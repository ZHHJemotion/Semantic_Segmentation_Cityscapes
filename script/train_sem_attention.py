import os
import time
import math
import json
import random
import argparse
import multiprocessing
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from tqdm import tqdm, tqdm_notebook
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datasets.cityscapes_loader import CityscapesLoader
from datasets.augmentations import *
from script.config import Config
from script.script_utils import init_weights, poly_topk_scheduler, poly_lr_scheduler
from model.attention_net import DANet
from model.attention_resnet import resnet50, resnet101
from model.losses_sem import bootstrapped_cross_entropy2d, SemanticEncodingLoss, SegmentationMultiLosses
from model.metrics import RunningScore


config = Config()


def train(args):
    weight_dir = args.log_root  # os.path.join(args.log_root, 'weights')
    log_dir = os.path.join(args.log_root, 'logs', 'SS-Net-{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                   time.localtime())))

    data_dir = os.path.join(args.data_root, args.dataset)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Setup DataLoader
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 0. Setting up DataLoader...")
    net_h, net_w = int(args.img_row * args.crop_ratio), int(args.img_col * args.crop_ratio)
    augment_train = Compose([RandomHorizontallyFlip(), RandomSized((0.5, 0.75)),
                             RandomRotate(5), RandomCrop((net_h, net_w))])
    augment_valid = Compose([RandomHorizontallyFlip(), Scale((args.img_row, args.img_col)),
                             CenterCrop((net_h, net_w))])

    train_loader = CityscapesLoader(data_dir, gt='gtFine', split='train',
                                    img_size=(args.img_row, args.img_col),
                                    is_transform=True, augmentations=augment_train)

    valid_loader = CityscapesLoader(data_dir, gt='gtFine', split='val',
                                    img_size=(args.img_row, args.img_col),
                                    is_transform=True, augmentations=augment_valid)

    num_classes = train_loader.n_classes

    tra_loader = data.DataLoader(train_loader, batch_size=args.batch_size,
                                 num_workers=int(multiprocessing.cpu_count() / 2),
                                 shuffle=True)

    val_loader = data.DataLoader(valid_loader, batch_size=args.batch_size,
                                 num_workers=int(multiprocessing.cpu_count() / 2))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 2. Setup Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Setting up Model...")
    backbone = resnet50(multi_grid=args.multi_grid, multi_dilation=[4, 8, 16])

    if args.pre_trained is not None:
        if 'resnet50' in args.pre_trained:
            backbone = resnet50(multi_grid=args.multi_grid, multi_dilation=[4, 8, 16])
        elif 'resnet101' in args.pre_trained:
            backbone = resnet101(multi_grid=args.multi_grid, multi_dilation=[4, 8, 16])

    model = DANet(nclass=num_classes, backbone=backbone, norm_layer=nn.BatchNorm2d)  # nn.BatchNor2d
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    # model = DataParallelModel(model, device_ids=[0,1,2,3]).cuda()  # multi-gpu

    # 2.1 Setup Optimizer
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # Check if model has custom optimizer
    if hasattr(model.module, 'optimizer'):
        print('> Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.90,
                                    weight_decay=5e-4, nesterov=True)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 2.2 Setup Loss
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    class_weight = np.array([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
                             5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
                             5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
                             2.78376194], dtype=float)
    class_weight = torch.from_numpy(class_weight).float().cuda()

    # sem_loss = bootstrapped_cross_entropy2d
    sem_loss = SegmentationMultiLosses(nclass=num_classes, ignore_index=250)

    """
    # multi-gpu
    bootstrapped_cross_entropy2d = ContextBootstrappedCELoss2D(num_classes=num_classes,
                                                               ignore=250,
                                                               kernel_size=5,
                                                               padding=4,
                                                               dilate=2,
                                                               use_gpu=True)
    loss_sem = DataParallelCriterion(bootstrapped_cross_entropy2d, device_ids=[0, 1]) 
    """

    # 2.3 Setup Metrics
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # !!!!! Here Metrics !!!!!
    metrics = RunningScore(num_classes)  # num_classes = 93

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 3. Resume Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 2. Model state init or resume...")
    args.start_epoch = 1
    args.start_iter = 0
    beat_map = 0.
    if args.resume is not None:
        full_path = os.path.join(os.path.join(weight_dir, 'train_model'), args.resume)
        if os.path.isfile(full_path):
            print("> Loading model and optimizer from checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(full_path)

            args.start_epoch = checkpoint['epoch']
            args.start_iter = checkpoint['iter']
            beat_map = checkpoint['beat_map']
            model.load_state_dict(checkpoint['model_state'])  # weights
            optimizer.load_state_dict(checkpoint['optimizer_state'])  # gradient state
            del checkpoint

            print("> Loaded checkpoint '{}' (epoch {}, iter {})".format(args.resume, args.start_epoch, args.start_iter))

        else:
            print("> No checkpoint found at '{}'".format(full_path))
            raise Exception("> No checkpoint found at '{}'".format(full_path))
    else:
        if args.pre_trained is not None:
            print("> Loading weights from pre-trained model '{}'".format(args.pre_trained))
            full_path = os.path.join(args.log_root, args.pre_trained)

            pre_weight = torch.load(full_path)
            model_dict = model.state_dict()

            if 'imagenet' in args.pre_trained:
                prefix = "module.backbone."
                pretrained_dict = {(prefix + k): v for k, v in pre_weight.items() if (prefix + k) in model_dict}
            else:
                pre_weight = pre_weight['model_state']
                pretrained_dict = {k: v for k, v in pre_weight.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            del pre_weight
            del model_dict
            del pretrained_dict

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 4. Train Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 4.0. Setup tensor-board for visualization
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    writer = None
    if args.tensor_board:
        writer = SummaryWriter(log_dir=log_dir, comment="SSnet_Attention_Cityscapes")
        # dummy_input = Variable(torch.rand(1, 3, args.img_row, args.img_col).cuda(), requires_grad=True)
        # writer.add_graph(model, dummy_input)

    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 3. Model Training start...")
    topk_init = 512
    num_batches = int(math.ceil(len(tra_loader.dataset.files[tra_loader.dataset.split]) /
                                float(tra_loader.batch_size)))

    for epoch in np.arange(args.start_epoch-1, args.num_epochs):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 4.1 Mini-Batch Training
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        model.train()
        topk_base = topk_init

        if epoch == args.start_epoch-1:
            pbar = tqdm(np.arange(args.start_iter, num_batches))
            start_iter = args.start_iter
        else:
            pbar = tqdm(np.arange(num_batches))
            start_iter = 0

        lr = args.learning_rate

        # scheduler.step()
        # for train_i, (images, gt_masks) in enumerate(tra_loader):  # One mini-Batch datasets, One iteration
        for train_i, (images, gt_masks) in zip(range(start_iter, num_batches), tra_loader):

            full_iter = (epoch * num_batches) + train_i + 1

            lr = poly_lr_scheduler(optimizer, init_lr=args.learning_rate, iter=full_iter,
                                   lr_decay_iter=1, max_iter=args.num_epochs*num_batches, power=0.9)

            images = images.cuda().requires_grad_()
            gt_masks = gt_masks.cuda()

            topk_base = poly_topk_scheduler(init_topk=topk_init, iter=full_iter, topk_decay_iter=1,
                                            max_iter=args.num_epochs * num_batches, power=0.95)

            optimizer.zero_grad()

            sem_seg_pred = model(images)  # tuple: sasc, sa, sc

            # --------------------------------------------------- #
            # Compute loss
            # --------------------------------------------------- #
            """
            # using bootstrapped_cross_entropy2d
            topk = topk_base * 512
            if random.random() < 0.20:
                train_loss = sem_loss(input=sem_seg_pred[0], target=gt_masks, K=topk, weight=class_weight)
            else:
                train_loss = sem_loss(input=sem_seg_pred[0], target=gt_masks, K=topk, weight=None)

            aux_loss = 0.0
            if args.aux:
                aux_loss = sem_loss(input=sem_seg_pred[1], target=gt_masks, K=topk, weight=None)\
                           + sem_loss(input=sem_seg_pred[2], target=gt_masks, K=topk, weight=None)

            loss = train_loss + args.aux_weight * aux_loss

            """
            # using SegmentationLosses
            train_loss = sem_loss(sem_seg_pred, gt_masks)
            loss = train_loss


            loss.backward()  # back-propagation

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
            optimizer.step()  # parameter update based on the current gradient

            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1, args.num_epochs))
            pbar.set_postfix(Train_Loss=train_loss.item(), TopK=topk_base)
            # pbar.set_postfix(Train_Loss=train_loss.item(), TopK=topk_base)

            # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 4.1.1 Verbose training process
            # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
            if (train_i + 1) % args.verbose_interval == 0:
                # ---------------------------------------- #
                # 1. Training Losses
                # ---------------------------------------- #
                loss_log = "Epoch [%d/%d], Iter: %d Loss1: \t %.4f " % (epoch + 1, args.num_epochs,
                                                                        train_i + 1, loss.item())

                # ---------------------------------------- #
                # 2. Training Metrics for sasc_output
                # ---------------------------------------- #
                sem_seg_pred = F.softmax(sem_seg_pred[0], dim=1)
                pred = sem_seg_pred.data.max(1)[1].cpu().numpy()
                gt = gt_masks.data.cpu().numpy()

                metrics.update(gt, pred)  # accumulate the metrics (confusion_matrix and ious)
                score, _ = metrics.get_scores()

                metric_log = ""
                for k, v in score.items():
                    metric_log += " {}: \t %.4f, ".format(k) % v
                metrics.reset()  # reset the metrics for each train_i steps

                logs = loss_log + metric_log

                if args.tensor_board:
                    writer.add_scalar('Training/Train_Loss', train_loss.item(), full_iter)
                    writer.add_scalar('Training/Loss', loss.item(), full_iter)
                    writer.add_scalar('Training/Lr', lr, full_iter)
                    writer.add_scalars('Training/Metrics', score, full_iter)
                    writer.add_text('Training/Text', logs, full_iter)

                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), full_iter)

        # end of this training phase
        state = {"epoch": epoch + 1,
                 "iter": num_batches,
                 'beat_map': beat_map,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict()}

        save_dir = os.path.join(os.path.join(args.log_root, 'train_model'),
                                "ssnet_model_sem_se_{}_{}epoch_{}iter.pkl".format(args.model_details,
                                                                                  epoch+1, num_batches))
        torch.save(state, save_dir)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 4.2 Mini-Batch Validation
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        model.eval()

        val_loss = 0.0
        vali_count = 0

        with torch.no_grad():
            for i_val, (images_val, gt_masks_val) in enumerate(val_loader):
                vali_count += 1

                images_val = images_val.cuda()
                gt_masks_val = gt_masks_val.cuda()

                sem_seg_pred_val = model(images_val)  # tuple: sasc, sa, sc

                # !!!!!! Loss !!!!!!
                # topk_val = topk_base * 512
                # loss = sem_loss(sem_seg_pred_val[0], gt_masks_val, topk_val, weight=None) + \
                #        args.aux_weight * (sem_loss(sem_seg_pred_val[1], gt_masks_val, topk_val, weight=None) +
                #                           sem_loss(sem_seg_pred_val[2], gt_masks_val, topk_val, weight=None))

                # using SegmentationLoss
                loss = sem_loss(sem_seg_pred_val, gt_masks_val)
                val_loss += loss.item()

                # accumulating the confusion matrix and ious
                sem_seg_pred_val = F.softmax(sem_seg_pred_val[0], dim=1)
                pred = sem_seg_pred_val.data.max(1)[1].cpu().numpy()
                gt = gt_masks_val.data.cpu().numpy()
                metrics.update(gt, pred)

            # ---------------------------------------- #
            # 1. Validation Losses
            # ---------------------------------------- #
            val_loss /= vali_count

            loss_log = "Epoch [%d/%d], Loss: \t %.4f" % (epoch + 1, args.num_epochs, val_loss)

            # ---------------------------------------- #
            # 2. Validation Metrics
            # ---------------------------------------- #
            metric_log = ""
            score, _ = metrics.get_scores()
            for k, v in score.items():
                metric_log += " {}: \t %.4f, ".format(k) % v
            metrics.reset()  # reset the metrics

            logs = loss_log + metric_log

            pbar.set_postfix(Vali_Loss=val_loss, Lr=lr, Vali_mIoU=score['Mean_IoU'])  # Train_Loss=train_loss.item()

            if args.tensor_board:
                writer.add_scalar('Validation/Loss', val_loss, epoch)
                writer.add_scalars('Validation/Metrics', score, epoch)
                writer.add_text('Validation/Text', logs, epoch)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 4.3 End of one Epoch
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # !!!!! Here choose suitable Metric for the best model selection !!!!!

        if score['Mean_IoU'] >= beat_map:
            beat_map = score['Mean_IoU']
            state = {"epoch": epoch + 1,
                     "beat_map": beat_map,
                     "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict()}

            save_dir = os.path.join(weight_dir, "SSnet_best_sem_se_{}_model.pkl".format(args.model_details))
            torch.save(state, save_dir)

        # Note that step should be called after validate()
        pbar.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 4.4 End of Training process
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    if args.tensor_board:
        # export scalar datasets to JSON for external processing
        # writer.export_scalars_to_json("{}/all_scalars.json".format(log_dir))
        writer.close()
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Training Done!!!")
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")


if __name__ == "__main__":
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 0. Hyper-params
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    parser = argparse.ArgumentParser(description='Hyper-params')

    parser.add_argument('--dataset', nargs='?', type=str, default='Cityscapes',
                        help='Dataset to use | Cityscapes by  default')
    parser.add_argument('--data_root', nargs='?', type=str, default='/home/liuhuijun/Datasets',
                        help='Dataset to use | /PycharmProject by  default')

    parser.add_argument('--img_row', nargs='?', type=int, default=512,
                        help='Height of the input image | 512 by  default')
    parser.add_argument('--img_col', nargs='?', type=int, default=1024,
                        help='Width of the input image | 512 by  default')
    parser.add_argument('--crop_ratio', nargs='?', type=float, default=0.875,  # 0.875
                        help='The ratio to crop the input image')

    parser.add_argument('--num_epochs', nargs='?', type=int, default=120,
                        help='# of the epochs used for training process | 120 by  default')
    parser.add_argument('--start_decay_at_epoch', nargs='?', type=int, default=16,
                        help='# of the epochs used for beginning decay | 21 by  default')

    parser.add_argument('--verbose_interval', nargs='?', default=200,
                        help='The interval for training result verbose | 200 by  default')
    parser.add_argument('--iter_interval_save_model', nargs='?', default=2000,
                        help='The iteration interval for saving model during training | 2000 by  default')
    parser.add_argument('--batch_size', nargs='?', default=8,  # 32
                        help='Batch size | 8 by  default')
    parser.add_argument('--learning_rate', nargs='?', type=float, default=4e-3,
                        help='Learning rate | 2.5e-4 by  default')

    parser.add_argument('--multi_grid', nargs='?', type=str, default=True,
                        help='multi_grid | True by  default')
    parser.add_argument('--multi_dilation', nargs='+', type=int, default=[4, 8, 16],
                        help='multi_dilation | None by  default')
    parser.add_argument('--se_loss', nargs='?', type=str, default=False,
                        help='se loss | False by  default')
    parser.add_argument('--se_weight', nargs='?', default=0.2,
                        help='the ratio of se loss | 0.2 by default')
    parser.add_argument('--aux', nargs='?', type=str, default=True,
                        help='auxiliary loss | False by  default')
    parser.add_argument('--aux_weight', nargs='?', default=0.5,
                        help='the ratio of auxiliary loss | 0.2 by default')

    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from | None by  default')
    parser.add_argument('--pre_trained', nargs='?', type=str, default='SSnet_best_sem_se_attention_model_resnet50_688.pkl', # 'resnet50_imagenet.pth',
                        help='Path to pre-trained  model to init from | None by  default')
                        # choices=[None, 'resnet101_imagenet.pth', 'resnet50_imagenet.pth'])
    parser.add_argument('--model_details', nargs='?', type=str, default='attention',
                        help='Some details for this model | None by  default')

    parser.add_argument('--tensor_board', nargs='?', type=bool, default=True,
                        help='Show visualization(s) through tensor-board | True by  default')
    parser.add_argument('--log_root', nargs='?', type=str,
                        default='/home/liuhuijun/tmp/pycharm_project_309/weights/',
                        help='Dataset to use | /home/liuhuijun/tmp/pycharm_project_309/weights by default')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Train the Deep Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    train_args = parser.parse_args()
    train(train_args)
