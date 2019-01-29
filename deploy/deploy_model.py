import argparse
import os

import torch.nn.functional as F
import numpy as np
import torch
import time
import cv2

from datasets.cocostuff_dataloader import COCOStuffLoader
from model.retinanet import RetinaNet
import torchvision.transforms as transforms
from torch.autograd import Variable
from deploy.image_utils import decode_segmap

try:
    import pydensecrf.densecrf as dcrf
except:
    print("Failed to import pydensecrf, CRF post-processing will not work")


def test():
    net_h, net_w = 512, 512   # 512, 1024  768, 1536  896, 1792  1024, 2048
    # Setup image
    # print("Read Input Image from : {}".format(args.img_path))
    deploy_img_file = '/home/pingguo/PycharmProject/SSnet_cityscape/deploy/deploy_img'
    img_path = os.path.join(deploy_img_file, "frankfurt_000000_014480_leftImg8bit.png")
    mask_path = os.path.join(deploy_img_file, "frankfurt_000000_014480_gtFine_color.png")

    img = cv2.imread(img_path)
    img = img[:, :, ::-1]  # bgr --> rgb
    msk = cv2.imread(mask_path)
    msk = msk[:, :, ::-1]  # bgr --> rgb

    data_path = "/home/pingguo/PycharmProject/COCO"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    loader = COCOStuffLoader(data_path, split='val', year="2017",
                             img_size=(net_h, net_w),
                             transform=transform, is_augment=None)
    n_classes = loader.num_classes
    im_mean = [0.485, 0.456, 0.406]
    im_std = [0.229, 0.224, 0.225]
    im_mean = np.array(im_mean).reshape([3, 1, 1])
    im_std = np.array(im_std).astype(float).reshape([3, 1, 1])

    # Setup Model
    print("> 1. Setting up Model...")
    model = RetinaNet(num_classes=n_classes, input_size=(net_h, net_w))
    model = torch.nn.DataParallel(model, device_ids=[1]).cuda()

    pre_weight = torch.load("/home/pingguo/PycharmProject/dl_project/Weights/PSnet/weights/{}".format(
        "psnet_model_sem.pkl"))
    pre_weight = pre_weight['model_state']
    model.load_state_dict(pre_weight)

    resized_img = cv2.resize(img, (loader.img_size[0], loader.img_size[1]), interpolation=cv2.INTER_CUBIC)
    resized_img = np.array(resized_img)

    # NHWC -> NCWH
    img = img.transpose(2, 0, 1)
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (loader.img_size[0], loader.img_size[1]), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img.astype(np.float32) / 255.
    img -= im_mean
    img = img / im_std

    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float
    img = img.unqueeze(0)

    model.eval()

    images = Variable(img.cuda(), volatile=True)

    start_time = time.time()
    outputs = F.softmax(model(images), dim=1)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    print("Inference time: {}s".format(time.time()-start_time))

    # color map for predicted mask
    decoded = pred*255
    decoded = decoded.astype(np.uint8)

    img_msk = cv2.addWeighted(resized_img, 0.60, decoded, 0.40, 0)
    fun_classes = np.unique(pred)
    print('> {} Classes found: {}'.format(len(fun_classes), fun_classes))

    out_path = "/home/pingguo/PycharmProject/dl_project/PSnet/deploy/{}".format("000000000872_imgmsk.png")
    img_msk.save(out_path)
    out_path = "/home/pingguo/PycharmProject/dl_project/PSnet/deploy/{}".format("000000000872_msk.png")
    decoded.imsave(out_path)
    print("> Segmentation Mask Saved at: {}".format(out_path))

    msk = cv2.resize(msk, (loader.img_size[0], loader.img_size[1]))
    cv2.namedWindow("Org Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Org Mask", msk)
    cv2.namedWindow("Pre Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Pre Mask", decoded[:, :, ::-1])
    cv2.namedWindow("Image Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Mask", img_msk[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--model_path', nargs='?', type=str, default='cityscapes_best_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                        help='Dataset to use [\'cityscapes, mvd etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None,
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None,
                        help='Path of the output segmap')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    deploy_args = parser.parse_args()
    # test(deploy_args)
    test()
