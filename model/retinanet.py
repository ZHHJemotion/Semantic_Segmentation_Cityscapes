"""
    The RetinaNet built by InceptionV4 or InceptionResNetV2 as base net
    
    paper: Focal Loss for Dense Object Detection
    
"""
import torch.nn as nn
import torch.onnx

from script.config import Config
# from model.dilated_fpn import DilatedFPN
from model.dilated_cat_fpn import DilatedFPN
from model.subnets import RPN, SemanticSegBranch, SemanticSegContextEncodingBranch, \
    SemanticSegDeeplabContextEncodingBranch
# from modules.bn import InPlaceABNSync, InPlaceABN


class RetinaNet(nn.Module):
    num_anchors = 3  # 9 anchors per cell !!!!!!!!!!!!
    # num_classes = 81 denotes the number of classes in COCO database

    def __init__(self, num_classes=2, input_size=(512, 512), norm_layer=None):   # num_classes = 2 or 81
        super(RetinaNet, self).__init__()
        self.config = Config()
        self.num_classes = num_classes
        self.input_size = input_size

        # self.fpn = FPN(self.input_size)
        self.fpn = DilatedFPN(self.input_size, norm_act=norm_layer)
        # self.rpn = RPN(self.num_anchors)
        # self.sem_seg_subnet = SemanticSegBranch(num_classes=num_classes, input_size=input_size)
        # self.sem_seg_ce_subnet = SemanticSegContextEncodingBranch(num_classes=num_classes, input_size=input_size)

        # using syn bn
        self.sem_seg_ce_subnet = SemanticSegDeeplabContextEncodingBranch(num_classes=num_classes,
                                                                         input_size=input_size,
                                                                         norm_act=norm_layer)

        # self.classifier = Classifier(256, self.config.POOL_SIZE, self.config.IMAGE_SHAPE, self.config.NUM_CLASSES)
        # self.mask = Mask(256, self.config.MASK_POOL_SIZE, self.config.IMAGE_SHAPE, self.config.NUM_CLASSES)

    def forward(self, x):
        feat_pyramid = self.fpn(x)  # for rpn and mask (n2, n3, n4, n5, p6)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #   Instance Segmentation Part
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # output_rpn = []
        # for fp_i in feat_pyramid:
        #     output_rpn.append(self.rpn(fp_i))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #   Semantic Segmentation Part
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # sem_seg_feat = self.sem_seg_subnet(feat_pyramid)
        se, sem_seg_feat = self.sem_seg_ce_subnet(feat_pyramid)

        # return feat_pyramid, output_rpn, sem_seg_feat
        return se, sem_seg_feat

    def freeze_bn(self):
        """
            Freeze BatchNorm layers. 
            
            Reason: 
                1. using the pre-trained net whose BNs have been trained. 
                2. The Batch-size in object detection is small, hard to make BN parameter stable.
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == "__main__":

    from model.losses_retinanet import Losses
    from torch.autograd import Variable

    import numpy as np
    from tqdm import tqdm

    model = RetinaNet()
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)

    loss_fn = Losses()

    start_epoch = 0
    num_batches = 58633
    for epoch in np.arange(start_epoch, 120):
        pbar = tqdm(np.arange(num_batches))

        for train_i in range(num_batches):  # One mini-Batch datasets, One iteration

            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1, 120))

            # images = Variable(torch.rand(1, 3, 512, 512).cuda(), requires_grad=True)  # Image feed into the deep neural network
            cls_gt = Variable(torch.LongTensor(2, 196416).random_(-1, 81).cuda(), requires_grad=False)
            loc_gt = Variable(torch.randn(2, 196416, 4).cuda(), requires_grad=False)

            cls_pred = Variable(torch.FloatTensor(2, 196416, 81).random_(0, 81).cuda(), requires_grad=True)
            loc_pred = Variable(torch.randn(2, 196416, 4).cuda(), requires_grad=True)

            """cls_pred, loc_pred = model(images)

            optimizer.zero_grad()

            loss = Variable(torch.rand(1).cuda(), requires_grad=True)
            """
            loss = loss_fn(loc_pred, loc_gt, cls_pred, cls_gt)

            loss.backward()
            optimizer.step()

            pbar.set_postfix(Losses=loss.data[0])