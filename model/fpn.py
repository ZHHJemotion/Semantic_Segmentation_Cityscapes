"""
    The Feature Pyramid Networks (FPN) for the RetinaNet
    paper: Feature Pyramid Networks for Object Detection

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import resnet101
from modules.normalization import SwitchNorm2d


class FPN(nn.Module):
    def __init__(self, input_size=(512, 1024)):
        super(FPN, self).__init__()
        self.input_size = input_size
        self.input_row = input_size[0]
        self.input_col = input_size[1]

        # self.base_net = InceptionResNetV2()
        self.base_net = resnet101()

        # !!!! Can be replaced with light-head/RFB/ASPP ect. to improve the receptive field !!!!
        self.c6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.c5_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1)
        self.c4_down = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        self.c3_down = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        # !!!! I add another two dilated convolution to it to improve the receptive field !!!!
        # !!!! Of course they can be replaced by light-head/ASPP/RBF etc. !!!!
        # !!!! Of course they can also be replaced by Inception Block !!!!
        self.c2_up = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        # !!!! Can be replaced with light-head/RFB/ASPP etc. to improve the receptive field !!!!
        # self.conv_fuse = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.p4_atrous = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
                                       SwitchNorm2d(256, using_moving_average=True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                       SwitchNorm2d(256, using_moving_average=True),
                                       nn.ReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2))
        #                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
        #                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8))

        # downsample Conv
        self.conv_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv1x1_cat = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.sn = SwitchNorm2d(256, using_moving_average=True)
        self.bn = nn.BatchNorm2d(256, eps=0.001)

    @staticmethod
    def _upsmaple_add(x, y, size):
        x = F.upsample(x, size, mode='bilinear')
        return torch.add(x, 1, y)

    def _downsample_add(self, x, y):
        x = self.conv_down(x)
        x = self.sn(x)
        x = F.relu(x)
        return torch.add(x, 1, y)

    def forward(self, x):
        c2, c3, c4, c5 = self.base_net(x)           # 1/4, 1/8, 1/16, 1/32, 1/64 of 512

        # shortcut
        p4_sc = F.adaptive_avg_pool2d(c2, (int(self.input_row/16), int(self.input_col/16)))
        p5_sc = F.adaptive_avg_pool2d(c2, (int(self.input_row/32), int(self.input_col/32)))
        n5_sc = F.adaptive_avg_pool2d(c2, (int(self.input_row/32), int(self.input_col/32)))

        # Up-bottom
        p5 = self.c5_down(c5)                       # (256, 16, 32)    1/32
        p5 = p5 + p5_sc
        p5 = self.sn(p5)
        p5 = F.relu(p5)

        p6 = self.c6(F.relu(c5))                    # (256, 8, 16)     1/64

        p4 = self.c4_down(c4)
        p4 = self.sn(p4)
        p4 = F.relu(p4)
        p4 = self._upsmaple_add(p5, p4, (int(self.input_row/16), int(self.input_col/16)))  # (256, 32, 64)    1/16
        p4 = self.conv_fuse(p4)
        p4 = self.sn(p4)
        # p4 = p4 + p4_sc
        p4 = F.relu(p4)                     # add low level feature to p4

        p4_out = self.p4_atrous(p4)

        p3 = self.c3_down(c3)
        p3 = self.sn(p3)
        p3 = F.relu(p3)
        p3 = self._upsmaple_add(p4, p3, (int(self.input_row/8), int(self.input_col/8)))   # (256, 64, 128)   1/8
        p3 = self.conv_fuse(p3)
        p3 = self.sn(p3)
        p3 = F.relu(p3)

        p2 = self.c2_up(c2)
        p2 = self.sn(p2)
        p2 = F.relu(p2)
        p2 = self._upsmaple_add(p3, p2, (int(self.input_row/4), int(self.input_col/4)))  # (256, 128, 256) 1/4
        p2 = self.conv_fuse(p2)
        p2 = self.sn(p2)
        p2 = F.relu(p2)

        # Bottom-up
        """
        n2 = p2                                             # 1/4  for semantic seg
        n3 = self._downsample_add(n2, p3)                   # 1/8
        n4 = self._downsample_add(n3, p4)                   # 1/16  for semantic seg
        n5 = self._downsample_add(n4, p5) + n5_sc
        # n5 = F.relu(self._downsample_add(n4, p5) + n5_sc)   # 1/32
        """

        # return [n2, n3, n4, n5, p6]
        return p2, p3, p4_out, p5, p6


if __name__ == "__main__":
    import time
    from torch.autograd import Variable

    input_model = Variable(torch.randn(1, 3, 512, 1024))

    fpn = FPN()

    start_time = time.time()
    output_model = fpn(input_model)
    end_time = time.time()
    print("FPN inference time: {}s".format(end_time - start_time))

    print(output_model[0].size())
