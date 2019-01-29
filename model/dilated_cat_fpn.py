"""
    The Feature Pyramid Networks (FPN) for the RetinaNet
    paper: Feature Pyramid Networks for Object Detection

    This one is used to fit the dilated ResNet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dilated_resnet import dilated_resnet50
from modules.normalization import SwitchNorm2d
# from modules.bn import InPlaceABNSync, InPlaceABN
# from encoding.nn import BatchNorm2d


# --------------------------------------------------------------------- #
# Dilated FPN with anyone norm_act
# --------------------------------------------------------------------- #
class DilatedFPN(nn.Module):
    """
    TODO: using InPlaceABN
    """

    def __init__(self, input_size=(512, 1024), norm_act=None):
        super(DilatedFPN, self).__init__()
        self.input_size = input_size
        self.input_row = input_size[0]
        self.input_col = input_size[1]

        self.base_net = dilated_resnet50()
        self.out_se = nn.Sequential(ModifiedSCSEBlock(channel=(512+256+128+64+16), reduction=16))

        # !!!! Can be replaced with light-head/RFB/ASPP ect. to improve the receptive field !!!!
        # self.c5_down = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=8, dilation=8)
        # self.c4_down = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.c3_down = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2)

        # reduce the output channel
        self.c5_down = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)  # 512 320
        self.c4_down = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # 256 160
        self.c3_down = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)  # 128 96
        self.c2_down = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)  # 64 64
        self.c1_down = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)  # 16 32

        # dilated convolution to improve the receptive field !!!!
        # !!!! Of course they can be replaced by light-head/ASPP/RBF etc. !!!!
        # !!!! Of course they can also be replaced by Inception Block !!!!
        """
        self.c1_up = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),  # 6
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=4, dilation=4),  # 12
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=8, dilation=8),  # 18
                                   nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=16, dilation=16))  # 24
        self.c2_up = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),  # 6
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4),  # 12
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=8, dilation=8),  # 18
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=16, dilation=16))  # 24
        """
        # !!!! Can be replaced with light-head/RFB/ASPP etc. to improve the receptive field !!!!
        # self.conv_fuse = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.atrous = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
        #                             # SwitchNorm2d(256, using_moving_average=True),
        #                             nn.BatchNorm2d(256, eps=0.001),
        #                             nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #                             # SwitchNorm2d(256, using_moving_average=True),
        #                             nn.BatchNorm2d(256, eps=0.001))
        # self.conv_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2))

        # downsample Conv
        # self.conv_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv1x1_cat = nn.Conv2d(in_channels=(512+256+128+64+16), out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.conv1x1_fuse = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # self.sn = SwitchNorm2d(256, using_moving_average=True)
        # self.bn = norm_act(256)  # syn_bn
        # self.bn_1 = norm_act(64)

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
        c1, c2, c3, c4, c5 = self.base_net(x)           # 1/2, 1/4, 1/8, 1/8, 1/8

        # reduce channel number
        c1_down = self.c1_down(c1)  # 1/2 64   --> 32
        c2_down = self.c2_down(c2)  # 1/4 256  --> 64
        c3_down = self.c3_down(c3)  # 1/8 512  --> 96
        c4_down = self.c4_down(c4)  # 1/8 1024 --> 160
        c5_down = self.c5_down(c5)  # 1/8 2048 --> 320

        # shortcut to 1/8
        c1_sc_1 = F.max_pool2d(c1_down, kernel_size=3, stride=2, padding=1)  # 1/2 --> 1/4
        c1_sc_2 = F.max_pool2d(c1_sc_1, kernel_size=3, stride=2, padding=1)  # 1/4 --> 1/8

        c2_sc = F.max_pool2d(c2_down, kernel_size=3, stride=2, padding=1)  # 1/4 --> 1/8


        """
        p5 = self.c5_down(c5)                       # 1/8
        p5 = p5 + sc
        p5 = self.conv1x1_fuse(p5)
        p5 = self.sn(p5)
        p5 = F.relu(p5)

        p4 = self.c4_down(c4)
        p4 = p4 + sc                                # add low level feature to p4
        p4 = self.conv1x1_fuse(p4)
        p4 = self.sn(p4)
        p4 = F.relu(p4)

        p3 = self.c3_down(c3)                       # 1/8
        p3 = p3 + sc
        p3 = self.conv1x1_fuse(p3)
        p3 = self.sn(p3)
        p3 = F.relu(p3)
        """

        # high feature 1/8
        # with concatenation and atrous as fusion
        # high = torch.cat((p5, p4, p3), dim=1)
        # high = self.conv1x1_cat(high)
        # high = self.atrous(high)

        # with SCSE Block
        high = torch.cat((c5_down, c4_down, c3_down, c2_sc, c1_sc_2), dim=1)  # 320+160+96+64+32
        # high = self.conv1x1_cat(high)  # 3840-->256
        # high = self.bn(high)
        high = self.out_se(high)
        # high = F.relu(high)

        # low feature 1/4
        # low_1 = self.c1_up(c1_sc_1)
        # low_1 = self.bn_1(low_1)

        # low_2 = self.c2_up(c2_down)                         # 1/4
        # low_2 = self.bn(low_2)

        # concate low_1 and low_2 into low, but without conv 1x1
        # because it will be concatenated with high passed ASPP
        low = torch.cat((c2_down, c1_sc_1), dim=1)

        return low, high


# --------------------------------------------------------------------- #
# Spatial-Channel Sequeeze & Excitation Block
# --------------------------------------------------------------------- #
class ModifiedSCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ModifiedSCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        batch, chn, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(batch, chn)
        chn_se = self.channel_excitation(chn_se).view(batch, chn, 1, 1)

        spa_se = self.spatial_se(x)
        # return torch.mul(torch.mul(x, chn_se), spa_se)
        return torch.add(torch.mul(x, chn_se), 1, torch.mul(x, spa_se))


if __name__ == "__main__":
    import time
    from torch.autograd import Variable

    input_model = Variable(torch.randn(1, 3, 512, 1024))

    fpn = DilatedFPN()

    start_time = time.time()
    output_model = fpn(input_model)
    end_time = time.time()
    print("FPN inference time: {}s".format(end_time - start_time))

    print(output_model[0].size())
