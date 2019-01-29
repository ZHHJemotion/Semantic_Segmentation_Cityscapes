"""
    The Feature Pyramid Networks (FPN) for the RetinaNet
    paper: Feature Pyramid Networks for Object Detection

    This one is used to fit the dilated ResNet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dilated_resnet import dilated_resnet101
from modules.normalization import SwitchNorm2d


class DilatedFPN(nn.Module):
    def __init__(self, input_size=(512, 1024)):
        super(DilatedFPN, self).__init__()
        self.input_size = input_size
        self.input_row = input_size[0]
        self.input_col = input_size[1]

        self.base_net = dilated_resnet101()
        self.out_se = nn.Sequential(ModifiedSCSEBlock(channel=256*3, reduction=16))

        # !!!! Can be replaced with light-head/RFB/ASPP ect. to improve the receptive field !!!!
        # self.c5_down = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=8, dilation=8)
        # self.c4_down = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.c3_down = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2)

        self.c5_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.c4_down = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.c3_down = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # dilated convolution to improve the receptive field !!!!
        # !!!! Of course they can be replaced by light-head/ASPP/RBF etc. !!!!
        # !!!! Of course they can also be replaced by Inception Block !!!!
        self.c2_up = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        # self.c2_up = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=6, dilation=6),  # 2
        #                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12),  # 4
        #                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=18, dilation=18),  # 8
        #                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=24, dilation=24))  # 16

        # !!!! Can be replaced with light-head/RFB/ASPP etc. to improve the receptive field !!!!
        # self.conv_fuse = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.atrous = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
                                    SwitchNorm2d(256, using_moving_average=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                    SwitchNorm2d(256, using_moving_average=True),
                                    nn.ReLU(inplace=True))
        # self.conv_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2))

        # downsample Conv
        self.conv_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv1x1_cat = nn.Conv2d(256*3, 256, kernel_size=1, stride=1, padding=0)
        self.conv1x1_fuse = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
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
        c2, c3, c4, c5 = self.base_net(x)           # 1/4, 1/8, 1/8, 1/8

        # shortcut
        # sc = F.adaptive_avg_pool2d(c2, (int(self.input_row/8), int(self.input_col/8)))  # 1/4 --> 1/8
        sc = F.max_pool2d(c2, kernel_size=3, stride=2, padding=1)

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

        # high feature 1/8
        # with concatenation and atrous as fusion
        # high = torch.cat((p5, p4, p3), dim=1)
        # high = self.conv1x1_cat(high)
        # high = self.atrous(high)

        # with SCSE Block
        high = torch.cat((p5, p4, p3), dim=1)
        high = self.out_se(high)
        high = self.conv1x1_cat(high)

        # low feature 1/4
        p2 = self.c2_up(c2)                         # 1/4
        p2 = self.sn(p2)
        low = F.relu(p2)

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
        return torch.mul(torch.mul(x, chn_se), spa_se)


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
