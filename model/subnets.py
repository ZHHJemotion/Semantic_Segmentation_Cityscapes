"""
    Region Proposal Network
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.normalization import SwitchNorm2d
# from modules.bn import ABN, InPlaceABN, InPlaceABNSync
# from encoding.nn import BatchNorm2d


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


############################################################
#  RPN
############################################################
class RPN(nn.Module):
    """Builds the model of Region Proposal Network.
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, num_anchors):
        super(RPN, self).__init__()
        self.num_anchors = num_anchors  # 9 --> 3 !!!!!!!!!!!!

        self.softmax = nn.Softmax(dim=1)
        self.cls_subnet = self._make_cls_subnet(self.num_anchors * 2)
        self.reg_subnet = self._make_reg_subnet(self.num_anchors * 4)

    def forward(self, x):
        # Anchor Score. [batch, height, width, anchors per location * 2].
        rpn_class_logits = self.cls_subnet(x)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size(0), -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, 4]
        # where 4 means [tx, ty, tw, th]
        rpn_bbox = self.reg_subnet(x)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size(0), -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]

    @staticmethod
    def _make_cls_subnet(out_channel):
        layers = []

        # !!!!! Here should be modified, not use the normal arch. !!!!!
        # !!!!! Please thinking why you add 4 convlayer here, why 3x3 kernel size, why not other conv. types to use.
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))  # add dilate convolution
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, out_channel, kernel_size=3, stride=1, padding=1, bias=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_reg_subnet(out_channel):
        layers = []

        # !!!!! Here should be modified, not use the normal arch. !!!!!
        # !!!!! Please thinking why you add 4 convlayer here, why 3x3 kernel size, why not other conv. types to use.
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))  # add dilate convolution
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, out_channel, kernel_size=3, stride=1, padding=1, bias=True))
        return nn.Sequential(*layers)


############################################################
#  Semantic Segmentation Branch
############################################################
class SemanticSegBranch(nn.Module):
    """
        TODO: using InPlaceABNSync
        using Switchable Norm to replace of BN
    """
    def __init__(self, num_classes, input_size=(512, 512)):
        super(SemanticSegBranch, self).__init__()
        self.input_size = input_size

        self.conv1x1_a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True))
        self.aspp_bra1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12),  # 12
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True))
        self.aspp_bra2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=24, dilation=24),  # 24
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True))
        self.aspp_bra3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=36, dilation=36),  # 36
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True))
        self.gave_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.conv1x1_b = nn.Sequential(nn.Conv2d(256*6, 256, kernel_size=1, stride=1, padding=0),  # 256*5 --> 256*6
                                       # SwitchNorm2d(256, using_moving_average=True),
                                       nn.BatchNorm2d(256, eps=0.001),
                                       nn.ReLU(inplace=True))
        self.conv3x3 = nn.Sequential(# SwitchNorm2d(512, using_moving_average=True),
                                     nn.BatchNorm2d(256, eps=0.001),
                                     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                     # SwitchNorm2d(256, using_moving_average=True),
                                     nn.BatchNorm2d(256, eps=0.001),
                                     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.Upsample(size=self.input_size, mode='bilinear'))  # delation????

    def forward(self, feat_pyramid):
        low_feat = feat_pyramid[0]  # 1/4
        high_feat = feat_pyramid[1]  # 1/8

        high_feat = torch.cat((high_feat,
                               self.conv1x1_a(high_feat),
                               self.aspp_bra1(high_feat),
                               self.aspp_bra2(high_feat),
                               self.aspp_bra3(high_feat),
                               F.upsample(self.gave_pool(high_feat), size=(high_feat.size(2), high_feat.size(3)), mode='bilinear')
                               ), dim=1)
        high_feat = self.conv1x1_b(high_feat)
        high_feat = F.upsample(high_feat, size=(low_feat.size(2), low_feat.size(3)), mode='bilinear')

        low_feat = self.conv1x1_a(low_feat)

        feat = torch.cat((low_feat, high_feat), dim=1)  # add a conv 1x1
        seg_feat = self.conv3x3(feat)

        return seg_feat


############################################################
#  Semantic Segmentation with Context Encoding Branch
############################################################
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16, num_classes=19):
        super(SEBlock, self).__init__()
        self.num_classes = num_classes
        self.se = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                nn.Linear(int(channel//reduction), self.num_classes),
                                nn.Sigmoid())

    def forward(self, x):
        return self.se(x)


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


# no deeplab v3+
class SemanticSegContextEncodingBranch(nn.Module):
    """
        TODO: using InPlaceABN
        using Switchable Norm to replace of BN
    """
    def __init__(self, num_classes, input_size=(512, 512)):
        super(SemanticSegContextEncodingBranch, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Semantic Encoding
        self.score_se = nn.Sequential(ModifiedSCSEBlock(channel=256, reduction=16))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fcs = nn.Sequential(nn.Linear(256, int(256 / 16)),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Linear(int(256 / 16), self.num_classes),
                                    nn.Sigmoid())

        # ASPP
        self.aspp = ASPP(in_chs=256, out_chs=256, rate=(12, 24, 36))

        # Feature Fusion
        self.conv1x1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(256, eps=0.001))
        self.conv3x3 = nn.Sequential(# SwitchNorm2d(256, using_moving_average=True),
                                     nn.BatchNorm2d(256, eps=0.001),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                     # SwitchNorm2d(256, using_moving_average=True),
                                     nn.BatchNorm2d(256, eps=0.001),
                                     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.Upsample(size=self.input_size, mode='bilinear'))  # delation????

        self.sn = SwitchNorm2d(256, using_moving_average=True)
        self.bn = nn.BatchNorm2d(256, eps=0.001),

    def forward(self, feat_pyramid):
        low_feat = feat_pyramid[0]  # 1/4
        high_feat = feat_pyramid[1]  # use 1/8

        # ASPP Branch of deeplab v3+
        high_feat = self.aspp(high_feat)
        feat = F.upsample(high_feat, size=(low_feat.size(2), low_feat.size(3)), mode='bilinear')  # 1/8 --> 1/4

        # Context Encoding Branch
        se_feat = self.score_se(feat)

        # pixel-wise classifier
        batch, chn, _, _ = se_feat.size()
        se = self.avg_pool(se_feat.clone()).view(batch, chn)
        se = self.se_fcs(se)

        sem_feat = self.conv3x3(se_feat)
        sem_feat = torch.mul(sem_feat, se.clone().view(batch, self.num_classes, 1, 1))

        return se, sem_feat

"""
# Deeplab v3+
# using SynBN of Encoding
class SynBNSemanticSegDeeplabContextEncodingBranch(nn.Module):
    
    def __init__(self, num_classes, input_size=(512, 512)):
        super(SynBNSemanticSegDeeplabContextEncodingBranch, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Semantic Encoding
        self.score_se = nn.Sequential(ModifiedSCSEBlock(channel=256, reduction=16))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fcs = nn.Sequential(nn.Linear(256, int(256 / 16)),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Linear(int(256 / 16), self.num_classes),
                                    nn.Sigmoid())

        # ASPP
        self.syn_aspp = SynASPP(in_chs=672, out_chs=256, rate=(12, 24, 36))

        # Feature Fusion
        self.conv1x1_b = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                     # SwitchNorm2d(256, using_moving_average=True),
                                     BatchNorm2d(256),
                                     nn.ReLU(inplace=True))
        self.conv1x1_cat = nn.Sequential(nn.Conv2d(576, 256, kernel_size=1, stride=1, padding=0),
                                         BatchNorm2d(256))
        self.conv3x3 = nn.Sequential(# SwitchNorm2d(256, using_moving_average=True),
                                     BatchNorm2d(256),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                     # SwitchNorm2d(256, using_moving_average=True),
                                     BatchNorm2d(256),
                                     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.Upsample(size=self.input_size, mode='bilinear'))  # delation????
        self.sn = SwitchNorm2d(256, using_moving_average=True)
        self.bn = BatchNorm2d(256)

    def forward(self, feat_pyramid):
        low_feat = feat_pyramid[0]  # 1/4
        high_feat = feat_pyramid[1]  # use 1/8

        # ASPP Branch of deeplab v3+
        high_feat = self.syn_aspp(high_feat)
        high_feat = F.upsample(high_feat, size=(low_feat.size(2), low_feat.size(3)), mode='bilinear')

        # low_feat = self.conv1x1_b(low_feat)

        feat = torch.cat((low_feat, high_feat), dim=1)  # add a conv 1x1

        # Context Encoding Branch
        se_feat = self.conv1x1_cat(feat)
        se_feat = self.score_se(se_feat)

        # pixel-wise classifier
        batch, chn, _, _ = se_feat.size()
        se = self.avg_pool(se_feat.clone()).view(batch, chn)
        se = self.se_fcs(se)

        sem_feat = self.conv3x3(se_feat)
        sem_feat = torch.mul(sem_feat, se.clone().view(batch, self.num_classes, 1, 1))

        return se, sem_feat
"""


# Deeplab v3+
# with/without SynBN
class SemanticSegDeeplabContextEncodingBranch(nn.Module):
    """
        using Switchable Norm OR InPlaceABN to replace of BN
    """
    def __init__(self, num_classes, input_size=(512, 512), norm_act=None):
        super(SemanticSegDeeplabContextEncodingBranch, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Semantic Encoding
        self.score_se = nn.Sequential(ModifiedSCSEBlock(channel=256, reduction=16))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fcs = nn.Sequential(nn.Linear(256, int(256 / 16)),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Linear(int(256 / 16), self.num_classes),
                                    nn.Sigmoid())

        # ASPP
        self.aspp = ASPP(in_chs=(512+256+128+64+16), out_chs=256, rate=(12, 24, 36), norm_act=norm_act)

        # Feature Fusion
        self.conv1x1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                     # SwitchNorm2d(256, using_moving_average=True),
                                     norm_act(256),
                                     nn.ReLU(inplace=True))

        self.conv1x1_cat = nn.Sequential(nn.Conv2d((256+64+16), 256, kernel_size=1, stride=1, padding=0),  # 32 --> 16
                                         norm_act(256))
        self.conv3x3 = nn.Sequential(# SwitchNorm2d(256, using_moving_average=True),
                                     norm_act(256),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                     # SwitchNorm2d(256, using_moving_average=True),
                                     norm_act(256),
                                     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
                                     nn.Upsample(size=self.input_size, mode='bilinear'))  # delation????
        # self.sn = SwitchNorm2d(256, using_moving_average=True)
        self.bn = norm_act(256),

    def forward(self, feat_pyramid):
        low_feat = feat_pyramid[0]  # 1/4 of concate c1 and c2
        high_feat = feat_pyramid[1]  # use 1/8

        # ASPP Branch of deeplab v3+
        high_feat = self.aspp(high_feat)
        high_feat = F.upsample(high_feat, size=(low_feat.size(2), low_feat.size(3)), mode='bilinear')

        # low_feat = self.conv1x1(low_feat)

        feat = torch.cat((high_feat, low_feat), dim=1)  # add a conv 1x1

        # Context Encoding Branch
        se_feat = self.conv1x1_cat(feat)
        se_feat = self.score_se(se_feat)

        # pixel-wise classifier
        batch, chn, _, _ = se_feat.size()
        se = self.avg_pool(se_feat.clone()).view(batch, chn)
        se = self.se_fcs(se)

        sem_feat = self.conv3x3(se_feat)
        sem_feat = torch.mul(sem_feat, se.clone().view(batch, self.num_classes, 1, 1))

        return se, sem_feat


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ASPP
# Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
# https://arxiv.org/pdf/1802.02611v2.pdf
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class ASPP(nn.Module):
    def __init__(self, in_chs, out_chs, feat_size=(56, 112), rate=(12, 24, 36), norm_act=None):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                     # SwitchNorm2d(out_chs, using_moving_average=True),
                                     norm_act(out_chs, eps=0.001))

        self.aspp_bra1 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                 padding=rate[0], dilation=rate[0]),  # 6
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       norm_act(out_chs, eps=0.001),
                                       nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       norm_act(out_chs, eps=0.001))

        self.aspp_bra2 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                 padding=rate[1], dilation=rate[1]),  # 12
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       norm_act(out_chs, eps=0.001),
                                       nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       norm_act(out_chs, eps=0.001))

        self.aspp_bra3 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                 padding=rate[2], dilation=rate[2]),  # 18
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       norm_act(out_chs, eps=0.001),
                                       nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       norm_act(out_chs, eps=0.001))

        self.gave_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       nn.Upsample(size=feat_size, mode='bilinear'))

        self.aspp_catdown = nn.Sequential(nn.Conv2d(out_chs*5, out_chs, kernel_size=1, stride=1, padding=0),
                                          # SwitchNorm2d(out_chs, using_moving_average=True),
                                          norm_act(out_chs, eps=0.001),
                                          nn.ReLU(inplace=True))

    def forward(self, x):
        out = torch.cat((self.conv1x1(x),
                         self.aspp_bra1(x),
                         self.aspp_bra2(x),
                         self.aspp_bra3(x),
                         self.gave_pool(x)), dim=1)

        out = self.aspp_catdown(out)
        return out

"""
class SynASPP(nn.Module):
    def __init__(self, in_chs, out_chs, feat_size=(56, 112), rate=(12, 24, 36)):
        super(SynASPP, self).__init__()

        self.conv1x1 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                     # SwitchNorm2d(out_chs, using_moving_average=True),
                                     BatchNorm2d(out_chs))

        self.aspp_bra1 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                 padding=rate[0], dilation=rate[0]),  # 12
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       BatchNorm2d(out_chs),
                                       nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       BatchNorm2d(out_chs))

        self.aspp_bra2 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                 padding=rate[1], dilation=rate[1]),  # 24
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       BatchNorm2d(out_chs),
                                       nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       BatchNorm2d(out_chs))

        self.aspp_bra3 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                 padding=rate[2], dilation=rate[2]),  # 36
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       BatchNorm2d(out_chs),
                                       nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       # SwitchNorm2d(out_chs, using_moving_average=True),
                                       BatchNorm2d(out_chs))

        self.gave_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
                                       nn.Upsample(size=feat_size, mode='bilinear'))

        self.aspp_catdown = nn.Sequential(nn.Conv2d(out_chs*5, out_chs, kernel_size=1, stride=1, padding=0),
                                          # SwitchNorm2d(out_chs, using_moving_average=True),
                                          BatchNorm2d(out_chs),
                                          nn.ReLU(inplace=True))

    def forward(self, x):
        out = torch.cat((self.conv1x1(x),
                         self.aspp_bra1(x),
                         self.aspp_bra2(x),
                         self.aspp_bra3(x),
                         self.gave_pool(x)), dim=1)

        out = self.aspp_catdown(out)
        return out
"""


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Vortex Pooling: Improving Context Representation in Semantic Segmentation
# https://arxiv.org/abs/1804.06242v1
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class VortexPooling(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, rate=(3, 9, 27)):
        super(VortexPooling, self).__init__()
        self.gave_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1,
                                                 padding=0, groups=1, bias=False, dilation=1),
                                       nn.Upsample(size=feat_res, mode='bilinear'),
                                       nn.BatchNorm2d(num_features=out_chs))

        self.conv3x3 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                               padding=1, bias=False, groups=1, dilation=1),
                                     nn.BatchNorm2d(num_features=out_chs))

        self.vortex_bra1 = nn.Sequential(nn.AvgPool2d(kernel_size=rate[0], stride=1,
                                                      padding=int((rate[0] - 1) / 2), ceil_mode=False),
                                         nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                   padding=rate[0], bias=False, groups=1, dilation=rate[0]),
                                         nn.BatchNorm2d(num_features=out_chs))

        self.vortex_bra2 = nn.Sequential(nn.AvgPool2d(kernel_size=rate[1], stride=1,
                                                      padding=int((rate[1] - 1) / 2), ceil_mode=False),
                                         nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                   padding=rate[1], bias=False, groups=1, dilation=rate[1]),
                                         nn.BatchNorm2d(num_features=out_chs))

        self.vortex_bra3 = nn.Sequential(nn.AvgPool2d(kernel_size=rate[2], stride=1,
                                                      padding=int((rate[2] - 1) / 2), ceil_mode=False),
                                         nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                   padding=rate[2], bias=False, groups=1, dilation=rate[2]),
                                         nn.BatchNorm2d(num_features=out_chs))

        self.vortex_catdown = nn.Sequential(nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1,
                                                      padding=1, bias=False, groups=1, dilation=1),
                                            nn.BatchNorm2d(num_features=out_chs),
                                            nn.Dropout2d(p=0.2, inplace=True))

        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    def forward(self, x):
        out = torch.cat([self.gave_pool(x),
                         self.conv3x3(x),
                         self.vortex_bra1(x),
                         self.vortex_bra2(x),
                         self.vortex_bra3(x)], dim=1)

        out = self.vortex_catdown(out)
        return self.upsampling(out)


class AcceleratedVortexPooling(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, rate=(1, 3, 9, 27)):
        super(AcceleratedVortexPooling, self).__init__()

        self.avg_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=int((rate[0] - 1) / 2), ceil_mode=False)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=int((rate[1] - 1) / 2), ceil_mode=False)
        self.avg_pool_3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=int((rate[2] - 1) / 2), ceil_mode=False)

        self.gave_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1,
                                                 padding=0, groups=1, bias=False, dilation=1),
                                       nn.Upsample(size=feat_res, mode='bilinear'),
                                       nn.BatchNorm2d(num_features=out_chs))

        self.conv3x3 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                               padding=1, bias=False, groups=1, dilation=1),
                                     nn.BatchNorm2d(num_features=out_chs))

        self.vortex_bra1 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                   padding=rate[1], bias=False, groups=1, dilation=rate[1]),
                                         nn.BatchNorm2d(num_features=out_chs))

        self.vortex_bra2 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                   padding=rate[2], bias=False, groups=1, dilation=rate[2]),
                                         nn.BatchNorm2d(num_features=out_chs))

        self.vortex_bra3 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                                   padding=rate[3], bias=False, groups=1, dilation=rate[3]),
                                         nn.BatchNorm2d(num_features=out_chs))

        self.vortex_catdown = nn.Sequential(nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1,
                                                      padding=1, bias=False, groups=1, dilation=1),
                                            nn.BatchNorm2d(num_features=out_chs),
                                            nn.Dropout2d(p=0.2, inplace=True))

        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    def forward(self, x):
        out_x = self.conv3x3(x)
        out_y1 = self.avg_pool_1(x)
        out_y2 = self.avg_pool_2(out_y1)
        out_y3 = self.avg_pool_3(out_y2)

        out = torch.cat([self.gave_pool(x),
                         out_x,
                         self.vortex_bra1(out_y1),
                         self.vortex_bra2(out_y2),
                         self.vortex_bra3(out_y3)], dim=1)

        out = self.vortex_catdown(out)
        return self.upsampling(out)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# DenseASPP for Semantic Segmentation in Street Scenes
# http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class DenseASPP(nn.Module):
    def __init__(self, in_chs, out_chs, rate=(3, 6, 12, 18, 24)):
        super(DenseASPP, self).__init__()

        self.dense3 = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1,
                                              padding=rate[0], dilation=rate[0]),
                                    SwitchNorm2d(out_chs, using_moving_average=True),
                                    nn.ReLU(inplace=True))
        self.dense6 = nn.Sequential(nn.Conv2d(in_chs + out_chs*1, out_chs, kernel_size=3, stride=1,
                                              padding=rate[1], dilation=rate[1]),
                                    SwitchNorm2d(out_chs, using_moving_average=True),
                                    nn.ReLU(inplace=True))
        self.dense12 = nn.Sequential(nn.Conv2d(in_chs + out_chs*2, out_chs, kernel_size=3, stride=1,
                                               padding=rate[2], dilation=rate[2]),
                                     SwitchNorm2d(out_chs, using_moving_average=True),
                                     nn.ReLU(inplace=True))
        self.dense18 = nn.Sequential(nn.Conv2d(in_chs + out_chs*3, out_chs, kernel_size=3, stride=1,
                                               padding=rate[3], dilation=rate[3]),
                                     SwitchNorm2d(out_chs, using_moving_average=True),
                                     nn.ReLU(inplace=True))
        self.dense24 = nn.Sequential(nn.Conv2d(in_chs + out_chs*4, out_chs, kernel_size=3, stride=1,
                                               padding=rate[4], dilation=rate[4]),
                                     SwitchNorm2d(out_chs, using_moving_average=True),
                                     nn.ReLU(inplace=True))

        self.dense_catdown = nn.Sequential(nn.Conv2d(out_chs*6, out_chs, kernel_size=1, stride=1,
                                                     padding=1, bias=False),
                                           SwitchNorm2d(out_chs, using_moving_average=True),
                                           nn.ReLU(inplace=True))  # why dropout?????

    def forward(self, x):
        aspp3 = self.dense3(x)
        aspp6 = self.dense6(torch.cat((x, aspp3), dim=1))
        aspp12 = self.dense12(torch.cat((x, aspp3, aspp6), dim=1))
        aspp18 = self.dense18(torch.cat((x, aspp3, aspp6, aspp12), dim=1))
        aspp24 = self.dense24(torch.cat((x, aspp3, aspp6, aspp12, aspp18), dim=1))

        out = torch.cat((x, aspp3, aspp6, aspp12, aspp18, aspp24), dim=1)
        out = self.dense_catdown(out)

        return out
