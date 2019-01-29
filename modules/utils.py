import torch
import torch.nn as nn
import torch.nn.functional as F


class PBCSABlock(nn.Module):
    def __init__(self, in_chns, reduct_ratio=16, dilation=4, is_res=True, scale=1.0):
        super(PBCSABlock, self).__init__()
        self.is_res = is_res
        self.scale = scale
        # ------------------------------------------ #
        # Channel-wise Attention Model
        # ------------------------------------------ #
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)

        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.LeakyReLU(negative_slope=0.15, inplace=True),
                                      nn.Conv2d(in_chns // reduct_ratio, in_chns,
                                                kernel_size=1, stride=1, padding=0))
        # self.ch_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns,
        #                                        kernel_size=1, stride=1, padding=0),
        #                              nn.BatchNorm2d(in_chns))

        self.sp_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, 1, kernel_size=1,
                                               stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------ #
        # 1. Channel-wise Attention Model
        # ------------------------------------------ #
        res = x
        avg_p = self.se_block(self.ch_avg_pool(x))
        max_p = self.se_block(self.ch_max_pool(x))
        ch_att = avg_p + max_p

        ch_att = torch.mul(x, self.sigmoid(ch_att).exp())

        # ------------------------------------------ #
        # 2. Spatial-wise Attention Model
        # ------------------------------------------ #
        sp_att = torch.mul(x, self.sigmoid(self.sp_conv(x)).exp())

        if self.is_res:
            return sp_att + res + ch_att

        return sp_att + ch_att


def soft_jaccard_loss(inputs, targets, weights=None, ignore_index=None):
    """
    inputs : NxCxHxW Variable
    targets :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1.0
    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.log_softmax(inputs, dim=1).exp()
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)  # the shape of mask as same as encoded_target
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = inputs * encoded_target
    numerator = intersection.sum(dim=0).sum(dim=1).sum(dim=1)
    denominator = inputs + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0

    denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
    loss_per_channel = weights * (1.0 - torch.log((numerator + smooth) / (denominator - numerator + smooth)))

    return loss_per_channel.sum() / inputs.size(1)


def dice_loss(inputs, targets, weights=None, ignore_index=None):
    """
    inputs : NxCxHxW Variable
    targets :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1.0
    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.log_softmax(inputs, dim=1).exp()
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = inputs * encoded_target
    numerator = 2.0 * intersection.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
    denominator = inputs + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0

    denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
    loss_per_channel = weights * (1.0 - (numerator / denominator))

    return loss_per_channel.sum() / inputs.size(1)


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, model, layer_name, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        layers = self.model._modules
        target_layer = layers[self.layer_name]

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)

        self.model(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale:
            self.rescale_output_array(x.size())

        return self.inputs, self.outputs


"""
image = data_preprocess(img_org, mean)
    _, out_feat = extractor.forward(image)

    out_feat = out_feat.numpy()
    out_feat = np.squeeze(out_feat)

for chn, feat in enumerate(out_feat):
        print("> +++++++++++++++++++++++++++++++++++++++++++++ <")
        print("> Processing Channel: {}...".format(chn))
        feat = 255.0 * ((feat - feat.min().min()) / (feat.max().max() - feat.min().min()))
        feat = feat.astype(np.uint8)
        feat = misc.imresize(feat, (net_h, net_w), interp="bilinear")

        # cv2.namedWindow("ImageOut", cv2.WINDOW_NORMAL)
        # cv2.imshow("ImageOut", feat)
        # cv2.waitKey()

        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(14, 6))
        axs[0].set_title('blue should be up')
        pos = axs[0].imshow(img_org_copy, origin='left')
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[0].set_title("Base Image")
        fig.colorbar(pos, ax=axs[0])

        axs[1].set_title('blue should be down')
        axs[1].imshow(feat, origin='right')
        pos = axs[1].imshow(img_org_copy, origin='right', alpha=0.225)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        axs[1].set_title("HeatMap on Image")
        fig.colorbar(pos, ax=axs[1])
        plt.tight_layout()

        # plt.imshow(feat)
        # plt.imshow(img_org_copy, alpha=0.425)
        # plt.colorbar()
        # plt.savefig(os.path.join(save_root, str(chn) + "_heatmap_2d.png"))
        plt.show()

"""