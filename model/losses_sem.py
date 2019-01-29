from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class CrossEntropy2D(nn.Module):
    def __init__(self, reduction='none', ignore_label=255):
        super(CrossEntropy2D, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)

        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
        return loss


def cross_entropy2d(input, target, weight=None, size_average=True, reduction='none'):
    # 1. input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # 2. log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)

    # 3. log_p: (n*h*w, c) - contiguous() required if transpose() is used before view().
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    # 4. target: (n*h*w,)
    mask = target >= 0
    target = target[mask]

    loss = F.nll_loss(log_p, target, ignore_index=250, weight=weight, size_average=False, reduction=reduction)
    if size_average:
        loss /= mask.data.sum()
        # loss /= mask.sum().datasets[0]
    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, reduction='none'):
    """A categorical cross entropy loss for 4D tensors.
        We assume the following layout: (batch, classes, height, width)
        Args:
            input: The outputs.
            target: The predictions.
            K: The number of pixels to select in the bootstrapping process.
               The total number of pixels is determined as 512 * multiplier.
        Returns:
            The pixel-bootstrapped cross entropy loss.
    """
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, reduction='none'):
        n, c, h, w = input.size()

        # 1. The log softmax. log_p: (n, c, h, w)
        log_p = F.log_softmax(input, dim=1)

        # 2. log_p: (n*h*w, c) - contiguous() required if transpose() is used before view().
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        # 3. target: (n*h*w,)
        mask = target >= 0
        target = target[mask]

        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, reduction=reduction)

        # For each element in the batch, collect the top K worst predictions
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           reduction=reduction)
    return loss / float(batch_size)


class SemanticEncodingLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25):
        super(SemanticEncodingLoss, self).__init__()
        self.alpha = alpha

        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def unique_encode(self, cls_targets):
        batch_size, _, _ = cls_targets.size()
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)
        cls_targets = [cls_targets[idx].masked_select(target_mask[idx]) for idx in np.arange(batch_size)]

        # unique_cls = [np.unique(label.numpy(), return_counts=True) for label in cls_targets]
        unique_cls = [np.unique(label.numpy()) for label in cls_targets]

        encode = np.zeros((batch_size, self.num_classes), dtype=np.uint8)

        for idx in np.arange(batch_size):
            np.put(encode[idx], unique_cls[idx], 1)

        return torch.from_numpy(encode).float()

    def forward(self, predicts, enc_cls_target, size_average=True, reduction='elementwise_mean'):
        return self.alpha * F.binary_cross_entropy(predicts, enc_cls_target, weight=None,
                                                   size_average=size_average, reduction=reduction)


class ContextBootstrappedCELoss2D(nn.Module):
    """
    Context SoftMax Cross Entropy Loss
    """
    def __init__(self, num_classes=19, ignore=250, kernel_size=5, padding=4, dilate=2, use_gpu=True):
        super(ContextBootstrappedCELoss2D, self).__init__()
        self.num_classes = num_classes
        self.ignore = ignore

        # self.kernel_size = kernel_size
        self.padding = padding
        self.dilate = dilate

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.filter = nn.Parameter(torch.randn(num_classes, 1, kernel_size, kernel_size).cuda(),
                                       requires_grad=True)
        else:
            self.filter = nn.Parameter(torch.randn(num_classes, 1, kernel_size, kernel_size),
                                       requires_grad=True)

    def __bootstrap_xentropy_single(self, feat, target, top_k, weight=None, reduction='elementwise_mean'):
        n, c, h, w = feat.size()

        # 1. The log softmax. log_p: (n, c, h, w)
        log_p = F.log_softmax(feat, dim=1)

        # 2. log_p: (n*h*w, c) - contiguous() required if transpose() is used before view().
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        # 3. target: (n*h*w,)
        mask = target >= 0
        target = target[mask]

        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=self.ignore,
                          reduce=False, reduction=reduction)

        # For each element in the batch, collect the top top_k worst predictions
        topk_loss, _ = loss.topk(top_k)
        return topk_loss.sum() / top_k

    def forward(self, cls_feat, cls_target, top_k, weight=None, reduction='none'):
        # context_feat = F.conv2d(input=cls_feat, weight=self.filter, bias=None,
        #                         stride=1, padding=self.padding,
        #                         dilation=self.dilate, groups=self.num_classes)
        # batch_size = context_feat.size(0)
        batch_size = cls_feat.size(0)

        loss = 0.0
        # Bootstrap from each image not entire batch
        for batch in range(batch_size):
            loss += self.__bootstrap_xentropy_single(feat=torch.unsqueeze(cls_feat[batch], dim=0),
                                                     target=torch.unsqueeze(cls_target[batch], dim=0),
                                                     top_k=top_k, weight=weight, reduction=reduction)
        return loss / float(batch_size)


class FocalLoss2D(nn.Module):
    """
    Focal Loss, which is proposed in:
        "Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002v2)"
    """
    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25, gamma=2, size_average=True):
        """
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        :param num_classes:   (int) num of the classes
        :param ignore_label:  (int) ignore label
        :param alpha:         (1D Tensor or Variable) the scalar factor
        :param gamma:         (float) gamma > 0;
                                      reduces the relative loss for well-classified examples (probabilities > .5),
                                      putting more focus on hard, mis-classified examples
        :param size_average:  (bool): By default, the losses are averaged over observations for each mini-batch.
                                      If the size_average is set to False, the losses are
                                      instead summed for each mini-batch.
        """
        super(FocalLoss2D, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.size_average = size_average
        self.one_hot = Variable(torch.eye(self.num_classes))

    def forward(self, cls_preds, cls_targets):
        """

        :param cls_preds:    (n, c, h, w)
        :param cls_targets:  (n, h, w)
        :return:
        """
        assert not cls_targets.requires_grad
        assert cls_targets.dim() == 3
        assert cls_preds.size(0) == cls_targets.size(0), "{0} vs {1} ".format(cls_preds.size(0), cls_targets.size(0))
        assert cls_preds.size(2) == cls_targets.size(1), "{0} vs {1} ".format(cls_preds.size(2), cls_targets.size(1))
        assert cls_preds.size(3) == cls_targets.size(2), "{0} vs {1} ".format(cls_preds.size(3), cls_targets.size(3))

        if cls_preds.is_cuda:
            self.one_hot = self.one_hot.cuda()

        n, c, h, w = cls_preds.size()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. target reshape and one-hot encode
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1.1. target: (n*h*w,)
        cls_targets = cls_targets.view(n * h * w, 1)
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)

        cls_targets = cls_targets[target_mask]
        cls_targets = self.one_hot.index_select(dim=0, index=cls_targets)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. compute focal loss for multi-classification
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2.1. The softmax. prob: (n, c, h, w)
        prob = F.softmax(cls_preds, dim=1)
        # 2.2. prob: (n*h*w, c) - contiguous() required if transpose() is used before view().
        prob = prob.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        prob = prob[target_mask.repeat(1, c)]
        prob = prob.view(-1, c)  # (n*h*w, c)

        probs = torch.clamp((prob * cls_targets).sum(1).view(-1, 1), min=1e-8, max=1.0)
        batch_loss = -self.alpha * (torch.pow((1 - probs), self.gamma)) * probs.log()

        # For each element in the batch, collect the top K worst predictions
        # topk_loss, _ = batch_loss.topk(K)
        # reduced_topk_loss = topk_loss.sum() / K

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class CenterLoss2D(nn.Module):
    """
    Center loss.

        Reference:
        Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """
    def __init__(self, num_classes=19, ignore_label=250, feat_dim=256, use_gpu=True):
        """
        :param num_classes:
        :param feat_dim:
        :param use_gpu:
        :return:
        """
        super(CenterLoss2D, self).__init__()
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        self.one_hot = Variable(torch.eye(self.num_classes))  # be changed to fit for Pytorch 0.4.1

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, feat, cls_targets):
        """

        :param cls_preds:
        :param cls_targets:
        :return:
        """
        if feat.is_cuda:
            self.one_hot = self.one_hot.cuda()

        batch_size, feat_dim, height, width = feat.size()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. target reshape
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1.1. target: (batch_size * height * width,)
        cls_targets = cls_targets.view(batch_size * height * width, 1)
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)

        cls_targets = cls_targets[target_mask]
        cls_targets = self.one_hot.index_select(dim=0, index=cls_targets)

        # 2.2. feat: (batch_size * height * width, feat_dim) contiguous() required if transpose() is used before view().
        feat = feat.transpose(1, 2).transpose(2, 3).contiguous().view(-1, feat_dim)
        feat = feat[target_mask.repeat(1, feat_dim)]
        feat = feat.view(-1, feat_dim)  # (batch_size * height * width, feat_dim)

        count = feat.size(0)
        dist_mat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(count, self.num_classes) + \
                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, count).t()
        dist_mat.addmm_(1, -2, feat, self.centers.t())  # out = \beta\ mat + \alpha\ (mat1_i \mathbin{@} mat2_i)

        """
        dist = []
        for idx in range(count):
            cls_idx = cls_targets[idx].byte()
            value = dist_mat[idx][cls_idx]
            value = value.clamp(min=1e-8, max=1e+8)  # for numerical stability
            dist.append(value)

        dist = torch.cat(dist)
        """

        dist = torch.clamp(torch.masked_select(dist_mat, cls_targets.byte()), min=1e-6, max=1e+6)
        return dist.mean()


class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss for Attention"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight, size_average)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


class SegmentationMultiLosses(nn.Module):
    """2D Cross Entropy Loss with Multi-L1oss for Attention"""
    def __init__(self, nclass=-1, weight=None, ignore_index=-1, reduction='elementwise_mean'):
        super(SegmentationMultiLosses, self).__init__()
        self.nclass = nclass
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.celoss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, *inputs):

        preds, target = tuple(inputs)
        pred1, pred2, pred3= tuple(preds)

        loss1 = self.celoss(pred1, target)
        loss2 = self.celoss(pred2, target)
        loss3 = self.celoss(pred3, target)

        # loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)
        # loss2 = super(SegmentationMultiLosses, self).forward(pred2, target)
        # loss3 = super(SegmentationMultiLosses, self).forward(pred3, target)
        loss = loss1 + loss2 + loss3

        return loss


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    net_h, net_w = 448, 896

    # loss_fn = FocalLoss2D(num_classes=19)
    loss_fn = ContextBootstrappedCELoss2D(num_classes=19, ignore=250,
                                          kernel_size=5, padding=4,
                                          dilate=2, use_gpu=False)
    i = 0

    # pytorch 0.3.1 need to be updated
    while True:
        i += 1
        print("iter :", i)
        dummy_input = Variable(torch.rand(2, 19, net_h, net_w), requires_grad=True)
        dummy_target = torch.LongTensor(2, net_h, net_w).random_(-1, 19)
        dummy_target = Variable(dummy_target, requires_grad=False)

        loss = loss_fn(cls_feat=dummy_input, cls_target=dummy_target, top_k=512, weight=None, size_average=True)
        loss.backward()
        print("Loss: {}".format(loss.data[0]))
