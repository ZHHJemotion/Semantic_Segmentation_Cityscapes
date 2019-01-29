'''Some helper functions for PyTorch.'''
import os
import math

import torch
import torch.nn as nn


def init_weights(model, pi=0.01, pre_trained=None):
    """
        The initialization for network:
            step 1: initializing the whole net --> weight: gaussian of std = 0.01, bias = 0
            step 2: initializing the base net by using the weights from pre-trained model
            step 3: initializing the cls_layer of subnet in RetinaNet --> bias = - log((1-pi)/pi)
    """
    init_fn = nn.init.kaiming_normal_
    # init_fn = nn.init.normal_

    cls_bias = - math.log((1 - pi) / pi)
    cls_prefix = "module.cls_subnet.8"

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            init_fn(m.weight, 0)

            if hasattr(m, "bias") and m.bias is not None:
                if cls_prefix in name:
                    nn.init.constant_(m.bias, cls_bias)  # initializing the cls_layer of cls subnet
                else:
                    nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    if pre_trained is not None:  # initializing the base net by pre-trained model
        if 'mobilenetv2' in pre_trained:
            pre_weight = torch.load(pre_trained)

            model_dict = model.state_dict()
            model_dict_tolist = list(model_dict.items())
            count = 0
            for key, value in pre_weight.items():
                if "feature" in key:
                    layer_name, weights = model_dict_tolist[count]
                    model_dict[layer_name] = value
                    count += 1

            model.load_state_dict(model_dict)
        else:
            pre_weight = torch.load(pre_trained)
            prefix = "module.fpn.base_net."

            model_dict = model.state_dict()

            pretrained_dict = {(prefix + k): v for k, v in pre_weight.items() if (prefix + k) in model_dict}
            model_dict.update(pretrained_dict)

            model.load_state_dict(model_dict)


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        lr = init_lr*(1 - iter/max_iter)**power
        param_group['lr'] = lr

        return lr


def adjust_learning_rate(optimizer, init_lr, decay_rate=.1, curr_epoch=0, epoch_step=10,
                         start_decay_at_epoch=10, total_epoch=40, mode='exp'):
    lr = init_lr
    if mode == 'step':
        lr = init_lr * (decay_rate ** (curr_epoch // epoch_step))

    elif mode == 'exp':
        if curr_epoch < start_decay_at_epoch:
            lr = init_lr
        else:
            lr = (init_lr * (0.01 ** (float(curr_epoch + 1 - start_decay_at_epoch)
                                       / (total_epoch + 1 - start_decay_at_epoch))))

    elif mode == 'poly':
        lr = init_lr * ((1 - curr_epoch / total_epoch) ** 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def cosine_annealing_lr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    # \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
    # \cos(\frac{T_{cur}}{T_{max}}\pi))

    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))


def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0:
            param_group['lr'] = lr
    return optimizer


def get_mean_and_std(dataset, max_load=10000):
    '''Compute the mean and std value of dataset.'''
    # dataloader = torch.utils.datasets.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        print(i)
        im, _, _ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:, j, :, :].mean()
            std[j] += im[:, j, :, :].std()
    mean.div_(N)
    std.div_(N)
    return mean, std


def poly_topk_scheduler(init_topk, iter, topk_decay_iter=1, max_iter=89280, power=0.9):
    curr_topk = init_topk
    if iter % topk_decay_iter or iter > max_iter:
        return curr_topk

    curr_topk = int(init_topk * (1 - iter / max_iter) ** power)
    if curr_topk <= 128:
        curr_topk = 128

    return curr_topk


def mask_select(input, mask, dim=0):
    '''Select tensor rows/cols using a mask tensor.
    Args:
      input: (tensor) input tensor, sized [N,M].
      mask: (tensor) mask tensor, sized [N,] or [M,].
      dim: (tensor) mask dim.
    Returns:
      (tensor) selected rows/cols.
    Example:
    >>> a = torch.randn(4,2)
    >>> a
    -0.3462 -0.6930
     0.4560 -0.7459
    -0.1289 -0.9955
     1.7454  1.9787
    [torch.FloatTensor of size 4x2]
    >>> i = a[:,0] > 0
    >>> i
    0
    1
    0
    1
    [torch.ByteTensor of size 4]
    >>> masked_select(a, i, 0)
    0.4560 -0.7459
    1.7454  1.9787
    [torch.FloatTensor of size 2x2]
    '''
    index = mask.nonzero().squeeze(1)
    return input.index_select(dim, index)


def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.
    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.
    Returns:
      (tensor) meshgrid, sized [x*y,2]
    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]
    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)


def change_box_order(boxes, order):
    """
    Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).
    
    :param boxes: (tensor) bounding boxes, sized [N,4].
    :param order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
    :return: (tensor) converted bounding boxes, sized [N,4].
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]  # xmin, ymin or xcenter, ycenter
    b = boxes[:, 2:]  # xmax, ymax or width, height

    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2, b-a+1], dim=1)
    return torch.cat([a-b/2, a+b/2], dim=1)


def box_iou(box1, box2, order='xyxy'):
    """
    Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    
    :param box1:  (tensor) bounding boxes, sized [N,4].
    :param box2:  (tensor) bounding boxes, sized [M,4].
    :param order: (str) box order, either 'xyxy' or 'xywh'.
    :return:      (tensor) iou, sized [N,M].
    """

    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)  # [x_min, y_min, x_max, y_max]
    M = box2.size(0)  # [x_min, y_min, x_max, y_max]

    # [N,2] -> [N,1,2] -> [N,M,2]   [M,2] -> [1,M,2] -> [N,M,2]
    lt = torch.max(box1[:, :2].unsqueeze(1).expand(N, M, 2), box2[:, :2].unsqueeze(0).expand(N, M, 2))

    # [N,2] -> [N,1,2] -> [N,M,2]   [M,2] -> [1,M,2] -> [N,M,2]
    rb = torch.min(box1[:, 2:].unsqueeze(1).expand(N, M, 2), box2[:, 2:].unsqueeze(0).expand(N, M, 2))

    wh = (rb - lt).clamp(min=0)        # [N,M,2]  Clamps all elements in input to be larger or equal min
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)

    # lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    # rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    # wh = (rb-lt+1).clamp(min=0)        # [N,M,2]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # area1 = (box1[:, 2]-box1[:, 0]+1) * (box1[:, 3]-box1[:, 1]+1)  # [N,]
    # area2 = (box2[:, 2]-box2[:, 0]+1) * (box2[:, 3]-box2[:, 1]+1)  # [M,]
    # iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def msr_init(net):
    '''Initialize layer parameters.'''
    for layer in net:
        if type(layer) == nn.Conv2d:
            n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2./n))
            layer.bias.data.zero_()
        elif type(layer) == nn.BatchNorm2d:
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif type(layer) == nn.Linear:
            layer.bias.data.zero_()


def convert_state_dict(state_dict):
    """
    Converts a state dict saved from a dataParallel module to normal module state_dict inplace
    :param state_dict is the loaded DataParallel model_state
    """
    new_state = {}
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state[name] = v
    return new_state


def voc_ap(rec, prec):
    """
         Calculate the AP given the recall and precision array
            1st) We compute a version of the measured precision/recall curve with
                    precision monotonically decreasing
            2nd) We compute the AP as the area under this curve by numerical integration.
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]

    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    """
        This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    """
        This part creates a list of indexes where the recall changes
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    """
        The Average Precision (AP) is the area under the curve
        (numerical integration)
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i-1]) * mpre[i])
    return ap