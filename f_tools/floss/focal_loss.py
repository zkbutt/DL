from abc import abstractmethod

import torch
import torch.nn.functional as F

from f_tools.floss.f_lossfun import x_bce, t_showpic_one
import torch.nn as nn


class FocallossBase(nn.Module):

    def __init__(self, reduction='none'):
        super(FocallossBase, self).__init__()
        self.reduction = reduction

    @abstractmethod
    def get_args(self, _mask_pos, pcls_sigmoid, gcls):
        weight, label = None, None
        return weight, label

    def forward(self, pcls_sigmoid, gcls, mask_pos, mash_ignore=None, is_debug=False):
        '''

        :param pcls_sigmoid:  这个要求 3D torch.Size([5, 3614, 3])
        :param gcls:
        :param mask_pos: 这个要求 2D
        :param mash_ignore:
        :param is_debug:
        :return:
        '''
        eps = torch.finfo(torch.float16).eps
        pcls_sigmoid = pcls_sigmoid.clamp(min=eps, max=1 - eps)

        # 定位到具体的正例
        mask_pos4cls = gcls > 1  # 同维 torch.Size([5, 3614, 3])
        _mask_pos = torch.logical_and(mask_pos.unsqueeze(-1).repeat(1, 1, gcls.shape[-1]), mask_pos4cls)

        '''--- 这个要复写 ---'''
        weight, _label = self.get_args(_mask_pos, pcls_sigmoid, gcls)

        bce = x_bce(pcls_sigmoid, _label)

        if mash_ignore is None:
            loss_val = weight * bce
        else:
            loss_val = weight * bce * torch.logical_not(mash_ignore)

        if is_debug:
            if mash_ignore is None:
                _mask_neg = torch.logical_not(_mask_pos)
            else:
                mash_ignore = mash_ignore.unsqueeze(-1).repeat(1, 1, gcls.shape[-1])
                _mask_neg = torch.logical_not(torch.logical_or(_mask_pos, mash_ignore))
            l_pos = loss_val * _mask_pos
            l_neg = loss_val * _mask_neg
            return l_pos, l_neg

        if self.reduction == 'mean':
            return loss_val.mean(-1)
        elif self.reduction == 'sum':
            return loss_val.sum()
        else:  # 'none'
            return loss_val


class Focalloss(FocallossBase):

    def __init__(self, alpha=0.25, gamma=2, reduction='none'):
        super(Focalloss, self).__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def get_args(self, _mask_pos, pcls_sigmoid, pgcls):
        # 正例-0.25 反例-0.75
        alpha_ts = torch.tensor(self.alpha, device=pcls_sigmoid.device)
        _alpha_factor = torch.where(_mask_pos, alpha_ts, 1. - alpha_ts)
        # 简单的分低 难的分高
        weight = torch.where(_mask_pos, 1. - pcls_sigmoid, pcls_sigmoid)
        weight = _alpha_factor * torch.pow(weight, self.gamma)
        return weight, pgcls


class GeneralizedFocalloss(FocallossBase):

    def __init__(self, beta=2.0, reduction='none'):
        super().__init__(reduction)
        self.beta = beta

    def get_args(self, _mask_pos, pcls_sigmoid, gcls4iou):
        weight = torch.where(_mask_pos, torch.abs(gcls4iou - pcls_sigmoid), pcls_sigmoid)
        weight = weight.pow(self.beta)  # 难易加成 -> 正负不平衡
        _label = torch.where(_mask_pos, gcls4iou, torch.zeros_like(pcls_sigmoid, device=pcls_sigmoid.device))
        return weight, _label


def focalloss(pcls_sigmoid, gcls, mask_pos, mash_ignore=None,
              alpha=0.25, gamma=2, is_debug=False):
    '''
    针对离散的 label
    :param pcls_sigmoid:
    :param gcls:
    :param mask_pos: mask为2D
    :param mash_ignore: mask为2D
    :param alpha:
    :param gamma:
    :param reduction:
    :param is_debug:
    :return:
    '''
    eps = torch.finfo(torch.float16).eps
    pcls_sigmoid = pcls_sigmoid.clamp(min=eps, max=1 - eps)

    # 定位到具体的正例
    if gcls.dim() == 3:
        mask_pos4cls = gcls == 1  # 同维 torch.Size([5, 3614, 3])
        _mask_pos = torch.logical_and(mask_pos.unsqueeze(-1).repeat(1, 1, gcls.shape[-1]), mask_pos4cls)
    else:
        _mask_pos = mask_pos

    '''-----这里和输入复写-----'''
    # 正例-0.25 反例-0.75
    alpha_ts = torch.tensor(alpha, device=pcls_sigmoid.device)
    _alpha_factor = torch.where(_mask_pos, alpha_ts, 1. - alpha_ts)
    # 简单的分低 难的分高
    focal_weight = torch.where(_mask_pos, 1. - pcls_sigmoid, pcls_sigmoid)
    focal_weight = _alpha_factor * torch.pow(focal_weight, gamma)
    # bce = -(gcls * torch.log(pcls) + (1.0 - gcls) * torch.log(1.0 - pcls))
    bce = x_bce(pcls_sigmoid, gcls)
    '''-----这里和输入复写完成-----'''

    if mash_ignore is None:
        loss_val = focal_weight * bce
    else:
        if gcls.dim() == 3:
            _mash_ignore = mash_ignore.unsqueeze(-1).repeat(1, 1, gcls.shape[-1])
        else:
            _mash_ignore = mash_ignore
        loss_val = focal_weight * bce * torch.logical_not(_mash_ignore)

    if is_debug:
        if mash_ignore is None:
            _mask_neg = torch.logical_not(_mask_pos)
        else:
            if gcls.dim() == 3:
                mash_ignore = mash_ignore.unsqueeze(-1).repeat(1, 1, gcls.shape[-1])
            _mask_neg = torch.logical_not(torch.logical_or(_mask_pos, mash_ignore))
        l_pos = loss_val * _mask_pos
        l_neg = loss_val * _mask_neg
        return l_pos, l_neg

    return loss_val


def focalloss_center(pcls_sigmoid, gcls, alpha=2., beta=4.):
    mask_pos_3d = gcls == 1
    mask_neg_3d = gcls != 1
    eps = torch.finfo(torch.float16).eps
    pcls_sigmoid = pcls_sigmoid.clamp(min=eps, max=1 - eps)
    l_pos = - ((1.0 - pcls_sigmoid) ** alpha) * torch.log(pcls_sigmoid) * mask_pos_3d.float()
    l_neg = -  ((1 - gcls) ** beta) * (pcls_sigmoid ** alpha) * torch.log(1.0 - pcls_sigmoid) * mask_neg_3d.float()
    return l_pos, l_neg


def focalloss_fcos(pcls_sigmoid, gcls, alpha=0.25, gamma=2):
    mask_pos_3d = gcls == 1
    mask_neg_3d = gcls != 1
    # mask_neg_3d = 1.0 - mask_pos_3d.float()
    eps = torch.finfo(torch.float16).eps
    pcls_sigmoid = pcls_sigmoid.clamp(min=eps, max=1 - eps)
    l_pos = -mask_pos_3d.float() * torch.pow(1 - pcls_sigmoid, gamma) * alpha * torch.log(pcls_sigmoid)
    l_neg = -mask_neg_3d.float() * torch.pow(pcls_sigmoid, gamma) * (1 - alpha) * torch.log(1 - pcls_sigmoid)
    return l_pos, l_neg


def quality_focal_loss(pcls_sigmoid, gcls, score_iou, mask_pos, mash_ignore=None, beta=2.0, is_debug=False):
    '''
    用于连续label的 score_iou
    分离训练 -> cls分数低 iou分数高 --- 标签分类回归联合 ^^ iou
    :param pcls_sigmoid:  torch.Size([5, 3614, 3])
    :param gcls: 是1的标签 torch.Size([5, 3614, 3]) 用于判断正例位置
    :param score_iou:  torch.Size([5, 3614]) box的iou分数 回归分数 0~1
    :param beta:
    :return:
    '''
    eps = torch.finfo(torch.float16).eps
    pcls_sigmoid = pcls_sigmoid.clamp(min=eps, max=1 - eps)

    # 定位到具体的正例
    mask_pos4cls = gcls == 1  # 同维 torch.Size([5, 3614, 3])
    # ([5, 3614] -> [5, 3614,1] -> [5, 3614, 3]) ^^ [5, 3614, 3]
    _mask_pos = torch.logical_and(mask_pos.unsqueeze(-1).repeat(1, 1, gcls.shape[-1]), mask_pos4cls)

    '''-----这里和输入复写-----'''
    # 维度匹配 [5, 3614] -> [5, 3614,1] -> [5, 3614, 3]
    _score_iou = score_iou.unsqueeze(-1).repeat(1, 1, gcls.shape[-1])
    # 正反例处理
    scale_factor = torch.where(_mask_pos, torch.abs(_score_iou - pcls_sigmoid), pcls_sigmoid)
    scale_factor = scale_factor.pow(beta)  # 难易加成 -> 正负不平衡
    _label = torch.where(_mask_pos, _score_iou, torch.zeros_like(pcls_sigmoid, device=pcls_sigmoid.device))
    bce = x_bce(pcls_sigmoid, _label)
    '''-----这里和输入复写完成-----'''

    if mash_ignore is None:
        loss_val = scale_factor * bce
    else:
        loss_val = scale_factor * bce * torch.logical_not(mash_ignore)

    if is_debug:
        if mash_ignore is None:
            _mask_neg = torch.logical_not(_mask_pos)
        else:
            mash_ignore = mash_ignore.unsqueeze(-1).repeat(1, 1, gcls.shape[-1])
            _mask_neg = torch.logical_not(torch.logical_or(_mask_pos, mash_ignore))
        l_pos = loss_val * _mask_pos
        l_neg = loss_val * _mask_neg
        return l_pos, l_neg

    return loss_val


def quality_focal_loss2(pcls_sigmoid, gcls4iou, mask_pos, mash_ignore=None, beta=2.0, is_debug=False):
    '''
    用于连续label的 gcls 0~1的标签
    分离训练 -> cls分数低 iou分数高 --- 标签分类回归联合 ^^ iou
    :param pcls_sigmoid:  torch.Size([5, 3614, 3])
    :param gcls4iou: 是0~1的标签， torch.Size([5, 3614, 3]) 用于判断正例位置
    :param beta:
    :return:
    '''
    eps = torch.finfo(torch.float16).eps
    pcls_sigmoid = pcls_sigmoid.clamp(min=eps, max=1 - eps)

    # 定位到具体的正例
    mask_pos4cls = gcls4iou > 0  # 同维 torch.Size([5, 3614, 3])
    # ([5, 3614] -> [5, 3614,1] -> [5, 3614, 3]) ^^ [5, 3614, 3]
    _mask_pos = torch.logical_and(mask_pos.unsqueeze(-1).repeat(1, 1, gcls4iou.shape[-1]), mask_pos4cls)

    gcls4iou = gcls4iou.type(pcls_sigmoid.type())  # 解决半精度训练的BUG
    # 正反例处理
    scale_factor = torch.where(_mask_pos, torch.abs(gcls4iou - pcls_sigmoid), pcls_sigmoid)
    scale_factor = scale_factor.pow(beta)  # 难易加成 -> 正负不平衡
    _label = torch.where(_mask_pos, gcls4iou, torch.zeros_like(pcls_sigmoid, device=pcls_sigmoid.device))
    bce = x_bce(pcls_sigmoid, _label)
    '''-----这里和输入复写完成-----'''

    if mash_ignore is None:
        loss_val = scale_factor * bce
    else:
        loss_val = scale_factor * bce * torch.logical_not(mash_ignore)

    if is_debug:
        if mash_ignore is None:
            _mask_neg = torch.logical_not(_mask_pos)
        else:
            mash_ignore = mash_ignore.unsqueeze(-1).repeat(1, 1, gcls4iou.shape[-1])
            _mask_neg = torch.logical_not(torch.logical_or(_mask_pos, mash_ignore))
        l_pos = loss_val * _mask_pos
        l_neg = loss_val * _mask_neg
        return l_pos, l_neg

    return loss_val


def distribution_focal_loss(cfg, preg_32d, label, mask_pos=None):
    '''
    Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>

    :param preg_32d: torch.Size([5, 3614, 32])
                    Predicted general distribution of bounding boxes
                    (before softmax) with shape (N, n+1), n is the max value of the
                    integral set `{0, ..., n}` in paper.
    :param label: 是anc和GT的计算出来的偏移, 差距很小在 0~7 之间 torch.Size([5, 3614, 4])
                Target distance label for bounding boxes with shape (N,).
    :param mask_pos: 用于测试 实际无用
    :return:Loss tensor with shape (N,).
    '''
    # CE只支持3D^^2D [5, 3614, 4] -> [5*3614, 4]
    batch, dim, _ = label.shape
    label = label.view(-1, 4)
    disl = torch.floor(label).long()  # 向下取整 [5*3614, 4]
    disr = torch.ceil(label).long()  # 向上取整 [5*3614, 4]
    weight_left = disr - label  # 下限偏移权重 [5*3614, 4]
    weight_right = label - disl  # torch.Size([18070, 4])

    # torch.Size([18070, 8, 4])
    preg_32d = preg_32d.view(-1, cfg.NUM_REG, 4)

    # 4D不支持 pred.view(*pred.shape[:2],8,-1)   ->  torch.Size([5, 3614, 8, 4]) ^^ torch.Size([5, 3614, 4])
    # 输出 torch.Size([18070, 4])
    loss = F.cross_entropy(preg_32d, disl, reduction='none') * weight_left \
           + F.cross_entropy(preg_32d, disr, reduction='none') * weight_right
    loss = loss.view(batch, dim, -1)  # torch.Size([5, 3614, 4]) 恢复尺寸
    return loss


def t图():
    global target
    import matplotlib.pyplot as plt
    pred = torch.Tensor([0.1, .5, .7])
    target = torch.Tensor([1, 1, 1])
    score = torch.Tensor([0.5, 0.7, 1])

    funs_loss = {
        # 'distribution_focal_loss': distribution_focal_loss,  # x_bce
        # # 'x_bce': x_bce,
        # 'focalloss_simple': focalloss_simple,
        # 'gaussian_focal_loss': gaussian_focal_loss,
        'quality_focal_loss': quality_focal_loss,
    }
    for k, v in funs_loss.items():
        # t_showpic_one(v, k, y=0.7)
        t_showpic_one(v, k, y=1.)
        # t_showpic_one(v, k, y=0.1)
    plt.show()
    # print(loss)


def focalloss_simple(pcls, gcls, alpha=0.25, gamma=2, reduction='none'):
    '''
    基本不用 适用于 G = 1 简单化版 无需 正反忽略例 不带忽略
    :param pcls: 支持 2D or 3D
    :param gcls:
    :param alpha:
    :param gamma:
    :param reduction:
    :return:
    '''
    assert reduction in (None, 'none', 'mean', 'sum')
    eps = torch.finfo(torch.float16).eps
    # eps = 1e-6
    pcls = pcls.clamp(min=eps, max=1 - eps)

    # 正例-0.25 反例-0.75
    pt = (1 - pcls) * gcls + pcls * (1 - gcls)
    weight = (alpha * gcls + (1 - alpha) * (1 - gcls)) * pt.pow(gamma)

    loss_val = x_bce(pcls, gcls, weight)

    if reduction == 'mean':
        return loss_val.mean(-1)
    elif reduction == 'sum':
        return loss_val.sum()
    else:  # 'none'
        return loss_val


def gaussian_focal_loss(pcls_sigmoid, gcls, alpha=2.0, gamma=4.0, is_debug=False):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.
    focal_loss 的一种 离散的 label
    Args:
        pcls_sigmoid (torch.Tensor): The prediction.
        gcls (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = torch.finfo(torch.float16).eps
    pcls_sigmoid = pcls_sigmoid.clamp(min=eps, max=1 - eps)

    pos_weights = gcls.eq(1)
    neg_weights = (1 - gcls).pow(gamma)

    l_pos = -pcls_sigmoid.log() * (1 - pcls_sigmoid).pow(alpha) * pos_weights
    l_neg = -(1 - pcls_sigmoid).log() * pcls_sigmoid.pow(alpha) * neg_weights

    if is_debug:
        return l_pos, l_neg

    return l_pos + l_neg


class HeatmapLoss(nn.Module):
    def __init__(self, weight=None, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0 - inputs) ** self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets) ** self.beta * (inputs) ** self.alpha * torch.log(1.0 - inputs + 1e-14)

        return center_loss + other_loss


class BCE_focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        eps = torch.finfo(torch.float16).eps
        inputs = inputs.clamp(min=eps, max=1 - eps)

        loss = (1.0 - inputs) ** self.gamma * (targets) * torch.log(inputs) + \
               (inputs) ** self.gamma * (1.0 - targets) * torch.log(1.0 - inputs)
        loss = -torch.sum(torch.sum(loss, dim=-1), dim=-1)
        return loss


if __name__ == '__main__':
    # t图()
    pred_sigmoid = torch.Tensor([[0.1, .5, .7], [0.12, .4, .6]])
    glabel = torch.Tensor([[1, 1, 1], [1, 1, 1]])
    gscore = torch.Tensor([[0.5, 0.7, 1], [0.5, 0.7, 1]])
    loss = quality_focal_loss(pred_sigmoid, glabel, gscore)
