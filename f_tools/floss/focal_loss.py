import torch
import torch.nn.functional as F

from f_tools.floss.f_lossfun import x_bce, t_showpic_one


def focalloss(pcls, gcls, mask_pos, mash_ignore=None,
              alpha=0.25, gamma=2, reduction='none', is_debug=False):
    eps = torch.finfo(torch.float16).eps
    # eps = 1e-6
    pcls = pcls.clamp(min=eps, max=1 - eps)

    # 正例-0.25 反例-0.75
    alpha_ts = torch.tensor(alpha, device=pcls.device)
    _alpha_factor = torch.where(mask_pos, alpha_ts, 1. - alpha_ts)
    focal_weight = torch.where(mask_pos, 1. - pcls, pcls)
    focal_weight = _alpha_factor * torch.pow(focal_weight, gamma)
    # bce = -(gcls * torch.log(pcls) + (1.0 - gcls) * torch.log(1.0 - pcls))
    bce = x_bce(pcls, gcls)

    if mash_ignore is None:
        loss_val = focal_weight * bce
    else:
        loss_val = focal_weight * bce * torch.logical_not(mash_ignore)

    if is_debug:
        mask_neg = torch.logical_not(torch.logical_or(mask_pos, mash_ignore))
        l_pos = loss_val * mask_pos
        l_neg = loss_val * mask_neg
        return l_pos, l_neg

    if reduction == 'mean':
        return loss_val.mean(-1)
    elif reduction == 'sum':
        return loss_val.sum()
    else:  # 'none'
        return loss_val


def focalloss_simple(pcls, gcls, alpha=0.25, gamma=2, reduction='none'):
    '''
    适用于 G = 1 简单化版 无需 正反忽略例
    :param pcls:
    :param gcls:
    :param alpha:
    :param gamma:
    :param reduction:
    :return:
    '''
    eps = torch.finfo(torch.float16).eps
    # eps = 1e-6
    pcls = pcls.clamp(min=eps, max=1 - eps)

    # 正例-0.25 反例-0.75
    pt = (1 - pcls) * gcls + pcls * (1 - gcls)
    focal_weight = (alpha * gcls + (1 - alpha) * (1 - gcls)) * pt.pow(gamma)
    loss_val = x_bce(pcls, gcls, focal_weight)

    if reduction == 'mean':
        return loss_val.mean(-1)
    elif reduction == 'sum':
        return loss_val.sum()
    else:  # 'none'
        return loss_val


def gaussian_focal_loss(pcls, gpcls, mask_pos=None, mask_neg=None, alpha=2.0, gamma=4.0, is_debug=False):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pcls (torch.Tensor): The prediction.
        gpcls (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = torch.finfo(torch.float16).eps

    if mask_pos is None:
        pos_weights = gpcls.eq(1)
    else:
        pos_weights = mask_pos

    if mask_neg is None:
        neg_weights = (1 - gpcls).pow(gamma)
    else:
        neg_weights = (1 - gpcls).pow(gamma) * mask_neg

    pos_loss = -(pcls + eps).log() * (1 - pcls).pow(alpha) * pos_weights
    neg_loss = -(1 - pcls + eps).log() * pcls.pow(alpha) * neg_weights

    if is_debug:
        l_pos = pos_loss
        l_neg = neg_loss
        return l_pos, l_neg

    return pos_loss + neg_loss


def quality_focal_lossv2(pcls, gcls, score, mask_pos, beta=2.0):
    '''

    :param pcls:  (n, 80)
    :param gcls: (n) 0, 1-80: 0 is neg, 1-80 is positive
    :param score: (n) reg target 0-1, only positive is good
    :param mask_pos: (n)
    :param beta:
    :return:
    '''
    # all goes to 0
    zerolabel = torch.zeros_like(pcls)
    loss = x_bce(pcls, zerolabel) * pcls.pow(beta)  # 这个是求反例

    indx_dim1 = gcls[mask_pos].long()

    # 正例引入质量 positive goes to bbox quality
    pt = score[mask_pos] - pcls[mask_pos, indx_dim1]
    loss[mask_pos, indx_dim1] = x_bce(pcls[mask_pos, indx_dim1], score[mask_pos]) * pt.pow(beta)

    return loss


def quality_focal_loss(pred, label, score, beta=2.0):
    '''

    :param pred:  (n, 80)
    :param label: (n) 0, 1-80: 0 is neg, 1-80 is positive
    :param score: (n) reg target 0-1, only positive is good
    :param beta:
    :return:
    '''
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction='none') * pt.pow(beta)

    label = label - 1
    pos = (label >= 0).nonzero().squeeze(1)
    a = pos
    b = label[pos].long()

    # positive goes to bbox quality
    pt = score[a] - pred_sigmoid[a, b]
    loss[a, b] = F.binary_cross_entropy_with_logits(pred[a, b], score[a], reduction='none') * pt.pow(beta)

    return loss


def distribution_focal_loss(pred, label):
    disl = label.long()
    disr = disl + 1

    wl = disr.float() - label
    wr = label - disl.float()

    # loss = F.cross_entropy(pred, disl, reduction='none') * wl + F.cross_entropy(pred, disr, reduction='none') * wr
    loss = x_bce(pred, disl, reduction='none') * wl + x_bce(pred, disr, reduction='none') * wr
    return loss


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pred = torch.Tensor([0.1, .5, .7])
    target = torch.Tensor([1, 1, 1])
    loss = distribution_focal_loss(pred, target)
    funs_loss = {
        'distribution_focal_loss': distribution_focal_loss,  # x_bce
        # 'x_bce': x_bce,
        'focalloss_simple': focalloss_simple,
        'gaussian_focal_loss': gaussian_focal_loss,
        'quality_focal_loss': quality_focal_loss,
    }
    for k, v in funs_loss.items():
        # t_showpic_one(v, k, y=0.7)
        t_showpic_one(v, k, y=1.)
        # t_showpic_one(v, k, y=0.1)
    plt.show()
    print(loss)
