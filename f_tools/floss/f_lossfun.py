import torch
from torch import Tensor, nn

import numpy as np
from torch.nn import BCEWithLogitsLoss, BCELoss
import torch.nn.functional as F
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import fmatch_OHEM, pos_match_retina4conf
from f_tools.fits.f_predictfun import batched_nms_auto
from f_tools.fits.ghm2 import GHMC_Loss2, GHMC_Loss3
from f_tools.floss.balanced_l1_loss import BalancedL1Loss
from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2ltwh, ltrb2xywh, calc_iou4ts, \
    bbox_iou4one
from f_tools.pic.f_show import f_show_3box4pil, show_anc4pil
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')


def x_bce(pconf_sigmoid, gconf, weight=1., reduction='none'):
    '''
    只支持二维
    :param pconf_sigmoid: 值必须为0~1之间 float
    :param gconf: 值为 float
    :return:
    '''
    eps = torch.finfo(torch.float16).eps
    pconf_sigmoid = pconf_sigmoid.clamp(min=eps, max=1 - eps)
    # loss = np.round(-(gconf * np.log(pconf) + (1 - gconf) * np.log(1 - pconf)), 4)
    loss = -(torch.log(pconf_sigmoid) * gconf + torch.log(1 - pconf_sigmoid) * (1 - gconf)) * weight
    return loss


def x_smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def f_ohem(scores, num_neg_ts, mask_pos, mash_ignore=None, pboxes_ltrb=None, threshold_iou=0.5):
    '''
    选反例最大的
    :param pboxes_ltrb:  (batch,anc)
    :param scores:gconf  (batch,anc)
    :param num_neg_ts: 每批负例数 3倍正例 int64 tensor([   3,    6, 3075,  768,    3,    3,    3,    3,    3,   3])
    :param mask_pos:
    :param threshold_iou:
    :return: mask_hard 蒙板
    '''

    scores_neg = scores.clone().detach()
    ts0 = torch.tensor(0.0, device=scores_neg.device)
    # 正例 or 忽略 分数置0
    if mash_ignore is not None:
        scores_neg[torch.logical_or(mask_pos, mash_ignore)] = ts0
    else:
        scores_neg[mask_pos] = ts0

    if pboxes_ltrb is None:
        return f_ohem_simpleness(scores_neg, num_neg_ts)

    # 下载几乎不用 --- 速度慢
    mask_neg_hard = torch.zeros_like(mask_pos, dtype=torch.bool, device=pboxes_ltrb.device)
    num_batch, dim_ancs, _ = pboxes_ltrb.shape

    # 只对框框和损失进行nms
    for i in range(num_batch):  # 每批NMS
        # 所有预测ltrb,
        keep = torch.ops.torchvision.nms(pboxes_ltrb[i], scores_neg[i], threshold_iou)
        keep = keep[:num_neg_ts[i]]
        mask_neg_hard[i, keep] = True  # 反例代表记损失

    # 批量nms
    # keep, ids_keep = batched_nms_auto(pboxes_ltrb, scores=scores, threshold_iou=threshold_iou)
    # for i in range(num_batch):
    #     mask = ids_keep == i
    #     if torch.any(mask):
    #         k = keep[mask][:num_pos_ts[i]]
    #         mask_neg_[i][k - i * dim_ancs] = True

    return mask_neg_hard


def f_ohem_simpleness(scores_neg, num_neg_ts):
    '''

    :param scores_neg:  已正例 及忽略置0的分数
    :param num_neg_ts:  tensor([111., 501.]) 反例个数 3倍
    :return:
    '''
    '''-----------简易版难例----------'''
    _, l_sort_ids = scores_neg.sort(dim=-1, descending=True)  # descending 倒序
    # 得每一个图片batch个 最大值索引排序 torch.Size([32, 169])
    _, l_sort_ids_rank = l_sort_ids.sort(dim=-1)  # 两次索引排序 用于取最大的n个布尔索引
    mask_neg_hard = l_sort_ids_rank < num_neg_ts.unsqueeze(-1)  # 选出最大的n个的mask  Tensor [batch, 8732]
    return mask_neg_hard  # torch.Size([32, 169])


def show_distribution(pconf, num_bins=10):
    '''
    显示分布
    :param pconf:
    :param num_bins:
    :return:
    '''
    pconf = pconf.clone().detach()
    float_eps = torch.finfo(torch.float16).eps
    inds_edges = torch.floor(pconf * (num_bins - float_eps)).long()  # 数据对应在哪个格子中 格子index
    nums_edges = torch.zeros(num_bins)  # 区间样本数量
    for i in range(num_bins):
        nums_edges[i] = (inds_edges == i).sum().item()
    flog.debug('show_dis: %s' % [i for i in nums_edges.tolist()])


def show_res(v_, input, target, name):
    s = []
    for i in range(len(v_)):
        _t = '{:.4f}({:.2f},{:.2f})'.format(v_[i], input[i], target[i])
        s.append(_t)
    print(name, '\t'.join(s))


def t_多值交叉熵():
    ''' 1,3,2 -> 1,2 '''
    # 1,2,3
    input = torch.tensor([[[1, 1, 1], [-1., 1, 6]]])
    # 1,2,3 -> 1,3,2
    input = input.permute(0, 2, 1)
    target = torch.tensor([[2, 1]])  # (1,2)

    # # 4d不支持 1,3,2 -> 1,1,3,2
    # input = input.unsqueeze(0)
    # # 1,2 -> 1,1,2
    # target = target.unsqueeze(0)

    print(input.shape)
    print(target.shape)

    obj_ce = nn.CrossEntropyLoss(reduction='none')
    print('CrossEntropyLoss', obj_ce(input, target))


def loss_cre_pdata_t1(nums_bins=(8000, 500, 6, 5, 30, 0, 0, 2, 1, 1)):
    pconf = torch.zeros([0])
    for i, num_bins in enumerate(nums_bins):
        if num_bins > 0:
            _offset = i / 10
            _tmp = torch.rand(num_bins) * 0.1 + _offset
            pconf = torch.cat([pconf, _tmp], -1)
    index = [i for i in range(len(pconf))]
    np.random.shuffle(index)
    pconf = pconf[index]
    return pconf


def t_图形测试():
    '''
    二值交叉熵可通过独热处理多分类
    :return:
    '''

    # bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    p_cls = [[0.4, 0.8, 0.99], [0.2, 0.7, 0.8]]
    g_cls = [[0.4, 0.2, 1], [0, 0, 1]]  # float

    p_cls = torch.arange(0, 1, 0.01)

    y = 0.7
    g_cls = torch.full_like(p_cls, y)

    xs = []
    ys = []
    labels = []
    alpha = 0.5
    gamma = 2

    # out = focalloss_v2(p_cls, g_cls, alpha=alpha, gamma=gamma, is_merge=True)
    # xs.append(p_cls)
    # ys.append(out)
    # labels.append('focalloss_v2')

    # out = focalloss_v3(p_cls, g_cls, alpha=alpha, gamma=gamma, mask_pos=g_cls > 0)
    # xs.append(p_cls)
    # ys.append(out)
    # labels.append('focalloss_retina')
    #
    # bce_obj = torch.nn.BCELoss(reduction='none')
    # out = bce_obj(p_cls, g_cls)
    # xs.append(p_cls)
    # ys.append(out)

    balanced_l1_obj = BalancedL1Loss()
    out = balanced_l1_obj(p_cls, g_cls)
    xs.append(p_cls)
    ys.append(out)
    labels.append('balanced_l1_obj')

    # ghmc_obj = GHMC_Loss()
    # out = ghmc_obj(p_cls, g_cls, is_debug=True)
    # xs.append(p_cls)
    # ys.append(out)
    # labels.append('ghmc_obj')

    # ghmr_obj = GHMR_Loss(0.2)
    # out = ghmr_obj(p_cls, g_cls, is_debug=True)
    # xs.append(p_cls)
    # ys.append(out)
    # labels.append('mse_loss')

    # out = F.mse_loss(p_cls, g_cls, reduction='none')
    # xs.append(p_cls)
    # ys.append(out)
    # labels.append('mse_loss')

    # loss_draw_res(xs, ys, labels=labels, title=y)
    loss_draw_res(xs, ys, title=y)


def loss_draw_res(xs, ys, labels=None, title=None, is_xylim=True):
    plt.xlabel('p_cls')
    plt.ylabel('loss')
    if title is not None:
        plt.title(title)
    if is_xylim:
        plt.xlim(0, 1)
        plt.ylim(0, 5)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if labels is not None:
            plt.plot(x, y, label=labels[i], alpha=0.9)
        else:
            plt.plot(x, y, alpha=0.9)
    plt.legend()


def t_损失值测试():
    ghmc_obj = GHMC_Loss2(10, 0.2)
    ghmc_obj = GHMC_Loss3()
    # (8000, 500, 6, 5, 30, 0, 0, 2, 1, 1)
    p_cls = loss_cre_pdata_t1()
    yy = torch.arange(0, 1, 0.1)

    labels = []
    labels.append('ghmc_obj')
    x1 = []
    y1 = []

    labels.append('x_bce')
    x2 = []
    y2 = []

    labels.append('focalloss')
    x3 = []
    y3 = []

    for i, y in enumerate(yy):
        g_cls = torch.full_like(p_cls, y)
        # out = ghmc_obj(p_cls, g_cls, is_debug=True)
        # out = ghmc_obj(p_cls, g_cls)
        out = ghmc_obj.calc(p_cls, g_cls)
        x1.append(y.item())
        _val = out.sum().item()
        y1.append(_val)
        print('ghmc_obj', _val)

        out = x_bce(p_cls, g_cls)
        x2.append(y.item())
        _val = out.sum().item()
        y2.append(_val)
        print('x_bce', _val)

        out = focalloss_v3(p_cls, g_cls, mask_pos=p_cls > 0)
        print('focalloss', out.sum().item())
        x3.append(y.item())
        _val = out.sum().item()
        y3.append(_val)

    print(y1)
    print(y2)
    print(y3)
    # _draw_res([x1, x2, x3], [y1, y2, y3], labels=labels, is_xylim=False)
    loss_draw_res([x1], [y1], labels=labels[0], is_xylim=False)


def t_showpic_one(fun_loss, text_labels, y=0.5, **kwargs):
    xs = []
    ys = []
    labels = []
    p_cls = torch.arange(0, 1, 0.01)
    g_cls = torch.full_like(p_cls, y)
    out = fun_loss(p_cls, g_cls, **kwargs)
    xs.append(p_cls)
    ys.append(out)
    labels.append(text_labels)
    loss_draw_res(xs, ys, labels=labels, title=y)


if __name__ == '__main__':
    import numpy as np

    np.random.seed(20201031)

    # t_多值交叉熵()
    # t_图形测试()

    p = torch.tensor([-1.1, 1.4, 1.5])
    # g = torch.tensor([0.03, 0.03, 0.7])
    g = torch.tensor([0.03, 0.03, 0.9])
    print(p.sigmoid())
    print(F.binary_cross_entropy(p.sigmoid(), g, reduction='none'))
    print(F.binary_cross_entropy_with_logits(p, g, reduction='none', pos_weight=torch.tensor(4)))
    # BCEWithLogitsLoss
    # BCELoss()

    # t_损失值测试()
    # pred = torch.Tensor([0, 2, 3])
    # target = torch.Tensor([1, 1, 1])
    # weight = torch.Tensor([1, 0, 1])
