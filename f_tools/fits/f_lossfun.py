import torch
from torch import Tensor, nn

import numpy as np
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import fmatch_OHEM, pos_match_retinaface
from f_tools.fits.f_predictfun import batched_nms_auto
from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2ltwh, diff_bbox, diff_keypoints, ltrb2xywh, calc_iou4ts, \
    bbox_iou4one, xy2offxy, offxy2xy, get_boxes_colrow_index, fix_boxes4yolo3
from f_tools.pic.f_show import f_show_3box4pil, show_anc4pil
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')


def f_bce(pconf, gconf, weight=1.):
    '''
    只支持二维
    :param pconf: 值必须为0~1之间 float
    :param gconf: 值为 float
    :return:
    '''
    torch.clamp(pconf, min=1e-6, max=1 - 1e-6)
    # loss = np.round(-(gconf * np.log(pconf) + (1 - gconf) * np.log(1 - pconf)), 4)
    loss = -(torch.log(pconf) * gconf + torch.log(1 - pconf) * (1 - gconf)) * weight
    return loss


def cneg_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    pos_inds = targets.eq(1).float()  # heatmap为1的部分是正样本
    neg_inds = targets.lt(1).float()  # 其他部分为负样本

    neg_weights = torch.pow(1 - targets, 4)  # 对应(1-Yxyc)^4

    loss = 0
    for pred in preds:  # 预测值
        # 约束在0-1之间
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss  # 只有负样本
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def t_focal_loss():
    # # 正例差距小 0.1054 0.0003
    # pconf, gconf = torch.tensor(0.9), torch.tensor(1.)
    # # 负例差距小 0.1054 0.0008
    # pconf, gconf = torch.tensor(0.1), torch.tensor(0.)
    # # 负例差距大 2.3026 1.3988 这个是重点
    # pconf, gconf = torch.tensor(0.9), torch.tensor(0.)
    # # 正例差距大 2.3026 0.4663
    # pconf, gconf = torch.tensor(0.1), torch.tensor(1.)
    '''
    alpha 越大 负债损失越小
    gamma 越大 中间损失越小
    :return:
    '''
    # 交叉熵是一样的
    # tensor([100.00000,   2.30258,   1.60944,   1.20397,   0.91629,   0.69315,   0.51083,   0.35667,   0.22314,   0.10536,   0.00000])
    # 反例 ： tensor([   75.00000,     1.39882,     0.77253,     0.44246,     0.24740,     0.12997,     0.06130,     0.02408,     0.00669,     0.00079,     0.00000])
    # 正例 ： tensor([   25.00000,     0.46627,     0.25751,     0.14749,     0.08247,     0.04332,     0.02043,     0.00803,     0.00223,     0.00026,    -0.00000])
    alpha, gamma = 0.25, 2.  # alpha 0.5以下 负例主导,越小负例影响大
    # tensor([   50.00000,     0.93255,     0.51502,     0.29497,     0.16493,     0.08664,     0.04087,     0.01605,     0.00446,     0.00053,     0.00000])
    # tensor([   50.00000,     0.93255,     0.51502,     0.29497,     0.16493,     0.08664,     0.04087,     0.01605,     0.00446,     0.00053,    -0.00000])
    alpha, gamma = 0.5, 2.  # alpha 是正负样本主导比例
    alpha, gamma = 0.5, 0.1  # gamma 0.5倍 一半时与交叉熵相同 恢复线性
    alpha, gamma = 0.75, 0.9  # 正例主导 正例3倍
    alpha, gamma = 0.25, 2  # 上了 0.5 就减损失
    # alpha, gamma = 2, 4  # 上了 0.5 就减损失

    torch.manual_seed(7)
    # pconf = torch.tensor([1.1] * 9)
    # gconf = torch.zeros_like(pconf)

    print('--------------- 1-0.05,  1-0.5,  1-0.8,  0.6-0.56, 0.5-0.05, 0.1-0.9, 0-095,0-0.5, 0-0.2,  0-0.1')
    gconf = torch.tensor([1, 1, 1., 0.6, 0.5, 0.1, 0, 0, 0, 0, 0])
    pconf = torch.tensor([0.05, 0.5, 0.8, 0.56, 0.05, 0.9, 0.95, 0.5, 0.2, 0.1, 0.99])

    loss_show = F.binary_cross_entropy(pconf, gconf, reduction='none')
    # loss_show = F.binary_cross_entropy(pconf, gconf, reduction='sum')
    print('bce_综合', loss_show)

    # bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    # obj_FocalLoss = FocalLoss(torch.nn.BCELoss(reduction='none'), alpha=alpha, gamma=gamma)
    # print('obj_FocalLoss', obj_FocalLoss(pconf, gconf))
    # obj_FocalLoss_v2 = FocalLoss_v2(is_oned=True, alpha=alpha, gamma=gamma)
    # print('FocalLoss_v2_综合', obj_FocalLoss_v2(pconf, gconf))
    print('focalloss_v2_综合', focalloss_v2(pconf, gconf, alpha=alpha, gamma=gamma))
    # print('focalloss_v3  _综合', focalloss_v3(pconf, gconf, alpha=alpha, gamma=gamma))
    # print('focal_loss4center_neg', focal_loss4center(pconf, gconf, a=a, b=b))

    a = 3  # 减小这个弧度上升
    b = 0.05
    ratio = 1

    print('focal_loss4center_综合', focal_loss4center3(pconf, gconf, a=a, b=b, ratio=ratio))
    # print('focal_loss4center_neg', focal_loss4center2(pconf, gconf, reduction='sum', a=2, b=4))

    # print('-----------------------下面是反例------------------------------------')
    pconf = torch.arange(1.0, -0.01, -0.1)
    gconf = torch.zeros_like(pconf)
    # # print('FocalLoss_v2_neg', obj_FocalLoss_v2(pconf, gconf))
    # # print('focalloss_v2_neg', focalloss_v2(pconf, gconf, alpha=alpha, gamma=gamma))
    # print('focalloss_v3  _neg', focalloss_v3(pconf, gconf, alpha=alpha, gamma=gamma))
    print('focal_loss4center_neg', focal_loss4center3(pconf, gconf, a=a, b=b, ratio=ratio))
    #
    # print('-----------------------下面是正例------------------------------------')
    pconf = torch.arange(0, 1.01, 0.1)
    gconf = torch.ones_like(pconf)
    # print(pconf, gconf)
    #
    # # loss_show = F.binary_cross_entropy(pconf, gconf, reduction='none')
    # # loss_show = F.binary_cross_entropy(pconf, gconf, reduction='sum')
    # # print('bce_pos', loss_show)
    #
    # # print('obj_FocalLoss_pos', obj_FocalLoss(pconf, gconf))
    print('obj_FocalLoss_v2_pos', obj_FocalLoss_v2(pconf, gconf))
    y = focalloss_v2(pconf, gconf, alpha=alpha, gamma=gamma)
    print('focalloss_v2_pos', y)
    plt.plot(pconf, y, color='r')
    # print('focalloss_v3  _pos', focalloss_v2(pconf, gconf, alpha=alpha, gamma=gamma))
    # # print('focal_loss4center_pos', focal_loss4center(pconf, gconf, a=a, b=b))
    y = focal_loss4center3(pconf, gconf, a=a, b=b)
    print('focal_loss4center_pos', y)

    plt.plot(pconf, y)
    plt.ylim([0.0, 5])
    plt.show()


def t_多值交叉熵():
    input = torch.tensor([[[1, 1, 1], [-1., 1, 6]]])
    input = input.permute(0, 2, 1)
    target = torch.tensor([[2, 1]])  # (1,2)
    print(input)
    print(target)

    obj_ce = nn.CrossEntropyLoss(reduction='none')
    print('CrossEntropyLoss', obj_ce(input, target))

    obj_fl4m = FocalLoss4mclass(alpha=0.25, gamma=2)
    print('obj_ce', obj_fl4m(input, target))


def f_二值交叉熵2():
    torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')
    # input 与 target 一一对应 同维
    size = (2, 3)  # 两个数据
    # input = torch.tensor([random.random()] * 6).reshape(*size)
    # print(input)
    # input = torch.randn(*size, requires_grad=True)
    input = torch.tensor([[0.8, 0.8],
                          [0.5, 2.]], dtype=torch.float)  # 默认是int64

    input = torch.tensor([[0.8, 0.8],
                          [0.5, 0.3]], dtype=torch.float)  # 默认是int64
    # input = torch.randn(*size, requires_grad=True)
    # [0,2)
    # target = torch.tensor(np.random.randint(0, 5, size), dtype=torch.float)
    # target = torch.tensor(np.random.randint(0, 2, size), dtype=torch.float)
    target = torch.tensor([
        [1, 0],
        [0, 1.],
    ], dtype=torch.float)

    # loss1 = F.binary_cross_entropy(torch.sigmoid(input), target, reduction='none')  # 独立的
    loss1 = F.binary_cross_entropy(input, target, reduction='none')  # 独立的
    print(loss1)

    # bce_loss = torch.nn.BCELoss(reduction='none')
    # focal_loss = FocalLoss(bce_loss, gamma=2)
    # loss3 = focal_loss(input, target)
    # print(loss3)

    loss2 = F.binary_cross_entropy_with_logits(input, target, reduction='none')  # 无需sigmoid
    # loss2 = F.binary_cross_entropy_with_logits(torch.sigmoid(input), target, reduction='none')  # 无需sigmoid
    print(loss2)

    # loss = F.binary_cross_entropy(F.softmax(input, dim=-1), target, reduction='none')  # input不为1要报错
    # loss = F.cross_entropy(input, target, reduction='none')  # 二维需要拉平
    loss3 = f_bce(input, target)
    print(loss3)

    # loss.backward()
    pass


def f_二值交叉熵1():
    '''
    二值交叉熵可通过独热处理多分类
    :return:
    '''
    bce_loss = torch.nn.BCELoss(reduction='none')
    # bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    p_cls = [[0.3, 0.8, 0.9], [0.2, 0.7, 0.8]]
    g_cls = [[0., 0, 1], [0, 0, 1]]  # float

    p_cls_np = np.array(p_cls)
    g_cls_np = np.array(g_cls)
    print(x_bce(p_cls_np, g_cls_np))

    p_cls_ts = torch.tensor(p_cls)
    g_cls_ts = torch.tensor(g_cls)
    print(bce_loss(p_cls_ts, g_cls_ts))

    floss = FocalLoss4Center()
    print(floss(p_cls_ts, g_cls_ts))


def focal_loss4center(pconf, gconf, reduction='none', a=2., b=4.):
    eps = torch.finfo(torch.float).eps
    pconf = pconf.clamp(min=eps, max=1 - eps)
    # ^Y=pconf  Y=gconf
    l_pos = gconf * torch.pow((1 - pconf), a) * torch.log(pconf)  # 正例
    # p=0.99 a=0.8   (1−0.8)**4*(0.99)**2*log(1−0.99)
    l_neg = (1 - gconf) * torch.pow((1 - gconf), b) * torch.pow(pconf, a) * torch.log(1 - pconf)  # 反例
    loss = -(l_pos + l_neg)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def focal_loss4center2(pconf, gconf, reduction='none', a=2., b=4., ratio=1.):
    eps = torch.finfo(torch.float).eps
    pconf = pconf.clamp(min=eps, max=1 - eps)
    mask_pos = gconf == 1
    mask_neg = torch.logical_not(mask_pos)

    # ^Y=pconf  Y=gconf
    l_pos = mask_pos * torch.pow(1 - pconf, a) * torch.log(pconf)  # 正例
    # p=0.99 a=0.8   (1−0.8)**4*(0.99)**2*log(1−0.99)
    neg_weights = torch.pow((1 - gconf), b)
    l_neg = ratio * mask_neg * neg_weights * torch.pow(pconf, a) * torch.log(1 - pconf)  # 反例
    loss = -(l_pos + l_neg)

    show_distribution(pconf)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def focal_loss4center3(pconf, gconf, reduction='none', a=2.):
    '''a=3'''
    eps = torch.finfo(torch.float).eps
    pconf = pconf.clamp(min=eps, max=1 - eps)
    # mask_pos = gconf > 0
    # mask_neg = torch.logical_not(mask_pos)

    # ^Y=pconf  Y=gconf
    abs_val = torch.abs(gconf - pconf)
    weights = torch.pow(abs_val, a)
    # weights = torch.pow(1 - pconf, 6)
    # weights = 1
    loss = weights * 0.3 / torch.log(abs_val)  # 正例
    loss = -loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


class FocalLoss4Center(nn.Module):

    def __init__(self, a=2., b=4., reduction='none'):
        super(FocalLoss4Center, self).__init__()
        self.a = a
        self.b = b
        self.reduction = reduction

    def forward(self, pconf, gconf):
        loss = focal_loss4center(pconf, gconf, self.reduction, self.a, self.b)
        return loss


def focalloss_v2(pconf, gconf, alpha=0.25, gamma=2, reduction='none', is_merge=True):
    '''
    这个与公式相同,正式版本 适用于GT=1的情况
    :param pconf:
    :param gconf: 这个必须归一化
    :param alpha: 正反例比例  0.25/0.75 正反例比 1/3
    :param gamma: 越小 是弧度向上
    :param reduction:
    :return:
    '''
    # eps = torch.finfo(torch.float).eps
    eps = 1e-6
    pconf = pconf.clamp(min=eps, max=1 - eps)
    # mask_pos = gconf == 1  # 可以直接用gconf 这里确定正反例 >0
    mask_pos = gconf > 0
    mask_neg = torch.logical_not(mask_pos)
    # mask_pos = gconf
    # mask_neg = 1 - gconf

    w_pos = alpha * torch.pow((1 - pconf), gamma)
    l_pos = mask_pos * w_pos * torch.log(pconf)  # 正例

    w_neg = (1 - alpha) * torch.pow(pconf, gamma)
    l_neg = mask_neg * w_neg * torch.log(1 - pconf)  # 反例

    if reduction == 'none' and not is_merge:  # 分离返回
        return -l_pos, -l_neg

    loss = l_pos + l_neg
    if reduction == 'mean':
        return loss.mean(-1)
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def focalloss_v3(pconf, gconf, alpha=0.25, gamma=2, reduction='none', is_merge=True):
    '''
    这个与公式相同,正式版本 适用于GT=任意的情况
    :param pconf:
    :param gconf: 这个必须归一化
    :param alpha: 正反例比例  0.25/0.75 正反例比 1/3
    :param gamma: 越小 是弧度向上
    :param reduction:
    :return:
    '''

    # mask_pos = gconf == 1  # 可以直接用gconf 这里确定正反例 >0
    mask_pos = gconf > 0
    mask_neg = torch.logical_not(mask_pos)
    # mask_pos = gconf
    # mask_neg = 1 - gconf

    abs_val = torch.abs(gconf - pconf)
    # eps = torch.finfo(torch.float).eps
    eps = 1e-6
    abs_val = abs_val.clamp(min=eps, max=1 - eps)
    val_log = -1 / torch.log(abs_val)

    sc_ = torch.pow(abs_val, gamma)  # 差距越大 难例加成
    w_pos = alpha * sc_  # 正例比例
    l_pos = mask_pos * w_pos * val_log  # 正例

    w_neg = (1 - alpha) * sc_
    l_neg = mask_neg * w_neg * val_log  # 反例
    if reduction == 'none' and not is_merge:  # 分离返回
        return l_pos, l_neg

    loss = l_pos + l_neg
    if reduction == 'mean':
        return loss.mean(-1)
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


class FocalLoss_v2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., is_oned=False, reduction='none'):
        '''
        gt 大于 lt 小于
        ne 不等于 eq 等于
        torch.finfo(torch.float16).eps
        :param gamma: 为1 难易比 扩大10倍
        :param alpha: 0~0.5 降低正样本权重
            (1-pt) ** gamma *log(pt)
        '''
        super(FocalLoss_v2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.is_oned = is_oned
        self.reduction = reduction

    def forward(self, pconf, gconf):
        '''
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        :param pconf: torch.Size([10, 10752])
        :param gconf: torch.Size([10, 10752])
        :return:
        '''
        if self.is_oned:
            _pconf = pconf
        else:
            _pconf = torch.sigmoid(pconf)
        logp = F.binary_cross_entropy_with_logits(pconf, gconf, reduction='none')
        pt = gconf * _pconf + (1 - gconf) * (1 - _pconf)  # >0正例
        modulating_factor = (1.0 - pt) ** self.gamma
        # tensor([0.25000, 0.75000, 0.25000, 0.75000])
        alpha_t = gconf * self.alpha + (1 - gconf) * (1 - self.alpha)
        weight = alpha_t * modulating_factor
        loss = logp * weight

        show_distribution(pconf)

        if self.reduction == 'mean':
            return loss.mean(-1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class FocalLoss4mclass(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='none'):
        super(FocalLoss4mclass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.elipson = 0.000001

    def forward(self, pconf, gconf):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length   torch.Size([32, 3, 10752])
        labels: batch_size * seq_length      torch.Size([32, 10752])
        """
        device = pconf.device
        if gconf.dim() > 2:
            gconf = gconf.contiguous().view(gconf.size(0), gconf.size(1), -1)
            gconf = gconf.transpose(1, 2)
            gconf = gconf.contiguous().view(-1, gconf.size(2)).squeeze()
        if pconf.dim() > 3:
            pconf = pconf.contiguous().view(pconf.size(0), pconf.size(1), pconf.size(2), -1)
            pconf = pconf.transpose(2, 3)
            pconf = pconf.contiguous().view(-1, pconf.size(1), pconf.size(3)).squeeze()
        assert (pconf.size(0) == gconf.size(0))
        assert (pconf.size(2) == gconf.size(1))

        # torch.Size([batch, 2]) -> (batch,class,anc)
        new_label = gconf.unsqueeze(1)
        # 匹配成onehot
        label_onehot = torch.zeros_like(pconf, device=device).scatter_(1, new_label, 1)
        # print(label_onehot)
        # log_softmax能够解决函数overflow和underflow，加快运算速度，提高数据稳定性
        log_p = torch.log_softmax(pconf, dim=1)
        # log_p = torch.softmax(pconf, dim=1)
        # print('torch.log_softmax\n', log_p)
        pt = label_onehot * log_p
        loss_v = -self.alpha * ((1 - pt) ** self.gamma) * log_p
        if self.reduction == 'mean':
            # [32, 3, 10752] -> [32, 10752]
            return loss_v.mean(dim=1).mean(dim=1)
        elif self.reduction == 'sum':
            return loss_v.mean(dim=1).sum(dim=1)
        else:  # 'none'  [32, 3, 10752] -> [32, 10752]
            return loss_v.mean(dim=1)


def f_ohem(scores, nums_neg, mask_pos, mash_ignore, pboxes_ltrb=None, threshold_iou=0.7):
    '''

    :param pboxes_ltrb:  (batch,anc)
    :param scores:gconf  (batch,anc)
    :param nums_neg: tensor([   3,    6, 3075,  768,    3,    3,    3,    3,    3,   3])
    :param mask_pos:
    :param threshold_iou:
    :return: 每一批的损失合 (batch)
    '''
    device = pboxes_ltrb.device
    scores_neg = scores.clone().detach()
    # 正样本及忽略 先置0 使其独立 排除正样本,选最大的
    # scores_neg[torch.logical_or(mask_pos, mash_ignore)] = torch.tensor(0.0, device=scores_neg.device)
    scores_neg[mask_pos] = torch.tensor(0.0, device=scores_neg.device)

    # mask_nms = torch.zeros_like(mask_pos, dtype=torch.bool, device=device)
    mask_nms = mask_pos  # 正例全部记损失
    num_batch, num_anc, _ = pboxes_ltrb.shape
    # 只对框框和损失进行nms
    for i in range(num_batch):
        keep = torch.ops.torchvision.nms(pboxes_ltrb[i], scores[i], threshold_iou)
        keep = keep[:nums_neg[i]]
        mask_nms[i, keep] = True  # 反例代表记损失
    loss_confs = (scores * mask_nms.float()).sum(dim=-1)

    # 批量nms
    # mask_nms_ = torch.zeros_like(mask_pos, dtype=torch.bool, device=device)
    # keep, ids_keep = batched_nms_auto(boxes_ltrb, scores=scores, threshold_iou=threshold_iou)
    # # mask_nms_[ids_keep][keep] = True # 批量全处理 无法对每批操作
    # for i in range(num_batch):
    #     mask = ids_keep == i
    #     if torch.any(mask):
    #         k = keep[mask][:nums_neg[i]]
    #         mask_nms_[i][k - i * num_anc] = True

    return loss_confs


def f_ohem_simpleness(scores, nums_neg, mask_pos, mash_ignore):
    '''

    :param scores:
    :param nums_neg:  负例数 = 推荐正例 * 3
    :param mask_pos:
    :return:
    '''
    scores_neg = scores.clone().detach()
    # 正样本及忽略 先置0 使其独立 排除正样本,选最大的
    scores_neg[torch.logical_or(mask_pos, mash_ignore)] = torch.tensor(0.0, device=scores_neg.device)

    '''-----------简易版难例----------'''
    _, l_sort_ids = scores_neg.sort(dim=-1, descending=True)  # descending 倒序
    # 得每一个图片batch个 最大值索引排序
    _, l_sort_ids_rank = l_sort_ids.sort(dim=-1)  # 两次索引排序 用于取最大的n个布尔索引
    # 计算每一层的反例数   [batch] -> [batch,1]  限制正样本3倍不能超过负样本的总个数  基本不可能超过总数
    # neg_num = torch.clamp(self.neg_ratio * pos_num, max=mask_pos.size(1)).unsqueeze(-1)
    nums_neg = nums_neg.unsqueeze(-1)
    mask_neg = l_sort_ids_rank < nums_neg  # 选出最大的n个的mask  Tensor [batch, 8732]
    # 正例索引 + 反例索引 得1 0 索引用于乘积筛选
    mask_val = mask_pos.float() + mask_neg.float()
    loss_confs = (scores * mask_val).sum(dim=-1)
    return loss_confs


def ohem_loss(pred, target, num_keep):
    loss = torch.nn.NLLLoss(reduce=False)(torch.log(pred), target)
    print(loss)
    loss_sorted, idx = torch.sort(loss, descending=True)
    loss_keep = loss_sorted[:num_keep]
    return loss_keep.sum() / num_keep


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


def f_ghmc_v3(pconf, gconf, mask=None, num_bins=10, momentum=0., nums_edges_last=None, reduction='none'):
    device = pconf.device
    if mask is None:  # 全部加入运算
        mask = torch.ones_like(pconf, dtype=torch.bool, device=device)
        # mask = gconf > 0
    g = torch.abs(pconf.detach() - gconf)  # 梯度模长 越大越难 恰好也是反向的梯度

    float_eps = torch.finfo(torch.float16).eps
    inds_edges = torch.floor(g * (num_bins - float_eps)).long()  # 数据对应在哪个格子中 格子index
    nums_edges = torch.zeros(num_bins, device=device)  # 区间样本数量
    for i in range(num_bins):
        nums_edges[i] = (inds_edges == i).sum().item()
    print('nums_edges', nums_edges)

    if nums_edges_last is None:
        nums_edges_last = nums_edges
    else:
        nums_edges = momentum * nums_edges_last + (1 - momentum) * nums_edges
        nums_edges_last = nums_edges

    n = mask.sum().item()  # 总样本数

    num_bins = (nums_edges > 0).sum().item()

    gd = nums_edges * num_bins
    gd = torch.clamp(gd, min=float_eps)
    weights = n / gd

    loss = F.binary_cross_entropy_with_logits(pconf, gconf, weights[inds_edges], reduction=reduction)
    return loss, nums_edges_last


class GHM_Loss_Base(nn.Module):

    def __init__(self, num_bins, momentum, reduction='none'):
        super().__init__()
        self.num_bins = num_bins
        self.reduction = reduction
        self.momentum = momentum  # 权重放大的作用 0~1 起放大作用 超过1变负值
        self.nums_edges_last = None

    def i_calc_gradient_length(self, pconf, gconf):
        raise NotImplementedError

    def i_loss_fun(self, pconf, gconf, weight):
        raise NotImplementedError

    def forward(self, pconf, gconf, mask=None):
        device = pconf.device
        if mask is None:  # 默认全部加入运算
            mask = torch.ones_like(pconf, dtype=torch.bool, device=device)
        float_eps = torch.finfo(torch.float16).eps
        g = self.i_calc_gradient_length(pconf, gconf)  # 梯度模长 越大越难 恰好也是反向的梯度

        # 计算梯度的 区间索引
        inds_edges = torch.floor(g * (self.num_bins - float_eps)).long().to(device)  # 数据对应在哪个格子中 格子index
        # 统计各区间样本数量
        nums_edges = torch.zeros(self.num_bins, device=device)
        for i in range(self.num_bins):
            nums_edges[i] = (inds_edges == i).sum().item()

        if self.nums_edges_last is None:
            self.nums_edges_last = nums_edges
        else:  # 前动量引入
            nums_edges = self.momentum * self.nums_edges_last + (1 - self.momentum) * nums_edges
            self.nums_edges_last = nums_edges

        n = mask.sum().item()  # 总样本数
        '''------调试代码--------'''
        flog.debug('GHM_Loss:%s', [round(d.item(), 2) for d in nums_edges])  # 梯度分布
        show_distribution(pconf)  # 在 pconf 分布
        # 统计有值的区间个数
        num_bins = (nums_edges > 0).sum().item()

        gd = nums_edges * num_bins  # 梯度
        gd = torch.clamp(gd, min=float_eps)
        weights = n / gd

        loss = self.i_loss_fun(pconf, gconf, weights[inds_edges])
        return loss


class GHMC_Loss(GHM_Loss_Base):

    def __init__(self, num_bins=10, momentum=0., reduction='none'):
        super(GHMC_Loss, self).__init__(num_bins=num_bins, momentum=momentum, reduction=reduction)

    def i_calc_gradient_length(self, pconf, gconf):
        return torch.abs(pconf.detach() - gconf)

    def i_loss_fun(self, pconf, gconf, weight):
        return f_bce(pconf, gconf, weight)


class GHMR_Loss(GHM_Loss_Base):
    def __init__(self, mu, num_bins=10, momentum=0., reduction='none'):
        super(GHMR_Loss, self).__init__(num_bins=num_bins, momentum=momentum, reduction=reduction)
        self.mu = mu

    def i_calc_gradient_length(self, pconf, gconf):
        d = pconf - gconf
        mu = self.mu
        return d / torch.sqrt(d * d + mu * mu)

    def i_loss_fun(self, pconf, gconf, weight):
        d = pconf - gconf
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        return loss * weight


def t_ghm():
    # 测试
    pconf = torch.arange(1.0, -0.01, -0.1)
    gconf = torch.zeros_like(pconf)

    # print('--------------- 1-0.05,  1-0.5,  1-0.8,  0.5-0.56, 0.5-0.05, 0.1-0.9, 0-0.95,0-0.5, 0-0.2,  0-0.1')
    # print('--------------- 大,  中,  小,  0.5超小, 0.5小, 0.1超大, 中,小中, 小,  大')

    print('--------------- 超小,  中,  大,  超大')
    input_1 = torch.tensor([0.56, 0.05, 0.9, 0.05, 0.5, 0.8, 0.95, 0.5, 0.2, 0.1, 0.99])
    target_1 = torch.tensor([0.5, 0.5, 0.1, 1, 1, 1., 0, 0, 0, 0, 0])

    # input_1 = torch.torch.Tensor([[0.05, 0.25, .5], [0.15, 0.65, .75]])
    # target_1 = torch.Tensor([[1.0, 0.0, 1.], [0.0, 1.0, 1.]])

    input_2 = torch.Tensor([[0.75, 0.65], [0.85, 0.05]])
    target_2 = torch.Tensor([[1.0, 0.0], [0.0, 0.0]])

    # ghmc = GHMC(num_bins=2, momentum=0.75)
    # print('ghmc', ghmc(pconf, gconf))
    print('ghmc', f_ghmc_v3(input_1, target_1))
    # print('ghmc', f_ghmc(input_1, target_1, momentum=0.75))
    ghmc_v2 = GHMC_Loss(num_bins=10, momentum=0.75)
    print('GHMC_Loss', ghmc_v2(input_1, target_1))
    print('bce', F.binary_cross_entropy(input_1, target_1, reduction='none'))
    # obj_FocalLoss_v2 = FocalLoss_v2()
    print('focalloss_v2', focalloss_v2(input_1, target_1))
    v_ = focalloss_v3(input_1, target_1)
    # print('focalloss_v3', v_)
    show_res(v_, input_1, target_1, 'f_v3')


def show_res(v_, input, target, name):
    s = []
    for i in range(len(v_)):
        _t = '{:.4f}({:.2f},{:.2f})'.format(v_[i], input[i], target[i])
        s.append(_t)
    print(name, '\t'.join(s))


if __name__ == '__main__':
    import numpy as np

    np.random.seed(20201031)

    t_ghm()
    # t_focal_loss()

    # t_多值交叉熵()
    # f_二值交叉熵2()
    # f_二值交叉熵1()
