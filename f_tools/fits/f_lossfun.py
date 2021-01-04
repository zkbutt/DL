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
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import f_show_3box4pil, show_anc4pil

torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')


def x_bce(i, o):
    '''
    同维
    :param i: 值必须为0~1之间 float
    :param o: 值为 float
    :return:
    '''
    return np.round(-(o * np.log(i) + (1 - o) * np.log(1 - i)), 4)


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

    torch.manual_seed(7)
    # pconf = torch.tensor([1.1] * 9)
    # gconf = torch.zeros_like(pconf)

    gconf = torch.tensor([1, 0, 1., 0])
    pconf = torch.tensor([0.95, 0.05, 0.5, 0.5])

    # pconf = torch.arange(1.0, -0.01, -0.1)
    # pconf = torch.rand((10000))
    # gconf = torch.zeros_like(pconf)
    loss_show = F.binary_cross_entropy(pconf, gconf, reduction='none')
    # loss_show = F.binary_cross_entropy(pconf, gconf, reduction='sum')
    print('bce_neg', loss_show)

    # bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    # obj_FocalLoss = FocalLoss(torch.nn.BCELoss(reduction='none'), alpha=alpha, gamma=gamma)
    # print('obj_FocalLoss', obj_FocalLoss(pconf, gconf))
    obj_FocalLoss_v2 = FocalLoss_v2(is_oned=True, alpha=alpha, gamma=gamma)
    print('FocalLoss_v2_neg', obj_FocalLoss_v2(pconf, gconf))
    a, b = alpha, gamma
    print('focal_loss4center_neg', focal_loss4center(pconf, gconf, a=a, b=b))

    print('-----------------------下面是正例------------------------------------')
    pconf = torch.arange(0, 1.01, 0.1)
    gconf = torch.ones_like(pconf)

    loss_show = F.binary_cross_entropy(pconf, gconf, reduction='none')
    # loss_show = F.binary_cross_entropy(pconf, gconf, reduction='sum')
    print('bce_pos', loss_show)

    # print('obj_FocalLoss_pos', obj_FocalLoss(pconf, gconf))
    print('obj_FocalLoss_v2_pos', obj_FocalLoss_v2(pconf, gconf))
    print('focal_loss4center_pos', focal_loss4center(pconf, gconf, a=a, b=b))


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

    obj_fl4m = FocalLoss(nn.CrossEntropyLoss, alpha=0.25, gamma=2)
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

    loss1 = F.binary_cross_entropy(torch.sigmoid(input), target, reduction='none')  # 独立的
    # loss1 = F.binary_cross_entropy(input, target, reduction='none')  # 独立的
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
    # print(loss)

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


# class FocalLoss(nn.Module):
#     '''
#     这个没有正负样本比例
#     '''
#
#     def __init__(self, ceobj, alpha=0.25, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         # self.ce = nn.CrossEntropyLoss(reduction='none')
#         self.ce = ceobj  # obj 和 fun
#         self.alpha = alpha
#
#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         pt = torch.exp(-logp)
#         loss = (1 - pt) ** self.gamma * logp
#         loss = self.alpha * loss
#         # 每一批的损失和
#         return loss


def focal_loss4center(pconf, gconf, reduction='none', a=2., b=4.):
    eps = torch.finfo(torch.float).eps
    pconf = pconf.clamp(min=eps, max=1 - eps)
    l_pos = gconf * torch.pow((1 - pconf), a) * torch.log(pconf)
    l_neg = (1 - gconf) * torch.pow((1 - gconf), b) * torch.pow(pconf, a) * torch.log(1 - pconf)
    loss = -(l_pos + l_neg)

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


class FocalLoss_v2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., is_oned=False, reduction='none'):
        '''
        gt 大于 lt 小于
        ne 不等于 eq 等于
        torch.finfo(torch.float32).eps
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
            logp = F.binary_cross_entropy(pconf, gconf, reduction='none')
        else:
            _pconf = torch.sigmoid(pconf)
            logp = F.binary_cross_entropy_with_logits(pconf, gconf, reduction='none')
        pt = gconf * _pconf + (1 - gconf) * (1 - _pconf)
        modulating_factor = (1.0 - pt) ** self.gamma
        # tensor([0.25000, 0.75000, 0.25000, 0.75000])
        alpha_t = gconf * self.alpha + (1 - gconf) * (1 - self.alpha)
        weight = alpha_t * modulating_factor
        loss = logp * weight

        if self.reduction == 'mean':
            return loss.mean(-1)
        elif self.reduction == 'sum':
            return loss.sum(-1)
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


class LossYOLOv3(nn.Module):

    def __init__(self, anc_obj, cfg):
        super(LossYOLOv3, self).__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj
        self.focal_loss = FocalLoss_v2(alpha=cfg.FOCALLOSS_ALPHA, gamma=cfg.FOCALLOSS_GAMMA, reduction='none')

    def hard_avg(self, cfg, values, mask_ignore, mask_pos):
        '''
        mask_ignore 没有算 反倒比例+难例
        :param values: 25200
        :param mask_ignore:
        :param mask_pos:
        :return:
        '''
        device = values.device
        _values = values.clone().detach()
        _values[torch.logical_or(mask_ignore, mask_pos)] = torch.tensor(0.0, device=device)

        _, max_idx1 = _values.sort(dim=-1, descending=True)  # descending 倒序
        _, max_idx2 = max_idx1.sort(dim=-1)
        num_pos = mask_pos.sum()
        num_neg = num_pos.item() * cfg.NEG_RATIO
        mask_neg = max_idx2 < num_neg

        # mask = mask_pos.float() + mask_neg.float()
        # 正反例可以一起算 
        l_conf_neg = (values * mask_neg.float()).sum() / num_pos
        l_conf_pos = (values * mask_pos.float()).sum() / num_pos
        return l_conf_pos, l_conf_neg

    def forward(self, p_yolo_ts, targets, imgs_ts=None):
        ''' 只支持相同的anc数

        :param p_yolo_ts: [5, 10647, 25])
        :param targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['size'] = target['size']
            target['image_id'] = int
        :return:
        '''

        # loss 基本参数
        batch = len(targets)
        device = p_yolo_ts.device
        # 层尺寸   tensor([[52., 52.], [26., 26.], [13., 13.]])
        # feature_sizes = np.array(self.anc_obj.feature_sizes)
        feature_sizes = torch.tensor(self.anc_obj.feature_sizes, dtype=torch.float32, device=device)
        # tensor([8112, 2028,  507], dtype=torch.int32)
        nums_feature_offset = feature_sizes.prod(dim=1) * torch.tensor(self.cfg.NUMS_ANC, device=device)  # 2704 676 169

        # 2704 676 169 -> tensor([8112, 2028,  507])
        nums_ceng = (feature_sizes.prod(axis=1) * 3).type(torch.int)
        # 转为np 用数组索引 tensor([[52., 52.], [26., 26.],[13., 13.]]) -> torch.Size([10647, 2]) 数组索引 只能用np
        fsize_p_anc = np.repeat(feature_sizes.cpu(), nums_ceng.cpu(), axis=0)
        fsize_p_anc = fsize_p_anc.clone().detach().to(device)  # cpu->gpu 安全

        # 匹配完成的数据
        _num_total = sum(nums_feature_offset)  # 10647
        _dim = 4 + 1 + self.cfg.NUM_CLASSES  # 25
        nums_feature_offset[2] = nums_feature_offset[0] + nums_feature_offset[1]
        nums_feature_offset[1] = nums_feature_offset[0]
        nums_feature_offset[0] = 0

        loss_cls_pos = 0
        loss_box_pos = 0  # 只计算匹配的
        loss_conf_pos = 0  # 正反例， 正例*100 取难例
        loss_conf_neg = 0  # 正反例， 正例*100 取难例

        # 分批处理
        for i in range(batch):
            '''------------------负样本选择-------------------'''
            target = targets[i]
            # 只能一个个的处理
            g_boxes_ltrb = target['boxes']  # ltrb
            g_labels = target['labels']
            if g_boxes_ltrb.shape[0] == 0:
                # 没有目标的图片不要
                flog.error('boxes==0 %s', g_boxes_ltrb.shape)
                continue
            if self.cfg.IS_VISUAL:
                # 可视化1 原目标图 --- 初始化图片
                img_ts = imgs_ts[i]
                from torchvision.transforms import functional as transformsF
                img_ts = f_recover_normalization4ts(img_ts)
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                # show_anc4pil(img_pil, g_boxes_ltrb, size=img_pil.size)
                # img_pil.save('./1.jpg')

            '''组装 BOX 对应的 n*9个 匹配9个anc 计算出网格index'''
            g_boxes_xywh = ltrb2xywh(g_boxes_ltrb)
            num_anc = np.array(self.cfg.NUMS_ANC).sum()  # anc总和数 [3,3,3].sum() -> 9
            # 单体复制 每一个box重复9次
            g_boxes_xywh_p = g_boxes_xywh.repeat_interleave(num_anc, dim=0)
            # 每一个bbox对应的9个anc tensor([[52., 52.], [26., 26.], [13., 13.]])
            # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True) torch复制
            # feature_sizes = torch.tensor(feature_sizes.copy(), dtype=torch.float32).to(device)
            # 与多个box匹配 只支持每层相同的anc数(索引不支持数组) [[52., 52.],[52., 52.],[52., 52.], ...[26., 26.]..., [13., 13.]...]
            num_boxes = g_boxes_xywh.shape[0]
            # 这里 fsize_p n*9个 单体 + 整体
            fsize_p_n9 = feature_sizes.repeat_interleave(self.cfg.NUMS_ANC[0], dim=0).repeat(num_boxes, 1)
            # XY对就 col rows
            colrow_index = get_boxes_colrow_index(g_boxes_xywh_p[:, :2], fsize_p_n9)

            '''构造 与输出匹配的 n*9个anc'''
            # 求出anc对应网络的的中心点  在网格左上角
            _ancs_xy = colrow_index / fsize_p_n9  # tensor([[52., 52.], 52., 52.], [52., 52.],[26., 26.],[26., 26.],[26., 26.],[13., 13.],[13., 13.],[13., 13.]])
            # 大特图对应小目标 _ancs_scale 直接作为wh
            _ancs_scale = torch.tensor(self.anc_obj.ancs_scale).to(device)
            _ancs_wh = _ancs_scale.reshape(-1, 2).repeat(num_boxes, 1)  # 拉平后整体复制
            ancs_xywh = torch.cat([_ancs_xy, _ancs_wh], dim=1)

            # --------------可视化调试----------------
            if self.cfg.IS_VISUAL:
                # 显示 boxes 中心点 黄色, 及计算出来的匹配 anc 的位置 3个层的中心点是不一样的 显示的大框 在网格左上角
                # ancs_ltrb = xywh2ltrb(ancs_xywh)
                # f_show_3box4pil(img_pil, g_boxes=g_boxes_ltrb,  # 黄色
                #                 boxes1=ancs_ltrb[:3, :],
                #                 boxes2=ancs_ltrb[3:6, :],
                #                 boxes3=ancs_ltrb[6:, :],
                #                 grids=self.anc_obj.feature_sizes[-1])
                pass

            '''批量找出每一个box对应的anc index'''
            # 主动构建偏移 使每个框的匹配过程独立 tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0')
            ids = torch.arange(num_boxes).to(device)  # 9个anc 0,1,2 只支持每层相同的anc数
            # (n,4) + (n,1)---(n,4) = (n,4)  扩展
            g_boxes_ltrb_offset = g_boxes_ltrb + ids[:, None]  # boxes加上偏移 1,2,3
            # 单体复制 对应anc数 1,2,3 -> 000000000 111111111 222222222
            ids_offset = ids.repeat_interleave(num_anc)
            # 与box 匹配的 anc
            ancs_ltrb = xywh2ltrb(ancs_xywh)  # 转ltrb 用于计算iou
            # 两边都加了同样的偏移
            ancs_ltrb_offset = ancs_ltrb + ids_offset[:, None]
            iou = calc_iou4ts(g_boxes_ltrb_offset, ancs_ltrb_offset, is_ciou=True)  # 这里都等于0
            # iou_sort = torch.argsort(-iou, dim=1) % num_anc  # 9个
            # iou_sort = torch.argsort(-iou, dim=1) % num_anc  # 求余数
            _, max_indexs = iou.max(dim=1)  # box对应anc的 index
            max_indexs = max_indexs % num_anc  # index 偏移值的修正

            '''---- 整体修复box  anc左上角+偏移  ---'''
            # 找出与pbox  与gbox 最大的index
            p_box_xywh = fix_boxes4yolo3(p_yolo_ts[i, :, :4], self.anc_obj.ancs, fsize_p_anc)

            p_box_ltrb = xywh2ltrb(p_box_xywh)
            ious = calc_iou4ts(g_boxes_ltrb, p_box_ltrb, is_ciou=True)  # 全部计算IOU
            # p_box_ltrb 对应的最大的 iou 值
            max_ious, _ = ious.max(dim=0)

            num_feature = len(self.cfg.NUMS_ANC)
            # 计算最大 anc 索引对应在哪个特图层
            match_feature_index = torch.true_divide(max_indexs, num_feature).type(torch.int64)  #
            # tensor([8112., 2028.,  507.]) ->[0., 8112,  2028.+8112.]
            num_feature_offset = nums_feature_offset[match_feature_index]
            colrow_num = colrow_index[torch.arange(num_boxes, device=device) * num_anc + max_indexs]
            _row_index = colrow_num[:, 1]  # [1,2]
            _col_index = colrow_num[:, 0]  # [3,2]
            # 特图层的行列偏移
            offset_colrow = _row_index * feature_sizes[match_feature_index][:, 0] + _col_index
            # 对应物图层的获取anc数
            _nums_anc = torch.tensor(self.cfg.NUMS_ANC, device=device)[match_feature_index]
            offset_total = num_feature_offset + offset_colrow * _nums_anc
            '''这里这个 match_index_pos 有可能已经用了'''
            match_index_pos = (offset_total + max_indexs % num_feature).type(torch.int64)

            '''---------  conf 损失 ---------'''
            # ----------选出难例负例----------
            mask_neg = max_ious > self.cfg.THRESHOLD_CONF_NEG
            mask_ignore = torch.logical_not(mask_neg)
            pconf = p_yolo_ts[i, :, 4]
            # p_yolo_ts[mask_ignore] = 0  # 忽略正反例
            # print(pconf[match_index_pos].sigmoid().tolist())
            gconf = torch.zeros_like(pconf)
            gconf[match_index_pos] = 1  # 25200
            mask_pos = gconf.type(torch.bool)
            _l_conf = F.binary_cross_entropy_with_logits(pconf, gconf, reduction='none')
            _l_conf_pos, _l_conf_neg = self.hard_avg(self.cfg, _l_conf, mask_ignore, mask_pos)

            # l_conf = self.focal_loss(pconf, gconf)
            # match_index_neg = fmatch_OHEM(l_conf, match_index_pos,
            #                               neg_ratio=self.cfg.NEG_RATIO, num_neg=2000, device=device, dim=-1)
            # _l_conf_pos += (l_conf.sum() / num_boxes)

            # focalloss
            # pconf = p_yolo_ts[i, :, 4]
            # # print(pconf[match_index_pos].sigmoid().tolist())
            # gconf = torch.zeros_like(pconf)
            # gconf[match_index_pos] = 1  # 25200
            # # _l_conf += (self.focal_loss(pconf, gconf) / num_boxes)
            # _l_conf += (self.focal_loss(pconf, gconf))

            '''-------- 计算正例 cls 损失 -----------'''
            pcls = p_yolo_ts[i, match_index_pos, 5:]
            # label是从1开始 找出正例
            _t = labels2onehot4ts(g_labels - 1, self.cfg.NUM_CLASSES)
            _l_cls_pos = (F.binary_cross_entropy_with_logits(pcls, _t.type(torch.float), reduction='sum') / num_boxes)
            # _l_cls_pos += (self.focal_loss(pcls, _t.type(torch.float)) / num_boxes)
            # _l_cls_pos += (self.focal_loss(pcls, _t.type(torch.float)))

            '''----- 正例box 损失 -----'''
            ious = bbox_iou4one(g_boxes_ltrb, p_box_ltrb[match_index_pos, :], is_ciou=True)
            # ious = bbox_iou4one(g_boxes_ltrb, p_box_ltrb[match_index_pos, :], is_giou=True)
            # ious = bbox_iou4one(g_boxes_ltrb, p_box_ltrb[match_index_pos, :], is_diou=True)
            w = 2 - g_boxes_xywh[:, 2] * g_boxes_xywh[:, 3]  # 增加小目标的损失权重
            _l_box_pos = torch.mean(w * (1 - ious))  # 每个框的损失
            # _l_box_pos += torch.sum(w * (1 - ious))  # 每个框的损失

            if self.cfg.IS_VISUAL:
                flog.debug('conf neg 损失 %s', _l_conf_neg)
                flog.debug('box 损失 %s', torch.mean(w * (1 - ious)))
                flog.debug('conf pos 损失 %s', _l_conf_pos)
                flog.debug('cls 损失 %s', _l_cls_pos)
                flog.debug('-------------------')
                f_show_3box4pil(img_pil, g_boxes=g_boxes_ltrb,
                                boxes1=p_box_ltrb[match_index_pos, :],
                                boxes2=xywh2ltrb(self.anc_obj.ancs[match_index_pos, :]),
                                grids=self.anc_obj.feature_sizes[-1],  # 网格
                                )

            loss_cls_pos = loss_cls_pos + _l_cls_pos
            loss_box_pos = loss_box_pos + _l_box_pos  # 上面已平均
            loss_conf_pos = loss_conf_pos + _l_conf_pos
            loss_conf_neg = loss_conf_neg + _l_conf_neg

        l_box_p = loss_box_pos / batch * self.cfg.LOSS_WEIGHT[0]
        l_conf_pos = loss_conf_pos / batch * self.cfg.LOSS_WEIGHT[1]
        l_conf_neg = loss_conf_neg / batch * self.cfg.LOSS_WEIGHT[2]
        l_cls_p = loss_cls_pos / batch * self.cfg.LOSS_WEIGHT[3]
        loss_total = l_box_p + l_conf_pos + l_conf_neg + l_cls_p

        # debug
        # _v = p_yolo_ts[:, :, 4].clone().detach().sigmoid()
        # print('min:%s mean:%s max:%s' % (_v.min().item(),
        #                                  _v.mean().item(),
        #                                  _v.max().item(),
        #                                           ))

        log_dict = {}
        log_dict['loss_total'] = loss_total.item()
        log_dict['l_box_p'] = l_box_p.item()
        log_dict['l_conf_p'] = l_conf_pos.item()
        log_dict['l_conf_n'] = l_conf_neg.item()
        log_dict['l_cls_p'] = l_cls_p.item()
        # log_dict['l_conf_min'] = _v.min().item()
        # log_dict['l_conf_mean'] = _v.mean().item()
        # log_dict['l_conf_max'] = _v.max().item()
        # log_dict['l_conf_top100'] = _v.topk(10).values.tolist() # 不支持数组
        # print(_v.topk(20).values.tolist())

        return loss_total, log_dict


def f_ohem(scores, nums_neg, mask_pos, boxes_ltrb=None, threshold_iou=0.7):
    '''

    :param boxes_ltrb:  (batch,anc)
    :param scores:gconf  (batch,anc)
    :param nums_neg: tensor([   3,    6, 3075,  768,    3,    3,    3,    3,    3,   3])
    :param mask_pos:
    :param threshold_iou:
    :return: 每一批的损失合 (batch)
    '''
    device = boxes_ltrb.device
    scores_neg = scores.clone().detach()
    # 正样本及忽略 先置0 使其独立 排除正样本,选最大的
    scores_neg[mask_pos] = torch.tensor(0.0, device=scores_neg.device)

    mask_nms = torch.zeros_like(mask_pos, dtype=torch.bool, device=device)
    num_batch, num_anc, _ = boxes_ltrb.shape
    for i in range(num_batch):
        keep = torch.ops.torchvision.nms(boxes_ltrb[i], scores[i], threshold_iou)
        keep = keep[:nums_neg[i]]
        mask_nms[i, keep] = True
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


def f_ohem_simpleness(scores, nums_neg, mask_pos):
    '''

    :param scores:
    :param nums_neg:  负例数 = 推荐正例 * 3
    :param mask_pos:
    :return:
    '''
    scores_neg = scores.clone().detach()
    # 正样本及忽略 先置0 使其独立 排除正样本,选最大的
    scores_neg[mask_pos] = torch.tensor(0.0, device=scores_neg.device)
    '''-----------简易版难例----------'''
    _, l_sort_ids = scores_neg.sort(dim=-1, descending=True)  # descending 倒序
    # 得每一个图片batch个 最大值索引排序
    _, l_sort_ids_rank = l_sort_ids.sort(dim=-1)  # 两次索引排序 用于取最大的n个布尔索引
    # 计算每一层的反例数   [batch] -> [batch,1]  限制正样本3倍不能超过负样本的总个数  基本不可能超过总数
    # neg_num = torch.clamp(self.neg_ratio * pos_num, max=mask_pos.size(1)).unsqueeze(-1)
    nums_neg = (nums_neg).unsqueeze(-1)
    mask_neg = l_sort_ids_rank < nums_neg  # 选出最大的n个的mask  Tensor [batch, 8732]
    # 正例索引 + 反例索引 得1 0 索引用于乘积筛选
    mask_val = mask_pos.float() + mask_neg.float()
    loss_confs = (scores * mask_val).sum(dim=-1)
    return loss_confs


if __name__ == '__main__':
    import numpy as np

    np.random.seed(20201031)

    # t_focal_loss()

    # t_多值交叉熵()
    f_二值交叉熵2()
    # f_二值交叉熵1()
