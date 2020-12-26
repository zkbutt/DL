import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from torch.nn import BCEWithLogitsLoss

from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import fmatch_OHEM
from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2ltwh, diff_bbox, diff_keypoints, ltrb2xywh, calc_iou4ts, \
    bbox_iou4one, xy2offxy, offxy2xy, get_boxes_colrow_index, fix_boxes4yolo3
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import show_bbox4ts, f_show_3box4pil, show_anc4pil

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
    alpha, gamma = 0.25, 1.5  # 上了 0.5 就减损失
    alpha, gamma = 0.75, 5  # 上了 0.5 就减损失
    alpha, gamma = 0.5, 0.001  # 上了 0.5 就减损失
    # alpha, gamma = 0.9998, 1  # gamma =0  失效
    # alpha, gamma = 0.7, 0.5  # gamma =0  失效

    torch.manual_seed(7)

    pconf = torch.arange(1.0, -0.01, -0.1)
    # pconf = torch.rand((10000))
    gconf = torch.zeros_like(pconf)
    loss_show = F.binary_cross_entropy(pconf, gconf, reduction='none')
    # loss_show = F.binary_cross_entropy(pconf, gconf, reduction='sum')
    print('bce_neg', loss_show)

    # bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    # fun_loss = FocalLoss_v2(alpha=alpha, gamma=gamma, is_oned=True, reduction='none')
    fun_loss = FocalLoss4Center(a=2, b=4., reduction='none')
    loss_show = fun_loss(pconf, gconf)
    # print(loss_show)
    print('FocalLoss4Center_neg', loss_show)
    print('focal_loss4center_neg', focal_loss4center(pconf, gconf, a=2, b=4))

    pconf = torch.arange(0, 1.01, 0.1)
    gconf = torch.ones_like(pconf)

    loss_show = F.binary_cross_entropy(pconf, gconf, reduction='none')
    # loss_show = F.binary_cross_entropy(pconf, gconf, reduction='sum')
    print('bce_pos', loss_show)
    print('focal_loss4center_pos', focal_loss4center(pconf, gconf, a=2, b=4))
    # loss_show = fun_loss(pconf, gconf)
    # print(loss_show)

    # pconf = torch.rand((1))
    # print(pconf)
    # fun_loss = FocalLoss_v2(alpha=alpha, gamma=gamma, is_oned=True, reduction='none')
    # fun_loss = FocalLoss_v2(alpha=alpha, gamma=gamma, is_oned=True, reduction='sum')
    loss_show = fun_loss(pconf, gconf)
    print('FocalLoss4Center_pos', loss_show)


def t_多值交叉熵():
    # 2维
    # input = torch.randn(2, 5, requires_grad=True)  # (batch,类别值)
    # # 多分类标签 [3,5) size=(2)
    # target = torch.randint(3, 5, (2,), dtype=torch.int64)  # (batch)  值是3~4
    # loss = F.cross_entropy(input, target, reduction='none')  # 二维需要拉平
    # print(loss)

    # 多维
    input = torch.randn(2, 7, 5, requires_grad=True)  # (batch,类别值,框)
    print(input.shape)
    # # 多分类标签 [3,5) size=(2,5)
    target = torch.randint(3, 5, (2, 5,), dtype=torch.int64)  # (batch,框)
    print(target.shape)
    loss = F.cross_entropy(input, target, reduction='none')  # 二维需要拉平
    print(loss)
    # loss.backward()


def f_二值交叉熵2():
    torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')
    # input 与 target 一一对应 同维
    size = (2, 3)  # 两个数据
    # input = torch.tensor([random.random()] * 6).reshape(*size)
    # print(input)
    # input = torch.randn(*size, requires_grad=True)
    input = torch.tensor([[0.8, 0.8],
                          [0.5, 0.6]], dtype=torch.float)  # 默认是int64
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


def focal_loss4center(pconf, gconf, reduction='none', a=2, b=4):
    '''

    :param pconf:
    :param gconf:
    :param reduction:
    :param threshold_pos: 这个是下拉阀值 越大拉得越低
        [   15.94238,     1.86509,     1.03004,     0.58995,     0.32986,     0.17329,     0.08173,     0.03210,     0.00893,     0.00105,    -0.00000]
    :param threshold_neg:
    :return:
    '''
    pos_inds = gconf.eq(1).float()
    neg_inds = gconf.lt(1).float()
    neg_weights = torch.pow(1 - gconf, b)

    eps = torch.finfo(torch.float).eps
    pconf = pconf.clamp(min=eps, max=1 - eps)

    pos_loss = torch.log(pconf) * torch.pow(1 - pconf, a) * pos_inds
    neg_loss = torch.log(1 - pconf) * torch.pow(pconf, a) * neg_weights * neg_inds

    loss = -(pos_loss + neg_loss)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


class FocalLoss4Center(nn.Module):

    def __init__(self, a=2, b=4., reduction='none'):
        super(FocalLoss4Center, self).__init__()
        self.a = a
        self.b = b
        self.reduction = reduction

    def forward(self, pconf, gconf):
        '''

        :param pconf: Y^ 预测
        :param gconf: Y
        :return:
        '''
        eps = torch.finfo(torch.float).eps
        pconf.clamp_(min=eps, max=1 - eps)
        l_pos = gconf * torch.pow((1 - pconf), self.a) * torch.log(pconf)
        l_neg = (1 - gconf) * torch.pow((1 - gconf), self.b) * torch.pow(pconf, self.a) * torch.log(1 - pconf)
        return -(l_pos + l_neg)


class FocalLoss_v2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., is_oned=False, reduction='none'):
        '''
        gt 大于 lt 小于
        ne 不等于 eq 等于
        torch.finfo(torch.float32).eps
        :param gamma: 0  .1  .2  .5  1.0 2.0 5.0 差距大放大  降低简单样本的权重
        :param alpha: 正样本的权重为0.75，负样本的权重为0.25  不是比例,平横权重
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
        :param pconf:
        :param gconf:
        :return:
        '''
        if self.is_oned:
            _pconf = pconf
        else:
            _pconf = torch.sigmoid(pconf)
        loss_none = F.binary_cross_entropy(_pconf, gconf, reduction='none')
        pt = gconf * _pconf + (1 - gconf) * (1 - _pconf)
        modulating_factor = (1.0 - pt) ** self.gamma
        alpha_factor = gconf * self.alpha + (1 - gconf) * (1 - self.alpha)
        loss_none *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss_none.mean()
        elif self.reduction == 'sum':
            return loss_none.sum()
        else:  # 'none'
            return loss_none


class LossRetinaface(nn.Module):

    def __init__(self, anc, loss_weight=(1., 1., 1.), neg_ratio=3, cfg=None):
        '''

        :param anc: torch.Size([1, 16800, 4])
        :param loss_weight:
        '''
        super().__init__()
        self.location_loss = nn.SmoothL1Loss(reduction='none')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        # self.focal_loss = FocalLoss(torch.nn.BCELoss(reduction='sum'), gamma=2., alpha=0.25)
        self.focal_loss = FocalLoss_v2(alpha=cfg.FOCALLOSS_ALPHA, gamma=cfg.FOCALLOSS_GAMMA)
        self.anc = anc.unsqueeze(dim=0)  # 这里加了一维
        self.loss_weight = loss_weight
        self.neg_ratio = neg_ratio
        self.cfg = cfg

    # def forward(self, p_bboxs_xywh, g_bboxs_ltrb, p_labels, g_labels, p_keypoints, g_keypoints, imgs_ts=None):
    def forward(self, outs, g_targets, imgs_ts=None):
        '''
        归一化的box 和 anc
        :param p_bboxs_xywh: torch.Size([batch, 16800, 4])
        :param g_bboxs_ltrb:
        :param p_labels: torch.Size([5, 16800, 1])
        :param g_labels: torch.Size([batch, 16800])
        :param p_keypoints:
        :param g_keypoints:
        :return:
        '''
        cfg = self.cfg
        p_bboxs_xywh, p_labels, p_keypoints = outs
        g_bboxs_ltrb, g_labels, g_keypoints = g_targets
        # 正样本标签布尔索引 [batch, 16800] 同维运算
        mask_pos = g_labels > 0
        mask_pos_neg = torch.logical_not(g_labels == -1)

        # 匹配正例可视化
        if self.cfg is not None and self.cfg.IS_VISUAL:
            flog.debug('显示匹配的框 %s 个', torch.sum(mask_pos))
            for i, (img_ts, mask) in enumerate(zip(imgs_ts, mask_pos)):
                # 遍历降维 self.anc([1, 16800, 4])
                _t = self.anc.view(-1, 4).clone()
                _t[:, ::2] = _t[:, ::2] * self.cfg.IMAGE_SIZE[0]
                _t[:, 1::2] = _t[:, 1::2] * self.cfg.IMAGE_SIZE[1]
                _t = _t[mask, :]
                show_bbox4ts(img_ts, xywh2ltrb(_t), torch.ones(200))

        # 每一个图片的正样本个数  [batch, 16800] ->[batch]
        pos_num = mask_pos.sum(dim=1)  # [batch] 这个用于batch中1个图没有正例不算损失和计算反例数

        '''-----------bboxs 损失处理-----------'''
        # [1, 16800, 4] ^^ [batch, 16800, 4] = [batch, 16800, 4]
        d_bboxs = diff_bbox(self.anc, ltrb2xywh(g_bboxs_ltrb))

        # [batch, 16800, 4] -> [batch, 16800] 得每一个特图 每一个框的损失和
        loss_bboxs = self.location_loss(p_bboxs_xywh, d_bboxs).sum(dim=-1)  # smooth_l1_loss
        # __d = 1
        # 正例损失过滤
        loss_bboxs = (mask_pos.float() * loss_bboxs).sum(dim=1)

        '''-----------keypoints 损失处理-----------'''
        if g_keypoints is not None:
            d_keypoints = diff_keypoints(self.anc, g_keypoints)
            loss_keypoints = self.location_loss(p_keypoints, d_keypoints).sum(dim=-1)
            # 全0的不计算损失 与正例的布尔索引取交集
            _mask = (mask_pos) * torch.all(g_keypoints > 0, dim=2)  # and一下 将全0的剔除
            loss_keypoints = (_mask.float() * loss_keypoints).sum(dim=1)
        else:
            loss_keypoints = 0
        '''-----------labels 损失处理 难例挖掘-----------'''
        # 三维需 移到中间才能计算 (batch,16800,2) -> (batch,2,16800)
        # batch = p_labels.shape[0]
        # loss_labels = self.focal_loss(p_labels[mask_pos_neg.unsqueeze(dim=-1)], g_labels[mask_pos_neg]) / batch

        '''难例挖掘'''
        # p_labels = p_labels.permute(0, 2, 1)
        # # F.cross_entropy
        #
        # (batch,2,16800)^[batch, 16800] ->[batch, 16800] 得同维每一个元素的损失
        if cfg.NUM_CLASSES == 1:
            loss_labels = F.binary_cross_entropy_with_logits(p_labels.squeeze(-1), g_labels, reduction='none')
            # loss_labels = loss_labels.permute(0, 2, 1)
        else:
            p_labels = torch.softmax(p_labels, dim=-1)
            loss_labels = self.confidence_loss(p_labels.permute(0, 2, 1), g_labels.long())
        # 分类损失 - 负样本选取  选损失最大的
        labels_neg = loss_labels.clone()
        labels_neg[mask_pos] = torch.tensor(0.0).to(loss_labels)  # 正样本先置0 使其独立 排除正样本,选最大的

        # 输入数组 选出反倒损失最大的N个 [33,11,55] -> 降[2,0,1] ->升[1,2,0] < 2 -> [T,F,T] con_idx(Tensor: [N, 8732])
        _, labels_idx = labels_neg.sort(dim=1, descending=True)  # descending 倒序
        # 得每一个图片batch个 最大值索引排序
        _, labels_rank = labels_idx.sort(dim=1)  # 两次索引排序 用于取最大的n个布尔索引
        # 计算每一层的反例数   [batch] -> [batch,1]  限制正样本3倍不能超过负样本的总个数  基本不可能超过总数
        neg_num = torch.clamp(self.neg_ratio * pos_num, max=mask_pos.size(1)).unsqueeze(-1)
        mask_neg = labels_rank < neg_num  # 选出最大的n个的mask  Tensor [batch, 8732]
        # 正例索引 + 反例索引 得1 0 索引用于乘积筛选
        mask_z = mask_pos.float() + mask_neg.float()
        loss_labels = (loss_labels * mask_z).sum(dim=1)

        '''-----------损失合并处理-----------'''
        # 没有正样本的图像不计算分类损失  pos_num是每一张图片的正例个数 eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=torch.finfo(torch.float).eps)  # 求平均 排除有0的情况取类型最小值

        # 计算总损失平均值 /正样本数量 [n] 加上超参系数
        loss_bboxs = (loss_bboxs * num_mask / pos_num).mean(dim=0)
        # __d = 1
        # loss_bboxs.detach().cpu().numpy()
        loss_labels = (loss_labels * num_mask / pos_num).mean(dim=0)
        loss_keypoints = (loss_keypoints * num_mask / pos_num).mean(dim=0)

        loss_total = self.loss_weight[0] * loss_bboxs \
                     + self.loss_weight[1] * loss_labels \
                     + self.loss_weight[2] * loss_keypoints

        log_dict = {}
        log_dict['loss_total'] = loss_total.detach().item()
        log_dict['l_bboxs'] = loss_bboxs.item()
        log_dict['l_labels'] = loss_labels.item()
        log_dict['l_keypoints'] = loss_keypoints.item()
        return loss_total, log_dict


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


class LossYOLOv1(nn.Module):

    def __init__(self, num_cls, grid=7, num_bbox=1, threshold_box=5, threshold_conf_neg=0.5, ):
        '''

        :param grid: 7代表将图像分为7x7的网格
        :param num_bbox: 2代表一个网格预测两个框
        :param threshold_box:
        :param threshold_conf_neg:
        '''
        super(LossYOLOv1, self).__init__()
        self.grid = grid
        self.num_bbox = num_bbox
        self.num_cls = num_cls
        self.threshold_box = threshold_box  # 5代表 λcoord  box位置的坐标预测 权重
        self.threshold_conf_neg = threshold_conf_neg  # 0.5代表没有目标的bbox的confidence loss 权重

    def calc_box_pos_loss(self, pbox, gbox):
        '''

        :param pbox: 已 sigmoid xy
        :param gbox:
        :return:
        '''
        dim = len(pbox.shape)
        if dim == 1:
            # loss_xy = F.mse_loss(torch.sigmoid(pbox[:2]), gbox[:2], reduction='sum')
            loss_xy = F.mse_loss(pbox[:2], gbox[:2], reduction='sum')
            loss_wh = F.mse_loss(pbox[2:4].abs().sqrt(), gbox[2:4].sqrt(), reduction='sum')
        else:
            # loss_xy = F.mse_loss(torch.sigmoid(pbox[:, :2]), gbox[:, :2], reduction='sum')
            loss_xy = F.mse_loss(pbox[:, :2], gbox[:, :2], reduction='sum')
            # 偏移相同的距离 小目录损失要大些
            loss_wh = F.mse_loss(pbox[:, 2:4].abs().sqrt(), gbox[:, 2:4].sqrt(), reduction='sum')
        return loss_xy + loss_wh

    def forward(self, p_yolo_ts, g_yolo_ts):
        '''
        4个损失:

        :param p_yolo_ts: torch.Size([5, 7, 7, 11])
        :param g_yolo_ts: torch.Size([5, 7, 7, 11])
        :return:
        '''
        device = p_yolo_ts.device
        '''生成有目标和没有目标的同维布尔索引'''
        # 获取有GT的框的布尔索引集,conf为1 4或9可作任选一个,结果一样的
        _is = self.num_bbox * 4
        mask_pos = g_yolo_ts[:, :, :, _is:_is + 1] > 0  # 同维(batch,7,7,25) -> (xx,7,7,1)
        # 没有GT的索引 mask ==0  coo_mask==False
        mask_neg = g_yolo_ts[:, :, :, _is:_is + 1] == 0  # (batch,7,7,25) ->(xx,7,7,1)
        # (batch,7,7,1) -> (batch,7,7,11) mask_coo 大部分为0 mask_noo  大部分为1
        mask_pos = mask_pos.expand_as(g_yolo_ts)
        mask_neg = mask_neg.expand_as(g_yolo_ts)

        num_dim = self.num_bbox * 5 + self.num_cls  # g_yolo_ts.shape[-1]
        # 尺寸拉平
        p_pos = p_yolo_ts[mask_pos].view(-1, num_dim).contiguous()
        g_pos = g_yolo_ts[mask_pos].view(-1, num_dim).contiguous()
        # 有目标的样本 (xxx,7,7,25) -> (xxx,xx,25)
        # p_coo_batch = p_coo.view(p_coo.shape[0], -1, num_dim).contiguous()
        # g_coo_batch = g_coo.view(p_coo.shape[0], -1, num_dim).contiguous()

        '''这个可以独立算  计算正例类别loss'''
        # batch拉平一起算,(xxx,25) -> (xxx,20)
        p_cls_pos = p_pos[:, self.num_bbox * 5:].contiguous()
        g_cls_pos = g_pos[:, self.num_bbox * 5:].contiguous()  # 这里全是1
        # loss_cls = F.mse_loss(p_cls_coo, g_cls_coo, reduction='sum')
        # 直接算无需 sigmoid
        loss_cls_pos = F.binary_cross_entropy_with_logits(p_cls_pos, g_cls_pos, reduction='sum')
        # bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        # fun_loss = FocalLoss(bce_loss)
        # loss_val = fun_loss(p_cls_pos, g_cls_pos)

        '''***计算有正例的iou好的那个的 box 损失   （x,y,w开方，h开方）'''
        # (xxx,25) -> (xxx,4)
        mask_pos_ = g_yolo_ts[:, :, :, _is] > 0  # 只要第一个匹配的GT >0即可
        # mask_neg_ = g_yolo_ts[:, :, :, _is] == 0
        ds0, ds1, ds2 = torch.where(mask_pos_)
        p_pos_ = p_yolo_ts[mask_pos_]
        g_pos_ = g_yolo_ts[mask_pos_]
        p_box_pos = p_pos_[:, :self.num_bbox * 4]
        g_box_pos = g_pos_[:, :self.num_bbox * 4]
        loss_box_pos = torch.tensor(0., dtype=torch.float, device=device)
        loss_conf_pos = torch.tensor(0., dtype=torch.float, device=device)

        for i, (pbox2, gbox2) in enumerate(zip(p_box_pos, g_box_pos)):
            colrow_index = torch.tensor([ds2[i], ds1[i]], device=device)
            grids = torch.tensor([self.grid] * 2, device=device)

            '''这个 fix 用于算 box 损失'''
            pbox = pbox2.view(-1, 4)
            p_offxy = torch.sigmoid(pbox[:, :2])  # 预测需要修正
            pbox_offxywh = torch.cat([p_offxy, pbox[:, 2:]], dim=1)
            gbox_offxywh = gbox2.view(-1, 4)

            '''这些用于算正例 conf 这里只是用来做索引 找出正例box'''
            with torch.no_grad():
                _xy = offxy2xy(p_offxy, colrow_index, grids)
                pbox_xywh = torch.cat([_xy, pbox[:, 2:]], dim=1)  # 这个用于算损失
                _xy = offxy2xy(gbox_offxywh[:, :2], colrow_index, grids)
                gbox_xywh = torch.cat([_xy, gbox_offxywh[:, 2:]], dim=1)

                pbox_ltrb = xywh2ltrb(pbox_xywh)  # 这个用于算IOU
                gbox_ltrb = xywh2ltrb(gbox_xywh)
                ious = calc_iou4ts(pbox_ltrb, gbox_ltrb)
                maxiou, ind_pbox = ious.max(dim=0)
                # 这里还需更新gt的 conf  匹配的为1  未匹配的为iou

            if torch.all(gbox_offxywh[0] == gbox_offxywh[1]):  # 两个 GT 是一样的 取最大的IOU
                # 只有一个box 则只选一个pbox 修正降低 ious
                pbox_ = pbox_offxywh[ind_pbox[0]]
                gbox_ = gbox_offxywh[0]
                # 只计算一个conf
                pconf = p_yolo_ts[ds0[i], ds1[i], ds2[i], _is + ind_pbox[1]]
                gconf = torch.tensor(1., dtype=torch.float, device=device)
                # '''计算有目标的置信度损失'''
                loss_conf_pos = loss_conf_pos + F.binary_cross_entropy_with_logits(pconf, gconf, reduction='sum')
            else:
                # 两个都计算  两个都有匹配正例 conf 不作处理
                if ind_pbox[0] == ind_pbox[1]:
                    # 如果 两个box 对一个 pbox 则直接相对匹配
                    pbox_ = pbox_offxywh
                    gbox_ = gbox_offxywh
                else:
                    # 如果 两个box对二个预测 则对应匹配
                    pbox_ = pbox_offxywh[ind_pbox, :]
                    gbox_ = gbox_offxywh
                pconf = p_yolo_ts[ds0[i], ds1[i], ds2[i], _is: _is + 2]  # 取两个 pconf
                gconf = torch.tensor([1.] * 2, dtype=torch.float, device=device)
                # '''计算有目标的置信度损失'''
                loss_conf_pos = loss_conf_pos + F.binary_cross_entropy_with_logits(pconf, gconf, reduction='sum')
            loss_box_pos = loss_box_pos + self.calc_box_pos_loss(pbox_, gbox_)

        '''反例conf损失'''
        # 选出没有目标的所有框拉平 (batch,7,7,30) -> (xx,7,7,30) -> (xxx,30) -> (xxx)
        p_conf_neg = p_yolo_ts[mask_neg].view(-1, num_dim)[:, _is:_is + 2]
        # g_conf_zero = torch.zeros(p_conf_noo.shape)
        g_conf_zero = torch.zeros_like(p_conf_neg)
        # 等价 g_conf_zero = g_yolo_ts[mask_noo].view(-1, num_dim)[:, 4]
        loss_conf_neg = F.binary_cross_entropy_with_logits(p_conf_neg, g_conf_zero, reduction='sum')

        batch = p_yolo_ts.shape[0]  # batch数 shape[0] 这个错了
        loss_box = self.threshold_box * loss_box_pos / batch
        loss_conf = (loss_conf_pos + self.threshold_conf_neg * loss_conf_neg) / batch
        loss_cls_pos = loss_cls_pos / batch
        loss_total = loss_box + loss_conf + loss_cls_pos

        log_dict = {}
        log_dict['loss_box'] = loss_box.item()
        log_dict['loss_conf'] = loss_conf.item()
        log_dict['loss_cls'] = loss_cls_pos.item()
        log_dict['loss_total'] = loss_total.item()
        return loss_total, log_dict


class ObjectDetectionLoss(nn.Module):
    """
        计算目标检测的 定位损失和分类损失
    """

    def __init__(self, ancs, neg_ratio, variance, loss_coefficient):
        '''

        :param ancs:  生成的基础ancs  xywh  基础的只匹配一张图  (m*w*h,4)
        :param neg_ratio: 反倒倍数  超参
        :param variance:  用于控制预测值缩放(0.1, 0.2)  预测值需限制缩小
        '''
        super(ObjectDetectionLoss, self).__init__()
        self.loss_coefficient = loss_coefficient  # 损失系数
        self.neg_ratio = neg_ratio
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        # 用anc 和 gt比较时需要放大差异
        self.scale_xy = 1.0 / variance[0]  # 10 用于控制预测值,使其变小,保持一致需放大
        self.scale_wh = 1.0 / variance[1]  # 5  与GT匹配时需要放大

        # loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # [num_anchors, 4] ->  [1, num_anchors, 4]
        self.ancs = nn.Parameter(ancs.unsqueeze(dim=0), requires_grad=False)

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        # self.confidence_loss = nn.BCELoss(reduction='none')
        # self.confidence_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _bbox_anc_diff(self, g_bboxs):
        '''
        计算ground truth相对anchors的回归参数
            self.dboxes 是def 是xywh self.dboxes
            两个参数只有前面几个用GT替代了的不一样 其它一个值 这里是稀疏
        :param g_bboxs: 已完成 正例匹配 torch.Size([3, 8732, 4])
        :return:
            返回 ground truth相对anchors的回归参数 这里是稀疏
        '''
        # type: (Tensor)
        # 用anc 和 gt比较时需要放大差异
        # 这里 scale_xy 是乘10   scale_xy * (bbox_xy - anc_xy )/ anc_wh
        gxy = self.scale_xy * (g_bboxs[:, :, :2] - self.ancs[:, :, :2]) / self.ancs[:, :, 2:]
        # 这里 scale_xy 是乘5   scale_wh * (bbox_wh - anc_wh ).log()
        gwh = self.scale_wh * (g_bboxs[:, :, 2:] / self.ancs[:, :, 2:]).log()
        return torch.cat((gxy, gwh), dim=-1).contiguous()

    def bboxs_loss(self, p_bboxs, g_bboxs, mask_pos):
        # -------------计算定位损失------------------(只有正样本) 很多0
        # 计算差异
        diff = self._bbox_anc_diff(g_bboxs)
        # 计算损失 [batch, 16800, 4] -> [batch, 16800] 得每一个特图 每一个框的损失和
        loss_bboxs = self.location_loss(p_bboxs, diff).sum(dim=-1)  # smooth_l1_loss
        # 正例损失过滤
        loss_bboxs = (mask_pos.float() * loss_bboxs).sum(dim=1)
        return loss_bboxs

    def labels_loss(self, p_labels, g_labels, mask_pos, pos_num):
        '''

        :param p_labels:
        :param g_labels:
        :param mask_pos: 正样本标签布尔索引 [batch, 16800] 根据gt算出来的
        :param pos_num: [batch]
        :return:
        '''
        # -------------计算分类损失------------------正样本很少
        # p_labels = F.softmax(p_labels, dim=-1)
        # if labels_p.shape[-1]>1:
        # else:
        #     labels_p = F.sigmoid(labels_p)
        # p_labels = F.softmax(p_labels, dim=-1)  # 预测值转 正数 加合为1
        # loss_labels = F.binary_cross_entropy(F.sigmoid(p_labels), g_labels, reduction='none')
        # loss_labels = F.binary_cross_entropy_with_logits(p_labels, g_labels, reduction='none')
        # one_hot = F.one_hot(masks, num_classes=args.num_classes)

        p_labels = p_labels.permute(0, 2, 1)  # (batch,16800,2) -> (batch,2,16800)
        loss_labels = self.confidence_loss(p_labels,
                                           g_labels.long())  # (batch,2,16800)^[batch, 16800] ->[batch, 16800] 得同维每一个元素的损失
        # F.cross_entropy(p_labels, g_labels)
        # 分类损失 - 负样本选取  选损失最大的
        labels_neg = loss_labels.clone()
        labels_neg[mask_pos] = torch.tensor(0.0).to(loss_labels)  # 正样本先置0 使其独立 排除正样本,选最大的

        # 输入数组 选出最大的N个 [33,11,55] -> 降[2,0,1] ->升[1,2,0] < 2 -> [T,F,T] con_idx(Tensor: [N, 8732])
        _, labels_idx = labels_neg.sort(dim=1, descending=True)  # descending 倒序
        # 得每一个图片batch个 最大值索引排序
        _, labels_rank = labels_idx.sort(dim=1)  # 两次索引排序 用于取最大的n个布尔索引
        # 计算每一层的反例数   [batch] -> [batch,1]  限制正样本3倍不能超过负样本的总个数  基本不可能超过总数
        neg_num = torch.clamp(self.neg_ratio * pos_num, max=mask_pos.size(1)).unsqueeze(-1)
        mask_neg = labels_rank < neg_num  # 选出最大的n个的mask  Tensor [batch, 8732]

        # 正例索引 + 反例索引 得1 0 索引用于乘积筛选
        mask_z = mask_pos.float() + mask_neg.float()
        # 总分类损失 Tensor[N] 每个图对应的 loss
        loss_labels = (loss_labels * (mask_z)).sum(dim=1)
        return loss_labels

    def get_loss_list(self, bboxs_p, bboxs_g, labels_p, labels_g, mask_pos, pos_num, *args):
        '''

        :param bboxs_p: 预测的 xywh  torch.Size([batch, 16800, 4])
        :param bboxs_g: 匹配的 xywh  torch.Size([batch, 16800, 4])
        :param labels_p:
        :param labels_g:匹配的 torch.Size([batch, 16800])
        :param mask_pos: 正样本标签布尔索引 [batch, 16800] 根据gt算出来的
        :param pos_num: 每一个图片的正样本个数  Tensor[batch] 1维降维
        :param args:用于添加其它损失项
        :return:
        '''
        ret = []
        ret.append(self.bboxs_loss(bboxs_p, bboxs_g, mask_pos))
        ret.append(self.labels_loss(labels_p, labels_g, mask_pos, pos_num))
        return ret

    def forward(self, p_bboxs, g_bboxs, p_labels, g_labels, *args):
        '''

        :param p_bboxs: 预测的 xywh  torch.Size([batch, 16800, 4])
        :param g_bboxs: 匹配的 xywh  torch.Size([batch, 16800, 4])
        :param p_labels:
        :param g_labels:匹配的 torch.Size([batch, 16800])
        :param args:用于添加其它损失项
        :return:
        '''
        # 正样本标签布尔索引 [batch, 16800]
        mask_pos = g_labels > 0
        # 每一个图片的正样本个数  Tensor[batch] 数据1维降维
        pos_num = mask_pos.sum(dim=1)  # [batch] 这个用于batch中1个图没有正例不算损失和计算反例数

        loss_list = self.get_loss_list(p_bboxs, g_bboxs, p_labels, g_labels, mask_pos, pos_num, *args)

        # 没有正样本的图像不计算分类损失  pos_num是每一张图片的正例个数 eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=torch.finfo(torch.float).eps)  # 求平均 排除有0的情况取类型最小值

        # 计算总损失平均值 /正样本数量 [n] 加上超参系数
        _s = self.loss_coefficient
        loss_total = 0
        log_dict = {}
        for i, key in enumerate(['loss_bboxs', 'loss_labels', 'loss_keypoints']):
            loss_total += _s[i] * loss_list[i]
            _t = (loss_list[i] * num_mask / pos_num).mean(dim=0)
            log_dict[key] = _t.item()
        loss_total = (loss_total * num_mask / pos_num).mean(dim=0)
        return loss_total, log_dict


if __name__ == '__main__':
    import numpy as np

    np.random.seed(20201031)

    # pconf = torch.arange(1.0, -0.01, -0.1)
    # print(pconf)

    # t_focal_loss()

    # py = torch.tensor(0.8)
    # gy = torch.tensor(0.8)
    # print(focal_loss4center(py, gy))
    # floss = FocalLoss4Center()
    # print(floss(py, gy))

    t_多值交叉熵()
    # f_二值交叉熵2()
    # f_二值交叉熵1()
    # t_LossYOLO()
