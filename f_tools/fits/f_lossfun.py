import torch
from torch import Tensor, nn
import torch.nn.functional as F

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import f_recover_normalization4ts
from f_tools.fun_od.f_boxes import batched_nms, xywh2ltrb, ltrb2ltwh, diff_bbox, diff_keypoints, ltrb2xywh, calc_iou4ts, \
    bbox_iou4ts
from f_tools.pic.f_show import show_od4ts


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class LossOD_K(nn.Module):

    def __init__(self, anc, loss_weight=(1., 1., 1.), neg_ratio=3, cfg=None):
        '''

        :param anc:
        :param loss_weight:
        '''
        super().__init__()
        self.location_loss = nn.SmoothL1Loss(reduction='none')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        self.anc = anc.unsqueeze(dim=0)
        self.loss_weight = loss_weight
        self.neg_ratio = neg_ratio
        self.cfg = cfg

    def forward(self, p_bboxs_xywh, g_bboxs_ltrb, p_labels, g_labels, p_keypoints, g_keypoints, imgs_ts=None):
        '''

        :param p_bboxs_xywh: torch.Size([batch, 16800, 4])
        :param g_bboxs_ltrb:
        :param p_labels:
        :param g_labels: torch.Size([batch, 16800])
        :param p_keypoints:
        :param g_keypoints:
        :return:
        '''
        # 正样本标签布尔索引 [batch, 16800] 同维运算
        mask_pos = g_labels > 0

        if self.cfg is not None and self.cfg.IS_VISUAL:
            flog.debug('显示匹配的框 %s 个', torch.sum(mask_pos))
            for i, (img_ts, mask) in enumerate(zip(imgs_ts, mask_pos)):
                # 遍历降维 self.anc([1, 16800, 4])
                _t = self.anc.view(-1, 4).clone()
                _t[:, ::2] = _t[:, ::2] * CFG.IMAGE_SIZE[0]
                _t[:, 1::2] = _t[:, 1::2] * CFG.IMAGE_SIZE[1]
                _t = _t[mask, :]
                show_od4ts(img_ts, xywh2ltrb(_t), torch.ones(200))

        # 每一个图片的正样本个数  [batch, 16800] ->[batch]
        pos_num = mask_pos.sum(dim=1)  # [batch] 这个用于batch中1个图没有正例不算损失和计算反例数

        '''-----------bboxs 损失处理-----------'''
        # [1, 16800, 4] ^^ [batch, 16800, 4] = [batch, 16800, 4]
        d_bboxs = diff_bbox(self.anc, ltrb2xywh(g_bboxs_ltrb))

        # [batch, 16800, 4] -> [batch, 16800] 得每一个特图 每一个框的损失和
        loss_bboxs = self.location_loss(p_bboxs_xywh, d_bboxs).sum(dim=-1)  # smooth_l1_loss
        __d = 1
        # 正例损失过滤
        loss_bboxs = (mask_pos.float() * loss_bboxs).sum(dim=1)

        '''-----------keypoints 损失处理-----------'''
        d_keypoints = diff_keypoints(self.anc, g_keypoints)
        loss_keypoints = self.location_loss(p_keypoints, d_keypoints).sum(dim=-1)
        # 全0的不计算损失 与正例的布尔索引取交集
        _mask = (mask_pos) * torch.all(g_keypoints > 0, dim=2)  # and一下 将全0的剔除
        loss_keypoints = (_mask.float() * loss_keypoints).sum(dim=1)

        '''-----------labels 损失处理-----------'''
        # 移到中间才能计算
        p_labels = p_labels.permute(0, 2, 1)  # (batch,16800,2) -> (batch,2,16800)
        # (batch,2,16800)^[batch, 16800] ->[batch, 16800] 得同维每一个元素的损失
        loss_labels = self.confidence_loss(p_labels, g_labels.long())
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
        loss_labels = (loss_labels * (mask_z)).sum(dim=1)

        '''-----------损失合并处理-----------'''
        # 没有正样本的图像不计算分类损失  pos_num是每一张图片的正例个数 eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=torch.finfo(torch.float).eps)  # 求平均 排除有0的情况取类型最小值

        # 计算总损失平均值 /正样本数量 [n] 加上超参系数
        loss_bboxs = (loss_bboxs * num_mask / pos_num).mean(dim=0)
        __d = 1
        # loss_bboxs.detach().cpu().numpy()
        loss_labels = (loss_labels * num_mask / pos_num).mean(dim=0)
        loss_keypoints = (loss_keypoints * num_mask / pos_num).mean(dim=0)

        loss_total = self.loss_weight[0] * loss_bboxs \
                     + self.loss_weight[1] * loss_labels \
                     + self.loss_weight[2] * loss_keypoints

        log_dict = {}
        log_dict['loss_bboxs'] = loss_bboxs.item()
        log_dict['loss_labels'] = loss_labels.item()
        log_dict['loss_keypoints'] = loss_keypoints.item()
        return loss_total, log_dict


class LossYOLO(nn.Module):

    def __init__(self, grid=7, num_bbox=2, l_coord=5, l_noobj=0.5):
        '''

        :param grid: 7代表将图像分为7x7的网格
        :param num_bbox: 2代表一个网格预测两个框
        :param l_coord:
        :param l_noobj:
        '''
        super(LossYOLO, self).__init__()
        self.S = grid
        self.B = num_bbox
        self.l_coord = l_coord  # 5代表 λcoord  更重视8维的坐标预测
        self.l_noobj = l_noobj  # 0.5代表没有object的bbox的confidence loss

    def calc_loss_conf_noo(self, p_yolo_ts, g_yolo_ts, mask_noo):
        # 选出没有目标的所有框拉平 (batch,7,7,30) -> (xx,7,7,30) -> (xxx,30)
        p_yolo_ts_noo = p_yolo_ts[mask_noo].view(-1, 30)
        g_yolo_ts_noo = g_yolo_ts[mask_noo].view(-1, 30)

        # 创建布尔索引选出---选出没有目标的置信度
        mask_conf_noo = torch.ByteTensor(p_yolo_ts_noo.size())  # 例如：[1496,30]
        mask_conf_noo.zero_()  # 初始化全为0
        mask_conf_noo[:, 4] = 1  # 选出置信度将第4、9  即有物体的confidence置为1
        mask_conf_noo[:, 9] = 1
        # (xxx,30) -> (xxx,2)
        p_conf_noo = p_yolo_ts_noo[mask_conf_noo]  # 选出没有目标的置信度
        # 这里恒为0
        g_conf_zero = g_yolo_ts_noo[mask_conf_noo]
        # nooobj_loss 一个标量, size_average=False 是对batch里的值求和
        loss_conf_noo = F.mse_loss(p_conf_noo, g_conf_zero, size_average=False)
        return loss_conf_noo

    def forward(self, p_yolo_ts, g_yolo_ts):
        '''
        4个损失:

        :param p_yolo_ts: (tensor) size(batch,7,7,2x5+20=30) [x,y,w,h,c]
        :param g_yolo_ts: (tensor) size(batch,7,7,30)
        :return:
        '''
        batch = p_yolo_ts.size()[0]  # shape[0]
        '''生成有目标和没有目标的同维布尔索引'''
        # 获取有GT的框的布尔索引集,conf为1 4或9可作任选一个,结果一样的
        mask_coo = g_yolo_ts[:, :, :, 4] > 0  # (batch,7,7,30) -> (batch,7,7)
        # 没有GT的索引 mask ==0  coo_mask==False
        mask_noo = g_yolo_ts[:, :, :, 4] == 0  # (batch,7,7,30) ->(batch,7,7)
        # (batch,7,7) -> (batch,7,7,30) unsqueeze(-1) 扩展最后一维  coo_mask 大部分为0 noo_mask  大部分为1
        mask_coo = mask_coo.unsqueeze(-1).expand_as(g_yolo_ts)
        mask_noo = mask_noo.unsqueeze(-1).expand_as(g_yolo_ts)

        '''组装 有GT的对应 box[x1,y1,w1,h1,c1] 及 后20个数'''
        # 在p_yolo_ts预测中选出有GT的 再拉成二维  (batch,7,7,30) -> (xx,30)
        p_yolo_ts_coo = p_yolo_ts[mask_coo].view(-1, 30)
        # 取预测框及置信度 2个展开 (xx,30)-> (xx,10) ->(xxx,5)   box[x1,y1,w1,h1,c1]
        box_pred = p_yolo_ts_coo[:, :10].contiguous().view(-1, 5)
        # 取预测类别  每个网格预测的类别  后20
        class_pred = p_yolo_ts_coo[:, 10:]

        # 对真实标签做同样操作
        g_yolo_ts_coo = g_yolo_ts[mask_coo].view(-1, 30)
        box_g = g_yolo_ts_coo[:, :10].contiguous().view(-1, 5)
        class_g = g_yolo_ts_coo[:, 10:]

        '''计算没有目标的置信度损失'''
        loss_conf_noo = self.calc_loss_conf_noo(p_yolo_ts, g_yolo_ts, mask_noo)

        # 计算包含obj损失  即本来有，预测有  和  本来有，预测无
        mask_max = torch.ByteTensor(box_g.size())
        mask_max.zero_()
        mask_max_noo = torch.ByteTensor(box_g.size())
        mask_max_noo.zero_()
        # 2个预测对一个,选择最好的IOU 0~gt个数 间隔为2
        for i in range(0, box_g.size()[0], 2):
            # 计算iou需要ltrb
            box_pred_2 = box_pred[i:i + 2]  # 取前两个
            box_pred_ltrb = torch.FloatTensor(box_pred_2.size())  # torch.tensor(x,y...).type(torch.float32)
            box_pred_ltrb[:, :2] = box_pred_2[:, :2] - 0.5 * box_pred_2[:, 2:4]
            box_pred_ltrb[:, 2:4] = box_pred_2[:, :2] + 0.5 * box_pred_2[:, 2:4]

            box_g_1 = box_g[i].view(-1, 5)  # GT框两个是一样的不需要用两个来比
            box_g_ltrb = torch.FloatTensor(box_g_1.size())
            box_g_ltrb[:, :2] = box_g_1[:, :2] - 0.5 * box_g_1[:, 2:4]
            box_g_ltrb[:, 2:4] = box_g_1[:, :2] + 0.5 * box_g_1[:, 2:4]
            iou = calc_iou4ts(box_pred_ltrb[:, :4], box_g_ltrb[:, :4])  # 2对1 返回(2,1)

            max_iou, max_index = iou.max(0)
            # max_index = max_index.data.cuda()
            mask_max[i + max_index] = 1
            mask_max_noo[i + 1 - max_index] = 1
        # 1.response loss响应损失，即本来有，预测有   有相应 坐标预测的loss  （x,y,w开方，h开方）参考论文loss公式
        # box_pred [144,5]   coo_response_mask[144,5]   box_pred_response:[72,5]
        # 选择IOU最好的box来进行调整  负责检测出某物体
        box_pred_max = box_pred[mask_max].view(-1, 5)
        box_g_random = box_g[mask_max].view(-1, 5)  # 维度1随便选一个即可
        '''计算有目标的iou好的那个的置信度损失'''
        loss_conf_coo_max = F.mse_loss(box_pred_max[:, 4], box_g_random[:, 4], size_average=False)
        '''计算有目标的iou好的那个的框损失   （x,y,w开方，h开方）'''
        loss_loc = F.mse_loss(box_pred_max[:, :2], box_g_random[:, :2], size_average=False) \
                   + F.mse_loss(torch.sqrt(box_pred_max[:, 2:4]), box_g_random[:, 2:4].sqrt(), size_average=False)

        # # 2.not response loss 未响应损失，即本来有，预测无   未响应
        # box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        # box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        # box_target_not_response[:, 4] = 0
        # box_pred_response: [72, 5]

        # not_contain_loss = F.mse_loss(box_pred_max[:, 4], box_g_random[:, 4], size_average=False)
        '''计算有物体的类别loss'''
        class_loss = F.mse_loss(class_pred, class_g, size_average=False)
        # 除以N  即平均一张图的总损失
        return (self.l_coord * loss_loc
                + loss_conf_noo
                + loss_conf_coo_max
                + self.l_noobj * loss_conf_noo
                + class_loss) / batch


class LossYOLOv1(nn.Module):

    def __init__(self, grid=7, num_bbox=1, l_coord=5, l_noobj=0.5, num_cls=20):
        '''

        :param grid: 7代表将图像分为7x7的网格
        :param num_bbox: 2代表一个网格预测两个框
        :param l_coord:
        :param l_noobj:
        '''
        super(LossYOLOv1, self).__init__()
        self.S = grid
        self.B = num_bbox
        self.num_cls = num_cls
        self.l_coord = l_coord  # 5代表 λcoord  box位置的坐标预测 权重
        self.l_noobj = l_noobj  # 0.5代表没有目标的bbox的confidence loss 权重

    def forward(self, p_yolo_ts, g_yolo_ts):
        '''
        4个损失:

        :param p_yolo_ts: (tensor) size(batch,7,7,4+1+num_class=25) [x,y,w,h,c]
        :param g_yolo_ts: (tensor) size(batch,7,7,4+1+num_class=25)
        :return:
        '''
        num_dim = self.B * 5 + self.num_cls
        '''生成有目标和没有目标的同维布尔索引'''
        # 获取有GT的框的布尔索引集,conf为1 4或9可作任选一个,结果一样的
        mask_coo = g_yolo_ts[:, :, :, 4] > 0  # (batch,7,7,25) -> (batch,7,7)
        # 没有GT的索引 mask ==0  coo_mask==False
        mask_noo = g_yolo_ts[:, :, :, 4] == 0  # (batch,7,7,25) ->(batch,7,7)
        # (batch,7,7) -> (batch,7,7,30) unsqueeze(-1) 扩展最后一维  coo_mask 大部分为0 noo_mask  大部分为1
        mask_coo = mask_coo.unsqueeze(-1).expand_as(g_yolo_ts)
        mask_noo = mask_noo.unsqueeze(-1).expand_as(g_yolo_ts)

        # (batch,7,7,25) -> (xxx) ->(xx,7,7,25)
        p_coo = p_yolo_ts[mask_coo].view(-1, 7, 7, num_dim)
        g_coo = g_yolo_ts[mask_coo].view(-1, 7, 7, num_dim)
        # 有目标的样本 (xxx,7,7,25) -> (xxx,xx,25)
        p_coo_batch = p_coo.view(p_coo.shape[0], -1, num_dim).contiguous()
        g_coo_batch = g_coo.view(p_coo.shape[0], -1, num_dim).contiguous()

        '''计算有目标的-------类别loss'''
        # p_cls_coo = p_coo_batch[:, :, self.B * 5:].reshape(-1,self.num_cls)
        p_cls_coo = p_coo_batch[:, :, self.B * 5:]
        # g_cls_coo = g_coo_batch[:, :, self.B * 5:].reshape(-1,self.num_cls)
        g_cls_coo = g_coo_batch[:, :, self.B * 5:]
        loss_cls = F.mse_loss(p_cls_coo, g_cls_coo, reduction='sum')

        '''计算有目标的iou好的那个的-------框损失   （x,y,w开方，h开方）'''
        # (batch, 7, 7, 25) -> (xx, 7, 7, 25) -> (xxx, xx,25) ->  (xxx, xx,4)
        p_boxes_coo = p_coo_batch[:, :, :4]
        g_boxes_coo = g_coo_batch[:, :, :4]
        loss_xy = F.mse_loss(p_boxes_coo[:, :2], g_boxes_coo[:, :2], reduction='sum')
        # 大小敏感
        loss_wh = F.mse_loss(p_boxes_coo[:, 2:4].sqrt(), g_boxes_coo[:, 2:4].sqrt(), reduction='sum')

        '''计算有目标的置信度损失???'''
        p_conf_coo = p_coo_batch[:, :, 4]  # [batch, 49, 25] ->[batch, 49]
        g_conf_one = torch.ones_like(p_conf_coo)
        loss_conf_coo = F.mse_loss(p_conf_coo, g_conf_one, reduction='sum')

        '''计算没有目标的置信度损失'''
        # 选出没有目标的所有框拉平 (batch,7,7,30) -> (xx,7,7,30) -> (xxx,30)
        p_conf_noo = p_yolo_ts[mask_noo].view(-1, num_dim)[:, 4]
        # zeros = torch.zeros(p_conf_noo.shape)
        g_conf_zero = g_yolo_ts[mask_noo].view(-1, num_dim)[:, 4]  # 全0
        loss_conf_noo = F.mse_loss(p_conf_noo, g_conf_zero, size_average=False)

        batch = p_yolo_ts.shape[0]  # batch数 shape[0]
        return (self.l_coord * (loss_xy + loss_wh)
                + loss_conf_coo
                + self.l_noobj * loss_conf_noo
                + loss_cls) / batch


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


class KeypointsLoss(ObjectDetectionLoss):

    def __init__(self, ancs, neg_ratio, variance, loss_coefficient):
        super().__init__(ancs, neg_ratio, variance, loss_coefficient)

    def _keypoints_anc_diff(self, gkeypoints):
        '''
        计算pkeypoints相对anchors的回归参数
            anc [1, num_anchors, 4]
            只计算中心点损失, 但有宽高
        :param gkeypoints: 已完成 正例匹配 torch.Size([batch, 16800, 10])
        :return:
            返回 ground truth相对anchors的回归参数 这里是稀疏
        '''
        # type: (Tensor)
        #  [1, anc个, 2] ->  [anc个, 2] -> [anc个, 2*5] -> [1, anc个, 10]
        ancs_ = self.ancs.squeeze(0)[:, :2]
        ancs_xy = ancs_.repeat(1, 5)[None]
        ancs_wh = ancs_.repeat(1, 5)[None]

        # 这里 scale_xy 是乘10 0维自动广播[batch, 16800, 10]/torch.Size([1, 16800, 10])
        gxy = self.scale_xy * (gkeypoints - ancs_xy) / ancs_wh
        return gxy

    def keypoints_loss(self, pkeypoints, gkeypoints, mask_pos):
        # -------------关键点损失------------------(只有正样本) 全0要剔除
        # 计算差异[batch, 16800, 10]
        diff = self._keypoints_anc_diff(gkeypoints)
        # 计算损失 smooth_l1_loss
        loss_keypoints = self.location_loss(pkeypoints, diff).sum(dim=-1)
        # 正例计算损失 (全0及反例过滤)
        __mask_pos = (mask_pos) * torch.all(gkeypoints > 0, 2)  # and一下 将全0的剔除
        loss_keypoints = (__mask_pos.float() * loss_keypoints).sum(dim=1)
        return loss_keypoints

    def get_loss_list(self, bboxs_p, bboxs_g, labels_p, labels_g, mask_pos, pos_num, *args):
        '''

        :param bboxs_p: 预测的 xywh  torch.Size([batch, 16800, 4])
        :param bboxs_g: 匹配的 xywh  torch.Size([batch, 16800, 4])
        :param labels_p:
        :param labels_g:匹配的 torch.Size([batch, 16800])
        :param mask_pos: 正样本标签布尔索引 [batch, 16800]
        :param pos_num: 每一个图片的正样本个数  Tensor[batch] 1维降维
        :param args:
           :param pkeypoints: torch.Size([5, 16800, 10])
           :param gkeypoints: 匹配的
        :return:
        '''
        loss_list = super().get_loss_list(bboxs_p, bboxs_g, labels_p, labels_g, mask_pos, pos_num, *args)
        loss_list.append(self.keypoints_loss(*args, mask_pos))

        return loss_list


class PredictOutput(nn.Module):
    def __init__(self, ancs, img_size, variance=(0.1, 0.2), scores_threshold=0.5, iou_threshold=0.5, max_output=100):
        '''

        :param ancs: 生成的基础ancs  xywh  只有一张图  (m*w*h,4)
        :param img_size: 输入尺寸 (300, 300) (w, h)
        :param variance: 用于控制预测值
        :param iou_threshold: iou阀值
        :param scores_threshold: scores 分数小于0.5的不要
        :param max_output: 最大预测数
        '''
        super(PredictOutput, self).__init__()
        self.ancs = nn.Parameter(ancs.unsqueeze(dim=0), requires_grad=False)
        # self.scale_xy = torch.tensor(float(variance[0])).type(torch.float)  # 这里是缩小
        # self.scale_wh = torch.tensor(float(variance[0])).type(torch.float)
        self.scale_xy = variance[0]  # 这里是缩小
        self.scale_wh = variance[1]

        self.img_size = img_size  # (w,h)

        self.iou_threshold = iou_threshold  # 非极大超参
        self.scores_threshold = scores_threshold  # 分数过小剔除
        self.max_output = max_output  # 最多100个目标

    def scale_back_batch(self, bboxs_p, labels_p, *args):
        '''
            修正def 并得出分数 softmax
            1）通过预测的 loc_p 回归参数与anc得到最终预测坐标 box
            2）将box格式从 xywh 转换回ltrb
            3）将预测目标 score通过softmax处理
        :param bboxs_p: 预测出的框 偏移量 xywh [N, 4, 8732]
        :param labels_p: 预测所属类别 [N, label_num, 8732]
        :return:  返回 anc+预测偏移 = 修复后anc 的 ltrb 形式
        '''
        # type: (Tensor, Tensor)
        # Returns a view of the original tensor with its dimensions permuted.
        # [batch, 4, 8732] -> [batch, 8732, 4]
        # loc_p = loc_p.permute(0, 2, 1)
        # [batch, label_num, 8732] -> [batch, 8732, label_num]
        # label_p = label_p.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        # ------------限制预测值------------
        bboxs_p[:, :, :2] = self.scale_xy * bboxs_p[:, :, :2]  # 预测的x, y回归参数
        bboxs_p[:, :, 2:] = self.scale_wh * bboxs_p[:, :, 2:]  # 预测的w, h回归参数
        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxs_p[:, :, :2] = bboxs_p[:, :, :2] * self.ancs[:, :, 2:] + self.ancs[:, :, :2]
        bboxs_p[:, :, 2:] = bboxs_p[:, :, 2:].exp() * self.ancs[:, :, 2:]

        # xywh -> ltrb 用于极大nms
        xywh2ltrb(bboxs_p)

        # scores_in: [batch, 8732, label_num]  输出8732个分数 -1表示最后一个维度
        if labels_p.dim() > 2:
            scores_p = F.softmax(labels_p, dim=-1)
        else:
            # scores_p = F.softmax(labels_p, dim=-1)
            scores_p = F.sigmoid(labels_p)
        return bboxs_p, scores_p

    def decode_single_new(self, bboxs_p, scores_p, criteria, num_output, keypoints_p=None, k_type='ltrb'):
        '''
        一张图片的  修复后最终框 通过极大抑制 的 ltrb 形式
        :param bboxs_p: (Tensor 8732 x 4)
        :param scores_p:
            scores_p: 单分类为一维数组
            scores_p (Tensor 8732 x nitems) 多类别分数
        :param criteria: IoU threshold of bboexes IoU 超参
        :param num_output: 最大预测数 超参
        :return: [bboxes_out, scores_out, labels_out, other_in] iou大于criteria 的num_output 个 最终nms出的框
        '''
        # type: (Tensor, Tensor, float, int)
        device = bboxs_p.device

        bboxs_p = bboxs_p.clamp(min=0, max=1)  # 对越界的bbox进行裁剪

        '''---组装数据 按21个类拉平数据 构建labels   只有一个类特殊处理--- 每一类需进行一次nms'''
        if scores_p.dim() > 1:
            num_classes = scores_p.shape[-1]  # 取类别数 21类
            # [8732, 4] -> [8732, 21, 4] 注意内存 , np高级 复制框预测框 为21个类
            bboxs_p = bboxs_p.repeat(1, num_classes).reshape(scores_p.shape[0], -1, 4)

            # 创建 21个类别 scores_p与 bboxs_p 对应 , 用于预测结果展视  用于表明nms 现在在处理第几个类型(每一类需进行一次nms)
            idxs = torch.arange(num_classes, device=device)
            # [num_classes] -> [8732, num_classes]
            idxs = idxs.view(1, -1).expand_as(scores_p)

            # 移除归为背景类别的概率信息 不要背景
            bboxs_p = bboxs_p[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
            scores_p = scores_p[:, 1:]  # [8732, 21] -> [8732, 20]
            idxs = idxs[:, 1:]  # [8732, 21] -> [8732, 20]

            # 将20个类拉平
            bboxs_p = bboxs_p.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
            scores_p = scores_p.reshape(-1)  # [8732, 20] -> [8732x20]
            idxs = idxs.reshape(-1)  # [8732, 20] -> [8732x20]

            if keypoints_p is not None:
                # 复制 - 剔背景 - 拉平类别
                keypoints_p = keypoints_p.repeat(1, num_classes).reshape(keypoints_p.shape[0], -1,
                                                                         keypoints_p.shape[-1])
                keypoints_p = keypoints_p[:, 1:, :]  # [8732, 21, 10] -> [8732, 20, 10]
                keypoints_p = keypoints_p.reshape(-1, 4)  # [8732, 20, 10] ->[8732 * 20, 10]
        else:
            # 组装 labels 按scores_p 进行匹配 只有一个类不用拉平  这里是一维
            idxs = torch.tensor([1], device=device)
            idxs = idxs.expand_as(scores_p)  # 一维直接全1匹配  保持与多类结构一致

        # 过滤...移除低概率目标，self.scores_thresh=0.05    16800->16736
        inds = torch.nonzero(scores_p > self.scores_threshold, as_tuple=False).squeeze(1)
        bboxs_p, scores_p, idxs = bboxs_p[inds, :], scores_p[inds], idxs[inds]

        # remove empty boxes 面积小的   ---长宽归一化值 小于一长宽分之一的不要
        ws, hs = bboxs_p[:, 2] - bboxs_p[:, 0], bboxs_p[:, 3] - bboxs_p[:, 1]
        keep = (ws >= 1. / self.img_size[0]) & (hs >= 1. / self.img_size[1])  # 目标大于1个像素的不要
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxs_p, scores_p, idxs = bboxs_p[keep], scores_p[keep], idxs[keep]

        # non-maximum suppression 将所有类别拉伸后 16736->8635
        keep = batched_nms(bboxs_p, scores_p, idxs, iou_threshold=criteria)  # criteria按iou进行选取

        # keep only topk scoring predictions
        keep = keep[:num_output]  # 最大100个目标
        bboxes_out = bboxs_p[keep, :]
        scores_out = scores_p[keep]
        labels_out = idxs[keep]

        if k_type == 'ltwh':  # 默认ltrb
            ltrb2ltwh(bboxes_out)

        ret = [bboxes_out, scores_out, labels_out]
        if keypoints_p is not None:
            ret.append(keypoints_p[keep, :].clamp(min=0, max=1))
        return ret

    def forward(self, bboxs_p, labels_p, keypoints_p=None, ktype='ltrb'):
        '''
        将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        :param bboxs_p: 预测出的框 偏移量 torch.Size([1, 4, 8732])
        :param labels_p: 预测所属类别的分数 torch.Size([1, 21, 8732])   只有一个类
        :param keypoints_p:
        :param ktype:ltrb  ltwh用于coco
        :return: imgs_rets 每一个张的最终输出 ltrb
            [bboxes_out, scores_out, labels_out, other_in]
        '''
        # 通过预测的boxes回归参数得到最终预测坐标, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxs_p, labels_p)

        # imgs_rets = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        imgs_rets = []

        # bboxes: [batch, 8732, 4] 0维分割 得每一个图片的框 zip[list(bboxe),list(prob)]
        _zip = [bboxes.split(1, 0), probs.split(split_size=1, dim=0)]

        if keypoints_p is not None:  # keypoints_p 特殊处理
            _zip = [bboxes.split(1, 0), probs.split(split_size=1, dim=0), keypoints_p.split(1, 0)]

        # 遍历一个batch中的每张image数据
        for index in range(bboxes.shape[0]):
            # bbox_p, prob, keypoints_p
            bbox_p = _zip[0][index]
            bbox_p = bbox_p.squeeze(0)  # 降维 每一张的

            prob = _zip[1][index]
            prob = prob.squeeze(0)  # 每一张的

            if len(_zip) == 2:  # 为keypoints_p 特殊 处理
                # _zip[0][index]
                imgs_rets.append(
                    self.decode_single_new(
                        bbox_p,
                        prob,
                        self.iou_threshold,
                        self.max_output,
                        None,
                        ktype
                    ))
            else:
                keypoints_p = _zip[2][index]
                keypoints_p = keypoints_p.squeeze(0)

                imgs_rets.append(
                    self.decode_single_new(
                        bbox_p,
                        prob,
                        self.iou_threshold,
                        self.max_output,
                        keypoints_p,
                        ktype
                    ))
        return imgs_rets


def f_bce(i, o):
    '''
    同维
    :param i: 值必须为0~1之间 float
    :param o: 值为 float
    :return:
    '''
    return np.round(-(o * np.log(i) + (1 - o) * np.log(1 - i)), 4)


def t_多值交叉熵():
    # 2维
    input = torch.randn(2, 5, requires_grad=True)  # (batch,类别值)
    # 多分类标签 [3,5) size=(2)
    target = torch.randint(3, 5, (2,), dtype=torch.int64)  # (batch)  值是3~4
    loss = F.cross_entropy(input, target, reduction='none')  # 二维需要拉平
    print(loss)
    # 多维
    input = torch.randn(2, 7, 5, requires_grad=True)  # (batch,类别值,框)
    # # 多分类标签 [3,5) size=(2,5)
    target = torch.randint(3, 5, (2, 5,), dtype=torch.int64)  # (batch,框)
    loss = F.cross_entropy(input, target, reduction='none')  # 二维需要拉平
    print(loss)
    # loss.backward()


def f_二值交叉熵2():
    # input 与 target 一一对应 同维
    size = (2, 3)  # 两个数据
    input = torch.randn(*size, requires_grad=True)
    # [0,2)
    # target = torch.tensor(np.random.randint(0, 5, size), dtype=torch.float)
    # target = torch.tensor(np.random.randint(0, 2, size), dtype=torch.float)
    target = torch.tensor([
        [1, 0, 0],
        [1, 0, 1],
    ], dtype=torch.float)

    loss1 = F.binary_cross_entropy(torch.sigmoid(input), target, reduction='none')  # 独立的
    print(loss1)

    loss2 = F.binary_cross_entropy_with_logits(input, target, reduction='none')
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
    print(f_bce(p_cls_np, g_cls_np))

    p_cls_ts = torch.tensor(p_cls)
    g_cls_ts = torch.tensor(g_cls)
    print(bce_loss(p_cls_ts, g_cls_ts))


def t_LossYOLO():
    # yolo = LossYOLO()
    yolo = LossYOLOv1()
    num_cls = 20
    num_boxes = 1
    num_dim = num_boxes * 5 + num_cls

    p_yolo_ts = torch.rand(7, 7, 7, num_dim, requires_grad=True)
    g_yolo_ts = torch.rand(7, 7, 7, num_dim, requires_grad=True)
    print(yolo(p_yolo_ts, g_yolo_ts))


class Loss(nn.Module):
    def __init__(self, lambda_coord, lambda_noobj):
        super(Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, p_yolo, g_yolo):
        """
            输入两个变量分别为，通过网络预测的张量和实际标签张量。两个张量的尺寸均为[batch_size，s，s，95]
        batch_size为批量处理的图像个数，s为网格尺寸，95就是5个box参数加90类，前5个参数为box属性。
        """
        """
            计算网格是否包含有目标，应从实际标签张量的box属性第5各参数来判定，该值表征某网格某box的预测概率为1
        逻辑mask应与原tensor尺寸相同，只包含0-1两个值，表示原tensor对应位置是否满足条件。
        """
        # 具有目标的标签逻辑索引
        coo_mask = g_yolo[:, :, :, 4] > 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(g_yolo)
        # 没有目标的标签逻辑索引
        noo_mask = g_yolo[:, :, :, 4] == 0
        noo_mask = noo_mask.unsqueeze(-1).expand_as(g_yolo)
        """
            计算每张图像中，每个目标对应的，最大IOU的预测box的定位误差、confidence误差、类别误差
            及每个不含目标的box的confidence误差。
        """
        xy_loss = 0
        wh_loss = 0
        con_obj_loss = 0
        nocon_obj_loss = 0
        # 遍历每一个batch
        for i in range(p_yolo.size()[0]):
            # (1,7,7,30) 中有目标的 (1,7,7,30) -> (xx,30)
            coo_targ = g_yolo[i][coo_mask[i]].view(-1, 95)
            # (xx,30) ->(xx,5)
            box_targ = coo_targ[:, :5].contiguous().view(-1, 5)

            # 提取预测box属性
            box_pred = p_yolo[i, :, :, :5].view(-1, 5)
            # 计算IOU张量，尺寸为N×M。
            if box_targ.size()[0] != 0:  # 如果有目标
                iou = self.cal_iou(box_targ, box_pred, coo_mask[i, :, :, 1])
                # 找到每列的最大值及对应行，即对应的真实box的最大IOU及box序号

                max_iou, max_sort = torch.max(iou, dim=0)
                # 计算定位误差
                xy_loss += F.mse_loss(box_pred[max_sort, :2], box_targ[max_sort, :2], reduction='sum')
                wh_loss += F.mse_loss(box_pred[max_sort, 2:4].sqrt(), box_targ[max_sort, 2:4].sqrt(), reduction='sum')

                # 计算confidence误差
                """
                    confidence误差，应为每一个网格内的每一个box的置信概率乘以该box的IOU值，该误差包括两个部分，一个是对于
                包含目标的box，上面已经计算出IOU值，可以直接进行计算，但对于另一部分，也就是不包含目标的box，由于其不包含
                box属性，所以真实confidence应该取0。对于预测的IOU可直接设为1。在计算损失函数时，为计算方便实际可分别设置
                为ones张量和zeros张量。
                """
                # 包含目标的box confidence误差
                con_obj_c = box_pred[max_sort][:, 4] * max_iou
                con_obj_loss += F.mse_loss(con_obj_c, torch.ones_like(con_obj_c), reduction='sum')

                # 不含目标的box confidence误差
                no_sort = torch.ones(box_pred.size()[0]).byte()
                no_sort[max_sort] = 0
                nocon_obj_c = box_pred[no_sort][:, 4]
                nocon_obj_loss += F.mse_loss(nocon_obj_c, torch.zeros_like(nocon_obj_c), reduction='sum')

        # 计算类别误差
        """
            由于类别是通过网格来确定的，每一个网格无论有几个box，一个所属类概率。
            在计算类别误差时，只对目标中心落在该其中的网格进行计算。
        """
        # coo_mask 表示在整个张量中，包含目标的网格点索引，所以可以不对每一个bitch进行分别计算，直接整体求和
        con_pre_class = p_yolo[coo_mask].view(-1, 95)[:, 5:]
        con_tar_class = g_yolo[coo_mask].view(-1, 95)[:, 5:]
        con_class_loss = F.mse_loss(con_pre_class, con_tar_class, reduction='sum')

        # 总损失函数求和
        loss_total = (self.lambda_coord * (xy_loss + xy_loss) + con_obj_loss
                      + self.lambda_noobj * nocon_obj_loss + con_class_loss) / p_yolo.size()[0]

        return loss_total

    def cal_iou(self, box_targ, box_pred, mask):
        # 计算box数量
        M = box_targ.size()[0]
        N = box_pred.size()[0]
        # 转化box参数，转化为统一坐标
        row = torch.arange(14, dtype=torch.float).unsqueeze(-1).expand_as(mask)[mask].cuda()
        col = torch.arange(14, dtype=torch.float).unsqueeze(0).expand_as(mask)[mask].cuda()
        box_targ[:, 0] = col / 14 + box_targ[:, 0] * 1 / 14
        box_targ[:, 1] = row / 14 + box_targ[:, 1] * 1 / 14

        exboxM = box_targ.unsqueeze(0).expand(N, M, 5)
        exboxN = box_pred.unsqueeze(1).expand(N, M, 5)
        dxy = (exboxM[:, :, :2] - exboxN[:, :, :2])
        swh = (exboxM[:, :, 2:4] + exboxN[:, :, 2:4])
        s_inter = swh / 2 - dxy.abs()
        s_inter = (s_inter[:, :, 0] * s_inter[:, :, 1]).clamp(min=0)
        s_union = exboxM[:, :, 2] * exboxM[:, :, 3] + exboxN[:, :, 2] * exboxN[:, :, 3] - s_inter
        iou = s_inter / s_union
        return iou


if __name__ == '__main__':
    import numpy as np

    np.random.seed(20201031)

    # t_多值交叉熵()
    # f_二值交叉熵2()
    # f_二值交叉熵1()

    t_LossYOLO()
