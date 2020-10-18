import torch
from torch import Tensor, nn
import torch.nn.functional as F

from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_boxes import batched_nms, xywh2ltrb, ltrb2ltwh, diff_bbox, diff_keypoints, ltrb2xywh
from f_tools.pic.f_show import show_od4ts
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG


class LossOD_K(nn.Module):

    def __init__(self, anc, loss_weight=(1., 1., 1.), neg_ratio=3):
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

        if CFG.IS_VISUAL:
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


def t多分类():
    # 2维
    input = torch.randn(2, 5, requires_grad=True)  # (batch,类别值)
    # 多分类标签 (3)
    target = torch.randint(3, 5, (2,), dtype=torch.int64)  # (batch)  值是3~4
    loss = F.cross_entropy(input, target, reduction='none')  # 二维需要拉平
    print(loss)
    # 多维
    input = torch.randn(2, 7, 5, requires_grad=True)  # (batch,类别值,框)
    # # 多分类标签 (3)
    target = torch.randint(3, 5, (2, 5,), dtype=torch.int64)  # (batch,框)
    loss = F.cross_entropy(input, target, reduction='none')  # 二维需要拉平
    print(loss)
    # loss.backward()


def t手写():
    pass


def t二分类():
    pass
    size = (2, 1000)  # 同维比较
    input = torch.randn(*size, requires_grad=True)
    # input = F.softmax(input,dim=-1)
    target = torch.tensor(np.random.randint(0, 2, size), dtype=torch.float)
    # loss = F.binary_cross_entropy(input, target, reduction='none')  # input不为1要报错
    loss = F.binary_cross_entropy(F.sigmoid(input), target, reduction='none')
    # loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

    # loss.backward()

    # loss = F.binary_cross_entropy(F.sigmoid(input), target, reduction='none')
    # loss = F.binary_cross_entropy_with_logits(predict, label, reduction='none')  #

    # print(loss)
    print(loss)
    # loss.backward()


if __name__ == '__main__':
    import numpy as np

    '''-------------多分类损失------------'''
    # t多分类()

    '''---------------二分类损失----------------'''
    t二分类()
