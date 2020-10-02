from typing import List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from f_tools.fun_od.f_boxes import batched_nms, xywh2ltrb, ltrb2ltwh


class MoreLabelsNumLossFun(nn.Module):
    def __init__(self, a=0.5, b=.5):
        # 这里定义超参 a,b 为系数值
        super().__init__()
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)

    def calculate(self, val):
        _t = torch.pow(val, 2)
        _t = torch.sum(_t, dim=1)
        _t = torch.sqrt(_t)
        ret = torch.mean(_t)
        return ret

    def forward(self, outputs: Tensor, y):
        # 输出一行20列
        # torch.Size([8, 20]) torch.Size([8, 20])
        # flog.debug('forward %s %s', outputs, y)

        y1 = torch.ones_like(outputs)
        y1[y <= 0] = 0  # 取没有目标的值
        y2 = torch.ones_like(outputs)
        y2[y > 0] = 0  # 取没有目标的值

        l1 = (y - outputs) * y1
        l2 = (y - outputs) * y2

        loss1 = self.calculate(l1)  # 已有目标的损失
        loss2 = self.calculate(l2)  # 其它类大于0则损失

        return self.a * loss1 + self.b * loss2


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
        self.loss_coefficient = loss_coefficient
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

        # self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        self.confidence_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _bbox_anc_diff(self, bboxs_p):
        '''
        计算ground truth相对anchors的回归参数
            self.dboxes 是def 是xywh self.dboxes
            两个参数只有前面几个用GT替代了的不一样 其它一个值 这里是稀疏
        :param bboxs_p: 已完成 正例匹配 torch.Size([3, 8732, 4])
        :return:
            返回 ground truth相对anchors的回归参数 这里是稀疏
        '''
        # type: (Tensor)
        # 用anc 和 gt比较时需要放大差异
        # 这里 scale_xy 是乘10   scale_xy * (bbox_xy - anc_xy )/ anc_wh
        gxy = self.scale_xy * (bboxs_p[:, :, :2] - self.ancs[:, :, :2]) / self.ancs[:, :, 2:]
        # 这里 scale_xy 是乘5   scale_wh * (bbox_wh - anc_wh ).log()
        gwh = self.scale_wh * (bboxs_p[:, :, 2:] / self.ancs[:, :, 2:]).log()
        return torch.cat((gxy, gwh), dim=-1).contiguous()

    def bboxs_loss(self, bboxs_p, bboxs_g, mask_pos):
        # -------------计算定位损失------------------(只有正样本) 很多0
        # 计算差异
        diff = self._bbox_anc_diff(bboxs_g)
        # 计算损失 [batch, 16800, 4] -> [batch, 16800] 得每一个特图 每一个框的损失和
        loss_bboxs = self.location_loss(bboxs_p, diff).sum(dim=-1)
        # 正例损失过滤
        loss_bboxs = (mask_pos.float() * loss_bboxs).sum(dim=1)
        return loss_bboxs

    def labels_loss(self, labels_p, labels_g, mask_pos, pos_num):
        # -------------计算分类损失------------------正样本很少
        # if labels_p.shape[-1]>1:
        #     labels_p = F.softmax(labels_p, dim=-1)
        # else:
        #     labels_p = F.sigmoid(labels_p)
        loss_labels = self.confidence_loss(labels_p, labels_g)  # [batch, 16800]

        # 分类损失 - 负样本选取  选损失最大的
        labels_neg = loss_labels.clone()
        labels_neg[mask_pos] = torch.tensor(0.0).to(loss_labels)  # 正样本先置0 使其独立

        # 按照 损失 降序排列 con_idx(Tensor: [N, 8732])
        _, labels_idx = labels_neg.sort(dim=1, descending=True)  # descending 倒序
        _, labels_rank = labels_idx.sort(dim=1)  # 两次索引排序 用于取最大的n个布尔索引
        # 计算每一层的反例数   [batch] -> [batch,1]  限制正样本3倍不能超过负样本的总个数
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
        :param mask_pos: 正样本标签布尔索引 [batch, 16800]
        :param pos_num: 每一个图片的正样本个数  Tensor[batch] 1维降维
        :param args:用于添加其它损失项
        :return:
        '''
        ret = []
        ret.append(self.bboxs_loss(bboxs_p, bboxs_g, mask_pos))
        ret.append(self.labels_loss(labels_p, labels_g, mask_pos, pos_num))
        return ret

    def forward(self, bboxs_p, bboxs_g, labels_p, labels_g, *args):
        '''

        :param bboxs_p: 预测的 xywh  torch.Size([batch, 16800, 4])
        :param bboxs_g: 匹配的 xywh  torch.Size([batch, 16800, 4])
        :param labels_p:
        :param labels_g:匹配的 torch.Size([batch, 16800])
        :param args:用于添加其它损失项
        :return:
        '''
        # 正样本标签布尔索引 [batch, 16800]
        mask_pos = labels_g > 0
        # 每一个图片的正样本个数  Tensor[batch] 1维降维
        pos_num = mask_pos.sum(dim=1)  # 这个用于batch中1个图没有正例不算损失和计算反例数

        loss_list = self.get_loss_list(bboxs_p, bboxs_g, labels_p, labels_g, mask_pos, pos_num, *args)

        # 没有正样本的图像不计算分类损失  pos_num是每一张图片的正例个数 eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=torch.finfo(torch.float).eps)  # 求平均 排除有0的情况取类型最小值

        # 总损失为[n] 加上超参系数
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

    def _keypoints_anc_diff(self, pkeypoints):
        '''
        计算pkeypoints相对anchors的回归参数
            anc [1, num_anchors, 4]
            只计算中心点损失, 但有宽高
        :param pkeypoints: 已完成 正例匹配 torch.Size([batch, 16800, 10])
        :return:
            返回 ground truth相对anchors的回归参数 这里是稀疏
        '''
        # type: (Tensor)
        #  [1, anc个, 2] ->  [anc个, 2] -> [anc个, 2*5] -> [1, anc个, 10]
        ancs_ = self.ancs.squeeze(0)[:, :2]
        ancs_xy = ancs_.repeat(1, 5)[None]
        ancs_wh = ancs_.repeat(1, 5)[None]

        # 这里 scale_xy 是乘10 0维自动广播[batch, 16800, 10]/torch.Size([1, 16800, 10])
        gxy = self.scale_xy * (pkeypoints - ancs_xy) / ancs_wh
        return gxy

    def keypoints_loss(self, pkeypoints, gkeypoints, mask_pos):
        # -------------关键点损失------------------(只有正样本) 全0要剔除
        # 计算差异[batch, 16800, 10]
        diff = self._keypoints_anc_diff(gkeypoints)
        # 计算损失
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
    def __init__(self, ancs, img_size, variance=(0.1, 0.2), scores_threshold=0.05, iou_threshold=0.5, max_output=100):
        '''

        :param ancs: 生成的基础ancs  xywh  只有一张图  (m*w*h,4)
        :param img_size: 输入尺寸 (300, 300) (w, h)
        :param variance: 用于控制预测值
        :param iou_threshold: iou阀值
        :param scores_threshold: scores分数过小剔除
        :param max_output: 最大预测数
        '''
        super(PredictOutput, self).__init__()
        self.ancs = nn.Parameter(ancs.unsqueeze(dim=0), requires_grad=False)
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
            scores_p = F.sigmoid(labels_p)
        return bboxs_p, scores_p

    def decode_single_new(self, bboxs_p, scores_p, criteria, num_output, keypoints_p=None, mode='ltrb'):
        '''
        一张图片的  修复后最终框 通过极大抑制 的 ltrb 形式
        :param bboxs_p: (Tensor 8732 x 4)
        :param scores_p:
            scores_p: 单分类为一维数组
            scores_p (Tensor 8732 x nitems) 多类别分数
        :param criteria: IoU threshold of bboexes IoU 超参
        :param num_output: 最大预测数 超参
        :return: [bboxes_out, scores_out, labels_out, other_in] 最终nms出的框
        '''
        # type: (Tensor, Tensor, float, int)
        device = bboxs_p.device

        bboxs_p = bboxs_p.clamp(min=0, max=1)  # 对越界的bbox进行裁剪

        '''---组装数据 按21个类拉平数据 构建labels---'''
        if scores_p.dim() > 1:
            num_classes = scores_p.shape[-1]  # 取类别数 21类
            # [8732, 4] -> [8732, 21, 4] 注意内存 , np高级 复制框预测框 为21个类
            bboxs_p = bboxs_p.repeat(1, num_classes).reshape(scores_p.shape[0], -1, 4)

            # 创建 21个类别 scores_p与 bboxs_p 对应 , 用于预测结果展视
            labels = torch.arange(num_classes, device=device)
            # [num_classes] -> [8732, num_classes]
            labels = labels.view(1, -1).expand_as(scores_p)

            # 移除归为背景类别的概率信息
            bboxs_p = bboxs_p[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
            scores_p = scores_p[:, 1:]  # [8732, 21] -> [8732, 20]
            labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

            # 将20个类拉平
            bboxs_p = bboxs_p.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
            scores_p = scores_p.reshape(-1)  # [8732, 20] -> [8732x20]
            labels = labels.reshape(-1)  # [8732, 20] -> [8732x20]

            if keypoints_p is not None:
                # 复制 - 剔背景 - 拉平类别
                keypoints_p = keypoints_p.repeat(1, num_classes).reshape(keypoints_p.shape[0], -1,
                                                                         keypoints_p.shape[-1])
                keypoints_p = keypoints_p[:, 1:, :]  # [8732, 21, 10] -> [8732, 20, 10]
                keypoints_p = keypoints_p.reshape(-1, 4)
        else:
            # 组装 labels 按scores_p 进行匹配
            labels = torch.tensor([1], device=device)
            labels = labels.expand_as(scores_p)

        # 过滤...移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores_p > self.scores_threshold, as_tuple=False).squeeze(1)
        bboxs_p, scores_p, labels = bboxs_p[inds, :], scores_p[inds], labels[inds]

        # remove empty boxes 面积小的
        ws, hs = bboxs_p[:, 2] - bboxs_p[:, 0], bboxs_p[:, 3] - bboxs_p[:, 1]
        keep = (ws >= 1. / self.img_size[0]) & (hs >= 1. / self.img_size[1])  # 目标大于1个像素的
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxs_p, scores_p, labels = bboxs_p[keep], scores_p[keep], labels[keep]

        # non-maximum suppression 将所有类别拉伸后
        keep = batched_nms(bboxs_p, scores_p, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]  # 最大100个目标
        bboxes_out = bboxs_p[keep, :]
        scores_out = scores_p[keep]
        labels_out = labels[keep]

        if mode == 'ltwh':
            ltrb2ltwh(bboxes_out)

        ret = [bboxes_out, scores_out, labels_out]
        if keypoints_p is not None:
            ret.append(keypoints_p[keep, :].clamp(min=0, max=1))
        return ret

    def forward(self, bboxs_p, labels_p, keypoints_p=None, mode='ltrb'):
        '''
        将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        :param bboxs_p: 预测出的框 偏移量 torch.Size([1, 4, 8732])
        :param labels_p: 预测所属类别的分数 torch.Size([1, 21, 8732])
        :param keypoints_p:
        :param mode:ltrb  ltwh用于coco
        :return: imgs_rets 每一个张的最终输出 ltrb
            [bboxes_out, scores_out, labels_out, other_in]
        '''
        # 通过预测的boxes回归参数得到最终预测坐标, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxs_p, labels_p)

        # imgs_rets = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        imgs_rets = []

        # bboxes: [batch, 8732, 4] 0维分割 得每一个图片
        _zip = [bboxes.split(1, 0), probs.split(split_size=1, dim=0)]
        if keypoints_p is not None:
            _zip = [bboxes.split(1, 0), probs.split(split_size=1, dim=0), keypoints_p.split(1, 0)]

        # 遍历一个batch中的每张image数据
        for index in range(bboxes.shape[0]):
            # bbox_p, prob, keypoints_p
            bbox_p = _zip[0][index]
            bbox_p = bbox_p.squeeze(0)

            prob = _zip[1][index]
            prob = prob.squeeze(0)

            if len(_zip) == 2:
                # _zip[0][index]
                imgs_rets.append(
                    self.decode_single_new(
                        bbox_p,
                        prob,
                        self.iou_threshold,
                        self.max_output,
                        None,
                        mode
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
                        mode
                    ))
        return imgs_rets


if __name__ == '__main__':
    model = nn.Linear(10, 1)
    # reduction 默认是取均值
    criterion = nn.BCEWithLogitsLoss()  # 输入值无要求 自动加sigmoid
    criterion = nn.BCELoss(reduction='none')  # 输入值必须在0~1之间 否则报错

    x = torch.randn(16, 10)
    y = torch.empty(16).random_(2)  # 随机int 0 1

    out = model(x)  # (16, 1)
    out = out.squeeze(dim=-1)  # (16, 1) ->(16, )

    loss = criterion(out, y)
    print(loss)