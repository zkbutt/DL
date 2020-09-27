from typing import List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from f_tools.fun_od.f_boxes import batched_nms, xywh2ltrb


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

    def __init__(self, ancs, neg_ratio, variance=(0.1, 0.2)):
        '''

        :param ancs:  生成的基础ancs  xywh  只有一张图  (m*w*h,4)
        :param neg_ratio: 反倒倍数  超参
        :param variance:  用于控制预测值
        '''
        super(ObjectDetectionLoss, self).__init__()
        self.neg_ratio = neg_ratio
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.scale_xy = 1.0 / variance[0]  # 10 用于控制预测值,使其变小,保持一致需放大
        self.scale_wh = 1.0 / variance[1]  # 5  用于控制预测值,使其变小,保持一致需放大

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        unsqueeze = ancs.transpose(0, 1).unsqueeze(dim=0)
        self.anc_bboxs = nn.Parameter(unsqueeze, requires_grad=False)

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, gloc):
        '''
        计算ground truth相对anchors的回归参数
            self.dboxes 是def 是xywh self.dboxes
            两个参数只有前面几个用GT替代了的不一样 其它一个值 这里是稀疏
        :param gloc: 已完成 正例匹配 torch.Size([3, 4, 8732])
        :return:
            返回 ground truth相对anchors的回归参数 这里是稀疏
        '''
        # type: (Tensor)
        # 这里 scale_xy 是乘10
        gxy = self.scale_xy * (gloc[:, :2, :] - self.anc_bboxs[:, :2, :]) / self.anc_bboxs[:, 2:, :]  # Nx2x8732
        gwh = self.scale_wh * (gloc[:, 2:, :] / self.anc_bboxs[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        '''

        :param ploc: 预测值 batch,4,8732
        :param plabel: 预测值  batch,label_numx,8732
        :param gloc: 已完成正例匹配---反倒一样 batch,4,8732
        :param glabel: 已完成正例匹配---反例为0 (batch,8732) 值为0~20
        :return:
        '''
        # type: (Tensor, Tensor, Tensor, Tensor)
        # 计算anc与bbox的差距 稀疏阵 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)
        # -------------计算定位损失------------------(只有正样本) 很多0
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732] 降维

        # 获取正样本的mask 同维运算 Tensor: [N, 8732]
        mask = glabel > 0  # mask1 = torch.nonzero(glabel)
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N] 只计算正例

        # batch的正样本个数 Tensor[N] 1维降维
        pos_num = mask.sum(dim=1)  # 这个用于batch中没有正例不算损失和反例数

        # -------------计算分类损失------------------正样本很少
        con = self.confidence_loss(plabel, glabel)  # Tenosr: [N, 8732]

        # 负样本选取  选损失最大的
        con_neg = con.clone()
        con_neg[mask] = torch.tensor(0.0).to(con)  # 正样本先置0 使其独立
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)  # descending 倒序
        _, con_rank = con_idx.sort(dim=1)  # 两次索引排序 用于取最大的n个布尔索引
        # 限制正样本3倍不能超过负样本的总个数
        neg_num = torch.clamp(self.neg_ratio * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num  # Tensor [N, 8732]

        # 正例索引+反例索引 得1 0 索引用于乘积筛选
        mask_z = mask.float() + neg_mask.float()
        # Tensor[N] 每个图对应的 loss
        con_loss = (con * (mask_z)).sum(dim=1)

        # 总损失为[n]
        total_loss = loc_loss + con_loss

        # 没有正样本的图像不计算分类损失  pos_num是每一张图片的正例个数 eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=torch.finfo(torch.float).eps)
        # 每个图片平均一下 输出一个值
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret


class PostProcess(nn.Module):
    def __init__(self, ancs, variance=(0.1, 0.2), criteria=0.5, max_output=100):
        '''

        :param ancs: 生成的基础ancs  xywh  只有一张图  (m*w*h,4)
        :param variance: 用于控制预测值
        :param criteria: iou阀值
        :param max_output: 最大预测数
        '''
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        unsqueeze = ancs.transpose(0, 1).unsqueeze(dim=0)
        self.anc_bboxs = nn.Parameter(unsqueeze, requires_grad=False)
        self.scale_xy = variance[0]
        self.scale_wh = variance[1]

        self.criteria = criteria  # 非极大超参
        self.max_output = max_output  # 最多100个目标

    def scale_back_batch(self, ploc, plabel):
        '''
            修正def 并得出分数 softmax
            1）通过预测的 boxes 回归参数得到最终预测坐标
            2）将box格式从 xywh 转换回ltrb
            3）将预测目标 score通过softmax处理
        :param ploc: 预测出的框 偏移量 xywh [N, 4, 8732]
        :param plabel: 预测所属类别 [N, label_num, 8732]
        :return:  返回 anc+预测偏移 = 修复后anc 的 ltrb 形式
        '''
        # type: (Tensor, Tensor)
        # Returns a view of the original tensor with its dimensions permuted.
        # [batch, 4, 8732] -> [batch, 8732, 4]
        ploc = ploc.permute(0, 2, 1)
        # [batch, label_num, 8732] -> [batch, 8732, label_num]
        plabel = plabel.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        # ------------限制预测值------------
        ploc[:, :, :2] = self.scale_xy * ploc[:, :, :2]  # 预测的x, y回归参数
        ploc[:, :, 2:] = self.scale_wh * ploc[:, :, 2:]  # 预测的w, h回归参数
        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        ploc[:, :, :2] = ploc[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        ploc[:, :, 2:] = ploc[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # 修复完成转回至ltrb
        xywh2ltrb(ploc)

        # scores_in: [batch, 8732, label_num]  输出8732个分数 -1表示最后一个维度
        return ploc, F.softmax(plabel, dim=-1)

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
        '''
        一张图片的  修复后最终框 通过极大抑制 的 ltrb 形式
        :param bboxes_in: (Tensor 8732 x 4)
        :param scores_in: scores_in (Tensor 8732 x nitems) 多类别分数
        :param criteria: IoU threshold of bboexes IoU 超参
        :param num_output: 最大预测数 超参
        :return:
        '''
        # type: (Tensor, Tensor, float, int)
        device = bboxes_in.device

        bboxes_in = bboxes_in.clamp(min=0, max=1)  # 对越界的bbox进行裁剪

        '''---组装数据 按21个类拉平数据---'''
        num_classes = scores_in.shape[-1]  # 取类别数 21类
        # [8732, 4] -> [8732, 21, 4] 注意内存 , np高级 复制框预测框 为21个类
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # 创建labels与 bboxes_in 对应 , 用于预测结果展视
        labels = torch.arange(num_classes, device=device)
        # [num_classes] -> [8732, num_classes]
        labels = labels.view(1, -1).expand_as(scores_in)

        # 移除归为背景类别的概率信息
        bboxes_in = bboxes_in[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
        scores_in = scores_in[:, 1:]  # [8732, 21] -> [8732, 20]
        labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

        # 将21个类拉平
        bboxes_in = bboxes_in.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
        scores_in = scores_in.reshape(-1)  # [8732, 20] -> [8732x20]
        labels = labels.reshape(-1)  # [8732, 20] -> [8732x20]

        # 过滤...移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        # remove empty boxes 面积小的
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)  # 目标大于1个像素的
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression 将所有类别拉伸后
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]  # 最大100个目录
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    def forward(self, ploc, plabel):
        '''
        将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        :param ploc: 预测出的框 偏移量 torch.Size([1, 4, 8732])
        :param plabel: 预测所属类别 torch.Size([1, 21, 8732])
        :return: list每一个图的 多个bboxes_out, labels_out, scores_out
        '''
        # 通过预测的boxes回归参数得到最终预测坐标, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(ploc, plabel)

        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        # 遍历一个batch中的每张image数据
        # bboxes: [batch, 8732, 4] 0维分割 得每一个图片
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(split_size=1, dim=0)):
            # bbox: [1, 8732, 4]
            bbox = bbox.squeeze(0)  # anc+预测偏移 = 修复后anc 的 ltrb 形式
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs
