import numpy as np
from math import sqrt
import itertools
import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List
from torch import nn, Tensor


# This function is from https://github.com/kuangliu/pytorch-ssd.
# def calc_iou_tensor(box1, box2):
#     """ Calculation of IoU based on two boxes tensor,
#         Reference to https://github.com/kuangliu/pytorch-src
#         input:
#             box1 (N, 4)  format [xmin, ymin, xmax, ymax]
#             box2 (M, 4)  format [xmin, ymin, xmax, ymax]
#         output:
#             IoU (N, M)
#     """
#     N = box1.size(0)
#     M = box2.size(0)
#
#     # (N, 4) -> (N, 1, 4) -> (N, M, 4)
#     be1 = box1.unsqueeze(1).expand(-1, M, -1)  # -1 means not changing the size of that dimension
#     # (M, 4) -> (1, M, 4) -> (N, M, 4)
#     be2 = box2.unsqueeze(0).expand(N, -1, -1)
#
#     # Left Top and Right Bottom
#     lt = torch.max(be1[:, :, :2], be2[:, :, :2])
#     rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])
#
#     # compute intersection area
#     delta = rb - lt  # width and height
#     delta[delta < 0] = 0
#     # width * height
#     intersect = delta[:, :, 0] * delta[:, :, 1]
#
#     # compute bel1 area
#     delta1 = be1[:, :, 2:] - be1[:, :, :2]
#     area1 = delta1[:, :, 0] * delta1[:, :, 1]
#     # compute bel2 area
#     delta2 = be2[:, :, 2:] - be2[:, :, :2]
#     area2 = delta2[:, :, 0] * delta2[:, :, 1]
#
#     iou = intersect / (area1 + area2 - intersect)
#     return iou


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou_tensor(boxes1, boxes2):
    '''

    :param boxes1: (Tensor[N, 4])
    :param boxes2: (Tensor[M, 4])
    :return: (Tensor[N, M])
    '''
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


class Encoder(object):
    """
    This function is from https://github.com/kuangliu/pytorch-ssd

        Inspired by https://github.com/kuangliu/pytorch-src
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes_ltrb = dboxes(order='ltrb')  # torch.Size([8732, 4])
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)  # torch.Size([1,8732, 4])
        self.nboxes = self.dboxes_ltrb.size(0)  # default boxes的数量
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, gt, labels_in, criteria=0.5):
        '''
        与 box_utils中 match 类似
        多张图片运行多次 难 编码类

        :param self.dboxes_ltrb def 计算iou用ltrb

        :param gt: 真实的GT框 (2,4) VOC 左上右下 --- 这是一张图片的 gt
        :param labels_in:  gt对应的 的类别值 1~20 --- 现在还没有背景
        :param criteria: iou threshold
        :return: 为什么要替换???     输出(x,y,w,h)
            在 一张图的 def 中找出正样本 并用对应的GT框替换def 并匹配其真实标签 1~20
            输出所有bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        '''
        ious = calc_iou_tensor(gt, self.dboxes_ltrb)  # [gt个, 8732]

        # 每个def匹配到的最大IoU gt [8732,] 个
        best_def_ious, best_def_idx = ious.max(dim=0)
        # 每个gt匹配到的最大IoU def [nboxes,]个
        best_gt_ious, best_gt_idx = ious.max(dim=1)

        # 每个gt匹配到的最大IoU 的 default box  iou分数设置为2 即保留为正例
        best_def_ious.index_fill_(0, best_gt_idx, 2.0)  # 保留gt个 有可能是多对一 实际保留<=gt的个数

        # 修正对应iou最大值的 gt  index 不会出错
        idx = torch.arange(0, best_gt_idx.size(0), dtype=torch.int64)
        best_def_idx[best_gt_idx[idx]] = idx

        # 寻找与bbox_in iou大于0.5的default box,对应论文中Matching strategy的第二条(这里包括了第一条匹配到的信息)
        masks = best_def_ious > criteria  # 布尔索引 是降维后的 1维

        # [8732,] 标识def 对应的 gt 索引
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)  # 标签默认为同维 类别0为负样本
        labels_out[masks] = labels_in[best_def_idx[masks]]  # 将正样本的GT 的真 实标签选出来 1~20
        bboxes_out = self.dboxes_ltrb.clone()
        # 把找出正样本的框 设置为 gt
        bboxes_out[masks, :] = gt[best_def_idx[masks], :]

        # 转换成 (x,y,w,h)
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])  # x
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])  # y
        w = bboxes_out[:, 2] - bboxes_out[:, 0]  # w
        h = bboxes_out[:, 3] - bboxes_out[:, 1]  # h
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out

    # def scale_back_batch(self, bboxes_in, scores_in):
    #     """
    #         将box格式从xywh转换回ltrb, 将预测目标score通过softmax处理
    #         Do scale and transform from xywh to ltrb
    #         suppose input N x 4 x num_bbox | N x label_num x num_bbox
    #
    #         bboxes_in: 是网络预测的xywh回归参数
    #         scores_in: 是预测的每个default box的各目标概率
    #     """
    #     if bboxes_in.device == torch.device("cpu"):
    #         self.dboxes_ltrb = self.dboxes_ltrb.cpu()
    #         self.dboxes_xywh = self.dboxes_xywh.cpu()
    #     else:
    #         self.dboxes_ltrb = self.dboxes_ltrb.cuda()
    #         self.dboxes_xywh = self.dboxes_xywh.cuda()
    #
    #     # Returns a view of the original tensor with its dimensions permuted.
    #     bboxes_in = bboxes_in.permute(0, 2, 1)
    #     scores_in = scores_in.permute(0, 2, 1)
    #     # print(bboxes_in.is_contiguous())
    #
    #     bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]  # 预测的x, y回归参数
    #     bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]  # 预测的w, h回归参数
    #
    #     # 将预测的回归参数叠加到default box上得到最终的预测边界框
    #     bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
    #     bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]
    #
    #     # transform format to ltrb
    #     l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
    #     t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
    #     r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
    #     b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]
    #
    #     bboxes_in[:, :, 0] = l  # xmin
    #     bboxes_in[:, :, 1] = t  # ymin
    #     bboxes_in[:, :, 2] = r  # xmax
    #     bboxes_in[:, :, 3] = b  # ymax
    #
    #     return bboxes_in, F.softmax(scores_in, dim=-1)

    # def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
    #     # 将box格式从xywh转换回ltrb（方便后面非极大值抑制时求iou）, 将预测目标score通过softmax处理
    #     bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
    #
    #     outputs = []
    #     # 遍历一个batch中的每张image数据
    #     for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
    #         bbox = bbox.squeeze(0)
    #         prob = prob.squeeze(0)
    #         outputs.append(self.decode_single_new(bbox, prob, criteria, max_output))
    #     return outputs

    # def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
    #     """
    #     decode:
    #         input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
    #         output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
    #         criteria : IoU threshold of bboexes
    #         max_output : maximum number of output bboxes
    #     """
    #     device = bboxes_in.device
    #     num_classes = scores_in.shape[-1]
    #
    #     # 对越界的bbox进行裁剪
    #     bboxes_in = bboxes_in.clamp(min=0, max=1)
    #
    #     # [8732, 4] -> [8732, 21, 4]
    #     bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)
    #
    #     # create labels for each prediction
    #     labels = torch.arange(num_classes, device=device)
    #     labels = labels.view(1, -1).expand_as(scores_in)
    #
    #     # 移除归为背景类别的概率信息
    #     bboxes_in = bboxes_in[:, 1:, :]
    #     scores_in = scores_in[:, 1:]
    #     labels = labels[:, 1:]
    #
    #     # batch everything, by making every class prediction be a separate instance
    #     bboxes_in = bboxes_in.reshape(-1, 4)
    #     scores_in = scores_in.reshape(-1)
    #     labels = labels.reshape(-1)
    #
    #     # 过滤...移除低概率目标，self.scores_thresh=0.05
    #     inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
    #     bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]
    #
    #     # remove empty boxes 面积小的
    #     ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
    #     keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
    #     keep = keep.nonzero(as_tuple=False).squeeze(1)
    #     bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]
    #
    #     # non-maximum suppression 将所有类别拉伸后
    #     keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)
    #
    #     # keep only topk scoring predictions
    #     keep = keep[:num_output]
    #     bboxes_out = bboxes_in[keep, :]
    #     scores_out = scores_in[keep]
    #     labels_out = labels[keep]
    #
    #     return bboxes_out, labels_out, scores_out

    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """  nvidia 非极大算法 nms
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        # Reference to https://github.com/amdegroot/ssd.pytorch
        bboxes_out = []
        scores_out = []
        labels_out = []

        # 非极大值抑制算法
        # scores_in (Tensor 8732 x nitems), 遍历返回每一列数据，即8732个目标的同一类别的概率
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            # [8732, 1] -> [8732]
            score = score.squeeze(1)

            # 虑除预测概率小于0.05的目标
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            # 按照分数从小到大排序
            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                # 获取排名前score_idx_sorted名的bboxes信息 Tensor:[score_idx_sorted, 4]
                bboxes_sorted = bboxes[score_idx_sorted, :]
                # 获取排名第一的bboxes信息 Tensor:[4]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                # 计算前score_idx_sorted名的bboxes与第一名的bboxes的iou
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()

                # we only need iou < criteria
                # 丢弃与第一名iou > criteria的所有目标(包括自己本身)
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                # 保存第一名的索引信息
                candidates.append(idx)

            # 保存该类别通过非极大值抑制后的目标信息
            bboxes_out.append(bboxes[candidates, :])  # bbox坐标信息
            scores_out.append(score[candidates])  # score信息
            labels_out.extend([i] * len(candidates))  # 标签信息

        if not bboxes_out:  # 如果为空的话，返回空tensor，注意boxes对应的空tensor size，防止验证时出错
            return [torch.empty(size=(0, 4)), torch.empty(size=(0,), dtype=torch.int64), torch.empty(size=(0,))]

        bboxes_out = torch.cat(bboxes_out, dim=0).contiguous()
        scores_out = torch.cat(scores_out, dim=0).contiguous()
        labels_out = torch.tensor(labels_out, dtype=torch.long)

        # 对所有目标的概率进行排序（无论是什么类别）,取前max_num个目标
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        '''

        :param fig_size: 原图尺寸 输入网络的尺寸 300
        :param feat_size:特图尺寸 [38, 19, 10, 5, 3, 1]
        :param steps: [8, 16, 32, 64, 100, 300] 不等于 原图/特图 这个值怎么来的?
        :param scales:[21, 45, 99, 153, 207, 261, 315]
        :param aspect_ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        :param scale_xy: 用于损失函数修正系数
        :param scale_wh: 用于损失函数修正系数
        '''
        self.fig_size = fig_size  # 输入网络的图像大小 300
        self.feat_size = feat_size  # 每个预测层的feature map尺寸

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        #
        self.steps = steps  # 每个特征层上的一个cell在原图上的跨度

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales  # 每个特征层上预测的default box的scale
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios  # 每个预测特征层上预测的default box的ratios

        fk = fig_size / np.array(steps)  # 计算每层特征层的fk 跨度值用于与调整系数相乘得真实框

        self.default_boxes = []
        # size of feature and number of feature
        # 遍历每层特征层，计算default box
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size  # scale转为相对值[0-1]
            sk2 = scales[idx + 1] / fig_size  # scale转为相对值[0-1] 下一层的尺寸
            sk3 = sqrt(sk1 * sk2)
            # 先添加两个1:1比例的default box宽和高 定制加入1比1
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # 再将剩下不同比例的default box宽和高添加到all_sizes中
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # 计算当前特征层对应原图上的所有default box
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):  # i -> 行（y）， j -> 列（x）
                    # 计算每个default box的中心坐标（范围是在0-1之间）
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        # 将default_boxes转为tensor格式 中心宽高
        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float32)  # 这里不转类型会报错
        self.dboxes.clamp_(min=0, max=1)  # 将坐标（x, y, w, h）都限制在0-1之间

        # For IoU calculation 转左上右下
        # ltrb is left top coordinate and right bottom coordinate
        # 将(x, y, w, h)转换成(xmin, ymin, xmax, ymax)，方便后续计算IoU(匹配正负样本时)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]  # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]  # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]  # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]  # ymax

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order='ltrb'):
        # 根据需求返回对应格式的default box
        if order == 'ltrb':
            return self.dboxes_ltrb

        if order == 'xywh':
            return self.dboxes


def dboxes300_coco():
    '''这个方法被调了两次 模型时 预处理时'''
    figsize = 300  # 输入网络的图像大小
    feat_size = [38, 19, 10, 5, 3, 1]  # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]  # 每个特征层上的一个cell在原图上的跨度
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]  # 每个特征层上预测的default box的scale
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # 每个预测特征层上预测的default box的ratios
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def nms(boxes, scores, iou_threshold):
    '''
         boxes (Tensor[N, 4])) – bounding boxes坐标. 格式：(x1, y1, x2, y2)
         scores (Tensor[N]) – bounding boxes得分
         iou_threshold (float) – IoU过滤阈值
     返回:NMS过滤后的bouding boxes索引（降序排列）
     '''
    # type: (Tensor, Tensor, float) -> Tensor
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    '''

    :param boxes: 拉平所有类别的box重复的 n*20,4
    :param scores:
    :param idxs:  真实类别index
    :param iou_threshold:
    :return:
    '''
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # torchvision.ops.boxes.batched_nms(boxes, scores, lvl, nms_thresh)
    # 根据最大的一个值确定每一类的偏移
    max_coordinate = boxes.max()  # 选出每个框的 坐标最大的一个值
    # idxs 的设备和 boxes 一致 , 真实类别index * (1+最大值) 则确保同类框向 左右平移 实现隔离
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes 加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4] 增加一维
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0),
                                        requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5  # 非极大超参
        self.max_output = 100  # 最多100个目标

    def scale_back_batch(self, bboxes_in, scores_in):
        '''
            修正def 并得出分数 softmax
            1）通过预测的 boxes 回归参数得到最终预测坐标
            2）将box格式从 xywh 转换回ltrb
            3）将预测目标 score通过softmax处理
        :param bboxes_in: 预测出的框 偏移量 xywh [N, 4, 8732]
        :param scores_in: 预测所属类别 [N, label_num, 8732]
        :return:  返回 anc+预测偏移 = 修复后anc 的 ltrb 形式
        '''
        # type: (Tensor, Tensor)
        # Returns a view of the original tensor with its dimensions permuted.
        # [batch, 4, 8732] -> [batch, 8732, 4]
        bboxes_in = bboxes_in.permute(0, 2, 1)
        # [batch, label_num, 8732] -> [batch, 8732, label_num]
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        # ------------固定公式修正预测框(前面有乘10,这里0.1) 0.1 -> 10 ;  0.2 -> 5 ------------
        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]  # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]  # 预测的w, h回归参数
        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # 修复完成转回至ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]
        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        # scores_in: [batch, 8732, label_num]  输出8732个分数 -1表示最后一个维度
        return bboxes_in, F.softmax(scores_in, dim=-1)

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

    def forward(self, bboxes_in, scores_in):
        '''
        将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        :param bboxes_in: 预测出的框 偏移量
        :param scores_in: 预测所属类别
        :return: list每一个图的 多个bboxes_out, labels_out, scores_out
        '''
        # 通过预测的boxes回归参数得到最终预测坐标, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        # 遍历一个batch中的每张image数据
        # bboxes: [batch, 8732, 4] 0维分割 得每一个图片
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(split_size=1, dim=0)):
            # bbox: [1, 8732, 4]
            bbox = bbox.squeeze(0)  # anc+预测偏移 = 修复后anc 的 ltrb 形式
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs
