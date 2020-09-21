import numpy as np
from torchvision.ops import nms
import torch
from object_detection.simple_faster_rcnn_pytorch_master.model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox


class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.
    roi处使用，roi和gt匹配得出正负样本的函数
    """

    def __init__(self, n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        '''

        :param n_sample: 超参 训练时输出的 样本数量
        :param pos_ratio: 超参 训练时输出的 正样本比例
        :param pos_iou_thresh:
        :param neg_iou_thresh_hi:
        :param neg_iou_thresh_lo:
        '''
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        '''

        :param roi: <class 'tuple'>: (1726, 4) 建议框 2000个 有可能被抑制
        :param bbox:  torch.Size([2, 4]) 标签框 GT
        :param label: 对应标签 torch.Size([1, 2])
        :param loc_normalize_mean: 修正归一化参数
        :param loc_normalize_std: 修正归一化参数
        :return:
            sample_roi 选出的128个样本 正反例 <class 'tuple'>: (128, 4)
            gt_roi_loc 选出的样本与GT损失 <class 'tuple'>: (128, 4)
            gt_roi_label 对应GT的标签 反例为0 <class 'tuple'>: (128,)
        '''
        n_bbox, _ = bbox.shape
        # 将GT也组合roi去
        roi = np.concatenate((roi, bbox), axis=0)
        # 需要的正样本个数,128*0.25
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        # 这里的最后两个一定是1 <class 'tuple'>: (1728, 2)
        iou = bbox_iou(roi, bbox)  # 获得shape=(建议框,GT)的iou 与_calc_ious方法
        max_iou = iou.max(axis=1)
        # 用于GT选择 和GT标签
        gt_assignment = iou.argmax(axis=1)  # 降维 每个建议框对应(属于)bbox的索引 根据最大IOU (根据IOU判断类型)
        gt_roi_label = label[gt_assignment] + 1  # 根据iou 形成每个建议框可能对应的类型 索引加1 为实际的标签

        # 超参iou正例阀值 影响目标交叉
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # 选出正例 index     最终正样本数取最小的
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:  # 在正例 index 中, 随机选出多少个索引
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # 选出正例 index
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image  # 留下反例数
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)  # 合并正反索引
        gt_roi_label = gt_roi_label[keep_index]  # <class 'tuple'>: (1872,)
        sample_roi = roi[keep_index]  # 正负样本一共 n_sample=128个
        gt_roi_label[pos_roi_per_this_image:] = 0  # 反例都在后面 ,反倒的标签全设为0 --> 0

        sample_bbox = bbox[gt_assignment[keep_index]]
        gt_roi_loc = bbox2loc(sample_roi, sample_bbox)  # 获得roi和GT的偏移值
        # 根据公式 偏移值需要归一化
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32))
                      / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):  # rpn处使用，anchor和gt匹配得到正负样本的函数
    """Assign the ground truth bounding boxes to anchors.
        anchor和gt匹配
    """

    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        '''

        :param n_sample:
        :param pos_iou_thresh:
        :param neg_iou_thresh:
        :param pos_ratio:
        '''
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        '''

        :param bbox: GT
        :param anchor: 所有anchor
        :param img_size: 原始尺寸
        :return:
            loc 生成的与gt的偏差 <class 'tuple'>: (18648, 4)
            label 根据GT算出来的类别 <class 'tuple'>: (18648,)
        '''

        img_H, img_W = img_size

        # 剔除超边界
        inside_index = _get_inside_index(anchor, img_H, img_W)
        n_anchor = len(anchor)  # 保存未筛选前的数量 用于后面修正还原
        anchor = anchor[inside_index]  # 筛选出来
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        # 获得建议框 和GT的偏移值
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # 将赛选还原回去
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        '''
        根据 与GT的比较 算出类型 和iou
        :param inside_index: 已剔除边界的indel
        :param anchor: 已剔除边界的 anchor, anchor = anchor[inside_index]
        :param bbox: GT框
        :return:
            argmax_ious : 选框对应的最大IOU索引值
            label : 选框的类别
        '''
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)  # 默认为-1 不处理

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        # 小于超参 标记为负
        label[max_ious < self.neg_iou_thresh] = 0

        # GT对应的最大的标记为正
        label[gt_argmax_ious] = 1

        # 及大于阀值的标签标记为正
        label[max_ious >= self.pos_iou_thresh] = 1

        # 超参正负样本比例
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:  # 正样多了就随机选
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1  # 剩余的标记为无效

        # 选负样
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        '''
        分别找出 索引 及 每一个选框对应的最大的IOU值
        :param anchor: 已剔除边界的 anchor, anchor = anchor[inside_index]
        :param bbox: GT
        :param inside_index: 剔除超出边际的后的index 这个参数 可以不要
        :return:
            argmax_ious  (2002) 个 与GT最大的索引值  选框每个对应的IOU值
            max_ious (2002) 对应的最大iou值
            gt_argmax_ious 每类GT 对应iou最大的选框 索引
        '''
        ious = bbox_iou(anchor, bbox)  # 返回 (2002, 2)
        argmax_ious = ious.argmax(axis=1)  # 每个选框 对哪个GT的IOU大索引 降维

        max_ious = ious[np.arange(len(inside_index)), argmax_ious]  # 每个选框的最大iou

        gt_argmax_ious = ious.argmax(axis=0)  # # 与GT最大的选框的 返回GT个数个 对应选框的索引 降维
        # 与GT最大的选框的 返回GT个数个 对应选框的索引
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]  # d1[[0, 1, 2], np.arange(3)] 这样一定取到最大值
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    '''
    将赛选还原回去---(与原始输入一直,并进行值修正)
    将没有选出来的选框偏移置0 类别为-1
    :param data: 对应正负无效样本的标签
    :param count: n_anchor
    :param index:
    :param fill:
    :return:
    '''
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)  # <class 'tuple'>: (6840,)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    '''
    选框超出图片的不要
    :param anchor:
    :param H:
    :param W:
    :return:
    '''
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:  # 创建roi的函数
    """Proposal regions are generated by calling this object.
        建议框选择器
    """

    def __init__(self, parent_model, nms_thresh=0.7,
                 n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms=6000, n_test_post_nms=300,
                 min_size=16):
        '''

        :param parent_model: 用于获取 self.parent_model.training = Ture 的参数 训练和预测选用不同的策略选择
        :param nms_thresh:
        :param n_train_pre_nms:
        :param n_train_post_nms:
        :param n_test_pre_nms:
        :param n_test_post_nms:
        :param min_size: 指定选框小阀值,hw小于此值的不要
        '''
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        '''
        NOTE: when test, remember
            faster_rcnn.eval()
            to set self.traing = False

        限制超出图片的roi(roi+img_size)
        用scale 剔除小框
        根据超参设备选分数高的 训练 预测不同 例如12000个
        根据nms 极大抑制 相交同类去除

        :param loc: 所有 anchor 取出来  (hh*ww*9,4)
        :param score: 前景概率 (hh*ww*9)
        :param anchor: (hh*ww*9,4) 已回归的 anchor
        :param img_size: 原图 size
        :param scale: 这个是超参 确定小建议框的阀值
        :return: 返回 (300,4) 建议框
        '''
        # 训练和预测时使用不同的策略
        if self.parent_model.training:  # 设置超参数
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # <class 'tuple'>: (18648, 4)
        roi = loc2bbox(anchor, loc)  # anchor 和 rpn预测出的偏移 返回全部已修正的

        # 这两步从高宽限制超出尺寸的roi 可用 _get_inside_index(anchor, H, W) 方法
        _slice = roi[:, slice(0, 4, 2)]
        # _slice1 = roi[:, 0:4:2] # 与直接切片等效
        roi[:, slice(0, 4, 2)] = np.clip(_slice, 0, img_size[0])  # 修正超出原图边界的 h
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # 用scale 剔除小框 Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale  # scale 这个是超参
        hs = roi[:, 2] - roi[:, 0]  # ymax - ymin = hs
        ws = roi[:, 3] - roi[:, 1]  # xmax - xmin = ws
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]  # 只要第一维即可
        roi = roi[keep, :]  # 选出hw同时大于的框
        score = score[keep]  # 选出hw同时大于的框对应的分数

        # 根据超参设置选分数高的 训练 预测不同 例如12000个
        order = score.ravel().argsort()[::-1]  # 分数从高到低排名 这里可以不要ravel()
        _order = score.argsort()[::-1]
        if n_pre_nms > 0:  # 数前n_pre_nms个框留下
            order = order[:n_pre_nms]  # 排序后的分数
        roi = roi[order, :]
        score = score[order]

        # 根据nms 极大抑制 相交同类去除
        '''
        arg
        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).
            boxes (Tensor[N, 4])) – boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2) format
            scores (Tensor[N]) – scores for each one of the boxes
            iou_threshold (float) – discards all overlapping boxes with IoU > iou_threshold
        ret: 返回int64 tensor 按分数递减
        '''
        keep = nms(torch.tensor(roi.copy()), torch.tensor(score.copy()), 0.7)
        if n_post_nms > 0:  # 非极大抑制2000个
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi
