import torch
import math
from torch.jit.annotations import List, Tuple
from torch import Tensor
from f_tools.GLOBAL_LOG import flog


# TODO: https://github.com/pytorch/pytorch/issues/26727


def zeros_like(tensor, dtype):
    # type: (Tensor, int) -> Tensor
    return torch.zeros_like(tensor, dtype=dtype, layout=tensor.layout,
                            device=tensor.device, pin_memory=tensor.is_pinned())


@torch.jit.script
class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float)
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor])
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs
        for matched_idxs_per_image in matched_idxs:
            # >= 1的为正样本, nonzero返回非零元素索引
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # = 0的为负样本
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            # 指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # 如果正样本数量不够就直接采用所有正样本
            num_pos = min(positive.numel(), num_pos)
            # 指定负样本数量
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 随机选择指定数量的正负样本
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = zeros_like(matched_idxs_per_image,
                                                dtype=torch.uint8).to(device=pos_idx_per_image.device)
            neg_idx_per_image_mask = zeros_like(matched_idxs_per_image,
                                                dtype=torch.uint8).to(device=pos_idx_per_image.device)

            pos_idx_per_image_mask[pos_idx_per_image] = torch.tensor(1, dtype=torch.uint8,
                                                                     device=pos_idx_per_image.device)
            neg_idx_per_image_mask[neg_idx_per_image] = torch.tensor(1, dtype=torch.uint8,
                                                                     device=pos_idx_per_image.device)

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit.script
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # parse coordinate of center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


@torch.jit.script
class BoxCoder(object):

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        '''
        ssd会使用到 weights
        :param weights: 这个是偏移系数 默认是1 是个超参 用于回归修复
        :param bbox_xform_clip: log(1000. / 16)超参
        '''
        # type: (Tuple[float, float, float, float], float)
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        '''

        :param reference_boxes:
        :param proposals:
        :return:
        '''
        # type: (List[Tensor], List[Tensor])
        # 统计每张图像的正负样本数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        '''

        :param reference_boxes:
        :param proposals:
        :return:
        '''
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        '''
        把两个图的 anchors 维度进行统一后进行 boxes 修复
        :param rel_codes: 回归参数 拉平后
        :param boxes:每个图 对应的拉平后的a
        :return:(nnn,1,4)
        '''
        # type: (Tensor, List[Tensor])
        assert isinstance(boxes, (list, tuple))
        # if isinstance(rel_codes, (list, tuple)):
        #     rel_codes = torch.cat(rel_codes, dim=0)
        assert isinstance(rel_codes, torch.Tensor)

        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)  # list(nn,4) 按图拉平的anc -> (nnn,4)与预测结果一致

        box_sum = 0  # 这个是总数 感觉直接取预测拉平数的shape[0]即可
        for val in boxes_per_image:
            box_sum += val
        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        # 这里 reshape 可能用于修复内存
        pred_boxes = self.decode_single(rel_codes.reshape(box_sum, -1), concat_boxes)
        return pred_boxes.reshape(box_sum, -1, 4) # (nnn,1,4)

    def decode_single(self, rel_codes, boxes):
        '''
        根据 回归参数 对 anchors 进行修复 维度相同
        :param rel_codes: 拉平后的回归参数 xywh
        :param boxes: 同维拉平后的 anchors
        :return: 输出(nnn,4)
        '''
        boxes = boxes.to(rel_codes.dtype)  # 设置成一样的设备和类型

        # xmin, ymin, xmax, ymax anchors 左上右下 -> whxy
        widths = boxes[:, 2] - boxes[:, 0]  # anchor宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor高度
        ctr_x = boxes[:, 0] + 0.5 * widths  # anchor中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor中心y坐标

        wx, wy, ww, wh = self.weights  # 这4个是超参 当修正时是10
        dx = rel_codes[:, 0::4] / wx  # 预测anchors的中心坐标x回归参数 间隔采样
        dy = rel_codes[:, 1::4] / wy  # 预测anchors的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww  # 预测anchors的宽度回归参数
        dh = rel_codes[:, 3::4] / wh  # 预测anchors的高度回归参数

        # self.bbox_xform_clip=math.log(1000. / 16) 防止出现指数暴炸情况
        dw = torch.clamp(dw, max=self.bbox_xform_clip)  # 宽高进行上限控制超参
        dh = torch.clamp(dh, max=self.bbox_xform_clip)  # 防止出现指数暴炸情况

        # anchors 使用预测进行修正 根据公式
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xywh -> xmin,ymin,xmax,ymax 转换为 用于nmv 分别得(nnn,1)
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_h
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_h
        # 感觉可以用一个cat代替 (nnn,4)
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


@torch.jit.script
class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        '''
        有点难
        :param high_threshold: 当iou大于fg_iou_thresh(0.7)时视为正样本
        :param low_threshold: 当iou小于bg_iou_thresh(0.3)时视为负样本
        :param allow_low_quality_matches:
        '''
        # type: (float, float, bool)
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        '''
        处理的是其中一个 anchors 的类别
        :param match_quality_matrix: 每一个gt 对应的 anchors iou分
        :return: 返回这个 anchors 对应正例的GT索引
        '''
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # 到这里---取 对应的 anchors 最高分 返回gt索引
        matched_vals, matches = match_quality_matrix.max(dim=0)  # 多个GT时选择对应的框
        if self.allow_low_quality_matches:  # 允许低质量的框 保留每个gt最大iou的索引
            all_matches = matches.clone()  # 复制 all_matches 用于保留原始分数
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算iou小于low_threshold的    索引
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的    索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        # iou小于low_threshold的matches索引置为-1 反例  这里出错 ??????????
        # flog.debug('这里出错------ %s', '123')
        matches[below_low_threshold] = torch.tensor(self.BELOW_LOW_THRESHOLD, device=matches.device)  # -1

        # iou在[low_threshold, high_threshold]之间的matches索引置为-2 丢弃
        matches[between_thresholds] = torch.tensor(self.BETWEEN_THRESHOLDS, device=matches.device)  # -2

        if self.allow_low_quality_matches:
            # 保存每一个GT最大的分数的 anchor，即使iou低于设定的阈值
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches  # 对应正例的GT索引 不合格的为 -1 或-2

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        '''
        保存每一个GT最大的分数的 anchor，即使iou低于设定的阈值
        :param matches:  以进行-1 -2 表示的索引
        :param all_matches:  未进行 -1 -2 表示的索引
        :param match_quality_matrix:  每一个gt 对应的 anchors 预测分
        :return:
        '''
        # NP高级 找出重复的多个max
        # 对于每个gt boxes寻找与其iou最大值的 anchor，
        # highest_quality_foreach_gt为匹配到的最大iou值 降维 GT个 (一个值)
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # 选出 anchor 的最大值.
        _t = match_quality_matrix == highest_quality_foreach_gt[:, None]  # 形成与原数据一致的2维布尔索引
        _nonzero = torch.nonzero(_t)
        gt_pred_pairs_of_highest_quality = _nonzero  # 选出索引
        # Example gt_pred_pairs_of_highest_quality: 当值相同时可能是多个值
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])

        # 这里 会多GT 出现同一索引anchor  (0,39796) (1, 39796)
        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]  # 只需要anchor
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]  # 将每个GT的最大的保留下来


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    '''

    :param input:
    :param target:
    :param beta:
    :param size_average:
    :return:
    '''
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
