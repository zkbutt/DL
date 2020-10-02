import torch
import numpy as np


def ltrb2xywh(bboxs):
    bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
    bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]
    bboxs[:, 0] = bboxs[:, 0] + 0.5 * bboxs[:, 2]
    bboxs[:, 1] = bboxs[:, 1] + 0.5 * bboxs[:, 3]
    return bboxs


def ltrb2ltwh(bboxs):
    '''
    :param bboxs:
    :return:
    '''
    bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
    bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]
    return bboxs


def xywh2ltrb(bboxs):
    '''

    :param bboxs: [N, 4, 8732]
    :return:
    '''
    l = bboxs[:, :, 0] - 0.5 * bboxs[:, :, 2]
    t = bboxs[:, :, 1] - 0.5 * bboxs[:, :, 3]
    r = bboxs[:, :, 0] + 0.5 * bboxs[:, :, 2]
    b = bboxs[:, :, 1] + 0.5 * bboxs[:, :, 3]
    bboxs[:, :, 0] = l  # xmin
    bboxs[:, :, 1] = t  # ymin
    bboxs[:, :, 2] = r  # xmax
    bboxs[:, :, 3] = b  # ymax
    return bboxs


def fix_anc4p(bboxs, loc, variances=(0.1, 0.2)):
    '''
    用于预测时
    :param bboxs: y,x,h,w
    :param loc: 修正系数
    :return: 中心点和 宽高的调整系数
    '''
    # 坐标移动
    bboxs[:, :2] = bboxs[:, :2] + loc[:, :2] * variances[0] * bboxs[:, :2]
    # 宽高缩放
    bboxs[:, 2:] = bboxs[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    return bboxs


def bbox_iou4np(bbox_a, bbox_b):
    '''
    求所有bboxs 与所有标定框 的交并比 ltrb
    返回一个数
    :param bbox_a: 多个预测框 (n,4)
    :param bbox_b: 多个标定框 (k,4)
    :return: <class 'tuple'>: (2002, 2) (n,k)
    '''
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    '''
    确定交叉部份的坐标  以下设 --- n = 3 , k = 4 ---
    广播 对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值 输出 (n,k,2)
    左上 右下
    '''
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # (3,1,2) (4,2)--->(3,4,2)
    # 选出前两个最小的 ymin，xmin 左上角的点 后面的通过广播
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    '''
    相交的面积 只有当右下的最小坐标  >>(xy全大于) 左上的最大坐标时才相交 用右下-左上得长宽
    '''
    # (2002,2,2)的每一第三维 降维运算(2002,2)  通过后面的是否相交的 降维运算 (2002,2)赛选
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # (2002,2) axis=1 2->1 行相乘 长*宽 --->降维运算(2002)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    # (2,2) axis=1 2->1 行相乘 长*宽 --->降维运算(2)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    #  (2002,1) +(2) = (2002,2) 每个的二维面积
    _a = area_a[:, None] + area_b
    _area_all = (_a - area_i)  # (2002,2)-(2002,2)
    return area_i / _area_all  # (2002,2)


def bbox_iou4ts(box_a, box_b):
    '''
       通过左上右下计算 IoU  全是 ltrb
       calc_iou4ts 这个方法更优雅
       :param box_a:  bboxs
       :param box_b: 先验框
       :return: (a.shape[0] b.shape[0])
   '''
    a = box_a.shape[0]
    b = box_b.shape[0]
    # x,2 -> x,1,2 -> x b 2  ,选出rb最大的
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(a, b, 2),
        box_b[:, 2:].unsqueeze(0).expand(a, b, 2)
    )
    # 选出lt最小的
    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(a, b, 2),
        box_b[:, :2].unsqueeze(0).expand(a, b, 2)
    )
    # 得有有效的长宽 np高级
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]  # torch.Size([a个, b个, 2])

    # # x,2 -> x,1,2 -> x b 2 直接算面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    iou = inter / union
    return iou


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


def calc_iou4ts(boxes1, boxes2):
    '''

    :param boxes1: (Tensor[N, 4])  bboxs
    :param boxes2: (Tensor[M, 4])  ancs
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


def resize_boxes4tensor(boxes, original_size, new_size):
    '''
    用于预处理 和 最后的测试(预测还原)
    :param boxes: 输入多个
    :param original_size: 图像缩放前的尺寸
    :param new_size: 图像缩放后的尺寸
    :return:
    '''
    # type: (Tensor, List[int], List[int]) -> Tensor
    # 输出数组   新尺寸 /旧尺寸 = 对应 h w 的缩放因子
    ratios = [
        torch.tensor(s_new, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s_new, s_orig in zip(new_size, original_size)
    ]

    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(dim=1)  # 分列
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


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


@torch.no_grad()
def pos_match(ancs, bboxs, criteria):
    '''
    正样本选取策略
        1. 每一个bboxs框 尽量有一个anc与之对应
        2. 每一个anc iou大于大于阀值的保留
    :param ancs:  anc模板
    :param bboxs: 标签框
    :param criteria: 小于等于0.35的反例
    :return:
        label_neg_mask: 返回反例的布尔索引
        anc_bbox_ind : anc对应的bbox的index 用于提出标签
    '''
    # (bboxs个,anc个)
    iou = calc_iou4ts(bboxs, ancs)
    # (1,anc个值)降维  每一个anc对应的bboxs  最大的iou和index
    anc_bbox_iou, anc_bbox_ind = iou.max(dim=0)  # 存的是 bboxs的index

    # (bbox个,1)降维  每一个bbox对应的anc  最大的iou和index
    bbox_anc_iou, bbox_anc_ind = iou.max(dim=1)  # 存的是 anc的index

    # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
    anc_bbox_iou.index_fill_(0, bbox_anc_ind, 2)  # dim index val

    # 确保保留的 anc 的 gt  index 不会出错 exp:一个anc对应多个bboxs, bbox只有这一个anc时
    _ids = torch.arange(0, bbox_anc_ind.size(0), dtype=torch.int64).to(anc_bbox_ind)
    anc_bbox_ind[bbox_anc_ind[_ids]] = _ids
    # for j in range(bbox_anc_ind.size(0)): # 与上句等价
    #     anc_bbox_ind[bbox_anc_ind[j]] = j

    # ----------正例的index 和 正例对应的bbox索引----------
    # masks = anc_bbox_iou > criteria  # anc 的正例index
    # masks = anc_bbox_ind[masks]  # anc对应最大bboxs的索引 进行iou过滤筛选 形成正例对应的bbox的索引

    # 这里改成反倒置0
    label_neg_mask = anc_bbox_iou <= criteria  # anc 的正例index

    return label_neg_mask, anc_bbox_ind


if __name__ == '__main__':
    pass