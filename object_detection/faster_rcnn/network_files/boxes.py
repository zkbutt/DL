import torch
from torch.jit.annotations import Tuple
from torch import Tensor
import torchvision


def nms(boxes, scores, iou_threshold):
    '''
    调用 封装的函数
    :param boxes:
    :param scores:
    :param iou_threshold:
    :return:
    '''
    # type: (Tensor, Tensor, float)
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    '''
    这里是遍历出来的 SSD有批量的方法
    :param boxes: 每一图的 torch.Size([8698, 4])
    :param scores: 每一图的 分数
    :param idxs: 每一图对应的特图索引
    :param iou_threshold: 超参
    :return:
    '''
    # type: (Tensor, Tensor, Tensor, float)
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # 为每一个类别生成一个很大的偏移量
    to = idxs.to(boxes)  # to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = to * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象 多特图同时nms的高级技巧
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    '''
    得分过小的不要
    :param boxes:
    :param min_size:  超参
    :return: 索引
    '''
    # type: (Tensor, float)
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  # 预测boxes的宽和高
    keep = (ws >= min_size) & (hs >= min_size)  # 当满足宽，高都大于给定阈值时为True
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    nonzero = keep.nonzero(as_tuple=False) # torch.Size([8741, 1]) 这个要报错 原因是keep是一维的
    # nonzero = torch.nonzero(keep[None, :])  # torch.Size([8741, 1]) 进行修正后 后续又要报错
    keep = nonzero.squeeze(1)  # torch.Size([8741])
    return keep  # torch.Size([8741])


def clip_boxes_to_image(boxes, size):
    '''
    boxes 进行修正
    :param boxes: 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
    :param size:
    :return:
    '''
    # type: (Tensor, Tuple[int, int])
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    if torchvision._is_tracing():  # 预测
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)  # 限制x坐标范围在[0,width]之间
        boxes_y = boxes_y.clamp(min=0, max=height)  # 限制y坐标范围在[0,height]之间

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


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


def box_iou(boxes1, boxes2):
    '''
    计算 各个框组的 iou
    :param boxes1: gt n
    :param boxes2: 建议框 m
    :return: boxes1 (n,m)
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
