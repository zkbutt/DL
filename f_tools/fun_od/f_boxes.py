import math
import torch
import numpy as np
from torch.nn import functional as F

from f_tools.GLOBAL_LOG import flog


def empty_bboxes(bboxs):
    if isinstance(bboxs, np.ndarray):
        bboxs_copy = np.empty_like(bboxs, dtype=np.float)
    elif isinstance(bboxs, torch.Tensor):
        device = bboxs.device
        bboxs_copy = torch.empty_like(bboxs, device=device, dtype=torch.float)
    else:
        raise Exception('类型错误', type(bboxs))
    return bboxs_copy


def ltrb2xywh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)  # 复制矩阵
    # wh = rb - lt
    bboxs_copy[..., 2:] = bboxs[..., 2:] - bboxs[..., :2]
    # xy = lt + 0.5* cwh    or (lt+rb)/2
    bboxs_copy[..., :2] = bboxs[..., :2] + 0.5 * bboxs_copy[..., 2:]
    return bboxs_copy


def ltrb2ltwh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)
    # lt = lt
    bboxs_copy[..., :2] = bboxs[..., :2]
    # wh = rb -lt
    bboxs_copy[..., 2:] = bboxs[..., 2:] - bboxs[..., :2]
    return bboxs_copy


def ltwh2ltrb(bboxs):
    bboxs_copy = empty_bboxes(bboxs)
    # lt = lt
    bboxs_copy[..., :2] = bboxs[..., :2]
    # rb= lt +wh
    bboxs_copy[..., 2:] = bboxs[..., :2] + bboxs[..., 2:]
    return bboxs_copy


def xywh2ltrb(bboxs):
    bboxs_copy = empty_bboxes(bboxs)

    if isinstance(bboxs_copy, np.ndarray):
        fdiv = np.true_divide
        v = 2
    elif isinstance(bboxs_copy, torch.Tensor):
        fdiv = torch.true_divide
        v = torch.tensor(2, device=bboxs.device)
    else:
        raise Exception('类型错误', type(bboxs))

    # lt = xy - wh/2
    bboxs_copy[..., :2] = bboxs[..., :2] - fdiv(bboxs[..., 2:], v)
    # rb = clt + wh
    bboxs_copy[..., 2:] = bboxs_copy[..., :2] + bboxs[..., 2:]
    return bboxs_copy


def xywh2ltwh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)

    if isinstance(bboxs_copy, np.ndarray):
        fdiv = np.true_divide
        v = 2
    elif isinstance(bboxs_copy, torch.Tensor):
        fdiv = torch.true_divide
        v = torch.tensor(2, device=bboxs.device)
    else:
        raise Exception('类型错误', type(bboxs))

    # lt = xy - wh/2
    bboxs_copy[..., :2] = bboxs[..., :2] - fdiv(bboxs[..., 2:], v)
    # wh=wh
    bboxs_copy[..., 2:] = bboxs[..., 2:]
    return bboxs_copy


def ltwh2xywh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)

    # xy= lt+ wh*0.5
    bboxs_copy[..., :2] = bboxs[..., :2] + bboxs[..., 2:] * 0.5
    # wh=wh
    bboxs_copy[..., 2:] = bboxs[..., 2:]
    return bboxs_copy


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


def bbox_iou4one(box1_ltrb, box2_ltrb, is_giou=False, is_diou=False, is_ciou=False):
    '''
    这个是一一对应  box1  box2
    :param box1_ltrb: 多个预测框 (n,4)
    :param box2_ltrb: 多个标定框 (n,4)
    iou: 目标框不相交时全为0 不能反映如何相交的  --- 重叠面积
    giou: 当目标框完全包裹预测框的时候 退化为iou   --- 中心点距离
    diou: 提高收敛速度及全包裹优化
    ciou: --- 长宽比  CIOU有点不一致
    :return: n
    '''
    max_lt = torch.max(box1_ltrb[:, :2], box2_ltrb[:, :2])  # left-top [N,M,2] 多维组合用法
    min_rb = torch.min(box1_ltrb[:, 2:], box2_ltrb[:, 2:])  # right-bottom [N,M,2]
    inter_wh = (min_rb - max_lt).clamp(min=0)  # [N,M,2]
    # inter_area = inter_wh[:, 0] * inter_wh[:, 1]  # [N,M] 降维
    inter_area = torch.prod(inter_wh, dim=1)

    # 并的面积
    area1 = box_area(box1_ltrb)  # 降维 n
    area2 = box_area(box2_ltrb)  # 降维 m
    union_area = area1 + area2 - inter_area  # A+B-交=并
    union_area = union_area.clamp(min=torch.finfo(torch.float16).eps)

    iou = inter_area / union_area  # 交一定小于并
    # flog.debug('iou %s', iou)

    if not (is_giou or is_diou or is_ciou):
        return iou
    else:
        # 最大矩形面积
        min_lt = torch.min(box1_ltrb[:, :2], box2_ltrb[:, :2])
        max_rb = torch.max(box1_ltrb[:, 2:], box2_ltrb[:, 2:])
        max_wh = max_rb - min_lt
        if is_giou:
            max_area = (max_wh[:, 0] * max_wh[:, 1]).clamp(min=torch.finfo(torch.float16).eps)
            giou = iou - (max_area - union_area) / max_area
            return giou

        c2 = max_wh[:, 0] ** 2 + max_wh[:, 1] ** 2 + torch.finfo(torch.float16).eps  # 最大矩形的矩离的平方
        box1_xywh = ltrb2xywh(box1_ltrb)
        box2_xywh = ltrb2xywh(box2_ltrb)
        xw2_xh2 = torch.pow(box1_xywh[:, :2] - box2_xywh[:, :2], 2)  # 中心点距离的平方
        d2 = xw2_xh2[:, 0] + xw2_xh2[:, 1]
        dxy = d2 / c2  # 中心比例距离
        if is_diou:
            diou = iou - dxy
            return diou

        if is_ciou:
            # [3, 1] / [3, 1]  => [3, 1]
            box1_atan_wh = torch.atan(box1_xywh[:, 2:3] / box1_xywh[:, 3:])
            box2_atan_wh = torch.atan(box2_xywh[:, 2:3] / box2_xywh[:, 3:])
            # torch.Size([3, 1])
            v = torch.pow(box1_atan_wh[:, :] - box2_atan_wh, 2) * (4 / math.pi ** 2)
            v = torch.squeeze(v, -1)  # m,n,1 -> m,n 去掉最后一维
            # v.squeeze_(-1)  # m,n,1 -> m,n 去掉最后一维
            with torch.no_grad():
                alpha = v / (1 - iou + v)
            ciou = iou - (dxy + v * alpha)
            return ciou


def bbox_iou4fcos(poff_ltrb_exp, goff_ltrb):
    w_gt = goff_ltrb[:, :, 0] + goff_ltrb[:, :, 2]
    h_gt = goff_ltrb[:, :, 1] + goff_ltrb[:, :, 3]
    w_pred = poff_ltrb_exp[:, :, 0] + poff_ltrb_exp[:, :, 2]
    h_pred = poff_ltrb_exp[:, :, 1] + poff_ltrb_exp[:, :, 3]
    S_gt = w_gt * h_gt
    S_pred = w_pred * h_pred
    I_h = torch.min(goff_ltrb[:, :, 1], poff_ltrb_exp[:, :, 1]) + torch.min(goff_ltrb[:, :, 3], poff_ltrb_exp[:, :, 3])
    I_w = torch.min(goff_ltrb[:, :, 0], poff_ltrb_exp[:, :, 0]) + torch.min(goff_ltrb[:, :, 2], poff_ltrb_exp[:, :, 2])
    S_I = I_h * I_w
    U = S_gt + S_pred - S_I + 1e-20
    IoU = S_I / U

    return IoU


def bbox_iou4y(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.t()
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_area(boxes):
    return torch.abs(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou4ts(box1, box2, is_giou=False, is_diou=False, is_ciou=False):
    '''

    :param box1:torch.Size([m, 4]) ltrb
    :param box2:torch.Size([n, 4]) ltrb
    :param is_giou: 重合度++
    :param is_diou: +中心点
    :param is_ciou: +宽高
    :return:(m,n)
    '''
    # 交集面积
    max_lt = torch.max(box1[:, None, :2], box2[:, :2])  # left-top [N,M,2] 多维组合用法
    min_rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # right-bottom [N,M,2]
    inter_wh = (min_rb - max_lt).clamp(min=0)  # [N,M,2]
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N,M] 降维

    # 并的面积
    area1 = box_area(box1)  # 降维 n
    area2 = box_area(box2)  # 降维 m
    union_area = area1[:, None] + area2 - inter_area + torch.finfo(torch.float16).eps  # 升维n m

    iou = inter_area / union_area

    if not (is_giou or is_diou or is_ciou):
        return iou
    else:
        # 最大矩形面积
        min_lt = torch.min(box1[:, None, :2], box2[:, :2])
        max_rb = torch.max(box1[:, None, 2:], box2[:, 2:])
        max_wh = max_rb - min_lt
        if is_giou:
            max_area = max_wh[:, :, 0] * max_wh[:, :, 1] + torch.finfo(torch.float16).eps  # 降维运算
            giou = iou - (max_area - union_area) / max_area
            return giou

        c2 = max_wh[:, :, 0] ** 2 + max_wh[:, :, 1] ** 2 + torch.finfo(torch.float16).eps  # 最大矩形的矩离的平方
        box1_xywh = ltrb2xywh(box1)
        box2_xywh = ltrb2xywh(box2)
        xw2_xh2 = torch.pow(box1_xywh[:, None, :2] - box2_xywh[:, :2], 2)  # 中心点距离的平方
        d2 = xw2_xh2[:, :, 0] + xw2_xh2[:, :, 1]
        dxy = d2 / c2  # 中心比例距离
        if is_diou:
            diou = iou - dxy
            return diou

        if is_ciou:
            box1_atan_wh = torch.atan(box1_xywh[:, 2:3] / box1_xywh[:, 3:])  # w/h
            box2_atan_wh = torch.atan(box2_xywh[:, 2:3] / box2_xywh[:, 3:])
            # torch.squeeze(ts)
            v = torch.pow(box1_atan_wh[:, None, :] - box2_atan_wh, 2) * (4 / math.pi ** 2)
            # v = torch.squeeze(v, -1)  # m,n,1 -> m,n 去掉最后一维
            v.squeeze_(-1)  # m,n,1 -> m,n 去掉最后一维
            with torch.no_grad():
                alpha = v / (1 - iou + v)
            ciou = iou - (dxy + v * alpha)
            return ciou


def resize_boxes4np(boxes, original_size, new_size):
    '''
    用于预处理 和 最后的测试(预测还原) 直接拉伸调整
    :param boxes: 输入多个 np shape(n,4)
    :param original_size: np (w,h)
    :param new_size: np (w,h)
    :return:
    '''
    # 输出数组   新尺寸 /旧尺寸 = 对应 h w 的缩放因子
    ratios = new_size / original_size
    boxes = boxes * np.concatenate([ratios, ratios]).astype(np.float32)
    return boxes


def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): 最大尺寸 Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


if __name__ == '__main__':
    b1_ltrb = torch.tensor([[1, 2, 3, 4]])
    # b1_ltrb.unsqueeze_(0).unsqueeze_(0)

    print(ltrb2xywh(b1_ltrb))  # tensor([[2, 3, 2, 2]])
    print(xywh2ltrb(torch.tensor([[2, 3, 2, 2]])))

    print(ltrb2ltwh(b1_ltrb))  # tensor([[1, 2, 2, 2]])
    print(ltwh2ltrb(torch.tensor([[1, 2, 2, 2]])))

    print(xywh2ltwh(b1_ltrb))  # tensor([[[-0.5000,  0.0000,  3.0000,  4.0000]]])
    print(ltwh2xywh(torch.tensor([[[-0.5000, 0.0000, 3.0000, 4.0000]]])))
