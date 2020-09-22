import torch
import numpy as np


def tlbr2yxhw(bboxs):
    '''
    上左下右 -> 中心高宽 输出 (n,4)
    :param bboxs:  t,l,b,r
    :return:y,x,h,w
    '''
    bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]  # h
    bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]  # w
    bboxs[:, 0] = bboxs[:, 0] + 0.5 * bboxs[:, 2]
    bboxs[:, 1] = bboxs[:, 1] + 0.5 * bboxs[:, 3]
    return bboxs


def tlbr2tlhw(bboxs):
    '''
    上左下右 -> 上左高宽 输出 (n,4)
    :param bboxs:  t,l,b,r
    :return:t,l,h,w
    '''
    bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]  # h
    bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]  # w
    return bboxs


def yxhw2ltrb(bboxs):
    '''
    中心高宽 -> 上左下右
    :bboxs y,x,h,w
    :return: t,l,b,r
    '''
    t = bboxs[:0] - 0.5 * bboxs[:, 2]
    l = bboxs[:1] - 0.5 * bboxs[:, 3]
    b = bboxs[:0] + 0.5 * bboxs[:, 2]
    r = bboxs[:1] + 0.5 * bboxs[:, 3]
    bboxs[:0] = t
    bboxs[:1] = l
    bboxs[:2] = b
    bboxs[:3] = r
    return bboxs


def fix_yxhw(bboxs, loc, variances=(0.1, 0.2)):
    '''
    用于 retinaface
    :param bboxs: y,x,h,w
    :param loc: 修正系数
    :return: 中心点和 宽高的调整系数
    '''
    # 坐标移动
    bboxs[:, :2] = bboxs[:, :2] + loc[:, :2] * variances[0] * bboxs[:, :2]
    # 宽高缩放
    bboxs[:, 2:] = bboxs[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    return bboxs


def loc_bbox4np(src_bbox, loc):
    ''' 只对 xywh 进行运算
     已知源bbox 和位置偏差dx，dy，dh，dw，求目标框G
    :param src_bbox: (n,4) 表示多个选区 , 左上右下角坐标
    :param loc: 回归修正参数 (n,4)  中心点 (dx,dy,dh,dw)
    :return: 修正后选区
    '''
    # 左上右下 ---> 中心点 (dx,dy,dh,dw)
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    _ = loc[:, 0]  # 这种取出来会降低维度
    _ = loc[:, :1]  # 与这个等效
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # 根据公式 进行选框回归 Gy Gx Gh Gw
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    # 开内存  中心点 (dx,dy,dh,dw) ---> 左上右下
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w
    return dst_bbox


def bbox_loc4np(src_bbox, dst_bbox):
    ''' 只对 xywh 进行运算
    已知源框和目标框求出其位置偏差
    :param src_bbox:  正选出的正样本
    :param dst_bbox: 左上右下
    :return: <class 'tuple'>: (128, 4)
    '''
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    # 左上右下转中心 hw
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 除法不能为0 log不能为负 取此类型的最小的一个数
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    # 按公式算偏差 , loc2bbox 这个是算偏差
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    '''
    求所有bboxs 与所有标定框 的交并比 第二维（ymin，xmin，ymax，xmax）
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
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
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
