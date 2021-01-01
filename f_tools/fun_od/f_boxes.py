import math
import torch
import numpy as np



def empty_bboxes(bboxs):
    if isinstance(bboxs, np.ndarray):
        bboxs_ = np.empty_like(bboxs)
    elif isinstance(bboxs, torch.Tensor):
        device = bboxs.device
        bboxs_ = torch.empty_like(bboxs, device=device)
    else:
        raise Exception('类型错误', type(bboxs))
    return bboxs_


def offxy2xy(xy, colrow_index, z_grids):
    '''

    :param xy:
    :param colrow_index:
    :param z_grids: [7,7]
    :return:
    '''
    # offxy = torch.true_divide(colrow_index, z_grids)
    # ret = torch.true_divide(xy, z_grids) + offxy
    ret2 = torch.true_divide(xy + colrow_index, z_grids)
    return ret2


def ltrb2xywh(bboxs):
    dim = len(bboxs.shape)
    bboxs_ = empty_bboxes(bboxs)
    if dim == 3:  # 可优化
        bboxs_[:, :, 2:] = bboxs[:, :, 2:] - bboxs[:, :, :2]
        bboxs_[:, :, :2] = bboxs[:, :, :2] + 0.5 * bboxs[:, :, 2:]
    elif dim == 2:
        bboxs_[:, 2:] = bboxs[:, 2:] - bboxs[:, :2]  # wh = rb -lt
        bboxs_[:, :2] = bboxs[:, :2] + 0.5 * bboxs_[:, 2:]  # xy = lt + 0.5*wh
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs_


def ltrb2ltwh(bboxs):
    dim = len(bboxs.shape)
    bboxs_ = empty_bboxes(bboxs)

    if dim == 3:
        bboxs_[:, :, 2:] = bboxs[:, :, 2:] - bboxs[:, :, :2]  # wh = rb - lt
    elif dim == 2:
        bboxs_[:, 2:] = bboxs[:, 2:] - bboxs[:, :2]
    else:
        raise Exception('维度错误', bboxs_.shape)
    return bboxs_


def ltwh2ltrb(bboxs):
    dim = len(bboxs.shape)
    bboxs_ = empty_bboxes(bboxs)

    if dim == 3:
        bboxs_[:, :, :2] = bboxs[:, :, :2]
        bboxs_[:, :, 2:] = bboxs[:, :, 2:] + bboxs[:, :, :2]  # rb = wh + lt
    elif dim == 2:
        bboxs_[:, :2] = bboxs[:, :2]
        bboxs_[:, 2:] = bboxs[:, 2:] + bboxs[:, :2]
    else:
        raise Exception('维度错误', bboxs_.shape)
    return bboxs_


def xywh2ltrb(bboxs):
    dim = len(bboxs.shape)
    bboxs_ = empty_bboxes(bboxs)

    if isinstance(bboxs_, np.ndarray):
        fdiv = np.true_divide
        v = 2
    elif isinstance(bboxs_, torch.Tensor):
        fdiv = torch.true_divide
        v = torch.tensor(2, device=bboxs.device)
    else:
        raise Exception('类型错误', type(bboxs))

    if dim == 2:
        bboxs_[:, :2] = bboxs[:, :2] - fdiv(bboxs[:, 2:], v)  # lt = xy + wh/2
        bboxs_[:, 2:] = bboxs_[:, :2] + bboxs[:, 2:]  # rb = nlt + wh
    elif dim == 3:
        bboxs_[:, :, :2] = bboxs[:, :, :2] - fdiv(bboxs[:, :, 2:], v)
        bboxs_[:, :, 2:] = bboxs_[:, :, :2] + bboxs[:, :, 2:]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs_


def xywh2ltwh(bboxs):
    dim = len(bboxs.shape)
    bboxs_ = empty_bboxes(bboxs)

    if isinstance(bboxs_, np.ndarray):
        fdiv = np.true_divide
        v = 2
    elif isinstance(bboxs_, torch.Tensor):
        fdiv = torch.true_divide
        v = torch.tensor(2, device=bboxs.device)
    else:
        raise Exception('类型错误', type(bboxs))

    if dim == 2:
        bboxs_[:, :2] = bboxs[:, :2] - fdiv(bboxs[:, 2:], v)  # lt = xy - wh/2
    elif dim == 3:
        bboxs_[:, :, :2] = bboxs[:, :, :2] - fdiv(bboxs[:, :, 2:], v)
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs_


def diff_keypoints(anc, g_keypoints, variances=(0.1, 0.2)):
    '''
    用anc同维 和 已匹配的GT 计算差异
    :param anc:
    :param g_keypoints:
    :param variances:
    :return:
    '''
    if len(anc.shape) == 2:
        ancs_xy = anc[:, :2].repeat(1, 5)
        ancs_wh = anc[:, 2:].repeat(1, 5)
        _t = (g_keypoints - ancs_xy) / variances[0] / ancs_wh
    elif len(anc.shape) == 3:
        ancs_xy = anc[:, :, :2].repeat(1, 1, 5)
        ancs_wh = anc[:, :, 2:].repeat(1, 1, 5)
        _t = (g_keypoints - ancs_xy) / variances[0] / ancs_wh
    else:
        raise Exception('维度错误', anc.shape)
    return _t


def diff_bbox(anc, g_bbox, variances=(0.1, 0.2)):
    '''
    用anc同维 和 已匹配的GT 计算差异
    :param anc: xywh  (nn,4) torch.Size([1, 16800, 4])
    :param g_bbox: gt torch.Size([5, 16800, 4])
    :return: 计算差异值   对应xywh
    '''
    if len(anc.shape) == 2:
        _a = (g_bbox[:, :2] - anc[:, :2]) / variances[0] / anc[:, 2:]
        _b = (g_bbox[:, 2:] / anc[:, 2:]).log() / variances[1]
        _t = torch.cat([_a, _b], dim=1)
    elif len(anc.shape) == 3:
        _a = (g_bbox[:, :, :2] - anc[:, :, :2]) / variances[0] / anc[:, :, 2:]
        _b = (g_bbox[:, :, 2:] / anc[:, :, 2:]).log() / variances[1]
        _t = torch.cat([_a, _b], dim=2)
    else:
        raise Exception('维度错误', anc.shape)
    return _t


def fix_bbox(anc, p_loc, variances=(0.1, 0.2)):
    '''
    用预测的loc 和 anc得到 修复后的框
    :param anc: xywh  (nn,4)
    :param p_loc: 修正系数 (nn,4)
    :return: 修复后的框
    '''
    if len(anc.shape) == 2:
        # 坐标移动
        xy = anc[:, :2] + p_loc[:, :2] * variances[0] * anc[:, 2:]
        # 宽高缩放
        wh = anc[:, 2:] * torch.exp(p_loc[:, 2:] * variances[1])
        xywh = torch.cat([xy, wh], dim=1)
    elif len(anc.shape) == 3:
        # 坐标移动
        xy = anc[:, :, :2] + p_loc[:, :, :2] * variances[0] * anc[:, :, 2:]
        # 宽高缩放
        wh = anc[:, :, 2:] * torch.exp(p_loc[:, :, 2:] * variances[1])
        xywh = torch.cat([xy, wh], dim=2)
    else:
        raise Exception('维度错误', anc.shape)
    return xywh


def fix_boxes4yolo3(p_yolo_boxes, anc, fsize_p_anc):
    '''
    支持三维 anc fsize_p_anc 是二维
    :param p_yolo_boxes: torch.Size([25200, 4])
    :param anc: torch.Size([25200, 4])
    :param fsize_p_anc:  torch.Size([25200, 2])
    :return:
    '''
    dim = len(p_yolo_boxes.shape)
    p_yolo_boxes_s = p_yolo_boxes.sigmoid()

    if dim == 2:
        p_offxy = p_yolo_boxes_s[:, :2] / fsize_p_anc  # xy 需要归一化修复
        # p_box_xy = p_offxy * 2. - 0.5 + anc[:, :2]  # xy修复
        p_wh = p_yolo_boxes_s[:, 2:4]
    elif dim == 3:
        p_offxy = p_yolo_boxes_s[:, :, :2] / fsize_p_anc  # xy 需要归一化修复
        p_wh = p_yolo_boxes_s[:, :, 2:4]

    else:
        raise Exception('维度错误', p_yolo_boxes.shape)
    p_box_xy = p_offxy + anc[:, :2]
    p_box_wh = p_wh.exp() * anc[:, 2:]  # wh修复
    p_box_xywh = torch.cat([p_box_xy, p_box_wh], dim=-1)
    return p_box_xywh


def fix_keypoints(anc, p_keypoints, variances=(0.1, 0.2)):
    '''
    可以不用返回值
    :param anc:
    :param p_keypoints:
    :param variances:
    :return:
    '''
    if len(anc.shape) == 2:
        # [anc个, 4] ->  [anc个, 2] -> [anc个, 2*5]
        ancs_xy = anc[:, :2].repeat(1, 5)
        ancs_wh = anc[:, 2:].repeat(1, 5)
        _t = ancs_xy + p_keypoints * variances[0] * ancs_wh
    elif len(anc.shape) == 3:
        # [anc个, 4] ->  [anc个, 2] -> [anc个, 2*5]
        ancs_xy = anc[:, :, :2].repeat(1, 1, 5)
        ancs_wh = anc[:, :, 2:].repeat(1, 1, 5)
        _t = ancs_xy + p_keypoints * variances[0] * ancs_wh
    else:
        raise Exception('维度错误', anc.shape)
    return _t


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


def bbox_iou4one(box1, box2, is_giou=False, is_diou=False, is_ciou=False):
    '''
    返回一个数
    :param bbox_a: 多个预测框 (n,4)
    :param bbox_b: 多个标定框 (n,4)
    :return: n
    '''
    max_lt = torch.max(box1[:, :2], box2[:, :2])  # left-top [N,M,2] 多维组合用法
    min_rb = torch.min(box1[:, 2:], box2[:, 2:])  # right-bottom [N,M,2]
    inter_wh = (min_rb - max_lt).clamp(min=0)  # [N,M,2]
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]  # [N,M] 降维

    # 并的面积
    area1 = box_area(box1)  # 降维 n
    area2 = box_area(box2)  # 降维 m
    union_area = area1 + area2 - inter_area + torch.finfo(torch.float32).eps  # 升维n m

    iou = inter_area / union_area

    if not (is_giou or is_diou or is_ciou):
        return iou
    else:
        # 最大矩形面积
        min_lt = torch.min(box1[:, :2], box2[:, :2])
        max_rb = torch.max(box1[:, 2:], box2[:, 2:])
        max_wh = max_rb - min_lt
        if is_giou:
            max_area = max_wh[:, 0] * max_wh[:, 1] + torch.finfo(torch.float32).eps  # 降维运算
            giou = iou - (max_area - union_area) / max_area
            return giou

        c2 = max_wh[:, 0] ** 2 + max_wh[:, 1] ** 2 + torch.finfo(torch.float32).eps  # 最大矩形的矩离的平方
        box1_xywh = ltrb2xywh(box1)
        box2_xywh = ltrb2xywh(box2)
        xw2_xh2 = torch.pow(box1_xywh[:, :2] - box2_xywh[:, :2], 2)  # 中心点距离的平方
        d2 = xw2_xh2[:, 0] + xw2_xh2[:, 1]
        dxy = d2 / c2  # 中心比例距离
        if is_diou:
            diou = iou - dxy
            return diou

        if is_ciou:
            box1_atan_wh = torch.atan(box1_xywh[:, 2:3] / box1_xywh[:, 3:])
            box2_atan_wh = torch.atan(box2_xywh[:, 2:3] / box2_xywh[:, 3:])
            # torch.squeeze(ts)
            v = torch.pow(box1_atan_wh[:, :] - box2_atan_wh, 2) * (4 / math.pi ** 2)
            v = torch.squeeze(v, -1)  # m,n,1 -> m,n 去掉最后一维
            v.squeeze_(-1)  # m,n,1 -> m,n 去掉最后一维
            alpha = v / (1 - iou + v)
            ciou = iou - (dxy + v * alpha)
            return ciou


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


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
    union_area = area1[:, None] + area2 - inter_area + torch.finfo(torch.float32).eps  # 升维n m

    iou = inter_area / union_area

    if not (is_giou or is_diou or is_ciou):
        return iou
    else:
        # 最大矩形面积
        min_lt = torch.min(box1[:, None, :2], box2[:, :2])
        max_rb = torch.max(box1[:, None, 2:], box2[:, 2:])
        max_wh = max_rb - min_lt
        if is_giou:
            max_area = max_wh[:, :, 0] * max_wh[:, :, 1] + torch.finfo(torch.float32).eps  # 降维运算
            giou = iou - (max_area - union_area) / max_area
            return giou

        c2 = max_wh[:, :, 0] ** 2 + max_wh[:, :, 1] ** 2 + torch.finfo(torch.float32).eps  # 最大矩形的矩离的平方
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


def calc_iou_wh4ts(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


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


def get_boxes_colrow_index(boxes_xy, fsize):
    '''
    同维 计算box对应层对应anc的 colrow_index
    :param boxes_xy: 归一化 [n,2]
    :param fsize: 格子数 [n,2]  [[52, 52]..., [26, 26]..., [13, 13]...] torch.Size([9, 2])
    :return:
    '''
    return (boxes_xy * fsize).type(torch.int16)  # 向下取整


def xy2offxy(boxes_xy, fsize):
    '''
    计算xy的相对偏移 同维
    :param boxes_xy: 归一化 [n,2]
    :param fsize: 格子数 [n,2]  [[52, 52], [26, 26], [13, 13]] torch.Size([9, 2])
    :return:
        返回offset_xy_wh
        返回 colrow
    '''
    colrow_index = get_boxes_colrow_index(boxes_xy, fsize)
    offset_xy = colrow_index / fsize  # 格子偏移对特图的
    offset_xy_wh = (boxes_xy - offset_xy) * fsize  # 特图坐标 - 格子偏移 * 对应格子数
    return offset_xy_wh


if __name__ == '__main__':
    from f_tools.datas.data_factory import VOCDataSet

    path = r'M:\AI\datas\VOC2012\trainval'

    dataset_train = VOCDataSet(
        path,
        'train.txt',  # 正式训练要改这里
        transforms=None,
        bbox2one=False,
        isdebug=True
    )
    img_pil, target_tensor = dataset_train[0]
    # bbox = target_tensor['boxes'].numpy()
    # labels = target_tensor["labels"].numpy()

    bbox = target_tensor['boxes']
    hw = target_tensor['height_width']
    wh = hw.numpy()[::-1]
    whwh = torch.tensor(wh.copy()).repeat(2)
    bbox /= whwh

    labels = target_tensor["labels"]
    target = boxes2yolo(bbox, labels)
    print(target.shape)
