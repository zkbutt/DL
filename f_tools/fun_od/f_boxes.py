import math
import torch
import numpy as np

from f_tools.GLOBAL_LOG import flog

from f_tools.pic.f_show import f_show_iou4plt, show_anc4pil, f_show_3box4pil


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


def ltrb2xywh(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))

    if dim == 3:  # 可优化
        bboxs[:, :, 2:] = bboxs[:, :, 2:] - bboxs[:, :, :2]
        bboxs[:, :, :2] = bboxs[:, :, :2] + 0.5 * bboxs[:, :, 2:]
    elif dim == 2:
        bboxs[:, 2:] = bboxs[:, 2:] - bboxs[:, :2]
        bboxs[:, :2] = bboxs[:, :2] + 0.5 * bboxs[:, 2:]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def ltrb2ltwh(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))

    if dim == 3:
        bboxs[:, :, 2:] = bboxs[:, :, 2:] - bboxs[:, :, :2]
    elif dim == 2:
        bboxs[:, 2:] = bboxs[:, 2:] - bboxs[:, :2]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def ltwh2ltrb(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))

    if dim == 3:
        bboxs[:, :, 2:] = bboxs[:, :, 2:] + bboxs[:, :, :2]
    elif dim == 2:
        bboxs[:, 2:] = bboxs[:, 2:] + bboxs[:, :2]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def xywh2ltrb(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))
    if dim == 2:
        bboxs[:, :2] -= bboxs[:, 2:] / 2  # 中点移左上
        bboxs[:, 2:] += bboxs[:, :2]
    elif dim == 3:
        bboxs[:, :, :2] -= bboxs[:, :, 2:] / 2  # 中点移左上
        bboxs[:, :, 2:] += bboxs[:, :, :2]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def xywh2ltwh(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))
    if dim == 2:
        bboxs[:, :2] = bboxs[:, :2] - bboxs[:, 2:] / 2.
    elif dim == 3:
        bboxs[:, :, :2] = bboxs[:, :, :2] - bboxs[:, :, 2:] / 2.
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


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
    :param p_loc: 修正系数 (nn,4)  torch.Size([5, 16800, 4])
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
        _a = anc[:, :2] + p_loc[:, :2] * variances[0] * anc[:, 2:]
        # 宽高缩放
        _b = anc[:, 2:] * torch.exp(p_loc[:, 2:] * variances[1])
        _t = torch.cat([_a, _b], dim=1)
    elif len(anc.shape) == 3:
        # 坐标移动
        _a = anc[:, :, :2] + p_loc[:, :, :2] * variances[0] * anc[:, :, 2:]
        # 宽高缩放
        _b = anc[:, :, 2:] * torch.exp(p_loc[:, :, 2:] * variances[1])
        _t = torch.cat([_a, _b], dim=2)
    else:
        raise Exception('维度错误', anc.shape)
    return _t


def fix_boxes4yolo3(boxes, anc):
    '''

    :param boxes: torch.Size([10647, 4])
    :param anc: torch.Size([10647, 4])
    :return:
    '''
    boxes
    return boxes


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


# @torch.no_grad()
# def pos_match(ancs, bboxs, criteria):
#     '''
#     正样本选取策略
#         1. 每一个bboxs框 尽量有一个anc与之对应
#         2. 每一个anc iou大于大于阀值的保留
#     :param ancs:  xywh
#     :param bboxs: 标签框 (xx,4)
#     :param criteria: 小于等于0.35的反例
#     :return:
#         label_neg_mask: 返回反例的布尔索引
#         anc_bbox_ind : 通过 g_bbox[anc_bbox_ind]  可选出标签 与anc对应 其它为0
#     '''
#     # (bboxs个,anc个)
#     # print(bboxs.shape[0])
#     iou = calc_iou4ts(bboxs, xywh2ltrb(ancs))
#     # (1,anc个值)降维  每一个anc对应的bboxs  最大的iou和index
#     anc_bbox_iou, anc_bbox_ind = iou.max(dim=0)  # 存的是 bboxs的index
#
#     # (bbox个,1)降维  每一个bbox对应的anc  最大的iou和index
#     bbox_anc_iou, bbox_anc_ind = iou.max(dim=1)  # 存的是 anc的index
#
#     # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
#     anc_bbox_iou.index_fill_(0, bbox_anc_ind, 2)  # dim index val
#
#     # 确保保留的 anc 的 gt  index 不会出错 exp:一个anc对应多个bboxs, bbox只有这一个anc时
#     _ids = torch.arange(0, bbox_anc_ind.size(0), dtype=torch.int64).to(anc_bbox_ind)
#     anc_bbox_ind[bbox_anc_ind[_ids]] = _ids
#     # for j in range(bbox_anc_ind.size(0)): # 与上句等价
#     #     anc_bbox_ind[bbox_anc_ind[j]] = j
#
#     # ----------正例的index 和 正例对应的bbox索引----------
#     # masks = anc_bbox_iou > criteria  # anc 的正例index
#     # masks = anc_bbox_ind[masks]  # anc对应最大bboxs的索引 进行iou过滤筛选 形成正例对应的bbox的索引
#
#     # 这里改成反倒置0
#     label_neg_mask = anc_bbox_iou <= criteria  # anc 的正例index
#
#     return label_neg_mask, anc_bbox_ind

# def pos_match4yolo(ancs, bboxs, criteria):
#     '''
#     正例与GT最大的 负例<0.5 >0.5忽略
#     :param ancs: xywh
#     :param bboxs: ltrb
#     :param criteria: 0.5
#     :return:
#
#     '''
#     threshold = 99
#     # (bboxs个,anc个)
#     gn = bboxs.shape[0]
#     num_anc = ancs.shape[0]
#     iou = calc_iou4ts(bboxs, xywh2ltrb(ancs), is_ciou=True)
#
#     maxs_iou_index = torch.argsort(-iou, dim=1)
#     anc_bbox_ind = torch.argsort(-iou, dim=0)
#     pos_ancs_index = []  # [n]
#     pos_bboxs_index = []  # [n]
#     for i in range(gn):
#         for j in range(num_anc):
#             # 已匹配好anc选第二大的
#             if maxs_iou_index[i][j] not in pos_ancs_index:
#                 iou[i, j] = threshold  # 强制最大 大于阀值即可
#                 pos_ancs_index.append(maxs_iou_index[i][j])
#                 pos_bboxs_index.append(anc_bbox_ind[i][j])
#                 break
#
#     # 反倒 torch.Size([1, 10647])
#     mask_neg = iou <= criteria  # anc 的正例index
#     # torch.Size([1, 10647])
#     mask_ignore = (iou >= criteria) == (iou < threshold)
#
#     return pos_ancs_index, pos_bboxs_index, mask_neg, mask_ignore

# def _ff_pos_match4yolo(ancs, bboxs, criteria):
#     '''
#     3个anc 一个GT
#     :param ancs:  ltrb
#     :param bboxs: 标签框 (xx,4)
#     :param criteria: 小于等于0.35的反例
#     :return:
#         返回 1~3个anc的索引 0,1,2
#     '''
#     # (bboxs个,anc个)
#     iou = calc_iou4ts(bboxs, ancs)
#
#     '''实现强制保留一个'''
#     # (1,anc个值)  每一个anc对应的bboxs  最大的iou和index
#     iou_max_gt_anc, anc_ind = iou.max(dim=0)  # 存的是 bboxs的index
#     # (bbox个,1) 每一个bbox对应的anc  最大的iou和index
#     iou_max_anc_gt, gt_ind = iou.max(dim=1)  # 存的是 anc的index
#     # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
#     iou_max_gt_anc.index_fill_(0, gt_ind, 2)  # dim index val
#
#     # gt_ind.size(0) = GT的个数  确保保留的 anc 的 gt  index 不会出错 exp:一个anc对应多个bboxs, bbox只有这一个anc时
#     _ids = torch.arange(0, gt_ind.size(0), dtype=torch.int64).to(anc_ind)
#     _gt = gt_ind[_ids]
#     anc_ind[_gt] = _ids
#
#     mask_pos = iou_max_gt_anc.view(-1) > criteria  # anc 的正例index
#
#     return mask_pos

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
