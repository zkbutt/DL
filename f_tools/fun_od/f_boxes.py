import math
import torch
import numpy as np

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_factory import VOCDataSet
from f_tools.pic.f_show import f_show_iou4plt, show_anc4pil, f_show_iou4pil


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
            box1_atan_wh = torch.atan(box1_xywh[:, 2:3] / box1_xywh[:, 3:])
            box2_atan_wh = torch.atan(box2_xywh[:, 2:3] / box2_xywh[:, 3:])
            # torch.squeeze(ts)
            v = torch.pow(box1_atan_wh[:, None, :] - box2_atan_wh, 2) * (4 / math.pi ** 2)
            v = torch.squeeze(v, -1)  # m,n,1 -> m,n 去掉最后一维
            v.squeeze_(-1)  # m,n,1 -> m,n 去掉最后一维
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


def nms(boxes, scores, iou_threshold):
    ''' IOU大于0.5的抑制掉
         boxes (Tensor[N, 4])) – bounding boxes坐标. 格式：(x1, y1, x2, y2)
         scores (Tensor[N]) – bounding boxes得分
         iou_threshold (float) – IoU过滤阈值
     返回:NMS过滤后的bouding boxes索引（降序排列）
     '''
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    '''

    :param boxes: 拉平所有类别的box重复的 n*20类,4
    :param scores: torch.Size([16766])
    :param idxs:  真实类别index 通过手动创建匹配的 用于表示当前 nms的类别 用于统一偏移 技巧
    :param iou_threshold:float 0.5
    :return:
    '''
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
    :param ancs:  xywh
    :param bboxs: 标签框 (xx,4)
    :param criteria: 小于等于0.35的反例
    :return:
        label_neg_mask: 返回反例的布尔索引
        anc_bbox_ind : 通过 g_bbox[anc_bbox_ind]  可选出标签 与anc对应 其它为0
    '''
    # (bboxs个,anc个)
    iou = calc_iou4ts(bboxs, xywh2ltrb(ancs))
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


@torch.no_grad()
def pos_match4yolo(ancs, bboxs, criteria):
    '''
    正例与GT最大的 负例<0.5 >0.5忽略
    :param ancs: xywh
    :param bboxs: ltrb
    :param criteria: 0.5
    :return:

    '''
    threshold = 99
    # (bboxs个,anc个)
    gn = bboxs.shape[0]
    num_anc = ancs.shape[0]
    iou = calc_iou4ts(bboxs, xywh2ltrb(ancs), is_ciou=True)

    maxs_iou_index = torch.argsort(-iou, dim=1)
    anc_bbox_ind = torch.argsort(-iou, dim=0)
    pos_ancs_index = []  # [n]
    pos_bboxs_index = []  # [n]
    for i in range(gn):
        for j in range(num_anc):
            # 已匹配好anc选第二大的
            if maxs_iou_index[i][j] not in pos_ancs_index:
                iou[i, j] = threshold  # 强制最大 大于阀值即可
                pos_ancs_index.append(maxs_iou_index[i][j])
                pos_bboxs_index.append(anc_bbox_ind[i][j])
                break

    # 反倒 torch.Size([1, 10647])
    mask_neg = iou <= criteria  # anc 的正例index
    # torch.Size([1, 10647])
    mask_ignore = (iou >= criteria) == (iou < threshold)

    return pos_ancs_index, pos_bboxs_index, mask_neg, mask_ignore


def _ff_pos_match4yolo(ancs, bboxs, criteria):
    '''
    3个anc 一个GT
    :param ancs:  ltrb
    :param bboxs: 标签框 (xx,4)
    :param criteria: 小于等于0.35的反例
    :return:
        返回 1~3个anc的索引 0,1,2
    '''
    # (bboxs个,anc个)
    iou = calc_iou4ts(bboxs, ancs)

    '''实现强制保留一个'''
    # (1,anc个值)  每一个anc对应的bboxs  最大的iou和index
    iou_max_gt_anc, anc_ind = iou.max(dim=0)  # 存的是 bboxs的index
    # (bbox个,1) 每一个bbox对应的anc  最大的iou和index
    iou_max_anc_gt, gt_ind = iou.max(dim=1)  # 存的是 anc的index
    # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
    iou_max_gt_anc.index_fill_(0, gt_ind, 2)  # dim index val

    # gt_ind.size(0) = GT的个数  确保保留的 anc 的 gt  index 不会出错 exp:一个anc对应多个bboxs, bbox只有这一个anc时
    _ids = torch.arange(0, gt_ind.size(0), dtype=torch.int64).to(anc_ind)
    _gt = gt_ind[_ids]
    anc_ind[_gt] = _ids

    mask_pos = iou_max_gt_anc.view(-1) > criteria  # anc 的正例index

    return mask_pos


def resize_boxes4np(boxes, original_size, new_size):
    '''
    用于预处理 和 最后的测试(预测还原)
    :param boxes: 输入多个 np shape(n,4)
    :param original_size: np (w,h)
    :param new_size: np (w,h)
    :return:
    '''
    # 输出数组   新尺寸 /旧尺寸 = 对应 h w 的缩放因子
    ratios = new_size / original_size
    boxes = boxes * np.concatenate([ratios, ratios])
    return boxes


def xy2offxy(p1, p2):
    '''

    :param p1:[n,2]
    :param p2: [m,2]  [[52, 52], [26, 26], [13, 13]]
    :return:
        返回offset_xy_wh
        返回 colrow
    '''
    colrow_index = (p1 * p2).type(torch.int16)  # 向下取整
    offset_xy = colrow_index / p2  # 格子偏移对特图的
    offset_xy_wh = (p1 - offset_xy) * p2  # 特图坐标 - 格子偏移 * 对应格子数
    return offset_xy_wh, colrow_index


def match4yolo1(boxes, labels, num_anc=1, num_bbox=2, num_class=20, grid=7, device=None):
    '''
    将ltrb 转换为 7*7*(2*(4+1)+20) = grid*grid*(num_bbox(4+1)+num_class)
    这个必须和anc的顺序一致
    :param boxes:已原图归一化的bbox ltrb
    :param labels: 1~n值
    :param num_bbox:
    :param num_class:
    :param grid:
    :return:
        torch.Size([7, 7, 25])  c r

    '''
    target = torch.zeros((grid, grid, 5 * num_bbox + num_class))
    if device is not None:
        target = target.to(device)
    # ltrb -> xywh
    boxes_xywh = ltrb2xywh(boxes)
    wh = boxes_xywh[:, 2:]
    cxcy = boxes_xywh[:, :2]

    for i in range(cxcy.size()[0]):  # 遍历每一个框 cxcy.size()[0]  与shape等价
        '''计算GT落在7x7哪个网格中 同时ij也表示网格的左上角的坐标'''
        # cxcy * 7  这里是 col row的索引, int(0.1,0.5)*3(格子) =(0,1)
        colrow_index = (cxcy[i] * grid).type(torch.int16)
        # 计算格子与大图的偏移(0,1) /3 =(0,0.3333)
        offset_xy = torch.true_divide(colrow_index, grid)
        # 大图xy-格子xy /格子 = 该格子的相对距离
        grid_xy = (cxcy[i] - offset_xy) / grid  # GT相对于所在网格左上角的值, 网格的左上角看做原点 需进行放大

        # 这里如果有两个GT在一个格子里将丢失
        for j in range(num_bbox):
            target[colrow_index[1], colrow_index[0], (j + 1) * 5 - 1] = 1  # conf
            start = j * 5
            target[colrow_index[1], colrow_index[0], start:start + 2] = grid_xy  # 相对于所在网格左上角
            target[colrow_index[1], colrow_index[0], start + 2:start + 4] = wh[i]  # wh值相对于整幅图像的尺寸
        target[colrow_index[1], colrow_index[0], labels[i] + 5 * num_bbox - 1] = 1
    target = target.repeat(1, 1, num_anc)
    return target


def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    # 这里的image是一个数字，代表是当前batch的第几个图片
    # x,y,w,h都进行了归一化，除以了宽或者高

    nt = len(targets)

    tcls, tbox, indices, av = [], [], [], []

    multi_gpu = type(model) in (nn.parallel.DataParallel,
                                nn.parallel.DistributedDataParallel)

    reject, use_all_anchors = True, True
    for i in model.yolo_layers:
        # yolov3.cfg中有三个yolo层，这部分用于获取对应yolo层的grid尺寸和anchor大小
        # ng 代表num of grid (13,13) anchor_vec [[x,y],[x,y]]
        # 注意这里的anchor_vec: 假如现在是yolo第一个层(downsample rate=32)
        # 这一层对应anchor为：[116, 90], [156, 198], [373, 326]
        # anchor_vec实际值为以上除以32的结果：[3.6,2.8],[4.875,6.18],[11.6,10.1]
        # 原图 416x416 对应的anchor为 [116, 90]
        # 下采样32倍后 13x13 对应的anchor为 [3.6,2.8]
        if multi_gpu:
            ng = model.module.module_list[i].ng
            anchor_vec = model.module.module_list[i].anchor_vec
        else:
            ng = model.module_list[i].ng,
            anchor_vec = model.module_list[i].anchor_vec

        # iou of targets-anchors
        # targets中保存的是ground truth
        t, a = targets, []

        gwh = t[:, 4:6] * ng[0]

        if nt:  # 如果存在目标
            # anchor_vec: shape = [3, 2] 代表3个anchor
            # gwh: shape = [2, 2] 代表 2个ground truth
            # iou: shape = [3, 2] 代表 3个anchor与对应的两个ground truth的iou
            iou = wh_iou(anchor_vec, gwh)  # 计算先验框和GT的iou

            if use_all_anchors:
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na).view(
                    (-1, 1)).repeat([1, nt]).view(-1)  # 构造 3x2 -> view到6
                # a = [0,0,1,1,2,2]
                t = targets.repeat([na, 1])
                # targets: [image, cls, x, y, w, h]
                # 复制3个: shape[2,6] to shape[6,6]
                gwh = gwh.repeat([na, 1])
                # gwh shape:[6,2]
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor
                # 取iou最大值是darknet的默认做法，返回的a是下角标

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                # 在这里将所有阈值小于ignore thresh的去掉
                j = iou.view(-1) > model.hyp['iou_t']
                # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        # 取的是targets[image, class, x,y,w,h]中 [image, class]

        gxy = t[:, 2:4] * ng[0]  # grid x, y

        gi, gj = gxy.long().t()  # grid x, y indices
        # 注意这里通过long将其转化为整形，代表格子的左上角

        indices.append((b, a, gj, gi))
        # indice结构体保存内容为：
        '''
        b: 一个batch中的角标
        a: 代表所选中的正样本的anchor的下角标
        gj, gi: 代表所选中的grid的左上角坐标
        '''
        # Box
        gxy -= gxy.floor()  # xy
        # 现在gxy保存的是偏移量，是需要YOLO进行拟合的对象
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        # 保存对应偏移量和宽高（对应13x13大小的）
        av.append(anchor_vec[a])  # anchor vec
        # av 是anchor vec的缩写，保存的是匹配上的anchor的列表

        # Class
        tcls.append(c)
        # tcls用于保存匹配上的类别列表
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())
    return tcls, tbox, indices, av


if __name__ == '__main__':
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
