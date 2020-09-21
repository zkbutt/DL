import torch
import numpy as np


def xywh2ltrb(boxes):
    # xywh -> ltrb
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_size(boxes):
    # 转换成中心宽高的形式
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    '''
    计算所有真实框和先验框的交面积
    :param box_a:真实框
    :param box_b:
    :return:
    '''
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    '''
    通过左上右下计算 IoU
    :param box_a:  真实框
    :param box_b: 先验框
    :return: (a个 b个)二维表
    '''
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def match(threshold, gt, anc, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    '''
    计算 标签 和 anc的差距

    ------------用anc与gt初选建议框,并标识-------------
    gt对应的最大的保留
    iou小的分数置0

    :param threshold: iou阀值 小于不要 设为0
    :param gt: 标签 GT
    :param anc: default boxes 左上右下
    :param variances: [0.1, 0.2] 写死的
    :param labels:  真实标签
    :param landms: 真实5点定位
    :param loc_t: 已定义的返回值
    :param conf_t: 已定义的返回值
    :param landm_t: 已定义的返回值
    :param idx: 一批图片的 index
    :return:
    '''

    # 计算交并比 m,n  m是gt框   n是anc框
    val = jaccard(gt, xywh2ltrb(anc))
    # 每个gt匹配到的最大IoU def gt个  keepdim=True和squeeze_(1) 可以去掉 结果一致
    gt_anc_val, gt_anc_ind = val.max(1, keepdim=True)
    gt_anc_ind.squeeze_(1)
    gt_anc_val.squeeze_(1)
    # 每个def匹配到的最大IoU gt 有可能多个anc对一个gt
    anc_gt_val, anc_gt_ind = val.max(0, keepdim=True)
    anc_gt_ind.squeeze_(0)
    anc_gt_val.squeeze_(0)

    # 修改gt个 anc 的值 确保对应的anc保留  anc可以是[1,gt个]之间
    anc_gt_val.index_fill_(0, gt_anc_ind, 2)
    # 确保保留的 anc 的 gt  index 不会出错
    for j in range(gt_anc_ind.size(0)):
        anc_gt_ind[gt_anc_ind[j]] = j

    # [anc个,4] 取出每一个anchor对应的 gt
    matches_gt = gt[anc_gt_ind]
    # [anc个] 取出每一个anchor对应的label
    conf = labels[anc_gt_ind]
    # [anc个,10] 取出每一个anchor对应的landms
    matches_landm = landms[anc_gt_ind]

    conf[anc_gt_val < threshold] = 0  # 未选中的全部置0为负样本

    loc = encode(matches_gt, anc, variances)
    landm = encode_landm(matches_landm, anc, variances)

    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm


def encode(matched, anc, variances):
    '''
    计算loss
    :param matched: 建议框 xywh
    :param anc: 所有的
    :param variances:
    :return:
    '''
    # 进行编码的操作
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - anc[:, :2]
    # 中心编码
    g_cxcy /= (variances[0] * anc[:, 2:])

    # 宽高编码
    g_wh = (matched[:, 2:] - matched[:, :2]) / anc[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_landm(matched, priors, variances):
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def decode(loc, priors, variances):
    '''
    调整并
    :param loc:  回归系数
    :param priors: anchors 调整比例 x y w h
    :param variances:
    :return:
    '''
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # 左上右下
    boxes[:, :2] -= boxes[:, 2:] / 2  # 中点移左上
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    # 关键点解码
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.3):
    '''
    左上右下
    :param boxes: # 已连接的 <class 'tuple'>: (37840, 15) 4+1+10
    :param conf_thres:
    :param nms_thres:
    :return:
    '''
    detection = boxes
    # 1 . 选出得分大于 0.5 的行
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]
    if not np.shape(detection)[0]:
        return []

    best_box = []
    scores = detection[:, 4]
    # 2 . 根据得分对框进行从大到小排序。
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]  # 行重组

    while np.shape(detection)[0] > 0:
        # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = jaccard(best_box[-1], detection[1:])  # 最新的一个
        detection = detection[1:][ious < nms_thres]  # 面积大于阀值说明是同类不要

    return np.array(best_box)

# def iou(b1, b2):
#     '''
#     计算预测框 和 anchors的iou
#     :param b1: 预测框 3个特图 每一个
#     :param b2: 多个目标
#     :return:
#     '''
#     # 只对框进行
#     b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
#     b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
#
#     inter_rect_x1 = np.maximum(b1_x1, b2_x1)
#     inter_rect_y1 = np.maximum(b1_y1, b2_y1)
#     inter_rect_x2 = np.minimum(b1_x2, b2_x2)
#     inter_rect_y2 = np.minimum(b1_y2, b2_y2)
#
#     # 相交面积
#     inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)
#
#     area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#     area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
#
#     iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
#     return iou
