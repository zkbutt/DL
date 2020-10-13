import torch
import numpy as np

from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_boxes import nms


def xywh2ltrb(boxes):
    # xywh -> ltrb
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def ltrb2xywh(bboxes):
    _l = (bboxes[:, :2] + bboxes[:, 2:] / 2,  # x, y
          bboxes[:, :2] - bboxes[:, 2:])  # w, h
    if isinstance(bboxes, np.ndarray):
        return np.concatenate(_l, axis=1)
    elif isinstance(bboxes, torch.Tensor):
        return torch.cat(_l, 1)  # w, h
    else:
        flog.error('bboxes类型错误 %s', type(bboxes))
        raise Exception('bboxes类型错误', type(bboxes))


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
        torch.Size([a个, b个, 2])
    '''
    a = box_a.shape[0]
    b = box_b.shape[0]
    # x,2 -> x,1,2 -> x b 2  ,选出rb最大的
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(a, b, 2),
        box_b[:, 2:].unsqueeze(0).expand(a, b, 2)
    )
    # 选出lt最小的
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(a, b, 2),
        box_b[:, :2].unsqueeze(0).expand(a, b, 2)
    )
    # 得有有效的长宽 np高级
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    '''
    通过左上右下计算 IoU  全是 ltrb
    :param box_a:  bboxs
    :param box_b: 先验框
    :return: (a.shape[0] b.shape[0])
    '''
    inter = intersect(box_a, box_b)
    # # x,2 -> x,1,2 -> x b 2 直接算面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def match(bboxs, labels, keypoints, anc, idx, loc_t, conf_t, landm_t,
          threshold, variances):
    '''
    计算 标签 和 anc的差距

    ------------用anc与gt初选建议框,并标识-------------
    gt对应的最大的保留
    iou小的分数置0

    :param bboxs: ltrb 标签 GT 真实目标 torch.Size([18, 4])
    :param labels:  真实标签
    :param keypoints: 真实5点定位
    :param anc:  xywh
    :param idx: 一批图片的 index

    :param loc_t: 已定义的返回值
    :param conf_t: 已定义的返回值
    :param landm_t: 已定义的返回值
    :param threshold: iou阀值 小于不要 设为0
    :param variances: [0.1, 0.2] 写死的
    :return:
    '''

    # 计算交并比 m,n  m是bboxs框   n是anc框  最输入 ltrb
    val = jaccard(bboxs, xywh2ltrb(anc))

    # 每个bbox匹配到的最大IoU anc   得gt个  torch.Size([40])
    bbox_anc_iou, bbox_anc_ind = val.max(1)  # 存的是 anc的index

    # 每个anc匹配到的最大IoU bbox 有可能多个anc对一个bbox keepdim squeeze_ 可以去掉
    anc_bbox_iou, anc_bbox_ind = val.max(0, keepdim=True)
    anc_bbox_ind.squeeze_(0)
    anc_bbox_iou.squeeze_(0)  # torch.Size([16800]) 存的是bbox的index

    # 修改 bbox个对应最好的 anc 的值 , 确保对应的anc保留  修改值anc为[1,bbox个]之间
    anc_bbox_iou.index_fill_(0, bbox_anc_ind, 2)  # 修改为2
    # 确保保留的 anc 的 gt  index 不会出错
    for j in range(bbox_anc_ind.size(0)):
        anc_bbox_ind[bbox_anc_ind[j]] = j

    # [anc个,4] 根据 anc个bbox索引 让bbox与anc同维便于后面运算
    matches_bbox = bboxs[anc_bbox_ind]  # 这里有问题? anc_bbox_ind这个还未进行阀值筛选
    # [anc个] 取出每一个anchor对应的label 同上
    conf = labels[anc_bbox_ind]
    # [anc个,10] 取出每一个anchor对应的landms
    matches_landm = keypoints[anc_bbox_ind]

    # 正负样本分类
    conf[anc_bbox_iou < threshold] = 0  # 值小于阀值的类别设置为 0负样本

    # matches_bbox ltrb 计算loss
    loc = encode(matches_bbox, anc, variances)
    landm = encode_landm(matches_landm, anc, variances)

    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm


def encode(matches_bbox, anc, variances):
    '''
    计算loss
    :param matches_bbox: 已anc同维的,已匹配bbox的框 ltrb
    :param anc: xywh
    :param variances:
    :return:
    '''
    # ltrb -> xywh
    xy = (matches_bbox[:, :2] + matches_bbox[:, 2:]) / 2
    wh = (matches_bbox[:, 2:] - matches_bbox[:, :2])

    # 中心值
    gap_xy = (xy - anc[:, :2]) / (variances[0] * anc[:, 2:])

    # 宽高差距
    gap_wh = torch.log(wh / anc[:, 2:]) / variances[1]
    return torch.cat([gap_xy, gap_wh], dim=1)


def encode_landm(matches_landm, anc, variances):
    '''
    
    :param matches_landm: [16800,10] xy,xy,xy,xy,xy
    :param anc: xywh torch.Size([16800, 4])
    :param variances:
    :return:
        gap_cxcy: torch.Size([16800, 10])
    '''
    size = matches_landm.size(0)  # [16800,10] -> [16800,5,2]
    matches_landm = torch.reshape(matches_landm, (size, 5, 2))
    # 分列组装  anc[:, 0] 降维 [size] -> [size,1] -> [size,5] -> [size,5,1]
    priors_cx = anc[:, 0].unsqueeze(1).expand(size, 5).unsqueeze(2)
    priors_cy = anc[:, 1].unsqueeze(1).expand(size, 5).unsqueeze(2)
    priors_w = anc[:, 2].unsqueeze(1).expand(size, 5).unsqueeze(2)
    priors_h = anc[:, 3].unsqueeze(1).expand(size, 5).unsqueeze(2)
    #  [size,5,1] 4个 ->  [size,5,4]
    anc = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    gap_cxcy = (matches_landm[:, :, :2] - anc[:, :, :2]) / (variances[0] * anc[:, :, 2:])
    gap_cxcy = gap_cxcy.reshape(gap_cxcy.size(0), -1)  # 拉平
    return gap_cxcy


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def decode(loc, priors, variances=(0.1, 0.2)):
    '''
    anc的使用loc进行调整,调整预测框并转换为ltrb 输入支持(xxx,4) xywh
    :param loc:  回归系数 torch.Size([37840, 4])
    :param priors: 这个就是 anchors 调整比例 x y w h
    :param variances: 限制调整在网格内
    :return:
    '''
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # xywh -> ltrb
    boxes[:, :2] -= boxes[:, 2:] / 2  # 中点移左上
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances=(0.1, 0.2)):
    # 关键点解码
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def non_max_suppression(detection, conf_thres=0.5, nms_thres=0.3):
    '''
    输入左上右下
    :param boxes: # 已连接的 <class 'tuple'>: (37840, 15) 4+1+10
    :param conf_thres: 类别得分阀值
    :param nms_thres: nms分数
    :return:
    '''
    # 1 . 选出得分大于 0.5 的行
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]
    if detection.shape[0] == 0:
        flog.warning('第一级 conf_thres 没有目标 %s', conf_thres)
        return []
    else:
        flog.debug('过滤后有 %s 个', detection.shape[0])
    scores = detection[:, 4]
    bbox = detection[:, :4]
    # 2 . 根据得分对框进行从大到小排序。
    keep = nms(bbox, scores, nms_thres)

    return detection[keep]

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
