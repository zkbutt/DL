import math

import numpy as np
import torch
import torch.nn.functional as F
from f_tools.f_general import labels2onehot4ts
from f_tools.fun_od.f_boxes import ltrb2xywh, xywh2ltrb, calc_iou4ts, offxy2xy, bbox_iou4one
from f_tools.pic.f_show import f_show_od_np4plt
from f_tools.yufa.x_calc_adv import f_mershgrid
from object_detection.z_center.utils import gaussian_radius, draw_gaussian

'''-------------------------------YOLO1 编解码--------------------------'''


def boxes_encode4yolo1(gboxes_ltrb, grid_h, grid_w, device, cfg):
    '''
    编码 GT 直接预测特图的 wh
    :param gboxes_ltrb: 归一化尺寸
    :param grid_h:
    :param grid_w:
    :return: 返回特图回归系数
    '''
    # ltrb -> xywh 原图归一化   编码xy与yolo2一样的
    gboxes_xywh = ltrb2xywh(gboxes_ltrb)
    whs = gboxes_xywh[:, 2:]
    # 算出中间点所在的格子
    cxys = gboxes_xywh[:, :2]
    grids_ts = torch.tensor([grid_h, grid_w], device=device, dtype=torch.int16)
    index_colrow = (cxys * grids_ts).type(torch.int16)  # 网格7的index
    offset_xys = torch.true_divide(index_colrow, grid_h)  # 网络index 对应归一化的实距

    txys = (cxys - offset_xys) * grids_ts  # 特图偏移
    twhs = (whs * torch.tensor(grid_h, device=device)).log()  # 直接预测特图长宽 log减小差距
    txywhs_g = torch.cat([txys, twhs], dim=-1)

    # 小目标损失加重
    weights = 2.0 - torch.prod(whs, dim=-1)
    return txywhs_g, weights, index_colrow


def boxes_decode4yolo1(ptxywh, grid_h, grid_w, cfg):
    '''
    解码出来是特图
    :param ptxywh: 预测的是在特图的 偏移 和 缩放比例
    :param cfg:
    :return: 输出原图归一化
    '''
    device = ptxywh.device
    # 单一格子偏移 + 特图格子偏移
    _xy_grid = torch.sigmoid(ptxywh[:, :, :2]) \
               + f_mershgrid(grid_h, grid_w, is_rowcol=False, num_repeat=cfg.NUM_ANC).to(device)
    hw_ts = torch.tensor((grid_h, grid_w), device=device)  # /13
    zxy = torch.true_divide(_xy_grid, hw_ts)  # 特图-》 原图归一化
    zwh = torch.exp(ptxywh[:, :, 2:]) / grid_h  # 特图-》原图归一化
    zxywh = torch.cat([zxy, zwh], -1)
    zltrb = xywh2ltrb(zxywh)
    return zltrb


'''-------------------------------YOLO2 编解码--------------------------'''


def boxes_encode4yolo2(gboxes_ltrb_b, mask_p, grid_h, grid_w, device, cfg):
    '''
    编码GT
    :param gboxes_ltrb_b: 归一化尺寸
    :return:
    '''
    # ltrb -> xywh 原图归一化  编码xy与yolo1一样的
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    whs = gboxes_xywh[:, 2:]
    # 算出中间点所在的格子
    cxys = gboxes_xywh[:, :2]
    grids_ts = torch.tensor([grid_h, grid_w], device=device, dtype=torch.int16)
    colrows_index = (cxys * grids_ts).type(torch.int16)  # 网格7的index
    offset_xys = torch.true_divide(colrows_index, grid_h)  # 网络index 对应归一化的实距
    txys = (cxys - offset_xys) * grids_ts  # 特图偏移

    # twhs = (whs * torch.tensor(grid_h, device=device)).log()  # 特图长宽 log减小差距
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)[mask_p]
    # torch.Size([1, 2]) ^ torch.Size([5, 2]) = [5,2]
    twhs = (whs / ancs_wh_ts).log()  # 特图长宽 log减小差距

    txywhs_g = torch.cat([txys, twhs], dim=-1)
    # 正例的conf
    weights = 2.0 - torch.prod(whs, dim=-1)
    return txywhs_g, weights, colrows_index


def boxes_encode4yolo2_4iou(gboxes_ltrb_b, preg_b, match_anc_ids, grid_h, grid_w, device, cfg):
    '''
    编码GT
    :param gboxes_ltrb_b: 归一化尺寸
    :param preg_b: yolo2只有一层 特图
    :return:
    '''
    # ltrb -> xywh 原图归一化  编码xy与yolo1一样的
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    whs = gboxes_xywh[:, 2:]

    cxys = gboxes_xywh[:, :2]
    grids_ts = torch.tensor([grid_h, grid_w], device=device, dtype=torch.int16)
    colrows_index = (cxys * grids_ts).type(torch.int16)  # 网格7的index
    offset_xys = torch.true_divide(colrows_index, grid_h)  # 网络index 对应归一化的实距
    txys = (cxys - offset_xys) * grids_ts  # 特图偏移

    # twhs = (whs * torch.tensor(grid_h, device=device)).log()  # 特图长宽 log减小差距
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)[match_anc_ids]
    # torch.Size([1, 2]) ^ torch.Size([5, 2]) = [5,2]
    twhs = (whs / ancs_wh_ts).log()  # 特图长宽 log减小差距

    txywhs_g = torch.cat([txys, twhs], dim=-1)
    # 正例的conf
    weights = 2.0 - torch.prod(whs, dim=-1)
    return txywhs_g, weights, colrows_index


def boxes_decode4yolo2(preg, grid_h, grid_w, cfg):
    '''
    解码  4维 -> 3维
    用于计算 iou得conf   及 预测
    :param preg: 原始预测 torch.Size([32, 169, 5, 4]) -> [3, 169*5, 4]
    :return: 输出原图归一化 [3, 169*5, 4]
    '''
    device = preg.device
    # 特图xy -> 原图
    pxy = torch.sigmoid(preg[:, :, :, :2]) \
          + f_mershgrid(grid_h, grid_w, is_rowcol=False, num_repeat=cfg.NUM_ANC) \
              .to(device).reshape(-1, cfg.NUM_ANC, 2)
    pxy = pxy / grid_h

    # 特图wh比例 -> 原图
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    # 比例 ( 比例不需要转换 ) * 特图anc_wh
    pwh = torch.exp(preg[:, :, :, 2:4]) * ancs_wh_ts  # torch.Size([3, 361, 5, 2])
    # fdebug 可视化匹配的anc
    # pwh = ancs_ts.view(1, 1, *ancs_ts.shape).repeat(*ptxywh[:, :, :, 2:4].shape[:2], 1, 1)

    pxywh = torch.cat([pxy, pwh], -1)  # torch.Size([3, 169, 5, 4])
    pxywh = pxywh.view(preg.shape[0], -1, 4)  # 原图归一化 [3, 169, 5, 4] -> [3, 169*5, 4]
    pltrb = xywh2ltrb(pxywh)
    return pltrb


'''-------------------------------YOLO3 编解码--------------------------'''


def boxes_decode4yolo3(ptxywh, cfg):
    '''
    :param ptxywh: torch.Size([3, 10647, 4]) 不要归一化
    :param cfg:
        cfg.NUMS_CENG [2704, 676, 169]
        cfg.NUMS_ANC [3, 3, 3]
    :return: 输出原图归一化 [3, 10647, 4]
    '''
    device = ptxywh.device
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)

    ptxywh_ = ptxywh.clone().detach()
    indexs_grid = []
    ancs_wh_match = []
    grids = []  # 用于原图归一化
    _s = 0
    for i, num_ceng in enumerate(cfg.NUMS_CENG):
        num_ceng_t = num_ceng * cfg.NUMS_ANC[i]  # 2704*3 = 8112
        _grid = int(math.sqrt(num_ceng))
        #  52 26 13
        grids.append(torch.tensor(_grid, device=device, dtype=torch.int32).repeat(num_ceng_t))

        _grids = f_mershgrid(_grid, _grid, is_rowcol=False, num_repeat=cfg.NUMS_ANC[i]).to(device)
        indexs_grid.append(_grids)  # 每层对应的colrow索引 torch.Size([8112, 2])

        _ancs_wh_match = ancs_wh_ts[_s:_s + cfg.NUMS_ANC[i]]  # 选出层对应的 anc
        _ancs_wh_match = _ancs_wh_match.repeat(num_ceng, 1)  # torch.Size([8112, 2])
        ancs_wh_match.append(_ancs_wh_match)
        _s += cfg.NUMS_ANC[i]
    # torch.Size([8112, 2])  torch.Size([2028, 2])  torch.Size([507, 2])
    indexs_grid = torch.cat(indexs_grid, 0)  # torch.Size([10647, 2])
    ancs_wh_match = torch.cat(ancs_wh_match, 0)  # torch.Size([10647, 2])
    grids = torch.cat(grids, -1)  # 10647

    pxy = ptxywh_[:, :, :2].sigmoid() + indexs_grid
    pxy = pxy / grids.view(1, -1, 1)
    pwh = ptxywh_[:, :, 2:4].exp() * ancs_wh_match
    pxywh = torch.cat([pxy, pwh], -1)  # torch.Size([32, 10647, 4])
    pltrb = xywh2ltrb(pxywh)
    return pltrb


'''-------------------------------SSD 编解码--------------------------'''


def boxes_encode4ssd(cfg, ancs_xywh, gboxes_xywh_match):
    ''' 与 retina 一致 '''
    txy = 10 * (gboxes_xywh_match[..., :2] - ancs_xywh[..., :2]) / ancs_xywh[..., 2:]
    twh = 5 * (gboxes_xywh_match[..., 2:] / ancs_xywh[..., 2:]).log()
    _gtxywh = torch.cat([txy, twh], dim=-1)
    return _gtxywh


def boxes_decode4ssd(cfg, preg, ancs_xywh):
    gxy = 0.1 * preg[..., :2] * ancs_xywh[..., 2:] + ancs_xywh[..., :2]
    gwh = (0.2 * preg[..., 2:]).exp() * ancs_xywh[..., 2:]
    _gtxywh = torch.cat([gxy, gwh], dim=-1)
    return _gtxywh


'''-------------------------------retina 编解码--------------------------'''


def boxes_encode4retina(cfg, anc_obj, gboxes_ltrb):
    '''
    用anc同维 和 已匹配的GT 计算差异
    :param cfg: cfg.tnums_ceng cfg.NUMS_ANC [8112, 2028, 507]  用于动态层 dim 数
    :param anc_obj:
        ancs_xywh  (nn,4) torch.Size([1, 16800, 4]) torch.Size([10647, 4])
        anc_obj.nums_dim_feature: [24336, 6084, 1521, 441, 144]
    :param gboxes_ltrb: gt torch.Size([5, 16800, 4]) torch.Size([10647, 4])
    :return: 计算差异值   对应xywh torch.Size([10647, 4])
    '''
    nums_ceng_np = np.array(anc_obj.nums_level, dtype=np.int32)
    nums_anc_np = np.array(cfg.NUMS_ANC, dtype=np.float32)
    grids_np = np.sqrt((nums_ceng_np / nums_anc_np))  # [ 52 26 13 7 4]
    match_grids_ts = torch.tensor(np.repeat(grids_np, nums_ceng_np, axis=-1), device=anc_obj.device, dtype=torch.float)
    match_grids_ts = match_grids_ts.view(-1, 1)  # torch.Size([32526, 1]) -> [32526, 4]

    gxywh = ltrb2xywh(gboxes_ltrb)
    gxywh_t = gxywh * match_grids_ts

    if gboxes_ltrb.dim() == 2:
        anc_xywh_t = anc_obj.ancs_xywh * match_grids_ts
        _a = 10 * (gxywh_t[:, :2] - anc_xywh_t[:, :2]) / anc_xywh_t[:, 2:]
        _b = 5 * (gxywh_t[:, 2:] / anc_xywh_t[:, 2:]).log()
    elif gboxes_ltrb.dim() == 3:
        anc_xywh_t = anc_obj.ancs_xywh.unsqueeze(0) * match_grids_ts
        _a = 10 * (gxywh_t[:, :, :2] - anc_xywh_t[:, :, :2]) / anc_xywh_t[:, :, 2:]
        _b = 5 * (gxywh_t[:, :, 2:] / anc_xywh_t[:, :, 2:]).log()
    else:
        raise Exception('维度错误', anc_obj.ancs_xywh.shape)
    _gtxywh = torch.cat([_a, _b], dim=-1)
    return _gtxywh


def boxes_decode4retina(cfg, anc_obj, ptxywh, variances=(0.1, 0.2)):
    '''
    需要统一 一维和二维
    :param cfg: cfg.tnums_ceng cfg.NUMS_ANC [8112, 2028, 507]  用于动态层 dim 数
    :param anc_obj: xywh  (nn,4) 这个始终是一维
    :param ptxywh: 修正系数 (nn,4)  torch.Size([32, 10647, 4])
    :return: 修复后的框
    '''
    nums_ceng_np = np.array(anc_obj.nums_level, dtype=np.int32)
    nums_anc_np = np.array(cfg.NUMS_ANC, dtype=np.float32)
    grids_np = np.sqrt((nums_ceng_np / nums_anc_np))
    match_grids_ts = torch.tensor(np.repeat(grids_np, nums_ceng_np, axis=-1), device=anc_obj.device, dtype=torch.float)
    match_grids_ts = match_grids_ts.view(-1, 1)  # torch.Size([10647, 1])

    # torch.Size([32, 10647, 4]) * torch.Size([10647, 1])
    ptxywh_t = ptxywh  # 这里是特图

    if ptxywh.dim() == 2:
        anc_xywh_t = anc_obj.ancs_xywh * match_grids_ts
        _anc_xy = anc_xywh_t[:, :2]
        _anc_wh = anc_xywh_t[:, 2:]
        _pxy = ptxywh_t[:, :2]
        _pwh = ptxywh_t[:, 2:]
    elif ptxywh.dim() == 3:
        anc_xywh_t = anc_obj.ancs_xywh.unsqueeze(0) * match_grids_ts
        _anc_xy = anc_xywh_t[:, :, :2]
        _anc_wh = anc_xywh_t[:, :, 2:]
        _pxy = ptxywh_t[:, :, :2]
        _pwh = ptxywh_t[:, :, 2:]
    else:
        raise Exception('维度错误', anc_obj.ancs_xywh.shape)
    xy = _anc_xy + _pxy * 0.1 * _anc_wh
    wh = _anc_wh * torch.exp(_pwh * 0.2)
    _pxywh = torch.cat([xy, wh], dim=-1)
    _pxywh = _pxywh / match_grids_ts
    return _pxywh


'''------------------------------- 其他 --------------------------'''


def boxes_encode4distribute(cfg, anc_obj, gboxes_ltrb_match, max_dis):
    '''
    for dfl
    :param cfg:
    :param anc_obj:
    :param gboxes_ltrb_match:
    :param max_dis: 根据定义的向量数 限制在向量-1 值以内
    :return:
    '''
    # anc 归一化 -> anc特图
    match_grids_ts, anc_xywh_t = anc_obj.match_anc_grids()
    gboxes_ltrb_t = gboxes_ltrb_match * match_grids_ts

    pt_lt = anc_xywh_t[:, :2] - gboxes_ltrb_t[:, :2]
    pt_rb = gboxes_ltrb_t[:, 2:] - anc_xywh_t[:, :2]
    gt_ltrb_t = torch.cat([pt_lt, pt_rb], -1)

    eps = torch.finfo(torch.float16).eps
    gt_ltrb_t.clamp_(min=0, max=max_dis - eps)  # 无需计算梯度

    return gt_ltrb_t


def boxes_decode4gfl(cfg, anc_obj, preg_32d_b, is_clamp=False, gboxes_ltrb_match=None):
    '''
    2D 对齐
    :param cfg:
        cfg.NUMS_CENG [2704, 676, 169, 49, 16]
    :param anc_obj:
        feature_sizes [[52, 52], [26, 26], [13, 13], [7, 7], [4, 4]]
        nums_level [2704, 676, 169, 49, 16]
        ancs_xywh torch.Size([3614, 4])
    :param poff_ltrb:
    :param is_clamp: 训练时为 False
    :param max_shape: 对应各城的尺寸
    :param gboxes_ltrb_match: 用于训练时计算IOU
    :return:
    '''
    # 得预测值
    # 看着任意概率分布 torch.Size([3614, 32]) -> torch.Size([14456, 8])
    _preg_32d_b = F.softmax(preg_32d_b.reshape(-1, cfg.NUM_REG), dim=1)
    # 与0~7值进行矩阵乘 得期望（预测值） 必然是0~7之间的 [8 (0~7) ]
    _arange = torch.arange(0, cfg.NUM_REG, device=preg_32d_b.device, dtype=torch.float)
    # 这个是矩阵乘法 [14456, 8] ^^ ( [8]广播[8,1]自动加T ) -> [14456, 1] -> [3614, 4]
    poff_ltrb_4d = F.linear(_preg_32d_b, _arange).reshape(-1, 4)  # 生成一堆 0~7的数

    # anc 归一化 -> anc特图
    match_grids_ts, anc_xywh_t = anc_obj.match_anc_grids()

    box_lt = anc_xywh_t[:, :2] - poff_ltrb_4d[:, :2]
    box_rb = anc_xywh_t[:, :2] + poff_ltrb_4d[:, 2:]
    boxes_t_ltrb = torch.cat([box_lt, box_rb], -1)

    if is_clamp:  # 预测时限制输出在特图长宽中
        index_start = 0
        for i, num_ceng in enumerate(anc_obj.nums_level):
            _index_end = index_start + num_ceng
            boxes_t_ltrb[index_start:_index_end, :] = boxes_t_ltrb[index_start:_index_end, :].clamp(
                min=0,
                max=anc_obj.feature_sizes[i][0])

    # 训练时用 这里用于计算IOU 实际IOU
    if gboxes_ltrb_match is not None:
        gboxes_t_ltrb_match = gboxes_ltrb_match * match_grids_ts
        ious_zg = bbox_iou4one(boxes_t_ltrb, gboxes_t_ltrb_match, is_ciou=True)  # [3614]
        return boxes_t_ltrb, ious_zg

    return boxes_t_ltrb


def boxes_decode4distribute(cfg, anc_obj, preg_32d_b, is_clamp=False, gboxes_ltrb_match=None):
    '''
    解码分布并计算IOU
    :param cfg:
        cfg.NUMS_CENG [2704, 676, 169, 49, 16]
    :param anc_obj:
        feature_sizes [[52, 52], [26, 26], [13, 13], [7, 7], [4, 4]]
        nums_level [2704, 676, 169, 49, 16]
        ancs_xywh torch.Size([3614, 4])
    :param preg_32d_b:
    :param is_clamp: 训练时为 False
    :param gboxes_ltrb_match: 用于训练时计算IOU
    :return:
        归一化的的预测框
    '''
    # 得预测值
    # 看着任意概率分布 torch.Size([3614, 32]) -> torch.Size([14456, 8])
    _preg_32d_b = F.softmax(preg_32d_b.reshape(-1, cfg.NUM_REG), dim=1)
    # 与0~7值进行矩阵乘 得期望（预测值） 必然是0~7之间的 [8 (0~7) ]
    _arange = torch.arange(0, cfg.NUM_REG, device=preg_32d_b.device, dtype=torch.float)
    # 这个是矩阵乘法 [14456, 8] ^^ ( [8]广播[8,1]自动加T ) -> [14456, 1] -> [3614, 4]
    poff_ltrb_t_4d = F.linear(_preg_32d_b, _arange).reshape(-1, 4)  # 生成一堆 0~7的数

    # anc 归一化 -> anc特图
    match_grids_ts, anc_xywh_t = anc_obj.match_anc_grids()

    box_lt = anc_xywh_t[:, :2] - poff_ltrb_t_4d[:, :2]
    box_rb = anc_xywh_t[:, :2] + poff_ltrb_t_4d[:, 2:]
    boxes_t_ltrb = torch.cat([box_lt, box_rb], -1)

    if is_clamp:  # 预测时限制输出在特图长宽中
        index_start = 0
        for i, num_ceng in enumerate(anc_obj.nums_level):
            _index_end = index_start + num_ceng
            boxes_t_ltrb[index_start:_index_end, :] = boxes_t_ltrb[index_start:_index_end, :].clamp(
                min=0,
                max=anc_obj.feature_sizes[i][0])

    # 特图归一化到原图
    boxes_ltrb = boxes_t_ltrb / match_grids_ts
    # 训练时用 这里用于计算IOU 实际IOU
    if gboxes_ltrb_match is not None:
        ious_zg = bbox_iou4one(boxes_ltrb, gboxes_ltrb_match, is_ciou=True)  # [3614]
        return boxes_ltrb, ious_zg

    return boxes_ltrb


'''------------------------------- 匹配 --------------------------'''


def matchs_gfl(cfg, dim, gboxes_ltrb_b, glabels_b, anc_obj, preg_32d_b, mode='atss', img_ts=None):
    '''
    cfg.NUMS_ANC = [3, 3, 3]

    :param cfg:
    :param dim: ious_zg-1 , cls-3,  与预测对应gt_ltrb-4
    :param gboxes_ltrb_b: torch.Size([ngt, 4])
    :param glabels_b: [ngt]
    :param anc_obj:
        anc_obj.nums_level: [24336, 6084, 1521, 441, 144]
        ancs_xywh  (nn,4)
    :param mode: topk atss iou
    :param preg_32d_b: torch.Size([3614, 32])
    :param img_ts: [3, 416, 416]
    :return:
    '''
    # mode = 'iou'  # topk atss iou
    if preg_32d_b is not None:
        device = preg_32d_b.device
    else:
        device = torch.device('cpu')

    # 计算 iou
    anc_xywh = anc_obj.ancs_xywh
    anc_ltrb = xywh2ltrb(anc_xywh)
    num_anc = anc_xywh.shape[0]
    # (anc 个,boxes 个) torch.Size([3, 10647])
    ious_ag = calc_iou4ts(anc_ltrb, gboxes_ltrb_b)
    num_gt = gboxes_ltrb_b.shape[0]

    # 全部ANC的距离
    gboxes_xywh_b = ltrb2xywh(gboxes_ltrb_b)
    # 中间点绝对距离 多维广播 (anc 个,boxes 个)  torch.Size([32526, 7])
    distances = (anc_xywh[:, None, :2] - gboxes_xywh_b[None, :, :2]).pow(2).sum(-1).sqrt()

    # conf-1, cls-3, txywh-4, keypoint-nn  = 8 + nn
    gretinas_one = torch.zeros((num_anc, dim), device=device)  # 返回值
    s_ = 1 + cfg.NUM_CLASSES  # 前面 两个是 conf-1, cls-3,

    if mode == 'atss':
        ''' 使正例框数量都保持一致 保障小目标也能匹配到多个anc
        用最大一个iou的均值和标准差,计算阀值,用IOU阀值初选正样本
        确保anc中心点在gt框中
        '''
        # 每层 anc 数是一致的
        num_atss_topk = 9  # 这个 topk = 要 * 该层的anc数

        idxs_candidate = []
        index_start = 0  # 这是每层的anc偏移值
        for i, num_dim_feature in enumerate(anc_obj.nums_level):  # [24336, 6084, 1521, 441, 144]
            '''每一层的每一个GT选 topk * anc数'''
            index_end = index_start + num_dim_feature
            # 取出某层的所有anc距离  中间点绝对距离 (anc 个,boxes 个)  torch.Size([32526, 7]) -> [nn, 7]
            distances_per_level = distances[index_start:index_end, :]
            # 确认该层的TOPK 不能超过该层总 anc 数 这里是一个数
            # topk = min(num_atss_topk * cfg.NUMS_ANC[i], num_dim_feature)
            topk = min(num_atss_topk, num_dim_feature)
            # 放 topk个 每个gt对应对的anc的index torch.Size([24336, box_n])---(anc,gt) -> torch.Size([topk, 1])
            _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)  # 只能在某一维top
            idxs_candidate.append(topk_idxs_per_level + index_start)
            index_start = index_end

        # 用于计算iou均值和方差 候选人，候补者；应试者 torch.Size([405, 1])
        idxs_candidate = torch.cat(idxs_candidate, dim=0)
        '''--- 选出每层每个anc对应的距离中心最近topk iou值 ---'''
        # ***************这个是ids选择 这个是多维筛选 ious---[anc,ngt]    [405, ngt] [0,1...ngt]-> [405,ngt]
        ious_candidate = ious_ag[idxs_candidate, torch.arange(num_gt)]
        mask_distances = torch.zeros_like(distances, device=device, dtype=torch.bool)
        mask_distances[idxs_candidate, torch.arange(idxs_candidate.shape[1])] = True

        # debug 可视  匹配正例可视化
        # mask_pos = torch.zeros(mask_distances.shape[0], device=device, dtype=torch.bool)
        # mask_distances_t = mask_distances.t()  # [32526 , 3] -> [3, 32526]
        # for m_pos_iou in mask_distances_t:  # 拉平 每个 32526
        #     mask_pos = torch.logical_or(m_pos_iou, mask_pos)
        # from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        # _img_ts = f_recover_normalization4ts(img_ts.clone())
        # from torchvision.transforms import functional as transformsF
        # img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        # import numpy as np
        # img_np = np.array(img_pil)
        #
        # f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b.cpu()
        #                  , pboxes_ltrb=anc_ltrb[mask_pos].cpu(),
        #                  is_recover_size=True)

        '''--- 用最大一个iou的均值和标准差,计算阀值 ---'''
        # 统计每一个 GT的均值 std [ntopk,ngt] -> [ngt] 个
        _iou_mean_per_gt = ious_candidate.mean(dim=0)  # 除维
        _iou_std_per_gt = ious_candidate.std(dim=0)
        _iou_thresh_per_gt = _iou_mean_per_gt + _iou_std_per_gt
        '''--- 用IOU阀值初选正样本 ---'''
        # torch.Size([32526, 1]) ^^ ([ngt] -> [1,ngt]) -> [32526,ngt]
        mask_pos4iou = ious_ag >= _iou_thresh_per_gt[None, :]  # 核心是这个选

        '''--- 中心点需落在GT中间 需要选出 anc的中心点-gt的lt为正, gr的rb-anc的中心点为正  ---'''
        # torch.Size([32526, 1, 2])
        dlt = anc_xywh[:, None, :2] - gboxes_ltrb_b[None, :, :2]
        drb = gboxes_ltrb_b[None, :, 2:] - anc_xywh[:, None, :2]
        # [32526, 1, 2] -> [32526, 1, 4] -> [32526, 1]
        mask_pos4in_gt = torch.all(torch.cat([dlt, drb], dim=-1) > 0.01, dim=-1)
        mask_pos_iou = torch.logical_and(torch.logical_and(mask_distances, mask_pos4iou), mask_pos4in_gt)

        '''--- 生成最终正例mask [32526, ngt] -> [32526] ---'''
        mask_pos = torch.zeros(mask_pos_iou.shape[0], device=device, dtype=torch.bool)
        mask_pos_iou = mask_pos_iou.t()  # [32526 , 3] -> [3, 32526]
        for m_pos_iou in mask_pos_iou:  # 拉平 每个 32526
            mask_pos = torch.logical_or(m_pos_iou, mask_pos)

        # mask_neg = torch.logical_not(mask_pos)

        '''--- 确定anc匹配 一个锚框被多个真实框所选择，则其归于iou较高的真实框  ---'''
        # (anc 个,boxes 个) torch.Size([3, 10647])
        anc_max_iou, boxes_index = ious_ag.max(dim=1)  # 存的是 bboxs的index

        '''--- GFL解码  ---'''
        # 解码 通过 anc ^^ 回归系数 = 最终预测框 和 giou值 用于giou损失
        gboxes_ltrb_match = gboxes_ltrb_b[boxes_index].type(torch.float)
        # zboxes_t_ltrb 这里解码出来 没用
        zboxes_t_ltrb, ious_zg = boxes_decode4distribute(cfg, anc_obj, preg_32d_b, is_clamp=False,
                                                         gboxes_ltrb_match=gboxes_ltrb_match)

        '''   ious_zg-1 , cls-3,  与预测对应gt_ltrb-4  '''
        gretinas_one[mask_pos, 0] = ious_zg[mask_pos].clamp(min=0)

        glabels_b_onehot = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
        gretinas_one[mask_pos, 1:s_] = glabels_b_onehot[boxes_index][mask_pos].type(torch.float)  # 正例才匹配

        # 用anc 和gt 编码实际值每个0~7 之间 最大8-1
        _gt_ltrb_t = boxes_encode4distribute(cfg, anc_obj, gboxes_ltrb_match, max_dis=cfg.NUM_REG - 1)
        gretinas_one[mask_pos, s_:s_ + 4] = _gt_ltrb_t[mask_pos]

        # 匹配正例可视化
        # from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        # _img_ts = f_recover_normalization4ts(img_ts.clone())
        # from torchvision.transforms import functional as transformsF
        # img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        # import numpy as np
        # img_np = np.array(img_pil)
        # match_grids_ts, anc_xywh_t = anc_obj.match_anc_grids()
        #
        # f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b.cpu()
        #                  , pboxes_ltrb=anc_ltrb[mask_pos].cpu(),
        #                  other_ltrb=(zboxes_t_ltrb / match_grids_ts)[mask_pos].detach().cpu(),
        #                  is_recover_size=True)

    elif mode == 'topk':
        '''--- 简单匹配9个 用IOU和距离相关计算 每特图取9个最好的 ---'''
        num_atss_topk = 9
        distances = distances / distances.max() / 1000
        mask_pos_candidate = torch.zeros_like(ious_ag, dtype=torch.bool)  # torch.Size([32526, 2])
        for ng in range(num_gt):  # 遍历每一个GT匹配9个
            # iou和距离差 [3614] -> [9]
            _, topk_idxs = (ious_ag[:, ng] - distances[:, ng]).topk(num_atss_topk, dim=0)
            anc_xy = anc_xywh[topk_idxs, :2]
            dlt = anc_xy - gboxes_ltrb_b[ng, :2]
            drb = - anc_xy + gboxes_ltrb_b[ng, 2:]
            # [topk,4] -> [topk]
            mask_4in_gt = torch.all(torch.cat([dlt, drb], dim=-1) > 0.01, dim=1)
            mask_pos_candidate[topk_idxs[mask_4in_gt], ng] = True

        # mask转换 [32526, 2] -> [32526]
        mask_pos = torch.zeros(mask_pos_candidate.shape[0], device=device, dtype=torch.bool)
        mask_pos_iou = mask_pos_candidate.t()  # [32526 , 3] -> [3, 32526]
        for m_pos_iou in mask_pos_iou:  # 拉平 每个 32526
            mask_pos = torch.logical_or(m_pos_iou, mask_pos)

        anc_max_iou, boxes_index = ious_ag.max(dim=1)  # 存的是 bboxs的index

        gretinas_one[mask_pos, 0] = torch.tensor(1., device=device)

        glabels_b_onehot = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
        gretinas_one[mask_pos, 1:s_] = glabels_b_onehot[boxes_index][mask_pos].type(torch.float)  # 正例才匹配
        _gtxywh = boxes_encode4retina(cfg, anc_obj, gboxes_ltrb_b[boxes_index], variances=cfg.variances)
        gretinas_one[mask_pos, s_:s_ + 4] = _gtxywh[mask_pos]

    elif mode == 'iou':
        '''--- iou阀值来确定 正反忽略 通常 >0.5 <0.4 正例至少匹配1个 ---'''
        # (anc 个,boxes 个) torch.Size([3, 10647])
        anc_max_iou, boxes_index = ious_ag.max(dim=1)  # 存的是 bboxs的index
        bbox_max_iou, anc_index = ious_ag.max(dim=0)  # 存的是 anc的index
        anc_max_iou.index_fill_(0, anc_index, 1)  # 最大的为正例, 强制为conf 为1
        gt_ids = torch.arange(0, anc_index.size(0), dtype=torch.int64).to(boxes_index)  # [0,1]
        anc_ids = anc_index[gt_ids]
        boxes_index[anc_ids] = gt_ids
        mask_pos = anc_max_iou >= cfg.THRESHOLD_CONF_POS  # [10647] anc 的正例 index 不管
        mask_neg = anc_max_iou < cfg.THRESHOLD_CONF_NEG  # anc 的反例 index
        mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
        # mask_neg 负例自动0

        gretinas_one[mask_pos, 0] = torch.tensor(1., device=device)
        gretinas_one[mash_ignore, 0] = torch.tensor(-1., device=device)

        glabels_b_onehot = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
        gretinas_one[mask_pos, 1:s_] = glabels_b_onehot[boxes_index][mask_pos].type(torch.float)  # 正例才匹配
        _gtxywh = boxes_encode4retina(cfg, anc_obj, gboxes_ltrb_b[boxes_index], variances=cfg.variances)
        gretinas_one[mask_pos, s_:s_ + 4] = _gtxywh[mask_pos]
    else:
        raise NotImplementedError

    return gretinas_one


def matchs_gt_b(cfg, gboxes_ltrb_b, glabels_b, anc_obj, mode='atss', ptxywh_b=None, img_ts=None, num_atss_topk=9):
    '''
    cfg.NUMS_ANC = [3, 3, 3]

    :param cfg:
    :param gboxes_ltrb_b: torch.Size([ngt, 4])
    :param glabels_b: [ngt]
    :param anc_obj:
        anc_obj.nums_level: [24336, 6084, 1521, 441, 144]
        ancs_xywh  (nn,4)
    :param mode: topk atss iou
    :param ptxywh_b: torch.Size([3614, 32])
    :param img_ts: [3, 416, 416]
    :param num_atss_topk: 9
    :return:
    '''
    # mode = 'iou'  # topk atss iou
    if ptxywh_b is not None:
        device = ptxywh_b.device
    else:
        device = torch.device('cpu')

    # 计算 iou
    anc_xywh = anc_obj.ancs_xywh
    anc_ltrb = xywh2ltrb(anc_xywh)
    # num_anc = anc_xywh.shape[0]
    # (anc 个,boxes 个) torch.Size([3, 10647])
    ious = calc_iou4ts(anc_ltrb, gboxes_ltrb_b)
    num_gt = gboxes_ltrb_b.shape[0]

    # 全部ANC的距离
    gboxes_xywh_b = ltrb2xywh(gboxes_ltrb_b)
    # 中间点绝对距离 多维广播 (anc 个,boxes 个)  torch.Size([32526, 7])
    distances = (anc_xywh[:, None, :2] - gboxes_xywh_b[None, :, :2]).pow(2).sum(-1).sqrt()

    mask_neg, mash_ignore = None, None
    if mode == 'atss':
        ''' 使正例框数量都保持一致 保障小目标也能匹配到多个anc
        用最大一个iou的均值和标准差,计算阀值,用IOU阀值初选正样本
        确保anc中心点在gt框中
        '''
        idxs_candidate = []
        index_start = 0  # 这是每层的anc偏移值
        for i, num_dim_feature in enumerate(anc_obj.nums_level):  # [24336, 6084, 1521, 441, 144]
            '''每一层的每一个GT选 topk * anc数'''
            index_end = index_start + num_dim_feature
            # 取出某层的所有anc距离  中间点绝对距离 (anc 个,boxes 个)  torch.Size([32526, 7]) -> [nn, 7]
            distances_per_level = distances[index_start:index_end, :]
            # 确认该层的TOPK 不能超过该层总 anc 数 这里是一个数
            # topk = min(num_atss_topk * cfg.NUMS_ANC[i], num_dim_feature)
            topk = min(num_atss_topk, num_dim_feature)
            # 放 topk个 每个gt对应对的anc的index torch.Size([24336, box_n])---(anc,gt) -> torch.Size([topk, 1])
            _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)  # 只能在某一维top
            idxs_candidate.append(topk_idxs_per_level + index_start)
            index_start = index_end

        # 用于计算iou均值和方差 候选人，候补者；应试者 torch.Size([405, 1])
        idxs_candidate = torch.cat(idxs_candidate, dim=0)
        '''--- 选出每层每个anc对应的距离中心最近topk iou值 ---'''
        # ***************这个是ids选择 这个是多维筛选 ious---[anc,ngt]    [405, ngt] [0,1...ngt]-> [405,ngt]
        ious_candidate = ious[idxs_candidate, torch.arange(num_gt)]
        mask_distances = torch.zeros_like(distances, device=device, dtype=torch.bool)
        mask_distances[idxs_candidate, torch.arange(idxs_candidate.shape[1])] = True

        # debug 可视  匹配正例可视化
        # mask_pos = torch.zeros(mask_distances.shape[0], device=device, dtype=torch.bool)
        # mask_distances_t = mask_distances.t()  # [32526 , 3] -> [3, 32526]
        # for m_pos_iou in mask_distances_t:  # 拉平 每个 32526
        #     mask_pos = torch.logical_or(m_pos_iou, mask_pos)
        # from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        # _img_ts = f_recover_normalization4ts(img_ts.clone())
        # from torchvision.transforms import functional as transformsF
        # img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        # import numpy as np
        # img_np = np.array(img_pil)
        #
        # f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b.cpu()
        #                  , pboxes_ltrb=anc_ltrb[mask_pos].cpu(),
        #                  is_recover_size=True)

        '''--- 用最大一个iou的均值和标准差,计算阀值 ---'''
        # 统计每一个 GT的均值 std [ntopk,ngt] -> [ngt] 个
        _iou_mean_per_gt = ious_candidate.mean(dim=0)  # 除维
        _iou_std_per_gt = ious_candidate.std(dim=0)
        _iou_thresh_per_gt = _iou_mean_per_gt + _iou_std_per_gt
        '''--- 用IOU阀值初选正样本 ---'''
        # torch.Size([32526, 1]) ^^ ([ngt] -> [1,ngt]) -> [32526,ngt]
        mask_pos4iou = ious >= _iou_thresh_per_gt[None, :]  # 核心是这个选

        '''--- 中心点需落在GT中间 需要选出 anc的中心点-gt的lt为正, gr的rb-anc的中心点为正  ---'''
        # torch.Size([32526, 1, 2])
        dlt = anc_xywh[:, None, :2] - gboxes_ltrb_b[None, :, :2]
        drb = gboxes_ltrb_b[None, :, 2:] - anc_xywh[:, None, :2]
        # [32526, 1, 2] -> [32526, 1, 4] -> [32526, 1]
        mask_pos4in_gt = torch.all(torch.cat([dlt, drb], dim=-1) > 0.01, dim=-1)
        mask_pos_iou = torch.logical_and(torch.logical_and(mask_distances, mask_pos4iou), mask_pos4in_gt)

        '''--- 生成最终正例mask [32526, ngt] -> [32526] ---'''
        mask_pos = torch.zeros(mask_pos_iou.shape[0], device=device, dtype=torch.bool)
        mask_pos_iou = mask_pos_iou.t()  # [32526 , 3] -> [3, 32526]
        for m_pos_iou in mask_pos_iou:  # 拉平 每个 32526
            mask_pos = torch.logical_or(m_pos_iou, mask_pos)

        # mask_neg = torch.logical_not(mask_pos)

        '''--- 确定anc匹配 一个锚框被多个真实框所选择，则其归于iou较高的真实框  ---'''
        # (anc 个,boxes 个) torch.Size([3, 10647])
        anc_max_iou, boxes_index = ious.max(dim=1)  # 存的是 bboxs的index

        # 匹配正例可视化
        # from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        # _img_ts = f_recover_normalization4ts(img_ts.clone())
        # from torchvision.transforms import functional as transformsF
        # img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        # import numpy as np
        # img_np = np.array(img_pil)
        #
        # f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b.cpu()
        #                  , pboxes_ltrb=anc_ltrb[mask_pos].cpu(),
        #                  is_recover_size=True)

    elif mode == 'topk':
        '''--- 简单匹配9个 用IOU和距离相关计算 每特图取9个最好的 ---'''
        distances = distances / distances.max() / 1000
        mask_pos_candidate = torch.zeros_like(ious, dtype=torch.bool)  # torch.Size([32526, 2])
        for ng in range(num_gt):  # 遍历每一个GT匹配9个
            # iou和距离差 [3614] -> [9]
            _, topk_idxs = (ious[:, ng] - distances[:, ng]).topk(num_atss_topk, dim=0)
            anc_xy = anc_xywh[topk_idxs, :2]
            dlt = anc_xy - gboxes_ltrb_b[ng, :2]
            drb = - anc_xy + gboxes_ltrb_b[ng, 2:]
            # [topk,4] -> [topk]
            mask_4in_gt = torch.all(torch.cat([dlt, drb], dim=-1) > 0.01, dim=1)
            mask_pos_candidate[topk_idxs[mask_4in_gt], ng] = True

        # mask转换 [32526, 2] -> [32526]
        mask_pos = torch.zeros(mask_pos_candidate.shape[0], device=device, dtype=torch.bool)
        mask_pos_iou = mask_pos_candidate.t()  # [32526 , 3] -> [3, 32526]
        for m_pos_iou in mask_pos_iou:  # 拉平 每个 32526
            mask_pos = torch.logical_or(m_pos_iou, mask_pos)

        anc_max_iou, boxes_index = ious.max(dim=1)  # 存的是 bboxs的index

    elif mode == 'iou':
        '''--- iou阀值来确定 正反忽略 通常 >0.5 <0.4 正例至少匹配1个 ---'''
        # (anc 个,boxes 个) torch.Size([3, 10647])
        anc_max_iou, boxes_index = ious.max(dim=1)  # 存的是 bboxs的index
        bbox_max_iou, anc_index = ious.max(dim=0)  # 存的是 anc的index
        anc_max_iou.index_fill_(0, anc_index, 1)  # 最大的为正例, 强制为conf 为1
        gt_ids = torch.arange(0, anc_index.size(0), dtype=torch.int64).to(boxes_index)  # [0,1]
        anc_ids = anc_index[gt_ids]
        boxes_index[anc_ids] = gt_ids
        mask_pos = anc_max_iou >= cfg.THRESHOLD_CONF_POS  # [10647] anc 的正例 index 不管
        mask_neg = anc_max_iou < cfg.THRESHOLD_CONF_NEG  # anc 的反例 index
        mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
        # mask_neg 负例自动0


    else:
        raise NotImplementedError

    return boxes_index, mask_pos, mask_neg, mash_ignore


@torch.no_grad()
def pos_match4ssd(ancs, bboxs, criteria):
    '''
    正样本选取策略
        1. 每一个bboxs框 尽量有一个anc与之对应
        2. 每一个anc iou大于大于阀值的保留
    :param ancs:  ltrb
    :param bboxs: 标签框 (xx,4) ltrb
    :param criteria: 小于等于0.35的反例
    :return:
        label_neg_mask: 返回反例的布尔索引
        anc_bbox_ind : 通过 g_bbox[anc_bbox_ind]  可选出标签 与anc对应 其它为0
    '''
    # (bboxs个,anc个)
    iou = calc_iou4ts(bboxs, ancs, is_ciou=True)
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


def pos_match4yolo(ancs, bboxs, criteria):
    '''
    正例与GT最大的 负例<0.5 >0.5忽略
    :param ancs: xywh
    :param bboxs: ltrb
    :param criteria: 0.5
    :return:

    '''
    threshold = 99  # 任意指定一个值
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


def pos_match_retina4conf(cfg, dim, anc_obj, gboxes_ltrb_b, glabels_b, gkeypoints_b, ptxywh_b=None, img_ts=None):
    '''

    :param cfg:
    :param anc_obj: torch.Size([10647, 4])
    :param gboxes_ltrb_b: torch.Size([xx, 4])
    :param glabels_b: tensor([1, 1, 1])
    :param gkeypoints_b: None
    :param img_ts:
    :param nums_dim_feature: 特图总个数
    :return:
    '''
    anc_ltrb = xywh2ltrb(anc_obj.ancs_xywh)
    # (boxes 个,anc 个) torch.Size([3, 10647])
    iou = calc_iou4ts(gboxes_ltrb_b, anc_ltrb)
    # 除维运行 anc对应box最大的  anc个 bbox_index
    anc_max_iou, boxes_index = iou.max(dim=0)  # 存的是 bboxs的index

    # box对应anc最大的  box个 anc_index
    bbox_max_iou, anc_index = iou.max(dim=1)  # 存的是 anc的index

    # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
    anc_max_iou.index_fill_(0, anc_index, 1)  # 最大的为正例, 强制为conf 为1

    # 一定层度上使gt均有对应的anc, 处理多个anc对一gt的问题, 若多个gt对一个anc仍不能解决(情况较少)会导致某GT学习不了...遍历每一个gt索引
    gt_ids = torch.arange(0, anc_index.size(0), dtype=torch.int64).to(boxes_index)  # [0,1]
    # gt对应最好的anc索引取出来
    anc_ids = anc_index[gt_ids]
    # 找到这些anc 写入gt
    boxes_index[anc_ids] = gt_ids

    # ----------正例的index 和 正例对应的bbox索引----------
    mask_pos = anc_max_iou >= cfg.THRESHOLD_CONF_POS  # 0.5 [10647] anc 的正例 index 不管
    mask_neg = anc_max_iou < cfg.THRESHOLD_CONF_NEG  # 0.4 anc 的反例 index
    mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))

    dim_total, _d4 = ptxywh_b.shape
    device = ptxywh_b.device
    # conf-1, cls-3, txywh-4, keypoint-nn  = 8 + nn
    gretinas_one = torch.zeros((dim_total, dim), device=device)  # 返回值

    labels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)

    # 整体匹配 正例全部为1 默认为0
    gretinas_one[mask_pos, 0] = torch.tensor(1., device=device)
    gretinas_one[mash_ignore, 0] = torch.tensor(-1., device=device)
    # gretinas_one[:, 1] = anc_max_iou  # iou赋值

    # 局部更新
    s_ = 1 + cfg.NUM_CLASSES  # 前面 两个是 conf-1, cls-3,
    gretinas_one[mask_pos, 1:s_] = labels_b[boxes_index][mask_pos].type(torch.float)  # 正例才匹配
    # 匹配的正box
    _gboxes_mp = gboxes_ltrb_b[boxes_index]
    # 解码
    # gretinas_one.numpy()[np.argsort(gretinas_one.numpy()[:,1])] np某一列排序
    _gtxywh = boxes_encode4retina(cfg, anc_obj, _gboxes_mp, variances=cfg.variances)
    gretinas_one[mask_pos, s_:s_ + 4] = _gtxywh[mask_pos]
    if gkeypoints_b is not None:
        gretinas_one[:, -cfg.NUM_KEYPOINTS:] = gkeypoints_b[boxes_index]

    return gretinas_one


def pos_match_retina4cls(cfg, dim, anc_obj, gboxes_ltrb_b, glabels_b, gkeypoints_b, ptxywh_b=None, img_ts=None):
    '''

    :param cfg:
    :param anc_obj: torch.Size([10647, 4])
    :param gboxes_ltrb_b: torch.Size([xx, 4])
    :param glabels_b: tensor([1, 1, 1])
    :param gkeypoints_b: None
    :param img_ts:
    :param nums_dim_feature: 特图总个数
    :return:
    '''
    anc_ltrb = xywh2ltrb(anc_obj.ancs_xywh)
    # (boxes 个,anc 个) torch.Size([3, 10647])
    iou = calc_iou4ts(gboxes_ltrb_b, anc_ltrb)
    # 除维运行 anc对应box最大的  anc个 bbox_index
    anc_max_iou, boxes_index = iou.max(dim=0)  # 存的是 bboxs的index

    # box对应anc最大的  box个 anc_index
    bbox_max_iou, anc_index = iou.max(dim=1)  # 存的是 anc的index

    # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
    anc_max_iou.index_fill_(0, anc_index, 1)  # 最大的为正例, 强制为conf 为1

    # 一定层度上使gt均有对应的anc, 处理多个anc对一gt的问题, 若多个gt对一个anc仍不能解决(情况较少)会导致某GT学习不了...遍历每一个gt索引
    gt_ids = torch.arange(0, anc_index.size(0), dtype=torch.int64).to(boxes_index)  # [0,1]
    # gt对应最好的anc索引取出来
    anc_ids = anc_index[gt_ids]
    # 找到这些anc 写入gt
    boxes_index[anc_ids] = gt_ids

    # ----------正例的index 和 正例对应的bbox索引----------
    mask_pos = anc_max_iou >= cfg.THRESHOLD_CONF_POS  # 0.5 [10647] anc 的正例 index 不管
    mask_neg = anc_max_iou < cfg.THRESHOLD_CONF_NEG  # 0.4 anc 的反例 index
    mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))

    dim_total, _d4 = ptxywh_b.shape
    device = ptxywh_b.device
    labels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)

    ''' conf-1, cls-3,  与预测对应gt_ltrb-4 ,ious_zg-1 ,'''
    gretinas_one = torch.zeros((dim_total, dim), device=device)  # 返回值

    # 整体匹配 正例全部为1 默认为0
    gretinas_one[:, -1] = anc_max_iou  # ious_zg

    gretinas_one[mask_pos, 0] = torch.tensor(1., device=device)
    gretinas_one[mash_ignore, 0] = torch.tensor(-1., device=device)
    gretinas_one[:, 1] = anc_max_iou  # iou赋值

    # 局部更新
    s_ = 1 + cfg.NUM_CLASSES  # 前面 两个是 conf-1, cls-3,
    gretinas_one[mask_pos, 1:s_] = labels_b[boxes_index][mask_pos].type(torch.float)  # 正例才匹配

    # 匹配的正box
    _gboxes_mp = gboxes_ltrb_b[boxes_index]
    # 解码
    # gretinas_one.numpy()[np.argsort(gretinas_one.numpy()[:,1])] np某一列排序
    _gtxywh = boxes_encode4retina(cfg, anc_obj, _gboxes_mp, variances=cfg.variances)
    gretinas_one[mask_pos, s_:s_ + 4] = _gtxywh[mask_pos]
    if gkeypoints_b is not None:
        gretinas_one[:, -cfg.NUM_KEYPOINTS:] = gkeypoints_b[boxes_index]

    return gretinas_one


def fmatch4yolov1_1(pboxes, boxes_ltrb, labels, num_bbox, num_class, grid, device=None, img_ts=None):
    '''
    一个网格匹配两个对象 只能预测一个类型
    输入一个图片的 target
    :param pboxes:
    :param boxes_ltrb: ltrb
    :param labels:
    :param num_bbox:
    :param num_class:
    :param grid:
    :param device:
    :return:
    '''
    p_yolo_one = torch.zeros((grid, grid, num_bbox * (5 + num_class)), device=device)
    # onehot 只有第一个类别 index 为0
    labels_onehot = labels2onehot4ts(labels - 1, num_class)

    # ltrb -> xywh
    boxes_xywh = ltrb2xywh(boxes_ltrb)
    wh = boxes_xywh[:, 2:]
    cxcy = boxes_xywh[:, :2]

    grids_ts = torch.tensor([grid] * 2, device=device, dtype=torch.int16)
    '''xy与 row col相反'''
    colrow_index = (cxcy * grids_ts).type(torch.int16)  # 网格7的index
    offset_xy = torch.true_divide(colrow_index, grid)  # 网络index 对应归一化的实距
    grid_xy = (cxcy - offset_xy) * grids_ts  # 归一尺寸 - 归一实距 / 网格数 = 相对一格左上角的偏移

    # 遍历格子
    for i, (col, row) in enumerate(colrow_index):
        # 修复xy
        _bbox = pboxes[col, row].clone().detach()
        _bbox[:, :2] = offxy2xy(_bbox[:, :2], torch.tensor((col, row), device=device), grids_ts)
        pbox_ltrb = xywh2ltrb(_bbox)
        ious = calc_iou4some_dim(pbox_ltrb, boxes_ltrb[i][None].repeat(2, 1)).view(-1, num_bbox)
        max_val, max_inx = ious.max(dim=-1)

        # 这里一定是一个gt 的处理 shape4
        offxywh_g = torch.cat([grid_xy[i], wh[i]], dim=0)
        # 正例的conf 和 onehot
        conf = torch.tensor([1], device=device, dtype=torch.int16)
        # 组装正例 yolo_data
        t = torch.cat([offxywh_g, conf, labels_onehot[i]], dim=0)
        idx_start = max_inx[0] * (5 + num_class)
        p_yolo_one[row, col, idx_start:idx_start + 5 + num_class] = t
        # p_yolo_one[row, col, :4] = offxywh
        # p_yolo_one[row, col, 5 + num_class:5 + num_class + 4] = offxywh
        # p_yolo_one[row, col, [4, 4 + 5 + num_class]] = offxywh
        # p_yolo_one[row, col, 5: 5 + num_class] = offxywh
    return p_yolo_one


def fmatch4yolov1_2(boxes_ltrb, labels, num_bbox, num_class, grid, device=None, img_ts=None):
    '''
    一个网格匹配两个对象 预测两个类型
    输入一个图片的 target
    :param boxes_ltrb: ltrb
    :param labels:
    :param num_bbox:
    :param num_class:
    :param grid:
    :param device:
    :return:
    '''
    p_yolo_one = torch.zeros((grid, grid, num_bbox * (5 + num_class)), device=device)
    # onehot 只有第一个类别 index 为0
    labels_onehot = labels2onehot4ts(labels - 1, num_class)

    # ltrb -> xywh
    boxes_xywh = ltrb2xywh(boxes_ltrb)
    wh = boxes_xywh[:, 2:]
    cxcy = boxes_xywh[:, :2]

    grids_ts = torch.tensor([grid] * 2, device=device, dtype=torch.int16)
    '''xy与 row col相反'''
    colrow_index = (cxcy * grids_ts).type(torch.int16)  # 网格7的index
    offset_xy = torch.true_divide(colrow_index, grid)  # 网络index 对应归一化的实距
    grid_xy = (cxcy - offset_xy) * grids_ts  # 归一尺寸 - 归一实距 / 网格数 = 相对一格左上角的偏移

    # 遍历格子
    for i, (col, row) in enumerate(colrow_index):
        offxywh_g = torch.cat([grid_xy[i], wh[i]], dim=0)
        # 正例的conf 和 onehot
        conf = torch.tensor([1], device=device, dtype=torch.int16)
        # 组装正例 yolo_data
        t = torch.cat([offxywh_g, conf, labels_onehot[i]] * 2, dim=0)
        p_yolo_one[row, col] = t
    return p_yolo_one


def fmatch4yolo1_v2(boxes_ltrb, labels, num_bbox, num_class, grid, device=None, img_ts=None):
    '''

    :param boxes_ltrb:
    :param labels:
    :param num_bbox:
    :param num_class:
    :param grid:
    :param device:
    :param img_ts:
    :return: offsetxywh_grid
    '''
    g_boxes_offsetxywh_grid_one = torch.empty((grid, grid, num_bbox * 4), device=device)
    g_confs_one = torch.zeros((grid, grid, num_bbox), device=device)
    g_clses_one = torch.empty((grid, grid, num_class), device=device)

    # onehot 只有第一个类别 index 为0
    labels_onehot = labels2onehot4ts(labels - 1, num_class)

    # ltrb -> xywh
    boxes_xywh = ltrb2xywh(boxes_ltrb)
    cxcy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:]

    grids_ts = torch.tensor([grid] * 2, device=device, dtype=torch.int16)
    '''求 cxcy 所在的格子 相同  xy与 row col相反'''
    colrow_index = (cxcy * grids_ts).type(torch.int16)  # 网格7的index
    offset_xy = torch.true_divide(colrow_index, grid)  # 网络index 对应归一化的实距
    grid_xy = (cxcy - offset_xy) * grids_ts  # 归一尺寸 - 归一实距 / 网格数 = 相对一格左上角的偏移
    # 这里如果有两个GT在一个格子里将丢失
    for i, (col, row) in enumerate(colrow_index):
        offsetxywh_grid = torch.cat([grid_xy[i], wh[i]], dim=0)
        # 正例的conf 和 onehot
        conf2 = torch.ones([2], device=device, dtype=torch.int16)
        g_confs_one[row, col] = conf2
        g_boxes_offsetxywh_grid_one[row, col] = offsetxywh_grid.repeat(2)
        g_clses_one[row, col] = labels_onehot[i]
    # f_plt_od_f(img_ts, boxes_ltrb) # 可视化显示
    return g_boxes_offsetxywh_grid_one, g_confs_one, g_clses_one


def fmatch_OHEM(l_conf, match_index_pos, neg_ratio, num_neg, device, dim=-1):
    '''
    60% 难例作负样本
    '''
    l_conf_neg = l_conf.clone().detach()
    l_conf_neg[match_index_pos] = torch.tensor(0.0, device=device)
    _, lconf_idx = l_conf_neg.sort(dim=dim, descending=True)  # descending 倒序
    # _, lconf_rank = lconf_idx.sort(dim=dim)
    # num_neg = min([neg_ratio * len(match_index_pos), num_neg])  # len(l_conf)
    num_neg = len(match_index_pos) * 1500
    # num_neg = int(len(l_conf_neg) * 0.75)
    match_index_neg = lconf_idx[:num_neg]  # 选出最大的n个的mask  Tensor [batch, 8732]
    return match_index_neg


def match4center(boxes_xywh, labels, fsize, target_center, num_keypoints=0, keypoints=None):
    '''
    这里处理一批的
    :param boxes_xywh: 按输入尺寸归一化 torch device
    :param labels: int torch  这个是从1开始的
    :param fsize: 特图size torch float
    :param target_center: torch.Size([128, 128, 24])
    :param num_keypoints: >0 表示有关键点 1个关键点由两个点构成
    :param keypoints: 关键点xy值
    :return:
    '''
    # 转换到对应特图的尺寸
    boxes_xywh_f = boxes_xywh * fsize.repeat(2)[None]
    # 输入归一化 尺寸
    xys = boxes_xywh_f[:, :2]
    whs = boxes_xywh_f[:, 2:4]
    # 限定为特图格子偏移 与yolo相同
    xys_int = xys.type(torch.int32)
    xys_offset = xys - xys_int  # 这个是偏置在 0~1 之间

    # 使用论文作半径
    radiuses_wh = gaussian_radius(whs, min_overlap=0.7)
    radiuses_wh.clamp_(min=0)
    # [nn] -> [nn,1] -> [nn,2]
    radiuses_wh = radiuses_wh.unsqueeze(-1).repeat(1, 2)

    # 使用真实长宽作半径
    # radiuses_wh = whs * 0.7

    labels_ = labels - 1
    # 根据中心点 和 半径生成 及labels 生成高斯图
    for i in range(len(labels_)):
        # 高斯叠加
        draw_gaussian(target_center[:, :, labels_[i]], xys_int[i], radiuses_wh[i])
        target_center[xys_int[i][1], xys_int[i][0], -4:-2] = xys_offset[i]  # 限定为特图格子偏移
        # target_center[xys_int[i][1], xys_int[i][0], -2:] = boxes_xywh[i][2:4] * fsize  # 按输入归一化
        target_center[xys_int[i][1], xys_int[i][0], -2:] = boxes_xywh[i][2:4]  # 按输入归一化
        if num_keypoints > 0:
            # 得 关键点到中心点的真实偏移
            k = boxes_xywh[i, :2].repeat(num_keypoints) - keypoints[i]
            target_center[xys_int[i][1], xys_int[i][0], -4 - num_keypoints * 2:-4] = k
            pass
