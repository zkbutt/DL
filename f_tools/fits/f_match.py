import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from f_tools.f_general import labels2onehot4ts
from f_tools.fun_od.f_boxes import ltrb2xywh, xywh2ltrb, calc_iou4ts, bbox_iou4one_2d
from f_tools.pic.f_show import f_show_od_np4plt, colormap
from f_tools.yufa.x_calc_adv import f_mershgrid, fcre_gaussian
from object_detection.z_center.nets.utils import gaussian_radius

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
    :param preg: [3, 169*5, 4]  == torch.Size([2, 845, 4])
    :return: 输出原图归一化 [3, 169*5, 4]
    '''
    device = preg.device
    # 特图xy -> 原图 [2, 845, 2] ^ [2, 845,2]
    pxy = torch.sigmoid(preg[..., :2]) \
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


def boxes_decode4yolo2_v2(preg, grid_h, grid_w, cfg):
    '''

    :param preg: torch.Size([2, 845, 4])
    :return: 输出原图归一化 [3, 169*5, 4]
    '''
    device = preg.device
    batch, dim, c = preg.shape
    # 特图xy -> 原图 [2, 845, 2] ^ [845,2]
    pxy = torch.sigmoid(preg[..., :2]) \
          + f_mershgrid(grid_h, grid_w, is_rowcol=False, num_repeat=cfg.NUM_ANC) \
              .to(device)
    pxy = pxy / grid_h  # 原图归一化

    # 原图anc
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    # 比例 ( 比例不需要转换 ) * 特图anc_wh [2, 845, 2] ^ [5,2]
    pwh = torch.exp(preg[..., 2:4]) * ancs_wh_ts.repeat(int(preg.shape[1] / cfg.NUM_ANC),
                                                        1)  # torch.Size([3, 361, 5, 2])
    # fdebug 可视化匹配的anc
    # pwh = ancs_ts.view(1, 1, *ancs_ts.shape).repeat(*ptxywh[:, :, :, 2:4].shape[:2], 1, 1)

    pxywh = torch.cat([pxy, pwh], -1)  # torch.Size([3, 169, 5, 4])
    pxywh = pxywh.view(preg.shape[0], -1, 4)  # 原图归一化 [3, 169, 5, 4] -> [3, 169*5, 4]
    pltrb = xywh2ltrb(pxywh)
    return pltrb


'''------------------------------- YOLO3 编解码--------------------------'''


def boxes_decode4yolo3(ptxywh, cfg, is_match=False, mask_pos=None):
    '''
    :param ptxywh: torch.Size([3, 10647, 4]) 不要归一化
    :param cfg:
        cfg.NUMS_CENG [2704, 676, 169]
        cfg.NUMS_ANC [3, 3, 3]
    :return: 输出原图归一化 [3, 10647, 4]
    '''
    device = ptxywh.device
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)  # torch.Size([9, 2])

    ptxywh_ = ptxywh.clone().detach()
    index_colrow = []
    ancs_wh_match = []
    grids = []  # 用于原图归一化
    _s = 0
    for i, num_ceng in enumerate(cfg.NUMS_CENG):
        num_ceng_t = num_ceng * cfg.NUMS_ANC[i]  # 2704*3 = 8112
        _grid = int(math.sqrt(num_ceng))
        #  52 26 13   形成8112个52
        grids.append(torch.tensor(_grid, device=device, dtype=torch.int32).repeat(num_ceng_t))

        _grids = f_mershgrid(_grid, _grid, is_rowcol=False, num_repeat=cfg.NUMS_ANC[i]).to(device)
        index_colrow.append(_grids)  # 每层对应的colrow索引 torch.Size([8112, 2])

        _ancs_wh_match = ancs_wh_ts[_s:_s + cfg.NUMS_ANC[i]]  # 选出层对应的 anc
        _ancs_wh_match = _ancs_wh_match.repeat(num_ceng, 1)  # torch.Size([8112, 2])
        ancs_wh_match.append(_ancs_wh_match)
        _s += cfg.NUMS_ANC[i]
    # torch.Size([8112, 2])  torch.Size([2028, 2])  torch.Size([507, 2])
    index_colrow = torch.cat(index_colrow, 0)  # torch.Size([10647, 2])
    ancs_wh_match = torch.cat(ancs_wh_match, 0)  # torch.Size([10647, 2])
    grids = torch.cat(grids, -1)  # 10647

    if is_match:
        pxy = ptxywh_[:, :, :2] + index_colrow
        pwh = ptxywh_[:, :, 2:4] * ancs_wh_match
    else:
        pxy = ptxywh_[:, :, :2].sigmoid() + index_colrow
        pwh = ptxywh_[:, :, 2:4].exp() * ancs_wh_match
    pxy = pxy / grids.view(1, -1, 1)
    pxywh = torch.cat([pxy, pwh], -1)  # torch.Size([32, 10647, 4])
    pltrb = xywh2ltrb(pxywh)
    return pltrb


def boxes_decode4yolo5(ptxywh, cfg):
    '''
    :param ptxywh: torch.Size([3, 10647, 4]) 不要归一化
    :param cfg:
        cfg.NUMS_CENG [2704, 676, 169]
        cfg.NUMS_ANC [3, 3, 3]
    :return: 输出原图归一化 [3, 10647, 4]
    '''
    device = ptxywh.device
    # (9,2) -> (3,3,2)
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    ancs_wh_ts = ancs_wh_ts.reshape(3, 3, 2)
    # torch.Size([10647, 2])
    ancs_p = torch.tensor(np.repeat(ancs_wh_ts.cpu().numpy(), cfg.NUMS_CENG, axis=0), device=device).reshape(-1, 2)

    ptxywh_sigmoid = ptxywh.sigmoid()

    _grids = [52, 26, 13]  # 这个是写死的
    # [3] -> [10647]
    _grids_p = torch.tensor(np.repeat(np.array(_grids), np.array(cfg.NUMS_CENG) * 3, axis=0), device=device)
    grids = []
    for i, num_ceng in enumerate(cfg.NUMS_CENG):
        _gp = f_mershgrid(_grids[i], _grids[i], is_rowcol=False, num_repeat=cfg.NUMS_ANC[i]).to(device)
        grids.append(_gp)
    # torch.Size([10647, 2])
    grid_ts_p = torch.cat(grids, 0)

    pxy = (ptxywh_sigmoid[..., :2] * 2) - 0.5
    # ([3, 10647, 2] + [10647, 2])/[1,10647, 1]
    pxy = (pxy + grid_ts_p) / _grids_p.view(1, -1, 1)
    pwh = (ptxywh_sigmoid[..., 2:4] * 2) ** 2 * ancs_p
    pxywh = torch.cat([pxy, pwh], -1)
    pltrb = xywh2ltrb(pxywh)
    return pltrb


'''------------------------------- SSD 编解码--------------------------'''


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


'''------------------------------- retina 编解码--------------------------'''


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


def boxes_decode4retina(cfg, anc_obj, ptxywh):
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


'''------------------------------- center 编解码--------------------------'''


def boxes_encode4center(cfg, anc_obj, gboxes_ltrb):
    return None


def boxes_decode4center(cfg, fsize_wh, ptxywh):
    ''' 解码 '''
    _pxywh = torch.zeros_like(ptxywh)
    device = ptxywh.device
    _grids = f_mershgrid(fsize_wh[0], fsize_wh[1], is_rowcol=False, num_repeat=1).to(device)
    _pxywh[:, :, :2] = (torch.sigmoid(ptxywh[:, :, :2]) + _grids) / fsize_wh[0]
    _pxywh[:, :, 2:] = torch.exp(ptxywh[:, :, 2:]) / fsize_wh[0]
    zltrb = xywh2ltrb(_pxywh)
    return zltrb


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
        ious_zg = bbox_iou4one_2d(boxes_t_ltrb, gboxes_t_ltrb_match, is_ciou=True)  # [3614]
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
        ious_zg = bbox_iou4one_2d(boxes_ltrb, gboxes_ltrb_match, is_ciou=True)  # [3614]
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
        if True:
            bbox_max_iou, anc_index = ious.max(dim=0)  # 存的是 anc的index
            ''' 强制确保每个GT对应最大的anc 这个可能会引起噪音'''
            anc_max_iou.index_fill_(0, anc_index, 1)  # 最大的为正例, 强制为conf 为1
            ''' 一定层度上使gt均有对应的anc, 处理多个anc对一gt的问题, 若多个gt对一个anc仍不能解决(情况较少)会导致某GT学习不了...遍历每一个gt索引 '''
            gt_ids = torch.arange(0, anc_index.size(0), dtype=torch.int64).to(boxes_index)  # [0,1]
            anc_ids = anc_index[gt_ids]
            boxes_index[anc_ids] = gt_ids
            mask_pos = anc_max_iou >= cfg.THRESHOLD_CONF_POS  # [10647] anc 的正例 index 不管
            mask_neg = anc_max_iou < cfg.THRESHOLD_CONF_NEG  # anc 的反例 index
            mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
            # mask_neg 负例自动0
        else:
            # fastrcnn 负样本分区间平均采样算法
            pass

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
    _gtxywh = boxes_encode4retina(cfg, anc_obj, _gboxes_mp)
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


def fmatch4yolov2_99(gboxes_ltrb_b, glabels_b, grid, gdim, device, cfg, preg_b, img_ts=None, val_iou=0.3):
    '''
    1. 计算GT 与anc的iou
        小于阀值则强制匹配一个iou最大的正例
        大于阀值选一个最好的赋为正例, 其它大于阀值的忽略
        conf 反倒为0 正例为1 忽略为-1
    # conf-1, cls-1, txywh-4, weight-1, gltrb-4
    :param gboxes_ltrb_b: ltrb
    :param glabels_b:
    :param grid: 13
    :param gdim:
    :param device:
    :return: 匹配中心与GT相同 iou 最大的一个anc  其余的全为0
    '''

    '''与yolo1相比多出确认哪个anc大 计算GT的wh与anc的wh 确定哪个anc的iou大'''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    anc_wh = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = anc_wh

    # 匹配一个最大的 用于获取iou index (n,4)^(5,4) -> (n,5)
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    # flog.debug('iou2d %s', iou2d)

    mask_ps = iou2d > val_iou

    # [gt,anc] GT对应格子中哪个最大 这个是必须要求的
    # index_p = iou2.max(-1)[1]  # 匹配最大的IOU
    ids_p = torch.argmax(iou2d, dim=-1)  # 匹配最大的IOU
    # ------------------------- yolo2 与yolo1一样只有一层 编码wh是比例 ------------------------------
    txywhs_g, weights, colrows_index = boxes_encode4yolo2_4iou(
        gboxes_ltrb_b=gboxes_ltrb_b,
        preg_b=preg_b,
        match_anc_ids=ids_p,  # 匹配的iou最大的anc索引
        grid_h=grid, grid_w=grid,
        device=device, cfg=cfg,
    )

    # conf-1, cls-1, txywh-4, weight-1, gltrb-4
    g_yolo_one = torch.zeros((grid, grid, cfg.NUM_ANC, gdim), device=device)

    # glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    # ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)

    # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
    for i, (col, row) in enumerate(colrows_index):
        conf = torch.tensor([1], device=device)
        if torch.any(mask_ps[i]):
            # 只要有一个匹配,则把其它匹配的置忽略
            g_yolo_one[row, col, mask_ps[i]] = -1

        # 如果一个都没有 则用一个最大的 [10]  最大的匹配
        _t = torch.cat([conf, glabels_b[i].unsqueeze(0) - 1, txywhs_g[i], weights[i].unsqueeze(0), gboxes_ltrb_b[i]],
                       dim=0)
        g_yolo_one[row, col, ids_p[i]] = _t

    return g_yolo_one


def fmatch4yolov3(gboxes_ltrb_b, glabels_b, dim, ptxywh_b, device, cfg, img_ts=None, pconf_b=None):
    '''
    只取最大一个IOU为正例
    :param gboxes_ltrb_b:
    :param glabels_b:
    :param dim:
    :param ptxywh_b: torch.Size([3, 10647, 4])
    :param device:
    :param cfg:
    :param img_ts:
    :return:
    '''
    ''' 只有wh计算与多层anc的IOU '''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = ancs_wh_ts

    # 匹配一个最大的anc 用于获取iou index  iou>0.5 忽略
    # ngt,4 ^^ 9,4 ->  ngt,9
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    '''取iou最大的一个anc匹配'''
    # 索引 0~8
    # iou_max_val, iou_max_index = iou2d.max(1) # 等价于下式
    ids_p_anc = torch.argmax(iou2d, dim=-1)  # 匹配最大的IOU
    # ------------------------- yolo23一样 ------------------------------

    # 匹配完成的数据 [2704, 676, 169] 层数量*anc数
    _dim_total = sum(cfg.NUMS_CENG) * 3  # 10647
    _num_anc_total = len(cfg.ANCS_SCALE)  # 9个
    # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
    g_yolo_one = torch.zeros((_dim_total, dim), device=device)  # 返回值

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    '''回归参数由于每层的grid不一样,需要遍历'''

    whs = gboxes_xywh[:, 2:]  # 归一化值
    weights = 2.0 - torch.prod(whs, dim=-1)

    grids = [52, 26, 13]
    t_ceng = [0, 2704, 676, 169]  # [0, 2704, 676, 169]

    '''遍历GT 匹配iou最大的  其它anc全部忽略'''
    for i in range(len(gboxes_ltrb_b)):
        cxy = gboxes_xywh[i][:2]
        index_p_anc = ids_p_anc[i]
        # 最大IOU匹配到哪一层 每层的anc数  8/3 向下取整 = 0~2    写死每层3个anc
        index_p_ceng = torch.true_divide(index_p_anc, cfg.NUMS_ANC[0]).type(torch.int32)
        index_anc_off = index_p_anc % cfg.NUMS_ANC[index_p_ceng]  # 0~2
        offset_ceng = torch.tensor(t_ceng, dtype=torch.int32)[:index_p_ceng + 1].sum() * cfg.NUMS_ANC[index_p_ceng]

        grid = grids[index_p_ceng]
        grids_ts = torch.tensor([grid, grid], device=device, dtype=torch.int32)
        index_colrow = (cxy * grids_ts).type(torch.int32)
        col, row = index_colrow
        offset_colrow = (row * grid + col) * cfg.NUMS_ANC[index_p_ceng]

        # cfg.NUMS_ANC[0] 每层anc数是一样的 索引要int
        _index_z = (offset_ceng + offset_colrow + index_anc_off).long()
        # 该层其它的anc忽略掉
        g_yolo_one[_index_z:_index_z + cfg.NUMS_ANC[index_p_ceng], 0] = -1

        conf = torch.tensor([1], device=device)

        # 编码  归一化网格左上角位置 用于原图归一化中心点减 求偏差
        offset_xy = torch.true_divide(index_colrow, grids_ts[0])
        # (真xy - 原图偏网络) -> 原图偏移 *grid  -> 一定在0~1之间
        txy_g = (cxy - offset_xy) * grids_ts  # 特图偏移
        anc_match_ts = ancs_wh_ts[index_p_anc]
        # 比例 /log
        twh_g = (gboxes_xywh[i][2:] / anc_match_ts).log()
        txywh_g = torch.cat([txy_g, twh_g], dim=-1)

        _t = torch.cat([conf, glabels_b[i], txywh_g, weights[i][None], gboxes_ltrb_b[i]], dim=0)
        g_yolo_one[_index_z] = _t  # 匹配到的正例

        if cfg.IS_VISUAL:
            ''' 可视化匹配最大的ANC '''
            from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            _img_ts = f_recover_normalization4ts(img_ts.clone())
            from torchvision.transforms import functional as transformsF
            img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
            import numpy as np
            img_np = np.array(img_pil)

            mask_pos = (g_yolo_one[:, 0] == 1).unsqueeze(0)
            anc_p = boxes_decode4yolo3(g_yolo_one[:, 4:8].unsqueeze(0), cfg, is_match=True, mask_pos=mask_pos)
            anc_p = anc_p[mask_pos]

            anc_o = torch.cat([cxy.repeat(ancs_wh_ts.shape[0], 1), ancs_wh_ts], dim=-1)
            f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b[i][None].cpu()
                             , pboxes_ltrb=xywh2ltrb(anc_p.cpu()),
                             other_ltrb=xywh2ltrb(anc_o.cpu()),
                             is_recover_size=True)

    return g_yolo_one


def fmatch4yolov3_iou(gboxes_ltrb_b, glabels_b, dim, ptxywh_b, device, cfg, img_ts=None, pconf_b=None, val_iou=0.3):
    '''

    :param gboxes_ltrb_b:
    :param glabels_b:
    :param dim:
    :param ptxywh_b: torch.Size([3, 10647, 4])
    :param device:
    :param cfg:
    :param img_ts:
    :return:
    '''
    ''' 只有wh计算与多层anc的IOU '''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = ancs_wh_ts

    # 匹配一个最大的anc 用于获取iou index  iou>0.5 忽略
    # ngt,4 ^^ 9,4 ->  ngt,9
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    '''取iou最大的一个anc匹配'''
    # mask_ps = iou2d > 0.3
    mask_ps = iou2d > val_iou

    # 匹配完成的数据 [2704, 676, 169] 层数量*anc数
    _dim_total = sum(cfg.NUMS_CENG) * 3  # 10647
    _num_anc_total = len(cfg.ANCS_SCALE)  # 9个
    # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
    g_yolo_one = torch.zeros((_dim_total, dim), device=device)  # 返回值

    ''' 这里开启强制匹配 '''
    ids_p_anc = torch.argmax(iou2d, dim=-1)
    mask_ps.index_fill_(1, ids_p_anc, True)
    # if not torch.any(mask_ps):
    #     ids_p_anc = torch.argmax(iou2d, dim=-1)
    #     mask_ps.index_fill_(1, ids_p_anc, True)
    # flog.warning('没有目标 %s', )

    ''' 这里用于容错 先忽略 '''
    # if not torch.any(mask_ps):
    #     flog.warning('没有目标 %s', )
    #     if cfg.IS_VISUAL:
    #         ''' 可视化匹配最大的ANC '''
    #         from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
    #         _img_ts = f_recover_normalization4ts(img_ts.clone())
    #         from torchvision.transforms import functional as transformsF
    #         img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
    #         import numpy as np
    #         img_np = np.array(img_pil)
    #         # 匹配的 anc 多个
    #         for gt in gboxes_ltrb_b:
    #             f_show_od_np4plt(img_np, gboxes_ltrb=gt[None].cpu(),
    #                              # , pboxes_ltrb=xywh2ltrb(anc_p.cpu()),
    #                              # other_ltrb=xywh2ltrb(anc_o.cpu()),
    #                              is_recover_size=True)
    #     return g_yolo_one

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    '''回归参数由于每层的grid不一样,需要遍历'''

    whs = gboxes_xywh[:, 2:]  # 归一化值
    weights = 2.0 - torch.prod(whs, dim=-1)

    grids = [52, 26, 13]
    t_ceng = [0, 2704, 676, 169]  # [0, 2704, 676, 169]

    '''遍历GT 匹配iou最大的  其它anc全部忽略'''
    for i in range(len(gboxes_ltrb_b)):
        cxy = gboxes_xywh[i][:2]
        indexs_p_anc = torch.where(mask_ps[i])[0]  # 核心是这个函数无法批量计算

        ''' 这里用于容错 先忽略 '''
        # if len(indexs_p_anc) == 0:
        #     flog.warning('没有合适的anc %s', )
        #     if cfg.IS_VISUAL:
        #         ''' 可视化匹配最大的ANC '''
        #     from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        #     _img_ts = f_recover_normalization4ts(img_ts.clone())
        #     from torchvision.transforms import functional as transformsF
        #     img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        #     import numpy as np
        #     img_np = np.array(img_pil)
        #     # 匹配的 anc 多个
        #     # anc_p = torch.cat([cxy.unsqueeze(0).repeat(len(anc_match_ts), 1), anc_match_ts], dim=-1)
        #     # anc_o = torch.cat([cxy.repeat(ancs_wh_ts.shape[0], 1), ancs_wh_ts], dim=-1)
        #     f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b[i][None].cpu(),
        #                      # pboxes_ltrb=xywh2ltrb(anc_p.cpu()),
        #                      # other_ltrb=xywh2ltrb(anc_o.cpu()),
        #                      is_recover_size=True)
        #     continue

        # 最大IOU匹配到哪一层 每层的anc数  8/3 向下取整 = 0~2    写死每层3个anc
        _t = torch.tensor([cfg.NUMS_ANC[0]] * len(indexs_p_anc), device=device)
        indexs_p_ceng = torch.true_divide(indexs_p_anc, _t).type(torch.long)
        nums_anc = torch.tensor(cfg.NUMS_ANC, device=device)[indexs_p_ceng]
        indexs_anc_off = indexs_p_anc % nums_anc  # 0~2

        offsets_ceng = []
        for j in indexs_p_ceng:
            _t = torch.tensor(t_ceng, dtype=torch.int32)[:j + 1].sum() * cfg.NUMS_ANC[j]
            offsets_ceng.append(_t)
        offsets_ceng = torch.tensor(offsets_ceng, device=device)

        # n -> n,1 -> n,2
        grid = torch.tensor(grids, device=device)[indexs_p_ceng]
        grids_ts = grid.unsqueeze(-1).repeat(1, 2)

        indexs_colrow = (cxy.unsqueeze(0) * grids_ts).type(torch.int32)
        row = indexs_colrow[:, 1]
        col = indexs_colrow[:, 0]
        offsets_colrow = (row * grid + col) * nums_anc

        # cfg.NUMS_ANC[0] 每层anc数是一样的 索引要int
        _index_z = (offsets_ceng + offsets_colrow + indexs_anc_off).long()
        # 该层其它的anc忽略掉
        # g_yolo_one[_index_z:_index_z + cfg.NUMS_ANC[indexs_p_ceng], 0] = -1

        conf = torch.tensor([1], device=device)

        # 编码  归一化网格左上角位置 用于原图归一化中心点减 求偏差
        # n,2 ^^ n
        offset_xy = torch.true_divide(indexs_colrow, grids_ts[:, 0].unsqueeze(-1))
        # (真xy - 原图偏网络) -> 原图偏移 *grid  -> 一定在0~1之间
        txy_g = (cxy.unsqueeze(0) - offset_xy) * grids_ts  # 特图偏移
        anc_match_ts = ancs_wh_ts[indexs_p_anc]
        # 比例 /log
        twh_g = (gboxes_xywh[i][2:].unsqueeze(0) / anc_match_ts).log()
        txywh_g = torch.cat([txy_g, twh_g], dim=-1)

        for _i, k in enumerate(txywh_g):
            _t = torch.cat([conf, glabels_b[i], k, weights[i][None], gboxes_ltrb_b[i]], dim=0)
            g_yolo_one[_index_z[_i]] = _t  # 匹配到的正例

        if cfg.IS_VISUAL:
            ''' 可视化匹配最大的ANC '''
            from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            _img_ts = f_recover_normalization4ts(img_ts.clone())
            from torchvision.transforms import functional as transformsF
            img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
            import numpy as np
            img_np = np.array(img_pil)
            # torch.Size([1, 10647, 4])
            mask_pos = (g_yolo_one[:, 0] == 1).unsqueeze(0)
            anc_p = boxes_decode4yolo3(g_yolo_one[:, 4:8].unsqueeze(0), cfg, is_match=True, mask_pos=mask_pos)
            anc_p = anc_p[mask_pos]

            # anc_p = torch.cat([cxy.unsqueeze(0).repeat(len(anc_match_ts), 1), anc_match_ts], dim=-1)
            anc_o = torch.cat([cxy.repeat(ancs_wh_ts.shape[0], 1), ancs_wh_ts], dim=-1)
            f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b[i][None].cpu()
                             , pboxes_ltrb=xywh2ltrb(anc_p.cpu()),
                             other_ltrb=xywh2ltrb(anc_o.cpu()),
                             is_recover_size=True)

    return g_yolo_one


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


def match4center(gboxes_ltrb_b, glabels_b, fsize_wh, dim, cfg, num_keypoints=0, gkeypoints=None, ):
    '''
    这里处理一批的
    :param gboxes_xywh_b: 按输入尺寸归一化 torch device
    :param glabels_b: int torch  这个是从1开始的
    :param fsize_wh: 特图size torch float
    :param num_keypoints: >0 表示有关键点 1个关键点由两个点构成
    :param gkeypoints: 关键点xy值
    :return:
    '''
    gboxes_xywh_b = ltrb2xywh(gboxes_ltrb_b)
    # 转换到对应特图的尺寸
    device = gboxes_xywh_b.device
    boxes_xywh_t = gboxes_xywh_b * fsize_wh.repeat(2).unsqueeze(0)
    # 输入特图 尺寸
    xys = boxes_xywh_t[:, :2]
    whs = boxes_xywh_t[:, 2:4]
    # 限定为特图格子偏移 与yolo相同
    indexs_colrow = xys.type(torch.int32)

    txy = xys - indexs_colrow  # 这个是偏置在 0~1 之间 与 yolo 一样偏置
    twh = torch.log(whs)
    txywh = torch.cat([txy, twh], -1)
    ''' 用这个判断回归正例 >0'''
    weights = torch.ones(boxes_xywh_t.shape[0])  # 有几个GT就创建几个 写死为1
    # weights = 2.0 - torch.prod(gboxes_xywh_b[:,2:4], dim=-1)

    ''' 这里用圆的效果没有影响 '''
    # 使用论文作半径  [ngt]
    radiuses = gaussian_radius(whs, min_overlap=0.7)
    # [ngt] -> [ngt,1] -> [ngt,2]
    r = radiuses.unsqueeze(-1).repeat(1, 2)
    ''' 这个为0 则是硬标签如yolo '''
    sigma_wh = torch.true_divide(r, torch.full_like(r, 3, device=device))
    # sigmawh = whs / 6

    # 使用真实长宽作半径
    # radiuses_wh = whs * 0.7

    labels_ = (glabels_b - 1).long()
    # 矩阵是先行后列
    g_res_one = torch.zeros((fsize_wh[0], fsize_wh[1], dim), device=device)

    # 根据中心点 和 半径生成 及labels 生成高斯图
    for i in range(len(labels_)):
        ''' 同一个格子有多个目标 只有一个有效  一个目标只有一个点有效 '''
        # 每一张图的 张一个标签
        g_res_one[indexs_colrow[i, 1], indexs_colrow[i, 0], labels_[i]] = 1
        g_res_one[indexs_colrow[i, 1], indexs_colrow[i, 0], cfg.NUM_CLASSES:cfg.NUM_CLASSES + 4] = txywh[i]
        g_res_one[indexs_colrow[i, 1], indexs_colrow[i, 0], cfg.NUM_CLASSES + 4] = weights[i]  # 这个判断回归正例

        # 软件阀值 标签是高斯范围
        heatmap = fcre_gaussian(index_colrow=indexs_colrow[i], fsize_wh=fsize_wh, sigma_wh=sigma_wh[i])
        pre_heatmap = g_res_one[:, :, labels_[i]]  # 该标签维度全部提出后取最大
        g_res_one[:, :, labels_[i]] = torch.max(heatmap, pre_heatmap)

    return g_res_one


def match4fcos(gboxes_ltrb_b, glabels_b, gdim, pcos, cfg, img_ts=None, ):
    '''
    这里处理一批的
    :img_ts: 一张图片
    :return:
    '''

    gboxes_xywh_b = ltrb2xywh(gboxes_ltrb_b)
    # 转换到对应特图的尺寸
    device = gboxes_ltrb_b.device
    dim_total = pcos.shape[1]
    labels_ = (glabels_b - 1).long()

    #  back cls centerness ltrb positivesample iou
    g_res_one = torch.zeros([dim_total, gdim], device=device)

    input_size = torch.tensor(cfg.IMAGE_SIZE, device=device)
    grids = torch.true_divide(input_size.unsqueeze(0),
                              torch.tensor(cfg.STRIDES, device=device).unsqueeze(-1).repeat(1, 2))
    grids = grids[:, 0].type(torch.int32).tolist()

    # 恢复到input
    gboxes_ltrb_b_input = gboxes_ltrb_b * input_size.repeat(2).unsqueeze(0)

    _s = 1 + cfg.NUM_CLASSES
    g_res_one[:, 0] = 1.0  # 默认全部为背景
    g_res_one[:, _s + 1 + 4 + 1 + 1] = 1.0

    # 遍历每一个标签, 的每一层的格子  找出格子是否在预测框中, 并记录差异
    for i in range(len(labels_)):
        l = gboxes_ltrb_b_input[i, 0]
        t = gboxes_ltrb_b_input[i, 1]
        r = gboxes_ltrb_b_input[i, 2]
        b = gboxes_ltrb_b_input[i, 3]
        start_index = 0
        for j in range(len(cfg.STRIDES)):

            for row in range(grids[j]):
                for col in range(grids[j]):
                    ''' 特图格子中心点 -> 原图格子中心点的映射 //是取下界 '''
                    x = col * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    y = row * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    if x >= l and x <= r and y >= t and y <= b:
                        # back cls center-ness tltrb positive iou
                        off_l = x - l  # 特图尺寸
                        off_t = y - t
                        off_r = r - x
                        off_b = b - y
                        M = max(off_l, off_t, off_r, off_b)
                        # 满足条件还需要确认是不是在这层的宽度中 [0, 49, 98, 196, 10000000000.0]
                        if M >= cfg.SCALE_THRESHOLDS[j] and M < cfg.SCALE_THRESHOLDS[j + 1]:
                            ''' 同一GT可能在不同的层中预测 '''
                            index = (row * grids[j] + col) + start_index

                            ''' 在本层中如果已匹配取面积最小的 前面默认已最大 '''
                            area = torch.prod(gboxes_xywh_b[i][2:])
                            if area >= g_res_one[index, _s + 1 + 4 + 1 + 1]:
                                # 已匹配点面积大的不要
                                continue

                            center_ness = torch.sqrt((torch.min(off_l, off_r) / torch.max(off_l, off_r))
                                                     * (torch.min(off_t, off_b) / torch.max(off_t, off_b)))
                            # 有可能重叠目标  避免框的类别错误
                            g_res_one[index, :_s] = torch.zeros(_s)  # 背景置0
                            # labels_ 1开始 需要减 1
                            g_res_one[index, 1 + labels_[i] - 1] = 1.
                            g_res_one[index, _s] = center_ness
                            g_res_one[index, _s + 1:_s + 1 + 4] = torch.stack([off_l, off_t, off_r, off_b])
                            g_res_one[index, _s + 1 + 4] = 1.
                            g_res_one[index, _s + 1 + 4 + 1] = 1.
                            g_res_one[index, _s + 1 + 4 + 1 + 1] = area  # 面积值

            start_index += (grids[j] ** 2)

    if cfg.IS_VISUAL:
        ''' 可视化匹配最大的ANC '''
        from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        _img_ts = f_recover_normalization4ts(img_ts.clone())
        from torchvision.transforms import functional as transformsF
        img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        import numpy as np
        img_np = np.array(img_pil)

        CLASS_COLOR = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
                       range(cfg.NUM_CLASSES)]

        start_index = 0
        for j in range(len(cfg.STRIDES)):

            for row in range(grids[j]):
                for col in range(grids[j]):
                    # 网络在原图的坐标
                    x = col * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    y = row * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    index = (row * grids[j] + col) + start_index
                    if g_res_one[index, _s + 1 + 4 + 1] == 1.:
                        off_l, off_t, off_r, off_b = g_res_one[index, _s + 1:_s + 1 + 4]
                        # 网络位置 求GT的位置
                        xmin = int(x - off_l)
                        ymin = int(y - off_t)
                        xmax = int(x + off_r)
                        ymax = int(y + off_b)

                        gcls = np.argmax(g_res_one[index, 1:_s], axis=-1)
                        mess = '%s' % (int(gcls))
                        cv2.circle(img_np, (int(x), int(y)), 5, CLASS_COLOR[int(gcls)], -1)
                        cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), CLASS_COLOR[int(gcls)], 2)
                        cv2.rectangle(img_np, (int(xmin), int(abs(ymin) - 15)),
                                      (int(xmin + (xmax - xmin) * 0.55), int(ymin)), CLASS_COLOR[int(gcls)], -1)
                        cv2.putText(img_np, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            start_index += (grids[j] ** 2)
        cv2.imshow('image', img_np)
        cv2.waitKey(0)
    return g_res_one


def match4fcos_v2(gboxes_ltrb_b, glabels_b, gdim, pcos, cfg, img_ts=None, ):
    '''
    这里处理一张图片
    :img_ts: 一张图片
    :return:
        v1 0.11269640922546387
        v2 0.001  提升1000倍
    '''

    gboxes_xywh_b = ltrb2xywh(gboxes_ltrb_b)
    # 转换到对应特图的尺寸
    device = gboxes_ltrb_b.device
    dim_total = pcos.shape[1]
    # labels_ = (glabels_b - 1).long()

    # back cls centerness ltrb positivesample iou area
    g_res_one = torch.zeros([dim_total, gdim], device=device)

    input_size = torch.tensor(cfg.IMAGE_SIZE, device=device)
    grids = torch.true_divide(input_size.unsqueeze(0),
                              torch.tensor(cfg.STRIDES, device=device).unsqueeze(-1).repeat(1, 2))
    grids = grids[:, 0].type(torch.int32).tolist()

    # 恢复到input
    gboxes_ltrb_b_input = gboxes_ltrb_b * input_size.repeat(2).unsqueeze(0)
    gboxes_xywh_b_input = ltrb2xywh(gboxes_ltrb_b_input)

    _s = 1 + cfg.NUM_CLASSES
    g_res_one[:, 0] = 1.0  # 默认全部为背景
    g_res_one[:, _s + 1 + 4 + 1 + 1] = 1.0  # 面积初始为最大1

    # back = torch.ones((dim_total, 1), device=device)
    positive_sample = torch.zeros((dim_total, 1), device=device)  # 半径正例
    # (4) -> (1,4) ->  (nn,4)  # 这里多加了一个背景
    labels_onehot = torch.zeros((dim_total, _s), device=device)  # 全为背景置1
    # labels_onehot[:, 0] = 1  # 全为背景置1

    # 遍历每一个标签, 的每一层的格子  找出格子是否在预测框中, 并记录差异
    for i in range(len(glabels_b)):
        l = gboxes_ltrb_b_input[i, 0]
        t = gboxes_ltrb_b_input[i, 1]
        r = gboxes_ltrb_b_input[i, 2]
        b = gboxes_ltrb_b_input[i, 3]
        area = torch.prod(gboxes_xywh_b[i][2:])

        index_colrow = []
        scale_thresholds = []
        radius = []

        image_size_ts = torch.tensor(cfg.IMAGE_SIZE, device=device)
        for j, s in enumerate(cfg.STRIDES):
            # 恢复到
            grid_wh = image_size_ts // s  # 特征格子
            _grids = f_mershgrid(grid_wh[1], grid_wh[0], is_rowcol=False).to(device)
            _grids = _grids * s + s // 2
            # dim, _ = _grids.shape
            index_colrow.append(_grids)

            _scale = torch.empty_like(_grids, device=device)
            _scale[:, 0] = cfg.SCALE_THRESHOLDS[j]
            _scale[:, 1] = cfg.SCALE_THRESHOLDS[j + 1]
            scale_thresholds.append(_scale)

            # [nn]
            _radius = torch.empty_like(_grids[:, 0], device=device)
            _radius[:] = cfg.MATCH_RADIUS * s  # 半径阀值
            radius.append(_radius)

        # (nn,2)
        index_colrow = torch.cat(index_colrow, 0)
        scale_thresholds = torch.cat(scale_thresholds, 0)
        radius = torch.cat(radius, 0)  # [nn]

        # (nn) # --- 是否框内条件 ---
        mask_col_lr = torch.logical_and(index_colrow[:, 0] >= l, index_colrow[:, 0] <= r)
        mask_row_tb = torch.logical_and(index_colrow[:, 1] >= t, index_colrow[:, 1] <= b)
        mask_D = torch.logical_and(mask_col_lr, mask_row_tb)

        # --- 中心格子半径条件 ---
        mask_radius = torch.logical_and(torch.abs(index_colrow[:, 0] - gboxes_xywh_b_input[i, 0]) < radius,
                                        torch.abs(index_colrow[:, 1] - gboxes_xywh_b_input[i, 1]) < radius)

        # (nn,2)
        off_lt = index_colrow - gboxes_ltrb_b_input[i, :2].unsqueeze(0)
        off_rb = gboxes_ltrb_b_input[i, 2:].unsqueeze(0) - index_colrow
        off_ltrb = torch.cat([off_lt, off_rb], -1)  # (nn,4)
        M, _ = torch.max(off_ltrb, -1)  # (nn,4) -> (nn)

        # (nn)  # --- 层阀值条件 ---
        mask_M = torch.logical_and(M >= scale_thresholds[:, 0], M <= scale_thresholds[:, 1])

        # --- 面积条件 ---
        mask_area = g_res_one[:, _s + 1 + 4 + 1 + 1] > area

        mask = torch.logical_and(torch.logical_and(mask_D, mask_M), mask_area)
        mask_radius = torch.logical_and(mask, mask_radius)

        # (nn) -> (nn,1)
        center_ness = torch.sqrt(torch.min(off_ltrb[:, ::2], -1)[0] / torch.max(off_ltrb[:, ::2], -1)[0]
                                 * torch.min(off_ltrb[:, 1::2], -1)[0] / torch.max(off_ltrb[:, 1::2], -1)[0])
        center_ness.unsqueeze_(-1)

        '''
        back 默认为1 有类别即付为 onehot 复写  -> back=0
        centerness 用于与回归 loss 权重
        ltrb 用于IOU计算
        positive_sample 用于计算conf 需结合半径
        
        '''
        labels_onehot[mask, 1 + glabels_b[i].long() - 1] = 1.  # 这个需要保留以前的值 本次只复写需要的

        positive_sample[mask_radius] = 1  # 这是一个较小的交集, 需要的付为1 用于 conf 计算
        area_ts = torch.empty_like(positive_sample)  # (nn,1)
        area_ts[:, 0] = area  # 面积更新 全部付这个GT的面积 后面通过 mask 过滤

        # back cls centerness ltrb positivesample iou(这个暂时无用) area [2125, 12]
        # g_res_one_tmp = torch.cat([labels_onehot, center_ness, off_ltrb, one_ts, one_ts, area_ts], -1)
        g_res_one_tmp = torch.cat([labels_onehot, center_ness, gboxes_ltrb_b[i].repeat(dim_total, 1),
                                   positive_sample, positive_sample, area_ts], -1)
        g_res_one[mask] = g_res_one_tmp[mask]

    if cfg.IS_VISUAL:
        ''' 可视化匹配最大的ANC '''
        from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        _img_ts = f_recover_normalization4ts(img_ts.clone())
        from torchvision.transforms import functional as transformsF
        img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        import numpy as np
        img_np = np.array(img_pil)

        CLASS_COLOR = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
                       range(cfg.NUM_CLASSES)]

        start_index = 0
        for j in range(len(cfg.STRIDES)):

            for row in range(grids[j]):
                for col in range(grids[j]):
                    # 网络在原图的坐标
                    x = col * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    y = row * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    index = (row * grids[j] + col) + start_index
                    # if g_res_one[index, _s + 1 + 4] == 1.: # 这个是半径正例
                    if g_res_one[index, 0] == 0:
                        # 正例
                        off_l, off_t, off_r, off_b = g_res_one[index, _s + 1:_s + 1 + 4]
                        # 网络位置 求GT的位置
                        xmin = int(x - off_l)
                        ymin = int(y - off_t)
                        xmax = int(x + off_r)
                        ymax = int(y + off_b)

                        gcls = np.argmax(g_res_one[index, 1:_s], axis=-1)
                        mess = '%s' % (int(gcls))
                        cv2.circle(img_np, (int(x), int(y)), 5, CLASS_COLOR[int(gcls)], -1)
                        cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), CLASS_COLOR[int(gcls)], 2)
                        cv2.rectangle(img_np, (int(xmin), int(abs(ymin) - 15)),
                                      (int(xmin + (xmax - xmin) * 0.55), int(ymin)), CLASS_COLOR[int(gcls)], -1)
                        cv2.putText(img_np, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            start_index += (grids[j] ** 2)
        cv2.imshow('image', img_np)
        cv2.waitKey(0)
    return g_res_one


def match4fcos_v3(gboxes_ltrb_b, glabels_b, gdim, pcos, cfg, img_ts=None, ):
    '''
    这里处理一张图片
    :img_ts: 一张图片
    :return:
        v1 0.11269640922546387
        v2 0.001  提升1000倍
    '''

    gboxes_xywh_b = ltrb2xywh(gboxes_ltrb_b)
    # 转换到对应特图的尺寸
    device = gboxes_ltrb_b.device
    dim_total = pcos.shape[1]
    # labels_ = (glabels_b - 1).long()

    # back cls centerness ltrb positivesample iou area
    g_res_one = torch.zeros([dim_total, gdim], device=device)

    input_size = torch.tensor(cfg.IMAGE_SIZE, device=device)
    grids = torch.true_divide(input_size.unsqueeze(0),
                              torch.tensor(cfg.STRIDES, device=device).unsqueeze(-1).repeat(1, 2))
    grids = grids[:, 0].type(torch.int32).tolist()

    # 恢复到input
    gboxes_ltrb_b_input = gboxes_ltrb_b * input_size.repeat(2).unsqueeze(0)
    gboxes_xywh_b_input = ltrb2xywh(gboxes_ltrb_b_input)

    # g_res_one[:, 0] = 1.0  # 默认全部为背景
    g_res_one[:, cfg.NUM_CLASSES + 1 + 4 + 1 + 1] = 1.0  # 面积初始为最大1

    # back = torch.ones((dim_total, 1), device=device)
    positive_radius = torch.zeros((dim_total, 1), device=device)  # 半径正例
    positive_ingt = torch.zeros((dim_total, 1), device=device)  # 半径正例
    # (4) -> (1,4) ->  (nn,4)  # 这里多加了一个背景
    labels_onehot = torch.zeros((dim_total, cfg.NUM_CLASSES), device=device)  # 全为背景置1
    # labels_onehot[:, 0] = 1  # 全为背景置1

    # 遍历每一个标签, 的每一层的格子  找出格子是否在预测框中, 并记录差异
    for i in range(len(glabels_b)):
        l = gboxes_ltrb_b_input[i, 0]
        t = gboxes_ltrb_b_input[i, 1]
        r = gboxes_ltrb_b_input[i, 2]
        b = gboxes_ltrb_b_input[i, 3]
        area = torch.prod(gboxes_xywh_b[i][2:])

        index_colrow = []
        scale_thresholds = []
        radius = []

        image_size_ts = torch.tensor(cfg.IMAGE_SIZE, device=device)
        for j, s in enumerate(cfg.STRIDES):
            # 恢复到
            grid_wh = image_size_ts // s  # 特征格子
            _grids = f_mershgrid(grid_wh[1], grid_wh[0], is_rowcol=False).to(device)
            _grids = _grids * s + s // 2
            # dim, _ = _grids.shape
            index_colrow.append(_grids)

            _scale = torch.empty_like(_grids, device=device)
            _scale[:, 0] = cfg.SCALE_THRESHOLDS[j]
            _scale[:, 1] = cfg.SCALE_THRESHOLDS[j + 1]
            scale_thresholds.append(_scale)

            # [nn]
            _radius = torch.empty_like(_grids[:, 0], device=device)
            _radius[:] = cfg.MATCH_RADIUS * s  # 半径阀值
            radius.append(_radius)

        # (nn,2)
        index_colrow = torch.cat(index_colrow, 0)
        scale_thresholds = torch.cat(scale_thresholds, 0)
        radius = torch.cat(radius, 0)  # [nn]

        # (nn) # --- 是否框内条件 ---
        mask_col_lr = torch.logical_and(index_colrow[:, 0] >= l, index_colrow[:, 0] <= r)
        mask_row_tb = torch.logical_and(index_colrow[:, 1] >= t, index_colrow[:, 1] <= b)
        mask_D = torch.logical_and(mask_col_lr, mask_row_tb)

        # --- 中心格子半径条件 ---
        mask_radius = torch.logical_and(torch.abs(index_colrow[:, 0] - gboxes_xywh_b_input[i, 0]) < radius,
                                        torch.abs(index_colrow[:, 1] - gboxes_xywh_b_input[i, 1]) < radius)

        # (nn,2)
        off_lt = index_colrow - gboxes_ltrb_b_input[i, :2].unsqueeze(0)
        off_rb = gboxes_ltrb_b_input[i, 2:].unsqueeze(0) - index_colrow
        off_ltrb = torch.cat([off_lt, off_rb], -1)  # (nn,4)
        M, _ = torch.max(off_ltrb, -1)  # (nn,4) -> (nn)

        # (nn)  # --- 层阀值条件 ---
        mask_M = torch.logical_and(M >= scale_thresholds[:, 0], M <= scale_thresholds[:, 1])

        # --- 面积条件 ---
        mask_area = g_res_one[:, cfg.NUM_CLASSES + 1 + 4 + 1 + 1] > area

        mask = torch.logical_and(torch.logical_and(mask_D, mask_M), mask_area)
        mask_radius = torch.logical_and(mask, mask_radius)

        # (nn) -> (nn,1)
        center_ness = torch.sqrt(torch.min(off_ltrb[:, ::2], -1)[0] / torch.max(off_ltrb[:, ::2], -1)[0]
                                 * torch.min(off_ltrb[:, 1::2], -1)[0] / torch.max(off_ltrb[:, 1::2], -1)[0])
        center_ness.unsqueeze_(-1)

        '''
        back 默认为1 有类别即付为 onehot 复写  -> back=0
        centerness 用于与回归 loss 权重
        ltrb 用于IOU计算
        positive_sample 用于计算conf 需结合半径
        
        '''
        labels_onehot[mask, glabels_b[i].long() - 1] = 1.  # 这个需要保留以前的值 本次只复写需要的
        positive_radius[mask_radius] = 1  # 这是一个较小的交集, 需要的付为1 用于 conf 计算
        positive_ingt[mask] = 1.  # gt框内的标志  reg正例
        area_ts = torch.empty_like(positive_radius)  # (nn,1)
        area_ts[:, 0] = area  # 面积更新 全部付这个GT的面积 后面通过 mask 过滤

        # cls3 centerness1 ltrb4 positive_radius1 positive_ingt1 area1 3+1+4+1+1+1=11
        g_res_one_tmp = torch.cat([labels_onehot, center_ness, gboxes_ltrb_b[i].repeat(dim_total, 1),
                                   positive_radius, positive_ingt, area_ts], -1)
        g_res_one[mask] = g_res_one_tmp[mask]

    if cfg.IS_VISUAL:
        ''' 可视化匹配最大的ANC '''
        from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        _img_ts = f_recover_normalization4ts(img_ts.clone())
        from torchvision.transforms import functional as transformsF
        img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        import numpy as np
        img_np = np.array(img_pil)

        CLASS_COLOR = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
                       range(cfg.NUM_CLASSES)]

        start_index = 0
        for j in range(len(cfg.STRIDES)):

            for row in range(grids[j]):
                for col in range(grids[j]):
                    # 网络在原图的坐标
                    x = col * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    y = row * cfg.STRIDES[j] + cfg.STRIDES[j] // 2
                    index = (row * grids[j] + col) + start_index
                    # if g_res_one[index, cfg.NUM_CLASSES + 1+4] == 1.: # 这个是半径正例
                    if g_res_one[index, cfg.NUM_CLASSES + 1 + 4 + 1] == 1:  # 这是框内正例
                        # 正例
                        off_l, off_t, off_r, off_b = g_res_one[index, cfg.NUM_CLASSES + 1:cfg.NUM_CLASSES + 1 + 4]
                        # 网络位置 求GT的位置
                        xmin = int(x - off_l)
                        ymin = int(y - off_t)
                        xmax = int(x + off_r)
                        ymax = int(y + off_b)

                        gcls = np.argmax(g_res_one[index, :cfg.NUM_CLASSES], axis=-1)
                        mess = '%s' % (int(gcls))
                        cv2.circle(img_np, (int(x), int(y)), 5, CLASS_COLOR[int(gcls)], -1)
                        cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), CLASS_COLOR[int(gcls)], 2)
                        cv2.rectangle(img_np, (int(xmin), int(abs(ymin) - 15)),
                                      (int(xmin + (xmax - xmin) * 0.55), int(ymin)), CLASS_COLOR[int(gcls)], -1)
                        cv2.putText(img_np, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            start_index += (grids[j] ** 2)
        cv2.imshow('image', img_np)
        cv2.waitKey(0)
    return g_res_one


def boxes_decode4fcos(cfg, poff_ltrb, is_t=False):
    '''
    将预测出来对应特图偏移 通过生成对应的特图的网格 求出真实坐标
    预测的是特图上的对应点到真实框的距离 计算特图和归一图iou的效果效果是一样的
    '''
    device = poff_ltrb.device
    weight = torch.tensor([-1, -1, 1, 1], device=device).view(1, 1, -1)

    index_colrow = []  # 这个是表示对应特图上的点, 感受野中心点
    # start_index = 0
    image_size_ts = torch.tensor(cfg.IMAGE_SIZE, device=device)
    for s in cfg.STRIDES:
        grid_wh = image_size_ts // s
        _grids = f_mershgrid(grid_wh[1], grid_wh[0], is_rowcol=False).to(device)
        _grids = _grids * s + s // 2  # 每层对应的特图感受野块
        index_colrow.append(_grids)
        # start_index += (grid_wh[1] * grid_wh[0])

    index_colrow = torch.cat(index_colrow, 0).repeat(1, 2).unsqueeze(0)
    # [2, 2125, 4]
    pboxes_ltrb1 = poff_ltrb * weight + index_colrow  # 得到真实 ltrb
    if not is_t:  # 是否特图尺寸  默认否为运行进行 归一化
        pboxes_ltrb1 = pboxes_ltrb1 / image_size_ts.repeat(2).view(1, 1, -1)  # 归一化尺寸
    return pboxes_ltrb1


def fmatch4yolov5(gboxes_ltrb_b, glabels_b, dim, ptxywh_b, device, cfg, img_ts=None, pconf_b=None):
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    # (ngt,4) -> (ngt,2)
    ngt = len(gboxes_xywh)

    # flog.debug('训练和验证的数据 %s %s', ngt, len(glabels_b))
    # from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
    # _img_ts = f_recover_normalization4ts(img_ts.clone())
    # from torchvision.transforms import functional as transformsF
    # img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
    # img_pil.show()

    # (ngt,4) -> (ngt,5)  n类,xywh
    g_nxywh = torch.cat([glabels_b.unsqueeze(-1), gboxes_xywh], -1)

    '''求 cr 偏xy  匹配:ceng偏移,grid,ids_anc'''
    # (ngt,5) -> (ngt,1,5)  -> (ngt,9,5)
    g_nxywh_p = g_nxywh.unsqueeze(1).repeat_interleave(len(cfg.ANCS_SCALE), dim=1)
    gwh = g_nxywh_p[..., 3:5]  # gwh 赋值 原图归一化用于与anc比较IOU

    # 匹配正例
    # 计算anc 与 gt的宽高比 剔除宽高比大于阀值的anc < 4.0
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)  # (9,2)
    ancs_wh_ts = ancs_wh_ts.unsqueeze(0).repeat(ngt, 1, 1)  # 系统可以自动广播
    # (ngt,2) -> (ngt,1,2) ^^ (9,2) -> (ngt,9,2) -> (ngt,9)
    swh = gwh / ancs_wh_ts
    # (ngt,9,2) -> (ngt,9)
    max_wh, max_indexs = swh.max(-1)  # 最大宽高比
    mask_pos_wh = torch.logical_and(max_wh < 4.0, max_wh > 0.25)
    # mask_pos_wh = max_wh < 4.0

    # 匹配格子 (9) 这里写死
    grid_ts_p = torch.tensor([52, 26, 13], device=device).repeat_interleave(3, dim=-1)
    # (9) -> (1,9,1)
    grid_ts_p = grid_ts_p.unsqueeze(0).unsqueeze(-1).repeat(ngt, 1, 1)

    # 计算 cr及偏xy
    # (num_gt,9,2) 格子左上角
    gxy = g_nxywh_p[..., 1:3]
    _xy = gxy * grid_ts_p  # 实际xy
    index_colrow = _xy.type(torch.int32)
    offset_xy = _xy - index_colrow

    # 匹配:ceng偏移
    # kk (ngt,5+2+2+2+1+1+1=14) nxywh0 + cr(2)5 + oxy(2)7 + swh(2)9 + offset_s(1)11 + grid(1)12 + ids_anc(1)13 + ancwh(2)14
    _t0 = torch.tensor(cfg.NUMS_CENG) * 3
    _t1 = _t0[0]
    _t2 = _t1 + _t0[1]
    offset_s = torch.tensor([0, 0, 0, _t1, _t1, _t1, _t2, _t2, _t2, ], device=device)
    # (9) -> (1,9,1) -> (ngt,9,1)
    offset_s = offset_s.unsqueeze(0).unsqueeze(-1).repeat(ngt, 1, 1)

    # 匹配 ids_anc
    _ids_anc = torch.arange(0, 3, device=device).repeat(3).unsqueeze(0).unsqueeze(-1).repeat(ngt, 1, 1)
    gres = torch.cat([g_nxywh_p, index_colrow, offset_xy, swh, offset_s, grid_ts_p, _ids_anc, ancs_wh_ts], -1)

    # kk (ngt,5+2+2+2+1+1+1=14) nxywh0 + cr(2)5 + oxy(2)7 + swh(2)9 + offset_s(1)11 + grid(1)12 + ids_anc(1)13 + ancwh(2)14
    gres_pos = gres[mask_pos_wh]

    ''' 匹配附近的偏移格子 偏移距离 '''
    # 这两个必须是对应位不同  例如 1,0  -1,0  不能都是1或0
    t1 = torch.tensor([[1, 0], [-1, 0]], device=device)
    t2 = torch.tensor([[0, 1], [0, -1]], device=device)
    # oxy(npos,9) -> ox或oy(npos,1)
    offset_x = gres_pos[..., 7:8]
    offset_y = gres_pos[..., 8:9]
    # (npos,1) -> (npos,2) 一个匹配的对应两个偏移列
    _offset_col = torch.where(offset_x > 0.5, t1[0], t1[1])
    _offset_row = torch.where(offset_y > 0.5, t2[0], t2[1])
    # (npos,2) -> (npos,2,1)
    _offset_col.unsqueeze_(-1)
    _offset_row.unsqueeze_(-1)
    # 得到扩展的两个正例的格子偏移 (npos,2,1) -> (npos,2,2)
    _offset_grid = torch.cat([_offset_col, _offset_row], -1)
    # 添加00偏移用于维度运算对齐 (npos,2,2) -> (npos,1,2)
    _t = torch.zeros_like(_offset_grid)[:, 0:1, :]
    # (npos,1,2) ^^ (npos,2,2) -> (npos,3,2)
    _offset_grid = torch.cat([_t, _offset_grid], 1)
    # 计算cr(2) 和 oxy(2) 网格-1 xy-0.5  (npos,3,2) -> (npos*3,2) -> (npos*3,4)
    _offset_grid_xy = _offset_grid.view(-1, 2).repeat(1, 2).float()
    _offset_grid_xy[..., 3:5] = _offset_grid_xy[..., 3:5] * 0.5

    # (npos,11) -> (npos*3,11)
    gres_pos = gres_pos.repeat_interleave(3, 0)
    # kk (ngt,5+2+2+2+1+1+1=14) nxywh0 + cr(2)5 + oxy(2)7 + swh(2)9 + offset_s(1)11 + grid(1)12 + ids_anc(1)13 + ancwh(2)14
    # (npos*3,11) ->(npos*3,4) ^^ (npos*3,4) ->(npos*3,4)
    gres_pos[..., 5:9] = gres_pos[..., 5:9] - _offset_grid_xy

    # 找出最终偏移  kk (ngt,5+2+2+2+1+1+1=14) nxywh0 + cr(2)5 + oxy(2)7 + swh(2)9 + offset_s(1)11 + grid(1)12 + ids_anc(1)13 + ancwh(2)14
    # (row*grid+col)*3+ids_anc
    ''' 确保加减后不会越界 '''
    m1 = torch.logical_and(gres_pos[:, 6] > 0, gres_pos[:, 6] < gres_pos[:, 12])
    m2 = torch.logical_and(gres_pos[:, 5] > 0, gres_pos[:, 5] < gres_pos[:, 12])
    m = torch.logical_and(m1, m2)
    m = torch.logical_not(m)
    if torch.any(m):
        a = 1
        pass
    gres_pos[torch.where(m)[0], 5] = 999999

    offset_colrow = (gres_pos[:, 6] * gres_pos[:, 12] + gres_pos[:, 5]) * 3 + gres_pos[:, 13]
    gres_pos[:, 11] = gres_pos[:, 11] + offset_colrow
    gres_pos = gres_pos[gres_pos[:, 5] < 999999]

    # ---------------------------------
    # print(gres_pos)

    _dim_total = sum(cfg.NUMS_CENG) * 3  # 10647
    g_yolo_one = torch.zeros((_dim_total, dim), device=device)

    # (ngt,5+2+2+2+1+1+1=14) nxywh0 + cr(2)5 + oxy(2)7 + swh(2)9 + offset_s(1)11 + grid(1)12 + ids_anc(1)13 + ancwh(2)14
    conf = torch.ones(gres_pos.shape[0], 1, device=device)
    # glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    label = gres_pos[:, 0:1]
    goff_xywh = gres_pos[:, 7:11]
    gxywh = gres_pos[:, 1:5]
    ancwh = gres_pos[:, 14:16]

    # conf1 + label1 + goff_xywh4 + gxywh4 +anc2 =12
    _t = torch.cat([conf, label, goff_xywh, gxywh, ancwh], dim=-1)
    # flog.debug('gres_pos[:, 11] %s', gres_pos[:, 11].max())
    g_yolo_one[gres_pos[:, 11].long()] = _t

    if cfg.IS_VISUAL:
        ''' 可视化匹配最大的ANC '''
        from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        _img_ts = f_recover_normalization4ts(img_ts.clone())
        from torchvision.transforms import functional as transformsF
        img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
        import numpy as np
        img_np = np.array(img_pil)

        # 解码
        pxy = gres_pos[::3, 9:11] + gres_pos[::3, 5:7]
        grid = gres_pos[::3, 12:13].repeat(1, 2)
        pxy = pxy / grid

        ancs_wh_ts = gres_pos[::3, 14:16]
        pxywh = torch.cat([pxy, ancs_wh_ts], -1)  # torch.Size([3, 169, 5, 4])
        pltrb1 = xywh2ltrb(pxywh)

        # ------------ 辅助框 -------------
        mask = torch.ones(gres_pos.shape[0]).to(torch.bool)
        mask[::3] = False
        oxy = gres_pos[mask, 7:9]
        # 修正 oxy 到 对应格子偏移
        oxy = torch.where(oxy < 0, 1 - oxy, oxy)
        oxy = torch.where(oxy > 1, oxy - 1, oxy)
        pxy = oxy + gres_pos[mask, 5:7]
        grid = gres_pos[mask, 12:13].repeat(1, 2)
        pxy = pxy / grid

        ancs_wh_ts = gres_pos[mask, 14:16]
        pxywh = torch.cat([pxy, ancs_wh_ts], -1)  # torch.Size([3, 169, 5, 4])
        pltrb2 = xywh2ltrb(pxywh)

        f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b.cpu()
                         , pboxes_ltrb=pltrb1,
                         other_ltrb=pltrb2,
                         is_recover_size=True)
    return g_yolo_one
