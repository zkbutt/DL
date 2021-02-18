import math

import numpy as np
import torch

from f_tools.f_general import labels2onehot4ts
from f_tools.fun_od.f_boxes import ltrb2xywh, xywh2ltrb, calc_iou4ts, offxy2xy
from f_tools.yufa.x_calc_adv import f_mershgrid
from object_detection.z_center.utils import gaussian_radius, draw_gaussian

'''-------------------------------解码--------------------------'''


def boxes_decode4yolo1(ptxywh, grid_h, grid_w, cfg):
    '''
    解码
    :param ptxywh: 预测的是在特图的 偏移 和 缩放比例
    :param grid_h:
    :param grid_w:
    :param cfg:
    :return: 输出归一化
    '''
    device = ptxywh.device
    _xy_grid = torch.sigmoid(ptxywh[:, :, :2]) \
               + f_mershgrid(grid_h, grid_w, is_rowcol=False, num_repeat=cfg.NUM_ANC).to(device)
    hw_ts = torch.tensor((grid_h, grid_w), device=device)  # /13
    ptxywh[:, :, :2] = torch.true_divide(_xy_grid, hw_ts)  # 原图归一化

    if cfg.loss_args['s_match'] == 'whoned':
        ptxywh[:, :, 2:4] = torch.sigmoid(ptxywh[:, :, 2:])
    elif cfg.loss_args['s_match'] == 'log':
        ptxywh[:, :, 2:4] = torch.exp(ptxywh[:, :, 2:]) / cfg.IMAGE_SIZE[0]  # wh log-exp
    elif cfg.loss_args['s_match'] == 'log_g':
        ptxywh[:, :, 2:4] = torch.exp(ptxywh[:, :, 2:]) / grid_h  # 原图归一化
    else:
        raise Exception('类型错误')
    # return ptxywh


def boxes_decode4yolo2(ptxywh, grid_h, grid_w, cfg):
    '''
    解码  4维 -> 3维
    用于计算 iou得conf   及 预测
    :param ptxywh: 原始预测 torch.Size([32, 169, 5, 4]) -> [3, 169*5, 4]
    :param grid_h:
    :param grid_w:
    :param cfg:
    :return: 输出原图归一化 [3, 169*5, 4]
    '''
    device = ptxywh.device
    # 特图xy -> 原图
    pxy = torch.sigmoid(ptxywh[:, :, :, :2]) \
          + f_mershgrid(grid_h, grid_w, is_rowcol=False, num_repeat=cfg.NUM_ANC) \
              .to(device).reshape(-1, cfg.NUM_ANC, 2)
    pxy = pxy / grid_h

    # 特图wh比例 -> 原图
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    # 比例 ( 比例不需要转换 ) * 特图anc_wh
    pwh = torch.exp(ptxywh[:, :, :, 2:4]) * ancs_wh_ts  # torch.Size([3, 361, 5, 2])
    # fdebug 可视化匹配的anc
    # pwh = ancs_ts.view(1, 1, *ancs_ts.shape).repeat(*ptxywh[:, :, :, 2:4].shape[:2], 1, 1)

    pxywh = torch.cat([pxy, pwh], -1)  # torch.Size([3, 169, 5, 4])
    pxywh = pxywh.view(ptxywh.shape[0], -1, 4)  # 原图归一化 [3, 169, 5, 4] -> [3, 169*5, 4]
    pltrb = xywh2ltrb(pxywh)
    return pltrb


def boxes_decode4yolo3(ptxywh, cfg):
    '''
    :param ptxywh: torch.Size([3, 10647, 4]) 不要归一化
    :param cfg:
        cfg.NUMS_ANC [2704, 676, 169]
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
    pxywh = torch.cat([pxy, pwh], -1) # torch.Size([32, 10647, 4])
    pltrb = xywh2ltrb(pxywh)
    return pltrb


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
        xy = _anc_xy + _pxy * variances[0] * _anc_wh
        wh = _anc_wh * torch.exp(_pwh * variances[1])
    elif ptxywh.dim() == 3:
        anc_xywh_t = anc_obj.ancs_xywh.unsqueeze(0) * match_grids_ts
        _anc_xy = anc_xywh_t[:, :, :2]
        _anc_wh = anc_xywh_t[:, :, 2:]
        _pxy = ptxywh_t[:, :, :2]
        _pwh = ptxywh_t[:, :, 2:]
        xy = _anc_xy + _pxy * variances[0] * _anc_wh
        wh = _anc_wh * torch.exp(_pwh * variances[1])
    else:
        raise Exception('维度错误', anc_obj.ancs_xywh.shape)
    _pxywh = torch.cat([xy, wh], dim=-1)
    _pxywh = _pxywh / match_grids_ts
    return _pxywh


def boxes_decode4retina2(cfg, anc_obj, ptxywh, variances=(0.1, 0.2)):
    '''
    需要统一 一维和二维
    :param cfg: cfg.tnums_ceng cfg.NUMS_ANC [8112, 2028, 507]  用于动态层 dim 数
    :param anc_obj: xywh  (nn,4) 这个始终是一维
    :param ptxywh: 修正系数 (nn,4)  torch.Size([32, 10647, 4])
    :return: 修复后的框
    '''
    size_ts4 = torch.tensor(cfg.IMAGE_SIZE, dtype=torch.float32, device=anc_obj.ancs_xywh.device).repeat(2)

    # torch.Size([32, 10647, 4]) * torch.Size([10647, 1])
    ptxywh_t = ptxywh  # 这里是特图

    if ptxywh.dim() == 2:
        anc_xywh_t = anc_obj.ancs_xywh * size_ts4
        _anc_xy = anc_xywh_t[:, :2]
        _anc_wh = anc_xywh_t[:, 2:]
        _pxy = ptxywh_t[:, :2]
        _pwh = ptxywh_t[:, 2:]
        xy = _anc_xy + _pxy * variances[0] * _anc_wh
        wh = _anc_wh * torch.exp(_pwh * variances[1])
    elif ptxywh.dim() == 3:
        anc_xywh_t = anc_obj.ancs_xywh.unsqueeze(0) * size_ts4
        _anc_xy = anc_xywh_t[:, :, :2]
        _anc_wh = anc_xywh_t[:, :, 2:]
        _pxy = ptxywh_t[:, :, :2]
        _pwh = ptxywh_t[:, :, 2:]
        xy = _anc_xy + _pxy * variances[0] * _anc_wh
        wh = _anc_wh * torch.exp(_pwh * variances[1])
    else:
        raise Exception('维度错误', anc_obj.ancs_xywh.shape)
    _pxywh = torch.cat([xy, wh], dim=-1)
    _pxywh = _pxywh / size_ts4
    return _pxywh


'''-------------------------------编码--------------------------'''


def boxes_encode4yolo1(gboxes_ltrb, grid_h, grid_w, device, cfg):
    '''
    编码GT
    :param gboxes_ltrb: 归一化尺寸
    :param grid_h:
    :param grid_w:
    :param device:
    :param cfg:
    :return: 返回特图回归系数
    '''
    # ltrb -> xywh 原图归一化   编码xy与yolo2一样的
    gboxes_xywh = ltrb2xywh(gboxes_ltrb)
    whs = gboxes_xywh[:, 2:]

    cxys = gboxes_xywh[:, :2]
    grids_ts = torch.tensor([grid_h, grid_w], device=device, dtype=torch.int16)
    index_colrow = (cxys * grids_ts).type(torch.int16)  # 网格7的index
    offset_xys = torch.true_divide(index_colrow, grid_h)  # 网络index 对应归一化的实距

    txys = (cxys - offset_xys) * grids_ts  # 特图偏移
    twhs = (whs * torch.tensor(grid_h, device=device)).log()  # 特图长宽 log减小差距
    txywhs_g = torch.cat([txys, twhs], dim=-1)

    # 小目标损失加重
    weights = 2.0 - torch.prod(whs, dim=-1)
    return txywhs_g, weights, index_colrow


def boxes_encode4yolo2(gboxes_ltrb, mask_p, grid_h, grid_w, device, cfg):
    '''
    编码GT
    :param gboxes_ltrb: 归一化尺寸
    :param grid_h:
    :param grid_w:
    :param device:
    :param cfg:
    :return:
    '''
    # ltrb -> xywh 原图归一化  编码xy与yolo1一样的
    gboxes_xywh = ltrb2xywh(gboxes_ltrb)
    whs = gboxes_xywh[:, 2:]

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


def boxes_encode4retina(cfg, anc_obj, gboxes_ltrb, variances=(0.1, 0.2)):
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
        _a = (gxywh_t[:, :2] - anc_xywh_t[:, :2]) / anc_xywh_t[:, 2:] / variances[0]
        _b = (gxywh_t[:, 2:] / anc_xywh_t[:, 2:]).log() / variances[1]
    elif gboxes_ltrb.dim() == 3:
        anc_xywh_t = anc_obj.ancs_xywh.unsqueeze(0) * match_grids_ts
        _a = (gxywh_t[:, :, :2] - anc_xywh_t[:, :, :2]) / anc_xywh_t[:, :, 2:] / variances[0]
        _b = (gxywh_t[:, :, 2:] / anc_xywh_t[:, :, 2:]).log() / variances[1]
    else:
        raise Exception('维度错误', anc_obj.ancs_xywh.shape)
    _gtxywh = torch.cat([_a, _b], dim=-1)
    return _gtxywh


def boxes_encode4retina2(cfg, anc_obj, gboxes_ltrb, variances=(0.1, 0.2)):
    '''
    用anc同维 和 已匹配的GT 计算差异
    :param cfg: cfg.tnums_ceng cfg.NUMS_ANC [8112, 2028, 507]  用于动态层 dim 数
    :param anc_obj:
        ancs_xywh  (nn,4) torch.Size([1, 16800, 4]) torch.Size([10647, 4])
        anc_obj.nums_dim_feature: [24336, 6084, 1521, 441, 144]
    :param gboxes_ltrb: gt torch.Size([5, 16800, 4]) torch.Size([10647, 4])
    :return: 计算差异值   对应xywh torch.Size([10647, 4])
    '''
    gxywh = ltrb2xywh(gboxes_ltrb)
    size_ts4 = torch.tensor(cfg.IMAGE_SIZE, dtype=torch.float32, device=anc_obj.ancs_xywh.device).repeat(2)
    gxywh_t = gxywh * size_ts4

    if gboxes_ltrb.dim() == 2:
        anc_xywh_t = anc_obj.ancs_xywh * size_ts4
        _a = (gxywh_t[:, :2] - anc_xywh_t[:, :2]) / anc_xywh_t[:, 2:] / variances[0]
        _b = (gxywh_t[:, 2:] / anc_xywh_t[:, 2:]).log() / variances[1]
    elif gboxes_ltrb.dim() == 3:
        anc_xywh_t = anc_obj.ancs_xywh.unsqueeze(0) * size_ts4
        _a = (gxywh_t[:, :, :2] - anc_xywh_t[:, :, :2]) / anc_xywh_t[:, :, 2:] / variances[0]
        _b = (gxywh_t[:, :, 2:] / anc_xywh_t[:, :, 2:]).log() / variances[1]
    else:
        raise Exception('维度错误', anc_obj.ancs_xywh.shape)
    _gtxywh = torch.cat([_a, _b], dim=-1)
    return _gtxywh


'''------------------------------- 匹配 --------------------------'''


def match_gt_b(cfg, dim, gboxes_ltrb_b, glabels_b, anc_obj, mode='iou', ptxywh_b=None, img_ts=None):
    '''
    cfg.NUMS_ANC = [3, 3, 3]

    :param cfg:
    :param dim: iou1, cls-1, txywh-4, gltrb-4,  keypoint-nn  = 10 + nn
    :param gboxes_ltrb_b: torch.Size([ngt, 4])
    :param glabels_b: [ngt]
    :param anc_obj:
        anc_obj.nums_level: [24336, 6084, 1521, 441, 144]
        ancs_xywh  (nn,4)
    :param mode: topk atss iou
    :param ptxywh_b: [32526, 4]
    :param img_ts: [3, 416, 416]
    :return:
    '''
    # mode = 'iou'  # topk atss iou
    if ptxywh_b is not None:
        device = ptxywh_b.device
    else:
        device = torch.device('cpu')
    num_atss_topk = 9
    # 计算 iou
    anc_xywh = anc_obj.ancs_xywh
    anc_ltrb = xywh2ltrb(anc_xywh)
    num_anc = anc_xywh.shape[0]
    # (anc 个,boxes 个) torch.Size([3, 10647])
    ious = calc_iou4ts(anc_ltrb, gboxes_ltrb_b)
    num_gt = gboxes_ltrb_b.shape[0]

    # 全部ANC的距离
    gboxes_xywh_b = ltrb2xywh(gboxes_ltrb_b)
    # (anc 个,boxes 个)  torch.Size([32526, 7])
    distances = (anc_xywh[:, None, :2] - gboxes_xywh_b[None, :, :2]).pow(2).sum(-1).sqrt()

    # cls-1, txywh-4, gltrb-4,  keypoint-nn  = 9 + nn
    gretinas_one = torch.zeros((num_anc, dim), device=device)  # 返回值

    if mode == 'atss':
        # 放 topk个 每个gt对应对的anc的index
        idxs_candidate = []
        index_start = 0  # 这是每层的anc偏移值
        for i, num_dim_feature in enumerate(anc_obj.nums_level):
            '''每一层的每一个GT选 topk*'''
            index_end = index_start + num_dim_feature
            distances_per_level = distances[index_start:index_end, :]
            # 确认该层的TOPK 不能超过该层总 anc 数 这里是一个数
            topk = min(num_atss_topk * cfg.NUMS_ANC[i], num_dim_feature)
            # torch.Size([24336, box_n])---(anc,gt) -> torch.Size([topk, 1]) 放 topk个 每个gt对应对的anc的index
            _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)  # 只能在某一维top
            idxs_candidate.append(topk_idxs_per_level + index_start)
            index_start = index_end

        # 候选人，候补者；应试者
        idxs_candidate = torch.cat(idxs_candidate, dim=0)
        '''--- 选出每层每个anc对应的距离中心最近topk iou值 ---'''
        # ***************这个是ids选择 这个是多维筛选 ious---[anc,ngt]    [405, ngt] [0,1...ngt]-> [405,ngt]
        ious_candidate = ious[idxs_candidate, torch.arange(num_gt)]

        '''--- 用最大一个iou的均值和标准差,计算阀值 ---'''
        # 统计每一个 GT的均值 std [ntopk,ngt] -> [ngt]
        _iou_mean_per_gt = ious_candidate.mean(dim=0)  # 除维
        _iou_std_per_gt = ious_candidate.std(dim=0)
        _iou_thresh_per_gt = _iou_mean_per_gt + _iou_std_per_gt
        '''--- 用IOU阀值初选正样本 ---'''
        # 每一个GT是阀值 [405,ngt] ^^ ([ngt] -> [1,ngt])->  [405,ngt]
        mask_pos4iou = ious_candidate >= _iou_thresh_per_gt[None, :]

        '''--- 中心点需落在GT中间 需要选出 anc的中心点-gt的lt为正, gr的rb-anc的中心点为正  ---'''
        for ng in range(num_gt):  # 拉平
            # 强制使每一个GT对应的anc 加个偏移
            idxs_candidate[:, ng] += ng * num_anc
        num_pos4iou = idxs_candidate.shape[0]
        idxs_candidate4gt = idxs_candidate.view(-1)
        # [65052, 4] -> [65052*num_anc, 2] 扩展后用于多GT筛选
        anc_xy4gt = anc_xywh.repeat(num_gt, 1)[:, :2]  # 用于与拉平的匹配
        # 单体复制 与 anc匹配
        gboxes_ltrb_b4gt = torch.repeat_interleave(gboxes_ltrb_b, num_pos4iou, dim=0)
        anc_xy4candidate = anc_xy4gt[idxs_candidate4gt]  # 把 anc 选出来
        dlt = anc_xy4candidate - gboxes_ltrb_b4gt[:, :2]
        drb = gboxes_ltrb_b4gt[:, 2:] - anc_xy4candidate
        # [405*ngt,4] -> [405*ngt]
        mask_pos4in_gt = torch.all(torch.cat([dlt, drb], dim=-1) > 0.01, dim=1)
        mask_pos4in_gt = mask_pos4in_gt.view(-1, num_gt)
        # 合并距离 和 中间点的 mask
        mask_pos = torch.logical_and(mask_pos4iou, mask_pos4in_gt)

        '''--- 一个锚框被多个真实框所选择，则其归于iou较高的真实框  ---'''
        # [32526, 1] -> [1, 32526] 创建一个按GT个数 -1 拉平的 match ().contiguous(默认负例为-1
        ious_z = torch.full_like(ious, -1).t().view(-1)
        # 取出所有的正例 [nnn] -> [74]
        index = idxs_candidate4gt[mask_pos.view(-1)]
        # [anc,gt] -> [gt,anc] ->[gt*anc] -> 最终选出来 mask_pos个 写出真实的iou
        ious_z[index] = ious.t().contiguous().view(-1)[index]
        # 恢复到最初 32526*ngt -> [ngt,32526] -> [32526, ngt]
        ious_z = ious_z.view(num_gt, -1).t()
        # [32526, ngt] -> [32526, 1] -> [32526] 最张index

        anc_max_iou, boxes_index = ious_z.max(dim=1)

        gretinas_one[:, 0] = glabels_b[boxes_index]
        gretinas_one[:, 0][anc_max_iou == -1] = 0

    elif mode == 'topk':
        '''--- 简单匹配9个 用IOU和距离相关取9个 ---'''
        distances = distances / distances.max() / 1000
        mask_pos = torch.zeros_like(ious, dtype=torch.bool)
        for ng in range(num_gt):
            # [3614] -> [9]
            _, topk_idxs = (ious[:, ng] - distances[:, ng]).topk(num_atss_topk, dim=0)
            anc_xy = anc_xywh[topk_idxs, :2]
            dlt = anc_xy - gboxes_ltrb_b[ng, :2]
            drb = - anc_xy + gboxes_ltrb_b[ng, 2:]
            # [topk,4] -> [topk]
            mask_4in_gt = torch.all(torch.cat([dlt, drb], dim=-1) > 0.01, dim=1)
            mask_pos[topk_idxs[mask_4in_gt], ng] = True
        ious[torch.logical_not(mask_pos)] = -1
        ious_z = ious

        anc_max_iou, boxes_index = ious_z.max(dim=1)

        gretinas_one[:, 0] = glabels_b[boxes_index]
        gretinas_one[:, 0][anc_max_iou == -1] = 0

    elif mode == 'iou':
        '''--- 这个有可能只能匹配1个 ---'''
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
        labels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)

        gretinas_one[mask_pos, :cfg.NUM_CLASSES] = labels_b[boxes_index][mask_pos].float()
        gretinas_one[mash_ignore, :cfg.NUM_CLASSES] = torch.tensor(-1., device=device)

    else:
        raise NotImplementedError

    # _gtxywh = boxes_encode4retina(cfg, anc_obj, gboxes_ltrb_b[boxes_index], variances=cfg.variances)
    _gtxywh = boxes_encode4retina2(cfg, anc_obj, gboxes_ltrb_b[boxes_index], variances=cfg.variances)
    gretinas_one[:, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 4] = _gtxywh
    # gretinas_one[:, 1 + 4:1 + 4 + 4] = gboxes_ltrb_b[boxes_index]
    return gretinas_one


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


def pos_match_retina(cfg, dim, anc_obj, gboxes_ltrb_b, glabels_b, gkeypoints_b, ptxywh_b=None, img_ts=None):
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
    mask_pos = anc_max_iou >= cfg.THRESHOLD_CONF_POS  # [10647] anc 的正例 index 不管
    mask_neg = anc_max_iou < cfg.THRESHOLD_CONF_NEG  # anc 的反例 index
    mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))

    dim_total, _d4 = ptxywh_b.shape
    device = ptxywh_b.device
    # conf-1, cls-3, txywh-4, keypoint-nn  = 8 + nn
    gretinas_one = torch.zeros((dim_total, dim), device=device)  # 返回值

    labels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)

    # 整体匹配 正例全部为1 默认为0
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
