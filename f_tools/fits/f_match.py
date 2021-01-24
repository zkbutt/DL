import numpy as np
import torch

from f_tools.f_general import labels2onehot4ts
from f_tools.fun_od.f_boxes import ltrb2xywh, xy2offxy, xywh2ltrb, calc_iou4ts, offxy2xy, calc_iou4some_dim
from f_tools.pic.f_show import f_plt_od_f
from object_detection.z_center.utils import gaussian_radius, draw_gaussian


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


def match4yolo3(targets, anchors_obj, nums_anc=(3, 3, 3), num_class=20, device=None, imgs_ts=None):
    '''

    :param targets:
    :param anchors_obj:
    :param nums_anc: 只支持每层相同的anc数
    :param num_class:
    :param device:
    :return:
    '''
    batch = len(targets)
    # 层尺寸   tensor([[52., 52.], [26., 26.], [13., 13.]])
    feature_sizes = np.array(anchors_obj.feature_sizes)
    num_ceng = feature_sizes.prod(axis=1)  # 2704 676 169

    # 匹配完成的数据
    _num_total = sum(num_ceng * nums_anc)  # 10647
    _dim = 4 + 1 + num_class + 2  # 25 +off_x  off_y
    # torch.Size([5, 10647, 25])
    targets_yolo = torch.zeros(batch, _num_total, _dim).to(device)

    # 分批处理
    for i in range(batch):
        target = targets[i]
        # 只能一个个的处理
        boxes = target['boxes'].to(device)  # ltrb
        labels = target['labels'].to(device)

        # 可视化1
        img_ts = imgs_ts[i]
        from torchvision.transforms import functional as F
        img_pil = F.to_pil_image(img_ts).convert('RGB')
        # show_anc4pil(img_pil,boxes,size=img_pil.size)
        # img_pil.show()

        boxes_xywh = ltrb2xywh(boxes)
        num_anc = np.array(nums_anc).sum()  # anc总和数
        # 优先每个bbox重复9次
        p1 = boxes_xywh.repeat_interleave(num_anc, dim=0)  # 单体复制 3,4 -> 6,4
        # 每套尺寸有三个anc 整体复制3,2 ->6,2
        _n = boxes_xywh.shape[0] * nums_anc[0]  # 只支持每层相同的anc数
        # 每一个bbox对应的9个anc tensor([[52., 52.], [26., 26.], [13., 13.]])
        _feature_sizes = torch.tensor(feature_sizes, dtype=torch.float32).to(device)
        # 单复制-整体复制 只支持每层相同的anc数 [[52., 52.],[52., 52.],[52., 52.], ...[26., 26.]..., [13., 13.]...]
        p2 = _feature_sizes.repeat_interleave(nums_anc[0], dim=0).repeat(boxes_xywh.shape[0], 1)
        offxy_xy, colrow_index = xy2offxy(p1[:, :2], p2)  # 用于最终结果

        '''batch*9 个anc的匹配'''
        # 使用网格求出anc的中心点
        _ancs_xy = colrow_index / p2
        # 大特图对应小目标 _ancs_scale 直接作为wh
        _ancs_scale = torch.tensor(anchors_obj.ancs_scale).to(device)
        _ancs_wh = _ancs_scale.reshape(-1, 2).repeat(boxes_xywh.shape[0], 1)  # 拉平后整体复制
        ancs_xywh = torch.cat([_ancs_xy, _ancs_wh], dim=1)

        # --------------可视化调试----------------
        # flog.debug(offxy_xy)
        # f_show_iou_0(ltrb2ltwh(boxes), xywh2ltwh(ancs_xywh))
        # f_show_iou4pil(img_pil, boxes, xywh2ltrb(ancs_xywh), grids=anchors_obj.feature_sizes[-1])

        ids = torch.arange(boxes.shape[0]).to(device)  # 9个anc 0,1,2 只支持每层相同的anc数
        _boxes_offset = boxes + ids[:, None]  # boxes加上偏移 1,2,3
        ids_offset = ids.repeat_interleave(num_anc)  # 1,2,3 -> 000000000 111111111 222222222
        # p1 = xywh2ltrb(p1)  # 匹配的bbox
        p2 = xywh2ltrb(ancs_xywh)

        # iou = calc_iou4ts(_boxes_offset, p2 + ids_offset[:, None])
        # iou = calc_iou4ts(_boxes_offset, p2 + ids_offset[:, None], is_giou=True)
        # iou = calc_iou4ts(_boxes_offset, p2 + ids_offset[:, None], is_diou=True)
        iou = calc_iou4ts(_boxes_offset, p2 + ids_offset[:, None], is_ciou=True)
        # 每一个GT的匹配降序再索引恢复(表示匹配的0~8索引)  15 % anc数9 难 后面的铁定是没有用的
        iou_sort = torch.argsort(-iou, dim=1) % num_anc  # 9个

        for j in range(iou_sort.shape[0]):  # gt个数
            for k in range(num_anc):  # 若同一格子有9个大小相同的对象则无法匹配
                # 第k个 一定在 num_anc 之中
                k_ = iou_sort[j][k]
                # 匹配特图的索引
                match_anc_index = torch.true_divide(k_, len(nums_anc)).type(torch.int16)  # 只支持每层相同的anc数
                # _match_anc_index = match_anc_index[j]  # 匹配特图的索引
                # 只支持每层相同的anc数 和正方形
                offset_ceng = 0
                if match_anc_index > 0:
                    offset_ceng = num_ceng[:match_anc_index].sum()
                # 取出 anc 对应的列行索引
                _row_index = colrow_index[j * num_anc + k_, 1]  # 行
                _col_index = colrow_index[j * num_anc + k_, 0]  # 列
                offset_colrow = _row_index * feature_sizes[match_anc_index][0] + _col_index
                # 这是锁定开始位置
                match_index_s = (offset_ceng + offset_colrow) * nums_anc[0]  # 只支持每层相同的anc数
                # if match_index > _num_total:
                #     flog.error(
                #         '行列偏移索引:%s，最终索引:%s，anc索引:%s,最好的特图层索引:%s,'
                #         'colrow_index:%s,box:%s/%s,当前index%s' % (
                #             offset_colrow.item(),  # 210
                #             match_index.item(),  # 10770超了
                #             iou_sort[:, :9],  # [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1]
                #             match_anc_index,  # [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1]
                #             colrow_index,  #
                #             j, iou_sort.shape[0],
                #             colrow_index[j * num_anc + k_],
                #         ))

                if targets_yolo[i, match_index_s + match_anc_index, 4] == 0:  # 同一特图一个网格只有一个目标
                    targets_yolo[i, match_index_s + match_anc_index, 4] = 1
                    # targets_yolo[i, match_index_s + match_anc_index, 0:2] = offxy_xy[j * num_anc + k_]
                    # targets_yolo[i, match_index_s + match_anc_index, 2:4] = boxes_xywh[j, 2:4]
                    targets_yolo[i, match_index_s + match_anc_index, :4] = boxes[j]
                    targets_yolo[i, match_index_s + match_anc_index, labels[j] + 5 - 1] = 1  # 独热
                    targets_yolo[i, match_index_s + match_anc_index, -2:] = offxy_xy[j * num_anc + k_]
                    # 可视化
                    # f_show_iou4pil(img_pil, boxes[j][None],
                    #                xywh2ltrb(anchors_obj.ancs[match_index_s + match_anc_index][None]),
                    #                is_safe=False)
                    break
                else:
                    # 取第二大的IOU的与之匹配
                    # flog.info('找到一个重复的')
                    pass
    return targets_yolo


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


def pos_match_retinaface(ancs_ltrb, g_bboxs_ltrb, g_labels, g_keypoints, threshold_pos=0.5, threshold_neg=0.3):
    '''

    :param ancs_ltrb:
    :param g_bboxs_ltrb:[n,4] ltrb
    :param g_labels: [n]
    :param g_keypoints:
    :param threshold_pos:
    :param threshold_neg:
    :return:
    '''
    # (bboxs个,anc个)
    # print(bboxs.shape[0])
    iou = calc_iou4ts(g_bboxs_ltrb, ancs_ltrb)
    # anc对应box最大的  anc个 bbox_index
    anc_max_iou, bbox_index = iou.max(dim=0)  # 存的是 bboxs的index

    # box对应anc最大的  box个 anc_index
    bbox_max_iou, anc_index = iou.max(dim=1)  # 存的是 anc的index

    # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
    anc_max_iou.index_fill_(0, anc_index, 1)  # 最大的为正例, 强制为conf 为1

    # 一定层度上使gt均有对应的anc, 处理多个anc对一gt的问题, 若多个gt对一个anc仍不能解决(情况较少)会导致某GT学习不了...遍历每一个gt索引
    gt_ids = torch.arange(0, anc_index.size(0), dtype=torch.int64).to(bbox_index)  # [0,1]
    # gt对应最好的anc索引取出来
    anc_ids = anc_index[gt_ids]
    # 找到这些anc 写入gt
    bbox_index[anc_ids] = gt_ids

    # ----------正例的index 和 正例对应的bbox索引----------
    mask_pos = anc_max_iou >= threshold_pos  # anc 的正例 index 不管
    mask_neg = anc_max_iou <= threshold_neg  # anc 的反例 index
    mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))

    match_conf = anc_max_iou  # 这里是全局真 iou 做为conf
    match_labels = g_labels[bbox_index]
    match_bboxs = g_bboxs_ltrb[bbox_index]
    if g_keypoints is not None:
        match_keypoints = g_keypoints[bbox_index]
    else:
        match_keypoints = None

    return match_bboxs, match_keypoints, match_labels, match_conf, mask_pos, mask_neg, mash_ignore


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
