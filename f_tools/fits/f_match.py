import numpy as np
import torch

from f_tools.fun_od.f_boxes import ltrb2xywh, xy2offxy, xywh2ltrb, calc_iou4ts


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
