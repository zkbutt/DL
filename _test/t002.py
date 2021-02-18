import sys

import torch

# datas_dt：id-1 cls-1 score-1 ltrb-4 match_id-1
from f_tools.fun_od.f_boxes import calc_iou4ts

datas_dt = torch.tensor([
    [2, 2, 0.4, 21, 21, 22, 22, 0],
    [2, 2, 0.5, 21, 21, 22, 22, 0],
    [0, 1, 0.4, 1, 1, 2, 2, 0],
    [0, 1, 0.4, 1, 1, 2, 2, 0],
    [0, 2, 0.6, 1.2, 1.2, 1.8, 1.8, 0],
    [1, 2, 0.9, 11, 11, 12, 12, 0],
    [1, 2, 0.5, 5.2, 5.2, 6.1, 6.1, 0],
    [0, 3, 0.5, 5.2, 5.2, 6.1, 6.1, 0],
])
# datas_gt：id-1 cls-1 ltrb-4
datas_gt = torch.tensor([
    [0., 1, 1, 1, 2, 2],
    [0., 2, 3, 3, 4, 4],
    [0., 3, 5, 5, 6, 6],
    [1, 2, 11, 11, 12, 12],
    [1, 2, 13, 13, 14, 14],
    [2, 2, 21, 21, 22, 22],
])

# 参数
gids = [0, 1, 2]
iou_std = 0.5
clses = [1, 2, 3]
debug = True

for gid in gids:  # 每一张图
    mask_gid = datas_dt[:, 0] == gid
    datas_dt_gid = datas_dt[mask_gid]
    datas_gt_gid = datas_gt[datas_gt[:, 0] == gid]
    ious = calc_iou4ts(datas_dt_gid[:, 3:7], datas_gt_gid[:, 2:6])
    # print(ious)
    for i in range(len(datas_gt_gid)):
        cls = datas_gt_gid[i][1]
        # clses.add(int(cls.item()))
        # 类别cls-1d
        mask_cls = datas_dt_gid[:, 1] == cls
        mask_nomatch = datas_dt_gid[:, -1] != 1
        mask_iou = ious[:, i] > iou_std
        _mask = torch.logical_and(mask_cls, mask_nomatch)
        _mask = torch.logical_and(_mask, mask_iou)
        if torch.any(_mask):
            # 分数
            index_score = datas_dt_gid[:, 2][_mask].max(0)[1]
            # datas_dt_gid[index_score, -1] = 1
            dim0 = torch.where(mask_gid)
            datas_dt[dim0[0][index_score], -1] = 1

    if not debug:
        continue
    for c in clses:  # 每个图片的每个类的 tp情况 debug
        num_gt = (datas_gt_gid[:, 1] == c).sum()
        mask_gid_cls = datas_dt_gid[:, 1] == c
        tp = (torch.logical_and(mask_gid_cls, datas_dt_gid[:, -1] == 1)).sum()
        fp = mask_gid_cls.sum() - tp
        fn = num_gt - tp
        print('gid=%s, cls=%s, tp=%s, fp=%s, fn=%s' % (gid, c, tp.item(), fp.item(), fn.item()))

# 总计算每类的 TP
for c in clses:  # 每个图片的每个类的 tp情况 debug
    num_gt = (datas_gt[:, 1] == c).sum().item()
    mask_gid_cls = datas_dt[:, 1] == c
    tp = (torch.logical_and(mask_gid_cls, datas_dt[:, -1] == 1)).sum().item()
    fp = mask_gid_cls.sum().item() - tp
    fn = num_gt - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / max((sys.float_info.min, precision + recall))
    print('总TP cls=%s, tp=%s, fp=%s, fn=%s, precision=%s, recall=%s, f1=%s'
          % (c, tp, fp, fn, precision, recall, f1_score))
print(datas_dt)