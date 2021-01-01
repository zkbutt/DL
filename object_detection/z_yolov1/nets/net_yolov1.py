import torch
from torch import nn
import torch.nn.functional as F
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_match import fmatch4yolo1, fmatch4yolo1_v2
from f_tools.fits.f_predictfun import label_nms
from f_tools.fun_od.f_boxes import offxy2xy, xywh2ltrb, calc_iou4ts


# class LossYOLOv1(nn.Module):
#
#     def __init__(self, num_cls, grid=7, num_bbox=1, threshold_box=5, threshold_conf_neg=0.5, cfg=None):
#         '''
#
#         :param grid: 7代表将图像分为7x7的网格
#         :param num_bbox: 2代表一个网格预测两个框
#         :param threshold_box:
#         :param threshold_conf_neg:
#         '''
#         super(LossYOLOv1, self).__init__()
#         self.cfg = cfg
#         self.grid = grid
#         self.num_bbox = num_bbox
#         self.num_cls = num_cls
#         self.threshold_box = threshold_box  # 5代表 λcoord  box位置的坐标预测 权重
#         self.threshold_conf_neg = threshold_conf_neg  # 0.5代表没有目标的bbox的confidence loss 权重
#
#     def calc_box_pos_loss(self, pbox, gbox):
#         '''
#
#         :param pbox: 已 sigmoid xy
#         :param gbox:
#         :return:
#         '''
#         dim = len(pbox.shape)
#         if dim == 1:
#             # loss_xy = F.mse_loss(torch.sigmoid(pbox[:2]), gbox[:2], reduction='sum')
#             loss_xy = F.mse_loss(pbox[:2], gbox[:2], reduction='sum')
#             loss_wh = F.mse_loss(pbox[2:4].abs().sqrt(), gbox[2:4].sqrt(), reduction='sum')
#         else:
#             # loss_xy = F.mse_loss(torch.sigmoid(pbox[:, :2]), gbox[:, :2], reduction='sum')
#             loss_xy = F.mse_loss(pbox[:, :2], gbox[:, :2], reduction='sum')
#             # 偏移相同的距离 小目录损失要大些
#             loss_wh = F.mse_loss(pbox[:, 2:4].abs().sqrt(), gbox[:, 2:4].sqrt(), reduction='sum')
#         return loss_xy + loss_wh
#
#     def forward(self, outs, targets, imgs_ts=None):
#         '''
#
#         :param outs: torch.Size([20, 8, 8, 11])
#         :param targets: tuple
#             target['boxes']
#             target['labels']
#             target['size']
#             target['image_id']
#         :param imgs_ts:
#         :return:
#         '''
#         cfg = self.cfg
#         device = outs.device
#         batch = outs.shape[0]
#         dim_yolo = cfg.NUM_BBOX * (4 + 1) + cfg.NUM_CLASSES
#
#         p_yolos = outs
#         g_yolos = torch.zeros((batch, cfg.NUM_GRID, cfg.NUM_GRID, dim_yolo), device=device)
#
#         for i, target in enumerate(targets):
#             boxes_one = target['boxes'].to(device)
#             labels_one = target['labels'].to(device)  # ltrb
#             g_yolo = fmatch4yolo1(boxes_ltrb=boxes_one, labels=labels_one, num_bbox=cfg.NUM_BBOX,
#                                   num_class=cfg.NUM_CLASSES, grid=cfg.NUM_GRID, device=device)
#             g_yolos[i] = g_yolo
#         '''生成有目标和没有目标的同维布尔索引'''
#         # 获取有GT的框的布尔索引集,conf为1 4或9可作任选一个,结果一样的
#         start_conf_index = self.num_bbox * 4
#         mask_pos = g_yolos[:, :, :, start_conf_index] > 0  # 同维(batch,7,7,25) -> (batch,7,7)
#         mask_neg = torch.logical_not(mask_pos)  # (batch,7,7,25) ->(batch,7,7)
#
#         # dim_yolo = self.num_bbox * 5 + self.num_cls  # g_yolo_ts.shape[-1]
#         p_yolos_pos = p_yolos[mask_pos]  # [nn, dim_yolo]
#         g_yolos_pos = g_yolos[mask_pos]
#
#         '''这个可以独立算  计算正例类别loss'''
#         # batch拉平一起算,(xxx,25) -> (xxx,20)
#         p_cls_pos = p_yolos_pos[:, self.num_bbox * 5:].contiguous()
#         g_cls_pos = g_yolos_pos[:, self.num_bbox * 5:].contiguous()  # 这里全是1
#         loss_cls_pos = F.mse_loss(p_cls_pos, g_cls_pos, reduction='sum')
#         # loss_cls_pos = F.binary_cross_entropy_with_logits(p_cls_pos, g_cls_pos, reduction='sum')
#
#         '''计算有正例的iou好的那个的 box 损失   （x,y,w开方，h开方）'''
#         # # (xxx,25) -> (xxx,4)
#         # mask_pos_ = g_yolos[:, :, :, start_conf_index] > 0  # 只要第一个匹配的GT >0即可
#         # # mask_neg_ = g_yolo_ts[:, :, :, _is] == 0
#         # p_pos_ = p_yolos[mask_pos_]
#         # g_pos_ = g_yolos[mask_pos_]
#         ds0, ds1, ds2 = torch.where(mask_pos)
#
#         p_loc2_pos = p_yolos_pos[:, :self.num_bbox * 4]
#         g_loc2_pos = g_yolos_pos[:, :self.num_bbox * 4]
#
#         loss_box_pos = torch.tensor(0., dtype=torch.float, device=device)
#         loss_conf_pos = torch.tensor(0., dtype=torch.float, device=device)
#
#         for i, (ploc2, gloc2) in enumerate(zip(p_loc2_pos, g_loc2_pos)):
#             colrow_index = torch.tensor([ds2[i], ds1[i]], device=device)
#             nums_grid = torch.tensor([self.grid] * 2, device=device)  # [7,7]
#
#             '''这个 fix 用于算 box 损失'''
#             plocs = ploc2.view(-1, 4)
#             p_offxy = torch.sigmoid(plocs[:, :2])  # 预测需要修正
#             p_offxywh = torch.cat([p_offxy, plocs[:, 2:]], dim=1)
#             g_offxywh = gloc2.view(-1, 4)
#
#             '''这些用于算正例 conf 这里只是用来做索引 找出正例box'''
#             with torch.no_grad():
#                 _xy = offxy2xy(p_offxy, colrow_index, nums_grid)
#                 pbox_xywh = torch.cat([_xy, plocs[:, 2:]], dim=1)  # 这个用于算损失
#                 _xy = offxy2xy(g_offxywh[:, :2], colrow_index, nums_grid)
#                 gbox_xywh = torch.cat([_xy, g_offxywh[:, 2:]], dim=1)
#
#                 pbox_ltrb = xywh2ltrb(pbox_xywh)  # 这个用于算IOU
#                 gbox_ltrb = xywh2ltrb(gbox_xywh)
#                 ious = calc_iou4ts(pbox_ltrb, gbox_ltrb)
#                 maxiou, ind_pbox = ious.max(dim=0)
#                 # 这里还需更新gt的 conf  匹配的为1  未匹配的为iou
#
#             if torch.all(g_offxywh[0] == g_offxywh[1]):  # 两个 GT 是一样的 取最大的IOU
#                 # 只有一个box 则只选一个pbox 修正降低 ious
#                 pbox_ = p_offxywh[ind_pbox[0]]
#                 gbox_ = g_offxywh[0]
#                 # 只计算一个conf
#                 pconf = p_yolos[ds0[i], ds1[i], ds2[i], start_conf_index + ind_pbox[1]]
#                 gconf = torch.tensor(1., dtype=torch.float, device=device)
#             else:
#                 # 两个都计算  两个都有匹配正例 conf 不作处理
#                 if ind_pbox[0] == ind_pbox[1]:
#                     # 如果 两个box 对一个 pbox 则直接相对匹配
#                     pbox_ = p_offxywh
#                     gbox_ = g_offxywh
#                     # 计算一个
#                     pconf = p_yolos[ds0[i], ds1[i], ds2[i], start_conf_index + ind_pbox[1]]
#                     gconf = torch.tensor(1., dtype=torch.float, device=device)
#                 else:
#                     # 如果 两个box对二个预测 则对应匹配
#                     pbox_ = p_offxywh[ind_pbox, :]
#                     gbox_ = g_offxywh
#                     # 计算两个
#                     pconf = p_yolos_pos[i, start_conf_index: start_conf_index + 2]
#                     gconf = torch.ones([2], dtype=torch.float, device=device)
#
#             '''---------------计算 1 个正例 box 损失------------------'''
#             loss_box_pos = loss_box_pos + self.calc_box_pos_loss(pbox_, gbox_)
#
#             '''---------------计算 2 个正例 置信度 损失------------------'''
#             loss_conf_pos = loss_conf_pos + F.mse_loss(pconf, gconf, reduction='sum')
#             # loss_conf_pos = loss_conf_pos + F.binary_cross_entropy_with_logits(pconf, gconf, reduction='sum')
#
#         '''-----------计算所有反例conf损失 7*7*batch - 正反比例 2 : 96 ---------------'''
#         p_conf_neg = p_yolos[mask_neg][:, start_conf_index:start_conf_index + 2]
#         g_conf_zero = torch.zeros_like(p_conf_neg)
#         loss_conf_neg = F.mse_loss(p_conf_neg, g_conf_zero, reduction='sum')
#         # loss_conf_neg = F.binary_cross_entropy_with_logits(p_conf_neg, g_conf_zero, reduction='sum')
#
#         loss_box = self.threshold_box * loss_box_pos / batch
#         loss_conf = (loss_conf_pos + self.threshold_conf_neg * loss_conf_neg) / batch
#         loss_cls_pos = loss_cls_pos / batch
#         loss_total = loss_box + loss_conf + loss_cls_pos
#
#         log_dict = {}
#         log_dict['loss_box'] = loss_box.item()
#         log_dict['loss_conf_pos'] = (loss_conf_pos / batch).item()
#         log_dict['loss_conf_neg'] = (loss_conf_neg / batch).item()
#         log_dict['loss_cls'] = loss_cls_pos.item()
#         log_dict['loss_total'] = loss_total.item()
#         return loss_total, log_dict
# class LossYOLOv1(nn.Module):
#
#     def __init__(self, threshold_box, threshold_conf_neg, cfg=None):
#         '''
#
#         :param threshold_box:
#         :param threshold_conf_neg:
#         '''
#         super(LossYOLOv1, self).__init__()
#         self.cfg = cfg
#         self.threshold_box = threshold_box  # 5代表 λcoord  box位置的坐标预测 权重
#         self.threshold_conf_neg = threshold_conf_neg  # 0.5代表没有目标的bbox的confidence loss 权重
#
#     def forward(self, outs, targets, imgs_ts=None):
#         '''
#
#         :param outs: torch.Size([20, 8, 8, 11]) 已 sigmoid 全部
#         :param targets: tuple
#             target['boxes']
#             target['labels']
#             target['size']
#             target['image_id']
#         :param imgs_ts:
#         :return:
#         '''
#         cfg = self.cfg
#         device = outs.device
#         batch = outs.shape[0]
#         dim_yolo = cfg.NUM_BBOX * (4 + 1) + cfg.NUM_CLASSES
#
#         start_conf_index = cfg.NUM_BBOX * 4
#         outs[:, :, :, :2] = outs[:, :, :, :2].sigmoid()
#         outs[:, :, :, 4:6] = outs[:, :, :, 4:6].sigmoid()
#         outs[:, :, :, start_conf_index:] = outs[:, :, :, start_conf_index:].sigmoid()
#
#         p_yolos = outs
#         g_yolos = torch.zeros((batch, cfg.NUM_GRID, cfg.NUM_GRID, dim_yolo), device=device)
#
#         for i, target in enumerate(targets):  # batch遍历
#             boxes_one = target['boxes'].to(device)
#             labels_one = target['labels'].to(device)  # ltrb
#             # g_yolo = fmatch4yolo1(boxes_ltrb=boxes_one, labels=labels_one, num_bbox=cfg.NUM_BBOX,
#             #                       num_class=cfg.NUM_CLASSES, grid=cfg.NUM_GRID, device=device)
#             # 返回 g_box2_xywh_pos
#             g_yolo = fmatch4yolo1_v2(boxes_ltrb=boxes_one, labels=labels_one, num_bbox=cfg.NUM_BBOX,
#                                      num_class=cfg.NUM_CLASSES, grid=cfg.NUM_GRID, device=device, img_ts=imgs_ts[i])
#             g_yolos[i] = g_yolo
#         '''生成有目标和没有目标的同维布尔索引'''
#         # 获取有GT的框的布尔索引集,conf为1 4或9可作任选一个,结果一样的
#
#         mask_pos = g_yolos[:, :, :, start_conf_index] > 0  # 同维(batch,7,7,25) -> (batch,7,7)
#         mask_neg = torch.logical_not(mask_pos)  # (batch,7,7,25) ->(batch,7,7)
#         # dim_yolo = self.num_bbox * 5 + self.num_cls  # g_yolo_ts.shape[-1]
#         p_yolos_pos = p_yolos[mask_pos]  # [nn, dim_yolo]
#         g_yolos_pos = g_yolos[mask_pos]
#
#         '''这个可以独立算  计算正例类别loss'''
#         # batch拉平一起算,(xxx,25) -> (xxx,20)
#         p_cls_pos = p_yolos_pos[:, cfg.NUM_BBOX * 5:].contiguous()
#         g_cls_pos = g_yolos_pos[:, cfg.NUM_BBOX * 5:].contiguous()  # 这里全是1
#         loss_cls_pos = F.mse_loss(p_cls_pos, g_cls_pos, reduction='sum')
#         # loss_cls_pos = F.binary_cross_entropy_with_logits(p_cls_pos, g_cls_pos, reduction='sum')
#
#         '''计算有正例的iou好的那个的 box 损失   （x,y,w开方，h开方）'''
#         ds0, ds1, ds2 = torch.where(mask_pos)
#
#         p_loc2_pos = p_yolos_pos[:, :cfg.NUM_BBOX * 4]
#         g_boxes_xywh_pos = g_yolos_pos[:, :4]  # box取一个即可
#
#         loss_box_pos = torch.tensor(0., dtype=torch.float, device=device)
#         loss_conf_pos = torch.tensor(0., dtype=torch.float, device=device)
#
#         for i, (ploc2, g_box_xywh) in enumerate(zip(p_loc2_pos, g_boxes_xywh_pos)):
#             # 这里是遍历每一个正例
#             colrow_index = torch.tensor([ds2[i], ds1[i]], device=device)
#             nums_grid = torch.tensor([cfg.NUM_GRID] * 2, device=device)  # [7,7]
#
#             '''这个 fix 用于算 box 损失'''
#             p_box2_loc = ploc2.view(-1, 4)
#             p_box2_offxy = p_box2_loc[:, :2]
#
#             '''这些用于算正例 conf 这里只是用来做索引 找出正例box'''
#             _xy = offxy2xy(p_box2_offxy, colrow_index, nums_grid)
#             pbox_xywh = torch.cat([_xy, p_box2_loc[:, 2:]], dim=1)  # 这个用于算损失
#
#             '''---------------计算 1 个正例 box 损失------------------'''
#             pbox_ltrb = xywh2ltrb(pbox_xywh)
#             gbox_ltrb = xywh2ltrb(g_box_xywh[None])
#             ious = calc_iou4ts(pbox_ltrb, gbox_ltrb)
#             max_iou, max_index = ious.max(dim=0)
#             loss_xy = F.mse_loss(pbox_xywh[max_index], g_box_xywh[None], reduction='sum')
#             loss_box_pos = loss_box_pos + loss_xy
#             # loss_xy = F.mse_loss(pbox_xywh[max_index][:, :2], g_box_xywh[:2][None], reduction='sum')
#             # loss_wh = F.mse_loss(pbox_xywh[max_index][:, 2:4].sqrt(), g_box_xywh[2:4][None].sqrt(), reduction='sum')
#             # loss_box_pos = loss_box_pos + loss_xy + loss_wh
#
#             '''---------------计算 1 个正例 置信度 损失------------------'''
#             pconf = p_yolos[ds0[i], ds1[i], ds2[i], start_conf_index + max_index]
#             gconf = torch.tensor(1., dtype=torch.float, device=device)
#             loss_conf_pos = loss_conf_pos + F.mse_loss(pconf, gconf, reduction='sum')
#
#         '''-----------计算所有反例conf损失 7*7*batch - 除了正例都在计划 正反比例 1 : 96 ---------------'''
#         p_conf_neg = p_yolos[mask_neg][:, start_conf_index:start_conf_index + 2]
#         g_conf_zero = torch.zeros_like(p_conf_neg)
#         loss_conf_neg = F.mse_loss(p_conf_neg, g_conf_zero, reduction='sum')
#
#         loss_box = self.threshold_box * loss_box_pos / batch
#         loss_conf_neg = self.threshold_conf_neg * loss_conf_neg / batch
#         loss_conf_pos = loss_conf_pos / batch
#         loss_cls_pos = loss_cls_pos / batch
#         loss_total = loss_box + loss_conf_pos + loss_conf_neg + loss_cls_pos
#
#         log_dict = {}
#         log_dict['loss_box'] = loss_box.item()
#         log_dict['loss_conf_pos'] = loss_conf_pos.item()
#         log_dict['loss_conf_neg'] = loss_conf_neg.item()
#         log_dict['loss_cls'] = loss_cls_pos.item()
#         log_dict['loss_total'] = loss_total.item()
#         return loss_total, log_dict

# class PredictYolov1(nn.Module):
#     def __init__(self, num_bbox, num_classes, num_grid, threshold_conf=0.5, threshold_nms=0.3, ):
#         super(PredictYolov1, self).__init__()
#         self.num_bbox = num_bbox
#         self.num_classes = num_classes
#         self.num_grid = num_grid
#         self.threshold_nms = threshold_nms
#         self.threshold_conf = threshold_conf
#
#     def forward(self, p_yolo_ts4, imgs_ts=None):
#         '''
#         批量处理 conf + nms
#         :param p_yolo_ts: torch.Size([5, 7, 7, 11])
#         :return:
#             ids_batch2 [nn]
#             p_boxes_ltrb2 [nn,4]
#             p_labels2 [nn]
#             p_scores2 [nn]
#         '''
#         # 确认一阶段有没有目标
#         start_conf_index = self.num_bbox * 4
#         # torch.sigmoid_(p_yolo_ts4[:, :, :, start_conf_index:])  # 处理conf 和 label
#         # torch.Size([104, 7, 7, 2])
#         mask_pos = p_yolo_ts4[:, :, :, start_conf_index: start_conf_index + 2] > self.threshold_conf
#         if not torch.any(mask_pos):  # 如果没有一个对象
#             p_scores = p_yolo_ts4[:, :, :, start_conf_index:start_conf_index + 2]
#             print('该批次没有找到目标 max:%s min:%s mean:%s' % (p_scores.max(), p_scores.min(), p_scores.mean()))
#             return [None] * 5
#
#         device = p_yolo_ts4.device
#         # batch = p_yolo_ts4.shape[0]
#
#         '''处理box'''
#         # [5, 7, 7, 8] -> [5, 7, 7, 2, 4]
#         p_boxes2_loc = p_yolo_ts4[:, :, :, :start_conf_index].view(*p_yolo_ts4.shape[:-1], self.num_bbox, 4)
#         # torch.Size([5, 7, 7, 2])
#
#         # [5, 7, 7, 2, 4]^^[5, 7, 7, 2] -> [nn,4] 全正例
#         p_boxes_loc_pos = p_boxes2_loc[mask_pos]
#         # torch.sigmoid_(_p_boxes[:, :2])
#         ids_batch1, ids_row, ids_col, ids_box = torch.where(mask_pos)
#         grids = torch.tensor([self.num_grid] * 2, device=device, dtype=torch.float)
#         colrow_index = torch.cat([ids_col[:, None], ids_row[:, None]], dim=1)
#         # 修复 p_boxes_pos
#         p_boxes_loc_pos[:, :2] = offxy2xy(p_boxes_loc_pos[:, :2], colrow_index, grids)
#         p_boxes_xywh_pos = p_boxes_loc_pos
#         p_boxes_ltrb1 = xywh2ltrb(p_boxes_xywh_pos)
#
#         '''处理 label scores'''
#         p_labels1 = []
#         p_scores1 = []
#         for i in range(p_boxes_ltrb1.shape[0]):
#             _label_start = self.num_bbox * 4 + self.num_bbox
#             _, max_index = p_yolo_ts4[ids_batch1[i], ids_row[i], ids_col[i], _label_start:].max(dim=0)
#             p_labels1.append(max_index + 1)  # 类型加1
#             _score = p_yolo_ts4[ids_batch1[i], ids_row[i], ids_col[i], self.num_bbox * 4 + ids_box[i]]
#             p_scores1.append(_score)  # 类型加1
#
#         # [5, 7, 7, 2] -> [5, 7, 7]
#         # mask_yolo = torch.any(mask_box, dim=-1)
#         p_labels1 = torch.tensor(p_labels1, device=device, dtype=torch.float)
#         p_scores1 = torch.tensor(p_scores1, device=device, dtype=torch.float)
#
#         # 分类 nms
#         ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
#                                                                     p_boxes_ltrb1,
#                                                                     p_labels1,
#                                                                     p_scores1,
#                                                                     device,
#                                                                     self.threshold_nms)
#
#         return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,

class LossYOLOv1(nn.Module):

    def __init__(self, threshold_box, threshold_conf_neg, cfg=None):
        '''

        :param threshold_box:
        :param threshold_conf_neg:
        '''
        super(LossYOLOv1, self).__init__()
        self.cfg = cfg
        self.threshold_box = threshold_box  # 5代表 λcoord  box位置的坐标预测 权重
        self.threshold_conf_neg = threshold_conf_neg  # 0.5代表没有目标的bbox的confidence loss 权重

    def compute_iou(self, box1, box2):
        '''
        同维IOU计算
        '''

        lt = torch.max(box1[:, :2], box2[:, :2])
        rb = torch.min(box1[:, 2:], box2[:, 2:])

        wh = rb - lt
        wh[wh < 0] = 0
        inter = wh[:, 0] * wh[:, 1]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        iou = inter / (area1 + area2 - inter + 1e-4)
        return iou

    def gen_anchor(self, ceil):
        '''
        mershgrid
        :param ceil:
        :return: 1,2,7,7
        '''
        w, h = ceil, ceil
        # 0~6 w -> 1,w -> h,w -> 1,h,w
        x = torch.linspace(0, w - 1, w).unsqueeze(dim=0).repeat(h, 1).unsqueeze(dim=0)
        # 0~6 h -> 1,h -> w,h -> 1,w,h -> 1,h,w
        y = torch.linspace(0, h - 1, h).unsqueeze(dim=0).repeat(w, 1).unsqueeze(dim=0).permute(0, 2, 1)
        anchor_xy = torch.cat((x, y), dim=0).view(-1, 2, h, w)
        return anchor_xy

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs:
            p_locs: torch.Size([20, 7, 7, 8])
            p_confs: torch.Size([20, 7, 7, 2])
            p_clses: torch.Size([20, 7, 7, 1])
        :param targets: tuple
            target['boxes']
            target['labels']
            target['size']
            target['image_id']
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        p_boxes_offsetxywh_grid, p_confs, p_clses = outs

        device = p_boxes_offsetxywh_grid.device
        batch = p_boxes_offsetxywh_grid.shape[0]

        # torch.Size([20, 7, 7, 4])
        g_boxes_offsetxywh_grid = torch.empty_like(p_boxes_offsetxywh_grid, device=device)
        g_confs = torch.empty_like(p_confs, device=device)  # 这个没有用 torch.Size([20, 7, 7, 2])
        g_clses = torch.empty_like(p_clses, device=device)  # torch.Size([20, 7, 7, 1])

        for i, target in enumerate(targets):  # batch遍历
            boxes_one = target['boxes'].to(device)
            labels_one = target['labels'].to(device)  # ltrb
            g_boxes_offsetxywh_grid[i], g_confs[i], g_clses[i] = fmatch4yolo1_v2(boxes_ltrb=boxes_one,
                                                                                 labels=labels_one,
                                                                                 num_bbox=cfg.NUM_BBOX,
                                                                                 num_class=cfg.NUM_CLASSES,
                                                                                 grid=cfg.NUM_GRID, device=device,
                                                                                 img_ts=imgs_ts[i])

        mask_pos = g_confs > 0  # 同维(batch,7,7,2) -> (batch,7,7,2)
        mask_pos = torch.any(mask_pos, dim=-1)  # (batch,7,7,2) -> (batch,7,7)
        # nums_pos = mask_pos.reshape(20, -1).sum(-1) # 正例数
        mask_neg = torch.logical_not(mask_pos)  # (batch,7,7,2) ->(batch,7,7) 降维

        '''计算有正例的iou好的那个的 box 损失   （x,y,w开方，h开方）'''
        # ds0, ds1, ds2 = torch.where(mask_pos)

        p_boxes_offsetxywh_grid_pos = p_boxes_offsetxywh_grid[mask_pos]  # [20, 7, 7, 8]  -> nn,8
        g_boxes_offsetxywh_grid_pos = g_boxes_offsetxywh_grid[mask_pos]  # [20, 7, 7, 8]  -> nn,8

        with torch.no_grad():
            _p_boxes_pos = p_boxes_offsetxywh_grid_pos.view(-1, 4)  # nn,8 -> nn,4
            _g_boxes_pos = g_boxes_offsetxywh_grid_pos.view(-1, 4)  # nn,8 -> nn,4
            __p_boxes_pos = _p_boxes_pos.clone().detach()
            __g_boxes_pos = _g_boxes_pos.clone().detach()
            __p_boxes_pos[:, :2] = __p_boxes_pos[:, :2] / cfg.NUM_GRID
            __g_boxes_pos[:, :2] = __g_boxes_pos[:, :2] / cfg.NUM_GRID
            pbox_ltrb = xywh2ltrb(__p_boxes_pos)
            gbox_ltrb = xywh2ltrb(__g_boxes_pos)
            # nn,nn -> 2*nn -> nn,2
            ious = self.compute_iou(pbox_ltrb, gbox_ltrb).view(-1, cfg.NUM_BBOX)

        #  nn,2 -> nn
        # max_inx = ious.argmax(dim=-1)
        max_val, max_inx = ious.max(dim=-1)
        max_inx = max_inx.unsqueeze(dim=-1)  # nn -> nn,1

        '''-----------正例conf---------------'''
        g_confs_pos = g_confs[mask_pos]  # [24, 2] ^ nn,1 = nn,1
        p_confs_pos = p_confs[mask_pos]  # nn,1
        loss_conf_pos = F.mse_loss(p_confs_pos.gather(1, max_inx), ious.gather(1, max_inx), reduction='sum')
        # loss_conf_pos = F.mse_loss(p_confs_pos.gather(1, max_inx), g_confs_pos.gather(1, max_inx), reduction='sum')

        '''-----------正例BOX损失---------------'''
        p_boxes_offsetxywh_grid_pos_max = p_boxes_offsetxywh_grid_pos.view(-1, cfg.NUM_BBOX, 4)  # nn,8 -> batch,2,4
        g_boxes_offsetxywh_grid_pos_max = g_boxes_offsetxywh_grid_pos.view(-1, cfg.NUM_BBOX, 4)  # nn,8 -> batch,2,4
        idx = max_inx.unsqueeze(dim=-1)  # nn,1 -> nn,1,1
        idx_ = idx.repeat(1, 1, 4)  # nn,1,1 -> nn,1,4
        loss_xy = F.mse_loss(p_boxes_offsetxywh_grid_pos_max.gather(1, idx_)[:, :, :2],
                             g_boxes_offsetxywh_grid_pos_max.gather(1, idx_)[:, :, :2], reduction='sum')
        loss_wh = F.mse_loss(p_boxes_offsetxywh_grid_pos_max.gather(1, idx_)[:, :, 2:4].sqrt(),
                             g_boxes_offsetxywh_grid_pos_max.gather(1, idx_)[:, :, 2:4].sqrt(), reduction='sum')
        loss_box_pos = loss_xy + loss_wh

        '''--------------网格的conf 反例损失---------'''
        _, min_inx = ious.min(dim=-1)
        min_inx = min_inx.unsqueeze(dim=-1)
        p_confs_pos_box = p_confs_pos.gather(1, min_inx)
        zeros = torch.zeros_like(p_confs_pos_box, device=device)
        loss_conf_neg_posbox = F.mse_loss(p_confs_pos_box, zeros, reduction='sum')

        '''-----------计算所有反例conf损失 7*7*batch - 除了正例都在计划 正反比例 1 : 96 ---------------'''
        p_conf_neg = p_confs[mask_neg]
        g_conf_zero = torch.zeros_like(p_conf_neg)
        loss_conf_neg = F.mse_loss(p_conf_neg, g_conf_zero, reduction='sum')

        '''-----------这个可以独立算  计算正例类别los ---------------'''
        if cfg.NUM_CLASSES == 1:
            loss_cls_pos = 0
        else:
            p_clses_pos = p_clses[mask_pos]  # [nn, dim_yolo]
            g_clses_pos = g_clses[mask_pos]
            loss_cls_pos = F.mse_loss(p_clses_pos, g_clses_pos, reduction='sum')

        loss_box = self.threshold_box * loss_box_pos / batch
        loss_conf_neg = self.threshold_conf_neg * loss_conf_neg / batch
        loss_conf_neg_posbox = loss_conf_neg_posbox / batch
        loss_conf_pos = 2 * loss_conf_pos / batch
        loss_cls_pos = loss_cls_pos / batch
        loss_total = loss_box + loss_conf_pos + loss_conf_neg + loss_cls_pos

        log_dict = {}
        log_dict['l_box'] = loss_box.item()
        log_dict['l_conf_pos'] = loss_conf_pos.item()
        log_dict['l_conf_neg'] = loss_conf_neg.item()
        log_dict['l_conf_neg_pbox'] = loss_conf_neg_posbox.item()
        if cfg.NUM_CLASSES != 1:
            log_dict['l_cls'] = loss_cls_pos.item()
        log_dict['loss_total'] = loss_total.item()
        return loss_total, log_dict


class PredictYolov1(nn.Module):
    def __init__(self, num_bbox, num_classes, num_grid, threshold_conf=0.5, threshold_nms=0.3, cfg=None):
        super(PredictYolov1, self).__init__()
        self.num_bbox = num_bbox
        self.num_classes = num_classes
        self.num_grid = num_grid
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf
        self.cfg = cfg

    def forward(self, outs, imgs_ts=None):
        '''
        批量处理 conf + nms
        :param outs:
            p_locs: torch.Size([20, 7, 7, 8])
            p_confs: torch.Size([20, 7, 7, 2])
            p_clses: torch.Size([20, 7, 7, 1])
        :return:
            ids_batch2 [nn]
            p_boxes_ltrb2 [nn,4]
            p_labels2 [nn]
            p_scores2 [nn]
        '''
        cfg = self.cfg
        p_boxes_offsetxywh_grid, p_confs, p_clses = outs
        mask_pos = p_confs > self.threshold_conf
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:%s min:%s mean:%s' % (p_confs.max().item(),
                                                       p_confs.min().item(),
                                                       p_confs.mean().item(),
                                                       ))
            return [None] * 5

        device = p_boxes_offsetxywh_grid.device
        batch = p_boxes_offsetxywh_grid.shape[0]
        ids_batch, ids_row, ids_col, ids_box = torch.where(mask_pos)

        '''处理 '''
        p_boxes_offsetxywh_pos = torch.empty((0, 4), device=device, dtype=torch.float)
        p_labels1 = []
        p_scores1 = []
        ids_batch1 = []
        for i in range(len(ids_batch)):
            _, max_index = p_clses[ids_batch[i], ids_row[i], ids_col[i]].max(dim=0)
            p_labels1.append(max_index + 1)
            _score = p_confs[ids_batch[i], ids_row[i], ids_col[i], ids_box[i]]
            p_scores1.append(_score)
            ids_batch1.append(ids_batch[i])

            # 修复xy
            _start_idx = ids_box[i] * 4
            _bbox = p_boxes_offsetxywh_grid[ids_batch[i], ids_row[i], ids_col[i], _start_idx:_start_idx + 4]
            nums_grid = torch.tensor([cfg.NUM_GRID] * 2, device=device)
            colrow_index = torch.tensor([ids_col[i], ids_row[i]], device=device)
            _bbox[:2] = offxy2xy(_bbox[:2], colrow_index, nums_grid)
            p_boxes_offsetxywh_pos = torch.cat([p_boxes_offsetxywh_pos, _bbox.unsqueeze(0)], dim=0)

        p_boxes_ltrb1 = xywh2ltrb(p_boxes_offsetxywh_pos)
        p_labels1 = torch.tensor(p_labels1, dtype=torch.float, device=device)
        p_scores1 = torch.tensor(p_scores1, dtype=torch.float, device=device)
        ids_batch1 = torch.tensor(ids_batch1, dtype=torch.float, device=device)

        # 分类 nms
        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    p_boxes_ltrb1,
                                                                    p_labels1,
                                                                    p_scores1,
                                                                    device,
                                                                    self.threshold_nms)

        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


class FPNYolov1(nn.Module):
    def __init__(self, dim_in, dim_layer=1024):
        super(FPNYolov1, self).__init__()
        self.fpn_yolov1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_layer, 3, padding=1),
            nn.BatchNorm2d(dim_layer),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(dim_layer, dim_layer, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim_layer),
            nn.LeakyReLU(),
            nn.Conv2d(dim_layer, dim_layer, 3, padding=1),
            nn.BatchNorm2d(dim_layer),
            nn.LeakyReLU(),
            nn.Conv2d(dim_layer, dim_layer, 3, padding=1),
            nn.BatchNorm2d(dim_layer),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.fpn_yolov1(x)
        return x


class Yolo_v1_1(nn.Module):
    def __init__(self, backbone, grid, num_classes, num_bbox, cfg=None):
        super(Yolo_v1_1, self).__init__()
        self.num_classes = num_classes
        self.grid = grid
        self.num_bbox = num_bbox

        self.yolo1 = backbone
        self.losser = LossYOLOv1(threshold_box=cfg.THRESHOLD_BOX,
                                 threshold_conf_neg=cfg.THRESHOLD_CONF_NEG,
                                 cfg=cfg
                                 )
        self.preder = PredictYolov1(num_bbox=num_bbox,
                                    num_classes=num_classes,
                                    num_grid=cfg.NUM_GRID,
                                    threshold_conf=cfg.THRESHOLD_PREDICT_CONF,
                                    threshold_nms=cfg.THRESHOLD_PREDICT_NMS,
                                    cfg=cfg
                                    )  # prediction

    def forward(self, x, targets=None):
        outs = self.yolo1(x)
        outs = outs.permute(0, 2, 3, 1).contiguous()  # torch.Size([-1,7,7,30])

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, None, p_labels, p_scores


class HeadYolov1(nn.Module):

    def __init__(self, dim_layer, num_bbox, num_classes, grid):
        super(HeadYolov1, self).__init__()
        self.num_classes = num_classes
        self.grid = grid
        self.num_bbox = num_bbox

        self.layer_p_loces = nn.Sequential(
            nn.Conv2d(dim_layer, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.num_bbox * 4, 1, stride=1, padding=0)
        )
        self.layer_p_confs = nn.Sequential(
            nn.Conv2d(dim_layer, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.num_bbox, 1, stride=1, padding=0)
        )
        self.layer_p_clses = nn.Sequential(
            nn.Conv2d(dim_layer, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.num_classes, 1, stride=1, padding=0)
        )

    def forward(self, x):
        batch = x.shape[0]
        p_locs = self.layer_p_loces(x).view(batch, self.num_bbox * 4, self.grid, self.grid)
        p_confs = self.layer_p_confs(x).view(batch, self.num_bbox, self.grid, self.grid)
        p_clses = self.layer_p_clses(x).view(batch, self.num_classes, self.grid, self.grid)

        p_locs = p_locs.sigmoid().permute(0, 2, 3, 1)
        p_confs = p_confs.sigmoid().permute(0, 2, 3, 1)
        p_clses = p_clses.sigmoid().permute(0, 2, 3, 1)
        return p_locs, p_confs, p_clses


class Yolo_v1(nn.Module):
    def __init__(self, backbone, dim_in, grid, num_classes, num_bbox, cfg=None):
        super(Yolo_v1, self).__init__()
        self.backbone = backbone

        # 以下是YOLOv1的最后四个卷积层
        dim_layer = 1024
        self.fpn_yolov1 = FPNYolov1(dim_in, dim_layer)

        # dim_out = self.num_bbox * (4 + 1) + self.num_classes
        # self.head_yolov1 = nn.Conv2d(dim_layer, dim_out, kernel_size=(1, 1))

        self.head_yolov1 = HeadYolov1(dim_layer, num_bbox, num_classes, grid)

        self.losser = LossYOLOv1(threshold_box=cfg.THRESHOLD_BOX,
                                 threshold_conf_neg=cfg.THRESHOLD_CONF_NEG,
                                 cfg=cfg
                                 )
        self.preder = PredictYolov1(num_bbox=num_bbox,
                                    num_classes=num_classes,
                                    num_grid=cfg.NUM_GRID,
                                    threshold_conf=cfg.THRESHOLD_PREDICT_CONF,
                                    threshold_nms=cfg.THRESHOLD_PREDICT_NMS,
                                    cfg=cfg
                                    )  # prediction

    def forward(self, x, targets=None):
        outs = self.backbone(x)  # 输出 torch.Size([1, 1280, 13, 13])
        outs = self.fpn_yolov1(outs)  # 输出torch.Size([1, 1024, 7, 7])
        outs = self.head_yolov1(outs)  # 输出torch.Size([1, 490, 7, 7])
        # outs = outs.permute(0, 2, 3, 1).contiguous()  # torch.Size([1, 11, 7, 7])

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, None, p_labels, p_scores


if __name__ == '__main__':
    pass
