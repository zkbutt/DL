import torch
from torch import nn
import torch.nn.functional as F

from f_pytorch.tools_model.f_model_api import finit_weights
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_match import fmatch4yolo1_v2, fmatch4yolov1_1, fmatch4yolov1_2
from f_tools.fits.f_predictfun import label_nms
from f_tools.fun_od.f_boxes import offxy2xy, xywh2ltrb, calc_iou4ts, calc_iou4some_dim


class LossYOLOv1(nn.Module):

    def __init__(self, cfg=None):
        '''

        :param threshold_box:
        :param threshold_conf_neg:
        '''
        super(LossYOLOv1, self).__init__()
        self.cfg = cfg
        # self.threshold_box = threshold_box  # 5代表 λcoord  box位置的坐标预测 权重
        # self.threshold_conf_neg = threshold_conf_neg  # 0.5代表没有目标的bbox的confidence loss 权重

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

        loss_box = cfg.LOSS_WEIGHT[0] * loss_box_pos / batch
        loss_conf_neg = cfg.LOSS_WEIGHT[1] * loss_conf_neg / batch
        loss_conf_neg_posbox = loss_conf_neg_posbox / batch
        loss_conf_pos = cfg.LOSS_WEIGHT[2] * loss_conf_pos / batch
        loss_cls_pos = cfg.LOSS_WEIGHT[3] * loss_cls_pos / batch
        loss_total = loss_box + loss_conf_pos + loss_conf_neg + loss_cls_pos

        log_dict = {}
        log_dict['p_confs max'] = p_confs.max().item()
        log_dict['p_confs min'] = p_confs.min().item()
        log_dict['p_confs mean'] = p_confs.mean().item()
        log_dict['l_box'] = loss_box.item()
        log_dict['l_conf_pos'] = loss_conf_pos.item()
        log_dict['l_conf_neg'] = loss_conf_neg.item()
        log_dict['l_conf_neg_pbox'] = loss_conf_neg_posbox.item()
        if cfg.NUM_CLASSES != 1:
            log_dict['l_cls'] = loss_cls_pos.item()
        log_dict['loss_total'] = loss_total.item()
        return loss_total, log_dict


class LossYOLOv1_1(nn.Module):

    def __init__(self, cfg=None):
        '''

        :param threshold_box:
        :param threshold_conf_neg:
        '''
        super(LossYOLOv1_1, self).__init__()
        self.cfg = cfg

    def forward(self, pyolos, targets, imgs_ts=None):
        '''
        输入 224,224  已全部sigmoid()
        :param pyolos: torch.Size([20, 7, 7, 12]) [box,conf,class,box,conf,class]
                        4+1+class+4+1+class =12
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, row, col, dim2 = pyolos.shape
        '''
        20批 1批1个GT框有两个框
        正反例 1*2 : 48*2
        rac class1:1d   conf_pos:2d     conf_neg:96d
        '''
        dim = 5 + cfg.NUM_CLASSES
        # pyolos_ = pyolos.contiguous().view(batch, row, col, cfg.NUM_BBOX, dim)

        # torch.Size([3, 7, 7, 12])
        gyolos = torch.empty_like(pyolos, device=device)

        for i, target in enumerate(targets):  # batch遍历
            boxes_one = target['boxes']
            labels_one = target['labels']  # ltrb
            gyolos[i] = fmatch4yolov1_2(
                # pboxes=pyolos_[i, :, :, :, :4],
                boxes_ltrb=boxes_one,
                labels=labels_one,
                num_bbox=cfg.NUM_BBOX,
                num_class=cfg.NUM_CLASSES,
                grid=cfg.NUM_GRID, device=device,
                img_ts=imgs_ts[i])

        gyolos = gyolos.contiguous().view(-1, dim)
        pyolos = pyolos.contiguous().view(-1, dim)

        mask_pos = gyolos[:, 4] == 1  # 同维布尔逻辑 torch.Size([1960, 6]) -> 1960
        num_pos = torch.floor_divide(mask_pos.sum(), 2)  # 正例个数 一个目标两个匹配正例 会大于批次图
        mask_neg = torch.logical_not(mask_pos)
        pyolos_pos = pyolos[mask_pos]  # torch.Size([nn, 6])
        gyolos_pos = gyolos[mask_pos]

        pyolos_neg = pyolos[mask_neg]
        gyolos_neg = gyolos[mask_neg]

        # 正例conf损失
        loss_conf_pos = F.binary_cross_entropy_with_logits(pyolos_pos[:, 4], gyolos_pos[:, 4], reduction="sum")
        # loss_conf_pos = F.mse_loss(pyolos_pos[:, 4], gyolos_pos[:, 4], reduction="sum")
        loss_conf_neg = F.binary_cross_entropy_with_logits(pyolos_neg[:, 4], gyolos_neg[:, 4], reduction="sum")
        # loss_conf_neg = F.mse_loss(pyolos_neg[:, 4], gyolos_neg[:, 4], reduction="sum")
        loss_box_pos = F.smooth_l1_loss(pyolos_pos[:, :4], gyolos_pos[:, :4], reduction="sum")
        loss_cls = F.binary_cross_entropy_with_logits(pyolos_pos[:, 5:], gyolos_pos[:, 5:], reduction="sum")
        # loss_box_pos = 0
        # loss_cls = 0

        loss_box_pos = cfg.LOSS_WEIGHT[0] * loss_box_pos / num_pos
        loss_conf_pos = cfg.LOSS_WEIGHT[1] * loss_conf_pos / num_pos
        loss_conf_neg = cfg.LOSS_WEIGHT[2] * loss_conf_neg / num_pos
        loss_cls = cfg.LOSS_WEIGHT[3] * loss_cls / num_pos
        loss_total = loss_conf_pos + loss_conf_neg + loss_box_pos + loss_cls

        log_dict = {}
        log_dict['l_box'] = loss_box_pos.item()
        log_dict['l_conf_pos'] = loss_conf_pos.item()
        log_dict['l_conf_neg'] = loss_conf_neg.item()
        log_dict['l_cls'] = loss_cls.item()
        log_dict['p_cls-max'] = pyolos_pos[:, 5:].max().item()
        log_dict['p_cls-min'] = pyolos_pos[:, 5:].min().item()
        log_dict['p_confs-max'] = pyolos[:, 4].max().item()
        log_dict['p_confs-min'] = pyolos[:, 4].min().item()
        log_dict['p_confs-mean'] = pyolos[:, 4].mean().item()
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
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(p_confs.max().item(),
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


class PredictYolov1_1(nn.Module):
    def __init__(self, num_bbox, num_classes, num_grid, threshold_conf=0.5, threshold_nms=0.3, cfg=None):
        super(PredictYolov1_1, self).__init__()
        self.num_bbox = num_bbox
        self.num_classes = num_classes
        self.num_grid = num_grid
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf
        self.cfg = cfg

    def forward(self, pyolos, imgs_ts=None):
        '''
        批量处理 conf + nms
        :param pyolos:torch.Size([20, 7, 7, 12])
        :return:
            ids_batch2 [nn]
            p_boxes_ltrb2 [nn,4]
            p_labels2 [nn]
            p_scores2 [nn]
        '''
        cfg = self.cfg
        batch, row, col, dim = pyolos.shape
        device = pyolos.device
        pyolos = pyolos.contiguous().view(batch, row, col, self.num_bbox, 5 + self.num_classes)
        # torch.Size([3, 7, 7, 2, 1])
        scores = torch.max(pyolos[:, :, :, :, 5:] * pyolos[:, :, :, :, 4:5], dim=-1, keepdim=True)[0]
        dim0, dim1, dim2, dim3, dim4 = torch.where(scores > self.threshold_conf)
        if len(dim0) == 0:  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(scores.max().item(),
                                                                          scores.min().item(),
                                                                          scores.mean().item(),
                                                                          ))
            return [None] * 5

        max_scores, max_scores_ids = scores.max(dim=-2, keepdim=True)
        idx = max_scores_ids.repeat(1, 1, 1, 1, 5 + self.num_classes)
        # 两个anc 去掉一半
        pyolos_max = torch.gather(pyolos, dim=-2, index=idx)
        pyolos_max = pyolos_max.squeeze(-2)  # 除维 [20, 7, 7,1, 6] -> [20, 7, 7, 6]

        p_boxes_offsetxywh_pos = torch.empty((0, 4), device=device, dtype=torch.float)
        p_labels1 = []
        p_scores1 = []
        ids_batch1 = []
        nums_grid = torch.tensor([cfg.NUM_GRID] * 2, device=device)

        for ds in zip(dim0, dim1, dim2, dim3, dim4):
            ids_batch1.append(ds[0])
            pyolos_ = pyolos_max[ds[0], ds[1], ds[2]]
            _, max_index = pyolos_[5:].max(dim=0)
            p_labels1.append(max_index + 1)
            p_scores1.append(scores[ds[0], ds[1], ds[2], ds[3], ds[4]])
            _bbox = pyolos_[:4]
            colrow_index = torch.tensor([ds[1], ds[2]], device=device)
            _bbox[:2] = offxy2xy(_bbox[:2], colrow_index, nums_grid)
            p_boxes_offsetxywh_pos = torch.cat([p_boxes_offsetxywh_pos, _bbox.unsqueeze(0)], dim=0)

        p_boxes_ltrb1 = xywh2ltrb(p_boxes_offsetxywh_pos)
        p_labels1 = torch.tensor(p_labels1, dtype=torch.float, device=device)
        p_scores1 = torch.tensor(p_scores1, dtype=torch.float, device=device)
        ids_batch1 = torch.tensor(ids_batch1, dtype=torch.float, device=device)

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
        finit_weights(self)

    def forward(self, x):
        x = self.fpn_yolov1(x)
        return x


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
    def __init__(self, backbone, dim_in, cfg):
        super(Yolo_v1, self).__init__()
        self.backbone = backbone

        # 以下是YOLOv1的最后四个卷积层
        dim_layer = 1024
        self.fpn_yolov1 = FPNYolov1(dim_in, dim_layer)

        # dim_out = self.num_bbox * (4 + 1) + self.num_classes
        # self.head_yolov1 = nn.Conv2d(dim_layer, dim_out, kernel_size=(1, 1))

        self.head_yolov1 = HeadYolov1(dim_layer, cfg.NUM_BBOX, cfg.NUM_CLASSES, cfg.NUM_GRID)

        self.losser = LossYOLOv1(cfg=cfg)
        self.preder = PredictYolov1(num_bbox=cfg.NUM_BBOX,
                                    num_classes=cfg.NUM_CLASSES,
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


class Yolo_v1_1(nn.Module):
    def __init__(self, backbone, dim_in, cfg, droprate=0.5, ):
        super(Yolo_v1_1, self).__init__()
        self.backbone = backbone

        # self.neck = nn.Sequential(
        #     # nn.Dropout(droprate),
        #     nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim_in),
        #     nn.LeakyReLU()
        # )

        dim_layer = 1024
        self.neck = FPNYolov1(dim_in, dim_layer)

        # 原始yolov1方式
        # self.head_yolov1 = nn.Sequential(
        #     nn.Conv2d(dim_in, cfg.NUM_BBOX * 5 + cfg.NUM_CLASSES, kernel_size=3, stride=1, padding=1),  # 每个anchor对应4个坐标
        #     nn.BatchNorm2d(cfg.NUM_BBOX * 5 + cfg.NUM_CLASSES),
        #     nn.Sigmoid()
        # )

        # 一个格子可以有两个目标?
        self.head_yolov1 = nn.Sequential(
            nn.Conv2d(dim_layer, cfg.NUM_BBOX * (5 + cfg.NUM_CLASSES), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cfg.NUM_BBOX * (5 + cfg.NUM_CLASSES)),
            nn.Sigmoid()
        )

        self.losser = LossYOLOv1_1(cfg=cfg)
        self.preder = PredictYolov1_1(num_bbox=cfg.NUM_BBOX,
                                      num_classes=cfg.NUM_CLASSES,
                                      num_grid=cfg.NUM_GRID,
                                      threshold_conf=cfg.THRESHOLD_PREDICT_CONF,
                                      threshold_nms=cfg.THRESHOLD_PREDICT_NMS,
                                      cfg=cfg
                                      )  # prediction

    def forward(self, x, targets=None):
        outs = self.backbone(x)
        outs = self.neck(outs)
        outs = self.head_yolov1(outs)
        outs = outs.permute(0, 2, 3, 1)  # (-1,7,7,30)
        # x = x.view(-1,self.num_anchors * (5 + self.num_classes))
        # x = x.reshape((-1,self.num_anchors * (5 + self.num_classes)))
        # x = x.reshape((-1,5 + self.num_classes))

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
    from torchvision import models
    from f_pytorch.tools_model.f_layer_get import ModelOut4Resnet18

    model = models.resnet18(pretrained=True)
    model = ModelOut4Resnet18(model)


    class CFG:
        pass


    cfg = CFG()
    cfg.NUM_CLASSES = 20
    cfg.NUM_BBOX = 2
    net = Yolo_v1_1(backbone=model, dim_in=model.dim_out, cfg=cfg)
    x = torch.rand([5, 3, 224, 224])
    print(net(x).shape)
