import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from f_pytorch.tools_model.f_model_api import finit_weights, CBL
from f_pytorch.tools_model.fmodels.model_modules import BottleneckCSP, SAM, SPP, SPPv2
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_lossfun import f_ghmc_v3, GHMC_Loss, focalloss_v2, FocalLoss_v2, focal_loss4center2, \
    show_distribution
from f_tools.fits.f_match import fmatch4yolo1_v2, fmatch4yolov1_1, fmatch4yolov1_2, fmatch4yolov1_3
from f_tools.fits.f_predictfun import label_nms
from f_tools.fun_od.f_boxes import offxy2xy, xywh2ltrb, calc_iou4ts, calc_iou4some_dim
from f_tools.pic.f_show import f_plt_show_cv
from f_tools.yufa.x_calc_adv import f_mershgrid


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


class LossYOLOv1_1(nn.Module):

    def __init__(self, cfg=None):
        super(LossYOLOv1_1, self).__init__()
        self.cfg = cfg
        self.ghmc_loss = GHMC_Loss(momentum=0.25)

    def forward(self, pyolos, targets, imgs_ts=None):
        '''

        :param pyolos: torch.Size([32, 6, 14, 14]) [conf-1,class-20,box4]
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape
        # b,c,h,w -> b,c,hw -> b,hw,c
        pyolos = pyolos.view(batch, c, -1).permute(0, 2, 1)

        # conf-1, cls-1, box-4, weight-1  torch.Size([3, 7, 7, 12])
        dim = 1 + 1 + 4 + 1
        gyolos = torch.empty((batch, h, w, dim), device=device)
        for i, target in enumerate(targets):  # batch遍历
            boxes_ltrb_one = target['boxes']  # ltrb
            labels_one = target['labels']

            # from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
            # img_ts = f_recover_normalization4ts(imgs_ts[i])
            # from torchvision.transforms import functional as transformsF
            # img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
            # import numpy as np
            # img_np = np.array(img_pil)
            # f_plt_show_cv(img_np, gboxes_ltrb=boxes_ltrb_one.cpu()
            #               , is_recover_size=True,
            #               grids=(h, w))

            gyolos[i] = fmatch4yolov1_3(
                boxes_ltrb=boxes_ltrb_one,
                labels=labels_one,
                grid=h,  # 7
                size_in=cfg.IMAGE_SIZE,
                device=device,
                img_ts=imgs_ts[i])

            '''可视化验证'''
            if cfg.IS_VISUAL:
                gyolo_test = gyolos[i].clone()
                gyolo_test = gyolo_test.view(-1, dim)
                gconf_one = gyolo_test[:, 0]  # torch.Size([13*13, 7])
                mask_pos = gconf_one == 1
                # 这里是修复是 xy
                _xy_grid = gyolo_test[:, 2:4] + f_mershgrid(h, w, is_rowcol=False).to(device)
                hw_ts = torch.tensor((h, w), device=device)
                gyolo_test[:, 2:4] = torch.true_divide(_xy_grid, hw_ts)
                gtxywh = gyolo_test[:, 2:6][mask_pos]
                gtxywh[:, 2:4] = gtxywh[:, 2:4].exp() / 416

                from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
                img_ts = f_recover_normalization4ts(imgs_ts[i])
                from torchvision.transforms import functional as transformsF
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                import numpy as np
                img_np = np.array(img_pil)
                f_plt_show_cv(img_np, gboxes_ltrb=boxes_ltrb_one.cpu()
                              , pboxes_ltrb=xywh2ltrb(gtxywh.cpu()), is_recover_size=True,
                              grids=(h, w))

        # a = 1.05
        # pconf = a * pyolos[:, :, 0].sigmoid() - (a - 1) / 2

        pconf = pyolos[:, :, 0].sigmoid()
        # b,hw,c -> b,hw torch.Size([32, 169])
        s_ = 1 + cfg.NUM_CLASSES
        pcls = pyolos[:, :, 1:s_].permute(0, 2, 1)
        # pbox = pyolos[:, :, s_:]
        ptxty = pyolos[:, :, s_:s_ + 2]
        ptwth = pyolos[:, :, s_ + 2:]  # 这里不需要归一

        gyolos = gyolos.view(batch, -1, dim)  # b,hw,7
        gconf = gyolos[:, :, 0]  # torch.Size([5, 169])
        gcls = gyolos[:, :, 1].long()  # torch.Size([5, 169])
        # gbox = gyolos[:, :, 2:6]
        gtxty = gyolos[:, :, 2:4]  # torch.Size([5, 169, 2])
        gtwth = gyolos[:, :, 4:6]
        weight = gyolos[:, :, -1]  # torch.Size([5, 169])

        '''-----------conf 正反例损失----------'''
        _loss_val = F.mse_loss(pconf, gconf, reduction="none")
        # _loss_val = F.binary_cross_entropy_with_logits(pconf, gconf, reduction="none")
        loss_conf_pos = (_loss_val * gconf).sum(-1).mean() * cfg.LOSS_WEIGHT[0]
        loss_conf_neg = (_loss_val * torch.logical_not(gconf)).sum(-1).mean() * cfg.LOSS_WEIGHT[1]
        # loss_conf = loss_conf_pos + loss_conf_neg

        # loss_conf = (self.ghmc_loss(pconf, gconf)).sum(-1).mean()
        # loss_conf, _ = f_ghmc_v3(pconf, gconf)
        # loss_conf = loss_conf.sum(-1).mean()
        # f_focalloss_v2 = FocalLoss_v2(alpha=0.25, is_oned=True)
        # loss_conf = (f_focalloss_v2(pconf, gconf)).sum(-1).mean()
        # loss_conf = (focal_loss4center2(pconf, gconf)).sum() / gconf.sum()

        '''-----------这两个只是正例  正例计算损失,按批量----------'''
        loss_cls = (F.cross_entropy(pcls, gcls, reduction="none") * gconf).sum(-1).mean()

        # _loss_val = F.binary_cross_entropy_with_logits(pbox, gbox, reduction="none")
        # loss_box = (_loss_val.sum(-1) * gconf * weight).sum(-1).mean()

        loss_txty = (F.binary_cross_entropy_with_logits(ptxty, gtxty, reduction="none").sum(-1) * gconf * weight) \
            .sum(-1).mean()
        loss_twth = (F.mse_loss(ptwth, gtwth, reduction="none").sum(-1) * gconf * weight).sum(-1).mean()

        # loss_total = loss_conf + loss_cls + loss_box
        loss_total = loss_conf_pos + loss_conf_neg + loss_cls + loss_txty + loss_twth

        log_dict = {}
        log_dict['l_total'] = loss_total.item()
        log_dict['l_conf_pos'] = loss_conf_pos.item()
        log_dict['l_conf_neg'] = loss_conf_neg.item()
        log_dict['l_cls'] = loss_cls.item()
        # log_dict['l_box'] = loss_box.item()
        log_dict['l_xy'] = loss_txty.item()
        log_dict['l_wh'] = loss_twth.item()
        log_dict['p_max'] = pconf.max().item()
        log_dict['p_min'] = pconf.min().item()
        log_dict['p_mean'] = pconf.mean().item()
        return loss_total, log_dict


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

        :param pyolos:torch.Size([5, 8, 13, 13])
        :return:
            ids_batch2 [nn]
            p_boxes_ltrb2 [nn,4]
            p_labels2 [nn]
            p_scores2 [nn]
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape

        # b,c,h,w -> b,c,hw -> b,hw,c
        pyolos = pyolos.view(batch, c, -1).permute(0, 2, 1)
        pconf = pyolos[:, :, 0].sigmoid()  # b,hw,c -> b,hw
        # b,hw,c -> b,hw,3 -> b,hw -> b,hw
        cls_conf, plabels = pyolos[:, :, 1:1 + cfg.NUM_CLASSES].softmax(-1).max(-1)
        pscores = cls_conf * pconf

        mask_pos = pscores > cfg.THRESHOLD_PREDICT_CONF  # b,hw
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(pconf.max().item(),
                                                                          pconf.min().item(),
                                                                          pconf.mean().item(),
                                                                          ))
            return [None] * 5

        ids_batch1, _ = torch.where(mask_pos)
        ptxywh = pyolos[:, :, 1 + cfg.NUM_CLASSES:]

        '''wh sigmoid 预测'''
        # ptxywh = ptxywh.sigmoid()
        # _xy_grid = ptxywh[:, :, :2] + f_mershgrid(h, w).to(device)

        # output = torch.zeros_like(ptxywh)
        # pred = ptxywh.clone()
        # stride = 32
        # pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + f_mershgrid(h, w).to(device)
        # pred[:, :, 2:] = torch.exp(pred[:, :, 2:])
        # output[:, :, 0] = pred[:, :, 0] * stride - pred[:, :, 2] / 2
        # output[:, :, 1] = pred[:, :, 1] * stride - pred[:, :, 3] / 2
        # output[:, :, 2] = pred[:, :, 0] * stride + pred[:, :, 2] / 2
        # output[:, :, 3] = pred[:, :, 1] * stride + pred[:, :, 3] / 2
        # pboxes_ltrb1 = output / 416

        '''wh log-exp 预测 这里是修复是 xy'''
        _xy_grid = torch.sigmoid(ptxywh[:, :, :2]) + f_mershgrid(h, w, is_rowcol=False).to(device)
        hw_ts = torch.tensor((h, w), device=device)  # /13
        ptxywh[:, :, :2] = torch.true_divide(_xy_grid, hw_ts)
        ptxywh[:, :, 2:4] = torch.exp(ptxywh[:, :, 2:])/ 416

        pboxes_ltrb1 = xywh2ltrb(ptxywh)[mask_pos]
        pboxes_ltrb1 = torch.clamp(pboxes_ltrb1, min=0., max=1.)

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1

        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    pboxes_ltrb1,
                                                                    plabels1,
                                                                    pscores1,
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
    def __init__(self, backbone, cfg):
        super(Yolo_v1_1, self).__init__()
        self.backbone = backbone

        dim_layer = 512
        self.spp = nn.Sequential(
            CBL(dim_layer, 256, k=1),
            SPPv2(),
            BottleneckCSP(256 * 4, dim_layer, n=1, shortcut=False)
        )
        self.sam = SAM(dim_layer)
        self.conv_set = BottleneckCSP(dim_layer, dim_layer, n=3, shortcut=False)

        dim_out = 1 + 4 + cfg.NUM_CLASSES
        self.head = nn.Conv2d(dim_layer, dim_out, 1)

        self.losser = LossYOLOv1_1(cfg=cfg)
        self.preder = PredictYolov1_1(num_bbox=cfg.NUM_BBOX,
                                      num_classes=cfg.NUM_CLASSES,
                                      num_grid=cfg.NUM_GRID,
                                      threshold_conf=cfg.THRESHOLD_PREDICT_CONF,
                                      threshold_nms=cfg.THRESHOLD_PREDICT_NMS,
                                      cfg=cfg
                                      )

    def forward(self, x, targets=None):
        outs = self.backbone(x)
        outs = self.spp(outs)
        outs = self.sam(outs)
        outs = self.head(outs)  # torch.Size([5, 30, 7, 7])

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
    cfg.THRESHOLD_PREDICT_CONF = 0.3
    cfg.THRESHOLD_PREDICT_NMS = 0.3
    cfg.NUM_BBOX = 2
    cfg.NUM_GRID = 7
    net = Yolo_v1_1(backbone=model, cfg=cfg)
    x = torch.rand([5, 3, 224, 224])
    print(net(x).shape)
