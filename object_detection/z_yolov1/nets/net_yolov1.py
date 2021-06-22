import torch
from torch import nn
import torch.nn.functional as F
from f_pytorch.tools_model.f_model_api import FConv2d
from f_pytorch.tools_model.fmodels.model_modules import BottleneckCSP, SAM, SPPv2
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import f_ohem, x_bce
from f_tools.fits.f_match import boxes_encode4yolo1, boxes_decode4yolo1, fmatch4yolov1
from f_tools.floss.focal_loss import focalloss
from f_tools.fun_od.f_boxes import xywh2ltrb, bbox_iou4one_2d, calc_iou4ts, bbox_iou4y, ltrb2xywh, bbox_iou4one
from f_tools.pic.f_show import f_show_od_np4plt
from f_tools.yufa.x_calc_adv import f_mershgrid


class LossYOLOv1(nn.Module):

    def __init__(self, cfg=None):
        super(LossYOLOv1, self).__init__()
        self.cfg = cfg

    def forward(self, pyolos, targets, imgs_ts=None):
        '''

        :param pyolos: torch.Size([32, 6, 14, 14]) [conf-1,class-20,box4]
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape  # torch.Size([32, 13,13, 8])
        # b,c,h,w -> b,c,hw -> b,hw,c  torch.Size([32, 169, 8])
        pyolos = pyolos.view(batch, c, -1).permute(0, 2, 1)
        s_ = 1 + cfg.NUM_CLASSES
        ptxywh = pyolos[..., s_:s_ + 4]  # torch.Size([32, 169, 4])

        # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
        gdim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4
        gyolos = torch.empty((batch, h, w, gdim), device=device)  # 每批会整体更新这里不需要赋0

        for i, target in enumerate(targets):  # batch遍历
            gboxes_ltrb_b = target['boxes']  # ltrb
            glabels_b = target['labels']

            '''
            yolo4
            1. 每层选一个匹配一个 anc与GT的IOU最大的一个
                技巧gt的xy可调整成 格子偏移与pxy匹配
            2. 其它的IOU>0.4忽略,除正例
            3. reg损失: 解码预测 pxy.sigmoid exp(pwh*anc) -> 进行IOU loss
                正例损失进行平均, 权重0.05
            4. cls损失: 
                label_smooth 标签平滑正则化, onehot* (1-0.01) + 0.01 /num_class
                pos_weight=0.5
                loss_weight=0.5 * num_classes / 80 = 0.01875
            5. conf损失:
                整体权重0.4  忽略的
            6. 每一层的损失全加起来
            '''
            gyolos[i] = fmatch4yolov1(
                gboxes_ltrb_b=gboxes_ltrb_b,
                glabels_b=glabels_b,
                grid=h,  # 7
                gdim=gdim,
                device=device,
                img_ts=imgs_ts[i],
                cfg=cfg,
                use_conf=True
            )

            '''可视化验证'''
            if cfg.IS_VISUAL:
                # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
                gyolo_test = gyolos[i].clone()  # torch.Size([32, 13, 13, 9])
                gyolo_test = gyolo_test.view(-1, gdim)
                gconf_one = gyolo_test[:, 0]
                mask_pos = gconf_one == 1  # [169]

                # torch.Size([169, 4])
                txywh_t = gyolo_test[:, 1 + cfg.NUM_CLASSES:1 + cfg.NUM_CLASSES + 4]

                # 这里是修复所有的xy
                zpxy_t = txywh_t[:, :2] + f_mershgrid(h, w, is_rowcol=False).to(device)
                hw_ts = torch.tensor((h, w), device=device)
                zpxy = torch.true_divide(zpxy_t, hw_ts)
                zpwh = torch.exp(txywh_t[:, 2:]) / hw_ts
                zpxywh_pos = torch.cat([zpxy, zpwh], dim=-1)[mask_pos]

                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                img_ts = f_recover_normalization4ts(imgs_ts[i])
                from torchvision.transforms import functional as transformsF
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                import numpy as np
                img_np = np.array(img_pil)
                f_show_od_np4plt(img_np,
                                 gboxes_ltrb=gboxes_ltrb_b.cpu(),
                                 pboxes_ltrb=xywh2ltrb(zpxywh_pos.cpu()),
                                 is_recover_size=True,
                                 grids=(h, w))

        gyolos = gyolos.view(batch, -1, gdim)  # b,hw,7
        gconf = gyolos[:, :, 0]  # torch.Size([5, 169])
        mask_pos = gconf > 0  # torch.Size([32, 169])
        # mask_pos = gconf == 1  # yolo1 gt 写死是1
        mask_neg = gconf == 0
        nums_pos = (mask_pos.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        nums_neg = (mask_neg.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        pyolos_pos = pyolos[mask_pos]  # torch.Size([32, 169, 13]) -> [nn, 13]
        gyolos_pos = gyolos[mask_pos]  # torch.Size([32, 169, 13]) -> [nn, 13]

        ''' ---------------- 类别-cls损失 ---------------- '''
        # # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
        pcls_sigmoid = pyolos[:, :, 1:s_].sigmoid()  # torch.Size([32, 169, 8])
        gcls = gyolos[:, :, 1:s_]  # torch.Size([32, 169, 13])
        _loss_val = x_bce(pcls_sigmoid, gcls, reduction="none")
        l_cls = ((_loss_val.sum(-1) * mask_pos).sum(-1) / nums_pos).mean()

        # pcls_sigmoid_pos = pyolos_pos[:, 1:s_].sigmoid()
        # gcls_pos = gyolos_pos[:, 1:s_]
        # _loss_val = x_bce(pcls_sigmoid_pos, gcls_pos, reduction="none")  # torch.Size([46, 3])
        # torch.Size([46, 3]) -> val
        # l_cls = _loss_val.sum(-1).mean()

        ''' ---------------- 类别-conf损失 ---------------- '''
        # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
        pconf_sigmoid = pyolos[:, :, 0].sigmoid()

        # ------------ conf-mse ------------''' 666666
        _loss_val = F.mse_loss(pconf_sigmoid, gconf, reduction="none")  # 用MSE效果更好
        l_conf_pos = ((_loss_val * mask_pos).sum(-1) / nums_pos).mean() * 5.
        l_conf_neg = ((_loss_val * mask_neg).sum(-1) / nums_pos).mean() * 1.

        # 效果一样 169:1
        # pos_ = _loss_val[mask_pos]
        # l_conf_pos = pos_.mean() * 1
        # l_conf_neg = _loss_val[mask_neg].mean() * 3

        # ------------ conf_ohem  ap26_26 ------------'''
        # _loss_val = x_bce(pconf_sigmoid, gconf)
        # mask_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
        # mask_neg_hard = f_ohem(_loss_val, nums_pos * 3, mask_pos=mask_pos, mash_ignore=mask_ignore)
        # l_conf_pos = ((_loss_val * mask_pos).sum(-1) / nums_pos).mean() * 3  # 正例越多反例越多
        # l_conf_neg = ((_loss_val * mask_neg_hard).sum(-1) / nums_pos).mean() * 3

        # ------------ focalloss   ------------
        # l_pos, l_neg = focalloss(pconf_sigmoid, gconf, mask_pos=mask_pos, is_debug=True, alpha=0.5)
        # l_conf_pos = (l_pos.sum(-1).sum(-1) / nums_pos).mean()
        # l_conf_neg = (l_neg.sum(-1).sum(-1) / nums_neg).mean() * 3

        log_dict = {}
        ''' ----------------回归损失   xy采用bce wh采用mes----------------- '''
        if cfg.MODE_TRAIN == 4:
            # ------------ iou损失   ------------
            # 解码pxywh 计算预测与 GT 的 iou 作为 gconf
            # preg_pos = pyolos_pos[:, s_:s_ + 4]
            # # 解码yolo1
            # pxy_pos_toff = preg_pos[..., :2].sigmoid()
            # pwh_pos = torch.exp(preg_pos[..., 2:])
            # pzxywh = torch.cat([pxy_pos_toff, pwh_pos], -1)

            # 这里是归一化的 gt
            gltrb_pos = gyolos_pos[:, s_ + 4 + 1:s_ + 4 + 1 + 4]

            ptxywh = pyolos[..., s_:s_ + 4]
            pltrb_pos = boxes_decode4yolo1(ptxywh, h, w, cfg)[mask_pos]

            iou_zg = bbox_iou4one(pltrb_pos, gltrb_pos, is_giou=True)
            # iou_zg = bbox_iou4y(xywh2ltrb4ts(pzxywh), gltrb_pos_tx, GIoU=True)
            # print(iou_zg)
            l_reg = (1 - iou_zg).mean() * 5

            ''' ---------------- loss完成 ----------------- '''
            l_total = l_conf_pos + l_conf_neg + l_cls + l_reg
            log_dict['l_reg'] = l_reg.item()
        else:
            # ------------ mse+bce   ------------ 666666
            # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
            # torch.Size([32, 169, 13])  9->实际是8
            ptxty_sigmoid = pyolos[:, :, s_:s_ + 2].sigmoid()  # 4:6
            ptwth = pyolos[:, :, s_ + 2:s_ + 4]  # 这里不需要归一

            weight = gyolos[:, :, s_ + 4]  # 这个是大小目标缩放比例
            gtxty = gyolos[:, :, s_:s_ + 2]  # torch.Size([5, 169, 2])
            gtwth = gyolos[:, :, s_ + 2:s_ + 4]

            # _loss_val = x_bce(ptxty_sigmoid, gtxty, reduction="none")
            _loss_val = F.mse_loss(ptxty_sigmoid, gtxty, reduction="none")
            l_txty = ((_loss_val.sum(-1) * mask_pos * weight).sum(-1) / nums_pos).mean()
            _loss_val = F.mse_loss(ptwth, gtwth, reduction="none")
            l_twth = ((_loss_val.sum(-1) * mask_pos * weight).sum(-1) / nums_pos).mean()

            ''' ---------------- loss完成 ----------------- '''
            l_total = l_conf_pos + l_conf_neg + l_cls + l_txty + l_twth
            log_dict['l_xy'] = l_txty.item()
            log_dict['l_wh'] = l_twth.item()

        log_dict['l_total'] = l_total.item()
        log_dict['l_conf_pos'] = l_conf_pos.item()
        log_dict['l_conf_neg'] = l_conf_neg.item()
        log_dict['l_cls'] = l_cls.item()

        log_dict['p_max'] = pconf_sigmoid.max().item()
        log_dict['p_min'] = pconf_sigmoid.min().item()
        log_dict['p_mean'] = pconf_sigmoid.mean().item()
        return l_total, log_dict


class PredictYOLOv1(Predicting_Base):
    def __init__(self, cfg=None):
        super(PredictYOLOv1, self).__init__(cfg)

    def p_init(self, pyolos):
        self.batch, self.c, self.h, self.w = pyolos.shape
        pyolos = pyolos.view(self.batch, self.c, -1).permute(0, 2, 1)
        return pyolos, pyolos.device

    def get_pscores(self, pyolos):
        # batch, c, h, w = pyolos.shape
        # b,c,h,w -> b,c,hw -> b,hw,c
        pconf = pyolos[:, :, 0].sigmoid()  # b,hw,c -> b,hw

        # b,hw,c -> b,hw,3 -> b,hw -> b,hw
        # cls_conf, plabels = pyolos[:, :, 1:1 + cfg.NUM_CLASSES].softmax(-1).max(-1)
        cls_conf, plabels = pyolos[:, :, 1:1 + self.cfg.NUM_CLASSES].sigmoid().max(-1)

        pscores = cls_conf * pconf  # torch.Size([32, 169])

        return pscores, plabels, pconf

    def get_stage_res(self, pyolos, mask_pos, pscores, plabels):
        batch, hw, c = pyolos.shape
        ids_batch1, _ = torch.where(mask_pos)

        ptxywh = pyolos[:, :, 1 + self.cfg.NUM_CLASSES:]
        ''' 预测 这里是修复是 xywh'''
        pboxes_ltrb1 = boxes_decode4yolo1(ptxywh, self.h, self.w, self.cfg)[mask_pos]

        pboxes_ltrb1.clamp_(min=0., max=1.)  # 预测不返回

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1

        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class LossYOLOv1_cr(nn.Module):
    ''' 只有cls及reg分支 '''

    def __init__(self, cfg=None):
        super(LossYOLOv1_cr, self).__init__()
        self.cfg = cfg

    def forward(self, pyolos, targets, imgs_ts=None):
        '''

        :param pyolos: torch.Size([32, 7, 13, 13]) cls-3,box-4
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape
        pyolos = pyolos.view(batch, c, -1).permute(0, 2, 1)

        # cls-num_class, txywh-4, weight-1, gltrb-4
        gdim = cfg.NUM_CLASSES + 4 + 1 + 4
        gyolos = torch.empty((batch, h, w, gdim), device=device)  # 每批会整体更新这里不需要赋0

        for i, target in enumerate(targets):  # batch遍历
            gboxes_ltrb_b = target['boxes']  # ltrb
            glabels_b = target['labels']

            gyolos[i] = fmatch4yolov1(
                gboxes_ltrb_b=gboxes_ltrb_b,
                glabels_b=glabels_b,
                grid=h,  # 7
                gdim=gdim,
                device=device,
                img_ts=imgs_ts[i],
                cfg=cfg,
                use_conf=False
            )

            '''可视化验证'''
            # if cfg.IS_VISUAL:
            #     # conf-1, cls-1, box-4, weight-1
            #     gyolo_test = gyolos[i].clone()  # torch.Size([32, 13, 13, 9])
            #     gyolo_test = gyolo_test.view(-1, gdim)
            #     gconf_one = gyolo_test[:, 0]
            #     mask_pos = gconf_one == 1  # [169]
            #
            #     # torch.Size([169, 4])
            #     gtxywh = gyolo_test[:, 1 + cfg.NUM_CLASSES:1 + cfg.NUM_CLASSES + 4]
            #
            #     # 这里是修复是 xy
            #     _xy_grid = gtxywh[:, :2] + f_mershgrid(h, w, is_rowcol=False).to(device)
            #     hw_ts = torch.tensor((h, w), device=device)
            #     gtxywh[:, :2] = torch.true_divide(_xy_grid, hw_ts)
            #     gtxywh = gtxywh[mask_pos]
            #     gtxywh[:, 2:4] = torch.exp(gtxywh[:, 2:]) / cfg.IMAGE_SIZE[0]
            #
            #     from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            #     img_ts = f_recover_normalization4ts(imgs_ts[i])
            #     from torchvision.transforms import functional as transformsF
            #     img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
            #     import numpy as np
            #     img_np = np.array(img_pil)
            #     f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b.cpu()
            #                      , pboxes_ltrb=xywh2ltrb(gtxywh.cpu()), is_recover_size=True,
            #                      grids=(h, w))

        # [32, 13, 13, 7] -> torch.Size([32, 169, 12])
        gyolos = gyolos.view(batch, -1, gdim)  # h*w
        gcls = gyolos[:, :, 0:cfg.NUM_CLASSES]  # torch.Size([5, 169])
        mask_pos_3d = gcls > 0  # torch.Size([32, 169, 3])
        mask_neg_3d = gcls == 0
        # [32, 169, 3] -> [32, 169]
        mask_pos_2d = torch.any(mask_pos_3d, dim=-1)
        # mask_pos = gconf == 1  # yolo1 gt 写死是1

        nums_pos = (mask_pos_2d.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        pyolos_pos = pyolos[mask_pos_2d]  # torch.Size([32, 169, 13]) -> [nn, 13]
        gyolos_pos = gyolos[mask_pos_2d]  # torch.Size([32, 169, 13]) -> [nn, 13]

        ''' ---------------- 类别损失 ---------------- '''
        # cls-num_class, txywh-4, weight-1, gltrb-4
        pcls_sigmoid = pyolos[:, :, 0:cfg.NUM_CLASSES].sigmoid()
        gcls = gyolos[:, :, 0:cfg.NUM_CLASSES]  # torch.Size([32, 169, 3])
        # 正反比 1:169*3
        # _loss_val = x_bce(pcls_sigmoid, gcls, reduction="none")  # torch.Size([46, 3])
        # l_cls_pos = ((_loss_val * mask_pos_3d).sum(-1).sum(-1) / nums_pos).mean()
        # l_cls_neg = ((_loss_val * mask_neg_3d).sum(-1).sum(-1) / nums_pos).mean()

        # ------------ conf-mse ------------''' 666666
        # _loss_val = F.mse_loss(pconf_sigmoid, gconf, reduction="none")  # 用MSE效果更好
        # _loss_val = x_bce(pconf_sigmoid, gconf, reduction="none")
        # l_conf_pos = ((_loss_val * mask_pos_3d).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[0]
        # l_conf_neg = ((_loss_val * mask_neg_3d).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[1]

        # ------------ conf_ohem  ap26_26 ------------'''
        # _loss_val = x_bce(pconf_sigmoid, gconf)
        # mask_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
        # mask_neg_hard = f_ohem(_loss_val, nums_pos * 3, mask_pos=mask_pos, mash_ignore=mask_ignore)
        # l_conf_pos = ((_loss_val * mask_pos).sum(-1) / nums_pos).mean() * 3  # 正例越多反例越多
        # l_conf_neg = ((_loss_val * mask_neg_hard).sum(-1) / nums_pos).mean() * 3

        # ------------ focalloss   ------------
        l_pos, l_neg = focalloss(pcls_sigmoid, gcls, mask_pos=mask_pos_2d, is_debug=True, alpha=0.75)
        l_cls_pos = (l_pos.sum(-1).sum(-1) / nums_pos).mean() * 30
        l_cls_neg = (l_neg.sum(-1).sum(-1) / nums_pos).mean() * 30

        ''' ----------------回归损失   xy采用bce wh采用mes----------------- '''
        # ------------ mse+bce   ------------ 666666
        # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
        # ptxty_sigmoid_pos = pyolos_pos[:, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 2].sigmoid()  # 这个需要归一化
        # ptwth_pos = pyolos_pos[:, cfg.NUM_CLASSES + 2:cfg.NUM_CLASSES + 4]
        #
        # # cls-num_class, txywh-4, weight-1, gltrb-4
        # # id = cfg.NUM_CLASSES + 4 +1 -1
        # weight_pos = gyolos_pos[:, cfg.NUM_CLASSES + 4]  # torch.Size([32, 845])
        # gtxty_pos = gyolos_pos[:, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 2]  # [nn]
        # gtwth_pos = gyolos_pos[:, cfg.NUM_CLASSES + 2:cfg.NUM_CLASSES + 4]
        #
        # _loss_val = x_bce(ptxty_sigmoid_pos, gtxty_pos, reduction="none")
        # l_txty = (_loss_val.sum(-1) * weight_pos).mean()
        # _loss_val = F.mse_loss(ptwth_pos, gtwth_pos, reduction="none")
        # l_twth = (_loss_val.sum(-1) * weight_pos).mean()

        # ------------ iou损失   ------------
        # 解码pxywh 计算预测与 GT 的 iou 作为 gconf
        ptxywh_pos = pyolos[:, :, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 4]
        # 这个是批量解码 3D 故解码出来再筛选
        zltrb_pos = boxes_decode4yolo1(ptxywh_pos, h, h, cfg)[mask_pos_2d]
        gltrb_pos = gyolos_pos[:, cfg.NUM_CLASSES + 4 + 1:cfg.NUM_CLASSES + 4 + 1 + 4]
        iou_zg = bbox_iou4one_2d(zltrb_pos, gltrb_pos, is_ciou=True)
        l_reg = (1 - iou_zg).mean()

        ''' ---------------- loss完成 ----------------- '''
        # loss_total = l_cls_pos + l_cls_neg + l_txty + l_twth
        loss_total = l_cls_pos + l_cls_neg + l_reg

        log_dict = {}
        log_dict['l_total'] = loss_total.item()
        log_dict['l_cls_pos'] = l_cls_pos.item()
        log_dict['l_cls_neg'] = l_cls_neg.item()
        log_dict['l_reg'] = l_reg.item()
        # log_dict['l_xy'] = l_txty.item()
        # log_dict['l_wh'] = l_twth.item()

        log_dict['p_max'] = pcls_sigmoid.max().item()
        log_dict['p_min'] = pcls_sigmoid.min().item()
        log_dict['p_mean'] = pcls_sigmoid.mean().item()
        return loss_total, log_dict


class PredictYOLOv1_cr(Predicting_Base):
    def __init__(self, cfg=None):
        super(PredictYOLOv1_cr, self).__init__(cfg)

    def p_init(self, pyolos):
        self.batch, self.c, self.h, self.w = pyolos.shape
        pyolos = pyolos.view(self.batch, self.c, -1).permute(0, 2, 1)
        return pyolos, pyolos.device

    def get_pscores(self, pyolos):
        pcls_sigmoid = pyolos[:, :, 0:self.cfg.NUM_CLASSES].sigmoid()
        pscores, plabels = pcls_sigmoid.max(-1)

        return pscores, plabels, pscores

    def get_stage_res(self, pyolos, mask_pos, pscores, plabels):
        # atch, c, h, w = pyolos.shape
        ids_batch1, _ = torch.where(mask_pos)
        ptxywh = pyolos[:, :, self.cfg.NUM_CLASSES:]

        ''' 预测 这里是修复是 xywh'''
        pboxes_ltrb1 = boxes_decode4yolo1(ptxywh, self.h, self.w, self.cfg)[mask_pos]
        pboxes_ltrb1.clamp_(min=0., max=1.)  # 预测不返回

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1

        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class YOLOv1_Net(nn.Module):
    def __init__(self, backbone, cfg, num_classes, num_reg=1):
        super(YOLOv1_Net, self).__init__()
        self.backbone = backbone

        dim_layer = backbone.dim_out  # 512
        dim256 = dim_layer // 2
        self.spp = nn.Sequential(
            FConv2d(dim_layer, dim256, k=1),
            SPPv2(),
            BottleneckCSP(dim256 * 4, dim_layer, n=1, shortcut=False)
        )
        self.sam = SAM(dim_layer)  # 有点类似 silu
        self.conv_set = BottleneckCSP(dim_layer, dim_layer, n=3, shortcut=False)

        self.head_conf_cls = nn.Conv2d(dim_layer, num_classes, 1)
        self.head_box = nn.Conv2d(dim_layer, num_reg * 4, 1)

        # 初始化分类 bias
        # num_pos = 1.5  # 正例数
        # num_tolal = cfg.NUM_GRID ** 2  # 总样本数
        # finit_conf_bias_one(self.head_conf_cls, num_tolal, num_pos, cfg.NUM_CLASSES)

    def forward(self, x, targets=None):
        outs = self.backbone(x)
        outs = self.spp(outs)
        outs = self.sam(outs)
        outs = self.conv_set(outs)
        out_conf_cls = self.head_conf_cls(outs)  # torch.Size([5, 1+3, 13, 13])
        out_box = self.head_box(outs)  # torch.Size([5, 4, 13, 13])
        outs = torch.cat([out_conf_cls, out_box], dim=1)
        return outs


class YOLOv1(nn.Module):
    def __init__(self, backbone, cfg):
        super(YOLOv1, self).__init__()
        if cfg.MODE_TRAIN == 1 or cfg.MODE_TRAIN == 4:  # base 或 IOU 损失及预测
            flog.warning('-------------- LossYOLOv1 ------------- %s', )
            self.net = YOLOv1_Net(backbone, cfg, cfg.NUM_CLASSES + 1)  # 带conf
            self.losser = LossYOLOv1(cfg=cfg)
            self.preder = PredictYOLOv1(cfg=cfg)

        elif cfg.MODE_TRAIN == 2:  # 去conf 只有cls reg分支
            flog.warning('-------------- LossYOLOv1_cr ------------- %s', )
            self.net = YOLOv1_Net(backbone, cfg, cfg.NUM_CLASSES)  # 只有cls及reg分支
            self.losser = LossYOLOv1_cr(cfg=cfg)
            self.preder = PredictYOLOv1_cr(cfg=cfg)

        elif cfg.MODE_TRAIN == 3:  # 任意分布 高级reg算法 暂时无效?
            self.net = YOLOv1_Net(backbone, cfg, cfg.NUM_CLASSES, num_reg=4 * cfg.NUM_REG)  # YOLOv1_Net
            self.losser = LossYOLOv1_cr(cfg=cfg)
            self.preder = PredictYOLOv1_cr(cfg=cfg)

    def forward(self, x, targets=None):
        outs = self.net(x)

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            '''------验证loss 待扩展------'''

            return loss_total, log_dict
        else:
            with torch.no_grad():  # 这个没用
                ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores


if __name__ == '__main__':
    from torchvision import models
    from f_pytorch.tools_model.f_layer_get import ModelOut4Resnet18

    model = models.resnet18(pretrained=True)
    model = ModelOut4Resnet18(model)


    class CFG:
        pass


    cfg = CFG()
    cfg.NUM_CLASSES = 3
    net = YOLOv1_Net(backbone=model, cfg=cfg)

    from f_pytorch.tools_model.model_look import f_look_tw

    f_look_tw(net, input=(5, 3, 416, 416), name='YOLOv1_Net')
