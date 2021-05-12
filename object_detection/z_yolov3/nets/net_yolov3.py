from collections import OrderedDict

import torch
import torch.nn as nn

from f_pytorch.tools_model.f_model_api import CBL, finit_conf_bias_one, Conv
from f_pytorch.tools_model.fmodels.model_modules import SPPv2, BottleneckCSP
from f_pytorch.tools_model.model_look import f_look_summary
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import boxes_decode4yolo3, fmatch4yolov5, boxes_decode4yolo5, fmatch4yolov3_iou
from f_tools.fits.f_predictfun import label_nms
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import x_bce, f_ohem
from f_tools.floss.focal_loss import focalloss

from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2xywh, calc_iou4ts, bbox_iou4one, xywh2ltrb4ts
import torch.nn.functional as F
import math

from f_tools.pic.f_show import f_show_od_ts4plt, f_show_od_np4plt
from f_tools.yufa.x_calc_adv import f_mershgrid
from object_detection.z_yolov3.nets.Layer_yolo3 import UpSample


class LossYOLO_v3(nn.Module):

    def __init__(self, cfg):
        super(LossYOLO_v3, self).__init__()
        self.cfg = cfg
        # self.focal_loss = FocalLoss_v2(alpha=cfg.FOCALLOSS_ALPHA, gamma=cfg.FOCALLOSS_GAMMA, reduction='none')

    def forward(self, p_yolo_tuple, targets, imgs_ts=None):
        ''' 只支持相同的anc数

        :param p_yolo_tuple: pconf pcls ptxywh
            pconf: torch.Size([3, 10647, 1])
            pcls: torch.Size([3, 10647, 3])
            ptxywh: torch.Size([3, 10647, 4])
        :param targets: list
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['size'] = target['size']
            target['image_id'] = int
        :return:
        '''
        cfg = self.cfg
        pconf, pcls, ptxywh = p_yolo_tuple
        device = ptxywh.device
        batch, hwa, c = ptxywh.shape  # [3, 10647, 4]

        if cfg.MODE_TRAIN == 5:
            # yolo5 conf1 + label1 + goff_xywh4 + gxywh4 =10
            gdim = 12
        else:
            # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
            gdim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4

        # h*w*anc

        gyolos = torch.empty((batch, hwa, gdim), device=device)

        # 匹配GT
        for i, target in enumerate(targets):  # batch遍历
            gboxes_ltrb_b = target['boxes']  # ltrb
            glabels_b = target['labels']

            ''' 可视化在里面 每层特图不一样'''

            if cfg.MODE_TRAIN == 5:
                gyolos[i] = fmatch4yolov5(gboxes_ltrb_b=gboxes_ltrb_b,
                                          glabels_b=glabels_b,
                                          dim=gdim,
                                          ptxywh_b=ptxywh[i],
                                          device=device, cfg=cfg,
                                          img_ts=imgs_ts[i],
                                          pconf_b=pconf[i])
            else:
                # gyolos[i] = fmatch4yolov3(gboxes_ltrb_b=gboxes_ltrb_b,
                #                           glabels_b=glabels_b,
                #                           dim=gdim,
                #                           ptxywh_b=ptxywh[i],
                #                           device=device, cfg=cfg,
                #                           img_ts=imgs_ts[i],
                #                           pconf_b=pconf[i])

                gyolos[i] = fmatch4yolov3_iou(gboxes_ltrb_b=gboxes_ltrb_b,
                                              glabels_b=glabels_b,
                                              dim=gdim,
                                              ptxywh_b=ptxywh[i],
                                              device=device, cfg=cfg,
                                              img_ts=imgs_ts[i],
                                              pconf_b=pconf[i],
                                              val_iou=0.3
                                              )

        # gyolos [3, 10647, 13] conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
        s_ = 1 + cfg.NUM_CLASSES

        gconf = gyolos[:, :, 0]  # 正例使用1

        mask_pos_2d = gconf > 0  # 同维bool索引 忽略的-1不计
        mask_neg_2d = gconf == 0  # 忽略-1 不管
        nums_pos = (mask_pos_2d.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)

        # # 使用 IOU 作为 conf 解码pxywh 计算预测与 GT 的 iou 作为 gconf
        # with torch.no_grad():  # torch.Size([3, 40, 13, 13])
        #     gyolos = gyolos.view(batch, -1, gdim)  # 4d -> 3d [3, 13, 13, 5, 13] -> [3, 169*5, 13]
        #     # mask_pos_2d = gyolos[:, :, 0] == 1  # 前面已匹配，降维运算 [3, xx, 13] -> [3, xx]
        #
        #     gltrb = gyolos[:, :, -4:]  # [3, 169*5, 13] ->  [3, 169*5, 4]
        #     pltrb = boxes_decode4yolo3(ptxywh, cfg)
        #
        #     _pltrb = pltrb.view(-1, 4)
        #     _gltrb = gltrb.view(-1, 4)  # iou 只支持2位
        #     iou_p = bbox_iou4one(_pltrb, _gltrb, is_ciou=True)  # 一一对应IOU
        #     iou_p = iou_p.view(batch, -1)  # 匹配每批的IOU [nn,1] -> [batch,nn/batch]
        #
        #     '''可视化 匹配的预测框'''
        #     debug = False  # torch.isnan(loss_conf_pos)
        #     if debug:  # debug
        #         # d0, d1 = torch.where(mask_pos_2d)  # [3,845]
        #         for i in range(batch):
        #             from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
        #             # img_ts = f_recover_normalization4ts(imgs_ts[i])
        #             # mask_ = d0 == i
        #             # _pltrb = _pltrb[d0[mask_], d1[mask_]].cpu()
        #             # _gbox_p = _gltrb[d0[mask_], d1[mask_]].cpu()
        #             _img_ts = imgs_ts[i].clone()
        #             _pltrb_show = pltrb[i][mask_pos[i]]
        #             _gltrb_show = gltrb[i][mask_pos[i]]
        #
        #             iou = bbox_iou4one(_pltrb_show, _gltrb_show)
        #             flog.debug('预测 iou %s', iou)
        #             _img_ts = f_recover_normalization4ts(_img_ts)
        #             f_show_od_ts4plt(_img_ts, gboxes_ltrb=_gltrb_show.detach().cpu()
        #                              , pboxes_ltrb=_pltrb_show.detach().cpu(), is_recover_size=True,
        #                              )
        #
        # gconf = iou_p  # 使用 iou赋值

        gyolos_pos = gyolos[mask_pos_2d]
        log_dict = {}

        if cfg.MODE_TRAIN == 5:
            # yolo5 conf1 + label1 + goff_xywh4 + gxywh4 +ancwh2 =12
            ''' ----------------cls损失 只计算正例---------------- '''
            pcls_sigmoid_pos = pcls[mask_pos_2d].sigmoid()  # 归一
            gcls_pos = gyolos_pos[:, 1:2]
            gcls_pos = labels2onehot4ts(gcls_pos - 1, cfg.NUM_CLASSES)
            _loss_val = x_bce(pcls_sigmoid_pos, gcls_pos, reduction="none")
            l_cls = _loss_val.sum(-1).mean()

            ''' ----------------box损失   iou----------------- '''
            ptxty_sigmoid_pos = ptxywh[mask_pos_2d][:, :2].sigmoid() * 2. - 0.5  # 格子偏移[-0.5 ~ 1.5]
            gxywh_pos = gyolos_pos[:, 6:10]
            _ancwh_pos = gyolos_pos[:, 10:12]

            ptwth_sigmoid_pos = (ptxywh[mask_pos_2d][:, 2:4].sigmoid() * 2) ** 2 * _ancwh_pos  # [0~4]
            pxywh_pos = torch.cat([ptxty_sigmoid_pos, ptwth_sigmoid_pos], -1)

            iou_zg = bbox_iou4one(xywh2ltrb4ts(pxywh_pos), xywh2ltrb4ts(gxywh_pos), is_giou=True)
            # iou_zg = bbox_iou4y(xywh2ltrb4ts(pzxywh), gltrb_pos_tx, GIoU=True)
            # print(iou_zg)
            l_reg = (1 - iou_zg).mean()

            ''' ----------------conf损失 ---------------- '''
            pconf_sigmoid = pconf.sigmoid().view(batch, -1)  # [3, 10647, 1] -> [3, 10647]
            gconf[mask_pos_2d] = iou_zg.detach().clamp(0)  # 使用IOU值修正正例值

            # ------------conf-mse ------------'''
            # _loss_val = F.mse_loss(pconf_sigmoid, gconf, reduction="none")
            # # _loss_val = F.binary_cross_entropy_with_logits(pconf_sigmoid, gconf, reduction="none")
            # l_conf_pos = ((_loss_val * mask_pos_2d).sum(-1) / nums_pos).mean() * 12
            # l_conf_neg = ((_loss_val * mask_neg_2d).sum(-1) / nums_pos).mean() * 2.5

            # pos_ = _loss_val[mask_pos_2d]
            # l_conf_pos = pos_.mean()
            # l_conf_neg = _loss_val[mask_neg_2d].mean()

            # ------------conf-focalloss ------------'''
            mask_ignore_2d = torch.logical_not(torch.logical_or(mask_pos_2d, mask_neg_2d))
            # l_pos, l_neg = focalloss(pconf_sigmoid, gconf, mask_pos=mask_pos_2d, mash_ignore=mask_ignore_2d,
            #                          is_debug=True)
            # l_conf_pos = (l_pos.sum(-1) / nums_pos).mean()
            # l_conf_neg = (l_neg.sum(-1) / nums_pos).mean()

            # ------------conf-ohem ------------'''
            _loss_val = x_bce(pconf_sigmoid, gconf, reduction="none")
            mask_neg_hard = f_ohem(_loss_val, nums_pos * 3, mask_pos=mask_pos_2d, mash_ignore=mask_ignore_2d)
            l_conf_pos = ((_loss_val * mask_pos_2d).sum(-1) / nums_pos).mean()
            l_conf_neg = ((_loss_val * mask_neg_hard).sum(-1) / nums_pos).mean()

            ''' ---------------- loss完成 ----------------- '''
            l_total = l_conf_pos + l_conf_neg + l_cls + l_reg

            log_dict['l_total'] = l_total.item()
            log_dict['l_conf_pos'] = l_conf_pos.item()
            log_dict['l_conf_neg'] = l_conf_neg.item()
            log_dict['l_cls'] = l_cls.item()
            log_dict['l_reg'] = l_reg.item()

        else:
            ''' ----------------cls损失---------------- '''
            pcls_sigmoid = pcls.sigmoid()  # 归一
            gcls = gyolos[:, :, 1:s_]
            _loss_val = x_bce(pcls_sigmoid, gcls, reduction="none")
            l_cls = ((_loss_val.sum(-1) * mask_pos_2d).sum(-1) / nums_pos).mean()

            ''' ----------------conf损失 ---------------- '''
            pconf_sigmoid = pconf.sigmoid().view(batch, -1)  # [3, 10647, 1] -> [3, 10647]

            # ------------conf-mse ------------'''
            # _loss_val = F.mse_loss(pconf_sigmoid, gconf, reduction="none")
            # _loss_val = F.binary_cross_entropy_with_logits(pconf_sigmoid, gconf, reduction="none")

            # l_conf_pos = ((_loss_val * mask_pos_2d).mean(-1)).mean() * 5.
            # l_conf_neg = ((_loss_val * mask_neg_2d).mean(-1)).mean() * 1.
            # l_conf_pos = ((_loss_val * mask_pos_2d).sum(-1) / nums_pos).mean() * 5.
            # l_conf_neg = ((_loss_val * mask_neg_2d).sum(-1) / nums_pos).mean() * 1.

            # l_conf_pos = _loss_val[mask_pos].mean() * 5
            # l_conf_neg = _loss_val[mask_neg].mean()

            # ------------conf-focalloss ------------'''
            mask_ignore_2d = torch.logical_not(torch.logical_or(mask_pos_2d, mask_neg_2d))
            # l_pos, l_neg = focalloss(pconf_sigmoid, gconf, mask_pos=mask_pos_2d, mash_ignore=mask_ignore_2d,
            #                          is_debug=True)
            # l_conf_pos = (l_pos.sum(-1) / nums_pos).mean()
            # l_conf_neg = (l_neg.sum(-1) / nums_pos).mean()

            # ------------conf-ohem ------------'''
            _loss_val = x_bce(pconf_sigmoid, gconf, reduction="none")
            mask_neg_hard = f_ohem(_loss_val, nums_pos * 3, mask_pos=mask_pos_2d, mash_ignore=mask_ignore_2d)
            l_conf_pos = ((_loss_val * mask_pos_2d).sum(-1) / nums_pos).mean()
            l_conf_neg = ((_loss_val * mask_neg_hard).sum(-1) / nums_pos).mean()

            ''' ----------------box损失   xy采用bce wh采用mes----------------- '''
            # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
            weight = gyolos[:, :, s_ + 4]  # torch.Size([32, 845])
            ptxty_sigmoid = ptxywh[:, :, :2].sigmoid()  # 这个需要归一化
            ptwth = ptxywh[:, :, 2:4]
            gtxty = gyolos[:, :, s_:s_ + 2]
            gtwth = gyolos[:, :, s_ + 2:s_ + 4]

            _loss_val = x_bce(ptxty_sigmoid, gtxty, reduction="none")
            l_txty = ((_loss_val.sum(-1) * mask_pos_2d * weight).sum(-1) / nums_pos).mean()
            _loss_val = F.mse_loss(ptwth, gtwth, reduction="none")
            l_twth = ((_loss_val.sum(-1) * mask_pos_2d * weight).sum(-1) / nums_pos).mean()

            l_total = l_conf_pos + l_conf_neg + l_cls + l_txty + l_twth

            log_dict['l_total'] = l_total.item()
            log_dict['l_conf_pos'] = l_conf_pos.item()
            log_dict['l_conf_neg'] = l_conf_neg.item()
            log_dict['l_cls'] = l_cls.item()
            log_dict['l_xy'] = l_txty.item()
            log_dict['l_wh'] = l_twth.item()

        # log_dict['p_max'] = pconf.max().item()
        # log_dict['p_min'] = pconf.min().item()
        # log_dict['p_mean'] = pconf.mean().item()
        return l_total, log_dict


class PredictYOLO_v3(Predicting_Base):
    def __init__(self, cfg=None):
        super(PredictYOLO_v3, self).__init__(cfg)
        self.cfg = cfg

    def p_init(self, outs):
        pconf, pcls, ptxywh = outs
        device = ptxywh.device
        return outs, device

    def get_pscores(self, outs):
        pconf, pcls, ptxywh = outs
        batch, hwa, c = ptxywh.shape
        pconf = pconf.sigmoid().view(batch, -1)  # [3, 10647, 1] -> [3, 10647]
        cls_conf, plabels = pcls.sigmoid().max(-1)
        pscores = cls_conf * pconf
        return pscores, plabels, pscores

    def get_stage_res(self, outs, mask_pos, pscores, plabels):
        pconf, pcls, ptxywh = outs
        ids_batch1, _ = torch.where(mask_pos)

        if self.cfg.MODE_TRAIN == 5:
            pltrb = boxes_decode4yolo5(ptxywh, self.cfg)
        else:
            pltrb = boxes_decode4yolo3(ptxywh, self.cfg)

        pboxes_ltrb1 = pltrb[mask_pos]
        # pboxes_ltrb1 = torch.clamp(pboxes_ltrb1, min=0., max=1.)

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class Yolo_v3_Net(nn.Module):

    def __init__(self, backbone, cfg):
        super(Yolo_v3_Net, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        dims_out = backbone.dims_out  # (128, 256, 512)

        # s32
        self.conv_set_3 = nn.Sequential(
            CBL(dims_out[2], dims_out[1], 1, leakyReLU=True),
            CBL(dims_out[1], dims_out[2], 3, padding=1, leakyReLU=True),
            CBL(dims_out[2], dims_out[1], 1, leakyReLU=True),
            CBL(dims_out[1], dims_out[2], 3, padding=1, leakyReLU=True),
            CBL(dims_out[2], dims_out[1], 1, leakyReLU=True),
        )

        self.conv_1x1_3 = CBL(dims_out[1], dims_out[0], 1, leakyReLU=True)
        self.extra_conv_3 = CBL(dims_out[1], dims_out[2], 3, padding=1, leakyReLU=True)
        self.pred_3 = nn.Conv2d(dims_out[2], cfg.NUMS_ANC[2] * (1 + 4 + cfg.NUM_CLASSES), 1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            CBL(dims_out[1] + dims_out[0], dims_out[0], 1, leakyReLU=True),
            CBL(dims_out[0], dims_out[1], 3, padding=1, leakyReLU=True),
            CBL(dims_out[1], dims_out[0], 1, leakyReLU=True),
            CBL(dims_out[0], dims_out[1], 3, padding=1, leakyReLU=True),
            CBL(dims_out[1], dims_out[0], 1, leakyReLU=True),
        )
        dim128 = int(dims_out[0] / 2)

        self.conv_1x1_2 = CBL(dims_out[0], dim128, 1, leakyReLU=True)
        self.extra_conv_2 = CBL(dims_out[0], dims_out[1], 3, padding=1, leakyReLU=True)
        self.pred_2 = nn.Conv2d(dims_out[1], cfg.NUMS_ANC[1] * (1 + 4 + cfg.NUM_CLASSES), 1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            CBL(dims_out[0] + dim128, dim128, 1, leakyReLU=True),
            CBL(dim128, dims_out[0], 3, padding=1, leakyReLU=True),
            CBL(dims_out[0], dim128, 1, leakyReLU=True),
            CBL(dim128, dims_out[0], 3, padding=1, leakyReLU=True),
            CBL(dims_out[0], dim128, 1, leakyReLU=True),
        )
        self.extra_conv_1 = CBL(dim128, dims_out[0], 3, padding=1, leakyReLU=True)
        self.pred_1 = nn.Conv2d(dims_out[0], cfg.NUMS_ANC[0] * (1 + 4 + cfg.NUM_CLASSES), 1)

        # 初始化分类 bias
        # num_pos = 1.5  # 正例数
        # num_tolal = 10  # 总样本数
        # finit_conf_bias_one(self.pred_3, num_tolal, num_pos, cfg.NUM_CLASSES)
        # finit_conf_bias_one(self.pred_2, num_tolal, num_pos, cfg.NUM_CLASSES)
        # finit_conf_bias_one(self.pred_1, num_tolal, num_pos, cfg.NUM_CLASSES)

    def forward(self, x):
        # [2, 256, 52, 52]  [2, 512, 26, 26]    [2, 1024, 13, 13]
        fmp_1, fmp_2, fmp_3 = self.backbone(x)

        # fpn
        fmp_3 = self.conv_set_3(fmp_3)
        fmp_3_up = F.interpolate(self.conv_1x1_3(fmp_3), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = self.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(self.conv_1x1_2(fmp_2), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = self.conv_set_1(fmp_1)

        # head
        # s = 32
        fmp_3 = self.extra_conv_3(fmp_3)
        pred_3 = self.pred_3(fmp_3)

        # s = 16
        fmp_2 = self.extra_conv_2(fmp_2)
        pred_2 = self.pred_2(fmp_2)

        # s = 8
        fmp_1 = self.extra_conv_1(fmp_1)
        pred_1 = self.pred_1(fmp_1)

        # 3*(1+3+4)  [2, 24, 52, 52]  [2, 24, 26, 26]  [2, 24, 13, 13]
        preds = [pred_1, pred_2, pred_3]
        pconf_l = []
        pcls_l = []
        ptxywh_l = []
        self.cfg.NUMS_CENG = []  # [2704, 676, 169] 这个用于后面匹配
        for i, pred in enumerate(preds):
            # 遍历每一层的输出  每层的每个网格有3个anc
            b, c, h, w = pred.shape
            self.cfg.NUMS_CENG.append(h * w)
            # [b, c, h, w] -> [b, h, w, c] -> [b, h*w, c]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
            # 24 =3*(1+3+4)
            s_ = self.cfg.NUMS_ANC[i]
            pconf_ = pred[:, :, :s_].reshape(b, -1, 1)  # 前3是conf
            pcls_ = pred[:, :, s_:s_ + s_ * self.cfg.NUM_CLASSES].reshape(b, -1, self.cfg.NUM_CLASSES)
            ptxywh_ = pred[:, :, -(s_ * 4):].reshape(b, -1, 4)
            pconf_l.append(pconf_)
            pcls_l.append(pcls_)
            ptxywh_l.append(ptxywh_)

        # 从大尺寸层开始 每个格子3个anc
        pconf = torch.cat(pconf_l, 1)
        pcls = torch.cat(pcls_l, 1)
        ptxywh = torch.cat(ptxywh_l, 1)

        return pconf, pcls, ptxywh


class Yolo_v3_Net_Slim(nn.Module):

    def forward(self, x):
        return x + self.block(x)

    def __init__(self, backbone, cfg):
        super(Yolo_v3_Net_Slim, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        dims_out = backbone.dims_out  # (128, 256, 512)

        # SPP
        self.spp = nn.Sequential(
            Conv(dims_out[2], dims_out[1], k=1),
            SPPv2(),
            BottleneckCSP(dims_out[1] * 4, dims_out[2], n=1, shortcut=False)
        )

        # head
        self.head_conv_0 = Conv(dims_out[2], dims_out[1], k=1)  # 10
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(dims_out[1] + dims_out[1], dims_out[1], n=1, shortcut=False)

        # P3/8-small
        self.head_conv_1 = Conv(dims_out[1], dims_out[0], k=1)  # 14
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(dims_out[0] + dims_out[0], dims_out[0], n=1, shortcut=False)

        # P4/16-medium
        self.head_conv_2 = Conv(dims_out[0], dims_out[0], k=3, p=1, s=2)
        self.head_csp_2 = BottleneckCSP(dims_out[0] + dims_out[0], dims_out[1], n=1, shortcut=False)

        # P8/32-large
        self.head_conv_3 = Conv(dims_out[1], dims_out[1], k=3, p=1, s=2)
        self.head_csp_3 = BottleneckCSP(dims_out[1] + dims_out[1], dims_out[2], n=1, shortcut=False)

        # det conv
        self.head_det_1 = nn.Conv2d(dims_out[0], cfg.NUMS_ANC[0] * (1 + cfg.NUM_CLASSES + 4), 1)
        self.head_det_2 = nn.Conv2d(dims_out[1], cfg.NUMS_ANC[1] * (1 + cfg.NUM_CLASSES + 4), 1)
        self.head_det_3 = nn.Conv2d(dims_out[2], cfg.NUMS_ANC[2] * (1 + cfg.NUM_CLASSES + 4), 1)

    def forward(self, x):
        # [2, 128, 52, 52]  [2, 256, 26, 26]    [2, 512, 13, 13]
        c3, c4, c5 = self.backbone(x)

        # neck
        c5 = self.spp(c5)

        # FPN + PAN
        # head
        c6 = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)  # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)  # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        # det
        pred_s = self.head_det_1(c13)
        pred_m = self.head_det_2(c16)
        pred_l = self.head_det_3(c19)
        preds = [pred_s, pred_m, pred_l]

        pconf_list = []
        pcls_list = []
        preg_list = []
        self.cfg.NUMS_CENG = []

        for i, pred in enumerate(preds):
            # 遍历每一层的输出  每层的每个网格有3个anc
            b, c, h, w = pred.shape
            self.cfg.NUMS_CENG.append(h * w)
            # [b, c, h, w] -> [b, h, w, c] -> [b, h*w, c]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
            # 24 =3*(1+3+4)  3 9 12
            s_ = self.cfg.NUMS_ANC[i]
            # 前3
            pconf_ = pred[:, :, :s_].reshape(b, -1, 1)  # 前3是conf
            # 3~12
            pcls_ = pred[:, :, s_:s_ + s_ * self.cfg.NUM_CLASSES].reshape(b, -1, self.cfg.NUM_CLASSES)
            ptxywh_ = pred[:, :, -(s_ * 4):].reshape(b, -1, 4)
            pconf_list.append(pconf_)
            pcls_list.append(pcls_)
            preg_list.append(ptxywh_)

        # 从大尺寸层开始 每个格子3个anc
        pconf = torch.cat(pconf_list, 1)
        pcls = torch.cat(pcls_list, 1)
        ptxywh = torch.cat(preg_list, 1)

        return pconf, pcls, ptxywh


class Yolo_v3(nn.Module):
    def __init__(self, backbone, cfg):
        '''
        层属性可以是 nn.Module nn.ModuleList(封装Sequential) nn.Sequential
        '''
        super(Yolo_v3, self).__init__()
        self.net = Yolo_v3_Net(backbone, cfg)
        # self.net = Yolo_v3_Net_Slim(backbone, cfg)

        self.losser = LossYOLO_v3(cfg)
        self.preder = PredictYOLO_v3(cfg)

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
        # if torch.jit.is_scripting():  # 这里是生产环境部署


if __name__ == '__main__':
    class CFG:
        pass


    from f_pytorch.tools_model.backbones.darknet import darknet53
    from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet53

    cfg = CFG()
    cfg.NUM_ANC = 9
    cfg.NUM_CLASSES = 3
    model = darknet53(pretrained=True, device='cpu')  # 2
    model = ModelOuts4DarkNet53(model)

    model = Yolo_v3_Net(model, cfg)
    f_look_summary(model, input=(3, 416, 416))
