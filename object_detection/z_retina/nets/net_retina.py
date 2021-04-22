from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from f_pytorch.tools_model.f_model_api import f_freeze_bn
from f_pytorch.tools_model.fmodels.model_fpns import FPN_out3, FPN_out5
from f_pytorch.tools_model.fmodels.model_modules import SSH
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts

from f_tools.fits.f_predictfun import label_nms
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import f_ohem, x_bce
from f_tools.fits.f_match import boxes_decode4retina, pos_match_retina4conf, matchs_gt_b, matchs_gfl, \
    pos_match_retina4cls, boxes_encode4retina, boxes_encode4ssd, boxes_decode4ssd
from f_tools.floss.focal_loss import focalloss, focalloss_simple, quality_focal_loss, distribution_focal_loss, \
    quality_focal_loss2
from f_tools.fun_od.f_anc import FAnchors

from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2xywh
from f_tools.pic.f_show import f_show_od_ts4plt
from object_detection.z_retina.nets.net_ceng import RegressionModel, ClassificationModel


class LossRetina2(nn.Module):
    '''带conf'''

    def __init__(self, cfg, anc_obj):
        super().__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([1, 32526, 4])

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs: tuple
            ptxywh, torch.Size([2, 32526, 4])
            pcls, torch.Size([2, 32526, 4]) 已归一化 cls+1
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        ptxywh, pcategory_sigmoid = outs
        pcategory_sigmoid = pcategory_sigmoid.sigmoid()  # 统一归一化
        pconf_sigmoid = pcategory_sigmoid[:, :, 0]  # torch.Size([2, 32526])
        pcls_sigmoid = pcategory_sigmoid[:, :, 1:]  # 已sigmoid
        device = ptxywh.device
        batch, pdim1, c = ptxywh.shape

        # conf-1, cls-num_class, txywh-4, keypoint-nn  = 8 + nn
        gdim = 1 + cfg.NUM_CLASSES + 4
        if cfg.NUM_KEYPOINTS > 0:
            gdim += cfg.NUM_KEYPOINTS
        gretinas = torch.zeros((batch, pdim1, gdim), device=device)
        s_ = 1 + cfg.NUM_CLASSES  # 前面 两个是 conf-1, cls-3,

        for i in range(batch):
            # if cfg.IS_VISUAL:
            #     _img_ts = imgs_ts[i].clone()
            #     from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            #     _img_ts = f_recover_normalization4ts(_img_ts)
            #     f_show_od_ts4plt(_img_ts, gboxes_ltrb=boxes_ltrb_one.cpu(),
            #                      is_recover_size=True,
            #                      # grids=grids_ts.cpu().numpy(),
            #                      # plabels_text=pconf_b[index_match_dim].sigmoid(),
            #                      # glabels_text=colrow_index[None]
            #                      )

            gboxes_ltrb_b = targets[i]['boxes']
            glabels_b = targets[i]['labels']
            if cfg.NUM_KEYPOINTS > 0:
                gkeypoints_b = targets['keypoints']  # torch.Size([batch, 10])
            else:
                gkeypoints_b = None

            # 这里是一批的 mask_neg_b, mash_ignore_b 可能为None
            boxes_index, mask_pos_b, mask_neg_b, mash_ignore_b = matchs_gt_b(cfg,
                                                                             gboxes_ltrb_b=gboxes_ltrb_b,
                                                                             glabels_b=glabels_b,
                                                                             anc_obj=self.anc_obj,
                                                                             mode='iou',
                                                                             # mode='atss',
                                                                             ptxywh_b=ptxywh[i],
                                                                             img_ts=imgs_ts[i],
                                                                             num_atss_topk=9)

            '''正反例设置'''
            gretinas[i][mask_pos_b, 0] = torch.tensor(1., device=device)
            if mash_ignore_b is not None:
                gretinas[i][mash_ignore_b, 0] = torch.tensor(-1., device=device)

            labels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
            gretinas[i][mask_pos_b, 1:s_] = labels_b[boxes_index][mask_pos_b].type(torch.float)  # 正例才匹配
            # _gtxywh = boxes_encode4retina(cfg, self.anc_obj, gboxes_ltrb_b[boxes_index])
            # _gtxywh = boxes_encode4ssd(cfg, self.anc_obj, gboxes_ltrb_b[boxes_index])
            gretinas[i][mask_pos_b, s_:s_ + 4] = gboxes_ltrb_b[boxes_index][mask_pos_b]

            # gretinas[i] = pos_match_retina(cfg, dim=gdim, gkeypoints_b=None,
            #                                gboxes_ltrb_b=gboxes_ltrb_b, glabels_b=glabels_b, anc_obj=self.anc_obj,
            #                                ptxywh_b=ptxywh[i], img_ts=imgs_ts[i])

            # 匹配正例可视化
            if cfg.IS_VISUAL:
                _mask_pos = gretinas[i, :, 0] > 0  # 3d ->1d
                _img_ts = imgs_ts[i].clone()
                anc_ltrb = xywh2ltrb(self.anc_obj.ancs_xywh)[_mask_pos]
                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                _img_ts = f_recover_normalization4ts(_img_ts)
                flog.debug('gt数 %s , 正例数量 %s' % (gboxes_ltrb_b.shape[0], anc_ltrb.shape[0]))
                f_show_od_ts4plt(_img_ts, gboxes_ltrb=gboxes_ltrb_b.cpu()
                                 , pboxes_ltrb=anc_ltrb.cpu(), is_recover_size=True,
                                 # grids=grids_ts.cpu().numpy(),
                                 # plabels_text=pconf_b[index_match_dim].sigmoid(),
                                 # glabels_text=colrow_index[None]
                                 )

        mask_pos_2d = gretinas[:, :, 0] > 0  # torch.Size([2, 32526])
        nums_pos = (mask_pos_2d.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        # mask_neg_2d = gretinas[:, :, 0] == 0
        mask_ignore_2d = gretinas[:, :, 0] == -1
        # s_ = 1 + cfg.NUM_CLASSES

        ''' ----------------cls损失---------------- '''
        # pcls_sigmoid 已归一
        gcls = gretinas[:, :, 1:s_]
        _loss_val = x_bce(pcls_sigmoid, gcls, reduction="none")
        l_cls = ((_loss_val.sum(-1) * mask_pos_2d).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[2]

        ''' ----------------conf损失 ---------------- '''
        # pconf_sigmoid 已归一
        gconf = gretinas[:, :, 0]  # 已归一化
        _loss_val = x_bce(pconf_sigmoid, gconf, reduction="none")
        mask_neg_hard = f_ohem(_loss_val, nums_pos * 3, mask_pos=mask_pos_2d, mash_ignore=mask_ignore_2d)
        l_conf_pos = ((_loss_val * mask_pos_2d).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[0]
        l_conf_neg = ((_loss_val * mask_neg_hard).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[1]

        # l_pos, l_neg = focalloss(pconf_sigmoid, gconf, mask_pos=mask_pos_2d, mash_ignore=mask_ignore_2d,
        #                          is_debug=True, alpha=0.5)
        # l_conf_pos = (l_pos.sum(-1).sum(-1) / nums_pos).mean() * 7
        # l_conf_neg = (l_neg.sum(-1).sum(-1) / nums_pos).mean() * 7

        # l_pos, l_neg = focalloss(pconf, gconf, mask_pos=mask_pos, mash_ignore=mash_ignore,
        #                          alpha=0.25, gamma=2,
        #                          reduction='none', is_debug=True)
        # loss_conf_pos = (l_pos.sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[0]
        # loss_conf_neg = l_neg.sum(-1).mean() * cfg.LOSS_WEIGHT[1]

        ''' ---------------- 回归损失 ----------------- '''
        # 正例筛选后计算
        gboxes_ltrb_m_pos = gretinas[:, :, s_:s_ + 4][mask_pos_2d]
        ancs_xywh_m_pos = self.anc_obj.ancs_xywh.unsqueeze(0).repeat(batch, 1, 1)[mask_pos_2d]
        gtxywh_pos = boxes_encode4ssd(cfg, ancs_xywh_m_pos, ltrb2xywh(gboxes_ltrb_m_pos))
        _loss_val = F.smooth_l1_loss(ptxywh[mask_pos_2d], gtxywh_pos, reduction="none")
        l_box = _loss_val.sum(-1).mean()

        log_dict = OrderedDict()
        loss_total = l_conf_pos + l_conf_neg + l_cls + l_box

        log_dict['l_total'] = loss_total.item()
        log_dict['l_conf_pos'] = l_conf_pos.item()
        log_dict['l_conf_neg'] = l_conf_neg.item()
        log_dict['loss_cls'] = l_cls.item()
        log_dict['l_box'] = l_box.item()

        log_dict['cls_max'] = pcls_sigmoid.max().item()
        log_dict['conf_max'] = pconf_sigmoid.max().item()

        log_dict['cls_mean'] = pcls_sigmoid.mean().item()
        log_dict['conf_mean'] = pconf_sigmoid.mean().item()

        log_dict['cls_min'] = pcls_sigmoid.min().item()
        log_dict['conf_min'] = pconf_sigmoid.min().item()

        return loss_total, log_dict


class PredictRetina2(Predicting_Base):
    def __init__(self, cfg, anc_obj):
        super(PredictRetina2, self).__init__(cfg)
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([10647, 4]) 原图归一化 anc

    def p_init(self, outs):
        ptxywh, pcategory = outs
        device = ptxywh.device
        return outs, device

    def get_pscores(self, outs):
        ptxywh, pcategory = outs
        pcategory = pcategory.sigmoid()
        pconf = pcategory[:, :, 0]
        pcls = pcategory[:, :, 1:]
        cls_conf, plabels = pcls.max(-1)
        pscores = cls_conf * pconf
        # torch.Size([32, 32526, 3]) -> torch.Size([32, 32526])
        return pscores, plabels, pscores

    def get_stage_res(self, outs, mask_pos, pscores, plabels):
        ptxywh, pcls = outs
        batch, hw, c = ptxywh.shape
        ids_batch1, _ = torch.where(mask_pos)
        ancs_xywh_3d = self.anc_obj.ancs_xywh.unsqueeze(0).repeat(batch, 1, 1)
        pxywh = boxes_decode4ssd(self.cfg, ptxywh, ancs_xywh_3d)

        # pxywh = boxes_decode4retina(self.cfg, self.anc_obj, ptxywh)
        pboxes_ltrb1 = xywh2ltrb(pxywh[mask_pos])
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class Retina_Net2(nn.Module):

    def __init__(self, backbone, cfg, num_classes, out_channels_fpn=256):
        '''

        :param backbone:
        :param cfg:
        :param num_classes: 可选 num_classes 或 num_classes+1
        :param out_channels_fpn:
        '''
        super(Retina_Net2, self).__init__()
        self.backbone = backbone
        self.cfg = cfg

        self.fpn = FPN_out5(backbone.dims_out, out_channels_fpn)

        num_anc = cfg.NUMS_ANC[0]
        self.regressionModel = RegressionModel(out_channels_fpn, num_anchors=num_anc,
                                               num_reg=cfg.NUM_REG,
                                               feature_size=out_channels_fpn)
        self.classificationModel = ClassificationModel(out_channels_fpn, num_anchors=cfg.NUMS_ANC[0],
                                                       num_classes=num_classes)

        # init_weight
        self.init_weight(self.fpn)  # 这个连backbone都初始化了

    def init_weight(self, model):
        import math
        for m in model.modules():  # 这个是深度优先遍历 model.children()只有一层
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.classificationModel.output.weight.data.fill_(0)
        prior = 0.01
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        # f_freeze_bn(self)

    def forward(self, x):
        # ceng1, ceng2, ceng3 [2, 128, 52, 52]  [2, 512, 13, 13] [2, 512, 13, 13]
        outs = self.backbone(x)

        # fpn [1, 256, 52, 52] [1, 256, 26, 26] [1, 256, 13, 13] [1, 256, 7, 7] [1, 256, 4, 4]
        features = self.fpn(outs)

        self.cfg.NUMS_CENG = []
        regression = []
        for i, feature in enumerate(features):
            # 遍历每一层的输出  每层的每个网格有3个anc
            b, c, h, w = feature.shape
            regression.append(self.regressionModel(feature))
            self.cfg.NUMS_CENG.append(h * w)

        self.cfg.NUMS_CENG = self.cfg.NUMS_CENG * self.cfg.NUMS_ANC[0]
        regression = torch.cat(regression, 1)
        # regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        # torch.Size([1, 32526, 4])

        return regression, classification


class Retina2(nn.Module):
    def __init__(self, backbone, cfg, device):
        super(Retina2, self).__init__()

        ''' 组装anc '''
        ancs_scale = []
        s = 0
        for num in cfg.NUMS_ANC:
            ancs_scale.append(cfg.ANCS_SCALE[s:s + num])
            s += num

        # self.anc_obj = Anchors(device=cfg.device)
        self.anc_obj = FAnchors(cfg.IMAGE_SIZE, ancs_scale, cfg.FEATURE_MAP_STEPS,
                                anchors_clip=True, is_xymid=True, is_real_size=False,
                                device=device)

        if cfg.MODE_TRAIN == 1 or cfg.MODE_TRAIN == 3:
            self.net = Retina_Net2(backbone, cfg, cfg.NUM_CLASSES + 1)
            self.losser = LossRetina2(cfg, self.anc_obj)
            self.preder = PredictRetina2(cfg, self.anc_obj)

        elif cfg.MODE_TRAIN == 2:
            self.net = Retina_Net2(backbone, cfg, cfg.NUM_CLASSES)
            self.losser = LossRetina4cls(cfg, self.anc_obj)
            self.preder = PredictRetina4cls(cfg, self.anc_obj)

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


# ---------------------- GFL --------------------------
class LossRetina4cls(nn.Module):

    def __init__(self, cfg, anc_obj):
        super().__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([1, 32526, 4])

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs: tuple
            ptxywh, torch.Size([32, 10842, 4*8])
            pcls, torch.Size([32, 10842, 3])
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        preg_32d, pcls = outs  # torch.Size([5, 3614, 4])
        device = preg_32d.device
        batch, pdim1, c = preg_32d.shape

        ''' conf-1, cls-3,  与预测对应gt_ltrb-4 ,ious_zg-1 ,'''
        gdim = 1 + cfg.NUM_CLASSES + 4 + 1
        if cfg.NUM_KEYPOINTS > 0:
            gdim += cfg.NUM_KEYPOINTS
        gretinas = torch.empty((batch, pdim1, gdim), device=device)

        for i in range(batch):
            # if cfg.IS_VISUAL:
            #     _img_ts = imgs_ts[i].clone()
            #     from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            #     _img_ts = f_recover_normalization4ts(_img_ts)
            #     f_show_od_ts4plt(_img_ts, gboxes_ltrb=boxes_ltrb_one.cpu(),
            #                      is_recover_size=True,
            #                      # grids=grids_ts.cpu().numpy(),
            #                      # plabels_text=pconf_b[index_match_dim].sigmoid(),
            #                      # glabels_text=colrow_index[None]
            #                      )

            gboxes_ltrb_b = targets[i]['boxes']
            glabels_b = targets[i]['labels']
            if cfg.NUM_KEYPOINTS > 0:
                gkeypoints_b = targets['keypoints']  # torch.Size([batch, 10])
            else:
                gkeypoints_b = None

            # gretinas[i] = matchs_gfl(cfg, dim=gdim,
            #                          gboxes_ltrb_b=gboxes_ltrb_b, glabels_b=glabels_b, anc_obj=self.anc_obj,
            #                          mode='atss', preg_32d_b=preg_32d[i], img_ts=imgs_ts[i])

            gretinas[i] = pos_match_retina4cls(cfg, dim=gdim, gkeypoints_b=None,
                                               gboxes_ltrb_b=gboxes_ltrb_b, glabels_b=glabels_b, anc_obj=self.anc_obj,
                                               ptxywh_b=preg_32d[i], img_ts=imgs_ts[i])

            # 匹配正例可视化
            if cfg.IS_VISUAL:
                _mask_pos = gretinas[i, :, 0] > 0  # 3d ->1d
                _img_ts = imgs_ts[i].clone()
                anc_ltrb = xywh2ltrb(self.anc_obj.ancs_xywh)[_mask_pos]
                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                _img_ts = f_recover_normalization4ts(_img_ts)
                flog.debug('gt数 %s , 正例数量 %s' % (gboxes_ltrb_b.shape[0], anc_ltrb.shape[0]))
                f_show_od_ts4plt(_img_ts, gboxes_ltrb=gboxes_ltrb_b.cpu()
                                 , pboxes_ltrb=anc_ltrb.cpu(), is_recover_size=True,
                                 # grids=grids_ts.cpu().numpy(),
                                 # plabels_text=pconf_b[index_match_dim].sigmoid(),
                                 # glabels_text=colrow_index[None]
                                 )

        mask_pos_2d = gretinas[:, :, 0] > 0  # torch.Size([2, 32526])
        nums_pos = (mask_pos_2d.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        mask_neg_2d = gretinas[:, :, 0] == 0  # 不需要负例
        mash_ignore_2d = gretinas[:, :, 0] == -1  # 忽略没有
        ''' conf-1, cls-3,  与预测对应gt_ltrb-4 ,ious_zg-1 ,'''
        s_ = 1 + cfg.NUM_CLASSES

        ''' ---------------- 类别损失 ---------------- '''
        pcls_sigmoid = pcls.sigmoid()  # 统一归一化 torch.Size([5, 3614, 3])
        gcls = gretinas[:, :, 1:s_]  # torch.Size([5, 3614, 3])
        l_pos, l_neg = focalloss(pcls_sigmoid, gcls, mask_pos=mask_pos_2d, mash_ignore=mash_ignore_2d,
                                 is_debug=True, alpha=0.5)
        l_cls_pos = (l_pos.sum(-1).sum(-1) / nums_pos).mean() * 7
        l_cls_neg = (l_neg.sum(-1).sum(-1) / nums_pos).mean() * 7

        ''' ---------------- box损失 ----------------- '''
        # 正例筛选后计算
        gtxywh = gretinas[:, :, s_:s_ + 4]
        # _loss_val = F.mse_loss(ptxywh, gtxywh, reduction="none") * mask_pos.unsqueeze(-1)
        _loss_val = F.smooth_l1_loss(preg_32d, gtxywh, reduction="none") * mask_pos_2d.unsqueeze(-1)
        # _loss_val = torch.abs(ptxywh - gtxywh) * mask_pos.unsqueeze(-1)
        l_box = (_loss_val.sum(-1).sum(-1) / nums_pos).mean()

        # ''' ---------------- 正例box损失 giou与cls分数容合 ----------------- '''
        # # 这个损失加和
        # weight_cls = pcls_sigmoid.detach()
        # weight_cls = weight_cls.max(dim=-1)[0]
        # ious_zg = gretinas[:, :, 0]
        # _loss_val = (1 - ious_zg) * weight_cls * mask_pos_2d
        # loss_box = (_loss_val.sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[2]  # 2
        # # loss_box = (_loss_val.sum(-1)).mean() * cfg.LOSS_WEIGHT[2]  # 2
        #
        # ''' ---------------- 正例dfl损失 ---------------- '''
        # ''' ious_zg-1 , cls-3,  与预测对应 gt_ltrb-4 '''
        # gt_ltrb = gretinas[:, :, s_:s_ + 4]  # 匹配的回归值 在0~7之间
        # # preg_32d torch.Size([5, 3614, 32]) gt_ltrb
        # # torch.Size([5, 3614, 4])
        # _loss_val = distribution_focal_loss(cfg, preg_32d, gt_ltrb, mask_pos_2d)
        # loss_dfl = ((_loss_val.sum(-1) * mask_pos_2d).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[3]
        #
        # ''' ---------------- gfl 损失 使用IOU分---------------- '''
        # # 这个需要每一个正例平均
        # gcls4iou = gretinas[:, :, 1:s_] * ious_zg.unsqueeze(-1)
        # l_pos, l_neg = quality_focal_loss2(pcls_sigmoid, gcls4iou, mask_pos=mask_pos_2d, is_debug=True)
        # # l_pos, l_neg = quality_focal_loss(pcls_sigmoid, gcls, ious_zg, mask_pos=mask_pos, is_debug=True)
        # loss_gfl_pos = (l_pos.sum(-1).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[0]  # 0.25
        # loss_gfl_neg = (l_neg.sum(-1).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[1]

        # l_pos, l_neg = focalloss(pcls_sigmoid, gcls, mask_pos=mask_pos, is_debug=True)
        # loss_gfl_pos = (l_pos.sum(-1).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[0]
        # loss_gfl_neg = (l_neg.sum(-1).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[1]
        # loss_total = loss_gfl_pos + loss_gfl_neg + loss_dfl + loss_box

        loss_total = l_cls_pos + l_cls_neg + l_box

        log_dict = OrderedDict()
        log_dict['l_total'] = loss_total.item()
        log_dict['l_cls_pos'] = l_cls_pos.item()
        log_dict['l_cls_neg'] = l_cls_neg.item()
        log_dict['l_box'] = l_box.item()

        log_dict['cls_max'] = pcls_sigmoid.max().item()
        log_dict['cls_mean'] = pcls_sigmoid.mean().item()
        log_dict['cls_min'] = pcls_sigmoid.min().item()

        return loss_total, log_dict


class PredictRetina4cls(Predicting_Base):
    def __init__(self, cfg, anc_obj):
        super(PredictRetina4cls, self).__init__(cfg)
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([10647, 4]) 原图归一化 anc

    def p_init(self, outs):
        ptxywh, pcategory = outs
        device = ptxywh.device
        return outs, device

    def get_pscores(self, outs):
        ptxywh, pcls = outs
        pcls = pcls.sigmoid()
        pscores, plabels = pcls.max(-1)
        # torch.Size([32, 32526, 3]) -> torch.Size([32, 32526])
        return pscores, plabels, pscores

    def get_stage_res(self, outs, mask_pos, pscores, plabels):
        ptxywh, pcls = outs
        ids_batch1, _ = torch.where(mask_pos)
        pxywh = boxes_decode4retina(self.cfg, self.anc_obj, ptxywh)
        pboxes_ltrb1 = xywh2ltrb(pxywh[mask_pos])
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class Retina3(nn.Module):
    def __init__(self, backbone, cfg, device):
        super(Retina3, self).__init__()
        self.net = Retina_Net2(backbone, cfg, cfg.NUM_CLASSES)

        ancs_scale = []
        s = 0
        for num in cfg.NUMS_ANC:
            ancs_scale.append(cfg.ANCS_SCALE[s:s + num])
            s += num

        # self.anc_obj = Anchors(device=cfg.device)
        self.anc_obj = FAnchors(cfg.IMAGE_SIZE, ancs_scale, cfg.FEATURE_MAP_STEPS,
                                anchors_clip=True, is_xymid=True, is_real_size=False,
                                device=device)

        self.losser = LossRetina4cls(cfg, self.anc_obj)

        self.preder = PredictRetina4cls(cfg, self.anc_obj)

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
    class CFG:
        pass


    from f_pytorch.tools_model.f_layer_get import ModelOuts4Resnet
    from f_pytorch.tools_model.model_look import f_look_tw

    cfg = CFG()
    cfg.NUMS_ANC = [3, 3, 3]
    cfg.NUM_CLASSES = 3
    cfg.IMAGE_SIZE = (416, 416)
    cfg.ANCS_SCALE = [[0.06, 0.06], [0.122, 0.098], [0.18, 0.19],
                      [0.382, 0.251], [0.262, 0.378], [0.408, 0.529],
                      [0.621, 0.415], [0.622, 0.704], [0.85, 0.61]]

    cfg.FEATURE_MAP_STEPS = [8, 16, 32, 64, 128]
    cfg.NUM_KEYPOINTS = 5
    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)

    model = Retina_Net2(model, cfg)
    f_look_tw(model, input=(1, 3, 416, 416), name='Retina_Net')
    # f_look_summary(model, input=(3, 416, 416))
    # model(torch.randn((2, 3, 416, 416)))
