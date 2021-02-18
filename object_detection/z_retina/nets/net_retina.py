import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from f_pytorch.tools_model.f_model_api import f_freeze_bn
from f_pytorch.tools_model.fmodels.model_fpns import FPNv1, FPNv2, RegressionModel, ClassificationModel
from f_pytorch.tools_model.fmodels.model_modules import SSH

from f_tools.fits.f_predictfun import label_nms
from f_tools.floss.f_lossfun import f_ohem, x_bce
from f_tools.fits.f_match import boxes_decode4retina, pos_match_retina, match_gt_b
from f_tools.floss.focal_loss import focalloss
from f_tools.fun_od.f_anc import FAnchors

from f_tools.fun_od.f_boxes import xywh2ltrb
from f_tools.pic.f_show import f_show_od_ts4plt
from object_detection.z_retina.nets.net_ceng import HeadBox


class LossRetina3(nn.Module):

    def __init__(self, cfg, anc_obj):
        super().__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([1, 32526, 4])

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs:
            ptxywh, torch.Size([2, 32526, 4])
            pcls, torch.Size([2, 32526, 3]) 已归一化
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        ptxywh, pcls = outs
        device = ptxywh.device
        batch, pdim1, c = ptxywh.shape

        # iou-1, cls-3, txywh-4, gltrb-4,  keypoint-nn  = 13 + nn
        # gdim = 2 + cfg.NUM_CLASSES + 4 + 4
        # cls-3, txywh-4, keypoint-nn  = 7 + nn
        gdim = cfg.NUM_CLASSES + 4
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

            gretinas[i] = match_gt_b(cfg, dim=gdim,
                                     gboxes_ltrb_b=gboxes_ltrb_b, glabels_b=glabels_b, anc_obj=self.anc_obj,
                                     mode='iou', ptxywh_b=ptxywh[i], img_ts=imgs_ts[i])

            # 匹配正例可视化
            if cfg.IS_VISUAL:
                _mask_pos = gretinas[i, :, :cfg.NUM_CLASSES] > 0
                # torch.Size([3614, 3]) -> [3614]
                _mask_pos = torch.any(_mask_pos, dim=-1)
                _img_ts = imgs_ts[i].clone()
                anc_ltrb = xywh2ltrb(self.anc_obj.ancs_xywh)[_mask_pos]
                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                _img_ts = f_recover_normalization4ts(_img_ts)
                f_show_od_ts4plt(_img_ts, gboxes_ltrb=gboxes_ltrb_b.cpu()
                                 , pboxes_ltrb=anc_ltrb.cpu(), is_recover_size=True,
                                 # grids=grids_ts.cpu().numpy(),
                                 # plabels_text=pconf_b[index_match_dim].sigmoid(),
                                 # glabels_text=colrow_index[None]
                                 )

        # mask_pos torch.Size([b,3614, 3]) -> [b,3614, 3] -> [b,3614]
        mask_pos = torch.any(gretinas[:, :, :cfg.NUM_CLASSES] > 0, dim=-1)
        num_pos = mask_pos.sum(-1).clamp(min=torch.finfo(torch.float16).eps) # 每张图匹配的正例数
        num_pos = torch.clamp(num_pos, min=1)  # 确保正例数最小为1 这句本例不起作用
        # mask_neg = torch.any(gretinas[:, :, :cfg.NUM_CLASSES] == 0, dim=-1)
        mash_ignore = torch.any(gretinas[:, :, :cfg.NUM_CLASSES] == -1, dim=-1)

        # ----------------cls损失  采用bce 采用筛选计算----------------
        pcls = pcls.sigmoid()
        gcls = gretinas[:, :, :cfg.NUM_CLASSES]
        # [b,3614] -> [b,3614,3]
        _mask_pos = mask_pos.unsqueeze(-1).repeat(1, 1, pcls.shape[-1])
        # _mask_neg = mask_neg.unsqueeze(-1).repeat(1, 1, pcls.shape[-1])
        _mash_ignore = mash_ignore.unsqueeze(-1).repeat(1, 1, pcls.shape[-1])
        l_pos, l_neg = focalloss(pcls, gcls, mask_pos=_mask_pos, mash_ignore=_mash_ignore,
                                 alpha=0.25, gamma=2,
                                 reduction='none', is_debug=True)
        loss_conf_pos = (l_pos.sum(-1).sum(-1) / num_pos).mean() * cfg.LOSS_WEIGHT[0]  # 批数平均
        loss_conf_neg = (l_neg.sum(-1).sum(-1) / num_pos).mean() * cfg.LOSS_WEIGHT[1]  # 批数平均

        # ----------------box损失   xy采用bce wh采用mes-----------------
        # 正例筛选后计算
        gtxywh = gretinas[:, :, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 4]
        ptxywh = ptxywh
        # _loss_val = F.smooth_l1_loss(ptxywh, gtxywh, reduction="none") * mask_pos.unsqueeze(-1)
        _loss_val = torch.abs(ptxywh - gtxywh) * mask_pos.unsqueeze(-1)  # l1 损失
        # 损失较小的放大 0.5 * 9.0 - 较大的减小 - 0.5 / 9.0
        _loss_val = torch.where(torch.le(_loss_val, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(_loss_val, 2),
                                _loss_val - 0.5 / 9.0
                                )
        loss_box = (_loss_val.sum(-1).sum(-1) / num_pos).mean()

        log_dict = {}
        loss_total = loss_conf_pos + loss_conf_neg + loss_box

        log_dict['l_total'] = loss_total.item()
        log_dict['l_conf_pos'] = loss_conf_pos.item()
        log_dict['l_conf_neg'] = loss_conf_neg.item()
        log_dict['l_box'] = loss_box.item()

        log_dict['p_max'] = pcls.max().item()
        log_dict['p_min'] = pcls.min().item()
        log_dict['p_mean'] = pcls.mean().item()

        return loss_total, log_dict


class LossRetina4(nn.Module):

    def __init__(self, cfg, anc_obj):
        super().__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([1, 32526, 4])

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs:
            ptxywh, torch.Size([2, 32526, 4])
            pcls, torch.Size([2, 32526, 4]) 已归一化 cls+1
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        ptxywh, pcategory = outs
        pcategory = pcategory.sigmoid()
        pconf = pcategory[:, :, 0]  # torch.Size([2, 32526])
        pcls = pcategory[:, :, 1:]  # 已sigmoid
        device = ptxywh.device
        batch, pdim1, c = ptxywh.shape

        # conf-1, cls-3, txywh-4, keypoint-nn  = 8 + nn
        gdim = 1 + cfg.NUM_CLASSES + 4
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

            # gretinas[i] = match_gt_b(cfg, dim=gdim,
            #                          gboxes_ltrb_b=gboxes_ltrb_b, glabels_b=glabels_b, anc_obj=self.anc_obj,
            #                          mode='iou', ptxywh_b=ptxywh[i], img_ts=imgs_ts[i])
            gretinas[i] = pos_match_retina(cfg, dim=gdim, gkeypoints_b=None,
                                           gboxes_ltrb_b=gboxes_ltrb_b, glabels_b=glabels_b, anc_obj=self.anc_obj,
                                           ptxywh_b=ptxywh[i], img_ts=imgs_ts[i])

            # 匹配正例可视化
            if cfg.IS_VISUAL:
                _mask_pos = gretinas[i, :, :cfg.NUM_CLASSES] > 0
                # torch.Size([3614, 3]) -> [3614]
                _mask_pos = torch.any(_mask_pos, dim=-1)
                _img_ts = imgs_ts[i].clone()
                anc_ltrb = xywh2ltrb(self.anc_obj.ancs_xywh)[_mask_pos]
                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                _img_ts = f_recover_normalization4ts(_img_ts)
                f_show_od_ts4plt(_img_ts, gboxes_ltrb=gboxes_ltrb_b.cpu()
                                 , pboxes_ltrb=anc_ltrb.cpu(), is_recover_size=True,
                                 # grids=grids_ts.cpu().numpy(),
                                 # plabels_text=pconf_b[index_match_dim].sigmoid(),
                                 # glabels_text=colrow_index[None]
                                 )

        # mask_pos torch.Size([b,3614, 3]) -> [b,3614, 3] -> [b,3614]
        mask_pos = gretinas[:, :, 0] > 0  # torch.Size([2, 32526])
        num_pos = mask_pos.sum(-1).clamp(min=torch.finfo(torch.float16).eps)  # 每张图匹配的正例数
        num_pos = torch.clamp(num_pos, min=1)  # 确保正例数最小为1 这句本例不起作用
        mask_neg = gretinas[:, :, 0] == 0
        mash_ignore = gretinas[:, :, 0] == -1

        # ----------------conf损失  ----------------
        gconf = gretinas[:, :, 0]
        l_pos, l_neg = focalloss(pconf, gconf, mask_pos=mask_pos, mash_ignore=mash_ignore,
                                 alpha=0.25, gamma=2,
                                 reduction='none', is_debug=True)
        loss_conf_pos = (l_pos.sum(-1).sum(-1) / num_pos).mean() * cfg.LOSS_WEIGHT[0]  # 批数平均
        loss_conf_neg = (l_neg.sum(-1).sum(-1) / num_pos).mean() * cfg.LOSS_WEIGHT[1]

        # ----------------cls损失  采用bce 采用筛选计算----------------
        pcls = pcls  # 正例损失 已归一
        gcls = gretinas[:, :, 1:1 + cfg.NUM_CLASSES]
        l_cls = x_bce(pcls, gcls).sum(-1) * mask_pos
        loss_cls = (l_cls.sum(-1)/num_pos).mean()

        # ----------------box损失   xy采用bce wh采用mes-----------------
        # 正例筛选后计算
        gtxywh = gretinas[:, :, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 4]
        ptxywh = ptxywh
        # _loss_val = F.smooth_l1_loss(ptxywh, gtxywh, reduction="none") * mask_pos.unsqueeze(-1)
        _loss_val = torch.abs(ptxywh - gtxywh) * mask_pos.unsqueeze(-1)  # l1 损失
        # 损失较小的放大 0.5 * 9.0 - 较大的减小 - 0.5 / 9.0
        _loss_val = torch.where(torch.le(_loss_val, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(_loss_val, 2),
                                _loss_val - 0.5 / 9.0
                                )
        loss_box = (_loss_val.sum(-1).sum(-1) / num_pos).mean()

        log_dict = {}
        loss_total = loss_conf_pos + loss_conf_neg + loss_cls + loss_box

        log_dict['l_total'] = loss_total.item()
        log_dict['l_conf_pos'] = loss_conf_pos.item()
        log_dict['l_conf_neg'] = loss_conf_neg.item()
        log_dict['loss_cls'] = loss_cls.item()
        log_dict['l_box'] = loss_box.item()

        log_dict['p_max'] = pcls.max().item()
        log_dict['p_min'] = pcls.min().item()
        log_dict['p_mean'] = pcls.mean().item()

        return loss_total, log_dict


class PredictRetina(nn.Module):
    def __init__(self, cfg, anc_obj):
        super(PredictRetina, self).__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([10647, 4]) 原图归一化 anc

    def forward(self, outs, imgs_ts=None):
        '''
        批量处理 conf + nms
        :param outs:
            ptxywh, torch.Size([32, 16731, 4])
            pcls, torch.Size([32, 16731, 4]) conf1 +  cls3
            pkeypoint None 或与关键点相同维度
        :return:
        '''
        cfg = self.cfg
        ptxywh, pcls = outs
        device = ptxywh.device

        pcls = pcls.sigmoid()
        pconf = pcls[:, :, 0]  # torch.Size([32, 10647])
        cls_conf, plabels = pcls[:, :, 1:].max(-1)
        pscores = cls_conf * pconf

        mask_pos = pscores >= cfg.THRESHOLD_PREDICT_CONF
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(pconf.max().item(),
                                                                          pconf.min().item(),
                                                                          pconf.mean().item(),
                                                                          ))
            return [None] * 5

        ids_batch1, _ = torch.where(mask_pos)
        # [10647, 4] -> torch.Size([1, 10647, 4])
        pxywh = boxes_decode4retina(cfg, self.anc_obj, ptxywh, variances=cfg.variances)
        pboxes_ltrb1 = xywh2ltrb(pxywh[mask_pos])

        # if cfg.NUM_KEYPOINTS > 0:
        #     pkeypoint = fix_keypoints(self.anchors, pkeypoint)
        #     p_keypoints1 = pkeypoint[mask_pos]
        # else:
        #     p_keypoints1 = None
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1

        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    pboxes_ltrb1,
                                                                    plabels1,
                                                                    pscores1,
                                                                    device,
                                                                    cfg.THRESHOLD_PREDICT_NMS)
        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


class PredictRetina2(nn.Module):
    def __init__(self, cfg, anc_obj):
        super(PredictRetina2, self).__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj  # torch.Size([10647, 4]) 原图归一化 anc

    def forward(self, outs, imgs_ts=None):
        '''
        批量处理 conf + nms
        :param outs:
            ptxywh, torch.Size([32, 32526, 4])
            pcls, torch.Size([32, 32526, 3])
            pkeypoint None 或与关键点相同维度
        :return:
        '''
        cfg = self.cfg
        ptxywh, pcls = outs
        pcls = pcls.sigmoid()
        device = ptxywh.device

        pscores, plabels = pcls.max(-1)
        # torch.Size([32, 32526, 3]) -> torch.Size([32, 32526])
        mask_pos = pscores >= cfg.THRESHOLD_PREDICT_CONF

        mask_pos4top = torch.zeros_like(pscores, device=device, dtype=torch.bool)
        val_topk, idx_topk = pscores.topk(1000, dim=1)
        mask_pos4top[torch.arange(1), idx_topk] = True

        mask_pos = torch.logical_and(mask_pos, mask_pos4top)
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(pcls.max().item(),
                                                                          pcls.min().item(),
                                                                          pcls.mean().item(),
                                                                          ))
            return [None] * 5
        # torch.Size([32, 32526]) ->
        # topk = torch.topk(pscores, k=1000)
        ids_batch1, _ = torch.where(mask_pos)
        # [10647, 4] -> torch.Size([1, 10647, 4])
        pxywh = boxes_decode4retina(cfg, self.anc_obj, ptxywh, variances=cfg.variances)
        pboxes_ltrb1 = xywh2ltrb(pxywh[mask_pos])

        # if cfg.NUM_KEYPOINTS > 0:
        #     pkeypoint = fix_keypoints(self.anchors, pkeypoint)
        #     p_keypoints1 = pkeypoint[mask_pos]
        # else:
        #     p_keypoints1 = None
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1

        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    pboxes_ltrb1,
                                                                    plabels1,
                                                                    pscores1,
                                                                    device,
                                                                    cfg.THRESHOLD_PREDICT_NMS)
        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


class Retina_Net(nn.Module):

    def __init__(self, backbone, cfg, out_channels_fpn=256):
        super(Retina_Net, self).__init__()
        self.backbone = backbone
        self.cfg = cfg

        self.fpn = FPNv1(backbone.dims_out, out_channels=out_channels_fpn)

        self.ssh1 = SSH(out_channels_fpn, out_channels_fpn)
        self.ssh2 = SSH(out_channels_fpn, out_channels_fpn)
        self.ssh3 = SSH(out_channels_fpn, out_channels_fpn)

        self.head_box = nn.ModuleList()
        self.head_cls = nn.ModuleList()
        if cfg.NUM_KEYPOINTS > 0:
            self.haed_keypoint = nn.ModuleList()
        else:
            self.haed_keypoint = None

        for i in range(3):  # 特图层输出数  ssh 输出
            self.head_box.append(HeadBox(out_channels_fpn, cfg.NUMS_ANC[i]))
            self.head_cls.append(HeadCls(out_channels_fpn, cfg.NUMS_ANC[i], cfg.NUM_CLASSES))
            if cfg.NUM_KEYPOINTS > 0:
                self.haed_keypoint.append(
                    HaedKeypoint(out_channels_fpn, num_anchors=cfg.NUMS_ANC[i], num_keypoint=cfg.NUM_KEYPOINTS))

        # 初始化分类 bias
        # num_pos = 1.5  # 正例数
        # num_tolal = 10  # 总样本数
        # finit_conf_bias_one(self.pred_3, num_tolal, num_pos, cfg.NUM_CLASSES)
        # finit_conf_bias_one(self.pred_2, num_tolal, num_pos, cfg.NUM_CLASSES)
        # finit_conf_bias_one(self.pred_1, num_tolal, num_pos, cfg.NUM_CLASSES)

    def forward(self, x):
        # ceng1, ceng2, ceng3 [2, 128, 52, 52]  [2, 512, 13, 13] [2, 512, 13, 13]
        outs = self.backbone(x)

        # fpn
        outs = self.fpn(outs)  # 尺寸不变 全256输出

        # SSH
        _ssh1 = self.ssh1(outs[0])  # torch.Size([2, 256, 52, 52])
        _ssh2 = self.ssh2(outs[1])  # torch.Size([2, 256, 26, 26])
        _ssh3 = self.ssh3(outs[2])  # torch.Size([2, 256, 13, 13])
        outs = [_ssh1, _ssh2, _ssh3]

        self.cfg.tnums_ceng = []
        ptxywh = []
        pcls = []
        for i, out in enumerate(outs):
            _t = self.head_box[i](out)
            ptxywh.append(_t)
            pcls.append(self.head_cls[i](out))
            self.cfg.tnums_ceng.append(_t.shape[1])

        # 三层输出堆叠后输出 torch.Size([2, 16731, 4])
        ptxywh = torch.cat(ptxywh, dim=1)
        # torch.Size([2, 16731, 4]) 这里cls 多一维 1+3
        pcls = torch.cat(pcls, dim=1)

        pkeypoint = None
        if self.haed_keypoint is not None:  # torch.Size([2, 16731, 5])
            pkeypoint = torch.cat([self.haed_keypoint[i](out) for i, out in enumerate(outs)], dim=1)

        return ptxywh, pcls, pkeypoint


class Retina_Net2(nn.Module):

    def __init__(self, backbone, cfg, num_classes, out_channels_fpn=256):
        super(Retina_Net2, self).__init__()
        self.backbone = backbone
        self.cfg = cfg

        self.fpn = FPNv2(backbone.dims_out, out_channels_fpn)

        # fpn输出是256
        # num_anc = np.array(cfg.NUMS_ANC).prod()
        num_anc = cfg.NUMS_ANC[0]
        self.regressionModel = RegressionModel(out_channels_fpn, num_anchors=num_anc, feature_size=out_channels_fpn)
        self.classificationModel = ClassificationModel(out_channels_fpn, num_anchors=num_anc, num_classes=num_classes)

        # init_weight
        self.init_weight()

    def init_weight(self):
        import math
        for m in self.modules():
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

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        # torch.Size([1, 32526, 4])

        return regression, classification


class Retina(nn.Module):
    def __init__(self, backbone, cfg, device):
        super(Retina, self).__init__()
        self.net = Retina_Net(backbone, cfg)

        # 分配anc
        ancs_scale = []
        s = 0
        for num in cfg.NUMS_ANC:
            ancs_scale.append(cfg.ANCS_SCALE[s:s + num])
            s += num

        # torch.Size([1, 32526, 4])
        self.anc_obj = FAnchors(cfg.IMAGE_SIZE, ancs_scale, cfg.FEATURE_MAP_STEPS,
                                anchors_clip=True, is_xymid=True, is_real_size=False,
                                device=device)

        self.losser = LossRetina(cfg, self.anc_obj)

        self.preder = PredictRetina(cfg, self.anc_obj)

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


class Retina2(nn.Module):
    def __init__(self, backbone, cfg, device):
        super(Retina2, self).__init__()
        self.net = Retina_Net2(backbone, cfg, cfg.NUM_CLASSES + 1)

        ancs_scale = []
        s = 0
        for num in cfg.NUMS_ANC:
            ancs_scale.append(cfg.ANCS_SCALE[s:s + num])
            s += num

        # self.anc_obj = Anchors(device=cfg.device)
        self.anc_obj = FAnchors(cfg.IMAGE_SIZE, ancs_scale, cfg.FEATURE_MAP_STEPS,
                                anchors_clip=True, is_xymid=True, is_real_size=False,
                                device=device)

        # self.losser = LossRetina3(cfg, self.anc_obj)
        self.losser = LossRetina4(cfg, self.anc_obj)
        # self.losser = LossRetina(cfg, self.anc_obj)

        self.preder = PredictRetina2(cfg, self.anc_obj)
        # self.preder = PredictRetina(cfg, self.anc_obj)

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
    # cfg.FEATURE_MAP_STEPS = [8, 16, 32]
    cfg.FEATURE_MAP_STEPS = [8, 16, 32, 64, 128]
    cfg.NUM_KEYPOINTS = 5
    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)

    model = Retina_Net2(model, cfg)
    f_look_tw(model, input=(1, 3, 416, 416), name='Retina_Net')
    # f_look_summary(model, input=(3, 416, 416))
    # model(torch.randn((2, 3, 416, 416)))
