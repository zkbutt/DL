import torch
import torch.nn as nn

from f_pytorch.tools_model.f_model_api import CBL
from f_pytorch.tools_model.model_look import f_look_summary
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import boxes_decode4yolo3
from f_tools.fits.f_predictfun import label_nms

from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2xywh, calc_iou4ts, bbox_iou4one2
import torch.nn.functional as F
import math


def calc_match_index(cxy, index_iou, cfg, device):
    match_ceng_index = torch.true_divide(index_iou, cfg.NUMS_ANC[0]).type(torch.int32)
    offset_ceng = 0
    if match_ceng_index > 0:
        # 计算出dim的层偏移    cfg.nums_ceng 在模型输出时计算出来了
        offset_ceng = torch.tensor(cfg.nums_ceng, dtype=torch.int32)[:match_ceng_index].sum()

    grid = math.sqrt(cfg.nums_ceng[match_ceng_index.item()])
    grids_ts = torch.tensor([grid, grid], device=device, dtype=torch.int32)
    colrow_index = (cxy * grids_ts).type(torch.int32)
    col, row = colrow_index
    offset_colrow = row * grid + col
    match_dim_s = ((offset_ceng + offset_colrow) * cfg.NUMS_ANC[0]).type(torch.int32)
    return colrow_index, grids_ts, match_dim_s, match_ceng_index


def fmatch4yolov3(gboxes_ltrb_b, labels, dim, ptxywh_b, device, cfg, img_ts=None):
    '''

    :param gboxes_ltrb_b:
    :param labels:
    :param dim:
    :param ptxywh_b: torch.Size([3, 10647, 4])
    :param device:
    :param cfg:
    :param img_ts:
    :return:
    '''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    anc_wh_ts = torch.tensor(cfg.ANC_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANC_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = anc_wh_ts

    # 匹配一个最大的 用于获取iou index  iou>0.5 忽略
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    mask = iou2d > 0.5
    iou_max_val, iou_max_index = iou2d.max(1)

    # 匹配完成的数据
    _dim_total = sum(cfg.nums_ceng) * 3  # 10647
    _num_anc_total = len(cfg.ANC_SCALE)
    # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
    g_yolo_one = torch.zeros((_dim_total, dim), device=device)

    labels = labels2onehot4ts(labels - 1, cfg.NUM_CLASSES)

    for i in range(len(gboxes_ltrb_b)):
        cxy = gboxes_xywh[i][:2]
        indexs_anc_ignore = torch.where(mask[i])
        for j in indexs_anc_ignore[0]:
            colrow_index, grids_ts, match_dim_s, index_match_ceng = calc_match_index(cxy, j, cfg, device)
            g_yolo_one[match_dim_s, 0] = -1
            g_yolo_one[match_dim_s, 1 + cfg.NUM_CLASSES + 4] = -1.  # weight

            '''可视化验证'''
            # if cfg.IS_VISUAL:
            #     _img_ts = img_ts.clone()
            #     flog.debug('iou %s', iou2d[i, j])
            #     anc_xywh = torch.cat([cxy, anc_wh_ts[j]], -1)
            #
            #     from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            #     _img_ts = f_recover_normalization4ts(_img_ts)
            #     f_show_od_ts4plt(_img_ts, gboxes_ltrb=gboxes_ltrb_b[i][None].cpu()
            #                      , pboxes_ltrb=xywh2ltrb(anc_xywh[None].cpu()), is_recover_size=True,
            #                      )

        index_match_anc = iou_max_index[i]
        colrow_index, grids_ts, match_dim_s, index_match_ceng = calc_match_index(cxy, index_match_anc, cfg, device)
        offset_xy = torch.true_divide(colrow_index, grids_ts[0])
        txy_g = (cxy - offset_xy) * grids_ts  # 特图偏移

        # 比例 /log
        anc_match_ts = torch.tensor(cfg.ANC_SCALE[iou_max_index[i]], device=device)
        twh_g = (gboxes_ltrb_b[i][2:] / anc_match_ts).log()
        txywh_g = torch.cat([txy_g, twh_g], dim=-1)

        with torch.no_grad():
            # (xx,4) -> [4]
            _ptxywh = ptxywh_b[index_match_anc]
            pxy = torch.sigmoid(_ptxywh[:2]) + colrow_index
            pxy = pxy / grids_ts

            pwh = _ptxywh[2:].exp() * anc_match_ts
            pxywh = torch.cat([pxy, pwh], -1)
            _pltrb = xywh2ltrb(pxywh[None])

            _gbox_p = gboxes_ltrb_b[i][None]
            iou = bbox_iou4one2(_pltrb, _gbox_p)

            '''可视化验证'''
            # if cfg.IS_VISUAL:
            #     _img_ts = img_ts.clone()
            #     flog.debug('预测 iou %s', iou[0].item())  # 只取第一个
            #
            #     from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            #     _img_ts = f_recover_normalization4ts(_img_ts)
            #     f_show_od_ts4plt(_img_ts, gboxes_ltrb=_gbox_p.cpu()
            #                      , pboxes_ltrb=_pltrb.cpu(), is_recover_size=True,
            #                      )

            conf = iou[0][None]

        weight = 2.0 - torch.prod(gboxes_ltrb_b[i][2:], dim=-1)  # 这是一个数需要加一维 1~2 小目标加成 这个不是数组
        _labels = labels[i]
        # labels恢复至1
        t = torch.cat([conf, _labels, txywh_g, weight[None], gboxes_ltrb_b[i]], dim=0)
        g_yolo_one[match_dim_s] = t
    return g_yolo_one


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

        # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
        gdim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4
        gyolos = torch.empty((batch, hwa, gdim), device=device)

        # 匹配GT
        for i, target in enumerate(targets):  # batch遍历
            boxes_ltrb_b = target['boxes']  # ltrb
            labels_b = target['labels']

            gyolos[i] = fmatch4yolov3(gboxes_ltrb_b=boxes_ltrb_b, labels=labels_b, dim=gdim,
                                      ptxywh_b=ptxywh[i], device=device, cfg=cfg, img_ts=imgs_ts[i])

        # gyolos [3, 10647, 13] conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
        s_ = 1 + cfg.NUM_CLASSES
        # ----------------conf损失  mes+倍数----------------
        pconf = pconf.sigmoid().view(batch, -1)  # [3, 10647, 1] -> [3, 10647]
        weight = gyolos[:, :, s_ + 4]  # 降维 -> [3, 10647]
        mask_pos = weight > 0  # 同维bool索引 忽略的-1不计
        mask_neg = weight == 0
        gconf = gyolos[:, :, 0]
        _loss_val = F.mse_loss(pconf, gconf, reduction="none")  # torch.Size([3, 10647])
        # _loss_val = F.binary_cross_entropy_with_logits(pconf, gconf, reduction="none")
        loss_conf_pos = (_loss_val * mask_pos).sum(-1).mean() * cfg.LOSS_WEIGHT[0]
        loss_conf_neg = (_loss_val * mask_neg).sum(-1).mean() * cfg.LOSS_WEIGHT[1]

        # ----------------cls损失  采用bce 采用筛选计算----------------
        pcls = pcls[mask_pos]
        gcls = gyolos[:, :, 1:s_][mask_pos]
        _loss_val = F.binary_cross_entropy_with_logits(pcls, gcls, reduction="none")
        loss_cls = _loss_val.sum(-1).mean()

        # ----------------box损失   xy采用bce wh采用mes-----------------
        ptxty = ptxywh[:, :, :2]  # 前面已完成归一化
        ptwth = ptxywh[:, :, 2:4]
        gtxty = gyolos[:, :, s_:s_ + 2]
        gtwth = gyolos[:, :, s_ + 2:s_ + 4]
        _loss_val = F.binary_cross_entropy_with_logits(ptxty, gtxty, reduction="none")
        loss_txty = (_loss_val.sum(-1) * mask_pos * weight).sum(-1).mean()
        _loss_val = F.mse_loss(ptwth, gtwth, reduction="none")
        loss_twth = (_loss_val.sum(-1) * mask_pos * weight).sum(-1).mean()

        log_dict = {}
        loss_total = loss_conf_pos + loss_conf_neg + loss_cls + loss_txty + loss_twth

        log_dict['l_total'] = loss_total.item()
        log_dict['l_conf_pos'] = loss_conf_pos.item()
        log_dict['l_conf_neg'] = loss_conf_neg.item()
        log_dict['l_cls'] = loss_cls.item()
        log_dict['l_xy'] = loss_txty.item()
        log_dict['l_wh'] = loss_twth.item()

        log_dict['p_max'] = pconf.max().item()
        log_dict['p_min'] = pconf.min().item()
        log_dict['p_mean'] = pconf.mean().item()

        return loss_total, log_dict


class PredictYOLO_v3(nn.Module):
    def __init__(self, cfg):
        super(PredictYOLO_v3, self).__init__()
        # self.num_bbox = num_bbox
        # self.num_classes = num_classes
        # self.num_grid = num_grid
        self.cfg = cfg

    def forward(self, p_yolo_tuple, imgs_ts=None):
        '''
        批量处理 conf + nms
        :param p_yolo_tuple: pconf pcls ptxywh
            pconf: torch.Size([3, 10647, 1])
            pcls: torch.Size([3, 10647, 3])
            ptxywh: torch.Size([3, 10647, 4])
        :param targets: list
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['size'] = target['size']
            target['image_id'] = int
        '''
        cfg = self.cfg
        pconf, pcls, ptxywh = p_yolo_tuple
        device = ptxywh.device
        batch, hwa, c = ptxywh.shape  # [3, 10647, 4]

        pconf = pconf.sigmoid().view(batch, -1)  # [3, 10647, 1] -> [3, 10647]
        cls_conf, plabels = pcls.sigmoid().max(-1)
        pscores = cls_conf * pconf

        mask_pos = pscores > cfg.THRESHOLD_PREDICT_CONF  # b,hw
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(pconf.max().item(),
                                                                          pconf.min().item(),
                                                                          pconf.mean().item(),
                                                                          ))
            return [None] * 5

        ids_batch1, _ = torch.where(mask_pos)
        # 解码txywh
        pltrb = boxes_decode4yolo3(ptxywh, cfg)
        pboxes_ltrb1 = pltrb[mask_pos]
        pboxes_ltrb1 = torch.clamp(pboxes_ltrb1, min=0., max=1.)

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1

        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    pboxes_ltrb1,
                                                                    plabels1,
                                                                    pscores1,
                                                                    device,
                                                                    self.cfg.THRESHOLD_PREDICT_NMS)

        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


class Yolo_v3_Net(nn.Module):

    def __init__(self, backbone, cfg):
        super(Yolo_v3_Net, self).__init__()
        self.cfg = cfg
        self.backbone = backbone

        # s32
        self.conv_set_3 = nn.Sequential(
            CBL(1024, 512, 1, leakyReLU=True),
            CBL(512, 1024, 3, padding=1, leakyReLU=True),
            CBL(1024, 512, 1, leakyReLU=True),
            CBL(512, 1024, 3, padding=1, leakyReLU=True),
            CBL(1024, 512, 1, leakyReLU=True),
        )
        self.conv_1x1_3 = CBL(512, 256, 1, leakyReLU=True)
        self.extra_conv_3 = CBL(512, 1024, 3, padding=1, leakyReLU=True)
        self.pred_3 = nn.Conv2d(1024, cfg.NUMS_ANC[2] * (1 + 4 + cfg.NUM_CLASSES), 1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            CBL(768, 256, 1, leakyReLU=True),
            CBL(256, 512, 3, padding=1, leakyReLU=True),
            CBL(512, 256, 1, leakyReLU=True),
            CBL(256, 512, 3, padding=1, leakyReLU=True),
            CBL(512, 256, 1, leakyReLU=True),
        )
        self.conv_1x1_2 = CBL(256, 128, 1, leakyReLU=True)
        self.extra_conv_2 = CBL(256, 512, 3, padding=1, leakyReLU=True)
        self.pred_2 = nn.Conv2d(512, cfg.NUMS_ANC[1] * (1 + 4 + cfg.NUM_CLASSES), 1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            CBL(384, 128, 1, leakyReLU=True),
            CBL(128, 256, 3, padding=1, leakyReLU=True),
            CBL(256, 128, 1, leakyReLU=True),
            CBL(128, 256, 3, padding=1, leakyReLU=True),
            CBL(256, 128, 1, leakyReLU=True),
        )
        self.extra_conv_1 = CBL(128, 256, 3, padding=1, leakyReLU=True)
        self.pred_1 = nn.Conv2d(256, cfg.NUMS_ANC[0] * (1 + 4 + cfg.NUM_CLASSES), 1)

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
        self.cfg.nums_ceng = []  # [2704, 676, 169] 这个用于后面匹配
        for i, pred in enumerate(preds):
            # 每层的每个网格有3个anc
            b, c, h, w = pred.shape
            self.cfg.nums_ceng.append(h * w)
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


class Yolo_v3(nn.Module):
    def __init__(self, backbone, cfg):
        '''
        层属性可以是 nn.Module nn.ModuleList(封装Sequential) nn.Sequential
        '''
        super(Yolo_v3, self).__init__()
        self.net = Yolo_v3_Net(backbone, cfg)

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
