import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from f_pytorch.tools_model.backbones.CSPDarknet import BottleneckCSP
from f_pytorch.tools_model.backbones.darknet import darknet19
from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet19, ModelOuts4Resnet
from f_pytorch.tools_model.f_model_api import CBL, ReorgLayer
from f_pytorch.tools_model.fmodels.model_modules import SPPv2, SAM
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import boxes_decode4yolo2, boxes_encode4yolo2
from f_tools.fits.f_predictfun import label_nms
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import x_bce
from f_tools.fun_od.f_boxes import xywh2ltrb, calc_iou4ts, ltrb2xywh, bbox_iou4one
from f_tools.pic.f_show import f_show_od_np4plt, f_show_od_ts4plt
from f_tools.yufa.x_calc_adv import f_mershgrid


def fmatch4yolov2(gboxes_ltrb_b, glabels_b, grid, dim, device, cfg, img_ts=None,
                  iou_pos=0.5, iou_neg=0.4):
    '''
    匹配 gyolo 如果需要计算IOU 需在这里生成
    :param gboxes_ltrb_b: ltrb
    :param glabels_b:
    :param grid: 13
    :param dim:
    :param device:
    :return: [13, 13, 5, 13]
    '''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    anc_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = anc_wh_ts

    # 匹配一个最大的 用于获取iou index  iou>0.5 忽略
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    # [gt,anc]
    # index_p = iou2.max(-1)[1]  # 匹配最大的IOU
    # mask_pos_p = iou2 > iou_pos
    # mask_neg_p = iou2 < iou_neg

    # 编码GT
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    whs = gboxes_xywh[:, 2:]

    # 编码xy
    cxys = gboxes_xywh[:, :2]
    grids_ts = torch.tensor([grid, grid], device=device, dtype=torch.int16)
    colrows_index = (cxys * grids_ts).type(torch.int16)  # 网格7的index
    offset_xys = torch.true_divide(colrows_index, grid)  # 网络index 对应归一化的实距
    txys = (cxys - offset_xys) * grids_ts  # 特图偏移

    # conf-1, cls-3, tbox-4, weight-1, gltrb-4   torch.Size([13, 13, 13])
    p_yolo_one = torch.zeros((grid, grid, cfg.NUM_ANC, dim), device=device)

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)

    # 遍历GT
    for i, (col, row) in enumerate(colrows_index):
        index_max = torch.argmax(iou2d, dim=-1)[i]  # 每个GT最大的IOU
        p_yolo_one[row, col, :, 0] = -1  # 全部-1

        indexs_ignore = torch.where(mask_pos_p[i])

        # 有正例,将正例赋0 用于忽略
        for j in indexs_ignore:
            p_yolo_one[row, col, j, 0] = -1.
            p_yolo_one[row, col, j, 1 + cfg.NUM_CLASSES + 4] = -1.

        # 编码 wh 这个获取的是比例故可以用原图尺寸
        twh_g = (whs[i] / ancs_wh_ts[index_max]).log()  # 这个是比例 [2]
        txy_g = txys[i]  # [2]
        txywh_g = torch.cat([txy_g, twh_g], dim=-1)  # [4]
        # 正例的conf
        conf = torch.tensor([1], device=device)
        weight = 2.0 - torch.prod(whs[i], dim=-1)  # 1~2 小目标加成 这个不是数组
        _labels = glabels_b[i]

        # labels恢复至1
        t = torch.cat([conf, _labels, txywh_g, weight[None], gboxes_ltrb_b[i]], dim=0)
        p_yolo_one[row, col, index_max] = t

    return p_yolo_one


def fmatch4yolov2_one(gboxes_ltrb_b, glabels_b, grid, dim, device, cfg, img_ts=None):
    '''
    匹配 gyolo 如果需要计算IOU 需在这里生成
    :param gboxes_ltrb_b: ltrb
    :param glabels_b:
    :param grid: 13
    :param dim:
    :param device:
    :return: 匹配中心与GT相同 iou 最大的一个anc  其余的全为0
    '''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    anc_wh = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = anc_wh

    # 匹配一个最大的 用于获取iou index
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    # [gt,anc] GT对应格子中哪个最大
    # index_p = iou2.max(-1)[1]  # 匹配最大的IOU
    ids_p = torch.argmax(iou2d, dim=-1)  # 匹配最大的IOU
    txywhs_g, weights, colrows_index = boxes_encode4yolo2(gboxes_ltrb_b, ids_p, grid, grid, device, cfg)

    # conf-1, cls-3, tbox-4, weight-1, gltrb-4   torch.Size([13, 13, 13])
    p_yolo_one = torch.zeros((grid, grid, cfg.NUM_ANC, dim), device=device)

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    # ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)

    # 遍历GT 匹配iou最大的  其它anc全部忽略
    for i, (col, row) in enumerate(colrows_index):
        p_yolo_one[row, col, :, 0] = -1  # 全部忽略
        # 正例的conf
        conf = torch.tensor([1], device=device)
        _t = torch.cat([conf, glabels_b[i], txywhs_g[i], weights[i][None], gboxes_ltrb_b[i]], dim=0)
        p_yolo_one[row, col, ids_p[i]] = _t

    return p_yolo_one


def calc_iou(ptxywh, gbox_p, batch, grid_h, grid_w, cfg, mask_pos_2, imgs_ts=None):
    '''
    可视化 匹配的预测框
    :param ptxywh: torch.Size([3, 169, 5, 4])
    :param gbox_p: [-1,4]
    :param grid_h:
    :param grid_w:
    :param cfg:
    :param mask_pos_2:  [3,xx]
    :param img_ts:
    :return:
    '''
    # 解码 可debug 原anc
    pltrb = boxes_decode4yolo2(ptxywh, grid_h, grid_w, cfg)

    if cfg.IS_VISUAL:
        d0, d1 = torch.where(mask_pos_2)  # [3,845]
        for i in range(batch):
            from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
            img_ts = f_recover_normalization4ts(imgs_ts[i])
            mask_ = d0 == i
            _pltrb = pltrb[d0[mask_], d1[mask_]]
            _gbox_p = gbox_p[d0[mask_], d1[mask_]]

            iou = bbox_iou4one(_pltrb, _gbox_p)
            flog.debug('预测 iou %s', iou)
            f_show_od_ts4plt(img_ts, gboxes_ltrb=_gbox_p, pboxes_ltrb=_pltrb,
                             is_recover_size=True, grids=(grid_h, grid_w))

    pltrb = pltrb.view(-1, 4)
    gbox_p = gbox_p.view(-1, 4)
    iou = bbox_iou4one(pltrb, gbox_p)  # 一一对应IOU
    return iou


class LossYOLO_v2(nn.Module):

    def __init__(self, cfg=None):
        super(LossYOLO_v2, self).__init__()
        self.cfg = cfg

    def forward(self, pyolos, targets, imgs_ts=None):
        '''

        :param pyolos: torch.Size([3, 40, 13, 13]) [conf-1,class-3,box4] 5*8=40
        :param targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['size'] = target['size']
            target['image_id'] = int
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape

        '''--------------gt匹配---------------'''
        # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
        gdim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4  # torch.Size([3, 13, 13, 5, 13])
        gyolos = torch.empty((batch, h, w, cfg.NUM_ANC, gdim), device=device)

        # 匹配GT
        for i, target in enumerate(targets):  # batch遍历
            boxes_ltrb_b = target['boxes']  # ltrb
            labels_b = target['labels']

            # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
            gyolos[i] = fmatch4yolov2_one(
                gboxes_ltrb_b=boxes_ltrb_b,
                glabels_b=labels_b,
                grid=h,  # 7
                dim=gdim,
                device=device,
                cfg=cfg,
                img_ts=imgs_ts[i],
            )

            '''可视化验证'''
            if cfg.IS_VISUAL:
                # conf-1, cls-3, box-4, weight-1
                gyolo_test = gyolos[i].clone()  # torch.Size([32, 13, 13, 9])
                gyolo_test = gyolo_test.view(-1, gdim)
                gconf_one = gyolo_test[:, 0]
                # mask_pos = torch.logical_or(gconf_one == 1, gconf_one == -1)
                mask_pos = gconf_one == 1

                gtxywh = gyolo_test[:, 1 + cfg.NUM_CLASSES:1 + cfg.NUM_CLASSES + 4]
                # 这里是修复是 xy
                _xy_grid = gtxywh[:, :2] + f_mershgrid(h, w, is_rowcol=False, num_repeat=cfg.NUM_ANC).to(device)
                hw_ts = torch.tensor((h, w), device=device)
                gtxywh[:, :2] = torch.true_divide(_xy_grid, hw_ts)
                gtxywh = gtxywh[mask_pos]

                # boxes_decode4yolo1 这个用于三维
                if cfg.loss_args['s_match'] == 'whoned':
                    gtxywh[:, 2:4] = torch.sigmoid(gtxywh[:, 2:])
                elif cfg.loss_args['s_match'] == 'log':
                    gtxywh[:, 2:4] = torch.exp(gtxywh[:, 2:]) / cfg.IMAGE_SIZE[0]  # wh log-exp
                elif cfg.loss_args['s_match'] == 'log_g':
                    gtxywh[:, 2:4] = torch.exp(gtxywh[:, 2:]) / h  # 原图归一化
                else:
                    raise Exception('类型错误')

                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                img_ts = f_recover_normalization4ts(imgs_ts[i])
                from torchvision.transforms import functional as transformsF
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                import numpy as np
                img_np = np.array(img_pil)
                f_show_od_np4plt(img_np, gboxes_ltrb=boxes_ltrb_b.cpu()
                                 , pboxes_ltrb=xywh2ltrb(gtxywh.cpu()), is_recover_size=True,
                                 grids=(h, w))

        # pbox解码计算 匹配的iou 作为正例conf
        s_ = 1 + cfg.NUM_CLASSES

        gconf = gyolos[:, :, 0]
        mask_pos = gconf > 0
        mask_neg = gconf == 0  # 忽略-1 不管
        nums_pos = (mask_pos.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)

        # [3, 40, 13, 13] -> [3, 8, 5, 13*13] -> [3, 169, 5, 8]
        pyolos = pyolos.view(batch, s_ + 4, cfg.NUM_ANC, - 1).permute(0, 3, 2, 1).contiguous()

        # 解码pxywh 计算预测与 GT 的 iou 作为 gconf
        with torch.no_grad():  # torch.Size([3, 40, 13, 13])
            ptxywh = pyolos[:, :, :, s_:s_ + 4]  # torch.Size([32, 169, 5, 4])

            gyolos = gyolos.view(batch, -1, gdim)  # 4d -> 3d [3, 13, 13, 5, 13] -> [3, 169*5, 13]
            mask_pos_2d = gyolos[:, :, 0] == 1  # 前面已匹配，降维运算 [3, xx, 13] -> [3, xx]
            gbox_p = gyolos[:, :, -4:]  # [3, 169*5, 13] ->  [3, 169*5, 4]

            # [nn,1]
            iou_p = calc_iou(ptxywh, gbox_p, batch, h, w, cfg, mask_pos_2d, imgs_ts)
            iou_p = iou_p.view(batch, -1)  # 匹配每批的IOU [nn,1] -> [batch,nn/batch]

        # gconf = gyolos[:, :, 0] # 正例使用1
        gconf = iou_p  # 使用 iou赋值

        # 4d -> 3d [32, 169, 5, 8] -> [32, 845, 8]
        pyolos = pyolos.view(batch, -1, 1 + cfg.NUM_CLASSES + 4)  # torch.Size([3, 169, 8])

        ''' ----------------cls损失---------------- '''
        pcls = pyolos[:, :, 1:s_].sigmoid()  # 归一
        gcls = gyolos[:, :, 1:s_]
        _loss_val = x_bce(pcls, gcls, reduction="none")
        loss_cls = ((_loss_val.sum(-1) * mask_pos).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[2]
        # flog.debug('loss_cls:%s', loss_cls)

        ''' ----------------conf损失 ---------------- '''
        pconf = pyolos[:, :, 0].sigmoid()  # 这个需要归一化 torch.Size([3, 845])

        # ------------conf-mse ------------
        _loss_val = F.mse_loss(pconf, gconf, reduction="none")
        # _loss_val = F.binary_cross_entropy_with_logits(pconf, gconf, reduction="none")
        loss_conf_pos = ((_loss_val * mask_pos).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[0]
        loss_conf_neg = ((_loss_val * mask_neg).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[1]
        # flog.debug('loss_conf_pos:%s', loss_conf_pos)
        # flog.debug('loss_conf_neg:%s', loss_conf_neg)

        # if cfg.loss_args['s_conf'] == 'foc':
        #     l_pos, l_neg = focalloss_v2(pconf, gconf, mask_pos=mask_pos, mask_neg=mask_neg,
        #                                 alpha=0.25, gamma=2, is_merge=False)
        #     loss_conf_pos = l_pos.sum(-1).mean()  # 批数平均
        #     loss_conf_neg = l_neg.sum(-1).mean()  # 批数平均
        # elif cfg.loss_args['s_conf'] == 'mse':
        #     _loss_val = F.mse_loss(pconf, gconf, reduction="none")
        #     _loss_val = F.binary_cross_entropy_with_logits(pconf, gconf, reduction="none")
        # loss_conf_pos = (_loss_val * mask_pos).sum(-1).mean() * cfg.LOSS_WEIGHT[0]
        # loss_conf_neg = (_loss_val * mask_neg).sum(-1).mean() * cfg.LOSS_WEIGHT[1]
        # else:
        #     raise Exception('类型错误')

        ''' ---------------- box损失 ----------------- '''
        # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
        weight = gyolos[:, :, s_ + 4]  # torch.Size([32, 845])
        ptxty = pyolos[:, :, s_:s_ + 2].sigmoid()  # 这个需要归一化
        ptwth = pyolos[:, :, s_ + 2:s_ + 4]
        gtxty = gyolos[:, :, s_:s_ + 2]
        gtwth = gyolos[:, :, s_ + 2:s_ + 4]

        _loss_val = x_bce(ptxty, gtxty, reduction="none")
        loss_txty = ((_loss_val.sum(-1) * mask_pos * weight).sum(-1) / nums_pos).mean()
        _loss_val = F.mse_loss(ptwth, gtwth, reduction="none")
        loss_twth = ((_loss_val.sum(-1) * mask_pos * weight).sum(-1) / nums_pos).mean()
        # flog.debug('loss_txty:%s', loss_txty)
        # flog.debug('loss_twth:%s', loss_twth)

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


class PredictYOLO_v2(Predicting_Base):
    def __init__(self, cfg=None):
        super(PredictYOLO_v2, self).__init__(cfg)
        self.cfg = cfg

    def p_init(self, pyolos):
        self.batch, self.c, self.h, self.w = pyolos.shape
        self.s_ = 1 + cfg.NUM_CLASSES
        # [3, 40, 13, 13] -> [3, 8, 5, 13*13] -> [3, 169, 5, 8]
        pyolos = pyolos.view(self.batch, self.s_ + 4, cfg.NUM_ANC, - 1).permute(0, 3, 2, 1).contiguous()
        return pyolos

    def get_pscores(self, pyolos):
        _pyolos = pyolos.view(self.batch, -1, self.s_ + 4)  # [3, 845, 8]
        pconf = _pyolos[:, :, 0].sigmoid()
        cls_conf, plabels = _pyolos[:, :, 1:self.s_].sigmoid().max(-1)
        pscores = cls_conf * pconf

        return pscores, plabels, pconf

    def get_stage_res(self, pyolos, mask_pos, pscores, plabels):
        ids_batch1, _ = torch.where(mask_pos)

        # torch.Size([3, 169, 5, 8]) -> [3, 169, 5, 4]
        ptxywh = pyolos[:, :, :, self.s_:self.s_ + 4]
        pltrb = boxes_decode4yolo2(ptxywh, self.h, self.w, self.cfg)  # 预测 -> ltrb [3, 845, 4]

        pboxes_ltrb1 = pltrb[mask_pos]
        pboxes_ltrb1 = torch.clamp(pboxes_ltrb1, min=0., max=1.)

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1

        return ids_batch1, pboxes_ltrb1, plabels1, pscores1

    def forward(self, pyolos, imgs_ts=None):
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape

        # 解码txywh
        s_ = 1 + cfg.NUM_CLASSES
        # [3, 40, 13, 13] -> [3, 8, 5, 13*13] -> [3, 169, 5, 8]
        pyolos = pyolos.view(batch, s_ + 4, cfg.NUM_ANC, - 1).permute(0, 3, 2, 1).contiguous()

        _pyolos = pyolos.view(batch, -1, s_ + 4)  # [3, 845, 8]
        pconf = _pyolos[:, :, 0].sigmoid()
        cls_conf, plabels = _pyolos[:, :, 1:s_].sigmoid().max(-1)
        pscores = cls_conf * pconf

        mask_pos = pscores > cfg.THRESHOLD_PREDICT_CONF  # b,hw
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(pconf.max().item(),
                                                                          pconf.min().item(),
                                                                          pconf.mean().item(),
                                                                          ))
            return [None] * 5

        ids_batch1, _ = torch.where(mask_pos)

        # torch.Size([3, 169, 5, 8]) -> [3, 169, 5, 4]
        ptxywh = pyolos[:, :, :, s_:s_ + 4]
        pltrb = boxes_decode4yolo2(ptxywh, h, w, cfg)  # 预测 -> ltrb [3, 845, 4]

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
                                                                    cfg.THRESHOLD_PREDICT_NMS)

        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


class Yolo_v2_Net(nn.Module):
    def __init__(self, backbone, cfg):
        super(Yolo_v2_Net, self).__init__()
        self.backbone = backbone
        dims_out = backbone.dims_out  # (128, 256, 512)

        # detection head
        self.convsets_1 = nn.Sequential(
            CBL(dims_out[2], dims_out[2], 3, padding=1, leakyReLU=True),
            CBL(dims_out[2], dims_out[2], 3, padding=1, leakyReLU=True)
        )

        self.route_layer = CBL(dims_out[1], 64, 1, leakyReLU=True)
        self.reorg = ReorgLayer(stride=2)
        self.convsets_2 = CBL(dims_out[1] + dims_out[2], 1024, 3, padding=1, leakyReLU=True)

        # conf1 + box4 + cls3
        self.head = nn.Conv2d(1024, cfg.NUM_ANC * (1 + 4 + cfg.NUM_CLASSES), 1)

    def forward(self, x):
        # ceng2 torch.Size([32, 256, 26, 26]) , ceng3 torch.Size([32, 512, 13, 13])
        _, ceng2, ceng3 = self.backbone(x)

        # 双CBL混合  dim不变 torch.Size([32, 512, 13, 13])
        ceng3 = self.convsets_1(ceng3)

        # reorg尺寸加倍再恢复 [5, 64, 38, 38] -> [5, 256, 19, 19]  route from 16th layer in darknet
        ceng2 = self.reorg(self.route_layer(ceng2))

        # torch.Size([5, 1280, 19, 19]) route concatenate
        fp = torch.cat([ceng2, ceng3], dim=1)
        fp = self.convsets_2(fp)  # torch.Size([5, 1024, 19, 19])
        out = self.head(fp)  # torch.Size([5, 40, 19, 19])
        return out


class Yolo_v2_Netv2(nn.Module):
    def __init__(self, backbone, cfg):
        super(Yolo_v2_Netv2, self).__init__()
        self.backbone = backbone
        dim_layer = backbone.dims_out[2]  # (128, 256, 512)
        dim256 = backbone.dims_out[1]

        self.spp = nn.Sequential(
            CBL(dim_layer, dim256, ksize=1),
            SPPv2(),
            BottleneckCSP(dim256 * 4, dim_layer, n=1, shortcut=False)
        )
        self.sam = SAM(dim_layer)
        self.conv_set = BottleneckCSP(dim_layer, dim_layer, n=3, shortcut=False)

        # conf1 + box4 + cls3
        self.head = nn.Conv2d(dim_layer, cfg.NUM_ANC * (1 + 4 + cfg.NUM_CLASSES), 1)

    def forward(self, x):
        # ceng2 torch.Size([32, 256, 26, 26]) , ceng3 torch.Size([32, 512, 13, 13])
        ceng1, ceng2, ceng3 = self.backbone(x)

        outs = self.spp(ceng3)
        outs = self.sam(outs)
        outs = self.conv_set(outs)
        out = self.head(outs)
        return out


class Yolo_v2(nn.Module):
    def __init__(self, backbone, cfg):
        super(Yolo_v2, self).__init__()
        # self.net = Yolo_v2_Net(backbone, cfg)
        self.net = Yolo_v2_Netv2(backbone, cfg)

        self.losser = LossYOLO_v2(cfg=cfg)
        self.preder = PredictYOLO_v2(cfg=cfg)

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
    # model = darknet19(pretrained=True)
    # model = ModelOuts4DarkNet19(model)

    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)


    class CFG:
        pass


    cfg = CFG()
    cfg.NUM_CLASSES = 3
    cfg.NUM_ANC = 5
    net = Yolo_v2_Net(backbone=model, cfg=cfg)
    # x = torch.rand([5, 3, 416, 416])
    # print(net(x).shape)

    from f_pytorch.tools_model.model_look import f_look_tw

    f_look_tw(net, input=(5, 3, 416, 416), name='Yolo_v2_Net')
