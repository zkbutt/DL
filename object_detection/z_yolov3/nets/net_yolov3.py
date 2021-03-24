import torch
import torch.nn as nn

from f_pytorch.tools_model.f_model_api import CBL, finit_conf_bias_one
from f_pytorch.tools_model.model_look import f_look_summary
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import boxes_decode4yolo3
from f_tools.fits.f_predictfun import label_nms
from f_tools.floss.f_lossfun import x_bce
from f_tools.floss.focal_loss import focalloss

from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2xywh, calc_iou4ts, bbox_iou4one
import torch.nn.functional as F
import math

from f_tools.pic.f_show import f_show_od_ts4plt, f_show_od_np4plt
from f_tools.yufa.x_calc_adv import f_mershgrid


def fmatch4yolov3(gboxes_ltrb_b, glabels_b, dim, ptxywh_b, device, cfg, img_ts=None, pconf_b=None):
    '''

    :param gboxes_ltrb_b:
    :param glabels_b:
    :param dim:
    :param ptxywh_b: torch.Size([3, 10647, 4])
    :param device:
    :param cfg:
    :param img_ts:
    :return:
    '''
    ''' 只有wh计算与多层anc的IOU '''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = ancs_wh_ts

    # 匹配一个最大的anc 用于获取iou index  iou>0.5 忽略
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    '''取iou最大的一个anc匹配'''
    # mask = iou2d > 0.5
    # iou_max_val, iou_max_index = iou2d.max(1)
    ids_p_anc = torch.argmax(iou2d, dim=-1)  # 匹配最大的IOU
    # ------------------------- yolo23一样 ------------------------------

    # 匹配完成的数据 [2704, 676, 169] 层数量*anc数
    _dim_total = sum(cfg.NUMS_CENG) * 3  # 10647
    _num_anc_total = len(cfg.ANCS_SCALE)  # 9个
    # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
    g_yolo_one = torch.zeros((_dim_total, dim), device=device)  # 返回值

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    '''回归参数由于每层的grid不一样,需要遍历'''

    whs = gboxes_xywh[:, 2:]  # 归一化值
    weights = 2.0 - torch.prod(whs, dim=-1)

    '''遍历GT 匹配iou最大的  其它anc全部忽略'''
    for i in range(len(gboxes_ltrb_b)):
        cxy = gboxes_xywh[i][:2]
        for index_p_anc in ids_p_anc:
            # 最大IOU匹配到哪一层 每层的anc数  7/3 向下取整
            index_p_ceng = torch.true_divide(index_p_anc, cfg.NUMS_ANC[0]).type(torch.int32)
            for j, num_ceng in enumerate(cfg.NUMS_CENG):
                offset_ceng = torch.tensor(cfg.NUMS_CENG, dtype=torch.int32)[:j].sum()  # 0个sum=0

                grid = math.sqrt(num_ceng)  # 本层的网格 52 或 26 或 13
                grids_ts = torch.tensor([grid, grid], device=device, dtype=torch.int32)
                index_colrow = (cxy * grids_ts).type(torch.int32)
                col, row = index_colrow

                offset_colrow = row * grid + col
                # cfg.NUMS_ANC[0] 每层anc数是一样的 索引要int
                _index_dim = ((offset_ceng + offset_colrow) * cfg.NUMS_ANC[0]).long()
                g_yolo_one[_index_dim:_index_dim + cfg.NUMS_ANC[j], 0] = -1  # 所在的格子全部忽略 -1
                if j == index_p_ceng:
                    offset_anc = index_p_anc % cfg.NUMS_ANC[j]  # 余数是每一层的偏移
                    index_p_dim = (_index_dim + offset_anc).type(torch.int32)
                    conf = torch.tensor([1], device=device)

                    # 编码  归一化网格左上角位置 用于原图归一化中心点减 求偏差
                    offset_xy = torch.true_divide(index_colrow, grids_ts[0])
                    txy_g = (cxy - offset_xy) * grids_ts  # 特图偏移
                    anc_match_ts = ancs_wh_ts[index_p_anc]
                    # 比例 /log
                    twh_g = (gboxes_xywh[i][2:] / anc_match_ts).log()
                    txywh_g = torch.cat([txy_g, twh_g], dim=-1)

                    _t = torch.cat([conf, glabels_b[i], txywh_g, weights[i][None], gboxes_ltrb_b[i]], dim=0)
                    g_yolo_one[index_p_dim] = _t  # 匹配到的正例

                    if cfg.IS_VISUAL:
                        ''' 可视化匹配最大的ANC '''
                        from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                        _img_ts = f_recover_normalization4ts(img_ts.clone())
                        from torchvision.transforms import functional as transformsF
                        img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
                        import numpy as np
                        img_np = np.array(img_pil)
                        anc_p = torch.cat([cxy, anc_match_ts], dim=-1)[None]
                        anc_o = torch.cat([cxy.repeat(ancs_wh_ts.shape[0], 1), ancs_wh_ts], dim=-1)
                        f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b[i][None].cpu()
                                         , pboxes_ltrb=xywh2ltrb(anc_p.cpu()),
                                         other_ltrb=xywh2ltrb(anc_o.cpu()),
                                         is_recover_size=True)

    return g_yolo_one


def fmatch4yolov3_v2(gboxes_ltrb_b, glabels_b, dim, ptxywh_b, device, cfg, img_ts=None, pconf_b=None):
    '''
    每层匹配一个IOU最好的 可选大于0.1 的
    :param gboxes_ltrb_b:
    :param glabels_b:
    :param dim:
    :param ptxywh_b: torch.Size([3, 10647, 4])
    :param device:
    :param cfg:
    :param img_ts:
    :return:
    '''
    ''' 只有wh计算与多层anc的IOU '''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    ngt, _ = gboxes_ltrb_b.shape
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = ancs_wh_ts

    # 匹配一个最大的anc 用于获取iou index  iou>0.5 忽略  (ngt,anc数)
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    # 每层选一个IOU最大的
    # indexs_p_anc = iou2d.view(ngt, len(cfg.NUMS_CENG), -1).argmax(-1)
    ious_p, indexs_p_anc = iou2d.view(ngt, len(cfg.NUMS_CENG), -1).max(-1)

    # # 匹配完成的数据 [2704, 676, 169] 层数量*anc数
    _dim_total = sum(cfg.NUMS_CENG) * 3  # 10647
    _num_anc_total = len(cfg.ANCS_SCALE)  # 9个
    # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
    g_yolo_one = torch.zeros((_dim_total, dim), device=device)  # 返回值

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    whs = gboxes_xywh[:, 2:]  # 归一化值
    weights = 2.0 - torch.prod(whs, dim=-1)

    '''遍历GT 匹配每层iou最大的三个  其它anc全部忽略'''
    for i in range(len(gboxes_ltrb_b)):
        cxy = gboxes_xywh[i][:2]
        indexs_cp = indexs_p_anc[i]  # tensor([1, 2, 1])
        for j, num_ceng in enumerate(cfg.NUMS_CENG):
            # flog.debug('ious_p %s', ious_p[i, j])
            if ious_p[i, j] <= 0.1:  # 提高准确降低召回
                continue  # 太小的不要

            offset_ceng = torch.tensor(cfg.NUMS_CENG, dtype=torch.int32)[:j].sum()  # 0个sum=0
            grid = math.sqrt(num_ceng)  # 本层的网格 52 或 26 或 13
            grids_ts = torch.tensor([grid, grid], device=device, dtype=torch.int32)  # (52,52)
            index_colrow = (cxy * grids_ts).type(torch.int32)
            col, row = index_colrow
            offset_colrow = row * grid + col  # cfg.NUMS_ANC=[3, 3, 3]
            _index_dim = ((offset_ceng + offset_colrow) * cfg.NUMS_ANC[j]).long()

            g_yolo_one[_index_dim:_index_dim + cfg.NUMS_ANC[j], 0] = -1  # 该层匹配的anc先全为-1
            index_p_dim = (_index_dim + indexs_cp[j]).type(torch.int32)

            conf = torch.tensor([1], device=device)

            # 编码  归一化网格左上角位置 用于原图归一化中心点减 求偏差
            offset_xy = torch.true_divide(index_colrow, grids_ts[0])
            txy_g = (cxy - offset_xy) * grids_ts  # 特图偏移
            anc_match_ts = ancs_wh_ts[j * cfg.NUMS_ANC[j] + indexs_cp[j]]
            # 比例 /log
            twh_g = (gboxes_xywh[i][2:] / anc_match_ts).log()
            txywh_g = torch.cat([txy_g, twh_g], dim=-1)

            _t = torch.cat([conf, glabels_b[i], txywh_g, weights[i][None], gboxes_ltrb_b[i]], dim=0)
            g_yolo_one[index_p_dim] = _t  # 匹配到的正例

            if cfg.IS_VISUAL:
                ''' 可视化匹配最大的ANC '''
                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                _img_ts = f_recover_normalization4ts(img_ts.clone())
                from torchvision.transforms import functional as transformsF
                img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
                import numpy as np
                img_np = np.array(img_pil)
                anc_p = torch.cat([cxy, anc_match_ts], dim=-1)[None]
                anc_o = torch.cat([cxy.repeat(ancs_wh_ts.shape[0], 1), ancs_wh_ts], dim=-1)
                f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b[i][None].cpu()
                                 , pboxes_ltrb=xywh2ltrb(anc_p.cpu()),
                                 other_ltrb=xywh2ltrb(anc_o.cpu()),
                                 is_recover_size=True)
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

        # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
        gdim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4
        # h*w*anc
        gyolos = torch.empty((batch, hwa, gdim), device=device)

        # 匹配GT
        for i, target in enumerate(targets):  # batch遍历
            gboxes_ltrb_b = target['boxes']  # ltrb
            glabels_b = target['labels']

            ''' 可视化在里面 每层特图不一样'''
            gyolos[i] = fmatch4yolov3(gboxes_ltrb_b=gboxes_ltrb_b,
                                      glabels_b=glabels_b,
                                      dim=gdim,
                                      ptxywh_b=ptxywh[i],
                                      device=device, cfg=cfg,
                                      img_ts=imgs_ts[i],
                                      pconf_b=pconf[i])

            # gyolos[i] = fmatch4yolov3_v2(gboxes_ltrb_b=gboxes_ltrb_b,
            #                              glabels_b=glabels_b,
            #                              dim=gdim,
            #                              ptxywh_b=ptxywh[i],
            #                              device=device, cfg=cfg,
            #                              img_ts=imgs_ts[i],
            #                              pconf_b=pconf[i])

        # gyolos [3, 10647, 13] conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
        s_ = 1 + cfg.NUM_CLASSES

        gconf = gyolos[:, :, 0]  # 正例使用1

        mask_pos = gconf > 0  # 同维bool索引 忽略的-1不计
        mask_neg = gconf == 0  # 忽略-1 不管
        nums_pos = (mask_pos.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)

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

        ''' ----------------cls损失---------------- '''
        pcls_sigmoid = pcls.sigmoid()  # 归一
        gcls = gyolos[:, :, 1:s_]
        _loss_val = x_bce(pcls_sigmoid, gcls, reduction="none")
        l_cls = ((_loss_val.sum(-1) * mask_pos).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[2]

        ''' ----------------conf损失 ---------------- '''
        pconf_sigmoid = pconf.sigmoid().view(batch, -1)  # [3, 10647, 1] -> [3, 10647]

        # ------------conf-mse ------------'''
        _loss_val = F.mse_loss(pconf_sigmoid, gconf, reduction="none")
        # _loss_val = F.binary_cross_entropy_with_logits(pconf_sigmoid, gconf, reduction="none")
        # l_conf_pos = ((_loss_val * mask_pos).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[0]
        # l_conf_neg = ((_loss_val * mask_neg).sum(-1) / nums_pos).mean() * cfg.LOSS_WEIGHT[1]

        pos_ = _loss_val[mask_pos]
        l_conf_pos = pos_.mean() * cfg.LOSS_WEIGHT[0] * 30
        l_conf_neg = _loss_val[mask_neg].mean() * cfg.LOSS_WEIGHT[1] * 30

        # ------------conf-focalloss ------------'''
        # mash_ignore = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
        # l_pos, l_neg = focalloss(pconf_sigmoid, gconf, mask_pos=mask_pos, mash_ignore=mash_ignore, is_debug=True)
        # l_conf_pos = (l_pos.sum(-1) / nums_pos).mean()
        # l_conf_neg = (l_neg.sum(-1) / nums_pos).mean()

        ''' ----------------box损失   xy采用bce wh采用mes----------------- '''
        # conf-1, cls-3, tbox-4, weight-1, gltrb-4  = 13
        weight = gyolos[:, :, s_ + 4]  # torch.Size([32, 845])
        ptxty_sigmoid = ptxywh[:, :, :2].sigmoid()  # 这个需要归一化
        ptwth = ptxywh[:, :, 2:4]
        gtxty = gyolos[:, :, s_:s_ + 2]
        gtwth = gyolos[:, :, s_ + 2:s_ + 4]

        _loss_val = x_bce(ptxty_sigmoid, gtxty, reduction="none")
        l_txty = ((_loss_val.sum(-1) * mask_pos * weight).sum(-1) / nums_pos).mean()
        _loss_val = F.mse_loss(ptwth, gtwth, reduction="none")
        l_twth = ((_loss_val.sum(-1) * mask_pos * weight).sum(-1) / nums_pos).mean()

        l_total = l_conf_pos + l_conf_neg + l_cls + l_txty + l_twth

        log_dict = {}
        log_dict['l_total'] = l_total.item()
        log_dict['l_conf_pos'] = l_conf_pos.item()
        log_dict['l_conf_neg'] = l_conf_neg.item()
        log_dict['l_cls'] = l_cls.item()
        log_dict['l_xy'] = l_txty.item()
        log_dict['l_wh'] = l_twth.item()

        log_dict['p_max'] = pconf.max().item()
        log_dict['p_min'] = pconf.min().item()
        log_dict['p_mean'] = pconf.mean().item()
        return l_total, log_dict


class PredictYOLO_v3(nn.Module):
    def __init__(self, cfg):
        super(PredictYOLO_v3, self).__init__()
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

        mask_pos = pscores > cfg.THRESHOLD_PREDICT_CONF
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(pconf.max().item(),
                                                                          pconf.min().item(),
                                                                          pconf.mean().item(),
                                                                          ))
            return [None] * 5

        # 最大1000个
        ids_topk = pscores.topk(500, dim=-1)[1]  # torch.Size([32, 1000])
        mask_topk = torch.zeros_like(mask_pos)
        mask_topk[torch.arange(ids_topk.shape[0])[:, None], ids_topk] = True
        mask_pos = torch.logical_and(mask_pos, mask_topk)

        ids_batch1, _ = torch.where(mask_pos)
        # 解码txywh 这个函数是预测的关键
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
                                                                    cfg.THRESHOLD_PREDICT_NMS)

        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


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
