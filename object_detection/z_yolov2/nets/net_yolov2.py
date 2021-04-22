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
from f_tools.fits.f_match import boxes_decode4yolo2, boxes_encode4yolo2, boxes_encode4yolo2_4iou
from f_tools.fits.f_predictfun import label_nms
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import x_bce
from f_tools.floss.focal_loss import focalloss
from f_tools.fun_od.f_boxes import xywh2ltrb, calc_iou4ts, ltrb2xywh, bbox_iou4one, xywh2ltrb4ts, bbox_iou4y
from f_tools.pic.f_show import f_show_od_np4plt, f_show_od_ts4plt
from f_tools.yufa.x_calc_adv import f_mershgrid


def fmatch4yolov2(gboxes_ltrb_b, glabels_b, grid, gdim, device, cfg, img_ts=None):
    '''
    匹配 gyolo 如果需要计算IOU 需在这里生成
    :param gboxes_ltrb_b: ltrb
    :param glabels_b:
    :param grid: 13
    :param gdim:
    :param device:
    :return: 匹配中心与GT相同 iou 最大的一个anc  其余的全为0
    '''

    '''与yolo1相比多出确认哪个anc大 计算GT的wh与anc的wh 确定哪个anc的iou大'''
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
    # ------------------------- yolo23一样 ------------------------------

    txywhs_g, weights, colrows_index = boxes_encode4yolo2(gboxes_ltrb_b, ids_p, grid, grid, device, cfg)

    # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4   torch.Size([13, 13, 13])
    g_yolo_one = torch.zeros((grid, grid, cfg.NUM_ANC, gdim), device=device)

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    # ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)

    # 遍历GT 匹配iou最大的  其它anc全部忽略
    for i, (col, row) in enumerate(colrows_index):
        g_yolo_one[row, col, :, 0] = -1  # 全部忽略
        # 正例的conf
        conf = torch.tensor([1], device=device)
        _t = torch.cat([conf, glabels_b[i], txywhs_g[i], weights[i][None], gboxes_ltrb_b[i]], dim=0)
        g_yolo_one[row, col, ids_p[i]] = _t

    return g_yolo_one


def fmatch4yolov2_4iou(gboxes_ltrb_b, glabels_b, grid, gdim, device, cfg, preg_b, img_ts=None):
    '''
    匹配 gyolo 如果需要计算IOU 需在这里生成
    :param gboxes_ltrb_b: ltrb
    :param glabels_b:
    :param grid: 13
    :param gdim:
    :param device:
    :return: 匹配中心与GT相同 iou 最大的一个anc  其余的全为0
    '''

    '''与yolo1相比多出确认哪个anc大 计算GT的wh与anc的wh 确定哪个anc的iou大'''
    # 提取[0,0,w,h]用于iou计算  --强制xy为0
    gboxes_xywh = ltrb2xywh(gboxes_ltrb_b)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]  # gwh 赋值 原图归一化用于与anc比较IOU

    anc_wh = torch.tensor(cfg.ANCS_SCALE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANCS_SCALE), 4), device=device)
    anciou_xywh[:, 2:] = anc_wh

    # 匹配一个最大的 用于获取iou index
    iou2d = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    # flog.debug('iou2d %s', iou2d)

    # [gt,anc] GT对应格子中哪个最大
    # index_p = iou2.max(-1)[1]  # 匹配最大的IOU
    ids_p = torch.argmax(iou2d, dim=-1)  # 匹配最大的IOU
    # ------------------------- yolo2 与yolo1一样只有一层 编码wh是比例 ------------------------------
    txywhs_g, weights, colrows_index = boxes_encode4yolo2_4iou(
        gboxes_ltrb_b=gboxes_ltrb_b,
        preg_b=preg_b,
        match_anc_ids=ids_p,  # 匹配的iou最大的anc索引
        grid_h=grid, grid_w=grid,
        device=device, cfg=cfg,
    )

    # gboxes_ltrb_b -> xy只取offxy
    _gboxes_ltrb_b_t = gboxes_ltrb_b * grid
    _gboxes_xywh_b_t = ltrb2xywh(_gboxes_ltrb_b_t)
    # 特图加偏移
    _gboxes_xywh_b_t[..., :2] = _gboxes_xywh_b_t[..., :2] - _gboxes_xywh_b_t[..., :2].long()
    gboxes_ltrb_b_toff = xywh2ltrb(_gboxes_xywh_b_t)

    # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4 , match_anc_ids-1,   torch.Size([13, 13, 13])
    g_yolo_one = torch.zeros((grid, grid, cfg.NUM_ANC, gdim), device=device)

    glabels_b = labels2onehot4ts(glabels_b - 1, cfg.NUM_CLASSES)
    # ancs_wh_ts = torch.tensor(cfg.ANCS_SCALE, device=device)

    # 遍历GT 匹配iou最大的  其它anc全部忽略
    for i, (col, row) in enumerate(colrows_index):
        g_yolo_one[row, col, :, 0] = -1  # 全部忽略
        # 正例的conf
        conf = torch.tensor([1], device=device)
        # 这里添加一个 匹配 index
        _t = torch.cat([conf, glabels_b[i], txywhs_g[i], weights[i][None],
                        gboxes_ltrb_b_toff[i], ids_p[i][None]], dim=0)
        g_yolo_one[row, col, ids_p[i]] = _t

    return g_yolo_one


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
        s_ = 1 + cfg.NUM_CLASSES
        # [3, 40, 13, 13] -> [3, 8, 5, 13*13] -> [3, 169, 5, 8]
        pyolos = pyolos.view(batch, s_ + 4, cfg.NUM_ANC, - 1).permute(0, 3, 2, 1).contiguous()
        # [3, 169, 5, 8] -> [3, 169*5, 8]
        pyolos = pyolos.view(batch, -1, s_ + 4)
        preg_pos = pyolos[..., s_:s_ + 4]

        '''--------------gt匹配---------------'''
        # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
        if cfg.MODE_TRAIN == 4:
            gdim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4 + 1  # torch.Size([3, 13, 13, 5, 13])
        else:
            gdim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4  # torch.Size([3, 13, 13, 5, 13])

        gyolos = torch.empty((batch, h, w, cfg.NUM_ANC, gdim), device=device)

        # 匹配GT
        for i, target in enumerate(targets):  # batch遍历
            gboxes_ltrb_b = target['boxes']  # ltrb
            glabels_b = target['labels']

            # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
            if cfg.MODE_TRAIN == 4:
                gyolos[i] = fmatch4yolov2_4iou(
                    gboxes_ltrb_b=gboxes_ltrb_b,
                    glabels_b=glabels_b,
                    grid=h,  # 7 只有一层
                    gdim=gdim,
                    device=device,
                    cfg=cfg,
                    preg_b=preg_pos[i],
                    img_ts=imgs_ts[i],
                )
            else:
                gyolos[i] = fmatch4yolov2(
                    gboxes_ltrb_b=gboxes_ltrb_b,
                    glabels_b=glabels_b,
                    grid=h,  # 7 只有一层
                    gdim=gdim,
                    device=device,
                    cfg=cfg,
                    img_ts=imgs_ts[i],
                )

            '''可视化验证'''
            if cfg.IS_VISUAL:
                # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
                gyolo_test = gyolos[i].clone()  # torch.Size([32, 13, 13, 9])
                gyolo_test = gyolo_test.view(-1, gdim)
                gconf_one = gyolo_test[:, 0]
                # mask_pos = torch.logical_or(gconf_one == 1, gconf_one == -1)
                mask_pos_2d = gconf_one == 1

                gtxywh = gyolo_test[:, 1 + cfg.NUM_CLASSES:1 + cfg.NUM_CLASSES + 4]
                # 这里是修复是 xy
                _xy_grid = gtxywh[:, :2] + f_mershgrid(h, w, is_rowcol=False, num_repeat=cfg.NUM_ANC).to(device)
                hw_ts = torch.tensor((h, w), device=device)
                gtxywh[:, :2] = torch.true_divide(_xy_grid, hw_ts)
                gtxywh = gtxywh[mask_pos_2d]

                gtxywh[:, 2:4] = torch.exp(gtxywh[:, 2:]) / h  # 原图归一化

                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                img_ts = f_recover_normalization4ts(imgs_ts[i])
                from torchvision.transforms import functional as transformsF
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                import numpy as np
                img_np = np.array(img_pil)
                f_show_od_np4plt(img_np, gboxes_ltrb=gboxes_ltrb_b.cpu()
                                 , pboxes_ltrb=xywh2ltrb(gtxywh.cpu()), is_recover_size=True,
                                 grids=(h, w))

        # torch.Size([32, 13, 13, 5, 13]) -> [32, 13*13*5, 13]
        gyolos = gyolos.view(batch, -1, gdim)
        gconf = gyolos[:, :, 0]  # 正例使用1  torch.Size([32, 910])
        mask_pos_2d = gconf > 0
        mask_neg_2d = gconf == 0  # 忽略-1 不管
        nums_pos = (mask_pos_2d.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        nums_neg = (mask_neg_2d.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        pyolos_pos = pyolos[mask_pos_2d]  # torch.Size([32, 845, 8]) -> torch.Size([40, 8])
        gyolos_pos = gyolos[mask_pos_2d]  # torch.Size([32, 845, 13]) -> torch.Size([40, 8])

        ''' ----------------cls损失---------------- '''
        pcls_sigmoid_pos = pyolos_pos[:, 1:s_].sigmoid()
        gcls_pos = gyolos_pos[:, 1:s_]
        _loss_val = x_bce(pcls_sigmoid_pos, gcls_pos, reduction="none")  # torch.Size([46, 3])
        # torch.Size([46, 3]) -> val
        l_cls = _loss_val.sum(-1).mean() * cfg.LOSS_WEIGHT[2]

        ''' ----------------conf损失 ---------------- '''
        pconf_sigmoid = pyolos[:, :, 0].sigmoid()  # 这个需要归一化 torch.Size([3, 845])

        # ------------conf-mse ------------
        # _loss_val = F.mse_loss(pconf_sigmoid, gconf, reduction="none")
        # l_conf_pos = ((_loss_val * mask_pos_2d).sum(-1) / nums_pos).mean() * 10
        # l_conf_neg = ((_loss_val * mask_neg_2d).sum(-1) / nums_neg).mean() * 30

        # ------------ focalloss   ------------
        mash_ignore_2d = torch.logical_not(torch.logical_or(mask_pos_2d, mask_neg_2d))
        l_pos, l_neg = focalloss(pconf_sigmoid, gconf, mask_pos=mask_pos_2d,
                                 mash_ignore=mash_ignore_2d, is_debug=True, alpha=0.5)
        l_conf_pos = (l_pos.sum(-1).sum(-1) / nums_pos).mean()
        l_conf_neg = (l_neg.sum(-1).sum(-1) / nums_neg).mean() * 3

        ''' ---------------- box损失 ----------------- '''
        log_dict = {}
        if cfg.MODE_TRAIN == 4:
            # ------------ iou损失   ------------
            # 解码pxywh 计算预测与 GT 的 iou 作为 gconf
            preg_pos = pyolos_pos[:, s_:s_ + 4]
            gltrb_pos_tx = gyolos_pos[:, s_ + 4 + 1:s_ + 4 + 1 + 4]
            match_anc_ids = gyolos_pos[:, s_ + 4 + 1 + 4]

            # 解码yolo2 特图尺寸
            pxy_pos_sigmoid = preg_pos[..., :2].sigmoid()

            # 这里与yolo1不一样
            match_ancs = torch.tensor(cfg.ANCS_SCALE, device=device)[match_anc_ids.long()]
            pwh_pos_scale = torch.exp(preg_pos[..., 2:4]) * match_ancs * h  # 恢复到特图
            pzxywh = torch.cat([pxy_pos_sigmoid, pwh_pos_scale], -1)

            iou_zg = bbox_iou4one(xywh2ltrb4ts(pzxywh), gltrb_pos_tx, is_giou=True)
            # iou_zg = bbox_iou4y(xywh2ltrb4ts(pzxywh), gltrb_pos_tx, GIoU=True)
            # print(iou_zg)
            l_reg = (1 - iou_zg).mean() * 2

            ''' ---------------- loss完成 ----------------- '''
            l_total = l_conf_pos + l_conf_neg + l_cls + l_reg
            log_dict['l_reg'] = l_reg.item()
        else:
            # ------------ mse+bce   ------------ 666666
            # conf-1, cls-num_class, txywh-4, weight-1, gltrb-4
            pxy_pos_sigmoid = pyolos_pos[:, s_:s_ + 2].sigmoid()  # 这个需要归一化
            pwh_pos_scale = pyolos_pos[:, s_ + 2:s_ + 4]
            weight_pos = gyolos_pos[:, s_ + 4 + 1]  # torch.Size([32, 845])
            gtxy_pos = gyolos_pos[:, s_:s_ + 2]  # [nn]
            gtwh_pos = gyolos_pos[:, s_ + 2:s_ + 4]

            _loss_val = x_bce(pxy_pos_sigmoid, gtxy_pos, reduction="none")
            l_txty = (_loss_val.sum(-1) * weight_pos).mean()
            _loss_val = F.mse_loss(pwh_pos_scale, gtwh_pos, reduction="none")
            l_twth = (_loss_val.sum(-1) * weight_pos).mean()

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


class PredictYOLO_v2(Predicting_Base):
    def __init__(self, cfg=None):
        super(PredictYOLO_v2, self).__init__(cfg)

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
    '''这个与 YOLO1 一致'''

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
