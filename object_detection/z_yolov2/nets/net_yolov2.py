import torch
from torch import nn
import torch.nn.functional as F
from f_pytorch.tools_model.backbones.darknet import darknet19
from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet
from f_pytorch.tools_model.f_model_api import finit_weights, CBL, finit_conf_bias, ReorgLayer
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_lossfun import f_ghmc_v3, GHMC_Loss, focalloss_v2, FocalLoss_v2, focal_loss4center2, \
    show_distribution, focalloss_v3, f_bce
from f_tools.fits.f_match import boxes_encode4yolo, boxes_decode4yolo
from f_tools.fits.f_predictfun import label_nms
from f_tools.fun_od.f_boxes import offxy2xy, xywh2ltrb, calc_iou4ts, calc_iou4some_dim, ltrb2xywh, bbox_iou4one
from f_tools.pic.f_show import f_plt_show_cv
from f_tools.yufa.x_calc_adv import f_mershgrid


def fmatch4yolov2(gboxes_ltrb, labels, grid, dim, device, cfg, img_ts=None):
    '''
    匹配 gyolo 如果需要计算IOU 需在这里生成
    :param gboxes_ltrb: ltrb
    :param labels:
    :param grid: 13
    :param dim:
    :param device:
    :return:
    '''
    # 提取wh用于iou计算  --需要在dataset时验证标注有效性 脏数据
    gboxes_xywh = ltrb2xywh(gboxes_ltrb)
    giou_boxes_xywh = torch.zeros_like(gboxes_xywh, device=device)
    giou_boxes_xywh[:, 2:] = gboxes_xywh[:, 2:]

    anc_wh = torch.tensor(cfg.ANC_SIZE, device=device)
    anciou_xywh = torch.zeros((len(cfg.ANC_SIZE), 4), device=device)
    anciou_xywh[:, 2:] = anc_wh

    # 匹配一个最大的  iou>0.5 忽略
    iou2 = calc_iou4ts(xywh2ltrb(giou_boxes_xywh), xywh2ltrb(anciou_xywh))
    mask = iou2 > 0.5

    txywhs_g, weights, colrows_index = boxes_encode4yolo(gboxes_ltrb, grid, grid, device, cfg)
    # conf-1, cls-3, tbox-4, weight-1, gbox-4   torch.Size([13, 13, 13])
    p_yolo_one = torch.zeros((grid, grid, cfg.NUM_ANC, dim), device=device)

    labels = labels2onehot4ts(labels - 1, cfg.NUM_CLASSES)

    # 遍历格子
    for i, (col, row) in enumerate(colrows_index):
        index_max = torch.argmax(iou2, dim=-1)[i]
        indexs_ignore = torch.where(mask[i])

        # 有正例,将正例赋0 用于忽略
        for j in indexs_ignore:
            p_yolo_one[row, col, j, 0] = -1.
            p_yolo_one[row, col, j, 1 + cfg.NUM_CLASSES + 4] = -1.

        txywh_g = txywhs_g[i]
        # 正例的conf
        conf = torch.tensor([1], device=device)
        weight = weights[i]  # 1~2 小目标加成
        _labels = labels[i]

        # labels恢复至1
        t = torch.cat([conf, _labels, txywh_g, weight[None], gboxes_ltrb[i]], dim=0)
        p_yolo_one[row, col, index_max] = t

    return p_yolo_one


class LossYOLO_v2(nn.Module):

    def __init__(self, cfg=None):
        super(LossYOLO_v2, self).__init__()
        self.cfg = cfg
        self.ghmc_loss = GHMC_Loss(momentum=0.25)

    def forward(self, pyolos, targets, imgs_ts=None):
        '''

        :param pyolos: torch.Size([3, 40, 13, 13]) [conf-1,class-3,box4] 5*8=40
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape

        '''--------------gt匹配---------------'''
        # conf-1, cls-3, tbox-4, weight-1, gbox-4  = 13
        dim = 1 + cfg.NUM_CLASSES + 4 + 1 + 4  # torch.Size([3, 13, 13, 5, 13])
        gyolos = torch.empty((batch, h, w, cfg.NUM_ANC, dim), device=device)

        for i, target in enumerate(targets):  # batch遍历
            boxes_ltrb_one = target['boxes']  # ltrb
            labels_one = target['labels']

            gyolos[i] = fmatch4yolov2(
                gboxes_ltrb=boxes_ltrb_one,
                labels=labels_one,
                grid=h,  # 7
                dim=dim,
                device=device,
                cfg=cfg,
                img_ts=imgs_ts[i],
            )

            '''可视化验证'''
            if cfg.IS_VISUAL:
                # conf-1, cls-1, box-4, weight-1
                gyolo_test = gyolos[i].clone()  # torch.Size([32, 13, 13, 9])
                gyolo_test = gyolo_test.view(-1, dim)
                gconf_one = gyolo_test[:, 0]
                mask_pos = torch.logical_or(gconf_one == 1, gconf_one == -1)
                # mask_pos = gconf_one == 1

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
                f_plt_show_cv(img_np, gboxes_ltrb=boxes_ltrb_one.cpu()
                              , pboxes_ltrb=xywh2ltrb(gtxywh.cpu()), is_recover_size=True,
                              grids=(h, w))

        gyolos = gyolos.view(batch, -1, dim)  # torch.Size([3, 845, 13])
        '''--------------pred提取---------------'''
        # [3, 40, 13, 13] -> [3, 8, 5, 13*13] -> [3, 169, 5, 8]
        pyolos = pyolos.view(batch, 1 + cfg.NUM_CLASSES + 4, cfg.NUM_ANC, - 1).permute(0, 3, 2, 1).contiguous()
        pyolos = pyolos.view(batch, -1, 1 + cfg.NUM_CLASSES + 4)  # torch.Size([3, 169, 8])

        # ----------------conf损失----------------
        pconf = pyolos[:, :, 0].sigmoid()  # torch.Size([3, 845])
        gconf = gyolos[:, :, 0]
        weight = gyolos[:, :, 8]
        mask_pos = gconf == 1  # 忽略的-1不计
        mask_neg = gconf == 0
        _loss_val = F.mse_loss(pconf, gconf, reduction="none")
        loss_conf_pos = (_loss_val * mask_pos).sum(-1).mean() * 5.
        loss_conf_neg = (_loss_val * mask_neg).sum(-1).mean() * 1.

        # ----------------cls损失----------------
        s_ = 1 + cfg.NUM_CLASSES
        pcls = pyolos[:, :, 1:s_][mask_pos]
        gcls = gyolos[:, :, 1:s_][mask_pos]
        _loss_val = F.binary_cross_entropy_with_logits(pcls, gcls, reduction="none")
        loss_cls = _loss_val.sum(-1).mean()

        # ----------------box损失-----------------
        ptxty = pyolos[:, :, s_:s_ + 2]
        ptwth = pyolos[:, :, s_ + 2:s_ + 4]
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


class PredictYOLO_v2(nn.Module):
    def __init__(self, cfg=None):
        super(PredictYOLO_v2, self).__init__()
        self.cfg = cfg

    def forward(self, pyolos, imgs_ts=None):
        '''

        :param pyolos: torch.Size([3, 40, 13, 13])
        :return:
            ids_batch2 [nn]
            p_boxes_ltrb2 [nn,4]
            p_labels2 [nn]
            p_scores2 [nn]
        '''
        cfg = self.cfg
        device = pyolos.device
        batch, c, h, w = pyolos.shape

        # conf-1, cls-3, tbox-4, weight-1, gbox-4  = 13
        dim = 1 + cfg.NUM_CLASSES + 4
        # [3, 40, 13, 13] -> [3, 8, 5, 13*13] -> [3, 169, 5, 8]
        pyolos = pyolos.view(batch, dim, cfg.NUM_ANC, - 1).permute(0, 3, 2, 1).contiguous()
        # [3, 169, 5, 8] -> [3, 169, 8]
        pyolos = pyolos.view(batch, -1, dim)

        pconf = pyolos[:, :, 0].sigmoid()  # b,hw,c -> b,hw
        cls_conf, plabels = pyolos[:, :, 1:1 + cfg.NUM_CLASSES].sigmoid().max(-1)
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

        ''' 预测 这里是修复是 xywh   torch.Size([3, 845, 4])'''
        boxes_decode4yolo(ptxywh, h, w, cfg)

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
                                                                    self.cfg.THRESHOLD_PREDICT_NMS)

        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


class Yolo_v2_Net(nn.Module):
    def __init__(self, backbone, cfg):
        super(Yolo_v2_Net, self).__init__()
        self.backbone = backbone

        # detection head
        self.convsets_1 = nn.Sequential(
            CBL(1024, 1024, 3, padding=1, leakyReLU=True),
            CBL(1024, 1024, 3, padding=1, leakyReLU=True)
        )

        self.route_layer = CBL(512, 64, 1, leakyReLU=True)
        self.reorg = ReorgLayer(stride=2)
        self.convsets_2 = CBL(1280, 1024, 3, padding=1, leakyReLU=True)

        # conf1 + box4 + cls3
        self.head = nn.Conv2d(1024, cfg.NUM_ANC * (1 + 4 + cfg.NUM_CLASSES), 1)

    def forward(self, x, targets=None):
        # ceng2, ceng3合并
        ceng1, ceng2, ceng3 = self.backbone(x)

        # conv层集合 2层  不变 [5, 1024, 19, 19]
        ceng3 = self.convsets_1(ceng3)

        # 尺寸加倍再恢复 [5, 64, 38, 38] -> [5, 256, 19, 19]  route from 16th layer in darknet
        ceng2 = self.reorg(self.route_layer(ceng2))

        # torch.Size([5, 1280, 19, 19]) route concatenate
        fp = torch.cat([ceng2, ceng3], dim=1)
        fp = self.convsets_2(fp)  # torch.Size([5, 1024, 19, 19])
        out = self.head(fp)  # torch.Size([5, 40, 19, 19])
        return out


class Yolo_v2(nn.Module):
    def __init__(self, backbone, cfg):
        super(Yolo_v2, self).__init__()
        self.net = Yolo_v2_Net(backbone, cfg)

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
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, None, p_labels, p_scores


if __name__ == '__main__':
    model = darknet19(pretrained=True)
    model = ModelOuts4DarkNet(model)


    class CFG:
        pass


    cfg = CFG()
    cfg.NUM_CLASSES = 3
    cfg.THRESHOLD_PREDICT_CONF = 0.3
    cfg.THRESHOLD_PREDICT_NMS = 0.3
    cfg.NUMS_ANC = [3, 3, 3]
    cfg.NUM_GRID = 7
    net = Yolo_v2_Net(backbone=model, cfg=cfg)
    # x = torch.rand([5, 3, 416, 416])
    # print(net(x).shape)

    from f_pytorch.tools_model.model_look import f_look_model

    f_look_model(net, input=(5, 3, 608, 608), name='Yolo_v2_Net')