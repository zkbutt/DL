import torch
from torch import nn
import torch.nn.functional as F
from f_pytorch.tools_model.f_model_api import finit_weights, CBL, finit_conf_bias
from f_pytorch.tools_model.fmodels.model_modules import BottleneckCSP, SAM, SPP, SPPv2
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_lossfun import f_ghmc_v3, GHMC_Loss, focalloss_v2, FocalLoss_v2, focal_loss4center2, \
    show_distribution, focalloss_v3, f_bce
from f_tools.fits.f_match import boxes_encode4yolo1, boxes_decode4yolo1
from f_tools.fits.f_predictfun import label_nms
from f_tools.fun_od.f_boxes import offxy2xy, xywh2ltrb, calc_iou4ts, calc_iou4some_dim, ltrb2xywh
from f_tools.pic.f_show import f_plt_show_cv
from f_tools.yufa.x_calc_adv import f_mershgrid


def fmatch4yolov1(boxes_ltrb, labels, grid, size_in, device, cfg, img_ts=None):
    '''
    匹配 gyolo 如果需要计算IOU 需在这里生成
    :param boxes_ltrb: ltrb
    :param labels:
    :param grid: 13
    :param size_in:
    :param device:
    :return:
    '''
    # 需要在dataset时验证标注有效性
    txywhs_g, weights, colrows_index = boxes_encode4yolo1(boxes_ltrb, grid, grid, device, cfg)

    if cfg.loss_args['s_cls'] == 'ce':  # 多值交叉熵匹配
        # conf-1,cls-1,box-4,weight-1
        p_yolo_one = torch.zeros((grid, grid, 1 + 1 + 4 + 1), device=device)
        labels = (labels - 1)[:, None]  # 后面加维
    elif cfg.loss_args['s_cls'] == 'bce':  # 二值交叉熵匹配
        # conf-1,cls-3,box-4,weight-1
        p_yolo_one = torch.zeros((grid, grid, 1 + cfg.NUM_CLASSES + 4 + 1), device=device)
        labels = labels2onehot4ts(labels - 1, cfg.NUM_CLASSES)
    else:
        raise Exception("cfg.loss_args['s_cls'] 类型错误 %s" % cfg.loss_args['s_cls'])

    # 遍历格子
    for i, (col, row) in enumerate(colrows_index):
        txywh_g = txywhs_g[i]
        # 正例的conf
        conf = torch.tensor([1], device=device, dtype=torch.int16)
        weight = weights[i]  # 1~2 小目标加成
        _labels = labels[i]
        # labels恢复至1
        t = torch.cat([conf, _labels, txywh_g, weight[None]], dim=0)
        p_yolo_one[row, col] = t

    return p_yolo_one


class LossYOLO_v1(nn.Module):

    def __init__(self, cfg=None):
        super(LossYOLO_v1, self).__init__()
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

        # conf-1, cls-1, box-4, weight-1
        if cfg.loss_args['s_cls'] == 'ce':
            dim = 1 + 1 + 4 + 1
        elif cfg.loss_args['s_cls'] == 'bce':  # 二值交叉熵匹配
            dim = 1 + cfg.NUM_CLASSES + 4 + 1
        else:
            raise Exception("cfg.loss_args['s_cls'] 类型错误 %s" % cfg.loss_args['s_cls'])

        gyolos = torch.empty((batch, h, w, dim), device=device)

        for i, target in enumerate(targets):  # batch遍历
            boxes_ltrb_one = target['boxes']  # ltrb
            labels_one = target['labels']

            gyolos[i] = fmatch4yolov1(
                boxes_ltrb=boxes_ltrb_one,
                labels=labels_one,
                grid=h,  # 7
                size_in=cfg.IMAGE_SIZE,
                device=device,
                img_ts=imgs_ts[i],
                cfg=cfg
            )

            '''可视化验证'''
            if cfg.IS_VISUAL:
                # conf-1, cls-1, box-4, weight-1
                gyolo_test = gyolos[i].clone()  # torch.Size([32, 13, 13, 9])
                gyolo_test = gyolo_test.view(-1, dim)
                gconf_one = gyolo_test[:, 0]
                mask_pos = gconf_one == 1

                gtxywh = gyolo_test[:, 1 + cfg.NUM_CLASSES:1 + cfg.NUM_CLASSES + 4]
                # 这里是修复是 xy
                _xy_grid = gtxywh[:, :2] + f_mershgrid(h, w, is_rowcol=False).to(device)
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

        # a = 1.05
        # pconf = a * pyolos[:, :, 0].sigmoid() - (a - 1) / 2

        s_ = 1 + cfg.NUM_CLASSES

        gyolos = gyolos.view(batch, -1, dim)  # b,hw,7
        gconf = gyolos[:, :, 0]  # torch.Size([5, 169])

        # ----------------cls损失----------------
        if cfg.loss_args['s_cls'] == 'ce':  # 二分类用这个
            pcls = pyolos[:, :, 1:s_].permute(0, 2, 1)  # 自动ce
            gcls = gyolos[:, :, 1].long()  # torch.Size([5, 169])

            '''-----------这两个只是正例  正例计算损失,按批量----------'''
            _loss_val = F.cross_entropy(pcls, gcls, reduction="none")
            loss_cls = (_loss_val * gconf).sum(-1).mean() * cfg.LOSS_WEIGHT[2]
            _dim = 1 + 1

        elif cfg.loss_args['s_cls'] == 'bce':  # 二值交叉熵匹配
            pcls = pyolos[:, :, 1:s_][gconf.type(torch.bool)]
            gcls = gyolos[:, :, 1:s_][gconf.type(torch.bool)]
            _loss_val = F.binary_cross_entropy_with_logits(pcls, gcls, reduction="none")
            loss_cls = _loss_val.sum(-1).mean() * cfg.LOSS_WEIGHT[2]

            _dim = 1 + cfg.NUM_CLASSES
        else:
            raise Exception("cfg.loss_args['s_cls'] 类型错误 %s" % cfg.loss_args['s_cls'])

        # pbox = pyolos[:, :, s_:]
        ptxty = pyolos[:, :, s_:s_ + 2]
        ptwth = pyolos[:, :, s_ + 2:]  # 这里不需要归一

        # gbox = gyolos[:, :, _dim:_dim + 4]
        gtxty = gyolos[:, :, _dim:_dim + 2]  # torch.Size([5, 169, 2])
        gtwth = gyolos[:, :, _dim + 2:_dim + 4]
        weight = gyolos[:, :, -1]  # torch.Size([5, 169])

        log_dict = {}

        # ----------------conf损失----------------
        pconf = pyolos[:, :, 0].sigmoid()  # -0.6~0.7
        if cfg.loss_args['s_conf'] == 'foc':
            # ghmc_loss = GHMC_Loss()  # 训不了
            # loss_conf = ghmc_loss(pconf, gconf).sum(-1).mean()
            l_pos, l_neg = focalloss_v2(pconf, gconf, alpha=cfg.arg_focalloss_alpha, is_merge=False)
            # l_pos, l_neg = focalloss_v3(pconf, gconf, alpha=0.25, is_merge=False)
            loss_conf_pos = l_pos.sum(-1).mean()
            loss_conf_neg = l_neg.sum(-1).mean()
        elif cfg.loss_args['s_conf'] == 'mse':
            _loss_val = F.mse_loss(pconf, gconf, reduction="none")  # 这个强
            # _loss_val = F.binary_cross_entropy_with_logits(pconf, gconf, reduction="none")
            loss_conf_pos = (_loss_val * gconf).sum(-1).mean() * cfg.LOSS_WEIGHT[0]
            loss_conf_neg = (_loss_val * torch.logical_not(gconf)).sum(-1).mean() * cfg.LOSS_WEIGHT[1]
            # loss_conf = loss_conf_pos + loss_conf_neg
        else:
            raise Exception('类型错误')

        # ----------------box损失-----------------
        if cfg.loss_args['s_match'] == 'whoned':
            # _loss_val = F.binary_cross_entropy_with_logits(ptxty, gtxty, reduction="none")
            # loss_txty = (_loss_val.sum(-1) * gconf * weight).sum(-1).mean()
            # _loss_val = F.binary_cross_entropy_with_logits(ptwth, gtwth, reduction="none")
            # loss_twth = (_loss_val.sum(-1) * gconf * weight).sum(-1).mean()
            pass
        elif cfg.loss_args['s_match'] == 'log' or cfg.loss_args['s_match'] == 'log_g':
            _loss_val = F.binary_cross_entropy_with_logits(ptxty, gtxty, reduction="none")
            # _loss_val = F.mse_loss(ptxty, gtxty, reduction="none")
            loss_txty = (_loss_val.sum(-1) * gconf * weight).sum(-1).mean()
            _loss_val = F.mse_loss(ptwth, gtwth, reduction="none")
            loss_twth = (_loss_val.sum(-1) * gconf * weight).sum(-1).mean()
        else:
            raise Exception('类型错误')

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


class PredictYolo_v1(nn.Module):
    def __init__(self, num_bbox, num_classes, num_grid, threshold_conf=0.5, threshold_nms=0.3, cfg=None):
        super(PredictYolo_v1, self).__init__()
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

        if cfg.loss_args['s_cls'] == 'ce':
            # b,hw,c -> b,hw,3 -> b,hw -> b,hw
            cls_conf, plabels = pyolos[:, :, 1:1 + cfg.NUM_CLASSES].softmax(-1).max(-1)
        elif cfg.loss_args['s_cls'] == 'bce':  # 二值交叉熵匹配
            cls_conf, plabels = pyolos[:, :, 1:1 + cfg.NUM_CLASSES].sigmoid().max(-1)
        else:
            raise Exception("cfg.loss_args['s_cls'] 类型错误 %s" % cfg.loss_args['s_cls'])

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

        ''' 预测 这里是修复是 xywh'''
        boxes_decode4yolo1(ptxywh, h, w, cfg)

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


class YOLOv1_Net(nn.Module):
    def __init__(self, backbone, cfg):
        super(YOLOv1_Net, self).__init__()
        self.backbone = backbone

        dim_layer = backbone.dim_out
        self.spp = nn.Sequential(
            CBL(dim_layer, 256, ksize=1),
            SPPv2(),
            BottleneckCSP(256 * 4, dim_layer, n=1, shortcut=False)
        )
        self.sam = SAM(dim_layer)
        self.conv_set = BottleneckCSP(dim_layer, dim_layer, n=3, shortcut=False)

        self.head_conf_cls = nn.Conv2d(dim_layer, 1 + cfg.NUM_CLASSES, 1)
        self.head_box = nn.Conv2d(dim_layer, 4, 1)

        # 初始化分类 bias
        # num_pos = 1.5
        # num_tolal = cfg.NUM_GRID ** 2
        # finit_conf_bias(self.head_conf_cls, num_tolal, num_pos, cfg.NUM_CLASSES)

    def forward(self, x, targets=None):
        outs = self.backbone(x)
        outs = self.spp(outs)
        outs = self.sam(outs)
        outs = self.conv_set(outs)
        out_conf_cls = self.head_conf_cls(outs)  # torch.Size([5, 1+3, 13, 13])
        out_box = self.head_box(outs)  # torch.Size([5, 4, 13, 13])
        outs = torch.cat([out_conf_cls, out_box], dim=1)
        return outs


class Yolo_v1_1(nn.Module):
    def __init__(self, backbone, cfg):
        super(Yolo_v1_1, self).__init__()
        self.net = YOLOv1_Net(backbone, cfg)

        self.losser = LossYOLO_v1(cfg=cfg)
        self.preder = PredictYolo_v1(num_bbox=cfg.NUM_BBOX,
                                     num_classes=cfg.NUM_CLASSES,
                                     num_grid=cfg.NUM_GRID,
                                     threshold_conf=cfg.THRESHOLD_PREDICT_CONF,
                                     threshold_nms=cfg.THRESHOLD_PREDICT_NMS,
                                     cfg=cfg
                                     )

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
    from torchvision import models
    from f_pytorch.tools_model.f_layer_get import ModelOut4Resnet18

    model = models.resnet18(pretrained=True)
    model = ModelOut4Resnet18(model)


    class CFG:
        pass


    cfg = CFG()
    cfg.NUM_CLASSES = 3
    cfg.THRESHOLD_PREDICT_CONF = 0.3
    cfg.THRESHOLD_PREDICT_NMS = 0.3
    cfg.NUM_BBOX = 2
    cfg.NUM_GRID = 7
    net = YOLOv1_Net(backbone=model, cfg=cfg)
    # x = torch.rand([5, 3, 416, 416])
    # print(net(x).shape)

    from f_pytorch.tools_model.model_look import f_look_tw

    f_look_tw(net, input=(5, 3, 416, 416), name='YOLOv1_Net')
