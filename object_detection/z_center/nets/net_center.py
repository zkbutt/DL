import json
import math
import tempfile

import torch
import torch.nn as nn
import numpy as np

from f_pytorch.tools_model.f_layer_get import ModelOut4Mobilenet_v2, ModelOut4Resnet18
from f_pytorch.tools_model.f_model_api import finit_weights, FConv2d
from f_pytorch.tools_model.fmodels.model_modules import SPPv2, DeConv
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_match import match4center, boxes_decode4center
from f_tools.fits.f_predictfun import label_nms4keypoints
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.fits.fitting.fcocoeval import FCOCOeval
from f_tools.floss.f_lossfun import x_bce
from f_tools.floss.focal_loss import focalloss_center
from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2xywh, bbox_iou4one
import torch.nn.functional as F


class FLoss(nn.Module):
    def __init__(self, cfg):
        super(FLoss, self).__init__()
        self.cfg = cfg
        # self.fun_ghmc = GHMC_Loss(num_bins=10, momentum=0.25, reduction='sum')
        # self.fun_ghmc = GHMC_Loss(num_bins=10, momentum=0.25, reduction='mean')
        # self.fun_focalloss_v2 = FocalLoss_v2(reduction='sum')

    def forward(self, p_center, targets, imgs_ts=None):
        '''

        :param p_center:
          :param targets: list
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['size'] = target['size']
            target['image_id'] = int
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        pcls, ptxy, ptwh = p_center
        device = pcls.device
        batch, c, h, w = pcls.shape

        # b,c,h,w -> b,h,w,c -> b,h*w,c
        pcls = pcls.permute(0, 2, 3, 1).contiguous().view(batch, -1, self.cfg.NUM_CLASSES)
        ptxy = ptxy.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
        ptwh = ptwh.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)

        fsize_wh = torch.tensor([h, w], device=device)

        # num_class + txywh + weight + gt4  conf通过高斯生成 热力图层数表示类别索引
        if cfg.NUM_KEYPOINTS > 0:
            gdim = cfg.NUM_CLASSES + cfg.NUM_KEYPOINTS * 2 + 4 + 1 + 4
        else:
            gdim = cfg.NUM_CLASSES + 4 + 1 + 4
        gres = torch.empty((batch, h, w, gdim), device=device)

        # 匹配GT
        for i, target in enumerate(targets):
            # batch 遍历每一张图
            gboxes_ltrb_b = targets[i]['boxes']
            glabels_b = targets[i]['labels']
            # 处理这张图的所有标签
            gres[i] = match4center(gboxes_ltrb_b=gboxes_ltrb_b,
                                   glabels_b=glabels_b,
                                   fsize_wh=fsize_wh,
                                   dim=gdim,
                                   cfg=cfg,
                                   )

            if cfg.IS_VISUAL:
                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                _img_ts = f_recover_normalization4ts(imgs_ts[i].clone())
                from torchvision.transforms import functional as transformsF
                img_pil = transformsF.to_pil_image(_img_ts).convert('RGB')
                import numpy as np
                # img_np = np.array(img_pil)

                '''plt画图部分'''
                from matplotlib import pyplot as plt
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
                plt.rcParams['axes.unicode_minus'] = False

                # 这里的热力图肯定的偏差  [128,128]
                data_hot = torch.zeros_like(gres[i, :, :, 0])  # 只需要一层即可
                for label in glabels_b.unique():
                    # print(ids2classes[str(int(label))])
                    # 类别合并输出
                    flog.debug(' %s', gres[i, :, :, 3:7][gres[i, :, :, (label - 1).long()] == 1])
                    torch.max(data_hot, gres[i, :, :, (label - 1).long()], out=data_hot)  # 这里是类别合并
                plt.imshow(data_hot.cpu())
                plt.imshow(img_pil.resize(fsize_wh), alpha=0.7)
                plt.colorbar()

                # x,y表示横纵坐标，color表示颜色：'r':红  'b'：蓝色 等，marker:标记，edgecolors:标记边框色'r'、'g'等，s：size大小
                boxes_xywh_cpu = ltrb2xywh(gboxes_ltrb_b).cpu()

                fsize_cpu = fsize_wh.cpu()
                xys_f = boxes_xywh_cpu[:, :2] * fsize_cpu
                plt.scatter(xys_f[:, 0], xys_f[:, 1], color='r', s=5)  # 红色

                boxes_ltrb_cpu = gboxes_ltrb_b.cpu()
                boxes_ltrb_f = boxes_ltrb_cpu * fsize_cpu.repeat(2)
                current_axis = plt.gca()
                for i, box_ltrb_f in enumerate(boxes_ltrb_f):
                    l, t, r, b = box_ltrb_f
                    # ltwh
                    current_axis.add_patch(plt.Rectangle((l, t), r - l, b - t, color='green', fill=False, linewidth=2))
                    # current_axis.text(l, t - 2, ids2classes[int(glabels[i])], size=8, color='white',
                    #                   bbox={'facecolor': 'green', 'alpha': 0.6})
                plt.show()

        gres = gres.reshape(batch, -1, gdim)

        ''' ---------------- cls损失 只计算正例---------------- '''
        gcls = gres[:, :, :cfg.NUM_CLASSES]
        # mask_pos_3d = gcls > 0  # torch.Size([3, 16384, 3])
        # mask_neg_3d = gcls == 0
        mask_pos_3d = gcls == 1  # 只有中心点为1正例 torch.Size([3, 16384, 3])
        mask_neg_3d = gcls != 1
        nums_pos = torch.sum(torch.sum(mask_pos_3d, dim=-1), dim=-1)
        # mask_pos_2d = torch.any(mask_pos_3d, -1)

        # focloss
        pcls_sigmoid = pcls.sigmoid()
        l_cls_pos, l_cls_neg = focalloss_center(pcls_sigmoid, gcls)
        l_cls_pos = torch.mean(torch.sum(torch.sum(l_cls_pos, -1), -1) / nums_pos)
        l_cls_neg = torch.mean(torch.sum(torch.sum(l_cls_neg, -1), -1) / nums_pos)

        # l_cls_neg = l_cls_neg.sum(-1).sum(-1).mean() # 等价

        ''' ---------------- box损失 ----------------- '''
        log_dict = {}
        # num_class + txywh + weight + gt4
        if cfg.MODE_TRAIN == 2:  # iou
            ptxywh = torch.cat([ptxy, ptwh], dim=-1)
            pboxes_ltrb = boxes_decode4center(self.cfg, fsize_wh, ptxywh)
            mask_pos_2d = torch.any(mask_pos_3d, -1)  # torch.Size([16, 16384])
            # torch.Size([16, 16384, 4])  -> torch.Size([19, 4])
            p_ltrb_pos = pboxes_ltrb[mask_pos_2d]
            g_ltrb_pos = gres[..., cfg.NUM_CLASSES + 4 + 1:cfg.NUM_CLASSES + 4 + 1 + 4][mask_pos_2d]
            iou = bbox_iou4one(p_ltrb_pos, g_ltrb_pos, is_giou=True)
            l_reg = 5 * torch.mean(1 - iou)

            l_total = l_cls_pos + l_cls_neg + l_reg

            log_dict['l_total'] = l_total.item()
            log_dict['l_cls_pos'] = l_cls_pos.item()
            log_dict['l_cls_neg'] = l_cls_neg.item()
            log_dict['l_reg'] = l_reg.item()

        elif cfg.MODE_TRAIN == 1:
            weight = gres[:, :, cfg.NUM_CLASSES + 4]  # 这个可以判断正例 torch.Size([32, 845])
            gtxy = gres[:, :, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 2]
            gtwh = gres[:, :, cfg.NUM_CLASSES + 2:cfg.NUM_CLASSES + 4]
            ptxy_sigmoid = ptxy.sigmoid()  # 这个需要归一化
            _loss_val = x_bce(ptxy_sigmoid, gtxy, reduction="none")
            # _loss_val = F.binary_cross_entropy_with_logits(ptxy, gtxy, reduction="none")
            # _loss_val[mask_pos_2d].sum() 与这个等价
            l_txty = torch.mean(torch.sum(torch.sum(_loss_val * weight.unsqueeze(-1), -1), -1) / nums_pos)
            _loss_val = F.smooth_l1_loss(ptwh, gtwh, reduction="none")
            l_twth = torch.mean(torch.sum(torch.sum(_loss_val * weight.unsqueeze(-1), -1), -1) / nums_pos)

            l_total = l_cls_pos + l_cls_neg + l_txty + l_twth

            log_dict['l_total'] = l_total.item()
            log_dict['l_cls_pos'] = l_cls_pos.item()
            log_dict['l_cls_neg'] = l_cls_neg.item()
            log_dict['l_xy'] = l_txty.item()
            log_dict['l_wh'] = l_twth.item()
        else:
            raise Exception('cfg.MODE_TRAIN = %s 不存在' % cfg.MODE_TRAIN)

        return l_total, log_dict


class FPredict(Predicting_Base):
    def __init__(self, cfg):
        super(FPredict, self).__init__(cfg)

    def p_init(self, outs):
        pcls, ptxy, ptwh = outs
        device = pcls.device
        return outs, device

    def get_pscores(self, outs):
        pcls, ptxy, ptwh = outs
        pcls_sigmoid_4d = pcls.sigmoid()
        # torch.Size([32, 3, 128, 128])
        batch, c, h, w = pcls.shape
        _pscores = F.max_pool2d(pcls_sigmoid_4d, kernel_size=5, padding=2, stride=1)
        mask_4d = _pscores == pcls_sigmoid_4d
        _pscores *= mask_4d

        # [3, 3, 128,128] -> [3, 128,128, 3] -> [3, -1, 3]
        _pscores = _pscores.permute(0, 2, 3, 1).contiguous().view(batch, -1, self.cfg.NUM_CLASSES)
        # [3, -1, 3] -> [3, -1]
        pscores, plabels = _pscores.max(-1)  # 一个格子取一个最大的分数即可 torch.Size([3, 16384])
        return pscores, plabels, pscores

    def get_stage_res(self, outs, mask_pos, pscores, plabels):
        '''

        :param outs:
        :param mask_pos: torch.Size([3, 49152])
        :param pscores:
        :param plabels: 这里传入为空
        :return:
        '''
        # ptxywh  xy是特图偏移  wh是特图真实长宽(非缩放比例)
        pcls, ptxy, ptwh = outs
        batch, c, h, w = pcls.shape
        # [3, 49152] -> [3, 3,-1] -> [3, -1,3]  获取mask
        # mask_pos_3d = mask_pos.view(batch, c, -1).permute(0, 2, 1)
        # plabels1 = torch.where(mask_pos_3d)[1]
        # mask_pos_2d = torch.any(mask_pos_3d, -1)

        device = ptxy.device
        fsize_wh = torch.tensor([h, w], device=device)

        ptxy = ptxy.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
        ptwh = ptwh.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)

        ptxywh = torch.cat([ptxy, ptwh], dim=-1)
        pboxes_ltrb1 = boxes_decode4center(self.cfg, fsize_wh, ptxywh)
        pboxes_ltrb1.clamp_(0, 1)
        pboxes_ltrb1 = pboxes_ltrb1[mask_pos]

        ids_batch1, _ = torch.where(mask_pos)

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class CenterNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(CenterNet, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        in_channel = backbone.dim_out

        # neck
        self.spp = nn.Sequential(
            SPPv2(),
            FConv2d(in_channel * 4, in_channel // 2, 1),
            FConv2d(in_channel // 2, in_channel, 3, p=1),
        )

        # head
        self.deconv5 = DeConv(in_channel, in_channel // 2, ksize=4, stride=2)  # 32 -> 16
        self.deconv4 = DeConv(in_channel // 2, in_channel // 2, ksize=4, stride=2)  # 16 -> 8
        self.deconv3 = DeConv(in_channel // 2, in_channel // 2, ksize=4, stride=2)  # 8 -> 4

        # 输出维度压缩4倍输出
        self.cls_pred = nn.Sequential(
            FConv2d(in_channel // 2, in_channel // 4, k=3, p=1),
            nn.Conv2d(in_channel // 4, cfg.NUM_CLASSES, kernel_size=1)
        )

        self.txty_pred = nn.Sequential(
            FConv2d(in_channel // 2, in_channel // 4, k=3, p=1),
            nn.Conv2d(in_channel // 4, 2, kernel_size=1)
        )

        self.twth_pred = nn.Sequential(
            FConv2d(in_channel // 2, in_channel // 4, k=3, p=1),
            nn.Conv2d(in_channel // 4, 2, kernel_size=1)
        )

    def forward(self, x, targets=None):
        # torch.Size([3, 512, 16, 16])
        c5 = self.backbone(x)
        batch = c5.shape[0]

        p5 = self.spp(c5)  # 维度放大混合后,恢复尺寸不变
        p4 = self.deconv5(p5)  # out torch.Size([3, 256, 32, 32])
        p3 = self.deconv4(p4)  # out 同维 64
        p2 = self.deconv3(p3)  # out 同维 128  torch.Size([3, 256, 128, 128])

        # head
        pcls = self.cls_pred(p2)  # torch.Size([b, num_class, 128, 128])
        ptxy = self.txty_pred(p2)  # torch.Size([3, 2, 128, 128])
        ptwh = self.twth_pred(p2)  # torch.Size([3, 2, 128, 128])

        return pcls, ptxy, ptwh


class Center(nn.Module):
    def __init__(self, backbone, cfg):
        '''
        层属性可以是 nn.Module nn.ModuleList(封装Sequential) nn.Sequential
        '''
        super(Center, self).__init__()
        self.net = CenterNet(backbone, cfg)

        self.losser = FLoss(cfg)
        self.preder = FPredict(cfg)

    def forward(self, x, targets=None, *args):
        outs = self.net(x)

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            '''------验证loss 待扩展------'''

            return loss_total, log_dict
        else:
            with torch.no_grad():  # 这个没用
                if self.cfg.CUSTOM_EVEL:
                    # 可以不要返回值
                    self.preder(outs, x, *args)
                    return
                else:
                    ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x, *args)
                    return ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores
        # if torch.jit.is_scripting():  # 这里是生产环境部署


if __name__ == '__main__':
    from torchvision import models
    from f_pytorch.tools_model.model_look import f_look_tw


    # model = models.mobilenet_v2(pretrained=True)
    # model = ModelOuts4Mobilenet_v2(model)
    # dims_out = [256, 512, 1024]
    class cfg:
        pass


    cfg.NUM_CLASSES = 3

    model = models.resnet18(pretrained=True)
    model = ModelOut4Resnet18(model)

    model = CenterNet(backbone=model, cfg=cfg, )
    model.eval()
    # 输出是 tuple 要报错但可以生成
    f_look_tw(model, input=(1, 3, 512, 512), name='CenterNet')
