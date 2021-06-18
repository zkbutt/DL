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
from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2xywh
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

        # num_class + txywh + weight  conf通过高斯生成 热力图层数表示类别索引
        if cfg.NUM_KEYPOINTS > 0:
            gdim = cfg.NUM_CLASSES + cfg.NUM_KEYPOINTS * 2 + 4 + 1
        else:
            gdim = cfg.NUM_CLASSES + 4 + 1
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
        mask_pos_3d = gcls == 1  # torch.Size([3, 16384, 3])
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
        # num_class + txywh + weight
        # ptxy_sigmoid_pos = ptxy[mask_pos_2d].sigmoid()  # 这个需要归一化
        # ptwh_pos = ptwh[mask_pos_2d]
        #
        # gres_pos = gres[mask_pos_2d]
        # weight_pos = gres_pos[:, cfg.NUM_CLASSES + 4]  # torch.Size([32, 845])
        # gtxy_pos = gres_pos[:, cfg.NUM_CLASSES:cfg.NUM_CLASSES + 2]
        # gtwh_pos = gres_pos[:, cfg.NUM_CLASSES + 2:cfg.NUM_CLASSES + 4]
        #
        # _loss_val = x_bce(ptxy_sigmoid_pos, gtxy_pos, reduction="none")
        # l_txty = (_loss_val * weight_pos.unsqueeze(-1)).sum(-1).mean()
        # _loss_val = F.smooth_l1_loss(ptwh_pos, gtwh_pos, reduction="none")
        # l_twth = (_loss_val * weight_pos.unsqueeze(-1)).sum(-1).mean()

        # 这个判断 正例
        weight = gres[:, :, cfg.NUM_CLASSES + 4]  # torch.Size([32, 845])

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

        log_dict = {}
        log_dict['l_total'] = l_total.item()
        log_dict['l_cls_pos'] = l_cls_pos.item()
        log_dict['l_cls_neg'] = l_cls_neg.item()
        log_dict['l_xy'] = l_txty.item()
        log_dict['l_wh'] = l_twth.item()

        return l_total, log_dict


# class FPredict(Predicting_Base):
#     def __init__(self, cfg):
#         super(FPredict, self).__init__(cfg)
#         self.cfg = cfg
#
#     def p_init(self, outs):
#         pcls, ptxy, ptwh = outs
#         device = pcls.device
#         return outs, device
#
#     def get_pscores(self, outs):
#         pcls, ptxy, ptwh = outs
#         pcls_sigmoid_4d = pcls.sigmoid()
#         # torch.Size([32, 3, 128, 128])
#         batch, c, h, w = pcls.shape
#         _pscores = F.max_pool2d(pcls_sigmoid_4d, kernel_size=5, padding=2, stride=1)
#         mask_4d = _pscores == pcls_sigmoid_4d
#         _pscores *= mask_4d
#
#         # [3, 3, 128,128] -> [3, 128,128, 3] -> [3, -1, 3]
#         _pscores = _pscores.permute(0, 2, 3, 1).contiguous().view(batch, -1, self.cfg.NUM_CLASSES)
#         # [3, -1, 3] -> [3, -1]
#         pscores, plabels = _pscores.max(-1)  # 一个格子取一个最大的分数即可 torch.Size([3, 16384])
#         return pscores, plabels, pscores
#
#     def get_stage_res(self, outs, mask_pos, pscores, plabels):
#         '''
#
#         :param outs:
#         :param mask_pos: torch.Size([3, 49152])
#         :param pscores:
#         :param plabels: 这里传入为空
#         :return:
#         '''
#         # ptxywh  xy是特图偏移  wh是特图真实长宽(非缩放比例)
#         pcls, ptxy, ptwh = outs
#         batch, c, h, w = pcls.shape
#         # [3, 49152] -> [3, 3,-1] -> [3, -1,3]  获取mask
#         # mask_pos_3d = mask_pos.view(batch, c, -1).permute(0, 2, 1)
#         # plabels1 = torch.where(mask_pos_3d)[1]
#         # mask_pos_2d = torch.any(mask_pos_3d, -1)
#
#         device = ptxy.device
#         fsize_wh = torch.tensor([h, w], device=device)
#
#         ptxy = ptxy.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
#         ptwh = ptwh.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
#
#         ptxywh = torch.cat([ptxy, ptwh], dim=-1)
#         pboxes_ltrb1 = boxes_decode4center(self.cfg, fsize_wh, ptxywh)
#         pboxes_ltrb1.clamp_(0, 1)
#         pboxes_ltrb1 = pboxes_ltrb1[mask_pos]
#
#         ids_batch1, _ = torch.where(mask_pos)
#
#         pscores1 = pscores[mask_pos]
#         plabels1 = plabels[mask_pos]
#         plabels1 = plabels1 + 1
#         return ids_batch1, pboxes_ltrb1, plabels1, pscores1


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


class FPredict2(nn.Module):
    def __init__(self, cfg):
        super(FPredict2, self).__init__()
        self.cfg = cfg
        self.stride = 4
        self.device = torch.device('cpu')
        input_size = [512, 512]
        self.grid_cell = self.create_grid(input_size)
        self.set_grid(input_size)
        self.topk = 100
        self.nms_thresh = 0.45

    def create_grid(self, input_size):
        h, w = input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 2)

        return grid_xy

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)  # the size of bbox
        order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

        keep = []  # store the final bounding boxes
        while order.size > 0:
            i = order[0]  # the index of the bbox with highest confidence
            keep.append(i)  # save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def decode_boxes(self, pred):
        """
        input box :  [delta_x, delta_y, sqrt(w), sqrt(h)]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        self.device = pred.device
        pred[:, :, :2] = (torch.sigmoid(pred[:, :, :2]) + self.grid_cell.to(self.device)) * self.stride
        pred[:, :, 2:] = (torch.exp(pred[:, :, 2:])) * self.stride

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2

        return output

    def set_grid(self, input_size):
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[0], input_size[1], input_size[0], input_size[1]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def _topk(self, scores):
        B, C, H, W = scores.size()
        # torch.Size([1, 3, 16384])  每个类找100个
        topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), self.topk)

        topk_inds = topk_inds % (H * W)  # 基本是一样的
        # 在选出的这300个中 选出最大的 100个  torch.Size([3, 100])
        topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), self.topk)

        topk_clses = torch.true_divide(topk_ind, self.topk).int()
        topk_inds = self._gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, self.topk)

        return topk_score, topk_inds, topk_clses

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def forward(self, outs, imgs_ts=None, *args):
        ids, sizes, coco_gt = args
        cls_preds, txty_preds, twth_preds = outs
        B = cls_preds.size(0)
        data_dict = []

        for i in range(B):
            cls_pred, txty_pred, twth_pred = cls_preds[i:i + 1], txty_preds[i:i + 1], twth_preds[i:i + 1]

            # batch_size = 1
            cls_pred = torch.sigmoid(cls_pred)
            # simple nms
            hmax = F.max_pool2d(cls_pred, kernel_size=5, padding=2, stride=1)
            keep = (hmax == cls_pred).float()
            cls_pred *= keep

            # decode box torch.Size([3, 16384, 4])
            txtytwth_pred = torch.cat([txty_pred, twth_pred], dim=1).permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
            # [B, H*W, 4] -> [H*W, 4]
            bbox_pred = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch.to(self.device))[0], 0., 1.)

            # topk  torch.Size([3, 3, 128, 128])
            scores, topk_inds, topk_clses = self._topk(cls_pred)

            scores = scores[0].cpu().numpy()
            cls_inds = topk_clses[0].cpu().numpy()
            bboxes = bbox_pred[topk_inds[0]].cpu().numpy()

            if True:
                # nms
                keep = np.zeros(len(bboxes), dtype=np.int)
                for j in range(self.cfg.NUM_CLASSES):
                    inds = np.where(cls_inds == j)[0]
                    if len(inds) == 0:
                        continue
                    c_bboxes = bboxes[inds]
                    c_scores = scores[inds]
                    c_keep = self.nms(c_bboxes, c_scores)
                    keep[inds[c_keep]] = 1

                keep = np.where(keep > 0)
                bboxes = bboxes[keep]
                scores = scores[keep]
                cls_inds = cls_inds[keep]

            id_ = ids[i]  # 外层
            scale = (sizes[i].repeat(2)).cpu().numpy()
            bboxes *= scale
            labels = cls_inds + 1

            for k, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                # label = self.dataset.class_ids[int(cls_inds[i])]

                # ltrb -> ltwh
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[k])  # object score * class score
                # 转换成int32 不能json序列化
                A = {"image_id": float(id_), "category_id": float(labels[k]), "bbox": bbox,
                     "score": score}  # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            # cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = coco_gt.loadRes(tmp)
            cocoEval = FCOCOeval(coco_gt, cocoDt, annType[1])
            # cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            coco_stats, print_coco = cocoEval.summarize()
            print(print_coco)
            return 0, 0
        else:
            return 0, 0


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
        if cfg.CUSTOM_EVEL:
            self.preder = FPredict2(cfg)  # 配合 cfg.CUSTOM_EVEL
        else:
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
