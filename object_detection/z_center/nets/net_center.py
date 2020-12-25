import torch
import torch.nn as nn

from f_pytorch.tools_model.f_layer_get import ModelOuts4Mobilenet_v2, ModelOut4Mobilenet_v2
from f_tools.GLOBAL_LOG import flog
from f_tools.f_predictfun import label_nms
from f_tools.fits.f_lossfun import FocalLoss_v2, focal_loss4center
from f_tools.fun_od.f_boxes import xywh2ltrb
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import show_anc4pil
import torch.nn.functional as F


class LoosCenter(nn.Module):
    def __init__(self, cfg):
        super(LoosCenter, self).__init__()
        self.cfg = cfg
        # self.focal_loss = FocalLoss_v2(alpha=cfg.FOCALLOSS_ALPHA, gamma=cfg.FOCALLOSS_GAMMA, reduction='none')

    def forward(self, p_center, g_center, imgs_ts=None):
        '''

        :param p_center:
            photmap: torch.Size([5, 20, 128, 128])  0~1
            pwh : torch.Size([5, 2, 128, 128])  归一化尺寸
            pxy_offset : torch.Size([5, 2, 128, 128]) 相对网格 同yolo
        :param g_center: torch.Size([5, 128, 128, 24])
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        pheatmap, pwh, pxy_offset, pkeypoint_offset = p_center
        pheatmap = pheatmap.permute(0, 2, 3, 1)

        gheatmap = g_center[:, :, :, :cfg.NUM_CLASSES]
        mask_pos = gheatmap.eq(1)
        mask_pos = torch.any(mask_pos, dim=-1)  # 降维运算
        g_center_pos = g_center[mask_pos]
        gxy_offset = g_center_pos[:, -4:-2]
        gwh = g_center_pos[:, -2:]
        num_pos = mask_pos.sum()
        num_pos.clamp_(min=torch.finfo(torch.float).eps)

        pxy_offset_pos = pxy_offset.permute(0, 2, 3, 1)[mask_pos]
        pwh_pos = pwh.permute(0, 2, 3, 1)[mask_pos]

        if pkeypoint_offset is not None:
            gkeypoint_offset = g_center_pos[:, :, :, cfg.NUM_CLASSES:-4]
            pkeypoint_offset = pkeypoint_offset.permute(0, 2, 3, 1)[mask_pos]
            loss_keypoint = F.l1_loss(pkeypoint_offset, gkeypoint_offset, reduction='sum') / num_pos \
                            * cfg.LOSS_WEIGHT[3]
        else:
            loss_keypoint = 0

        loss_conf = focal_loss4center(pheatmap, gheatmap, reduction='sum') / num_pos * cfg.LOSS_WEIGHT[0]
        # loss_xy = focal_loss4center(pxy_offset, gxy_offset, reduction='sum') / num_pos
        # loss_wh = focal_loss4center(pwh, gwh, reduction='sum') / num_pos
        loss_xy = F.l1_loss(pxy_offset_pos, gxy_offset, reduction='sum') / num_pos * cfg.LOSS_WEIGHT[1]
        loss_wh = F.l1_loss(pwh_pos, gwh, reduction='sum') / num_pos * cfg.LOSS_WEIGHT[2]

        loss_total = loss_conf + loss_xy + loss_wh + loss_keypoint
        log_dict = {}
        log_dict['loss_total'] = loss_total.item()
        log_dict['l_conf'] = loss_conf.item()
        log_dict['l_xy'] = loss_xy.item()
        log_dict['l_wh'] = loss_wh.item()
        if pkeypoint_offset is not None:
            log_dict['l_kp'] = loss_keypoint.item()
        return loss_total, log_dict


class PredictCenter(nn.Module):
    def __init__(self, cfg, threshold_conf=0.18, threshold_nms=0.3, topk=100):
        super(PredictCenter, self).__init__()
        self.cfg = cfg
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf
        self.topk = topk

    def pool_nms(self, phm, kernel=3):
        '''
        最大池化剔最大值 some 池化
        :param phm:
        :param kernel:
        :return:
        '''
        stride = 1
        pad = (kernel - stride) // 2
        hmax = nn.functional.max_pool2d(phm, (kernel, kernel), stride=stride, padding=pad)
        keep = (hmax == phm).float()  # 同维清零
        return phm * keep

    def forward(self, p_center, imgs_ts=None):
        '''

        :param p_center:
            phm : 归一化 conf b,c,h,w 这个是conf 和 label
            pwh :
            pxy_offset
        :param imgs_ts:
        :return:
        '''
        pheatmap, pwh, pxy_offset, pkeypoint_offset = p_center
        pheatmap_nms = self.pool_nms(pheatmap)  # 这个包含conf 和label
        # b,c,h,w -> b,h,w,c
        pheatmap_nms = pheatmap_nms.permute(0, 2, 3, 1)

        # torch.Size([5, 128, 128, 20]) -> [5, 128, 128]
        mask = pheatmap_nms > self.threshold_conf
        mask = torch.any(mask, dim=-1)
        if not torch.any(mask):  # 如果没有一个对象
            # flog.error('该批次没有找到目标')
            return [None] * 4

        device = pheatmap.device
        # batch, h, w, c = pheatmap_nms.shape
        pxy_offset = pxy_offset.permute(0, 2, 3, 1)
        pwh = pwh.permute(0, 2, 3, 1)

        '''第一阶段'''
        ids_batch1, ids_row, ids_col = torch.where(mask)
        # mask 降变二维 batch,h,w,c -> nn,20
        phm_nms1 = pheatmap_nms[mask]  # 这个是conf 和 label
        pscores1, plabels1 = phm_nms1.max(dim=-1)
        pwh1 = pwh[mask]

        '''修复获得 p_boxes_ltrb1'''
        fsize = torch.tensor(pheatmap_nms.shape[1:3], device=device)
        pxy_offset1 = pxy_offset[mask]  # torch.Size([6610, 2])
        # 后面增加一维 [nn]-> [nn,1] ->  [nn,2] 归一化运算
        colrows_index = torch.cat([ids_col.unsqueeze(-1), ids_row.unsqueeze(-1)], dim=-1).type(torch.float)
        pxy = (pxy_offset1 + colrows_index).true_divide(fsize)

        pboxes_xywh1 = torch.cat([pxy, pwh1], dim=-1)
        pboxes_ltrb1 = xywh2ltrb(pboxes_xywh1)

        if self.cfg.IS_VISUAL:
            # 可视化1 显示框框 原目标图 --- 初始化图片
            flog.debug('conf后 %s 个', pboxes_ltrb1.shape[0])
            img_ts = imgs_ts[0]
            from torchvision.transforms import functional as transformsF
            img_ts = f_recover_normalization4ts(img_ts)
            img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
            show_anc4pil(img_pil, pboxes_ltrb1, size=img_pil.size)

        pscores1_argsort = torch.argsort(pscores1, descending=True)
        pscores1_topk = pscores1_argsort[:self.topk]

        ids_batch2 = ids_batch1[pscores1_topk]
        plabels2 = pscores1[pscores1_topk]
        pboxes_ltrb2 = pboxes_ltrb1[pscores1_topk]
        pscores2 = pscores1[pscores1_topk]

        return ids_batch2, pboxes_ltrb2, plabels2, pscores2


class CenterHead(nn.Module):
    def __init__(self, num_classes, in_channels=64, out_channels=64, bn_momentum=0.1, num_keypoints=0):
        '''

        :param num_classes:
        :param out_channels:
        :param bn_momentum:
        '''
        super(CenterHead, self).__init__()
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding=0))
        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding=0))
        if num_keypoints > 0:
            self.keypoint_head = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, num_keypoints * 2, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        phm = self.cls_head(x).sigmoid()
        pwh = self.wh_head(x).sigmoid()
        pxy_offset = self.reg_head(x).sigmoid()
        # pwh = self.wh_head(x)
        # pxy_offset = self.reg_head(x)
        if hasattr(self, 'keypoint_head'):
            pkeypoint_offset = self.keypoint_head(x).sigmoid()
        else:
            pkeypoint_offset = None
        return phm, pwh, pxy_offset, pkeypoint_offset


class CenterUpsample3Conv(nn.Module):
    def __init__(self, dim_in_backbone, bn_momentum=0.1):
        super(CenterUpsample3Conv, self).__init__()
        self.bn_momentum = bn_momentum
        # [256, 512, 1024]
        # self.dim_in_backbone = dim_in_backbone
        self.deconv_layers = self._make_deconv_layer(dim_in_backbone)

    def _make_deconv_layer(self, dim_in_backbone, nums_dim_out=[256, 128, 64], nums_kernel=[4, 4, 4]):
        '''
        反卷积
        :param nums_dim_out:
        :param nums_kernel: 对应核数
        :return:
        '''
        layers = []
        # 16,16,2048 -> 32,32,256
        # 32,32,256 -> 64,64,128
        # 64,64,128 -> 128,128,64

        for dim_out, kernel in zip(nums_dim_out, nums_kernel):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in_backbone,
                    out_channels=dim_out,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(dim_out, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            dim_in_backbone = dim_out
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class CenterNet(nn.Module):
    def __init__(self, cfg, backbone, num_classes, dim_in_backbone):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        # [256, 512, 1024]
        self.upsample3conv = CenterUpsample3Conv(dim_in_backbone)
        self.head = CenterHead(num_classes=num_classes, num_keypoints=cfg.NUM_KEYPOINTS)
        self.pred = PredictCenter(cfg)
        self.loss = LoosCenter(cfg)

    def forward(self, x, targets=None):
        # torch.Size([1, 1280, 13, 13])
        outs = self.backbone(x)
        outs = self.upsample3conv(outs)  # torch.Size([1, 64, 104, 104])
        outs = self.head(outs)
        # 热力图预测 orch.Size([1, 20, 104, 104])
        # 中心点预测 torch.Size([1, 2, 104, 104])
        # 宽高预测 torch.Size([1, 2, 104, 104])
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.loss(outs, targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_labels, p_scores = self.pred(outs, x)
            return ids_batch, p_boxes_ltrb, p_labels, p_scores


if __name__ == '__main__':
    from torchvision import models
    from f_pytorch.tools_model.model_look import f_look_model

    num_classes = 20
    # model = models.mobilenet_v2(pretrained=True)
    # model = ModelOuts4Mobilenet_v2(model)
    # dims_out = [256, 512, 1024]

    model = models.mobilenet_v2(pretrained=True)
    model = ModelOut4Mobilenet_v2(model)

    model = CenterNet(cfg=None, backbone=model, num_classes=num_classes, dim_in_backbone=model.dim_out)
    model.eval()
    f_look_model(model, input=(1, 3, 416, 416))
