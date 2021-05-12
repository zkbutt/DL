import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Resnet
from f_pytorch.tools_model.f_model_api import CBL, FModelBase
from f_pytorch.tools_model.model_look import f_look_tw
from f_tools.fits.f_match import match4fcos, boxes_decode4fcos
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.focal_loss import BCE_focal_loss
from f_tools.fun_od.f_boxes import bbox_iou4one, bbox_iou4fcos
from f_tools.yufa.x_calc_adv import f_mershgrid
from object_detection.fcos.CONFIG_FCOS import CFG


class FLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs: torch.Size([2, 2125, 9])
        :param targets:
            'image_id': 413,
            'size': tensor([500., 309.])
            'boxes': tensor([[0.31400, 0.31715, 0.71000, 0.60841]]),
            'labels': tensor([1.])
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        device = outs.device
        batch, pdim, c = outs.shape

        #  back cls centerness ltrb positivesample iou area
        gdim = 1 + cfg.NUM_CLASSES + 1 + 4 + 1 + 1 + 1
        gres = torch.empty((batch, pdim, gdim), device=device)

        for i in range(batch):
            gboxes_ltrb_b = targets[i]['boxes']
            glabels_b = targets[i]['labels']

            gres[i] = match4fcos(gboxes_ltrb_b=gboxes_ltrb_b,
                                 glabels_b=glabels_b,
                                 gdim=gdim,
                                 pcos=outs,
                                 img_ts=imgs_ts[i],
                                 cfg=cfg, )

        s_ = 1 + cfg.NUM_CLASSES
        # outs = outs[:, :, :s_ + 1].sigmoid()
        mask_pos = gres[:, :, s_ + 1 + 4]
        nums_pos = torch.sum(mask_pos, dim=-1)
        nums_pos = torch.max(nums_pos, torch.ones_like(nums_pos, device=device))

        ''' ---------------- cls损失 ---------------- '''
        obj_cls_loss = BCE_focal_loss()
        # 这里多一个背景一起算
        pcls_sigmoid = outs[:, :, :s_].sigmoid()
        gcls = gres[:, :, :s_]
        l_cls = torch.mean(obj_cls_loss(pcls_sigmoid, gcls) / nums_pos)

        ''' ---------------- conf损失 只计算正例---------------- '''
        pconf_sigmoid = outs[:, :, s_].sigmoid()  # center_ness
        gconf = gres[:, :, s_]
        _loss_val = F.binary_cross_entropy(pconf_sigmoid, gconf, reduction="none")
        l_conf = 5. * torch.mean(torch.sum(_loss_val * mask_pos.float(), dim=-1) / nums_pos)

        ''' ---------------- box损失 只计算正例---------------- '''
        poff_ltrb_exp = torch.exp(outs[:, :, s_ + 1:s_ + 1 + 4])
        goff_ltrb = gres[:, :, s_ + 1:s_ + 1 + 4]
        giou = gres[:, :, s_ + 1 + 4 + 1]
        # torch.Size([2, 2125])
        iou = bbox_iou4fcos(poff_ltrb_exp, goff_ltrb)
        # 使用 iou 与 1 进行bce
        _loss_val = F.binary_cross_entropy(iou, giou, reduction="none")
        l_iou = torch.mean(torch.sum(_loss_val * mask_pos.float(), dim=-1) / nums_pos)

        l_total = l_cls + l_conf + l_iou

        log_dict = {}
        log_dict['l_total'] = l_total.item()
        log_dict['l_cls'] = l_cls.item()
        log_dict['l_conf'] = l_conf.item()
        log_dict['l_iou'] = l_iou.item()

        return l_total, log_dict


class FPredict(Predicting_Base):
    def __init__(self, cfg):
        super(FPredict, self).__init__(cfg)

    def p_init(self, outs):
        device = outs.device
        return outs, device

    def get_pscores(self, outs):
        s_ = 1 + self.cfg.NUM_CLASSES
        # 这里不再需要背景
        cls_conf, plabels = outs[:, :, 1:s_].sigmoid().max(-1)
        pconf_sigmoid = outs[:, :, s_].sigmoid()
        pscores = cls_conf * pconf_sigmoid
        return pscores, plabels, pscores

    def get_stage_res(self, outs, mask_pos, pscores, plabels):
        s_ = 1 + self.cfg.NUM_CLASSES
        ids_batch1, _ = torch.where(mask_pos)

        # 解码
        poff_ltrb_exp = torch.exp(outs[:, :, s_ + 1:s_ + 1 + 4])
        pboxes_ltrb1 = boxes_decode4fcos(self.cfg, poff_ltrb_exp)
        pboxes_ltrb1.clamp_(min=0., max=1.)  # 预测不返回

        pboxes_ltrb1 = pboxes_ltrb1[mask_pos]
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class FcosNet(nn.Module):
    def __init__(self, backbone, cfg, out_channels_fpn=256):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        d512 = backbone.dims_out[-1]
        d1024 = int(d512 * 2)
        d256 = int(d512 / 2)
        d128 = int(d256 / 2)
        d64 = int(d128 / 2)

        # C_5 -> C_6  [1, 1024, 7, 7]
        self.conv_3x3_C_6 = CBL(d512, d1024, 3, padding=1, stride=2)

        # detection head
        # All branches share the self.head
        # 1 is for background label

        # C_6 - P_6 torch.Size([1, 512, 7, 7])
        self.conv_set_6 = nn.Sequential(
            CBL(d1024, d512, 1),
            CBL(d512, d1024, 3, padding=1),
            CBL(d1024, d512, 1),
        )

        gdim = 1 + cfg.NUM_CLASSES + 1 + 4
        self.pred_6 = nn.Sequential(
            CBL(d512, d1024, 3, padding=1),
            nn.Conv2d(d1024, gdim, 1)
        )

        # P_5
        self.conv_set_5 = nn.Sequential(
            CBL(d512, d256, 1),
            CBL(d256, d512, 3, padding=1),
            CBL(d512, d256, 1)
        )
        self.conv_1x1_5 = CBL(d256, d128, 1)
        self.pred_5 = nn.Sequential(
            CBL(d256, d512, 3, padding=1),
            nn.Conv2d(d512, gdim, 1)
        )

        # P_4
        self.conv_set_4 = nn.Sequential(
            CBL(d256 + d128, d128, 1),
            CBL(d128, d256, 3, padding=1),
            CBL(d256, d128, 1)
        )
        self.conv_1x1_4 = CBL(d128, d64, 1)
        self.pred_4 = nn.Sequential(
            CBL(d128, d256, 3, padding=1),
            nn.Conv2d(d256, gdim, 1)
        )

        # P_3
        self.conv_set_3 = nn.Sequential(
            CBL(d128 + d64, d64, 1),
            CBL(d64, d128, 3, padding=1),
            CBL(d128, d64, 1)
        )
        self.pred_3 = nn.Sequential(
            CBL(d64, d128, 3, padding=1),
            nn.Conv2d(d128, gdim, 1)
        )

    def forward(self, x):
        # torch.Size([2, 3, 320, 320])
        cfg = self.cfg
        # backbone ([2, 128, 40, 40])  ([2, 256, 20, 20]) ([2, 512, 10, 10])
        C_3, C_4, C_5 = self.backbone(x)
        C_6 = self.conv_3x3_C_6(C_5)  # torch.Size([2, 1024, 5, 5])
        B = C_3.shape[0]

        # self.cfg.NUMS_CENG = []
        # self.cfg.NUMS_CENG.append(torch.prod(torch.tensor(C_3.shape[-2:])))
        # self.cfg.NUMS_CENG.append(torch.prod(torch.tensor(C_4.shape[-2:])))
        # self.cfg.NUMS_CENG.append(torch.prod(torch.tensor(C_5.shape[-2:])))
        # self.cfg.NUMS_CENG.append(torch.prod(torch.tensor(C_6.shape[-2:])))

        # P_6 torch.Size([2, 9, 25])
        gdim = 1 + cfg.NUM_CLASSES + 1 + 4
        pred_6 = self.pred_6(self.conv_set_6(C_6)).view(B, gdim, -1)

        # P_5 向上 torch.Size([2, 256, 10, 10])
        C_5 = self.conv_set_5(C_5)
        # 上采样 [2, 256, 10, 10] -> [2, 128, 20, 20]
        C_5_up = F.interpolate(self.conv_1x1_5(C_5), scale_factor=2.0, mode='bilinear', align_corners=True)
        pred_5 = self.pred_5(C_5).view(B, gdim, -1)  # torch.Size([2, 9, 100])

        # P_4 [2, 256, 20, 20] ^ [2, 128, 20, 20]
        C_4 = torch.cat([C_4, C_5_up], dim=1)
        C_4 = self.conv_set_4(C_4)
        # [2, 64, 40, 40]
        C_4_up = F.interpolate(self.conv_1x1_4(C_4), scale_factor=2.0, mode='bilinear', align_corners=True)
        pred_4 = self.pred_4(C_4).view(B, gdim, -1)

        # P_3
        C_3 = torch.cat([C_3, C_4_up], dim=1)
        C_3 = self.conv_set_3(C_3)
        pred_3 = self.pred_3(C_3).view(B, gdim, -1)

        # [2, 9, 1600] [2, 9, 400] [2, 9, 100] [2, 9, 25] 输出4层
        total_prediction = torch.cat([pred_3, pred_4, pred_5, pred_6], dim=-1).permute(0, 2, 1)

        return total_prediction


class Fcos(FModelBase):
    def __init__(self, backbone, cfg, device=torch.device('cpu')):
        net = FcosNet(backbone, cfg)
        losser = FLoss(cfg)
        preder = FPredict(cfg)
        super(Fcos, self).__init__(net, losser, preder)


if __name__ == '__main__':
    cfg = CFG()
    cfg.NUMS_ANC = [3, 3, 3]
    cfg.NUM_CLASSES = 3
    cfg.IMAGE_SIZE = (416, 416)
    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)
    model = FcosNet(model, cfg)
    model.eval()
    f_look_tw(model, input=(1, 3, 416, 416), name='fcos')
