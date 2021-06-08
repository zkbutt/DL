import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Resnet
from f_pytorch.tools_model.f_model_api import FModelBase, FConv2d
from f_pytorch.tools_model.fmodels.model_necks import FPN_out_v2
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_match import match4fcos, boxes_decode4fcos, match4fcos_v2
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import x_bce
from f_tools.floss.focal_loss import BCE_focal_loss, focalloss_fcos
from f_tools.fun_od.f_boxes import bbox_iou4one_2d, bbox_iou4fcos, bbox_iou4one_3d
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
        batch, dim_total, pdim = outs.shape

        #  back cls centerness ltrb positivesample iou area
        gdim = 1 + cfg.NUM_CLASSES + 1 + 4 + 1 + 1 + 1
        gres = torch.empty((batch, dim_total, gdim), device=device)

        for i in range(batch):
            gboxes_ltrb_b = targets[i]['boxes']
            glabels_b = targets[i]['labels']

            import time
            # start = time.time()
            gres[i] = match4fcos_v2(gboxes_ltrb_b=gboxes_ltrb_b,
                                    glabels_b=glabels_b,
                                    gdim=gdim,
                                    pcos=outs,
                                    img_ts=imgs_ts[i],
                                    cfg=cfg, )
            # gres[i] = match4fcos(gboxes_ltrb_b=gboxes_ltrb_b,
            #                      glabels_b=glabels_b,
            #                      gdim=gdim,
            #                      pcos=outs,
            #                      img_ts=imgs_ts[i],
            #                      cfg=cfg, )
            # flog.debug('show_time---完成---%s--' % (time.time() - start))

        s_ = 1 + cfg.NUM_CLASSES
        # outs = outs[:, :, :s_ + 1].sigmoid()
        mask_pos = gres[:, :, 0] == 0  # 背景为0 是正例
        nums_pos = torch.sum(mask_pos, dim=-1)
        nums_pos = torch.max(nums_pos, torch.ones_like(nums_pos, device=device))

        # back cls centerness ltrb positivesample iou(这个暂时无用) area [2125, 12]

        ''' ---------------- cls损失 计算全部样本,正反例,正例为框内本例---------------- '''
        # obj_cls_loss = BCE_focal_loss()
        # 这里多一个背景一起算
        pcls_sigmoid = outs[:, :, :s_].sigmoid()
        gcls = gres[:, :, :s_]
        # l_cls = torch.mean(obj_cls_loss(pcls_sigmoid, gcls) / nums_pos)
        l_cls_pos, l_cls_neg = focalloss_fcos(pcls_sigmoid, gcls)
        l_cls_pos = torch.mean(torch.sum(torch.sum(l_cls_pos, -1), -1) / nums_pos)
        l_cls_neg = torch.mean(torch.sum(torch.sum(l_cls_neg, -1), -1) / nums_pos)

        ''' ---------------- conf损失 只计算半径正例 center_ness---------------- '''
        # 和 positive sample 算正例
        mask_pp = gres[:, :, s_ + 1 + 4] == 1
        pconf_sigmoid = outs[:, :, s_].sigmoid()  # center_ness
        gcenterness = gres[:, :, s_]  # (nn,1) # 使用centerness

        # _loss_val = x_bce(pconf_sigmoid, gcenterness, reduction="none")
        _loss_val = x_bce(pconf_sigmoid, torch.ones_like(pconf_sigmoid), reduction="none")  # 用半径1

        # 只算半径正例,提高准确性
        l_conf = 5. * torch.mean(torch.sum(_loss_val * mask_pp.float(), dim=-1) / nums_pos)

        ''' ---------------- box损失 计算框内正例---------------- '''
        # conf1 + cls3 + reg4
        # poff_ltrb_exp = torch.exp(outs[:, :, s_:s_ + 4])
        poff_ltrb = outs[:, :, s_:s_ + 4]  # 这个全是特图的距离 全rule 或 exp
        # goff_ltrb = gres[:, :, s_ + 1:s_ + 1 + 4]
        g_ltrb = gres[:, :, s_ + 1:s_ + 1 + 4]

        # _loss_val = F.smooth_l1_loss(poff_ltrb, goff_ltrb, reduction='none')
        # _loss_val = F.mse_loss(poff_ltrb_exp, goff_ltrb, reduction='none')
        # l_reg = torch.sum(torch.sum(_loss_val, -1) * gconf * mask_pos.float(), -1)
        # l_reg = torch.mean(l_reg / nums_pos)

        # 这里是解析归一化图
        # pboxes_ltrb = boxes_decode4fcos(self.cfg, poff_ltrb, is_t=True)
        # p_ltrb_pos = pboxes_ltrb[mask_pos]
        # image_size_ts = torch.tensor(cfg.IMAGE_SIZE, device=device)
        # g_ltrb_pos = g_ltrb[mask_pos] * image_size_ts.repeat(2).view(1, -1)

        # giou = gres[:, :, s_ + 1 + 4 + 1]
        # torch.Size([2, 2125])
        # iou = bbox_iou4one_2d(p_ltrb_pos, g_ltrb_pos, is_giou=True)

        # 这里是解析归一化图
        pboxes_ltrb = boxes_decode4fcos(self.cfg, poff_ltrb)
        p_ltrb_pos = pboxes_ltrb[mask_pos]
        g_ltrb_pos = g_ltrb[mask_pos]
        # iou = bbox_iou4one_2d(p_ltrb_pos, g_ltrb_pos, is_giou=True)
        iou = bbox_iou4one_3d(p_ltrb_pos, g_ltrb_pos, is_giou=True)

        # 使用 iou 与 1 进行bce  debug iou.isnan().any() or iou.isinf().any()
        l_reg = 5 * torch.mean((1 - iou) * gcenterness[mask_pos])

        # iou2 = bbox_iou4one_3d(pboxes_ltrb, g_ltrb, is_giou=True) # 2D 和 3D效果是一样的
        # l_reg2 = torch.mean(torch.sum((1 - iou2) * gcenterness * mask_pos.float(), -1) / nums_pos)

        # _loss_val = x_bce(iou, giou, reduction="none")
        # l_iou = torch.mean(torch.sum(_loss_val * gconf * mask_pos.float(), dim=-1) / nums_pos)

        l_total = l_cls_pos + l_cls_neg + l_conf + l_reg

        log_dict = {}
        log_dict['l_total'] = l_total.item()
        log_dict['l_cls_pos'] = l_cls_pos.item()
        log_dict['l_cls_neg'] = l_cls_neg.item()
        log_dict['l_conf'] = l_conf.item()
        log_dict['l_reg'] = l_reg.item()
        # log_dict['l_iou_max'] = iou.max().item()

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
        # poff_ltrb_exp = torch.exp(outs[:, :, s_:s_ + 4])
        poff_ltrb_exp = outs[:, :, s_:s_ + 4]
        pboxes_ltrb1 = boxes_decode4fcos(self.cfg, poff_ltrb_exp)
        pboxes_ltrb1.clamp_(min=0., max=1.)  # 预测不返回

        pboxes_ltrb1 = pboxes_ltrb1[mask_pos]
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


# class FcosNet(nn.Module):
#     def __init__(self, backbone, cfg, o_ceng=4):
#         '''
#         输出 4 层
#         :param backbone:
#         :param cfg:
#         :param out_channels_fpn:
#         '''
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = backbone
#         channels_list = backbone.dims_out
#         self.fpn = FPN_out_v2(channels_list, feature_size=256, o_ceng=o_ceng)
#         self.head = FcosHead(self.cfg)
#
#     def forward(self, x):
#         # C_3, C_4, C_5
#         Bout_ceng3 = self.backbone(x)
#         # P3_x, P4_x, P5_x, P6_x
#         Pout_fpn5 = self.fpn(Bout_ceng3)
#         res = self.head(Pout_fpn5)
#         return res


class FcosNet(nn.Module):
    def __init__(self, backbone, cfg, out_channels_fpn=256, o_ceng=4):
        '''
        输出 4 层
        :param backbone:
        :param cfg:
        :param out_channels_fpn:
        '''
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        d512 = backbone.dims_out[-1]
        d1024 = int(d512 * 2)
        d256 = int(d512 / 2)
        d128 = int(d256 / 2)
        d64 = int(d128 / 2)

        is_bias = True # 原装是True

        # C_5 -> C_6  [1, 1024, 7, 7]
        self.conv_3x3_C_6 = FConv2d(d512, d1024, 3, p=1, s=2, is_bias=is_bias)

        # detection head
        # All branches share the self.head
        # 1 is for background label

        # C_6 - P_6 torch.Size([1, 512, 7, 7])
        self.conv_set_6 = nn.Sequential(
            FConv2d(d1024, d512, 1, is_bias=is_bias),
            FConv2d(d512, d1024, 3, p=1, is_bias=is_bias),
            FConv2d(d1024, d512, 1, is_bias=is_bias),
        )

        self.gdim = 1 + cfg.NUM_CLASSES + 1 + 4
        self.pred_6 = nn.Sequential(
            FConv2d(d512, d1024, 3, p=1, is_bias=is_bias),
            nn.Conv2d(d1024, self.gdim, 1)
        )

        # P_5
        self.conv_set_5 = nn.Sequential(
            FConv2d(d512, d256, 1, is_bias=is_bias),
            FConv2d(d256, d512, 3, p=1, is_bias=is_bias),
            FConv2d(d512, d256, 1, is_bias=is_bias)
        )
        self.conv_1x1_5 = FConv2d(d256, d128, 1, is_bias=is_bias)
        self.pred_5 = nn.Sequential(
            FConv2d(d256, d512, 3, p=1, is_bias=is_bias),
            nn.Conv2d(d512, self.gdim, 1)
        )

        # P_4
        self.conv_set_4 = nn.Sequential(
            FConv2d(d256 + d128, d128, 1, is_bias=is_bias),
            FConv2d(d128, d256, 3, p=1, is_bias=is_bias),
            FConv2d(d256, d128, 1, is_bias=is_bias)
        )
        self.conv_1x1_4 = FConv2d(d128, d64, 1, is_bias=is_bias)
        self.pred_4 = nn.Sequential(
            FConv2d(d128, d256, 3, p=1, is_bias=is_bias),
            nn.Conv2d(d256, self.gdim, 1)
        )

        # P_3
        self.conv_set_3 = nn.Sequential(
            FConv2d(d128 + d64, d64, 1, is_bias=is_bias),
            FConv2d(d64, d128, 3, p=1, is_bias=is_bias),
            FConv2d(d128, d64, 1, is_bias=is_bias)
        )
        self.pred_3 = nn.Sequential(
            FConv2d(d64, d128, 3, p=1, is_bias=is_bias),
            nn.Conv2d(d128, self.gdim, 1)
        )

    def forward(self, x):
        # torch.Size([2, 3, 320, 320])
        cfg = self.cfg
        # backbone ([2, 128, 40, 40])  ([2, 256, 20, 20]) ([2, 512, 10, 10])
        C_3, C_4, C_5 = self.backbone(x)
        B = C_3.shape[0]

        # 直接输出C6
        C_6 = self.conv_3x3_C_6(C_5)  # torch.Size([2, 1024, 5, 5])
        pred_6 = self.pred_6(self.conv_set_6(C_6)).view(B, self.gdim, -1)

        # P_5 向上 torch.Size([2, 256, 10, 10])
        C_5 = self.conv_set_5(C_5)
        # 上采样 [2, 256, 10, 10] -> [2, 128, 20, 20]
        C_5_up = F.interpolate(self.conv_1x1_5(C_5), scale_factor=2.0, mode='bilinear', align_corners=True)
        pred_5 = self.pred_5(C_5).view(B, self.gdim, -1)  # torch.Size([2, 9, 100])

        # P_4 [2, 256, 20, 20] ^ [2, 128, 20, 20]
        C_4 = torch.cat([C_4, C_5_up], dim=1)
        C_4 = self.conv_set_4(C_4)
        # [2, 64, 40, 40]
        C_4_up = F.interpolate(self.conv_1x1_4(C_4), scale_factor=2.0, mode='bilinear', align_corners=True)
        pred_4 = self.pred_4(C_4).view(B, self.gdim, -1)

        # P_3
        C_3 = torch.cat([C_3, C_4_up], dim=1)
        C_3 = self.conv_set_3(C_3)
        pred_3 = self.pred_3(C_3).view(B, self.gdim, -1)

        # [2, 9, 1600] [2, 9, 400] [2, 9, 100] [2, 9, 25] 输出4层
        total_prediction = torch.cat([pred_3, pred_4, pred_5, pred_6], dim=-1).permute(0, 2, 1)

        return total_prediction


class FcosHead(nn.Module):

    def __init__(self, cfg, num_conv=4, feature_size=256, prior_prob=0.01, ctn_on_reg=True):
        '''
        经过
        :param cfg:
        :param num_conv:  4层CGL
        :param feature_size:
        :param prior_prob: 这个用于初始化 cls 头
        '''
        super().__init__()
        self.num_ceng = len(cfg.STRIDES)
        self.num_conv = num_conv
        self.ctn_on_reg = ctn_on_reg

        self.convs_k3gg_cls = torch.nn.ModuleList()
        self.convs_k3gg_reg = torch.nn.ModuleList()

        for i in range(self.num_conv):
            # 创建4层共享 CGL
            convk3gg_cls = FConv2d(feature_size, feature_size, 3, s=1, is_bias=True, norm='gn', g=32,
                                   act='relu')
            convk3gg_reg = FConv2d(feature_size, feature_size, 3, s=1, is_bias=True, norm='gn', g=32,
                                   act='relu')
            self.convs_k3gg_cls.append(convk3gg_cls)
            self.convs_k3gg_reg.append(convk3gg_reg)

        convk3_cls = FConv2d(feature_size, cfg.NUM_CLASSES, 3, s=1, is_bias=True, act=None)
        # convk3_cls 初始化
        bias_init_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(convk3_cls.conv.bias, bias_init_value)
        self.convs_k3gg_cls.append(convk3_cls)

        convk3_reg = FConv2d(feature_size, 4, 3, s=1, is_bias=True, act=None)
        self.convs_k3gg_reg.append(convk3_reg)

        self.convk3_ctn = FConv2d(feature_size, 1, 3, s=1, is_bias=True, act=None)

        # self.scales_4_reg = torch.nn.ParameterList()  # 为每层 回归分支预设一个比例参数 学习
        # for i in range(self.num_ceng):
        #     scale = torch.nn.Parameter(torch.ones(1, ))
        #     self.scales_4_reg.append(scale)

    def forward(self, inputs):
        res_cls = []
        res_reg = []
        res_ctn = []
        for x in inputs:
            tcls = x
            treg = x
            ''' 先进行一次卷积 4层CBL '''
            for i in range(self.num_conv):
                tcls = self.convs_k3gg_cls[i](tcls)
                treg = self.convs_k3gg_reg[i](treg)

            batch, _, _, _ = tcls.shape

            ''' 这里输出ctn 建议在回归分之 '''
            if self.ctn_on_reg:  # conf在回归
                res_ctn.append(self.convk3_ctn(treg).view(batch, 1, -1))
            else:
                res_ctn.append(self.convk3_ctn(tcls).view(batch, 1, -1))

            ''' 进行最后一层卷积输出需要的维度 这里是强制变维head '''
            tcls = self.convs_k3gg_cls[-1](tcls)
            treg = self.convs_k3gg_reg[-1](treg)

            batch, c, h, w = tcls.shape
            tcls = tcls.view(batch, c, -1)
            batch, c, h, w = treg.shape
            treg = treg.view(batch, c, -1)

            res_cls.append(tcls)
            res_reg.append(treg)

        # 回归值的每一层多加一个参数
        # for i in range(self.num_ceng):
        #     res_reg[i] = res_reg[i] * self.scales_4_reg[i]

        res_cls = torch.cat(res_cls, -1)

        res_reg = torch.cat(res_reg, -1)
        # 使输出为正 或使用exp
        # res_reg = F.relu(res_reg)
        res_reg = res_reg.exp()

        res_ctn = torch.cat(res_ctn, -1)
        # cls3 + reg4 +conf1  输出 torch.Size([2, 8, 2134])
        res = torch.cat([res_cls, res_reg, res_ctn], 1).permute(0, 2, 1)
        return res


# class FcosNet_v2(nn.Module):
#     def __init__(self, backbone, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = backbone
#         channels_list = backbone.dims_out  # (512, 1024, 2048)
#         self.fpn5 = FPN_out_v2(channels_list, feature_size=256, o_ceng=5)
#         self.head = FcosHead(self.cfg)
#
#     def forward(self, x):
#         ''' [2, 512, 40, 40] [2, 1024, 20, 20] [2, 2048, 10, 10] '''
#         C_3, C_4, C_5 = self.backbone(x)
#         ''' ...256维 + [2, 256, 5, 5] [2, 256, 3, 3] '''
#         P3_x, P4_x, P5_x, P6_x, P7_x = self.fpn5((C_3, C_4, C_5))
#         res = self.head((P3_x, P4_x, P5_x, P6_x, P7_x))
#         return res


class Fcos(FModelBase):
    def __init__(self, backbone, cfg, device=torch.device('cpu')):
        if cfg.MODE_TRAIN == 1:
            net = FcosNet(backbone, cfg, o_ceng=4)
        elif cfg.MODE_TRAIN == 2:
            net = FcosNet(backbone, cfg, o_ceng=5)
        losser = FLoss(cfg)
        preder = FPredict(cfg)
        super(Fcos, self).__init__(net, losser, preder)


if __name__ == '__main__':
    from f_pytorch.tools_model.model_look import f_look_tw

    ''' 模型测试 '''
    cfg = CFG()
    cfg.NUMS_ANC = [3, 3, 3]
    cfg.NUM_CLASSES = 3
    cfg.IMAGE_SIZE = (416, 416)
    cfg.STRIDES = [8, 16, 32, 64]
    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)

    model = FcosNet(model, cfg)

    # cfg.STRIDES = [8, 16, 32, 64, 128]
    # model = FcosNet(model, cfg, o_ceng=5)

    # print(model(torch.rand(2, 3, 416, 416)).shape)
    # model.eval()
    f_look_tw(model, input=(1, 3, 416, 416), name='fcos')
