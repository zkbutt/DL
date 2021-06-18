import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Resnet
from f_pytorch.tools_model.f_model_api import FModelBase, FConv2d, Scale
from f_pytorch.tools_model.fmodels.model_necks import FPN_out_v3, FPN_out_v4
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_match import match4fcos, boxes_decode4fcos, match4fcos_v2, match4fcos_v3
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import x_bce
from f_tools.floss.focal_loss import BCE_focal_loss, focalloss_fcos
from f_tools.fun_od.f_boxes import bbox_iou4one_2d, bbox_iou4fcos, bbox_iou4one
from f_tools.yufa.x_calc_adv import f_mershgrid
from object_detection.fcos.CONFIG_FCOS import CFG


class FLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs:   torch.Size([2, 2125, 9])
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
        # p_ltrb_t_pos = pboxes_ltrb[mask_pos]
        # image_size_ts = torch.tensor(cfg.IMAGE_SIZE, device=device)
        # g_ltrb_t_pos = g_ltrb[mask_pos] * image_size_ts.repeat(2).view(1, -1)
        # iou = bbox_iou4one(p_ltrb_t_pos, g_ltrb_t_pos, is_giou=True)

        # 这里是解析归一化图  归一化与特图计算的IOU是一致的
        pboxes_ltrb = boxes_decode4fcos(self.cfg, poff_ltrb)
        p_ltrb_pos = pboxes_ltrb[mask_pos]
        g_ltrb_pos = g_ltrb[mask_pos]
        # iou = bbox_iou4one_2d(p_ltrb_pos, g_ltrb_pos, is_giou=True)
        iou = bbox_iou4one(p_ltrb_pos, g_ltrb_pos, is_giou=True)

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


class FLoss_v2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, outs, targets, imgs_ts=None):
        '''

        :param outs: cls1+conf1+ltrb4 torch.Size([2, 2125, 9])
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

        # cls3 centerness1 ltrb4 positive_radius1 positive_ingt1 area1 3+1+4+1+1+1=11
        gdim = cfg.NUM_CLASSES + 1 + 4 + 1 + 1 + 1
        gres = torch.empty((batch, dim_total, gdim), device=device)

        nums_pos = []

        for i in range(batch):
            gboxes_ltrb_b = targets[i]['boxes']
            glabels_b = targets[i]['labels']
            nums_pos.append(gboxes_ltrb_b.shape[0])

            # import time
            # start = time.time()
            gres[i] = match4fcos_v3(gboxes_ltrb_b=gboxes_ltrb_b,
                                    glabels_b=glabels_b,
                                    gdim=gdim,
                                    pcos=outs,
                                    img_ts=imgs_ts[i],
                                    cfg=cfg, )
            # flog.debug('show_time---完成---%s--' % (time.time() - start))

        # cls3 centerness1 ltrb4 positive_radius1 positive_ingt1 area1
        mask_pos = gres[:, :, cfg.NUM_CLASSES + 1 + 4 + 1] == 1  # 框内正例
        nums_pos = torch.tensor(nums_pos, device=device)

        ''' ---------------- cls损失 计算全部样本,正反例,正例为框内本例---------------- '''
        # 框内3D正例 可以用 mask_pos_3d = gcls == 1
        pcls_sigmoid = outs[:, :, :cfg.NUM_CLASSES].sigmoid()
        gcls = gres[:, :, :cfg.NUM_CLASSES]
        l_cls_pos, l_cls_neg = focalloss_fcos(pcls_sigmoid, gcls)
        l_cls_pos = torch.mean(torch.sum(torch.sum(l_cls_pos, -1), -1) / nums_pos)
        l_cls_neg = torch.mean(torch.sum(torch.sum(l_cls_neg, -1), -1) / nums_pos)

        ''' ---------------- conf损失 只计算半径正例 center_ness---------------- '''
        # 半径正例
        mask_pp = gres[:, :, cfg.NUM_CLASSES + 1 + 4] == 1  # 半径正例
        pconf_sigmoid = outs[:, :, cfg.NUM_CLASSES].sigmoid()  # center_ness
        gcenterness = gres[:, :, cfg.NUM_CLASSES]  # (nn,1) # 使用centerness

        # 与 gcenterness 还是以1为准
        # _loss_val = x_bce(pconf_sigmoid, gcenterness, reduction="none")
        _loss_val = x_bce(pconf_sigmoid, torch.ones_like(pconf_sigmoid), reduction="none")  # 用半径1

        # 只算半径正例,提高准确性
        l_conf = 5. * torch.mean(torch.sum(_loss_val * mask_pp.float(), dim=-1) / nums_pos)

        ''' ---------------- box损失 计算框内正例---------------- '''
        # cls3+ conf1+ reg4
        poff_ltrb = outs[:, :, cfg.NUM_CLASSES + 1:cfg.NUM_CLASSES + 1 + 4]  # 这个全是特图的距离 全rule 或 exp
        # goff_ltrb = gres[:, :, s_ + 1:s_ + 1 + 4]
        g_ltrb = gres[:, :, cfg.NUM_CLASSES + 1:cfg.NUM_CLASSES + 1 + 4]

        # 这里是解析归一化图  归一化与特图计算的IOU是一致的
        pboxes_ltrb = boxes_decode4fcos(self.cfg, poff_ltrb)
        # 这里采用的是正例计算 直接平均
        p_ltrb_pos = pboxes_ltrb[mask_pos]
        g_ltrb_pos = g_ltrb[mask_pos]
        iou = bbox_iou4one(p_ltrb_pos, g_ltrb_pos, is_giou=True)
        # 使用 iou 与 1 进行bce  debug iou.isnan().any() or iou.isinf().any()
        l_reg = 5 * torch.mean((1 - iou) * gcenterness[mask_pos])

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
        poff_ltrb = outs[:, :, s_:s_ + 4]
        pboxes_ltrb1 = boxes_decode4fcos(self.cfg, poff_ltrb)
        pboxes_ltrb1.clamp_(min=0., max=1.)  # 预测不返回

        pboxes_ltrb1 = pboxes_ltrb1[mask_pos]
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class FPredict_v2(Predicting_Base):
    def __init__(self, cfg):
        super(FPredict_v2, self).__init__(cfg)

    def p_init(self, outs):
        device = outs.device
        return outs, device

    def get_pscores(self, outs):
        # cls3+ conf1+ reg4  双torch.Size([2, 2125])  值+索引
        cls_conf, plabels = outs[:, :, :self.cfg.NUM_CLASSES].sigmoid().max(-1)
        pconf_sigmoid = outs[:, :, self.cfg.NUM_CLASSES].sigmoid()
        pscores = cls_conf * pconf_sigmoid
        return pscores, plabels, pscores

    def get_stage_res(self, outs, mask_pos, pscores, plabels):
        ids_batch1, _ = torch.where(mask_pos)

        # 解码 cls3+ conf1+ reg4
        poff_ltrb = outs[:, :, self.cfg.NUM_CLASSES + 1:self.cfg.NUM_CLASSES + 1 + 4]
        pboxes_ltrb1 = boxes_decode4fcos(self.cfg, poff_ltrb)
        pboxes_ltrb1.clamp_(min=0., max=1.)  # 预测不返回

        pboxes_ltrb1 = pboxes_ltrb1[mask_pos]
        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1 + 1
        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class FcosHead_v1(nn.Module):

    def __init__(self, cfg, in_channels_list, act='relu', pdim=None):
        '''
        这个是简易头 预测全部共享两层卷积层
        :param cfg:
        :param in_channels_list:in_channels = [16, 32, 64, 128, 256]
        '''
        super().__init__()
        ''' dim共用卷积 cls(3+1),centerness,box '''
        if pdim is None:
            pdim = 1 + cfg.NUM_CLASSES + 1 + 4

        # 分别为4组对应4个输入  每组一个序列
        self.conv_list = nn.ModuleList()
        for c in in_channels_list:
            _pred = nn.Sequential(
                FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=True),
                nn.Conv2d(c * 2, pdim, 1)
            )
            self.conv_list.append(_pred)

    def forward(self, inputs):
        res = []
        for i, x in enumerate(inputs):
            x = self.conv_list[i](x)
            # (b,c,h,w) -> (b,c,hw)
            x = x.view(*x.shape[:2], -1)
            res.append(x)
        # (b,c,hw) -> (b,hw,c)
        res = torch.cat(res, dim=-1).permute(0, 2, 1)
        return res


class FcosHead_v2(nn.Module):

    def __init__(self, cfg, in_channel, num_conv=4, prior_prob=0.01,
                 ctn_on_reg=True, use_dcn=False, o_ceng=5, is_norm=True):
        '''
        这个只能支持 单一输入维度
        :param cfg:
        :param num_conv:  4层CGL
        :param in_channel:
        :param prior_prob: 这个用于初始化 cls 头
        '''
        super().__init__()
        self.num_ceng = len(cfg.STRIDES)
        self.num_conv = num_conv
        self.ctn_on_reg = ctn_on_reg
        self.use_dcn = use_dcn
        self.is_norm = is_norm
        self.cfg = cfg

        self.branch_cls = torch.nn.ModuleList()
        # 第二种形式 self.cls_logits = [] + self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.branch_reg = torch.nn.ModuleList()

        for i in range(self.num_conv):
            # 创建4层共享 CGL
            _cls = FConv2d(in_channel, in_channel, 3, p=1, is_bias=True, norm='gn', g=32, act='relu')
            _reg = FConv2d(in_channel, in_channel, 3, p=1, is_bias=True, norm='gn', g=32, act='relu')
            self.branch_cls.append(_cls)
            self.branch_reg.append(_reg)

        self.branch_reg.append(nn.Conv2d(in_channel, 4, 3, stride=1, padding=1))
        self.branch_cls.append(nn.Conv2d(in_channel, cfg.NUM_CLASSES + 1, 3, stride=1, padding=1))

        self.centerness = nn.Conv2d(in_channel, 1, 3, stride=1, padding=1)
        self.initweight(prior_prob)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(o_ceng)])

    def initweight(self, prior_prob):
        for modules in self.branch_reg:
            for l in modules.modules():
                if hasattr(l, 'conv'):
                    torch.nn.init.normal_(l.conv.weight, std=0.01)
                    torch.nn.init.constant_(l.conv.bias, 0)

        for modules in self.branch_cls[:-1]:
            for l in modules.modules():
                if hasattr(l, 'conv'):
                    torch.nn.init.normal_(l.conv.weight, std=0.01)
                    torch.nn.init.constant_(l.conv.bias, 0)

        bias_init_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.branch_cls[-1].bias, bias_init_value)

        torch.nn.init.normal_(self.centerness.weight, std=0.01)
        torch.nn.init.constant_(self.centerness.bias, 0)

    def forward(self, inputs):
        res_cls = []
        res_reg = []
        res_ctn = []
        for i, x in enumerate(inputs):
            tcls = x
            treg = x
            ''' 先进行一次卷积 4层CBL '''
            for j in range(self.num_conv):
                tcls = self.branch_cls[j](tcls)
                treg = self.branch_reg[j](treg)

            batch, _, _, _ = tcls.shape

            ''' 这里输出ctn 建议在回归分之 '''
            if self.ctn_on_reg:  # conf在回归
                res_ctn.append(self.centerness(treg).view(batch, 1, -1))
            else:
                res_ctn.append(self.centerness(tcls).view(batch, 1, -1))

            ''' 进行最后一层卷积输出需要的维度 这里是强制变维head '''
            tcls = self.branch_cls[-1](tcls)
            batch, c, h, w = tcls.shape
            res_cls.append(tcls.view(batch, c, -1))

            treg = self.scales[i](self.branch_reg[-1](treg))  # 对应归一化系数
            treg = F.relu(treg)  # 这个确保为正
            batch, c, h, w = treg.shape
            res_reg.append(treg.view(batch, c, -1))

        # 回归值的每一层多加一个参数
        # for i in range(self.num_ceng):
        #     res_reg[i] = res_reg[i] * self.scales_4_reg[i]

        res_cls = torch.cat(res_cls, -1)
        res_reg = torch.cat(res_reg, -1)
        res_ctn = torch.cat(res_ctn, -1)
        # cls3 + reg4 +conf1  输出 torch.Size([2, 8, 2134])
        res = torch.cat([res_cls, res_reg, res_ctn], 1).permute(0, 2, 1)
        return res


class FcosHead_v3(nn.Module):

    def __init__(self, cfg, in_channel, num_conv=4, prior_prob=0.01,
                 ctn_on_reg=True, use_dcn=False, o_ceng=5, is_norm=True):
        '''
        这个是单头 只支持输入单层 每特图需创建一个
        :param cfg:
        :param num_conv:  4层CGL
        :param in_channel:
        :param prior_prob: 这个用于初始化 cls 头
        '''
        super().__init__()
        self.num_ceng = len(cfg.STRIDES)
        self.num_conv = num_conv
        self.ctn_on_reg = ctn_on_reg
        self.use_dcn = use_dcn
        self.is_norm = is_norm
        self.cfg = cfg

        self.branch_cls = torch.nn.ModuleList()
        # 第二种形式 self.cls_logits = [] + self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.branch_reg = torch.nn.ModuleList()

        for i in range(self.num_conv):
            # 创建4层共享 CGL
            _cls = FConv2d(in_channel, in_channel, 3, p=1, is_bias=True, norm='gn', g=32, act='relu')
            _reg = FConv2d(in_channel, in_channel, 3, p=1, is_bias=True, norm='gn', g=32, act='relu')
            self.branch_cls.append(_cls)
            self.branch_reg.append(_reg)

        self.branch_reg.append(nn.Conv2d(in_channel, 4, 3, stride=1, padding=1))
        self.branch_cls.append(nn.Conv2d(in_channel, cfg.NUM_CLASSES, 3, stride=1, padding=1))

        self.centerness = nn.Conv2d(in_channel, 1, 3, stride=1, padding=1)
        self.initweight(prior_prob)

        self.scales = Scale(init_value=1.0)

    def initweight(self, prior_prob):
        for modules in self.branch_reg:
            for l in modules.modules():
                if hasattr(l, 'conv'):
                    torch.nn.init.normal_(l.conv.weight, std=0.01)
                    torch.nn.init.constant_(l.conv.bias, 0)

        for modules in self.branch_cls[:-1]:
            for l in modules.modules():
                if hasattr(l, 'conv'):
                    torch.nn.init.normal_(l.conv.weight, std=0.01)
                    torch.nn.init.constant_(l.conv.bias, 0)

        bias_init_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.branch_cls[-1].bias, bias_init_value)

        torch.nn.init.normal_(self.centerness.weight, std=0.01)
        torch.nn.init.constant_(self.centerness.bias, 0)

    def forward(self, input):
        tcls = input
        treg = input
        ''' 先进行一次卷积 4层CBL '''
        for j in range(self.num_conv):
            tcls = self.branch_cls[j](tcls)
            treg = self.branch_reg[j](treg)

        batch, _, _, _ = tcls.shape

        ''' 这里输出ctn 建议在回归分之 '''
        if self.ctn_on_reg:  # conf在回归
            res_ctn = self.centerness(treg).view(batch, 1, -1)
        else:
            res_ctn = self.centerness(tcls).view(batch, 1, -1)

        ''' 进行最后一层卷积输出需要的维度 这里是强制变维head '''
        tcls = self.branch_cls[-1](tcls)
        batch, c, h, w = tcls.shape
        res_cls = tcls.view(batch, c, -1)

        treg = self.scales(self.branch_reg[-1](treg))  # 对应归一化系数
        treg = F.relu(treg)  # 这个确保为正
        batch, c, h, w = treg.shape
        res_reg = treg.view(batch, c, -1)

        # cls3 + reg4 +conf1  输出 torch.Size([2, 8, 2134])
        res = torch.cat([res_cls, res_reg, res_ctn], 1)
        return res


class FcosNet_v2(nn.Module):
    def __init__(self, backbone, cfg, o_ceng, num_conv):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        channels_list = backbone.dims_out  # (512, 1024, 2048)
        ''' 这两个是一套 '''
        if cfg.MODE_TRAIN == 3:
            # 保持输出维度为统一的 256
            self.fpn5 = FPN_out_v4(channels_list, out_channel=backbone.dims_out[0], o_ceng=o_ceng, num_conv=num_conv)
            self.head = FcosHead_v2(self.cfg, in_channel=backbone.dims_out[0])
        elif cfg.MODE_TRAIN == 5:
            self.fpn5 = FPN_out_v4(channels_list, o_ceng=o_ceng, num_conv=num_conv)
            self.head = FcosHead_v1(self.cfg, self.fpn5.dims_out, pdim=cfg.NUM_CLASSES + 1 + 4)
        else:
            # cfg.MODE_TRAIN == 2 与 cfg.MODE_TRAIN == 1一样的
            self.fpn5 = FPN_out_v4(channels_list, o_ceng=o_ceng, num_conv=num_conv)
            self.head = FcosHead_v1(self.cfg, self.fpn5.dims_out)

    def forward(self, x):
        '''尺寸由大到小 维度由低到高 [2, 512, 40, 40] [2, 1024, 20, 20] [2, 2048, 10, 10] '''
        C_3, C_4, C_5 = self.backbone(x)
        ''' ...256维 + [2, 256, 5, 5] [2, 256, 3, 3] '''
        res = self.fpn5((C_3, C_4, C_5))
        res = self.head(res)
        return res


class FcosNet_v3(nn.Module):
    '''
    这个用的原装fpn 输出不同维  采用专业头
    '''

    def __init__(self, backbone, cfg, o_ceng, num_conv):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        channels_list = backbone.dims_out  # (512, 1024, 2048)
        self.fpn5 = FPN_out_v4(channels_list, out_channel=None, o_ceng=o_ceng, num_conv=num_conv)
        self.heads = nn.ModuleList()
        for c in self.fpn5.dims_out:
            self.heads.append(FcosHead_v3(self.cfg, in_channel=c))

    def forward(self, x):
        '''尺寸由大到小 维度由低到高 [2, 512, 40, 40] [2, 1024, 20, 20] [2, 2048, 10, 10] '''
        C_3, C_4, C_5 = self.backbone(x)
        res = self.fpn5((C_3, C_4, C_5))
        res_list = []
        for i, model_head in enumerate(self.heads):
            res_list.append(model_head(res[i]))

        # 输出 torch.Size([2, 3614, 8])
        res = torch.cat(res_list, -1).permute(0, 2, 1)
        return res


class Fcos(FModelBase):
    def __init__(self, backbone, cfg, device=torch.device('cpu')):
        if cfg.MODE_TRAIN == 1:
            # net = FcosNet_v1(backbone, cfg)  # 这个基本不用
            pass
        elif cfg.MODE_TRAIN == 2 or cfg.MODE_TRAIN == 5:
            # fpn4+简易头
            net = FcosNet_v2(backbone, cfg, o_ceng=4, num_conv=3)
        elif cfg.MODE_TRAIN == 3:
            # fpn5(单一维度)+论文头+
            net = FcosNet_v2(backbone, cfg, o_ceng=5, num_conv=3)
        elif cfg.MODE_TRAIN == 4:
            # fpn5(多维度)+论文头
            net = FcosNet_v3(backbone, cfg, o_ceng=5, num_conv=3)
        else:
            raise Exception('cfg.MODE_TRAIN 错误 %s' % cfg.MODE_TRAIN)
        if cfg.MODE_TRAIN == 5:  # cls3
            losser = FLoss_v2(cfg)
            preder = FPredict_v2(cfg)
        else:
            losser = FLoss(cfg)
            preder = FPredict(cfg)
        super(Fcos, self).__init__(net, losser, preder)


if __name__ == '__main__':
    from f_pytorch.tools_model.model_look import f_look_tw

    ''' 模型测试 '''
    cfg = CFG()
    cfg.NUMS_ANC = [3, 3, 3]
    cfg.NUM_CLASSES = 3

    cfg.IMAGE_SIZE = (320, 320)
    cfg.STRIDES = [8, 16, 32, 64]
    cfg.MODE_TRAIN = 3

    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)

    # model = FcosNet_v1(model, cfg)
    # model = FcosNet_v2(model, cfg, o_ceng=4, num_conv=3)
    # model = FcosNet_v2(model, cfg, o_ceng=5, num_conv=3)
    model = FcosNet_v3(model, cfg, o_ceng=5, num_conv=3)
    model.train()

    # cfg.STRIDES = [8, 16, 32, 64, 128]
    # model = FcosNet(model, cfg, o_ceng=5)

    print(model(torch.rand(2, 3, 416, 416)).shape)
    # model.eval()
    f_look_tw(model, input=(1, 3, 320, 320), name='fcos')

    # from f_pytorch.tools_model.model_look import f_look_summary
    # f_look_summary(model)
