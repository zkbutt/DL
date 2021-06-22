import torch
import torch.nn as nn
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Resnet
from f_pytorch.tools_model.f_model_api import FModelBase, FConv2d, Scale
from f_pytorch.tools_model.fmodels.model_necks import FPN_out_v3, FPN_out_v4
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_match import boxes_decode4fcos, match4fcos_v2, match4fcos_v3_noback, match4fcos_keypoint
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
        模型尺寸list[20,10,5,3]
        :param outs:   torch.Size([2, 534, 7]) in 160 输出[]
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

        #  1 + cfg.NUM_CLASSES + 1 + 4 + cfg.NUM_KEYPOINTS * 2
        #  back cls centerness ltrb positivesample iou area
        gdim = 1 + cfg.NUM_CLASSES + 1 + 4 + 1 + 1 + 1
        gres = torch.empty((batch, dim_total, gdim), device=device)

        for i in range(batch):
            gboxes_ltrb_b = targets[i]['boxes']
            glabels_b = targets[i]['labels']

            gres[i] = match4fcos_v2(gboxes_ltrb_b=gboxes_ltrb_b,
                                    glabels_b=glabels_b,
                                    gdim=gdim,
                                    pcos=outs,
                                    img_ts=imgs_ts[i],
                                    cfg=cfg, )

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
        poff_ltrb = outs[:, :, s_:s_ + 4]  # 这个全是特图的距离 全rule 或 exp
        # goff_ltrb = gres[:, :, s_ + 1:s_ + 1 + 4]
        g_ltrb = gres[:, :, s_ + 1:s_ + 1 + 4]

        # 这里是解析归一化图  归一化与特图计算的IOU是一致的
        pboxes_ltrb = boxes_decode4fcos(self.cfg, poff_ltrb)
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


class KFcosHead(nn.Module):

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


class KFcosNet(nn.Module):
    def __init__(self, backbone, cfg, o_ceng, num_conv):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        channels_list = backbone.dims_out  # (512, 1024, 2048)
        self.fpn5 = FPN_out_v4(channels_list, o_ceng=o_ceng, num_conv=num_conv)
        self.head = KFcosHead(self.cfg, self.fpn5.dims_out)

    def forward(self, x):
        '''尺寸由大到小 维度由低到高 [2, 512, 40, 40] [2, 1024, 20, 20] [2, 2048, 10, 10] '''
        C_3, C_4, C_5 = self.backbone(x)
        ''' ...256维 + [2, 256, 5, 5] [2, 256, 3, 3] '''
        res = self.fpn5((C_3, C_4, C_5))
        res = self.head(res)
        return res


class KFcos(FModelBase):
    def __init__(self, backbone, cfg):
        if cfg.MODE_TRAIN == 1:  # 单人脸
            net = KFcosNet(backbone, cfg, o_ceng=4, num_conv=3)
            losser = FLoss(cfg)
            preder = FPredict(cfg)
        super(KFcos, self).__init__(net, losser, preder)


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
