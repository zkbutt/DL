import torch
from torch import nn
import torch.nn.functional as F

from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_match import matchs_gt_b, boxes_encode4ssd, boxes_decode4ssd
from f_tools.fits.f_predictfun import label_nms
from f_tools.fits.fitting.f_fit_class_base import Predicting_Base
from f_tools.floss.f_lossfun import x_bce, f_ohem
from f_tools.fun_od.f_anc import cre_ssd_ancs, FAnchors
from f_tools.fun_od.f_boxes import xywh2ltrb, bbox_iou4one_2d, calc_iou4ts, bbox_iou4y, xywh2ltrb4ts, ltrb2xywh
from f_tools.pic.f_show import f_show_od_np4plt
from f_tools.yufa.x_calc_adv import f_mershgrid


class LossSSD(nn.Module):

    def __init__(self, cfg, anc_obj):
        super(LossSSD, self).__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj

    def forward(self, pssd, targets, imgs_ts=None):
        '''

        :param pssd: preg, pcls = class+1 [2, 4, 8732]
        :param targets:
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        # pcls  classes+1 [b, 4, 8732]
        preg, pcls = pssd
        preg = preg.permute(0, 2, 1)  # [b, 4, 8732] -> [b, 8732, 4]
        device = preg.device
        batch, hw, c = preg.shape

        # cls_val-1, gltrb-4
        gdim = 1 + 4
        gssd = torch.empty((batch, hw, gdim), device=device)  # 每批会整体更新这里不需要赋0

        for i, target in enumerate(targets):  # batch遍历
            gboxes_ltrb_b = target['boxes']  # ltrb
            glabels_b = target['labels']

            boxes_index, mask_pos_b, mask_neg_b, mash_ignore_b = matchs_gt_b(cfg,
                                                                             gboxes_ltrb_b=gboxes_ltrb_b,
                                                                             glabels_b=glabels_b,
                                                                             anc_obj=self.anc_obj,
                                                                             mode='iou', ptxywh_b=preg[i],
                                                                             img_ts=imgs_ts[i],
                                                                             num_atss_topk=9)

            '''正反例设置 正例才匹配'''
            gssd[i][:, 0] = 0  # 是背景 这个要计算正反例 故需全fill 0
            gssd[i][mask_pos_b, 0] = glabels_b[boxes_index][mask_pos_b]
            gssd[i][mask_pos_b, 1:1 + 4] = gboxes_ltrb_b[boxes_index][mask_pos_b]  # gltrb-4

            '''可视化验证'''
            if cfg.IS_VISUAL:
                gssd_test = gssd[i].clone()
                # gssd_test = gssd_test.view(-1, gdim)
                gconf_one = gssd_test[:, 0]
                mask_pos_2d = gconf_one > 0
                flog.debug('mask_pos_2d 个数%s', mask_pos_2d.sum())
                # torch.Size([169, 4])
                anc_ltrb_pos = xywh2ltrb(self.anc_obj.ancs_xywh[mask_pos_2d])

                from f_tools.pic.enhance.f_data_pretreatment4pil import f_recover_normalization4ts
                img_ts = f_recover_normalization4ts(imgs_ts[i])
                from torchvision.transforms import functional as transformsF
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                import numpy as np
                img_np = np.array(img_pil)
                f_show_od_np4plt(img_np,
                                 gboxes_ltrb=gboxes_ltrb_b.cpu(),
                                 pboxes_ltrb=anc_ltrb_pos.cpu(),  # ltrb
                                 # other_ltrb=xywh2ltrb(self.anc_obj.ancs_xywh)[:100],
                                 is_recover_size=True,
                                 # grids=(h, w)
                                 )

        # cls_val-1, gltrb-4
        glabel = gssd[:, :, 0]  # 0为背景
        mask_pos_2d = glabel > 0
        nums_pos = (mask_pos_2d.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)
        # mask_neg_2d = glabel == 0 反例没用上
        # nums_neg = (mask_neg.sum(-1).to(torch.float)).clamp(min=torch.finfo(torch.float16).eps)

        ''' ---------------- 回归损失   ----------------- '''
        gboxes_ltrb_m_pos = gssd[:, :, 1:1 + 4][mask_pos_2d]
        ancs_xywh_m_pos = self.anc_obj.ancs_xywh.unsqueeze(0).repeat(batch, 1, 1)[mask_pos_2d]
        gtxywh_pos = boxes_encode4ssd(cfg, ancs_xywh_m_pos, ltrb2xywh(gboxes_ltrb_m_pos))
        _loss_val = F.smooth_l1_loss(preg[mask_pos_2d], gtxywh_pos, reduction="none")
        l_box = _loss_val.sum(-1).mean()

        ''' ---------------- 类别-cls损失 ---------------- '''
        # 自带softmax
        _loss_val = F.cross_entropy(pcls, glabel.long(), reduction="none")
        mask_neg_hard = f_ohem(_loss_val, nums_pos * 3, mask_pos=mask_pos_2d)
        l_conf_pos = ((_loss_val * mask_pos_2d).sum(-1) / nums_pos).mean()  # 正例越多反例越多
        l_conf_neg = ((_loss_val * mask_neg_hard).sum(-1) / nums_pos).mean()

        log_dict = {}

        ''' ---------------- loss完成 ----------------- '''
        l_total = l_box + l_conf_pos + l_conf_neg

        log_dict['l_total'] = l_total.item()
        log_dict['l_conf_pos'] = l_conf_pos.item()
        log_dict['l_conf_neg'] = l_conf_neg.item()
        log_dict['l_box'] = l_box.item()

        # log_dict['p_max'] = pcls.max().item()
        # log_dict['p_min'] = pcls.min().item()
        # log_dict['p_mean'] = pcls.mean().item()
        return l_total, log_dict


class PredictSSD(Predicting_Base):
    def __init__(self, cfg, anc_obj):
        super(PredictSSD, self).__init__(cfg)
        self.anc_obj = anc_obj

    def p_init(self, pssd):
        preg, pcls = pssd
        return pssd, preg.device

    def get_pscores(self, pssd):
        preg, pcls = pssd

        pcls = pcls.permute(0, 2, 1)
        # torch.Size([5, 8732])
        pscores, plabels = F.softmax(pcls, dim=-1).max(-1)
        pscores[plabels == 0] = 0  # 这里有背景 把背景置0
        return pscores, plabels, pscores

    def get_stage_res(self, pssd, mask_pos, pscores, plabels):
        ids_batch1, _ = torch.where(mask_pos)

        preg, pcls = pssd
        batch, c, hw = preg.shape
        preg = preg.permute(0, 2, 1)  # 回归参数

        ptxywh_3d = preg
        ''' 预测 这里是修复是 xywh'''
        # 装入GPU
        ancs_xywh_3d = self.anc_obj.ancs_xywh.unsqueeze(0).repeat(batch, 1, 1)
        gxywh = boxes_decode4ssd(self.cfg, ptxywh_3d, ancs_xywh_3d)
        pboxes_ltrb1 = xywh2ltrb(gxywh[mask_pos])
        pboxes_ltrb1.clamp_(min=0., max=1.)  # 预测不返回

        pscores1 = pscores[mask_pos]
        plabels1 = plabels[mask_pos]
        plabels1 = plabels1  # 这里不是用的bce 不需要再加1

        return ids_batch1, pboxes_ltrb1, plabels1, pscores1


class SSD_Net(nn.Module):
    def __init__(self, backbone, num_classes, cfg):
        super(SSD_Net, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone

        out_channels = [1024, 512, 512, 256, 256, 256]
        _additional_blocks = self._build_additional_features(out_channels)
        self.additional_blocks = nn.ModuleList(_additional_blocks)

        num_defaults = [4, 6, 6, 6, 4, 4]

        location_extractors = []
        confidence_extractors = []
        for nd, oc in zip(num_defaults, out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)

        self._init_weights()

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def _build_additional_features(self, input_size):
        """
        为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
        :param input_size: [1024, 512, 512, 256, 256, 256]
        :return:
        """
        additional_blocks = []
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        return additional_blocks

    def forward(self, x, targets=None):
        x = self.backbone(x)  # torch.Size([1, 1024, 38, 38])
        detection_features = []  # torch.Size([1, 2048, 10, 10])
        detection_features.append(x)  # 这里形成6层
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        preg = []
        pcls = []
        for f, l, c in zip(detection_features, self.loc, self.conf):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            preg.append(l(f).view(f.size(0), 4, -1))
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            pcls.append(c(f).view(f.size(0), self.num_classes, -1))

        # 最后一维堆叠 [1, 4, 8732]  [1, 3, 8732]
        preg, pcls = torch.cat(preg, 2).contiguous(), torch.cat(pcls, 2).contiguous()

        return preg, pcls


class SSD(nn.Module):
    def __init__(self, backbone, cfg, device):
        super(SSD, self).__init__()
        self.net = SSD_Net(backbone, cfg.NUM_CLASSES + 1, cfg)  # 带conf

        ancs_scale = []
        s = 0
        for num in cfg.NUMS_ANC:
            ancs_scale.append(cfg.ANCS_SCALE[s:s + num])
            s += num
        # self.anc_obj = FAnchors(cfg.IMAGE_SIZE, ancs_scale, feature_sizes=cfg.FEATURE_SIZES,
        #                         anchors_clip=True, is_xymid=True, is_real_size=False,
        #                         device=device)
        self.anc_obj = cre_ssd_ancs(device)

        self.losser = LossSSD(cfg=cfg, anc_obj=self.anc_obj)
        self.preder = PredictSSD(cfg=cfg, anc_obj=self.anc_obj)

    def forward(self, x, targets=None):
        outs = self.net(x)
        # return outs

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


class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        from torchvision import models
        net = models.resnet50(pretrained=True)  # 只存 1~4
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        # 只取7层
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        # 修改conv4_block1的步距，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


if __name__ == '__main__':
    from torchvision import models
    from f_pytorch.tools_model.f_layer_get import ModelOut4Resnet18, ModelOut4Resnet50

    # model = models.resnet50(pretrained=True)
    # model = ModelOut4Resnet50(model)
    model = Backbone()


    class CFG:
        pass


    cfg = CFG()
    cfg.NUM_CLASSES = 3
    cfg.NUMS_ANC = [4, 6, 6, 6, 4, 4]
    cfg.ANCS_SCALE = [[0.028, 0.034],
                      [0.028, 0.064],
                      [0.044, 0.05],
                      [0.062, 0.04],
                      [0.046, 0.072],
                      [0.038, 0.108],
                      [0.07, 0.078],
                      [0.106, 0.052],
                      [0.054, 0.158],
                      [0.084, 0.114],
                      [0.146, 0.092],
                      [0.078, 0.21],
                      [0.118, 0.154],
                      [0.15, 0.204],
                      [0.106, 0.294],
                      [0.242, 0.13],
                      [0.212, 0.234],
                      [0.162, 0.382],
                      [0.422, 0.198],
                      [0.298, 0.288],
                      [0.23, 0.412],
                      [0.47, 0.31],
                      [0.34, 0.444],
                      [0.274, 0.62187],
                      [0.792, 0.268],
                      [0.57806, 0.42],
                      [0.432, 0.6],
                      [0.843, 0.49],
                      [0.64, 0.732],
                      [0.942, 0.662], ]
    # net = SSD_Net(backbone=model, num_classes=cfg.NUM_CLASSES, cfg=cfg)
    model = SSD(backbone=model, cfg=cfg, device=torch.device('cpu'))
    from f_pytorch.tools_model.model_look import f_look_tw

    f_look_tw(model, input=(1, 3, 300, 300), name='SSD_Net')
