import torch
import torch.nn as nn

from f_pytorch.tools_model.fmodels.model_fpns import FPN
from f_pytorch.tools_model.fmodels.model_modules import SSH
from f_tools.GLOBAL_LOG import flog
from f_tools.f_predictfun import label_nms
from f_tools.fits.f_lossfun import LossRetinaface
from f_tools.fits.f_match import pos_match_retinaface
from f_tools.fun_od.f_anc import FAnchors
import numpy as np

from f_tools.fun_od.f_boxes import fix_bbox, fix_keypoints, xywh2ltrb


class PredictRetinaface(nn.Module):
    def __init__(self, anchors, device, threshold_conf=0.5, threshold_nms=0.3, cfg=None):
        super(PredictRetinaface, self).__init__()
        self.cfg = cfg
        self.device = device
        self.anchors = anchors[None]  # (1,xx,4)
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf

    def forward(self, p_tuple):
        '''
        批量处理 conf + nms
        :param p_tuple:
            p_loc: torch.Size([7, 16800, 4])
            p_conf: torch.Size([7, 16800, 1])
            p_landms: torch.Size([7, 16800, 10])
        :return:
            ids_batch2 [nn]
            p_boxes_ltrb2 [nn,4]
            p_labels2 [nn]
            p_scores2 [nn]
        '''
        p_loc, p_conf, p_landms = p_tuple
        p_scores = torch.sigmoid(p_conf)
        mask = p_scores >= self.threshold_conf
        if not torch.any(mask):  # 如果没有一个对象
            flog.error('该批次没有找到目标')
            return [None] * 4

        # p_boxes_xywh -> p_boxes_ltrb
        p_boxes = fix_bbox(self.anchors, p_loc)
        # xywh2ltrb(p_boxes, safe=False)
        p_keypoints = fix_keypoints(self.anchors, p_landms)

        '''第一阶段'''
        ids_batch1, _, _ = torch.where(mask)
        # 删除一维
        mask_box = mask.squeeze(-1)
        p_boxes_ltrb1 = xywh2ltrb(p_boxes[mask_box])
        p_scores1 = p_scores[mask]
        p_labels1 = torch.ones_like(p_scores1)

        # 分类 nms
        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    p_boxes_ltrb1,
                                                                    p_labels1,
                                                                    p_scores1,
                                                                    self.device,
                                                                    self.threshold_nms)

        return ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2


class ClassHead(nn.Module):
    def __init__(self, inchannels, num_anchors, num_classes):
        super(ClassHead, self).__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * num_classes, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_classes)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)  # 直接将特图  torch([batch, 64, 80, 80]) 拉成 num_anchors * 4
        out = out.permute(0, 2, 3, 1).contiguous()  # 将值放在最后
        return out.view(out.shape[0], -1, 4)  # 将 anc 拉平   num_anchors * 4 *每个层的h*w 叠加


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2, num_keypoint=10):
        super(LandmarkHead, self).__init__()
        self.num_keypoint = num_keypoint
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * num_keypoint, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_keypoint)


class RetinaFace(nn.Module):
    def __init__(self, backbone, num_classes, anchor_num, in_channels_fpn, ssh_channel=256, is_use_keypoint=False,
                 cfg=None, device=None):
        '''

        :param backbone:   这个的三层输出 in_channels_fpn
        :param return_layers: 定义 backbone 的输出名
        :param num_classes:
        :param anchor_num:
        :param in_channels_fpn: fpn的输入通道维度 数组对应输
        :param ssh_channel: FPN的输出 与SSH输出一致
        '''
        super(RetinaFace, self).__init__()
        self.backbone = backbone  # backbone转换
        self.fpn = FPN(in_channels_fpn, ssh_channel)
        self.ssh1 = SSH(ssh_channel, ssh_channel)
        self.ssh2 = SSH(ssh_channel, ssh_channel)
        self.ssh3 = SSH(ssh_channel, ssh_channel)

        self.ClassHead = self._make_class_head(fpn_num=len(in_channels_fpn),
                                               inchannels=ssh_channel,
                                               anchor_num=anchor_num,
                                               num_classes=num_classes)
        self.BboxHead = self._make_bbox_head(fpn_num=len(in_channels_fpn),
                                             inchannels=ssh_channel,
                                             anchor_num=anchor_num)

        self.use_keypoint = is_use_keypoint
        if is_use_keypoint:
            self.LandmarkHead = self._make_landmark_head(fpn_num=len(in_channels_fpn),
                                                         inchannels=ssh_channel,
                                                         anchor_num=anchor_num)

        anc_obj = FAnchors(cfg.IMAGE_SIZE, cfg.ANC_SCALE, cfg.FEATURE_MAP_STEPS,
                           anchors_clip=cfg.ANCHORS_CLIP, is_xymid=True, is_real_size=False,
                           device=device)

        if is_use_keypoint:
            losser = LossRetinaface(anc_obj.ancs, cfg.LOSS_WEIGHT, cfg.NEG_RATIO, cfg)
        else:
            losser = LossRetinaface(anc_obj.ancs, cfg.LOSS_WEIGHT, cfg.NEG_RATIO, cfg)
        self.losser = losser

        self.cfg = cfg
        self.anc_obj = anc_obj
        self.device = device

        self.preder = PredictRetinaface(anc_obj.ancs, device,
                                        cfg.THRESHOLD_PREDICT_CONF,
                                        cfg.THRESHOLD_PREDICT_NMS,
                                        cfg)

    def _make_class_head(self, fpn_num, inchannels, anchor_num, num_classes):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num, num_classes))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def data_packaging(self, outs, targets):
        '''

        :param outs:
        :param targets:
        :return:
        '''
        p_bboxs_xywh = outs[0]  # (batch,16800,4)  xywh  xywh
        p_labels = outs[1]  # (batch,16800,10)
        p_keypoints = outs[2]  # (batch,16800,2)
        num_batch = p_bboxs_xywh.shape[0]
        num_ancs = p_bboxs_xywh.shape[1]
        gbboxs_ltrb = torch.Tensor(*p_bboxs_xywh.shape).to(self.device)  # torch.Size([batch, 16800, 4])
        glabels = torch.Tensor(num_batch, num_ancs).to(self.device)  # 计算损失只会存在一维 无论多少类 标签只有一类
        gkeypoints = torch.Tensor(*p_keypoints.shape).to(self.device)  # 相当于empty
        for index in range(num_batch):
            g_bboxs = targets[index]['boxes']  # torch.Size([batch, 4])
            g_labels = targets[index]['labels']  # torch.Size([batch])
            g_keypoints = targets[index]['keypoints']  # torch.Size([batch, 10])
            match_bboxs, match_keypoints, match_labels = pos_match_retinaface(
                self.anc_obj.ancs, g_bboxs=g_bboxs,
                g_labels=g_labels,
                g_keypoints=g_keypoints,
                threshold_pos=self.cfg.THRESHOLD_PREDICT_CONF,
                threshold_neg=self.cfg.THRESHOLD_PREDICT_NMS
            )
            glabels[index] = match_labels
            gbboxs_ltrb[index] = match_bboxs
            gkeypoints[index] = match_keypoints

        return gbboxs_ltrb, glabels, gkeypoints

    def forward(self, x, targets=None):
        '''

        :param x: tensor(batch,c,h,w)
        :return:

        '''
        out = self.backbone(x)  # 输出字典 {1:层1输出,2:层2输出,3:层2输出}

        # FPN 输出 3个特图的统一维度(超参) tuple(tensor(层1),tensor(层2),tensor(层3))
        fpn = self.fpn(out)

        # SSH 串联 ssh
        feature1 = self.ssh1(fpn[0])  # in torch.Size([8, 64, 80, 80]) out一致
        feature2 = self.ssh2(fpn[1])  # in torch.Size([8, 128, 40, 40]) out一致
        feature3 = self.ssh3(fpn[2])  # in torch.Size([8, 256, 20, 20]) out一致
        features = [feature1, feature2, feature3]

        # 为每一输出的特图进行预测,输出进行连接 torch.Size([batch, 16800, 4])
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # torch.Size([batch, 8400, 2]) 这里可以优化成一个值
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # torch.Size([batch, 16800, 10])

        if self.use_keypoint:
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        else:
            ldm_regressions = None
        # if torchvision._is_tracing():  #预测模式 训练时是不会满足的 训练和预测进行不同的处理

        # 模型输出在这里
        outs = (bbox_regressions, classifications, ldm_regressions)

        if self.training:
            # 过渡处理
            g_targets = self.data_packaging(outs, targets)
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, g_targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_labels, p_scores = self.preder(outs)
            return ids_batch, p_boxes_ltrb, p_labels, p_scores


if __name__ == '__main__':
    pass
