import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from f_pytorch.tools_model.fmodels.model_fpns import FPN
from f_pytorch.tools_model.fmodels.model_modules import SSH
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_predictfun import label_nms4keypoints
from f_tools.fits.f_lossfun import FocalLoss_v2, f_ohem, f_ohem_simpleness
from f_tools.fits.f_match import pos_match_retinaface
from f_tools.fits.fitting.f_fit_eval_base import SmoothedValue, MetricLogger
from f_tools.fun_od.f_anc import FAnchors

from f_tools.fun_od.f_boxes import fix_bbox, fix_keypoints, xywh2ltrb, diff_bbox, ltrb2xywh, diff_keypoints, calc_iou4ts
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import f_show_3box4pil, f_plt_box2


class LossRetinaface(nn.Module):

    def __init__(self, anc_xywh, loss_weight=(1., 1., 1.), neg_ratio=3, cfg=None):
        '''

        :param anc_xywh: torch.Size([1, 16800, 4])
        :param loss_weight:
        '''
        super().__init__()
        # self.location_loss = nn.SmoothL1Loss(reduction='none')
        # self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        # self.focal_loss = FocalLoss(torch.nn.BCELoss(reduction='sum'), gamma=2., alpha=0.25)
        self.focal_loss = FocalLoss_v2(alpha=cfg.FOCALLOSS_ALPHA, gamma=cfg.FOCALLOSS_GAMMA, reduction='none')
        # self.anc = anc.unsqueeze(dim=0)  # 这里加了一维
        self.anc_xywh = anc_xywh  # 这里加了一维
        self.loss_weight = loss_weight
        self.neg_ratio = neg_ratio
        self.cfg = cfg

    def forward(self, outs, targets, imgs_ts=None):
        '''
        归一化的box 和 anc
        :param p_bboxs_xywh: torch.Size([batch, 16800, 4])
        :param p_labels:
        :param p_keypoints: torch.Size([5, 16800, 1])
        :param gbboxs_ltrb: torch.Size([batch, 16800])
        :param glabels:
        :param gkeypoints:
        :return:
        '''
        cfg = self.cfg
        p_locs_box, p_labels_p_scores, p_locs_keypoint = outs
        p_labels = p_labels_p_scores[:, :, 1:]
        p_confs = p_labels_p_scores[:, :, 0]  # head中0conf 1label
        num_batch = p_locs_box.shape[0]
        num_ancs = p_locs_box.shape[1]
        device = p_locs_box.device
        gbboxs_ltrb = torch.zeros_like(p_locs_box, device=device)
        glabels = torch.empty((num_batch, num_ancs), device=device)  # 这里只存label值 1位
        gconfs = torch.zeros((num_batch, num_ancs), device=device, dtype=torch.float)
        mask_pos = torch.zeros((num_batch, num_ancs), device=device, dtype=torch.bool)
        # -------------这两个没用------------
        mask_neg = torch.zeros((num_batch, num_ancs), device=device, dtype=torch.bool)
        mash_ignore = torch.zeros((num_batch, num_ancs), device=device, dtype=torch.bool)
        if cfg.NUM_KEYPOINTS > 0:
            gkeypoints_xy = torch.zeros_like(p_locs_keypoint, device=device)
        else:
            gkeypoints_xy = None
        # 匹配
        for index in range(num_batch):
            g_bboxs_ltrb = targets[index]['boxes']  # torch.Size([batch, 4])
            g_labels = targets[index]['labels']  # torch.Size([batch])
            if cfg.NUM_KEYPOINTS > 0:
                g_keypoints_batch = targets[index]['keypoints']  # torch.Size([batch, 10])
            else:
                g_keypoints_batch = None
            res = pos_match_retinaface(xywh2ltrb(self.anc_xywh),
                                       g_bboxs_ltrb=g_bboxs_ltrb, g_labels=g_labels,
                                       g_keypoints=g_keypoints_batch,
                                       threshold_pos=self.cfg.THRESHOLD_CONF_POS,
                                       threshold_neg=self.cfg.THRESHOLD_CONF_NEG)
            match_bboxs, match_keypoints, match_labels, match_conf, mask_pos_, mask_neg_, mash_ignore_ = res

            # match_conf[mask_pos_] = torch.tensor(1, device=device, dtype=torch.float)
            # match_conf[torch.logical_or(mask_neg_,mash_ignore_)] = torch.tensor(0, device=device, dtype=torch.float)
            # match_conf[mash_ignore_] = torch.tensor(-1, device=device, dtype=torch.float)
            gconfs[index] = match_conf
            gbboxs_ltrb[index] = match_bboxs
            glabels[index] = match_labels
            if g_keypoints_batch is not None:
                gkeypoints_xy[index] = match_keypoints
            mask_pos[index] = mask_pos_
            mask_neg[index] = mask_neg_
            mash_ignore[index] = mash_ignore_

        # 匹配正例可视化
        if self.cfg is not None and self.cfg.IS_VISUAL:
            flog.debug('显示整个批次匹配的框 %s 个', torch.sum(mask_pos))
            _anc_xywh = self.anc_xywh.view(-1, 4).clone()
            for i, (img_ts, mask) in enumerate(zip(imgs_ts, mask_pos)):
                flog.debug('单个画匹配的框 %s 个', torch.sum(mask))
                img_ts = f_recover_normalization4ts(img_ts)
                _t = _anc_xywh[mask]
                n = 999
                img_pil = transforms.ToPILImage()(img_ts)
                f_plt_box2(img_pil, g_boxes_ltrb=targets[i]['boxes'].cpu(),
                           boxes1_ltrb=xywh2ltrb(_t[:n]).cpu(),
                           )

        # 每一个图片的正样本个数  [batch, 16800] ->[batch]
        num_pos = mask_pos.sum(dim=1)  # [batch] 这个用于batch中1个图没有正例不算损失和计算反例数

        '''-----------bboxs 损失处理-----------'''
        # [1, 16800, 4] ^^ [batch, 16800, 4] = [batch, 16800, 4]
        # p_bboxs_xywh_fix = fix_bbox(self.anc_xywh.unsqueeze(dim=0), p_locs_box)
        # loss_bboxs = F.smooth_l1_loss(p_bboxs_xywh_fix, gbboxs_ltrb, reduction='none').sum(dim=-1)
        # loss_bboxs = 1-calc_iou4ts(p_bboxs_xywh_fix, gbboxs_ltrb)

        # 这个损失要大得多
        d_bboxs = diff_bbox(self.anc_xywh.unsqueeze(dim=0), ltrb2xywh(gbboxs_ltrb))
        loss_bboxs = F.smooth_l1_loss(p_locs_box, d_bboxs, reduction='none').sum(dim=-1)

        # 正例框损失过滤
        loss_bboxs = (mask_pos.float() * loss_bboxs).sum(dim=1)

        '''-----------keypoints 损失处理-----------'''
        if gkeypoints_xy is not None:
            d_keypoints = diff_keypoints(self.anc_xywh, gkeypoints_xy)
            loss_keypoints = self.location_loss(p_locs_keypoint, d_keypoints).sum(dim=-1)
            # 全0的不计算损失 与正例的布尔索引取交集
            _mask = mask_pos * torch.all(gkeypoints_xy > 0, dim=2)  # and一下 将全0的剔除
            loss_keypoints = (_mask.float() * loss_keypoints).sum(dim=1)
        else:
            loss_keypoints = 0

        '''-----------labels 损失处理-----------'''
        # softmax 概率维已平均 (batch,1,anc) ^^ (batch,1) = (batch,1)
        loss_labels = F.cross_entropy(p_labels.permute(0, 2, 1), (glabels - 1).long(), reduction='none')
        # 正例损失过滤
        loss_labels = (mask_pos.float() * loss_labels).sum(dim=1)

        '''-----------conf 损失处理-----------'''
        # mssk_ = torch.logical_or(mask_pos, mask_neg)
        # loss_confs = (self.focal_loss(p_conf, gconfs) * mssk_).sum(-1)
        gconfs[mask_pos] = 1
        gconfs[torch.logical_or(mask_neg, mash_ignore)] = 0

        # loss_confs = self.focal_loss(p_confs, gconfs).sum(-1)

        '''------------------------conf 难例挖掘----------------------------------'''
        # (batch,anc)
        loss_confs = F.binary_cross_entropy_with_logits(p_confs, gconfs, reduction='none')

        # loss_confs = f_ohem_simpleness(loss_confs, self.neg_ratio * num_pos, mask_pos, mash_ignore)

        p_bboxs_xywh_fix = fix_bbox(self.anc_xywh.unsqueeze(dim=0), p_locs_box)
        loss_confs = f_ohem(scores=loss_confs,
                            nums_neg=cfg.NEG_RATIO * num_pos,
                            mask_pos=mask_pos,
                            mash_ignore=mash_ignore,
                            pboxes_ltrb=xywh2ltrb(p_bboxs_xywh_fix),
                            threshold_iou=0.7)

        '''-----------损失合并处理-----------'''
        # 求平均 排除有0的情况取类型最小值  pos_num是每一张图片的正例个数 eg. [15, 3, 5, 0]
        num_pos = num_pos.float().clamp(min=torch.finfo(torch.float).eps)

        # 每批损失/ 每批正例 /整批平均
        loss_bboxs = (loss_bboxs / num_pos).mean(dim=0) * self.loss_weight[0]
        loss_confs = (loss_confs / num_pos).mean(dim=0) * self.loss_weight[1]
        loss_labels = (loss_labels / num_pos).mean(dim=0) * self.loss_weight[2]
        loss_keypoints = (loss_keypoints / num_pos).mean(dim=0) * self.loss_weight[3]

        loss_total = loss_bboxs + loss_confs + loss_labels + loss_keypoints

        log_dict = {}
        log_dict['loss_total'] = loss_total.detach().item()
        log_dict['l_confs'] = loss_confs.item()
        log_dict['l_bboxs'] = loss_bboxs.item()
        log_dict['l_labels'] = loss_labels.item()
        log_dict['l_keypoints'] = loss_keypoints.item()

        detach = p_confs.clone().detach().sigmoid()
        log_dict['p_confs-max'] = detach.max().item()
        log_dict['p_confs-min'] = detach.min().item()
        log_dict['p_confs-mean'] = detach.mean().item()
        return loss_total, log_dict


class PredictRetinaface(nn.Module):
    def __init__(self, anchors, device, threshold_conf, threshold_nms, cfg=None):
        super(PredictRetinaface, self).__init__()
        self.cfg = cfg
        self.device = device
        self.anchors = anchors[None]  # (1,xx,4)
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf

    def forward(self, p_tuple, imgs_ts=None):
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
        cfg = self.cfg
        p_bboxs_xywh, p_labels_p_scores, p_keypoints = p_tuple

        p_labels = torch.softmax(p_labels_p_scores[:, :, 1:], dim=-1)
        p_scores = torch.sigmoid(p_labels_p_scores[:, :, 0])

        mask = p_scores >= self.threshold_conf
        # mask = torch.any(mask, dim=-1) # 修正维度
        if not torch.any(mask):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(p_scores.max().item(),
                                                                          p_scores.min().item(),
                                                                          p_scores.mean().item(),
                                                                          ))
            return [None] * 5

        '''第一阶段'''
        ids_batch1, _ = torch.where(mask)

        p_boxes = fix_bbox(self.anchors, p_bboxs_xywh)
        p_boxes_ltrb1 = xywh2ltrb(p_boxes[mask])

        if cfg.NUM_KEYPOINTS > 0:
            p_keypoints = fix_keypoints(self.anchors, p_keypoints)
            p_keypoints1 = p_keypoints[mask]
        else:
            p_keypoints1 = None
        _, p_labels = p_labels[mask].max(dim=-1)
        p_labels1 = p_labels + 1
        p_scores1 = p_scores[mask]

        # 分类 nms
        ids_batch2, p_boxes_ltrb2, p_keypoints2, p_labels2, p_scores2 = label_nms4keypoints(
            ids_batch1,
            p_boxes_ltrb1,
            p_keypoints1,
            p_labels1,
            p_scores1,
            self.device,
            self.threshold_nms,
        )
        return ids_batch2, p_boxes_ltrb2, p_keypoints2, p_labels2, p_scores2


class ClassHead(nn.Module):
    def __init__(self, inchannels, num_anchors, num_classes):
        '''
        包含conf  index0
        :param inchannels:
        :param num_anchors:
        :param num_classes:
        '''
        super(ClassHead, self).__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * (num_classes + 1), kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_classes + 1)


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
    def __init__(self, backbone, num_classes, anchor_num, in_channels_fpn, ssh_channel=256, cfg=None, device=None):
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

        if cfg.NUM_KEYPOINTS > 0:
            self.LandmarkHead = self._make_landmark_head(fpn_num=len(in_channels_fpn),
                                                         inchannels=ssh_channel,
                                                         anchor_num=anchor_num)

        anc_obj = FAnchors(cfg.IMAGE_SIZE, cfg.ANC_SCALE, cfg.FEATURE_MAP_STEPS,
                           anchors_clip=cfg.ANCHORS_CLIP, is_xymid=True, is_real_size=False,
                           device=device)

        self.losser = LossRetinaface(anc_obj.ancs, cfg.LOSS_WEIGHT, cfg.NEG_RATIO, cfg)

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

    def forward(self, x, targets=None):
        '''

        :param x: tensor(batch,c,h,w)
        :return:

        '''
        cfg = self.cfg
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

        if hasattr(self, 'LandmarkHead'):
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        else:
            ldm_regressions = None
        # if torchvision._is_tracing():  #预测模式 训练时是不会满足的 训练和预测进行不同的处理

        # 模型输出在这里
        outs = (bbox_regressions, classifications, ldm_regressions)

        if self.training:
            # 过渡处理
            # g_targets = self.data_packaging(outs, targets)
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad():  # 这个没用
            #     loss_total, log_dict = self.losser(outs, targets, x)
            # for k, v in log_dict.items():
            #     print(k, v)
            # metric_logger = MetricLogger(delimiter="  ")
            # metric_logger.update(**log_dict)

            ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores


if __name__ == '__main__':
    pass
