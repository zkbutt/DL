import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import FitBase
from f_tools.fun_od.f_boxes import pos_match, xywh2ltrb, fix_bbox, fix_keypoints, nms, batched_nms, boxes2yolo, \
    match4yolo3
from f_tools.pic.f_show import show_anc4ts
from object_detection.f_yolov3.CONFIG_YOLO3 import CFG
import numpy as np


def _preprocessing_data(batch_data, device, anchors_obj, cfg):
    '''
    使用GPU处理
    :param batch_data:
    :param device:
    :param feature_sizes:
    :param cfg:
    :return:
        torch.Size([5, 10647, 25])
    '''
    images, targets = batch_data
    images = images.to(device)

    target_yolo = match4yolo3(targets, anchors_obj=anchors_obj,num_anc=cfg.NUMS_ANC, num_class=cfg.NUM_CLASSES, device=device)
    return images, target_yolo


class LossHandler(FitBase):
    '''
    前向和反向 loss过程函数
    '''

    def __init__(self, model, device, anc_obj, losser, cfg):
        super(LossHandler, self).__init__(model, device)
        self.anc_obj = anc_obj
        self.losser = losser
        self.cfg = cfg

    def forward(self, batch_data):
        # -----------------------输入模型前的数据处理 开始------------------------
        # targets torch.Size([5, 10647, 26])
        images, targets = _preprocessing_data(batch_data, self.device,
                                              self.anc_obj, self.cfg)
        # -----------------------输入模型前的数据处理 完成------------------------
        # 模型输出 torch.Size([5, 10647, 25])
        out = self.model(images)

        '''-----------------------------寻找正例匹配-----------------------'''
        # ------------修复每批的 type_index 使批的GT id都不一样------------------
        batch = len(targets)
        batch_len = targets.shape[1]
        # 使用 300 作偏移  一张图不会超过 300 个GT
        batch_offset = 300 * np.array([range(batch)], dtype=np.float32)
        # 单体复制 构造同维  5 -> 00..11..22.. -> batch,10647
        batch_offset_index = batch_offset.repeat(batch_len).reshape(batch, batch_len, 1)
        mask_match = targets[:, :, -1:] > 0
        targets[:, :, -1:][mask_match] = targets[:, :, -1:][mask_match] + batch_offset_index[mask_match]

        # 计算IOU

        p_bboxs_xywh = out[:, :, :4]  # torch.Size([5, 10647, 4])
        # sigmoid = torch.sigmoid(out[:, :, 4:])
        # p_conf = sigmoid[:, :, 0]
        # p_cls = sigmoid[:, :, 1:]
        p_conf = out[:, :, 5]  # torch.Size([5, 10647]) 预测的iou值
        p_cls = out[:, :, 5:]  # torch.Size([5, 10647, 20])

        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        num_batch = images.shape[0]
        num_ancs = self.anc_obj.ancs.shape[0]

        mg_bboxs_ltrb = torch.Tensor(*p_bboxs_xywh.shape).to(images)
        mg_labels = torch.Tensor(num_batch, num_ancs, device=images.device).type(torch.int64)  # 计算损失只会存在一维 无论多少类 标签只有一类
        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        for index in range(num_batch):
            g_bboxs = targets[index]['boxes']  # torch.Size([batch, 4])
            g_labels = targets[index]['labels']  # torch.Size([batch])

            '''
            将g_bboxs 进行重构至与 anc的结果一致,并在 g_labels 根据IOU超参,上进行正反例标注
            label_neg_mask: 反例的  布尔 torch.Size([16800]) 一维
            anc_bbox_ind : 正例对应的 g_bbox 的index  torch.Size([16800]) 一维
            '''
            label_neg_mask, anc_bbox_ind = pos_match(self.anc_obj, g_bboxs, self.neg_iou_threshold)

            # new_anchors = anchors.clone().detach() # 深复制
            # 将bbox取出替换anc对应位置 ,根据 bboxs 索引list 将bboxs取出与anc 形成维度对齐 便于后面与anc修复算最终偏差 ->[anc个,4]
            # 只计算正样本的定位损失,将正例对应到bbox标签 用于后续计算anc与bbox的差距
            match_bboxs = g_bboxs[anc_bbox_ind]

            # 构建正反例label 使原有label保持不变
            # labels = torch.zeros(num_ancs, dtype=torch.int64)  # 标签默认为同维 类别0为负样本
            # 正例保持原样 反例置0
            match_labels = g_labels[anc_bbox_ind]
            # match_labels[label_neg_mask] = torch.tensor(0).to(labels)
            match_labels[label_neg_mask] = 0
            mg_labels[index] = match_labels

            mg_bboxs_ltrb[index] = match_bboxs  # torch.Size([5, 10647])
        '''---------------------与输出进行维度匹配及类别匹配  完成-------------------------'''

        # ---------------损失计算 ----------------------
        # log_dict用于显示
        loss_total, log_dict = self.losser(p_bboxs_xywh, mg_bboxs_ltrb, p_cls, mg_labels, imgs_ts=images)

        # -----------------构建展示字典及返回值------------------------
        # 多GPU时结果处理 reduce_dict 方法
        # losses_dict_reduced = reduce_dict(losses_dict)
        show_dict = {"loss_total": loss_total.detach().item(), }
        show_dict.update(**log_dict)
        return loss_total, show_dict


class PredictHandler(FitBase):
    def __init__(self, model, device, anchors, threshold_conf=0.5, threshold_nms=0.3):
        super(PredictHandler, self).__init__(model, device)
        self.anchors = anchors[None]  # (1,xx,4)
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf

    def output_res(self, p_boxes, p_keypoints, p_scores, img_ts4=None):
        '''
        已修复的框 点 和对应的分数
            1. 经分数过滤
            2. 经NMS 出最终结果
        :param p_boxes:
        :param p_keypoints:
        :param p_scores:
        :param img_ts4: 用于显示
        :param idxs_img: 用于识别多个
        :return:
        '''
        # (batch,xx) -> (batch.xx)
        mask = p_scores >= self.threshold_conf
        # (batch,xx,4) -> (3,4) 拉伸降维
        p_boxes = p_boxes[mask]
        p_scores = p_scores[mask]
        p_keypoints = p_keypoints[mask]

        if p_scores.shape[0] == 0:
            flog.error('threshold_conf 过滤后 没有目标 %s', self.threshold_conf)
            return None, None, None

        if img_ts4 is not None:
            if CFG.IS_VISUAL:
                flog.debug('过滤后 threshold_conf %s', )
                # h, w = img_ts4.shape[-2:]
                show_anc4ts(img_ts4.squeeze(0), p_boxes, CFG.IMAGE_SIZE)

        # flog.debug('threshold_conf 过滤后有 %s 个', p_scores.shape[0])
        # 2 . 根据得分对框进行从大到小排序。
        # keep = batched_nms(p_boxes, p_scores, idxs_img, self.threshold_nms)
        keep = nms(p_boxes, p_scores, self.threshold_nms)
        # flog.debug('threshold_nms 过滤后有 %s 个', len(keep))
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]
        p_keypoints = p_keypoints[keep]
        return p_boxes, p_keypoints, p_scores

    @torch.no_grad()
    def predicting4one(self, img_ts4):
        '''
        相比 forward 没有预处理
        :param img_ts4:
        :return:
        '''
        # ------模型输出处理-------
        # (batch,xx,4)
        p_loc, p_conf, p_landms = self.model(img_ts4)
        # (batch,xx,1)->(batch.xx)
        p_scores = torch.nn.functional.softmax(p_conf, dim=-1)
        p_scores = p_scores[:, :, 1]
        # p_scores = p_scores.data.squeeze(0)
        # p_loc = p_loc.data.squeeze(0)
        # p_landms = p_landms.squeeze(0)

        # ---修复--并转换 xywh --> ltrb--variances = (0.1, 0.2)
        p_boxes = fix_bbox(self.anchors, p_loc)
        xywh2ltrb(p_boxes, safe=False)
        p_keypoints = fix_keypoints(self.anchors, p_landms)

        p_boxes, p_keypoints, p_scores = self.output_res(p_boxes, p_keypoints, p_scores, img_ts4)
        '''
        xx是最终框 (xx,4)  (xx,10)  (xx)
        '''
        return p_boxes, p_keypoints, p_scores

    def forward(self, batch_data):
        # ------数据处理-------
        images, _ = _preprocessing_data(batch_data, self.device)
        return self.predicting4one(images)

    @torch.no_grad()
    def handler_map_dt_txt(self, batch_data, path_dt_info, idx_to_class):
        # (batch,3,640,640)   list(batch{'size','boxes','labels'}) 转换到GPU设备
        images, targets = _preprocessing_data(batch_data, self.device, mode='bbox')
        sizes = []
        files_txt = []
        for target in targets:
            sizes.append(target['size'])
            files_txt.append(os.path.join(path_dt_info, target['name_txt']))

        idxs, p_boxes, p_labels, p_scores = self.predicting4many(images)

        for i, (szie, file_txt) in enumerate(zip(sizes, files_txt)):
            mask = idxs == i
            if torch.any(mask):
                lines_write = []
                for label, score, bbox in zip(p_labels[mask], p_scores[mask], p_boxes[mask]):
                    _bbox = [str(i.item()) for i in list((bbox * szie.repeat(2)).type(torch.int64).data)]
                    bbox_str = ' '.join(_bbox)
                    _line = idx_to_class[label.item()] + ' ' + str(score.item()) + ' ' + bbox_str + '\n'
                    lines_write.append(_line)
                with open(file_txt, "w") as f:
                    f.writelines(lines_write)
            else:
                # flog.warning('没有预测出框 %s', files_txt)
                pass
        return p_labels, p_scores, p_boxes, sizes, idxs

    def predicting4many(self, images):
        # (batch,xx,4)
        p_loc, p_conf, p_landms = self.model(images)
        # (batch,xx,1)->(batch.xx)
        p_scores = torch.nn.functional.softmax(p_conf, dim=-1)
        p_scores = p_scores[:, :, 1]
        # ---修复----variances = (0.1, 0.2)
        p_boxes = fix_bbox(self.anchors, p_loc)
        xywh2ltrb(p_boxes, safe=False)
        # (batch个) -> (batch,1) -> (batch, xx) 从1开始有利于批量nms
        idxs = torch.arange(1, p_boxes.shape[0] + 1, device=self.device)
        idxs = idxs.view(-1, 1).repeat(1, p_boxes.shape[1])
        # (batch,xx) -> (batch.xx)
        mask = p_scores >= self.threshold_conf  # 这里过滤有可能导致没得
        # (batch,xx,4) -> (3,4) 拉伸降维
        p_boxes = p_boxes[mask]
        p_scores = p_scores[mask]
        idxs = idxs[mask]
        keep = batched_nms(p_boxes, p_scores, idxs, self.threshold_nms)
        # flog.debug('threshold_nms 过滤后有 %s 个', len(keep))
        p_labels = torch.ones(len(keep), dtype=torch.int64).to(p_boxes.device)
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]
        idxs = idxs[keep] - 1  # 为batched_nms +1 恢复
        return idxs, p_boxes, p_labels, p_scores

    def to_map_res(self, p_labels, p_scores, p_boxes, sizes, idxs):
        '''
        tvmonitor 0.471781 0 13 174 244
        cup 0.414941 274 226 301 265
        :param p_labels:
        :param p_scores:
        :param p_boxes:
        :param sizes:
        :param idxs:
        :param idx_to_class:
        :return:
        '''
        idx_to_class = {}
        for i, szie in enumerate(sizes):
            mask = idxs == i
            if torch.any(mask):
                lines_write = []
                for label, score, bbox in zip(p_labels[mask], p_scores[mask], p_boxes[mask]):
                    _bbox = [str(i.item()) for i in list((bbox * szie.repeat(2)).type(torch.int64).data)]
                    bbox_str = ' '.join(_bbox)
                    line = idx_to_class[label.item()] + ' ' + str(score.item()) + ' ' + bbox_str
                with open(file_txt, "w") as f:
                    f.writelines(lines_write)
