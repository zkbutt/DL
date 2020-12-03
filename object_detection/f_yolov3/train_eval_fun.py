import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import FitBase
from f_tools.fits.f_match import match4yolo3
from f_tools.fun_od.f_boxes import xywh2ltrb, fix_bbox, nms, batched_nms, fix_boxes4yolo3
from f_tools.pic.enhance.data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import show_anc4ts, f_show_od4pil, f_show_od4pil_yolo
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

    # targets = match4yolo3(targets, anchors_obj=anchors_obj,
    #                       nums_anc=cfg.NUMS_ANC,
    #                       num_class=cfg.NUM_CLASSES,
    #                       device=device,
    #                       imgs_ts=images)
    # for target in targets:
    #     target['boxes'] = target['boxes'].to(device)
    #     target['labels'] = target['labels'].to(device)
    #     target['height_width'] = target['height_width'].to(device)
    return images, targets


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
        # p_bboxs_xywh = out[:, :, :4]  # (batch,16800,4)  xywh  xywh
        # p_conf = out[:, :, 4:5]
        # p_labels = out[:, :, 5:]  # (batch,16800,10)

        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        # num_batch = images.shape[0]
        # num_ancs = self.anc_obj.ancs.shape[0]
        # g_yolo = torch.zeros(num_batch, num_ancs, 5 + self.cfg.NUM_CLASSES)

        '''---------------------与输出进行维度匹配及类别匹配  完成--------------------'''
        # for index in range(num_batch):
        #     g_bboxs = targets[index]['boxes']
        #     g_labels = targets[index]['labels']
        #     g_labels = labels2onehot4ts(g_labels - 1, self.cfg.NUM_CLASSES)
        #
        #     '''
        #     将g_bboxs 进行重构至与 anc的结果一致,并在 g_labels 根据IOU超参,上进行正反例标注
        #     label_neg_mask: 反例的  布尔 torch.Size([16800]) 一维
        #     anc_bbox_ind : 正例对应的 g_bbox 的index  torch.Size([16800]) 一维
        #     '''
        #     pos_ancs_index, pos_bboxs_index, mask_neg, mask_ignore = pos_match4yolo(self.anc_obj.ancs,
        #                                                                             g_bboxs,
        #                                                                             self.cfg.NEG_IOU_THRESHOLD)
        #
        #     match_bboxs = g_bboxs[pos_bboxs_index]
        #     match_labels = g_labels[pos_bboxs_index]
        #     match_labels[label_neg_mask] = 0.
        #
        #     g_yolo[index] = match_labels
        #     g_bboxs_ltrb[index] = match_bboxs

        # ---------------损失计算 ----------------------
        # log_dict用于显示
        loss_total, log_dict = self.losser(out, targets, images)

        # -----------------构建展示字典及返回值------------------------
        # 多GPU时结果处理 reduce_dict 方法
        # losses_dict_reduced = reduce_dict(losses_dict)
        show_dict = {"loss_total": loss_total.detach().item(), }
        show_dict.update(**log_dict)
        return loss_total, show_dict


class PredictHandler(FitBase):
    def __init__(self, model, device, anc_obj,
                 predict_conf_threshold, predict_nms_threshold):
        super(PredictHandler, self).__init__(model, device)
        self.predict_conf_threshold = predict_conf_threshold
        self.predict_nms_threshold = predict_nms_threshold
        self.anc_obj = anc_obj

    @torch.no_grad()
    def predicting4one(self, img_ts4, img_pil=None, idx_to_class=None):
        '''
        相比 forward 没有预处理
        :param img_ts4: torch.Size([batch, 3, 416, 416])
        :param whwh:原图片的比例 torch.Size([batch, 4])
        :return:
            {0: [[81, 125, 277, 296, 14], [77, 43, 286, 250, 14]]}
        '''
        # ------模型输出处理-------
        p_yolos = self.model(img_ts4)  # torch.isnan(p_yolo).any()
        batch = p_yolos.shape[0]

        feature_sizes = np.array(self.anc_obj.feature_sizes)
        num_ceng = feature_sizes.prod(axis=1)
        # 匹配完成的数据
        _findex = np.arange(len(num_ceng)).repeat(num_ceng * 3)
        fsizes = np.array(self.anc_obj.feature_sizes)[_findex, 0]

        for i in range(batch):
            # img_ts = f_recover_normalization4ts(img_ts4[i])
            p_yolo = p_yolos[i]  # batch,10647,25
            p_box_xy = p_yolo[:, :2] / torch.tensor(fsizes)[:, None] + self.anc_obj.ancs[:, :2]
            p_box_wh = p_yolo[:, 2:4].exp() * self.anc_obj.ancs[:, 2:]
            p_box_xywh = torch.cat([p_box_xy, p_box_wh], dim=-1)
            mask = p_yolo[:, 4] > self.predict_conf_threshold  # 一维索引
            p_box_ltrb = xywh2ltrb(p_box_xywh[mask])
            flog.debug('predict_conf_threshold过滤后 %s 个', p_box_ltrb.shape[0])
            if p_box_ltrb.shape[0] == 0:
                continue
            # show_anc4ts(img_ts, p_box_ltrb, size)
            _p_box_ltrb = p_box_ltrb * torch.tensor(img_pil.size).repeat(2)
            p_yolo_ltrb = torch.cat([_p_box_ltrb, p_yolo[mask][:, 4:5], p_yolo[mask][:, 5:]], dim=-1)
            # f_show_od4pil_yolo(img_pil, p_yolo_ltrb, id_to_class=idx_to_class)

            p_scores = p_yolo[mask][:, 4]
            keep = nms(p_box_ltrb, p_scores, self.predict_nms_threshold)
            flog.debug('predict_nms_threshold 过滤后有 %s 个', len(keep))
            p_yolo_ltrb = p_yolo_ltrb[keep]
            f_show_od4pil_yolo(img_pil, p_yolo_ltrb, id_to_class=idx_to_class)

    def forward(self, batch_data):
        # ------数据处理-------
        images, _ = _preprocessing_data(batch_data, self.device)
        return self.predicting4one(images)

    @torch.no_grad()
    def handler_map_dt_txt(self, batch_data, path_dt_info, idx_to_class):
        # -----------------------输入模型前的数据处理 开始------------------------
        images, g_yolo = _preprocessing_data(batch_data, self.device, self.grid, self.num_classes)
        # -----------------------输入模型前的数据处理 完成------------------------
        '''
        模型输出 torch.Size([batch, 13, 13, 25])
        '''
        p_yolo = self.model(images)  # torch.isnan(p_yolo).any()

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
        p_boxes = fix_bbox(self.anc_obj, p_loc)
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
