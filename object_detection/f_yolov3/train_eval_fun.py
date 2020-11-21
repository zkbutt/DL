import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import FitBase
from f_tools.fun_od.f_boxes import pos_match, xywh2ltrb, fix_bbox, fix_keypoints, nms, batched_nms, boxes2yolo, \
    match4yolo3
from f_tools.pic.f_show import show_anc4ts
from object_detection.f_yolov3.CONFIG_YOLO3 import CFG


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

    targets_yolo = match4yolo3(targets, anchors_obj=anchors_obj,
                              nums_anc=cfg.NUMS_ANC,
                              num_class=cfg.NUM_CLASSES,
                              device=device)
    return images, targets_yolo


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
        images, targets_yolo = _preprocessing_data(batch_data, self.device,
                                              self.anc_obj, self.cfg)
        # -----------------------输入模型前的数据处理 完成------------------------
        # 模型输出 torch.Size([5, 10647, 25])
        out = self.model(images)

        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''

        '''---------------------与输出进行维度匹配及类别匹配  完成--------------------'''

        # ---------------损失计算 ----------------------
        # log_dict用于显示
        loss_total, log_dict = self.losser(out,targets_yolo)

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
