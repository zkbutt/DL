import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_boxes import pos_match, xywh2ltrb, fix_bbox, fix_keypoints, nms, batched_nms, ltrb2ltwh, \
    pos_match_retinaface
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import show_bbox_keypoints4ts, show_bbox4ts, show_anc4ts


def _preprocessing_data(batch_data, device, mode='keypoints'):
    '''
    cpu转gpu 输入模型前数据处理方法 定制
    :param batch_data:
    :param device:
    :return:
    '''
    images, targets = batch_data
    images = images.to(device)
    for target in targets:
        target['boxes'] = target['boxes'].to(device)
        target['labels'] = target['labels'].to(device)
        target['size'] = target['size'].to(device)
        if mode == 'keypoints':
            target['keypoints'] = target['keypoints'].to(device)

        # for key, val in target.items():
        #     target[key] = val.to(device)
    return images, targets


class FitBase(torch.nn.Module):
    def __init__(self, model, device):
        super(FitBase, self).__init__()
        self.model = model
        self.device = device


class LossHandler(FitBase):
    '''
    前向和反向 loss过程函数
    '''

    def __init__(self, model, device, anchors, losser, neg_iou_threshold):
        super(LossHandler, self).__init__(model, device)
        self.anchors = anchors  # 归一化的ancs
        self.losser = losser
        self.neg_iou_threshold = neg_iou_threshold

    def forward(self, batch_data):
        '''

        :param batch_data: tuple(images,targets)
            images:tensor(batch,c,h,w)
            list( # batch个
                target: dict{
                        image_id: int,
                        bboxs: np(num_anns, 4), ltrb ltrb ltrb ltrb
                        labels: np(num_anns),
                        keypoints: np(num_anns,10),
                    }
                )
        :return:
            loss_total: 这个用于优化
            show_dict:  log_dict
        '''
        # -----------------------输入模型前的数据处理 开始------------------------
        images, targets = _preprocessing_data(batch_data, self.device)
        cfg = self.model.cfg
        # if cfg.IS_VISUAL:
        #     flog.debug('查看模型输入和anc %s', )
        #     for img_ts, target in zip(images, targets):
        #         # 遍历降维
        #         _t = torch.cat([target['boxes'], target['keypoints']], dim=1)
        #         _t[:, ::2] = _t[:, ::2] * cfg.IMAGE_SIZE[0]
        #         _t[:, 1::2] = _t[:, 1::2] * cfg.IMAGE_SIZE[1]
        #         img_ts = f_recover_normalization4ts(img_ts)
        #         # show_bbox_keypoints4ts(img_ts, _t[:, :4], _t[:, 4:14], target['labels'])
        #
        #         whwh = torch.tensor(cfg.IMAGE_SIZE).repeat(2)[None]
        #         _t = self.anchors.view(-1, 4).clone() * whwh
        #         # _t[:, ::2] = _t[:, ::2] * cfg.IMAGE_SIZE[0]
        #         # _t[:, 1::2] = _t[:, 1::2] * cfg.IMAGE_SIZE[1]
        #         show_bbox4ts(img_ts, xywh2ltrb(_t)[-200:-195, :], torch.ones(200))

        # -----------------------输入模型前的数据处理 完成------------------------

        '''
           模型输出 tuple(# 预测器 框4 类别2 关键点10
               torch.Size([batch, 16800, 4]) # 框
               torch.Size([batch, 16800, 2]) # 类别 只有一个类是没有第三维
               torch.Size([batch, 16800, 10]) # 关键点
           )
        '''
        out = self.model(images)
        p_bboxs_xywh = out[0]  # (batch,16800,4)  xywh  xywh
        p_labels = out[1]  # (batch,16800,10)
        p_keypoints = out[2]  # (batch,16800,2)

        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        num_batch = images.shape[0]
        num_ancs = self.anchors.shape[0]

        gbboxs_ltrb = torch.Tensor(*p_bboxs_xywh.shape).to(images)  # torch.Size([batch, 16800, 4])
        glabels = torch.Tensor(num_batch, num_ancs).to(images)  # 计算损失只会存在一维 无论多少类 标签只有一类
        gkeypoints = torch.Tensor(*p_keypoints.shape).to(images)  # 相当于empty
        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        for index in range(num_batch):
            g_bboxs = targets[index]['boxes']  # torch.Size([batch, 4])
            g_labels = targets[index]['labels']  # torch.Size([batch])

            # id_ = int(targets[index]['image_id'].item())
            # info = dataset_train.coco.loadImgs(id_)[0]  # 查看图片信息
            # flog.debug(info)

            g_keypoints = targets[index]['keypoints']  # torch.Size([batch, 10])
            # '''
            # 将g_bboxs 进行重构至与 anc的结果一致,并在 g_labels 根据IOU超参,上进行正反例标注
            # label_neg_mask: 反例的  布尔 torch.Size([16800]) 一维
            # anc_bbox_ind : 正例对应的 g_bbox 的index  torch.Size([16800]) 一维
            # '''
            # label_neg_mask, anc_bbox_ind = pos_match(self.anchors, g_bboxs, self.neg_iou_threshold)
            #
            # # new_anchors = anchors.clone().detach()
            # # 将bbox取出替换anc对应位置 ,根据 bboxs 索引list 将bboxs取出与anc 形成维度对齐 便于后面与anc修复算最终偏差 ->[anc个,4]
            # # 只计算正样本的定位损失,将正例对应到bbox标签 用于后续计算anc与bbox的差距
            # match_bboxs = g_bboxs[anc_bbox_ind]
            # match_keypoints = g_keypoints[anc_bbox_ind]
            #
            # # 构建正反例label 使原有label保持不变
            # # labels = torch.zeros(num_ancs, dtype=torch.int64)  # 标签默认为同维 类别0为负样本
            # # 正例保持原样 反例置0
            # match_labels = g_labels[anc_bbox_ind]
            # # match_labels[label_neg_mask] = torch.tensor(0).to(labels)
            # match_labels[label_neg_mask] = 0.

            match_bboxs, match_keypoints, match_labels = pos_match_retinaface(self.anchors, g_bboxs=g_bboxs,
                                                                              g_labels=g_labels,
                                                                              g_keypoints=g_keypoints,
                                                                              threshold_pos=cfg.THRESHOLD_PREDICT_CONF,
                                                                              threshold_neg=cfg.THRESHOLD_PREDICT_NMS
                                                                              )

            glabels[index] = match_labels
            gbboxs_ltrb[index] = match_bboxs
            gkeypoints[index] = match_keypoints
        '''---------------------与输出进行维度匹配及类别匹配  完成-------------------------'''

        # ---------------损失计算 ----------------------
        # log_dict用于显示
        loss_total, log_dict = self.losser(p_bboxs_xywh, gbboxs_ltrb, p_labels, glabels, p_keypoints, gkeypoints,
                                           imgs_ts=images)

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
        cfg = self.model.cfg
        # (batch,xx) -> (batch.xx)
        mask = p_scores >= self.threshold_conf
        mask = mask.squeeze(-1)
        # (batch,xx,4) -> (3,4) 拉伸降维
        p_boxes = p_boxes[mask]
        p_scores = p_scores[mask]
        p_keypoints = p_keypoints[mask]

        if p_scores.shape[0] == 0:
            flog.error('threshold_conf 过滤后 没有目标 %s', self.threshold_conf)
            return None, None, None

        if img_ts4 is not None:
            if cfg.IS_VISUAL:
                flog.debug('过滤后 threshold_conf %s', )
                # h, w = img_ts4.shape[-2:]
                show_anc4ts(img_ts4.squeeze(0), p_boxes, cfg.IMAGE_SIZE)

        p_scores = p_scores.squeeze(-1)
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
        p_scores = torch.sigmoid(p_conf)
        # p_scores = p_scores[:, :, 1]
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
    def handler_fmap(self, batch_data, path_dt_info, idx_to_class):
        '''
        生成 dt_info
            tvmonitor 0.471781 0 13 174 244
            cup 0.414941 274 226 301 265
        :param batch_data:
        :param path_dt_info:
        :param idx_to_class:
        :return:
        '''
        # (batch,3,640,640)   list(batch{'size','boxes','labels'}) 转换到GPU设备
        images, targets = _preprocessing_data(batch_data, self.device, mode='bbox')
        sizes = []
        files_txt = []
        for target in targets:
            sizes.append(target['size'])
            files_txt.append(os.path.join(path_dt_info, target['name_txt']))

        idxs, p_boxes, p_labels, p_scores = self.predicting4many(images)

        for batch_index, (szie, file_txt) in enumerate(zip(sizes, files_txt)):
            mask = idxs == batch_index  # 构建 batch 次的mask
            if torch.any(mask):  # 如果最终有目标存在 将写出info中
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

    def predicting4many(self, img_ts4):
        '''
        返回 CONF 修复后 和 nms 后的结果 和对应的ids_batch
        :param img_ts4:
        :return:
            off_batch: tensor 10395
            p_boxes: ltrb torch.Size([10395, 4])
            p_labels: tensor 10395
            p_scores: tensor 10395
        '''
        # (batch,xx,4)
        p_loc, p_conf, p_landms = self.model(img_ts4)
        # (batch,xx,1)->(batch.xx)
        # p_scores = torch.nn.functional.softmax(p_conf, dim=-1)
        # p_scores = torch.softmax(p_conf, dim=-1)
        # p_scores = p_scores[:, :, 1]  # 取人脸
        p_scores = torch.sigmoid(p_conf)
        # ---修复----variances = (0.1, 0.2)
        p_boxes = fix_bbox(self.anchors, p_loc)  # (1,xx,4)

        xywh2ltrb(p_boxes, safe=False)  # 这里转换 ltrb
        # (batch个) -> (batch,1) -> (batch, xx) 从1开始有利于批量nms
        off_batch = torch.arange(1, p_boxes.shape[0] + 1, device=self.device)  #
        off_batch = off_batch.view(-1, 1).repeat(1, p_boxes.shape[1])
        # (batch,xx,1) -> (batch.xx,1) -> torch.Size([8, 16800, 1])
        mask = p_scores >= self.threshold_conf  # 这里过滤有可能导致没得
        # torch.Size([8, 16800, 1]) -> (8, 16800)
        mask = mask.squeeze(-1)  # 删除最后维度
        # (batch,xx,4)^(8, 16800) -> (xx,4) 拉伸降维
        p_boxes = p_boxes[mask]
        # batch,xx,1 ^ batch,xx ->  (xx,1) -> (xx)
        p_scores = p_scores[mask].squeeze(-1) # 删除最后维
        off_batch = off_batch[mask]  # 对应的batch index
        keep = batched_nms(p_boxes, p_scores, off_batch, self.threshold_nms)
        # flog.debug('threshold_nms 过滤后有 %s 个', len(keep))
        p_labels = torch.ones(len(keep), dtype=torch.int64).to(p_boxes.device)
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]
        off_batch = off_batch[keep] - 1  # 为batched_nms +1 恢复 为batch 的 ids
        return off_batch, p_boxes, p_labels, p_scores

    def handler_coco(self, batch_data):
        '''

        :param batch_data:
        :return:
            res[id] = {
                'boxes': boxes_ltwh,
                'labels': torch.tensor(np.array(labels)),
                'scores': torch.tensor(np.array(scores)),
            }
        '''
        images, targets = _preprocessing_data(batch_data, self.device, mode='bbox')
        sizes = []
        ids = []
        for target in targets:
            ids.append(target['image_id'])
            sizes.append(target['size'])  # tesor

        idxs, p_boxes, p_labels, p_scores = self.predicting4many(images)

        res = {}
        # 每一张图的 id 与批次顺序保持一致 选出匹配
        for batch_index, (size, image_id) in enumerate(zip(sizes, ids)):
            mask = idxs == batch_index  # 构建 batch 次的mask
            if torch.any(mask):  # 如果最终有目标存在 将写出info中
                boxes_ltwh = ltrb2ltwh(p_boxes[mask] * size.repeat(2)[None])

                res[image_id] = {
                    'boxes': boxes_ltwh,
                    'labels': p_labels[mask],
                    'scores': p_scores[mask],
                }
            else:
                # flog.warning('没有预测出框 %s', files_txt)
                pass
        return res
