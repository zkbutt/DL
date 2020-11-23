import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_fit_fun import FitBase
from f_tools.fits.f_match import match4yolo3
from f_tools.fun_od.f_boxes import xywh2ltrb, fix_bbox, nms, batched_nms
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
        loss_total, log_dict = self.losser(out, targets,images)

        # -----------------构建展示字典及返回值------------------------
        # 多GPU时结果处理 reduce_dict 方法
        # losses_dict_reduced = reduce_dict(losses_dict)
        show_dict = {"loss_total": loss_total.detach().item(), }
        show_dict.update(**log_dict)
        return loss_total, show_dict


class PredictHandler(FitBase):
    def __init__(self, model, device,
                 grid, num_bbox, num_cls,
                 threshold_conf=0.5, threshold_nms=0.3):
        super(PredictHandler, self).__init__(model, device)
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf
        self.grid = grid
        self.num_bbox = num_bbox
        self.num_cls = num_cls

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
    def predicting4one(self, img_ts4, whwh):
        '''
        相比 forward 没有预处理
        :param img_ts4: torch.Size([batch, 3, 416, 416])
        :param whwh:原图片的比例 torch.Size([batch, 4])
        :return:
            {0: [[81, 125, 277, 296, 14], [77, 43, 286, 250, 14]]}
        '''
        num_dim = self.num_bbox * 5 + self.num_cls
        # ------模型输出处理-------
        p_yolo = self.model(img_ts4)  # torch.isnan(p_yolo).any()
        mask_coo = p_yolo[:, :, 4:5] > self.threshold_conf
        axis0, axis1, axis2 = torch.where(mask_coo)  # 获取网络的坐标
        print(axis0)

        res = {}
        img_id = None
        # 遍历每一个
        for i, (img_index, dim1, dim2) in enumerate(zip(axis0, axis1, axis2)):
            t = []
            p_one = p_yolo[img_index, dim1]  # batch,7,7,25 7*7*25
            p_one[:2] = (p_one[:2] + torch.tensor([l, r]) + 1) / self.grid
            bboxs = p_one[:4] * whwh[img_index]
            # xywh -> ltrb
            bboxs[:2] -= bboxs[2:] / 2  # 中心到左上
            bboxs[2:] += bboxs[:2]  # wh加左上
            # t.extend(list(p_one[:4].type(torch.int64).numpy()))
            t.extend(list(bboxs.type(torch.int64).numpy()))
            _, max_index = torch.max(p_one[5:], dim=0)
            t.append(max_index.item() + 1)
            if img_id != img_index.item():
                res[img_index.item()] = [t]
                img_id = img_index
            else:
                res[img_index.item()].append(t)

        return res

    # size_ = torch.cat([axis1[None], axis2[None]], dim=0).T  # 1,2  3,4 ---> [[1,2][3,4]] -> [[1,3][2,4]]
    #
    # batch = img_ts4.shape[0]
    # for i in range(batch):
    #     p_boxes_i = p_yolo[i][:, :, 4:5]
    #
    # p_res = p_yolo[mask_coo].view(-1, num_dim).contiguous()
    # num_res = p_res.shape[0]
    # flog.debug('conf过滤后 %s 个', num_res)
    #
    # if num_res > 0:
    #     p_boxes_ = p_res[:, :4]
    # p_boxes_ = p_boxes_[:, :2]
    # _, max_index = torch.max(p_res[:, 5:], dim=1)
    # p_cls = max_index[0] + 1
    #
    # # ---修复--并转换 xywh --> ltrb--variances = (0.1, 0.2)
    # p_boxes = fix_bbox4yolo1(self.anchors, p_loc)
    # xywh2ltrb(p_boxes, safe=False)
    #
    # p_boxes, p_keypoints, p_scores = self.output_res(p_boxes, p_keypoints, p_scores, img_ts4)
    # '''
    # xx是最终框 (xx,4)  (xx,10)  (xx)
    # '''
    # return p_boxes, p_keypoints, p_scores

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
