import time

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_lossfun import PredictOutput
from f_tools.fun_od.f_boxes import pos_match
from object_detection.coco_t.coco_api import coco_eval


def preprocessing_data(batch_data, device):
    '''
    cpu转gpu 输入模型前数据处理方法 定制
    :param batch_data:
    :param device:
    :return:
    '''
    images, targets = batch_data
    images = images.to(device)
    for target in targets:
        for key, val in target.items():
            target[key] = val.to(device)
    return images, targets


class FitBase(torch.nn.Module):
    def __init__(self, model, device):
        super(FitBase, self).__init__()
        self.model = model
        self.device = device


class LossProcess(FitBase):
    '''
    前向和反向 loss过程函数
    '''

    def __init__(self, model, device, anchors, losser, neg_iou_threshold):
        super(LossProcess, self).__init__(model, device)
        self.anchors = anchors
        self.losser = losser
        self.neg_iou_threshold = neg_iou_threshold

    def forward(self, batch_data):
        '''

        :param batch_data: tuple(images,targets)
            images:tensor(batch,c,h,w)
            list( # batch个
                target: dict{
                        image_id: int,
                        bboxs: np(num_anns, 4),
                        labels: np(num_anns),
                        keypoints: np(num_anns,10),
                    }
                )
        :return:
            loss_total: 这个用于优化
            show_dict:  log_dict
        '''
        # -----------------------输入模型前的数据处理 开始------------------------
        images, targets = preprocessing_data(batch_data, self.device)

        # -----------------------输入模型前的数据处理 完成------------------------

        '''
           模型输出 tuple(# 预测器 框4 类别2 关键点10
               torch.Size([batch, 16800, 4]) # 框
               torch.Size([batch, 16800, 2]) # 类别 只有一个类是没有第三维
               torch.Size([batch, 16800, 10]) # 关键点
           )
        '''
        out = self.model(images)
        bboxs_p = out[0]
        labels_p = out[1]
        keypoints_p = out[2]

        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        num_batch = images.shape[0]
        num_ancs = self.anchors.shape[0]

        gbboxs = torch.Tensor(num_batch, num_ancs, 4).to(images)  # torch.Size([5, 16800, 4])
        glabels = torch.Tensor(num_batch, num_ancs).to(images)  # 这个只会存在一维 无论多少类
        gkeypoints = torch.Tensor(num_batch, num_ancs, 10).to(images)
        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        for index in range(num_batch):
            bboxs = targets[index]['bboxs']  # torch.Size([40, 4])
            labels = targets[index]['labels']  # torch.Size([40,2])

            # id_ = int(targets[index]['image_id'].item())
            # info = dataset_train.coco.loadImgs(id_)[0]  # 查看图片信息
            # flog.debug(info)

            keypoints = targets[index]['keypoints']  # torch.Size([40, 10])
            '''
            masks: 正例的 index 布尔
            bboxs_ids : 正例对应的bbox的index
            '''
            label_neg_mask, anc_bbox_ind = pos_match(self.anchors, bboxs, self.neg_iou_threshold)

            # new_anchors = anchors.clone().detach()
            # 将bbox取出替换anc对应位置 ,根据 bboxs 索引list 将bboxs取出与anc 形成维度对齐 便于后面与anc修复算最终偏差 ->[anc个,4]
            # 只计算正样本的定位损失,将正例对应到bbox标签 用于后续计算anc与bbox的差距
            match_bboxs = bboxs[anc_bbox_ind]
            match_keypoints = keypoints[anc_bbox_ind]

            # 构建正反例label 使原有label保持不变
            # labels = torch.zeros(num_ancs, dtype=torch.int64)  # 标签默认为同维 类别0为负样本
            # 正例保持原样 反例置0
            match_labels = labels[anc_bbox_ind]
            # match_labels[label_neg_mask] = torch.tensor(0).to(labels)
            match_labels[label_neg_mask] = 0.  # 这里没有to设备
            # labels[label_neg_ind] = 0 #  这是错误的用法
            glabels[index] = match_labels

            gbboxs[index] = match_bboxs
            gkeypoints[index] = match_keypoints
        '''---------------------与输出进行维度匹配及类别匹配  完成-------------------------'''

        # ---------------损失计算 ----------------------
        # log_dict用于显示
        loss_total, log_dict = self.losser(bboxs_p, gbboxs, labels_p, glabels, keypoints_p, gkeypoints)

        # -----------------构建展示字典及返回值------------------------
        # 多GPU时结果处理 reduce_dict 方法
        # losses_dict_reduced = reduce_dict(losses_dict)
        show_dict = {"loss_total": loss_total.detach().item(), }
        show_dict.update(**log_dict)
        return loss_total, show_dict


class ForecastProcess(PredictOutput):

    def __init__(self, model, device, ancs, img_size, coco, variance=(0.1, 0.2), iou_threshold=0.5, max_output=100,
                 eval_mode='bboxs'):
        '''

        :param model:
        :param ancs:
        :param img_size:
        :param coco:
        :param variance:
        :param iou_threshold:
        :param max_output:
        :param eval_mode: bboxs   keypoints
        '''
        super(ForecastProcess, self).__init__(ancs,
                                              img_size,
                                              variance=variance,
                                              iou_threshold=iou_threshold,
                                              max_output=max_output
                                              )
        self.model = model
        self.coco = coco
        self.eval_mode = eval_mode
        self.device = device

    def forward(self, batch_data, epoch, coco_res_bboxs, coco_res_keypoints=None):
        '''
        这是验证生成 coco_res_bboxs 的方法
        :param batch_data:
        :param epoch:
        :param coco_res_bboxs: 用于保存结果
        :param coco_res_keypoints:
        :return:
        '''
        # -----------------------输入模型前的数据处理 开始------------------------
        images, targets = preprocessing_data(batch_data, self.device)

        # -----------------------输入模型前的数据处理 完成------------------------

        '''
           模型输出 tuple(# 预测器 框4 类别2 关键点10
               torch.Size([batch, 16800, 4]) # 框
               torch.Size([batch, 16800]) # 类别 只有一个类是没有第三维
               torch.Size([batch, 16800, 10]) # 关键点
           )
        '''
        out = self.model(images)
        bboxs_p = out[0]
        labels_p = out[1]
        keypoints_p = out[2]

        if self.eval_mode == 'bboxs':
            keypoints_p = None

        # imgs_rets 每一个图的最终输出 ltrb  list([bboxes_out, scores_out, labels_out, other_in] * batch个 )
        imgs_rets = super().forward(bboxs_p, labels_p, keypoints_p, ktype='ltwh')  # 用父类的nms
        '''coco结果要求ltwh'''

        for i, ret in enumerate(imgs_rets):  # 这里遍历一个批量中每一张图片
            image_id = int(targets[i]['image_id'].cpu().item())

            w, h = self.img_size  # 使用模型进入尺寸

            # np高级 选框用特图尺寸恢复
            bboxs, scores, labels = ret[0], ret[1], ret[2]
            img_size = torch.tensor((w, h)).to(bboxs)
            bboxs = bboxs * img_size[None].repeat(1, 2)
            # 构造每一张图片的COCO结果json
            if self.eval_mode == 'bboxs':
                for bbox, score, label in zip(bboxs, scores, labels):
                    _t_bbox = {}
                    _t_bbox['image_id'] = image_id
                    _t_bbox['category_id'] = label.cpu().item()
                    _t_bbox['bbox'] = list(bbox.cpu().numpy())
                    _t_bbox['score'] = score.cpu().item()
                    coco_res_bboxs.append(_t_bbox)

            elif self.eval_mode == 'keypoints':
                keypoints = ret[3]
                keypoints = keypoints * img_size[None].repeat(1, 5)
                coco_keypoints = torch.zeros(
                    (int(keypoints.shape[0]), int(keypoints.shape[-1] / 2 + keypoints.shape[-1]))).to(keypoints)
                coco_keypoints[:, 2::3] = 2
                axis1, axis2 = torch.where(coco_keypoints != 2)
                coco_keypoints[:, torch.unique(axis2)] = keypoints

                # tensor也可以直接遍历
                for bbox, score, label, keypoint in zip(bboxs, scores, labels, coco_keypoints):
                    _t_bbox = {}
                    _t_bbox['image_id'] = image_id
                    _t_bbox['category_id'] = label.cpu().item()
                    _t_bbox['bbox'] = list(bbox.cpu().numpy())
                    _t_bbox['score'] = score.cpu().item()
                    coco_res_bboxs.append(_t_bbox)
                    _t_keypoints = {}
                    _t_keypoints['image_id'] = image_id
                    _t_keypoints['category_id'] = label.cpu().item()
                    _t_keypoints['keypoints'] = list(keypoint.cpu().numpy())
                    _t_keypoints['score'] = score.cpu().item()
                    coco_res_keypoints.append(_t_keypoints)
