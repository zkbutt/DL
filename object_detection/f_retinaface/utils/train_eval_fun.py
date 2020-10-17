import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_boxes import pos_match, xywh2ltrb
from f_tools.pic.f_show import show_od_keypoints4ts, show_od4ts
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG


def _preprocessing_data(batch_data, device):
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


class LossHandler(FitBase):
    '''
    前向和反向 loss过程函数
    '''

    def __init__(self, model, device, anchors, losser, neg_iou_threshold):
        super(LossHandler, self).__init__(model, device)
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
                        bboxs: np(num_anns, 4), ltrb
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
        # if CFG.IS_VISUAL:
        #     flog.debug('查看模型输入和anc %s', )
        #     for img_ts, target in zip(images, targets):
        #         # 遍历降维
        #         _t = torch.cat([target['bboxs'], target['keypoints']], dim=1)
        #         _t[:, ::2] = _t[:, ::2] * CFG.IMAGE_SIZE[0]
        #         _t[:, 1::2] = _t[:, 1::2] * CFG.IMAGE_SIZE[1]
        #         show_od_keypoints4ts(img_ts, _t[:, :4], _t[:, 4:14], target['labels'])
        #
        #         _t = self.anchors.view(-1, 4).clone()
        #         _t[:, ::2] = _t[:, ::2] * CFG.IMAGE_SIZE[0]
        #         _t[:, 1::2] = _t[:, 1::2] * CFG.IMAGE_SIZE[1]
        #         show_od4ts(img_ts, xywh2ltrb(_t)[:999, :], torch.ones(200))

        # -----------------------输入模型前的数据处理 完成------------------------

        '''
           模型输出 tuple(# 预测器 框4 类别2 关键点10
               torch.Size([batch, 16800, 4]) # 框
               torch.Size([batch, 16800, 2]) # 类别 只有一个类是没有第三维
               torch.Size([batch, 16800, 10]) # 关键点
           )
        '''
        out = self.model(images)
        p_bboxs = out[0]  # (batch,16800,4)
        p_labels = out[1]  # (batch,16800,10)
        p_keypoints = out[2]  # (batch,16800,2)

        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        num_batch = images.shape[0]
        num_ancs = self.anchors.shape[0]

        gbboxs = torch.Tensor(*p_bboxs.shape).to(images)  # torch.Size([batch, 16800, 4])
        glabels = torch.Tensor(num_batch, num_ancs).to(images)  # 计算损失只会存在一维 无论多少类 标签只有一类
        gkeypoints = torch.Tensor(*p_keypoints.shape).to(images)  # 相当于empty
        '''---------------------与输出进行维度匹配及类别匹配-------------------------'''
        for index in range(num_batch):
            g_bboxs = targets[index]['bboxs']  # torch.Size([batch, 4])
            g_labels = targets[index]['labels']  # torch.Size([batch])

            # id_ = int(targets[index]['image_id'].item())
            # info = dataset_train.coco.loadImgs(id_)[0]  # 查看图片信息
            # flog.debug(info)

            g_keypoints = targets[index]['keypoints']  # torch.Size([batch, 10])
            '''
            将g_bboxs 进行重构至与 anc的结果一致,并在 g_labels 根据IOU超参,上进行正反例标注
            label_neg_mask: 反例的  布尔 torch.Size([16800]) 一维
            anc_bbox_ind : 正例对应的 g_bbox 的index  torch.Size([16800]) 一维
            '''
            label_neg_mask, anc_bbox_ind = pos_match(self.anchors, g_bboxs, self.neg_iou_threshold)

            # new_anchors = anchors.clone().detach()
            # 将bbox取出替换anc对应位置 ,根据 bboxs 索引list 将bboxs取出与anc 形成维度对齐 便于后面与anc修复算最终偏差 ->[anc个,4]
            # 只计算正样本的定位损失,将正例对应到bbox标签 用于后续计算anc与bbox的差距
            match_bboxs = g_bboxs[anc_bbox_ind]
            match_keypoints = g_keypoints[anc_bbox_ind]

            # 构建正反例label 使原有label保持不变
            # labels = torch.zeros(num_ancs, dtype=torch.int64)  # 标签默认为同维 类别0为负样本
            # 正例保持原样 反例置0
            match_labels = g_labels[anc_bbox_ind]
            # match_labels[label_neg_mask] = torch.tensor(0).to(labels)
            match_labels[label_neg_mask] = 0.
            glabels[index] = match_labels

            gbboxs[index] = match_bboxs
            gkeypoints[index] = match_keypoints
        '''---------------------与输出进行维度匹配及类别匹配  完成-------------------------'''

        # ---------------损失计算 ----------------------
        # log_dict用于显示
        loss_total, log_dict = self.losser(p_bboxs, gbboxs, p_labels, glabels, p_keypoints, gkeypoints, imgs_ts=images)

        # -----------------构建展示字典及返回值------------------------
        # 多GPU时结果处理 reduce_dict 方法
        # losses_dict_reduced = reduce_dict(losses_dict)
        show_dict = {"loss_total": loss_total.detach().item(), }
        show_dict.update(**log_dict)
        return loss_total, show_dict
