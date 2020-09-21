from __future__ import absolute_import
from __future__ import division

import torch
import torch as t
import numpy as np
# import cupy as cp
from torchvision.ops import nms

from object_detection.simple_faster_rcnn_pytorch_master.utils import array_tool as at
from object_detection.simple_faster_rcnn_pytorch_master.model.utils.bbox_tools import loc2bbox
# from object_detection.simple_faster_rcnn_pytorch_master.model.utils.nms import non_maximum_suppression

from torch import nn
from object_detection.simple_faster_rcnn_pytorch_master.data.dataset import preprocess
from torch.nn import functional as F
from object_detection.simple_faster_rcnn_pytorch_master.utils.config import opt


def nograd(f):  # 不计算梯度函数
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)

    return new_f


class FasterRCNN(nn.Module):  # 定义FasterRCNN架构类
    """Base class for Faster R-CNN.
       是 FasterRCNNVGG16 的父亲

    """

    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        '''

        :param extractor: 主干网
        :param rpn:
        :param head: roi + classifier
        :param loc_normalize_mean:
        :param loc_normalize_std:
        '''
        super(FasterRCNN, self).__init__()
        self.extractor = extractor  # 特征提取网络
        self.rpn = rpn  # rpn网络
        self.head = head  # roi pooling + 全连接层网络

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean  # 坐标归一化参数
        self.loc_normalize_std = loc_normalize_std  # 坐标归一化参数
        self.use_preset('evaluate')  # 可视化内容，（跳过）

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.


        """
        img_size = x.shape[2:]  # 得到图片尺寸(H,W)

        h = self.extractor(x)  # 特征提取
        # 只用 300精选建议框
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        # roi pooling + 全连接层网络，输入特征提取的feature map和rpn网络输出的rois,rois_indices
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):  # 预设超参数
        """Use the given preset during prediction.


        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        '''
        抑制输出(预测时使用)
        :param raw_cls_bbox:
        :param raw_prob:
        :return:
        '''
        bbox = list()  # 最终的输出框
        label = list()  # 最终的输出label
        score = list()  # 最终的输出分数
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):  # 忽略cls_id=0，因为是背景类。以类别为单位
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]  # 该类别的bbox
            prob_l = raw_prob[:, l]  # 该类别概率
            mask = prob_l > self.score_thresh  # 第一轮筛选，得到分数大于阈值的索引
            cls_bbox_l = cls_bbox_l[mask]  # 得到需要的该类别的框
            prob_l = prob_l[mask]  # 得到需要的该类别概率

            # 第二轮筛选，非极大抑制，输入该类别的框和该类别概率
            keep = nms(torch.tensor(cls_bbox_l), torch.tensor(prob_l), self.nms_thresh, )
            bbox.append(cls_bbox_l[keep])  # 两轮筛选后的框
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))  # label在[0,self.n_class - 2]为了对应索引,考虑背景和物理下标则减2
            score.append(prob_l[keep])  # 两类筛选后的类别概率
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)  # 最终的输出框
        label = np.concatenate(label, axis=0).astype(np.int32)  # 最终的输出label
        score = np.concatenate(score, axis=0).astype(np.float32)  # 最终的输出分数
        return bbox, label, score

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):  # 预测函数
        '''

        :param imgs: 测试集图片 torch.Size([1, 3, 600, 901])
        :param sizes: 原图尺寸  [[333, 500]]
        :param visualize:
        :return:
        '''
        self.eval()  # 网络设置为eval模式(禁用BatchNorm和Dropout)
        if visualize:  # 可视化内容，（跳过）
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()  # 最终的输出框
        labels = list()  # 最终的输出label
        scores = list()  # 最终的输出分数
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()  # 增加batch维 迭代出来是3维 再加一1
            scale = img.shape[3] / size[1]  # 获得预处理时的   scale
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)  # 前向
            #  roi_cls_loc torch.Size([250, 84]) , roi_scores torch.Size([250, 21]) , rois (250, 4)
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale  # 把rois 缩放到预处理后的图片上

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).repeat(self.n_class)[None]  # 去重
            std = t.Tensor(self.loc_normalize_std).repeat(self.n_class)[None]
            roi_cls_loc = (roi_cls_loc * std + mean)  # 将偏移 反归一化
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)  # 一个框对应n_class个loc，所以要expand_as到同维度后面可以二次修正框

            #
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # 剔除边界
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])  # 限制超出尺寸的框
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])  # 限制超出尺寸的框
            # softmax得到每个框的类别概率 21 个类别 300个框 共6300个结果
            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)
            # 输入框以及对应的类别概率，抑制输出
            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)

            # 输出坐标，类别，该类别概率
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        # self.use_preset('evaluate')  # 可视化内容，（跳过）
        self.train()  # 返回train模式
        return bboxes, labels, scores

    def get_optimizer(self):  # 得到optimizer函数
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr  # 配置超参
        params = []  # 添加多个配置字典
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:  # 没有锁定时进入 , 初始化优化策略和超参
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:  # optimizer = optim.Adam(net.parameters(), lr=0.0002)
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):  # 衰减学习率函数
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
