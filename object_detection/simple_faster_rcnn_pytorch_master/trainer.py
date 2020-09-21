from __future__ import absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from .model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from .utils import array_tool as at
from .utils.vis_tool import Visualizer

from .utils.config import opt

# from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',  # 类似c的结构体，保存个batch训练的loss
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):  # 定义训练器类
    """wrapper for conveniently training. return losses

    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn  # 训练的网络
        self.rpn_sigma = opt.rpn_sigma  # 计算rpn_loc_loss的超参数
        self.roi_sigma = opt.roi_sigma  # 计算roi_loc_loss的超参数

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()  # rpn处使用的，实例化anchor和gt匹配函数
        self.proposal_target_creator = ProposalTargetCreator()  # roi处使用的，实例化提取roi正负样本函数

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean  # 图片归一化参数mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std  # 图片归一化参数std

        self.optimizer = self.faster_rcnn.get_optimizer()  # 实例化得到optimizer函数
        # visdom wrapper
        # self.vis = Visualizer(env=opt.env)  # 可视化内容，（跳过）
        # indicators for training status
        # self.rpn_cm = ConfusionMeter(2)  # 可视化内容，（跳过）
        # self.roi_cm = ConfusionMeter(21)  # 可视化内容，（跳过）
        # self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss，可视化内容，（跳过）

    def forward(self, imgs, bboxes, labels, scale):  # 前向函数
        '''

        :param imgs:4维训练图 Tensor 已归一化 (1,c,h,w) 只有一张
        :param bboxes: 3维 Tensor(1,1,4) 输入支持多个 标签的坐标
        :param labels: 2维 Tensor(1,1)
        :param scale:  尺寸缩放比例---原图
        :return:
        '''
        # n = batch_size,bboxes.shape = (N,k)
        n = bboxes.shape[0]  # 取出选框 只训练一个
        # 只支持batch_size=1的训练
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        # 主干网络,输出特图
        features = self.faster_rcnn.extractor(imgs)  # 特征提取部分,vgg的conv5_3输出的下采样为16,输出为(N,C,H/16,W/16)

        # rpn部分
        _, _, H, W = imgs.shape  # img.shape=(N,C,H,W)
        img_size = (H, W)  # 原始图尺寸
        # RegionProposalNetwork 这个类 forward 方法
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # 方法只支持一张图片
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois  # 改多张图时 注意这个

        # 提取正负样本个数,默认总共128个 对应 ProposalTargetCreator 的 forward
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # VGG16RoIHead  forward 输出结果
        sample_roi_index = t.zeros(len(sample_roi))  # 因为batch_size=1,所以128个rois对应的图片下标都是第一个(下标0)
        # 返回roi_cls_loc---torch.Size([128, 84])  roi_score---torch.Size([128, 21])
        roi_cls_loc, roi_score = self.faster_rcnn.head(  # VGG16RoIHead roi pooling+全连接部分
            features,  # torch.Size([1, 512, 37, 50])
            sample_roi,  # <class 'tuple'>: (128, 4)
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        # anchor和gt匹配得到用于loss计算的gt AnchorTargetCreator
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),  # torch.Size([4, 4])
            anchor,  # <class 'tuple'>: (18648, 4)
            img_size)  # 原始图尺寸
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        # 用gt的坐标、label和预测框的坐标来计算rpn_loc_loss,label用于前景背景，坐标用于
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, # rpn预测出的偏差
            gt_rpn_loc, # 实际的偏差
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # 用gt的label和预测框的坐标来计算rpn_cls_loss# 未加 softmax torch.Size([16650, 2])
        rpn_cls_loss = F.cross_entropy(rpn_score,
                                       gt_rpn_label,  # 16650
                                       ignore_index=-1
                                       )

        # 可视化内容，（跳过）
        # _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        # _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        # self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]  # 正负样本筛选后数量，默认128
        # torch.Size([128, 84]) ---> torch.Size([128, 21, 4])
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)  # roi_cls_loc.shape=(n_sample, num_class*4)
        # 选取每个roi对应的class_loc进行训练
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long(), at.totensor(gt_roi_label).long()]

        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        # 回归损失
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        # 分类损失 用roi_score和匹配正负样本的gt_roi_label计算loss
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

        # 可视化
        # self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())  # 可视化内容，（跳过）

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]  # 保存4个loss
        losses = losses + [sum(losses)]  # 保存总和loss

        return LossTuple(*losses)  # 返回5个loss

    def train_step(self, imgs, bboxes, labels, scale):
        '''

        :param imgs:4维训练图 Tensor 已归一化 (1,c,h,w) 只有一张
        :param bboxes: 3维 Tensor(1,1,4)
        :param labels: 2维 Tensor(1,1)
        :param scale:  尺寸缩放比例---原图
        :return: 返回一次loss
        '''
        self.optimizer.zero_grad()  # 梯度清零用于反向传播
        losses = self.forward(imgs, bboxes, labels, scale)  # 计算loss
        losses.total_loss.backward()  # 反向传播
        self.optimizer.step()  # 更新梯度
        # self.update_meters(losses)  # 可视化内容，（跳过）
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):  # 模型保存函数
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):  # 模型加载函数
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):  # 可视化内容，（跳过）
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):  # 可视化内容，（跳过）
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    '''

    :param x: 生成的所有框 torch.Size([18648, 4])
    :param t: torch.Size([18648, 4])
    :param in_weight: pred_loc 正例索引 正例才计算
    :param sigma: 超参 RPN loss的占比
    :return:
    '''
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)  # 回归出的偏差 和实际的偏差之间的差异 只取类别大于1的
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()  # 偏差小于某值时算法不一样
    # smooth l1 loss  原文 sigma2 =1
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    '''

    :param pred_loc:  rpn预测出的偏差
    :param gt_loc:  实际的偏差  loc <class 'tuple'>: (18648, 4)
    :param gt_label: 实际的类别   label  <class 'tuple'>: (18648,)
    :param sigma: 超参 RPN loss的占比
    :return: 返回一个数
    '''
    in_weight = t.zeros(gt_loc.shape)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    _t1 = (gt_label > 0).view(-1, 1)  # 1维>0 取正例 拉成2维
    _t2 = _t1.expand_as(in_weight)
    in_weight[_t2] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())  # 忽略 gt_label==-1 for rpn_loss
    return loc_loss
