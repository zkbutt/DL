import numpy as np
from torch.nn import functional as F
from torch import nn

from object_detection.simple_faster_rcnn_pytorch_master.model.utils.bbox_tools import generate_anchor_base
from object_detection.simple_faster_rcnn_pytorch_master.model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.
        核心 model
    """

    def __init__(self, in_channels=512, mid_channels=512,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 feat_stride=16, proposal_creator_params=dict(), ):
        '''

        :param in_channels:
        :param mid_channels:rpn第一个3卷积输出的维度 通常与 in_channels 相同
        :param ratios:
        :param anchor_scales:
        :param feat_stride: 下采样倍数
        :param proposal_creator_params: ProposalCreator 构造方法的参数, 默认不传
        '''
        super(RegionProposalNetwork, self).__init__()
        # 生成 anchor_base 9个模板
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride  # 下采样倍数

        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)  # 实例化生成roi函数
        n_anchor = self.anchor_base.shape[0]  # anchor_base的数量
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)  # some卷积

        # 1卷积 (9 * 2)  rpn分类层
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 1卷积 (9 * 4) rpn回归层 修正参数 用于调整中心点和hw
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        normal_init(self.conv1, 0, 0.01)  # 初始化3x3卷积核参数
        normal_init(self.score, 0, 0.01)  # 初始化rpn分类层参数
        normal_init(self.loc, 0, 0.01)  # 初始化rpn回归层参数

    def forward(self, x, img_size, scale=1.):
        '''

        :param imgs:
            4维训练图 Tensor 已归一化 (1,c,h,w) torch.Size([1, 512, 35, 62]) 只有一张
            用于 clip 建议框
        :param bboxes: 3维 Tensor(1,1,4) 输入支持多个
        :param scale:  尺寸缩放比例---原图 未在 600 - 800 不为1 用于与原图一起进行还原
        :return:
            rpn_locs 所有 anchor 的调整系数 torch.Size([1, 16650(h*w*9), 4])
            rpn_scores 所有 anchor 的得分 torch.Size([1, 16650(h*w*9), 2])
            rois 建议框 <class 'tuple'>: (2000, 4)
            roi_indices 建议框对应的图片索引  多图时 <class 'tuple'>: (2000,)
            anchor 原始的 anchor    (hh*ww*9,4) <class 'tuple'>: (16650, 4)
        '''

        h = F.relu(self.conv1(x))  # 经过3x3卷积核 (1,c,h,w)
        # 特图尺寸(每一个点,确定生成多少套 anchor 每一个anchor 4维表示, )
        rpn_locs = self.loc(h)  # 回归层输出每一个 anchor 的调整系数，shape=(n,一套anchor数*4,h,w)
        # pytorch (n,c,h,w) 调整c到最后,再分4 用于对应生成的 anchor
        n, _, hh, ww = x.shape  # 只支持 1 张特图
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # $$$这个深复制感觉没用 原装结果要返回

        rpn_scores = self.score(h)  # 分类层,shape=(n,num_anchor*2,h,w) 未加 softmax
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  # 调整c=9 * 2到最后 (n,h,w,num_anchor*2)

        # 生成feature map 的 hh*ww*9 个 anchor (hh*ww*9,4)
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww)

        # 与这个相同 n_anchor = self.anchor_base.shape[0]
        n_anchor = anchor.shape[0] // (hh * ww)  # feature map的 anchor 数量 输出9

        _rpn_scores = rpn_scores.view(n, hh, ww, n_anchor, 2)  # 调整并在第4维上 n_anchor 上求值
        rpn_softmax_scores = F.softmax(_rpn_scores, dim=4)  # softmax得到每个点的9个框分类概率 0 是背景 1是前景
        # 定义1维是前景 0是背景 这里应该用二分类
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # 只取前景的概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # 拉平与 hh*ww对应 anchor 的得分是不是前景

        rpn_scores = rpn_scores.view(n, -1, 2)  # 图片所有anchor的前背景概率 这个要返回

        rois = list()  # 保存roi的list
        roi_indices = list()  # 保存roi对应哪张图片的list
        for i in range(n):
            # 建议框生成策略
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),  # 把所有 anchor 取出来  (hh*ww*9,4)
                rpn_fg_scores[i].cpu().data.numpy(),  # 对应前景取出来 (hh*ww*9)
                anchor,  # (hh*ww*9,4)
                img_size,  # 原图片的size(用于限制超出图片的框)
                scale=scale)

            # 生成2000个 1 * 0
            batch_index = i * np.ones((len(roi),), dtype=np.int32)  # 有多张图时, 图与建议框的对应关系
            # rois.shape = (n,S,4), roi_indices.shape=(n,S)
            rois.append(roi)  # 保存roi
            roi_indices.append(batch_index)  # 保存roi对应该图片

        # _rois = np.array(rois)  # 多图片时考虑用这个
        # 当只有一个时 为shape=(2000,4)
        rois = np.concatenate(rois, axis=0)  # 多个时转换为np.ndarray类型，shape=(n,S,4)
        # 当只有一个时 为shape=(2000,)
        roi_indices = np.concatenate(roi_indices, axis=0)  # 转换为np.ndarray类型，shape=(n,S)
        # 最后返回rpn自己的输出rpn_locs, rpn_scores以及给下面继续使用的输出rois, roi_indices, anchor
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


# 获得feature_map的anchor
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    '''
    对应 AnchorsGenerator __grid_anchors
    :param anchor_base: 一套 00点 9 个anchor (9,4)
    :param feat_stride: 下采样倍数
    :param height: 特图的h
    :param width: 特图的w
    :return: 生成原图对应所有 anchor 的坐标 (K*A,4)个
        所有 anchor 对应在原图的框坐标(K,A,4)
        K是对应原图的拉平点
        A是anchor 4代表真实左上右下坐标
    '''
    '''---根据偏移生成图片的网点坐标(anchor的中点,左上角点)---'''
    import numpy as np
    # 对应原图上的列坐标 --- 每一点对应原图的 top 一维np数组
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    # 对应原图上的列坐标 --- 每一点对应原图的 left
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    # arg1扩展成行 2维 ,arg2扩展成列 2维, 形成网 h*w 个左上角的点 --- 偏移量
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    '''拉平后表示图片中每一个网点的选框调整值'''
    # 左上和右下平移比例是一样的 转置后 列合并 输出 (K,4)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]  # 取 anchor_base 的数量
    K = shift.shape[0]  # K = height*width  映射回原图的 点位数

    # 最终要形成(K,A,4)
    _r = anchor_base.reshape((1, A, 4))
    _t = shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = _r + _t

    # _r = anchor_base  # 使用np自动广播
    # _t = shift.reshape((K, 1, 4))
    # anchor1 = _r + _t

    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  # 修正后拉平
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    import numpy as xp

    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):  # 初始化网络参数
    '''
        weight initalizer: truncated normal and random normal.
    :param m:  模型
    :param mean:  均值
    :param stddev: 方差
    :param truncated:
    :return:
    '''
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
