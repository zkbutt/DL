from __future__ import absolute_import
import torch as t
import torch
from torch import nn
from torchsummary import summary
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from object_detection.simple_faster_rcnn_pytorch_master.model.region_proposal_network import RegionProposalNetwork
from object_detection.simple_faster_rcnn_pytorch_master.model.faster_rcnn import FasterRCNN
from object_detection.simple_faster_rcnn_pytorch_master.utils import array_tool as at
from object_detection.simple_faster_rcnn_pytorch_master.utils.config import opt, Config


def decom_vgg16():  # 获得vgg16的backbone卷积层和classifier全连接层
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:  # 加载caffe预训练模型两个条件：1.caffe_pretrain不为空，2.load_path为空
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        # 使用这个 pretrained=True
        # model = vgg16(not opt.load_path)  # 若load_path为空即没有FasterRCNN预训练权重则使用pytorch权重
        model = vgg16()  # 本地加载
        # model.load_state_dict(torch.load(r'D:\down\AI\weights\pytorch\vgg16-397923af.pth'))
        model.load_state_dict(torch.load(Config.PATH_VGG16_WEIGHT))

    # ----------- 取出各层 和 classifier重组 重新返回 ----------
    # 没有要最后一层最大池化层
    features = list(model.features)[:30]  # vgg的特征提取层
    # features = list(model.features)  # vgg的特征提取层
    classifier = model.classifier  # vgg的全连接层

    # 替换原有的 classifier 层
    classifier = list(classifier)
    del classifier[6]  # 删除最后一层
    if not opt.use_drop:  # 如设置不使用dropout则删除dropout层
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:  # 冻结层
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.

    """
    # 这个参数是根据模型预设的
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16 vgg16下采样16倍

    def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        # 获得vgg16层信息 nn.Sequential
        extractor, classifier = decom_vgg16()

        # nn.Module 接vgg16的输出 512 生成 anchor 模板
        rpn = RegionProposalNetwork(in_channels=512, mid_channels=512,
                                    ratios=ratios, anchor_scales=anchor_scales,
                                    feat_stride=self.feat_stride,
                                    )

        # 实例化head，
        head = VGG16RoIHead(n_class=n_fg_class + 1, roi_size=7,
                            spatial_scale=(1. / self.feat_stride),
                            classifier=classifier
                            )

        # 特殊化的模块组合成FasterRCNN架构
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head, )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.

    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        '''

        :param n_class: 输出21类
        :param roi_size: ROI输出7个格子
        :param spatial_scale: 下采样比例  1. / self.feat_stride 1/16
        :param classifier:  分类层
        '''
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier  # vgg16的全连接层 接两个下级
        self.cls_loc = nn.Linear(4096, n_class * 4)  # 21*4=84 最后一层的回归层
        self.score = nn.Linear(4096, n_class)  # 21 最后一层的分类层

        normal_init(self.cls_loc, 0, 0.001)  # 初始化回归层参数
        normal_init(self.score, 0, 0.01)  # 初始化分类层参数

        self.n_class = n_class  # 类别数量
        self.roi_size = roi_size  # roi pooling切分格子
        self.spatial_scale = spatial_scale
        # ----------换成py方法---------
        # self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)  # roi pooling网络

        # box_roi_pool = MultiScaleRoIAlign(
        #                 featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
        #                 output_size=[7, 7],
        #                 sampling_ratio=2)

        self.roi = RoIPool(self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        '''

        :param x:  torch.Size([1, 512, 37, 50])
        :param rois:  <class 'tuple'>: (128, 4)
        :param roi_indices:  t.zeros(len(sample_roi))
        :return:
        '''
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)  # shape=(N,R,5)
        # 多维 列转换
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]  # yx->xy
        indices_and_rois = xy_indices_and_rois.contiguous()
        # RoIPool 输出torch.Size([128, 512, 7, 7])
        pool = self.roi(x, indices_and_rois)  # feature_map+rois的ROIPOOLING,得到7*7的向量
        pool = pool.view(pool.size(0), -1)  # 向量拉平进入全连接层
        fc7 = self.classifier(pool)  # 全连接层前向
        roi_cls_locs = self.cls_loc(fc7)  # 回归层
        roi_scores = self.score(fc7)  # 分类层
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):  # 初始化网络参数的函数
    '''

    :param m:
    :param mean:
    :param stddev:
    :param truncated:
    :return:
    '''
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


if __name__ == '__main__':
    model = vgg16(not opt.load_path)
    print(model.features)
    print(model.named_parameters())
    for i in model.named_parameters():
        print(i)

    # features = list(model.features)[:30]
    # summary(model, (3, 500, 500))  # 通过2D卷积 in=3 判断输入
