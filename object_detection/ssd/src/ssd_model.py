import os
from collections import OrderedDict

from f_tools.GLOBAL_LOG import flog
from object_detection.ssd.CONFIG_SSD import NEG_RATIO
from object_detection.ssd.src.res50_backbone import resnet50
from torch import nn, Tensor
import torch
from torch.jit.annotations import Optional, List, Dict, Tuple, Module
from object_detection.ssd.src.utils import dboxes300_coco, Encoder, PostProcess


class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()  # 采用默认构造
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path and os.path.exists(pretrain_path):
            # 只是载入resnet pytorch
            # 只加载部份参数
            pretrained_dict = torch.load(pretrain_path)
            model_dict = net.state_dict()  # 模型k
            d = OrderedDict()
            for k, v in pretrained_dict.items():
                if k == 'layer4.0.conv1':
                    break
                d[k] = v
            model_dict.update(d)
            net.load_state_dict(model_dict)

            flog.debug('resnet权重加载成功 %s', pretrain_path)

        # 只取另一模型的某连接的模块 返回Sequential
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]  # 取出最后一层的(残差下采样层)
        # 修改conv4_block1的步距，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    '''核心类'''

    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            # self.out_channels = [1024, 512, 512, 256, 256, 256] 这个是超参
            raise Exception("the backbone not has attribute: out_channel")
        self.feature_extractor = backbone

        self.num_classes = num_classes
        # 输入 out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        '''返回 self.additional_blocks ---nn.ModuleList(additional_blocks)'''
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]  # 定义 6 个特图的 anc模板数
        location_extractors = []
        confidence_extractors = []

        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        # ---这个后面属于损失的，与模型无关---
        default_box = dboxes300_coco()
        self.compute_loss = Loss(default_box)
        # self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)  # 后处理生成框

    def _build_additional_features(self, input_size):
        """
        为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
        :param input_size:  [1024, 512, 512, 256, 256, 256]
        :return:
        """
        additional_blocks = []
        # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            # 这里要增加5层(有一层是输入)
            padding, stride = (1, 2) if i < 3 else (0, 1)  # 前3层尺寸减半,后二层尺寸-2
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        '''初始化权重'''
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, features, loc_extractor, conf_extractor):
        '''
        计算并组合多个特图的 anc 调整 及 对应分数
        :param features: 6个特图
        :param loc_extractor: 最后输出6个loc的层
        :param conf_extractor:最后输出6个conf的层
        :return: 组合后的 locs confs tensor格式
        '''
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1] = [batch, 4, n * w * h]
            locs.append(l(f).view(f.size(0), 4, -1))
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1] = [batch, 21, n * w * h]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        # 相当于把每一个6个特图出来的list 在最后一维进行拉平堆叠
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):
        '''

        :param image: 批画像
        :param targets: 其中bboxs是与GT匹配后的
        :return:
        '''
        x = self.feature_extractor(image)

        # Feature Map 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        detection_features = torch.jit.annotate(List[Tensor], [])  # [x]
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)  # 这里为什么不直接连接运行完成

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets 不能为空")
            # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            # print(bboxes_out.is_contiguous())
            labels_out = targets['labels']
            # print(labels_out.is_contiguous())

            # ploc, plabel, gloc, glabel
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {"total_losses": loss}  # 组装成字典返回

        # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        # results = self.encoder.decode_batch(locs, confs)
        results = self.postprocess(locs, confs)
        return results


class SSD300_Test(SSD300):

    def forward(self, image, targets=None):
        '''

        :param image: 批画像
        :param targets:
        :return:
        '''
        x = self.feature_extractor(image)

        # Feature Map 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        detection_features = torch.jit.annotate(List[Tensor], [])  # [x]
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)  # 这里为什么不直接连接运行完成

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        return locs, confs


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        '''

        :param dboxes:  def 类对像
        '''
        super(Loss, self).__init__()
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10 这个值是实验出来的结果值 用于回归参数计算
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5  这个值是实验出来的结果值 用于回归参数计算

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # np高级与预测对齐 [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        unsqueeze = dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0)
        self.dboxes = nn.Parameter(unsqueeze, requires_grad=False)  # 这个def是原版的

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, loc):
        '''
        这里是 匹配的框 的第一步, 计划 anc与GT的调整系数
        计算ground truth相对anchors的回归参数
            self.dboxes 是def 是xywh self.dboxes
            两个参数只有前面几个用GT替代了的不一样 其它一个值 这里是稀疏
        :param loc: 已完成 def 匹配的框 n,4,8732 正例已修改为GT  N,4,8732
        :return:
            返回 ground truth相对anchors的回归参数 这里是稀疏
        '''
        # type: (Tensor)
        # 这里 scale_xy 是乘10
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        '''

        :param ploc: 预测的 N,4,8732
        :param plabel: 预测的  N,label_numx,8732
        :param gloc: 已完成 def 匹配的框 n,4,8732 正例已修改为GT
        :param glabel: 已完成 def 匹配的真实标签 0~20  n,8732  anchor匹配到的对应GTBOX的分数
        :return:
        '''
        # type: (Tensor, Tensor, Tensor, Tensor)
        # 计算gt的location回归参数 稀疏阵 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)

        # 获取正样本的mask  Tensor: [N, 8732]
        mask = glabel > 0
        # mask1 = torch.nonzero(glabel)

        # batch中的每张图片的正样本个数 Tensor: [N] 降维
        pos_num = mask.sum(dim=1)  # 这个用于统计布尔标签总数

        # -------------计算定位损失------------------(只有正样本) 很多0
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732] 降维
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N] 只计算正例

        # -------------计算分类损失------------------正样本很少
        con = self.confidence_loss(plabel, glabel)  # Tenosr: [N, 8732]
        # 负样本选取  选损失最大的
        con_neg = con.clone()
        con_neg[mask] = torch.tensor(0.0).to(con)  # 正样本先置0 使其独立

        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)  # descending 倒序
        _, con_rank = con_idx.sort(dim=1)  # 两次索引排序 用于取最大的n个布尔索引
        # 负样本数是正样本的3倍，但不能超过总样本数8732
        neg_num = torch.clamp(NEG_RATIO * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num  # Tensor [N, 8732]

        # confidence 最终loss使用选取的正样本loss+ 选取的负样本loss
        mask_z = mask.float() + neg_mask.float()  # 这两个是独立的 不干绕加
        con_loss = (con * (mask_z)).sum(dim=1)  # Tensor [N]

        # 总损失为[n]
        total_loss = loc_loss + con_loss

        # 没有正样本的图像不计算分类损失  eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=torch.finfo(torch.float).eps)
        # 每个图片平均一下
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret


if __name__ == '__main__':
    from f_tools.model.analyse import f_tensorwatch, f_summary

    # model = resnet50()
    backbone = Backbone()
    model = SSD300_Test(backbone)  # 这个分析不了
    # f_summary(model)
    f_tensorwatch(model, 'model_analyse.xlsx', size=(1, 3, 300, 300))
