import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from torchvision import models

from f_tools.GLOBAL_LOG import flog
from object_detection.retinaface.nets.mobilenet025 import MobileNetV1
from object_detection.retinaface.nets.layers import FPN, SSH


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, model_name, pretrain_path,
                 in_channels, out_channel, return_layers):
        '''

        :param in_channels: fpn的输入通道维度 数组对应输
        :param out_channel: FPN的输出 与SSH输出一致
        :param path_model_weight: backbone 权重文件路径
        :param return_layers:  定义 backbone 的输出名
        '''
        super(RetinaFace, self).__init__()
        if model_name == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pretrain_path and os.path.exists(pretrain_path):
                checkpoint = torch.load(pretrain_path, map_location=torch.device('cpu'))
                # checkpoint = torch.load(pretrain_path) # 这里不用cpu要报错
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # 移除后7层
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
                flog.debug('backbone  权重加载成功 %s', pretrain_path)
        elif model_name == 'Resnet50':
            backbone = models.resnet50()

        # ---根据字典抽取结构输出---
        # import torchvision.models._utils as _utils
        # self.body = _utils.IntermediateLayerGetter(backbone, {'stage1': 1, 'stage2': 2, 'stage3': 3})
        # 生成 IntermediateLayerGetter 对象 通过 out = self.body(inputs) 即可输出多个形成数组
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)  # backbone转换
        in_channels_fpn = [
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
        ]
        self.fpn = FPN(in_channels_fpn, out_channel)
        self.ssh1 = SSH(out_channel, out_channel)
        self.ssh2 = SSH(out_channel, out_channel)
        self.ssh3 = SSH(out_channel, out_channel)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channel)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channel)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channel)

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN out OrderdDict
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])  # in torch.Size([8, 64, 80, 80]) out一致
        feature2 = self.ssh2(fpn[1])  # in torch.Size([8, 128, 40, 40]) out一致
        feature3 = self.ssh3(fpn[2])  # in torch.Size([8, 256, 20, 20]) out一致
        features = [feature1, feature2, feature3]

        # torch.Size([8, 16800, 4])
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # torch.Size([8, 16800, 2])
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # torch.Size([8, 16800, 10])
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # if torchvision._is_tracing():  #预测模式 训练时是不会满足的 训练和预测进行不同的处理
        if self.training:
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output


if __name__ == '__main__':
    pass
