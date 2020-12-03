import torch
import torch.nn as nn

from f_pytorch.tools_model.fmodels.model_fpns import FPN
from f_pytorch.tools_model.fmodels.model_modules import SSH


class ClassHead(nn.Module):
    def __init__(self, inchannels, num_anchors, num_classes):
        super(ClassHead, self).__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * num_classes, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_classes)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)  # 直接将特图  torch([batch, 64, 80, 80]) 拉成 num_anchors * 4
        out = out.permute(0, 2, 3, 1).contiguous()  # 将值放在最后
        return out.view(out.shape[0], -1, 4)  # 将 anc 拉平   num_anchors * 4 *每个层的h*w 叠加


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2, num_keypoint=10):
        super(LandmarkHead, self).__init__()
        self.num_keypoint = num_keypoint
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * num_keypoint, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_keypoint)


class RetinaFace(nn.Module):
    def __init__(self, backbone, num_classes, anchor_num, in_channels_fpn, ssh_channel=256, use_keypoint=False):
        '''

        :param backbone:   这个的三层输出 in_channels_fpn
        :param return_layers: 定义 backbone 的输出名
        :param num_classes:
        :param anchor_num:
        :param in_channels_fpn: fpn的输入通道维度 数组对应输
        :param ssh_channel: FPN的输出 与SSH输出一致
        '''
        super(RetinaFace, self).__init__()
        self.backbone = backbone  # backbone转换
        self.fpn = FPN(in_channels_fpn, ssh_channel)
        self.ssh1 = SSH(ssh_channel, ssh_channel)
        self.ssh2 = SSH(ssh_channel, ssh_channel)
        self.ssh3 = SSH(ssh_channel, ssh_channel)

        self.ClassHead = self._make_class_head(fpn_num=len(in_channels_fpn),
                                               inchannels=ssh_channel,
                                               anchor_num=anchor_num,
                                               num_classes=num_classes)
        self.BboxHead = self._make_bbox_head(fpn_num=len(in_channels_fpn),
                                             inchannels=ssh_channel,
                                             anchor_num=anchor_num)

        self.use_keypoint = use_keypoint
        if use_keypoint:
            self.LandmarkHead = self._make_landmark_head(fpn_num=len(in_channels_fpn),
                                                         inchannels=ssh_channel,
                                                         anchor_num=anchor_num)

    def _make_class_head(self, fpn_num, inchannels, anchor_num, num_classes):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num, num_classes))
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
        '''

        :param inputs: tensor(batch,c,h,w)
        :return:

        '''
        out = self.backbone(inputs)  # 输出字典 {1:层1输出,2:层2输出,3:层2输出}

        # FPN 输出 3个特图的统一维度(超参) tuple(tensor(层1),tensor(层2),tensor(层3))
        fpn = self.fpn(out)

        # SSH 串联 ssh
        feature1 = self.ssh1(fpn[0])  # in torch.Size([8, 64, 80, 80]) out一致
        feature2 = self.ssh2(fpn[1])  # in torch.Size([8, 128, 40, 40]) out一致
        feature3 = self.ssh3(fpn[2])  # in torch.Size([8, 256, 20, 20]) out一致
        features = [feature1, feature2, feature3]

        # 为每一输出的特图进行预测,输出进行连接 torch.Size([batch, 16800, 4])
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # torch.Size([batch, 8400, 2]) 这里可以优化成一个值
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # torch.Size([batch, 16800, 10])

        if self.use_keypoint:
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        else:
            ldm_regressions = None
        # if torchvision._is_tracing():  #预测模式 训练时是不会满足的 训练和预测进行不同的处理
        output = (bbox_regressions, classifications, ldm_regressions)
        return output


if __name__ == '__main__':
    pass
