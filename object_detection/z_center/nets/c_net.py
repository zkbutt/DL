import torch
from torch import nn
import torch.nn.functional as f
import torchvision.models as models
import numpy as np

"""
这个文件是centerNet的网络结构
"""

# 预训练模型的路径
# BACKBONE = "G:/工作空间/预训练模型/resnet18-5c106cde.pth"


class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x


class CenterNet(nn.Module):
    # backbone是预训练模型的路径
    # class_num是分类数量，voc数据集中分类数量是20
    # feature是上采样之后卷积层的通道数
    def __init__(self, backbone=None, class_num=20):
        super(CenterNet, self).__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        # [1,3,500,500] -> [1,256,32,32]
        self.stage1 = nn.Sequential(*list(self.backbone.children())[:-3])

        """
        # [1,64,125,125] -> [1,128,63,63]
        self.stage2 = nn.Sequential(list(backbone.children())[-5])
        # [1,128,63,63] -> [1,256,32,32]
        self.stage3 = nn.Sequential(list(backbone.children())[-4])
        """

        # 改变通道数
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

        batchNorm_momentum = 0.1
        self.block = nn.Sequential(
            SepConv(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(),
        )
        # head的内容
        self.head = nn.Sequential(
            self.block,
            self.block,
            self.block,
            self.block
        )
        # 分类预测
        self.head_cls = nn.Conv2d(64, class_num, kernel_size=3, padding=1, stride=1)
        # 偏移量修正预测
        self.head_offset = nn.Conv2d(64, 2, kernel_size=3, padding=1, stride=1)
        # 回归框大小预测
        self.head_size = nn.Conv2d(64, 2, kernel_size=3, padding=1, stride=1)

    # 上采样，mode参数默认的是"nearest",使用mode="bilinear"的时候会有warning
    def upsampling(self, src, width, height, mode="nearest"):
        # target的形状举例 torch.Size([1, 256, 50, 64])
        return f.interpolate(src, size=[width, height], mode=mode)

    def forward(self, input):
        output = self.stage1(input)
        # 将通道数由256变为128
        output = self.conv1(output)
        width = input.shape[2] // 8
        height = input.shape[3] // 8
        output = self.upsampling(output, width, height)
        # 将通道数由128变为64
        output = self.conv2(output)
        width = input.shape[2] // 4
        height = input.shape[3] // 4
        output = self.upsampling(output, width, height)
        output = self.head(output)
        # 分类预测
        classes = self.head_cls(output)
        # 偏移量预测
        offset = self.head_offset(output)
        # 回归框大小预测
        size = self.head_size(output)
        # 由于分类值输出在[0,1]之间，所以需要使用sigmoid函数
        # classes = nn.Sigmoid()(classes)
        # 使用softmax函数
        classes = self.softmax(classes)
        # 回归值为正
        size = torch.exp(size)
        return classes, offset, size


if __name__ == "__main__":
    network = CenterNet()
    img = torch.rand(1, 3, 512, 512)
    output = network(img)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
