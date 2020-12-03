import torch
from torch import nn

from f_pytorch.tools_model.f_model_api import conv2d, conv_bn_no_relu, conv_bn


class SSH(nn.Module):
    '''
    与SPP类似 多尺寸卷积堆叠
    长宽不变 深度不变
    '''

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, inputs):
        conv3X3 = self.conv3X3(inputs)

        conv5X5_1 = self.conv5X5_1(inputs)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = torch.relu(out)
        return out


class SPP(nn.Module):
    '''
    与SPP类似 长宽不变, 深度是原来的4倍 calc_oshape_pytorch
    '''

    def __init__(self, in_channel):
        super(SPP, self).__init__()
        self.maxpool5X5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool9X9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool13X13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.spp_out = conv2d(in_channel * 4, in_channel, kernel_size=1)

    def forward(self, inputs):
        maxpool5X5 = self.maxpool5X5(inputs)  # torch.Size([1, 512, 13, 13])
        maxpool9X9 = self.maxpool9X9(inputs)
        maxpool13X13 = self.maxpool13X13(inputs)
        out = torch.cat([inputs, maxpool5X5, maxpool9X9, maxpool13X13], dim=1)
        out = self.spp_out(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    '''
    CBAM
    通道注意力机制
    '''

    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    '''
    CBAM
    空间注意力机制
    '''

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
