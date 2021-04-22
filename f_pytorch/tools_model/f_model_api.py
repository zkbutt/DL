from math import log

import torch
import torch.nn as nn

from collections import OrderedDict

'''-----------------模型方法区-----------------------'''


def f_freeze_bn(model_nn):
    '''Freeze BatchNorm layers. 冻结BN层'''
    for layer in model_nn.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def finit_conf_bias_one(model_nn, num_tolal, num_pos, num_classes):
    '''初始化分类层 使 conf 输出靠向负例'''
    # pi = num_pos / num_tolal / cfg.NUM_CLASSES
    b = -log(num_tolal / num_pos * num_classes - 1)
    model_nn.bias.data += b


def finit_conf_bias(model, num_tolal, num_pos, num_classes):
    '''初始化分类层 使 conf 输出靠向负例'''
    # pi = num_pos / num_tolal / cfg.NUM_CLASSES
    b = -log(num_tolal / num_pos * num_classes - 1)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            model.bias.data += b


def finit_weights(model):
    # for m in model.modules():
    #     t = type(m)
    #     if t is nn.Conv2d:
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     elif t is nn.BatchNorm2d:
    #         m.eps = 1e-4
    #         m.momentum = 0.03
    #     elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
    #         m.inplace = True

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight.data)
            # nn.init.xavier_normal_(m.weight, gain=1.0)  # 正态分布
            # nn.init.xavier_uniform_(m.weight, gain=1.)
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')  # 正态分布
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))  # 均匀分布
            # nn.init.xavier_normal_(m.weight.data)
            # nn.init.kaiming_normal_(m.weight.data)  # 卷积层参数初始化
            # m.bias.data.fill_(0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data)  # normal: mean=0, std=1
            # nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.neural_num))  ## normal: mean=0, std=1/n
            # nn.init.constant_(m.weight, 0)
            # nn.init.constant_(m.bias, 0)

            nn.init.uniform_(m.weight.data, -a, a)


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU()),
    ]))


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


'''-----------------模型层区-----------------------'''


class CBL(nn.Module):
    '''基础层'''

    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0, dilation=1, groups=1, leakyReLU=True):
        super(CBL, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation,
                      groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class NoneLayer(nn.Module):
    def __init__(self):
        super(NoneLayer, self).__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    '''
    CBAM
    Convolutional bottleneck attention module
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class ReorgLayer(nn.Module):
    '''宽高 self.stride缩小倍数 ->  转通道增加2*2=4 '''

    def __init__(self, stride):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same' 默认为自动same
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


'''-----------------模型转换-----------------------'''


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


class FModelBase(nn.Module):
    def __init__(self, net, losser, preder):
        super(FModelBase, self).__init__()
        self.net = net
        self.losser = losser
        self.preder = preder

    def forward(self, x, targets=None):
        outs = self.net(x)

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            '''------验证loss 待扩展------'''

            return loss_total, log_dict
        else:
            with torch.no_grad():  # 这个没用
                ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores


if __name__ == '__main__':
    pass
