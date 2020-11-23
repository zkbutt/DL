import torch
import torch.nn as nn
from torchvision.models import _utils
from collections import OrderedDict

'''-----------------模型方法区-----------------------'''


def init_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-4
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU6(inplace=True), # 最大值为6
        nn.LeakyReLU(inplace=True),  # 负x轴给一个固定的斜率
        # nn.RReLU(inplace=True),  # 负x轴给定范围内随机
    )


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU()),
    ]))


'''-----------------模型层区-----------------------'''


class SPP(nn.Module):
    '''
    长宽不变, 深度是原来的4倍 calc_oshape_pytorch
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


class Output4Densenet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        layer1_od = OrderedDict()
        layer2_od = OrderedDict()
        layer3_od = OrderedDict()
        # pool = AvgPool2d(kernel_size=2, stride=2, padding=0)

        for name1, module1 in backbone._modules.items():
            if name1 == 'features':
                _od = layer1_od
                for name2, module2 in module1._modules.items():
                    if name2 == 'transition2':
                        _od = layer2_od
                    elif name2 == 'transition3':
                        _od = layer3_od
                    elif name2 == 'norm5':
                        break
                    _od[name2] = module2
            break

        self.layer1 = nn.Sequential(layer1_od)
        self.layer2 = nn.Sequential(layer2_od)
        self.layer3 = nn.Sequential(layer3_od)

    def forward(self, inputs):
        out1 = self.layer1(inputs)  # torch.Size([1, 512, 52, 52])
        out2 = self.layer2(out1)  # torch.Size([1, 1024, 26, 26])
        out3 = self.layer3(out2)
        return out1, out2, out3


class Output4Return(nn.Module):

    def __init__(self, backbone, return_layers) -> None:
        super().__init__()
        self.layer_out = _utils.IntermediateLayerGetter(backbone, return_layers)

    def forward(self, inputs):
        out = self.layer_out(inputs)
        out = list(out.values())
        return out


'''-----------------模型组合-----------------------'''


def x_model_group():
    model = nn.Sequential()
    model.add_module('conv', nn.Conv2d(3, 3, 3))
    model.add_module('batchnorm', nn.BatchNorm2d(3))
    model.add_module('activation_layer', nn.ReLU())

    model = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.BatchNorm2d(3),
        nn.ReLU()
    )

    from collections import OrderedDict
    model = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(3, 3, 3)),
        ('batchnorm', nn.BatchNorm2d(3)),
        ('activation_layer', nn.ReLU())
    ]))

    # ModuleList 类似 list ，内部没有实现 forward 函数
    model = nn.ModuleList([nn.Linear(3, 4),
                           nn.ReLU(),
                           nn.Linear(4, 2)])
