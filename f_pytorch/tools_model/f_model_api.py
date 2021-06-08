from math import log

import torch
import torch.nn as nn

from collections import OrderedDict

from f_pytorch.tools_model.fmodels.init_weights import kaiming_init, constant_init
from f_pytorch.tools_model.fun_regulariz import AffineChannel
from f_pytorch.tools_model.f_fun_activation import Mish, SiLU

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


'''-----------------模型层区-----------------------'''


class FConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 k,  # kernel_size
                 s=1,  # stride
                 p=None,  # padding
                 d=1,  # dilation空洞
                 g=1,  # g groups 一般不动
                 is_bias=False,
                 norm='bn',  # bn,gn,af
                 act='leaky',  # relu leaky mish silu
                 is_freeze=False,
                 use_dcn=False):
        '''
        有 bn 建议不要 bias
        # 同维
        Conv2dUnit(in_channels_list[0], out_channels, k=1, bn=True, act='leaky', is_bias=False)
        # 降维
        Conv2dUnit(in_channels_list[2], feature_size, k=3, s=2, p=1, bn=True, act='leaky', is_bias=False)
        '''
        super(FConv2d, self).__init__()
        self.groups = g
        self.act = act
        self.is_freeze = is_freeze
        self.use_dcn = use_dcn

        if p is None:
            # conv默认为0
            p = conv_same(k)

        # conv
        if use_dcn:
            pass
        else:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s,
                                        padding=p, dilation=d, bias=is_bias)
        # 正则化方式 normalization
        if norm == 'bn':
            self.normalization = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.normalization = torch.nn.GroupNorm(num_groups=g, num_channels=out_channels)
        elif norm == 'af':
            self.normalization = AffineChannel(out_channels)
        else:
            self.normalization = None

        # act
        if act == 'relu':
            self.act = torch.nn.ReLU(inplace=True)
        elif act == 'leaky':
            self.act = torch.nn.LeakyReLU(0.1, inplace=True)
        elif act == 'mish':
            self.act = Mish()
        elif act == 'silu':
            self.act = SiLU()
        else:
            self.act = None

        self.name_act = act

        if is_freeze:
            self.freeze()

    def init_weights(self):
        if self.name_act == 'leaky':
            nonlinearity = 'leaky_relu'
        elif self.name_act == 'leaky':
            nonlinearity = 'relu'
        else:
            raise Exception('self.name_act 错误%s' % self.name_act)

        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.bn:
            constant_init(self.norm, 1, bias=0)

    def freeze(self):
        # 冻结
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
        if self.bn is not None:
            self.bn.weight.requires_grad = False
            self.bn.bias.requires_grad = False
        if self.gn is not None:
            self.gn.weight.requires_grad = False
            self.gn.bias.requires_grad = False
        if self.af is not None:
            self.af.weight.requires_grad = False
            self.af.bias.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x


class NoneLayer(nn.Module):
    def __init__(self):
        super(NoneLayer, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    # 在全局平均池化以后使用，去掉2个维度
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


def conv_same(k):
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, bias=True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, conv_same(k), groups=g, bias=bias)
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
