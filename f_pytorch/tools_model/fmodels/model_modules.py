import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from f_pytorch.tools_model.f_model_api import conv2d, conv_bn_no_relu, conv_bn, CBL


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.relu6(x + 3.0) / 6.0


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


class SPPv2(torch.nn.Module):
    '''
    一个输入 由多个尺寸的核 池化后 再进行堆叠
    '''

    def __init__(self, seq='asc'):
        super(SPPv2, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq

    def __call__(self, x):
        x_1 = x
        x_2 = F.max_pool2d(input=x, kernel_size=5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, 1, 4)
        x_4 = F.max_pool2d(x, 13, 1, 6)
        if self.seq == 'desc':
            out = torch.cat([x_4, x_3, x_2, x_1], dim=1)
        else:
            out = torch.cat([x_1, x_2, x_3, x_4], dim=1)
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


class SAM(nn.Module):
    """ conv -> sigmoid * x -> 对原值进行缩放  Parallel CBAM from yolov4"""

    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Spatial Attention Module """
        x_attention = self.conv(x)

        return x * x_attention


class _Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBL(c1, c_, ksize=1)
        self.cv2 = CBL(c_, c2, ksize=3, padding=1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBL(c1, c_, ksize=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)
        self.cv4 = CBL(2 * c_, c2, ksize=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class DCNv2(torch.nn.Module):
    '''
    咩酱自实现的DCNv2，咩酱的得意之作，Pytorch的纯python接口实现，效率极高。
    '''

    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 distribution='normal',
                 gain=1):
        super(DCNv2, self).__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.conv_offset = torch.nn.Conv2d(input_dim, filter_size * filter_size * 3, kernel_size=filter_size,
                                           stride=stride,
                                           padding=padding, bias=True)
        # 初始化代码摘抄自SOLOv2  mmcv/cnn/weight_init.py  里的代码
        torch.nn.init.constant_(self.conv_offset.weight, 0.0)
        torch.nn.init.constant_(self.conv_offset.bias, 0.0)

        self.sigmoid = torch.nn.Sigmoid()

        self.dcn_weight = torch.nn.Parameter(torch.randn(filters, input_dim, filter_size, filter_size))
        self.dcn_bias = None
        if bias_attr:
            self.dcn_bias = torch.nn.Parameter(torch.randn(filters, ))
            torch.nn.init.constant_(self.dcn_bias, 0.0)
        if distribution == 'uniform':
            torch.nn.init.xavier_uniform_(self.dcn_weight, gain=gain)
        else:
            torch.nn.init.xavier_normal_(self.dcn_weight, gain=gain)

    def gather_nd(self, input, index):
        # 不被合并的后面的维
        keep_dims = []
        # 被合并的维
        first_dims = []
        dim_idx = []
        dims = index.shape[1]
        for i, number in enumerate(input.shape):
            if i < dims:
                dim_ = index[:, i]
                dim_idx.append(dim_)
                first_dims.append(number)
            else:
                keep_dims.append(number)

        # 为了不影响输入index的最后一维，避免函数副作用
        target_dix = torch.zeros((index.shape[0],), dtype=torch.long, device=input.device) + dim_idx[-1]
        new_shape = (-1,) + tuple(keep_dims)
        input2 = torch.reshape(input, new_shape)
        mul2 = 1
        for i in range(dims - 1, 0, -1):
            mul2 *= first_dims[i]
            target_dix += mul2 * dim_idx[i - 1]
        o = input2[target_dix]
        return o

    def forward(self, x):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias

        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
        mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
        mask = self.sigmoid(mask)

        # ===================================
        N, in_C, H, W = x.shape
        out_C, in_C, kH, kW = dcn_weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride

        # 1.先对图片x填充得到填充后的图片pad_x
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = torch.zeros((N, in_C, pad_x_H, pad_x_W), dtype=torch.float32, device=x.device)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x

        # 卷积核中心点在pad_x中的位置
        rows = torch.arange(0, out_W, dtype=torch.float32, device=dcn_weight.device) * stride + padding
        cols = torch.arange(0, out_H, dtype=torch.float32, device=dcn_weight.device) * stride + padding
        rows = rows[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis].repeat((1, out_H, 1, 1, 1))
        cols = cols[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis].repeat((1, 1, out_W, 1, 1))
        start_pos_yx = torch.cat([cols, rows], dim=-1)  # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = start_pos_yx.repeat((N, 1, 1, kH * kW, 1))  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置

        # 卷积核内部的偏移
        half_W = (kW - 1) // 2
        half_H = (kW - 1) // 2
        rows2 = torch.arange(0, kW, dtype=torch.float32, device=dcn_weight.device) - half_W
        cols2 = torch.arange(0, kH, dtype=torch.float32, device=dcn_weight.device) - half_H
        rows2 = rows2[np.newaxis, :, np.newaxis].repeat((kH, 1, 1))
        cols2 = cols2[:, np.newaxis, np.newaxis].repeat((1, kW, 1))
        filter_inner_offset_yx = torch.cat([cols2, rows2], dim=-1)  # [kH, kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = torch.reshape(filter_inner_offset_yx,
                                               (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = filter_inner_offset_yx.repeat(
            (N, out_H, out_W, 1, 1))  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移

        mask = mask.permute(0, 2, 3, 1)  # [N, out_H, out_W, kH*kW*1]
        offset = offset.permute(0, 2, 3, 1)  # [N, out_H, out_W, kH*kW*2]
        offset_yx = torch.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终位置。其实也不是最终位置，为了更快速实现DCNv2，还要给y坐标（代表行号）加上图片的偏移来一次性抽取，避免for循环遍历每一张图片。
        start_pos_y.requires_grad = False
        start_pos_x.requires_grad = False
        filter_inner_offset_y.requires_grad = False
        filter_inner_offset_x.requires_grad = False
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = torch.clamp(pos_y, 0.0, H + padding * 2 - 1.0)
        pos_x = torch.clamp(pos_x, 0.0, W + padding * 2 - 1.0)
        ytxt = torch.cat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

        pad_x = pad_x.permute(0, 2, 3, 1)  # [N, pad_x_H, pad_x_W, C]
        pad_x = torch.reshape(pad_x, (N * pad_x_H, pad_x_W, in_C))  # [N*pad_x_H, pad_x_W, C]

        ytxt = torch.reshape(ytxt, (N * out_H * out_W * kH * kW, 2))  # [N*out_H*out_W*kH*kW, 2]
        _yt = ytxt[:, :1]  # [N*out_H*out_W*kH*kW, 1]
        _xt = ytxt[:, 1:]  # [N*out_H*out_W*kH*kW, 1]

        # 为了避免使用for循环遍历每一张图片，还要给y坐标（代表行号）加上图片的偏移来一次性抽取出更兴趣的像素。
        row_offset = torch.arange(0, N, dtype=torch.float32, device=dcn_weight.device) * pad_x_H  # [N, ]
        row_offset = row_offset[:, np.newaxis, np.newaxis].repeat(
            (1, out_H * out_W * kH * kW, 1))  # [N, out_H*out_W*kH*kW, 1]
        row_offset = torch.reshape(row_offset, (N * out_H * out_W * kH * kW, 1))  # [N*out_H*out_W*kH*kW, 1]
        row_offset.requires_grad = False
        _yt += row_offset

        _y1 = torch.floor(_yt)
        _x1 = torch.floor(_xt)
        _y2 = _y1 + 1.0
        _x2 = _x1 + 1.0
        _y1x1 = torch.cat([_y1, _x1], -1)
        _y1x2 = torch.cat([_y1, _x2], -1)
        _y2x1 = torch.cat([_y2, _x1], -1)
        _y2x2 = torch.cat([_y2, _x2], -1)

        _y1x1_int = _y1x1.long()  # [N*out_H*out_W*kH*kW, 2]
        v1 = self.gather_nd(pad_x, _y1x1_int)  # [N*out_H*out_W*kH*kW, in_C]
        _y1x2_int = _y1x2.long()  # [N*out_H*out_W*kH*kW, 2]
        v2 = self.gather_nd(pad_x, _y1x2_int)  # [N*out_H*out_W*kH*kW, in_C]
        _y2x1_int = _y2x1.long()  # [N*out_H*out_W*kH*kW, 2]
        v3 = self.gather_nd(pad_x, _y2x1_int)  # [N*out_H*out_W*kH*kW, in_C]
        _y2x2_int = _y2x2.long()  # [N*out_H*out_W*kH*kW, 2]
        v4 = self.gather_nd(pad_x, _y2x2_int)  # [N*out_H*out_W*kH*kW, in_C]

        lh = _yt - _y1  # [N*out_H*out_W*kH*kW, 1]
        lw = _xt - _x1
        hh = 1 - lh
        hw = 1 - lw
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # [N*out_H*out_W*kH*kW, in_C]
        mask = torch.reshape(mask, (N * out_H * out_W * kH * kW, 1))
        value = value * mask
        value = torch.reshape(value, (N, out_H, out_W, kH, kW, in_C))
        new_x = value.permute(0, 1, 2, 5, 3, 4)  # [N, out_H, out_W, in_C, kH, kW]

        # 旧的方案，使用逐元素相乘，慢！
        # new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))  # [N, out_H, out_W, in_C * kH * kW]
        # new_x = new_x.permute(0, 3, 1, 2)  # [N, in_C*kH*kW, out_H, out_W]
        # exp_new_x = new_x.unsqueeze(1)  # 增加1维，[N,      1, in_C*kH*kW, out_H, out_W]
        # reshape_w = torch.reshape(dcn_weight, (1, out_C, in_C * kH * kW, 1, 1))  # [1, out_C,  in_C*kH*kW,     1,     1]
        # out = exp_new_x * reshape_w  # 逐元素相乘，[N, out_C,  in_C*kH*kW, out_H, out_W]
        # out = out.sum((2,))  # 第2维求和，[N, out_C, out_H, out_W]

        # 新的方案，用等价的1x1卷积代替逐元素相乘，快！
        new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))  # [N, out_H, out_W, in_C * kH * kW]
        new_x = new_x.permute(0, 3, 1, 2)  # [N, in_C*kH*kW, out_H, out_W]
        rw = torch.reshape(dcn_weight, (
            out_C, in_C * kH * kW, 1, 1))  # [out_C, in_C, kH, kW] -> [out_C, in_C*kH*kW, 1, 1]  变成1x1卷积核
        out = F.conv2d(new_x, rw, stride=1)  # [N, out_C, out_H, out_W]
        return out
