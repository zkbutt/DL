import torch
import torch.nn as nn
from collections import OrderedDict

from f_pytorch.backbone_t.model_look import f_look
from object_detection.yolo3.nets.darknet import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    '''
    5次卷积出结果
    :param filters_list:[512, 1024] 交替Conv
    :param in_filters: 1024 输入维度
    :param out_filter: 输出 维度 （20+1+4）*anc数
    :return:
    '''
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),  # 1卷积 + 3卷积 交替
        conv2d(filters_list[0], filters_list[1], 3),  #
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                  stride=1, padding=0, bias=True)
    ])
    return m


class YoloBody(nn.Module):
    def __init__(self, backbone, nums_anc, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = backbone
        self.nums_anc = nums_anc

        out_filters = self.backbone.layers_out_filters
        '''根据不同的输出层的anc参数，确定输出结果'''
        #  final_out_filter3 最终输出层输出维度，根据不同的anc尺寸数量 * （类别数 + 1 + 4）
        final_out_filter3 = nums_anc[0] * (1 + 4 + num_classes)
        self.last_layer3 = make_last_layers([512, 1024], out_filters[-1], final_out_filter3)

        #  第二输出定义
        final_out_filter2 = nums_anc[1] * (1 + 4 + num_classes)
        self.last_layer2_conv = conv2d(512, 256, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter2)

        #  第一输出定义
        final_out_filter1 = nums_anc[2] * (1 + 4 + num_classes)
        self.last_layer1_conv = conv2d(256, 128, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter1)

    def forward(self, x):
        def _branch(last_layer, layer_in):
            '''

            :param last_layer: 五层CONV
            :param layer_in: 输入数据
            :return:
                layer_in 输出数据
                out_branch 上采样输入
            '''
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)  #
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        #  backbone torch.Size([5, 256, 52, 52]) torch.Size([5, 512, 26, 26])  torch.Size([5, 1024, 13, 13])
        backbone_out1, backbone_out2, backbone_out3 = self.backbone(x)  # out3, out4, out5
        #  yolo 终层处理
        out3, out3_branch = _branch(self.last_layer3, backbone_out3)  # torch.Size([5, 75, 13, 13])

        #  yolo 第二层处理
        _out = self.last_layer2_conv(out3_branch)
        _out = self.last_layer2_upsample(_out)
        _out = torch.cat([_out, backbone_out2], 1)
        out2, out2_branch = _branch(self.last_layer2, _out)  # torch.Size([5, 75, 26, 26])

        #  yolo 第一层处理
        _out = self.last_layer1_conv(out2_branch)
        _out = self.last_layer1_upsample(_out)
        _out = torch.cat([_out, backbone_out1], 1)
        out1, _ = _branch(self.last_layer1, _out)  # torch.Size([5, 75, 52, 52])
        outs = self.data_packaging([out1, out2, out3], self.nums_anc)

        return outs

    def data_packaging(self, outs, nums_anc):
        '''

        :param outs:
        :param nums_anc: ans数组
        :return: torch.Size([5, 10647, 25])
        '''
        _ts = []
        for out, num_anc in zip(outs, nums_anc):
            batch, o, w, h = out.shape  # torch.Size([5, 75, 52, 52])
            out = out.permute(0, 2, 3, 1)  # torch.Size([5, 52, 52, 75])
            _ts.append(out.reshape(batch, -1, int(o / num_anc)).contiguous()) # torch.Size([5, 52*52*3, 25])
        # torch.Size([5, 2704, 75])，torch.Size([5, 676, 75])，torch.Size([5, 169, 75]) -> torch.Size([5, 3549, 75])
        return torch.cat(_ts, dim=1)


if __name__ == '__main__':
    from object_detection.yolo3.utils.config import Config

    model = YoloBody(Config, [3, 3, 3], 20)
    f_look(model)
