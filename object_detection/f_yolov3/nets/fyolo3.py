import torch
import torch.nn as nn
from collections import OrderedDict

from torchvision import models
from torchvision.models import _utils

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
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),  # 这里输出上采样
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                  stride=1, padding=0, bias=True)
    ])
    return m


class YoloBody(nn.Module):
    def __init__(self, backbone, nums_anc, num_classes, dims_rpn_in):
        super(YoloBody, self).__init__()
        assert len(dims_rpn_in) == 3, 'yolo_in_dims 的长度必须是3'
        #  backbone
        self.backbone = backbone
        self.nums_anc = nums_anc

        '''根据不同的输出层的anc参数，确定输出结果'''
        dim_out = dims_rpn_in[-1]
        final_out_filter1 = nums_anc[0] * (1 + 4 + num_classes)
        self.last_layer1 = make_last_layers([int(dim_out / (2 * 2 * 2)), int(dim_out / (2 * 2))],
                                            dims_rpn_in[0],
                                            final_out_filter1)

        final_out_filter2 = nums_anc[1] * (1 + 4 + num_classes)
        self.last_layer2 = make_last_layers([int(dim_out / 4), int(dim_out / 2)],
                                            dims_rpn_in[1] + int(dim_out / 4),  # 输入叠加
                                            final_out_filter2)
        self.last_layer2_conv = conv2d(int(dim_out / 2), int(dim_out / (2 * 2)), 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        final_out_filter3 = nums_anc[2] * (1 + 4 + num_classes)
        self.last_layer3 = make_last_layers([int(dim_out / 2), dim_out],
                                            dims_rpn_in[2],
                                            final_out_filter3)
        self.last_layer3_conv = conv2d(int(dim_out/2), int(dim_out / 4), 1)  # 决定上采的维度
        self.last_layer3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

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

        #  backbone ([batch, 512, 52, 52]) ([batch, 1024, 26, 26])  ([batch, 1024, 13, 13])
        backbone_out1, backbone_out2, backbone_out3 = self.backbone(x)
        #  [batch, 1024, 13, 13] -> [batch, 75, 13, 13],[batch, 2048, 13, 13]
        out3, out3_branch = _branch(self.last_layer3, backbone_out3)
        # [batch, 2048, 13, 13] -> [batch, 1024, 13, 13]
        _out = self.last_layer3_conv(out3_branch)
        _out = self.last_layer3_upsample(_out)
        _out = torch.cat([_out, backbone_out2], 1)  # 叠加 torch.Size([1, 1280, 26, 26])
        out2, out2_branch = _branch(self.last_layer2, _out)  # torch.Size([batch, 75, 26, 26])

        _out = self.last_layer2_conv(out2_branch)  # torch.Size([1, 128, 26, 26])
        _out = self.last_layer2_upsample(_out)  # torch.Size([batch, 128, 26, 26]) -> torch.Size([1, 128, 52, 52])
        _out = torch.cat([_out, backbone_out1], 1)  # 叠加
        out1, _ = _branch(self.last_layer1, _out)  # torch.Size([batch, 75, 52, 52])

        # 自定义数据重装函数 torch.Size([1, 10647, 25])
        outs = self.data_packaging([out1, out2, out3], self.nums_anc)
        return outs

    def data_packaging(self, outs, nums_anc):
        '''
        3个输入 合成一个输出 与anc进行拉伸
        :param outs: [out1, out2, out3]
        :param nums_anc: ans数组
        :return: torch.Size([1, 10647, 25])
        '''
        _ts = []
        for out, num_anc in zip(outs, nums_anc):
            batch, o, w, h = out.shape  # torch.Size([5, 75, 52, 52])
            out = out.permute(0, 2, 3, 1)  # torch.Size([5, 52, 52, 75])
            # [5, 52, 52, 75] -> [5, 52*52*3, 25]
            _ts.append(out.reshape(batch, -1, int(o / num_anc)).contiguous())
        # torch.Size([1, 8112, 25])，torch.Size([1, 2028, 25])，torch.Size([1, 507, 25]) -> torch.Size([1, 10647, 25])
        return torch.cat(_ts, dim=1)


class OIMO4Densenet(nn.Module):

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


class OIMO4Resnext(nn.Module):

    def __init__(self, backbone, return_layers) -> None:
        super().__init__()
        self.layer_out = _utils.IntermediateLayerGetter(backbone, return_layers)

    def forward(self, inputs):
        out = self.layer_out(inputs)
        out = list(out.values())
        return out


if __name__ == '__main__':
    # model = models.densenet121(pretrained=True)
    # dim_layer1_out = model.features.transition2.conv.in_channels  # 512
    # dim_layer2_out = model.features.transition3.conv.in_channels  # 1024
    # dim_layer3_out = model.classifier.in_features  # 1024
    # dims_out = [dim_layer1_out, dim_layer2_out, dim_layer3_out]
    # model = OIMO4Densenet(model)

    model = models.resnext50_32x4d(pretrained=True)
    return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    model = OIMO4Resnext(model, return_layers)
    dims_out = [512, 1024, 2048]

    nums_anc = [3, 3, 3]
    num_classes = 20
    model = YoloBody(model, nums_anc, num_classes, dims_out)
    f_look(model, input=(1, 3, 416, 416))
