import torch
import torch.nn as nn
from collections import OrderedDict

from f_pytorch.backbone_t.f_model_api import SPP
from f_pytorch.backbone_t.f_models.darknet import Darknet
from f_pytorch.backbone_t.model_look import f_look, f_look2
from object_detection.yolo3.nets.darknet import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU()),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    '''
    5次卷积出结果
    :param filters_list:[512, 1024] 交替Conv
    :param in_filters: 1024 输入维度
    :param out_filter: 输出 维度 （20+1+4）*anc数
    :return:
    '''
    m = nn.ModuleList([  # 共7层
        conv2d(in_filters, filters_list[0], 1),  # 1卷积 + 3卷积 交替
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),  # 这里加spp
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),  # 这里输出上采样
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    ])
    return m


class YoloV3SPP(nn.Module):
    def __init__(self, backbone, nums_anc, num_classes, dims_rpn_in, is_spp=False):
        '''
        层属性可以是 nn.Module nn.ModuleList(封装Sequential) nn.Sequential
        '''
        super(YoloV3SPP, self).__init__()
        assert len(dims_rpn_in) == 3, 'yolo_in_dims 的长度必须是3'
        #  backbone
        self.backbone = backbone
        self.nums_anc = nums_anc
        if is_spp:
            self.is_spp = is_spp
            self.spp = SPP(int(dims_rpn_in[2] / 2))
        else:
            is_spp = None

        '''根据不同的输出层的anc参数，确定输出结果'''
        final_out_filter1 = nums_anc[0] * (1 + 4 + num_classes)
        self.last_layer1 = make_last_layers([int(dims_rpn_in[0] / 2), dims_rpn_in[0]],
                                            dims_rpn_in[0] + int(dims_rpn_in[1] / 4),  # # 叠加上层的输出
                                            final_out_filter1)

        final_out_filter2 = nums_anc[1] * (1 + 4 + num_classes)
        self.last_layer2 = make_last_layers([int(dims_rpn_in[1] / 2), dims_rpn_in[1]],  # 小大震荡
                                            dims_rpn_in[1] + int(dims_rpn_in[2] / 4),  # 叠加上层的输出
                                            final_out_filter2)
        self.last_layer2_conv = conv2d(int(dims_rpn_in[1] / 2), int(dims_rpn_in[1] / 4), 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        final_out_filter3 = nums_anc[2] * (1 + 4 + num_classes)
        self.last_layer3 = make_last_layers([int(dims_rpn_in[2] / 2), dims_rpn_in[2]],  # 小大震荡 输入的一半 and 还原
                                            dims_rpn_in[2],
                                            final_out_filter3)
        self.last_layer3_conv = conv2d(int(dims_rpn_in[2] / 2), int(dims_rpn_in[2] / 4), 1)  # 决定上采的维度
        # F.interpolate(x,scale_factor=2,mode='nearest')
        self.last_layer3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.sigmoid_out = nn.Sigmoid()

    def forward(self, x):
        def _branch(last_layer, layer_in, is_spp=False):
            '''

            :param last_layer: 五层CONV
            :param layer_in: 输入数据
            :return:
                layer_in 输出数据
                out_branch 上采样输入
            '''
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 2 and is_spp:
                    layer_in = self.spp(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        #  backbone ([batch, 512, 52, 52]) ([batch, 1024, 26, 26])  ([batch, 1024, 13, 13])
        backbone_out1, backbone_out2, backbone_out3 = self.backbone(x)
        #  [batch, 1024, 13, 13] -> [batch, 75, 13, 13],[batch, 2048, 13, 13]
        out3, out3_branch = _branch(self.last_layer3, backbone_out3, self.is_spp)
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
        '''这里输出每层每格的对应三个anc'''
        outs[:, :, :2] = self.sigmoid_out(outs[:, :, :2])  # xy归一
        outs[:, :, 4:] = self.sigmoid_out(outs[:, :, 4:])  # 支持多标签
        '''为每一个特图预测三个尺寸的框,拉平堆叠'''
        return outs

    def data_packaging(self, outs, nums_anc):
        '''
        3个输入 合成一个输出 与anc进行拉伸
        :param outs: [out1, out2, out3] b,c,h,w
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


if __name__ == '__main__':
    # model = models.densenet121(pretrained=True)
    # dim_layer1_out = model.features.transition2.conv.in_channels  # 512
    # dim_layer2_out = model.features.transition3.conv.in_channels  # 1024
    # dim_layer3_out = model.classifier.in_features  # 1024
    # dims_out = [dim_layer1_out, dim_layer2_out, dim_layer3_out]
    # model = Output4Densenet(model)

    # model = models.resnext50_32x4d(pretrained=True)
    # return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    # model = Output4Return(model, return_layers)
    # dims_out = [512, 1024, 2048]

    model = Darknet(nums_layer=(1, 2, 8, 8, 4))
    return_layers = {'block3': 1, 'block4': 2, 'block5': 3}
    model = Output4Return(model, return_layers)
    dims_out = [256, 512, 1024]

    nums_anc = [3, 3, 3]
    num_classes = 20
    model = YoloV3SPP(model, nums_anc, num_classes, dims_out, is_spp=True)
    # f_look(model, input=(1, 3, 416, 416))
    # f_look2(model, input=(3, 416, 416))

    # torch.save(model, 'yolov3spp.pth')
