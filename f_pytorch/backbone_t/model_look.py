import os
from collections import OrderedDict

import torch
import torchvision.models as models
from torch import nn
from torch.nn import AvgPool2d

from f_pytorch.backbone_t.f_models.darknet import darknet53
from f_pytorch.backbone_t.f_models.mobilenet025 import MobileNetV1
from f_tools.GLOBAL_LOG import flog


class FModelOne2More(nn.Module):

    def __init__(self, backbone, return_layers):
        super().__init__()
        import torchvision.models._utils as _utils
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)  # backbone转换

    def forward(self, inputs):
        out = self.body(inputs)
        d = 1
        return out


class FRebuild4densenet161(nn.Module):

    def __init__(self, backbone, return_layers):
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
        out = self.body(inputs)
        return out

    def handler_pool(self, name, module, od):
        pool = module._modules.pop('pool')
        od[name] = module
        return pool


def f替换(model):
    features = model.classifier.in_features  # 提取输入维度
    model.classifier = nn.Linear(features, 5)


def other():
    pass
    # 修正版本报错
    # torch.onnx.set_training = torch.onnx.select_model_mode_for_export
    # 生成模型结构图
    # img = tw.draw_model(model, [1, 3, 224, 224])
    # print(type(img))
    # img.save(r'model.jpg')
    # # GRAPHVIZ+TORCHVIZ
    # x = torch.rand(8, 3, 256, 512)
    # y = model(x)


def f_look(model, input=(1, 3, 416, 416)):
    import tensorwatch as tw

    # 用这个即可---查看网络的统计结果---
    args_pd = tw.model_stats(model, input)
    args_pd.to_excel('model_look.xlsx')


if __name__ == '__main__':
    '''
    '''
    # data_inputs_list = [1, 3, 640, 640]
    data_inputs_list = [1, 3, 416, 416]
    torch.random.manual_seed(20201025)  # 3746401707500
    data_inputs_ts = torch.rand(data_inputs_list, dtype=torch.float)

    # model = models.densenet161(pretrained=True)  # 能力 22.35  6.20  ---top2
    # model = FRebuild4densenet161(model, None)
    # return_layers = {'layer1': 1, 'layer2': 2, 'layer3': 3}

    # model = models.wide_resnet50_2(pretrained=True)  # 能力 21.49 5.91  ---top1
    # model = models.resnext50_32x4d(pretrained=True)  # 能力 22.38 6.30 ---top3
    # model = models.mobilenet_v2(pretrained=True)  # 能力 28.12 9.71 ---速度top1

    # model = models.resnet50(pretrained=True)  # 下采样倍数32 能力23.85 7.13
    # return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    # my_model = FModelOne2More(model, return_layers)
    # data_outputs = my_model(data_inputs_ts)
    # for k, v in data_outputs.items():
    #     print(v.shape)

    # f替换(model)

    model = darknet53()
    '''
    torch.Size([1, 512, 80, 80])
    torch.Size([1, 1024, 40, 40])
    torch.Size([1, 2048, 20, 20])
    '''

    # model = MobileNetV1()  # 下采样倍数32 能力23.85 7.13
    # model = models.squeezenet1_0(pretrained=True)
    # model = models.vgg.vgg16(pretrained=True)
    # model = models.shufflenet_v2_x1_0(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.mnasnet1_0(pretrained=True)  # 能力 26.49 8.456
    # model = models.inception_v3(pretrained=True)  # 能力 22.55 6.44

    '''-----------------模型分析 开始-----------------------'''
    import tensorwatch as tw

    # 用这个即可---查看网络的统计结果---
    args_pd = tw.model_stats(model, data_inputs_list)
    args_pd.to_excel('model_log.xlsx')

    # # print(type(args_pd))
    # print(args_pd)

    from torchsummary import summary

    # summary1 = summary(model, (3, 640, 640))
    # print(type(summary1))

    # other()
    '''-----------------模型分析 完成-----------------------'''

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
