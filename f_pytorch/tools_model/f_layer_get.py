import torch
from torchvision.models import _utils
import torch.nn as nn

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import save_weight, load_weight

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


'''-----------------模型提取-----------------------'''


class ModelOut4Densenet121(nn.Module):

    def __init__(self, backbone, layer_name, ret_name_dict):
        super().__init__()
        submodule = backbone._modules[layer_name]
        self.outs_model = ModelOut4Utils(submodule, ret_name_dict)

    def forward(self, inputs):
        out1, out2, out3 = self.outs_model(inputs)
        return out1, out2, out3


class ModelOuts4Mobilenet_v2(nn.Module):
    def __init__(self, model):
        super().__init__()
        # del model.classifier
        self.model_hook = model

        self.dims_out = [192, 576, 1280]
        model.features[7].conv[1][0].register_forward_hook(self.fun_layer1)
        model.features[14].conv[1][0].register_forward_hook(self.fun_layer2)
        model.features[18][2].register_forward_hook(self.fun_layer3)
        self.out_layout1, self.out_layout2, self.out_layout3 = [0] * 3

    def fun_layer1(self, module, input, output):
        self.out_layout1 = input[0]

    def fun_layer2(self, module, input, output):
        self.out_layout2 = input[0]

    def fun_layer3(self, module, input, output):
        self.out_layout3 = output

    def forward(self, inputs):
        hook = self.model_hook(inputs)
        # torch.Size([10, 192, 52, 52]))  torch.Size([10, 576, 26, 26]) torch.Size([10, 1280, 13, 13])
        return self.out_layout1, self.out_layout2, self.out_layout3
        # return hook


class ModelOut4Mobilenet_v2(nn.Module):
    def __init__(self, model):
        super().__init__()
        # del model.classifier
        self.model_hook = model

        self.dim_out = 1280
        model.features[18][2].register_forward_hook(self.fun_layer1)
        self.out_layout1 = None

    def fun_layer1(self, module, input, output):
        self.out_layout1 = output

    def forward(self, inputs):
        hook = self.model_hook(inputs)
        return self.out_layout1


class ModelOut4Utils(nn.Module):

    def __init__(self, submodule, ret_name_dict) -> None:
        '''
        只支持 层名为顶层
        '''
        super().__init__()
        self.layer_out = _utils.IntermediateLayerGetter(submodule, ret_name_dict)

    def forward(self, inputs):
        out = self.layer_out(inputs)
        out = list(out.values())
        # torch.Size([1, 1024, 26, 26])
        return out


class FeatureExtractor(nn.Module):

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)

        return outputs


if __name__ == '__main__':
    from f_pytorch.tools_model.model_look import f_look_model
    from torchvision import models

    '''通过 层层遍历 提取'''
    # model = models.densenet121(pretrained=True)
    # # f_look(model, input=(1, 3, 416, 416))
    # # conv 可以取 in_channels 不支持数组层
    # dims_out = [512, 1024, 1024]
    # print(dims_out)
    # ret_name_dict = {'denseblock2': 1, 'denseblock3': 2, 'denseblock4': 3}
    # model = ModelOut4Densenet121(model, 'features', ret_name_dict)
    # f_look_model(model, input=(1, 3, 416, 416))

    # 这个分不出来
    # model = models.mobilenet_v2(pretrained=True)
    # f_look(model, input=(1, 3, 416, 416))
    # model = ModelOut4Mobilenet_v2(model)
    # dims_out = [192, 576, 1280]
    # f_look_model(model, input=(1, 3, 416, 416))

    '''通过 只支持顶层 _utils.IntermediateLayerGetter(backbone, return_layers) 提取'''


    # model = models.resnext50_32x4d(pretrained=True)
    # model = models.densenet161(pretrained=True)
    # ret_name_dict = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    # model = Output4Return(model, return_layers)
    # dims_out = [512, 1024, 2048]
    # f_look(model, input=(1, 3, 416, 416))

    def fun1(module, input, output):
        print(module, input, output)
        print(input[0])  # hook1_obj hook2_obj
        print(output)  # hook1


    # model = models.resnet50(pretrained=True)
    # model = models.wide_resnet50_2(pretrained=True)
    # model = models.mnasnet0_5(pretrained=True)
    model = models.mobilenet_v2(pretrained=True)
    model = ModelOuts4Mobilenet_v2(model)
    save_weight('.', model, '123')
    load_weight('./123-1_None.pth', model)

    # checkpoint = torch.load('./123-1_None.pth')
    # pretrained_dict = checkpoint['model']
    # keys_missing, keys_unexpected = model.load_state_dict(pretrained_dict, strict=False)
    # if len(keys_missing) > 0 or len(keys_unexpected):
    #     flog.error('missing_keys %s', keys_missing)
    #     flog.error('unexpected_keys %s', keys_unexpected)
    # else:
    #     flog.info('完成')

    # dims_out = [240, 576, 1280]
    # dims_out = [144, 288, 1280]
    # layer1 = model.layers[10][0].layers[3]
    # layer2 = model.layers[12][0].layers[3]
    # layer3 = model.layers[10][0].layers[3]
    # ModelOuts4Mnasnet(model)
    #
    # hook1_obj = model.layers[10][0].layers[3].register_forward_hook(fun1)  # 提取输入
    # hook2_obj = model.layers[12][0].layers[3].register_forward_hook(fun1)  # 提取输入
    # hook3_obj = model.layers[16].register_forward_hook(fun1)  # 提取输入
    # hook_obj.remove()
    # del model.classifier
    # ModelOut4Mnasnet1_0(model,)
    # print(hook_obj)
    f_look_model(model, input=(10, 3, 416, 416))
    pass
