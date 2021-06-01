import torch
from torchvision.models import _utils
import torch.nn as nn
import types

from f_pytorch.tools_model.f_model_api import NoneLayer
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


class FNoProcessing(nn.Module):
    '''
    不处理的模型
    '''

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


'''-----------------模型提取双-----------------------'''


class ModelOutsUtils(nn.Module):

    def __init__(self, submodule, ret_name_dict) -> None:
        '''
        只支持 层名为顶层
        '''
        super().__init__()
        self.layer_out = _utils.IntermediateLayerGetter(submodule, ret_name_dict)

    def forward(self, inputs):
        out = self.layer_out(inputs)
        out = list(out.outs())
        # torch.Size([1, 1024, 26, 26])
        return out


class ModelOuts4Resnet(nn.Module):

    def __init__(self, model, dims_out=(512, 1024, 2048)):
        super().__init__()
        self.dims_out = dims_out

        self.model_hook = model
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        # self.model_hook.classifier = NoneLayer()  # 直接输出层
        self.model_hook.layer3[0].conv1.register_forward_hook(self.fun_layer1)
        self.model_hook.layer4[0].conv1.register_forward_hook(self.fun_layer2)
        # model.features[18][2].register_forward_hook(self.fun_layer3)

        self.dims_out = dims_out
        self.out_layout1, self.out_layout2 = [0] * 2

    def foverwrite(self, model, x):
        # 重写流程 model 替换self
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x

    def fun_layer1(self, module, input, output):
        '''input是默认tuple '''
        self.out_layout1 = input[0]

    def fun_layer2(self, module, input, output):
        self.out_layout2 = input[0]

    def forward(self, inputs):
        outs = self.model_hook(inputs)
        # torch.Size([10, 192, 52, 52]))  torch.Size([10, 576, 26, 26]) torch.Size([10, 1280, 13, 13])
        return self.out_layout1, self.out_layout2, outs
        # return hook


class ModelOuts4Resnet_4(nn.Module):

    def __init__(self, model, dims_out=(512, 1024, 2048)):
        super().__init__()
        self.dims_out = dims_out

        self.model_hook = model
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        # self.model_hook.classifier = NoneLayer()  # 直接输出层
        self.model_hook.layer2[0].conv1.register_forward_hook(self.fun_layer1)
        self.model_hook.layer3[0].conv1.register_forward_hook(self.fun_layer2)
        self.model_hook.layer4[0].conv1.register_forward_hook(self.fun_layer3)
        # model.features[18][2].register_forward_hook(self.fun_layer3)

        self.dims_out = dims_out
        self.out_layout1, self.out_layout2 = [0] * 2

    def foverwrite(self, model, x):
        # 重写流程 model 替换self
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x

    def fun_layer1(self, module, input, output):
        '''input是默认tuple '''
        self.out_layout1 = input[0]

    def fun_layer2(self, module, input, output):
        self.out_layout2 = input[0]

    def fun_layer3(self, module, input, output):
        self.out_layout3 = input[0]

    def forward(self, inputs):
        outs = self.model_hook(inputs)
        # torch.Size([10, 192, 52, 52]))  torch.Size([10, 576, 26, 26]) torch.Size([10, 1280, 13, 13])
        return self.out_layout1, self.out_layout2, self.out_layout3, outs
        # return hook


class ModelOuts4Densenet121(nn.Module):

    def __init__(self, backbone, layer_name, ret_name_dict, dims_out=[512, 1024, 1024]):
        super().__init__()
        submodule = backbone._modules[layer_name]
        self.outs_model = ModelOutsUtils(submodule, ret_name_dict)
        self.dims_out = dims_out

    def forward(self, inputs):
        out1, out2, out3 = self.outs_model(inputs)
        return out1, out2, out3


class ModelOuts4Mobilenet_v2(nn.Module):
    def __init__(self, model):
        super().__init__()
        del model.classifier
        self.model_hook = model
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        # self.model_hook.classifier = NoneLayer()  # 直接输出层
        self.model_hook.features[7].conv[1][0].register_forward_hook(self.fun_layer1)
        self.model_hook.features[14].conv[1][0].register_forward_hook(self.fun_layer2)
        # model.features[18][2].register_forward_hook(self.fun_layer3)

        self.dims_out = [192, 576, 1280]
        self.out_layout1, self.out_layout2 = [0] * 2

    def foverwrite(self, model, x):
        # 重写流程
        x = model.features(x)
        return x

    def fun_layer1(self, module, input, output):
        self.out_layout1 = input[0]

    def fun_layer2(self, module, input, output):
        self.out_layout2 = input[0]

    def forward(self, inputs):
        outs = self.model_hook(inputs)
        # torch.Size([10, 192, 52, 52]))  torch.Size([10, 576, 26, 26]) torch.Size([10, 1280, 13, 13])
        return self.out_layout1, self.out_layout2, outs
        # return hook


class ModelOuts4DarkNet19(nn.Module):
    def __init__(self, model, dims_out=(256, 512, 1024)):
        super().__init__()
        del model.conv_7
        del model.avgpool
        self.model_hook = model
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        self.dims_out = dims_out

    def foverwrite(self, model, x):
        x = model.conv_1(x)
        x = model.conv_2(x)
        x = model.conv_3(x)
        ceng1 = model.conv_4(x)
        ceng2 = model.conv_5(model.maxpool_4(ceng1))
        ceng3 = model.conv_6(model.maxpool_5(ceng2))
        return ceng1, ceng2, ceng3

    def forward(self, inputs):
        outs = self.model_hook(inputs)
        return outs


class ModelOuts4DarkNet19_Tiny(nn.Module):
    def __init__(self, model, dims_out=(128, 256, 512)):
        super().__init__()
        del model.conv_6
        del model.avgpool
        self.model_hook = model
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        self.dims_out = dims_out

    def foverwrite(self, model, x):
        x = model.conv_1(x)
        x = model.conv_2(x)
        ceng1 = model.conv_3(x)
        ceng2 = model.conv_4(ceng1)
        ceng3 = model.conv_5(ceng2)
        return ceng1, ceng2, ceng3

    def forward(self, inputs):
        outs = self.model_hook(inputs)
        return outs


class ModelOuts4DarkNet53(nn.Module):
    def __init__(self, model, dims_out=(256, 512, 1024)):
        super().__init__()
        del model.avgpool
        del model.fc
        self.model_hook = model
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        self.dims_out = dims_out

    def foverwrite(self, model, x):
        x = model.layer_1(x)
        x = model.layer_2(x)
        ceng1 = model.layer_3(x)
        ceng2 = model.layer_4(ceng1)
        ceng3 = model.layer_5(ceng2)
        return ceng1, ceng2, ceng3

    def forward(self, inputs):
        outs = self.model_hook(inputs)
        return outs


'''-----------------单输出---------------------'''


class ModelOut4DarkNet19(nn.Module):
    def __init__(self, model):
        super().__init__()
        del model.conv_7
        del model.avgpool
        self.model = model
        # 重写流程
        self.model._forward_impl = types.MethodType(self.foverwrite, model)
        self.dim_out = 1024

    def foverwrite(self, model, x):
        x = model.conv_1(x)
        x = model.conv_2(x)
        x = model.conv_3(x)
        x = model.conv_4(x)
        x = model.conv_5(model.maxpool_4(x))
        x = model.conv_6(model.maxpool_5(x))

        # x = self.avgpool(x)
        # x = self.conv_7(x)
        # x = x.view(x.size(0), -1)
        return x

    def forward(self, inputs):
        ous = self.model(inputs)
        return ous


class ModelOut4Resnet18(nn.Module):
    def __init__(self, model):
        super().__init__()
        del model.avgpool
        del model.fc
        self.model = model
        # 重写流程
        self.model._forward_impl = types.MethodType(self.foverwrite, model)
        self.dim_out = 512

    def foverwrite(self, model, x):
        # 重写流程
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

    def forward(self, inputs):
        ous = self.model(inputs)
        return ous


class ModelOut4Resnet50(ModelOut4Resnet18):

    def __init__(self, model):
        super().__init__(model)
        self.dim_out = 2048


class ModelOut4Mobilenet_v2(nn.Module):
    def __init__(self, model):
        super().__init__()
        del model.classifier
        self.model_hook = model
        # 直接全部改了
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        self.dim_out = 1280
        # model.features[18][2].register_forward_hook(self.fun_layer1)
        # self.out_layout1 = None

    def foverwrite(self, model, x):
        # 重写流程
        x = model.features(x)
        return x

    def forward(self, inputs):
        ous = self.model_hook(inputs)
        return ous


class ModelOut4Mobilenet_v3(nn.Module):
    def __init__(self, model):
        super().__init__()
        del model.linear3
        del model.bn3
        del model.hs3
        del model.linear4
        self.model_hook = model
        # 类中调用 第一参数是self  必须定义一个参数 是第二个参数, 调用传入的是第三个
        self.model_hook._forward_impl = types.MethodType(self.foverwrite, model)
        self.dim_out = 576

    def foverwrite(self, model, x):
        '''

        :param model: MethodType 必须传一个没用的参数
        :param x:
        :return:
        '''
        out = model.hs1(model.bn1(model.conv1(x)))
        out = model.bneck(out)
        out = model.hs2(model.bn2(model.conv2(out)))
        # out = F.avg_pool2d(out, 7)
        # out = out.view(out.size(0), -1)
        # out = self.hs3(self.bn3(self.linear3(out)))
        # out = self.linear4(out)
        return out

    def forward(self, inputs):
        ous = self.model_hook(inputs)
        return ous


if __name__ == '__main__':
    from f_pytorch.tools_model.model_look import f_look_tw
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

    # model = models.resnet50(pretrained=True)
    # model = models.resnext50_32x4d(pretrained=True)
    # dims_out = (512, 1024, 2048)
    model = models.resnet18(pretrained=True)
    dims_out = (128, 256, 512)
    model = ModelOuts4Resnet(model, dims_out)
    # f_look_model(model, input=(1, 3, 416, 416))

    x = torch.zeros((1, 3, 416, 416))
    outs = model(x)
    print(outs)

    # model = models.resnext50_32x4d(pretrained=True)
    # model = models.densenet161(pretrained=True)
    # ret_name_dict = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    # model = Output4Return(model, return_layers)

    # dims_out = [512, 1024, 2048]
    # f_look(model, input=(1, 3, 416, 416))

    # model = models.resnet50(pretrained=True)
    # model = models.wide_resnet50_2(pretrained=True)
    # model = models.mnasnet0_5(pretrained=True)
    # model = models.mobilenet_v2(pretrained=True)
    # model = ModelOuts4Mobilenet_v2(model)
    # save_weight('.', model, '123')
    # load_weight('./123-1_None.pth', model)

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
    # f_look_model(model, input=(10, 3, 416, 416))
    pass
