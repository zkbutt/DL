import torch
import torch.nn as nn
import os

from f_pytorch.tools_model.f_layer_get import ModelOuts4DarkNet19, ModelOuts4DarkNet53


class Conv_LeakyReLU_Res_BN(nn.Module):
    def __init__(self, in_ch, ksize, padding=0, stride=1, dilation=1, depthwise=False):
        super(Conv_LeakyReLU_Res_BN, self).__init__()

        inter_ch = in_ch // 2
        # depth-wise conv
        if depthwise:
            groups = inter_ch
        else:
            groups = 1

        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1, bias=False),
            nn.Conv2d(inter_ch, inter_ch, ksize, padding=padding, stride=stride, dilation=dilation, bias=False,
                      groups=groups),
            nn.Conv2d(inter_ch, in_ch, 1, bias=False)
        )

        # # res-connect
        # self.conv1x1_compress = nn.Conv2d(in_ch, inter_ch // 4, 1)

        self.bn_leakyRelu = nn.Sequential(
            nn.BatchNorm2d(in_ch * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        # x += x + self.convs(x)
        # x_spp = self.conv1x1(x)
        # x_spp_1 = F.max_pool2d(x_spp, 3, stride=1, padding=1)
        # x_spp_2 = F.max_pool2d(x_spp, 5, stride=1, padding=2)
        # x_spp_3 = F.max_pool2d(x_spp, 7, stride=1, padding=3)
        # x_spp = torch.cat([x_spp, x_spp_1, x_spp_2, x_spp_3], dim=1)
        # x = self.bn_leakyRelu(torch.cat([x, x_spp], dim=1))
        x = self.bn_leakyRelu(torch.cat([x, self.convs(x)], dim=1))

        return x


class Conv_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch // 2, 1),
                Conv_BN_LeakyReLU(ch // 2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_19(nn.Module):
    def __init__(self, num_classes=1000):
        print("Initializing the darknet19 network ......")

        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )

        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

        self.conv_7 = nn.Conv2d(1024, 1000, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_impl(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(self.maxpool_4(x))
        x = self.conv_6(self.maxpool_5(x))

        x = self.avgpool(x)
        x = self.conv_7(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class DarkNet_53(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """

    def __init__(self, num_classes=1000):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _forward_impl(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class DarkNet_Tiny(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet_Tiny, self).__init__()
        # backbone network : DarkNet-Tiny
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            Conv_BN_LeakyReLU(32, 32, 3, padding=1, stride=2)
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            Conv_BN_LeakyReLU(64, 64, 3, padding=1, stride=2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 128, 3, padding=1, stride=2),
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 256, 3, padding=1, stride=2),
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 512, 3, padding=1, stride=2),
        )

        self.conv_6 = nn.Conv2d(512, 1000, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        x = self.avgpool(x)
        x = self.conv_6(x)
        x = x.view(x.size(0), -1)
        return x


class DarkNet_Lite(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet_Lite, self).__init__()
        # backbone network : DarkNet-Tiny
        # output : stride = 2, c = 32
        self.layer_1 = nn.Sequential(
            Conv_LeakyReLU(3, 16, ksize=3, padding=1),
            Conv_LeakyReLU_Res_BN(in_ch=16, ksize=3, padding=1),  # out_ch = 32
            nn.MaxPool2d((3, 3), stride=2, padding=1)
        )

        # output : stride = 4, c = 64
        self.layer_2 = nn.Sequential(
            Conv_LeakyReLU_Res_BN(in_ch=32, ksize=3, padding=1),  # out_ch = 64
            nn.MaxPool2d((2, 2), stride=2)
        )

        # output : stride = 8, c = 128
        self.layer_3 = nn.Sequential(
            Conv_LeakyReLU_Res_BN(in_ch=64, ksize=3, padding=1, depthwise=True),  # out_ch = 128
            nn.MaxPool2d((2, 2), stride=2)
        )

        # output : stride = 16, c = 256
        self.layer_4 = nn.Sequential(
            Conv_LeakyReLU_Res_BN(in_ch=128, ksize=3, padding=1, depthwise=True),  # out_ch = 256
            nn.MaxPool2d((2, 2), stride=2)
        )

        # output : stride = 32, c = 512
        self.layer_5 = nn.Sequential(
            Conv_LeakyReLU_Res_BN(in_ch=256, ksize=3, padding=1, depthwise=True),  # out_ch = 512
            nn.MaxPool2d((2, 2), stride=2),
            Conv_BN_LeakyReLU(512, 512, ksize=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_6 = nn.Conv2d(512, 1000, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = self.conv_6(x)
        x = x.view(x.size(0), -1)
        return x


class DarkNet_Light(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet_Light, self).__init__()
        # backbone network : DarkNet
        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)  # stride = 2

        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)  # stride = 4

        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)  # stride = 8

        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)  # stride = 16

        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)  # stride = 32

        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d((2, 2), 1)  # stride = 32
        )

        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_8 = nn.Conv2d(1024, 1000, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        x = self.conv_5(x)
        x = self.maxpool_5(x)
        x = self.conv_6(x)
        x = self.maxpool_6(x)
        x = self.conv_7(x)

        x = self.avgpool(x)
        x = self.conv_8(x)
        x = x.view(x.size(0), -1)
        return x


def _init_path_root():
    import socket

    host_name = socket.gethostname()
    if host_name == 'Feadre-NB':
        PATH_HOST = 'M:'
        # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
    elif host_name == 'e2680v2':
        PATH_HOST = ''
    return PATH_HOST


def _load_weight_base(model, file_weight, device):
    path_root = _init_path_root()
    model.load_state_dict(
        torch.load(os.path.join(path_root, file_weight), map_location=device),
        strict=False)
    return model


def darknet19(pretrained=False, device='cpu', **kwargs):
    """Constructs a darknet-19 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = DarkNet_19()
    if pretrained:
        print('Loading the darknet19 ...')
        # file_weight = 'AI/weights/pytorch/darknet/darknet19_72.96.pth'
        file_weight = 'AI/weights/pytorch/darknet/darknet19_hr_75.52_92.73.pth'
        model = _load_weight_base(model, file_weight, device)
    return model


def darknet53(pretrained=False, device='cpu', **kwargs):
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_53()
    if pretrained:
        print('Loading the darknet53 ...')
        # file_weight = 'AI/weights/pytorch/darknet/darknet53_75.42.pth'
        file_weight = 'AI/weights/pytorch/darknet/darknet53_hr_77.76.pth'
        model = _load_weight_base(model, file_weight, device)
    return model


def darknet_tiny(pretrained=False, device='cpu', **kwargs):
    """Constructs a darknet-tiny model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_Tiny()
    if pretrained:
        print('Loading the darknet_tiny ...')
        # file_weight = 'AI/weights/pytorch/darknet/darknet_tiny_63.50_85.06.pth'
        file_weight = 'AI/weights/pytorch/darknet/darknet_tiny_hr_61.85.pth'
        model = _load_weight_base(model, file_weight, device)
    return model


def darknet_lite(pretrained=False, device='cpu', **kwargs):
    """Constructs a darknet-tiny model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_Lite()
    if pretrained:
        print('Loading the darknet_lite ...')
        # file_weight = 'AI/weights/pytorch/darknet/darknet_tiny_63.50_85.06.pth'
        file_weight = 'AI/weights/pytorch/darknet/darknet_tiny_hr_61.85.pth'
        model = _load_weight_base(model, file_weight, device)
    return model


def darknet_light(pretrained=False, device='cpu', **kwargs):
    """Constructs a darknet model.
    TinyYOLOv3 一样
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_Light()
    if pretrained:
        print('Loading the darknet_light ...')
        # file_weight = 'AI/weights/pytorch/darknet/darknet_light_90_58.99.pth'
        file_weight = 'AI/weights/pytorch/darknet/darknet_light_hr_59.61.pth'
        model = _load_weight_base(model, file_weight, device)
    return model


if __name__ == '__main__':
    from f_pytorch.tools_model.model_look import f_look_summary, f_look_tw

    # model = darknet19(pretrained=True, device='cpu') # 1
    model = darknet53(pretrained=True, device='cpu')  # 2
    model = ModelOuts4DarkNet53(model)
    # model = darknet_tiny(pretrained=True, device='cpu')  # 1/4
    # model = darknet_lite(pretrained=True, device='cpu')
    # model = darknet_light(pretrained=True, device='cpu')
    f_look_summary(model, input=(3, 416, 416))
    # f_look_tw(model, input=(1, 3, 416, 416), name='model_look')
    # print(model)
