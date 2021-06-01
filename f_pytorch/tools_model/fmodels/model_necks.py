import torch
from torch import nn
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Resnet, ModelOut4Resnet18
from f_pytorch.tools_model.f_model_api import FConv2d
from f_pytorch.tools_model.fmodels.model_modules import Focus, SPPv3, SPPv2


class FPN_out_v2(nn.Module):
    ''' FPN '''

    def __init__(self, in_channels_list, feature_size=256, o_ceng=3):
        '''
        input = torch.arange(1, 5).view(1, 1, 2, 2).float()
        print(input)
        #tensor([[[[1., 2.],
        #          [3., 4.]]]])

        Upsample:
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        print(upsample(input))
        #tensor([[[[1., 1., 2., 2.],
        #          [1., 1., 2., 2.],
        #          [3., 3., 4., 4.],
        #          [3., 3., 4., 4.]]]])
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        print(upsample(input))
        #tensor([[[[1.0000, 1.2500, 1.7500, 2.0000],
        #          [1.5000, 1.7500, 2.2500, 2.5000],
        #          [2.5000, 2.7500, 3.2500, 3.5000],
        #          [3.0000, 3.2500, 3.7500, 4.0000]]]])

        Interpolate:
        upsample = F.interpolate(input, size=(4, 4), mode='nearest')
        print(upsample)
        #tensor([[[[1., 1., 2., 2.],
        #          [1., 1., 2., 2.],
        #          [3., 3., 4., 4.],
        #          [3., 3., 4., 4.]]]])
        upsample = F.interpolate(input, size=(4, 4), mode='bilinear')
        print(upsample)
        #tensor([[[[1.0000, 1.2500, 1.7500, 2.0000],
        #          [1.5000, 1.7500, 2.2500, 2.5000],
        #          [2.5000, 2.7500, 3.2500, 3.5000],
        #          [3.0000, 3.2500, 3.7500, 4.0000]]]])


        :param in_channels_list: dims_out = (128, 256, 512)
        :param feature_size:
        '''
        super(FPN_out_v2, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = FConv2d(in_channels_list[2], feature_size, k=1, norm='bn', act='leaky', is_bias=False)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = FConv2d(feature_size, feature_size, k=3, norm='bn', act='leaky', is_bias=False)

        # add P5 elementwise to C4
        self.P4_1 = FConv2d(in_channels_list[1], feature_size, k=1, norm='bn', act='leaky', is_bias=False)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = FConv2d(feature_size, feature_size, k=3, norm='bn', act='leaky', is_bias=False)

        # add P4 elementwise to C3
        self.P3_1 = FConv2d(in_channels_list[0], feature_size, k=1, norm='bn', act='leaky', is_bias=False)
        self.P3_2 = FConv2d(feature_size, feature_size, k=3, norm='bn', act='leaky', is_bias=False)

        self.o_ceng = o_ceng
        if self.o_ceng == 3:
            return
        elif self.o_ceng == 4:
            # "P6 is obtained via a 3x3 stride-2 conv on C5"
            self.P6 = FConv2d(in_channels_list[2], feature_size, k=3, s=2, p=1, norm='bn', act='leaky',
                              is_bias=False)
        elif self.o_ceng == 5:
            self.P6 = FConv2d(in_channels_list[2], feature_size, k=3, s=2, p=1, norm='bn', act='leaky',
                              is_bias=False)
            # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
            self.P7 = FConv2d(feature_size, feature_size, k=3, s=2, p=1, norm='bn', act='leaky', is_bias=False)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        # self.P4_upsampled(P4_x) 与 F.interpolate 等价
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        if self.o_ceng == 3:
            return [P3_x, P4_x, P5_x]
        elif self.o_ceng == 4:
            P6_x = self.P6(C5)
            return [P3_x, P4_x, P5_x, P6_x]
        elif self.o_ceng == 5:
            P6_x = self.P6(C5)
            P7_x = self.P7(P6_x)
            return [P3_x, P4_x, P5_x, P6_x, P7_x]


class Yolo4(nn.Module):

    def __init__(self, backbone):
        super(Yolo4).__init__()
        self.backbone = backbone
        self.down_sample1 = conv2d(128, 256, 3, stride=2)

    def forward(self, inputs):
        pass


if __name__ == '__main__':
    from f_pytorch.tools_model.model_look import f_look_tw

    model = models.resnet18(pretrained=True)
    model = ModelOut4Resnet18(model)

    print(model(torch.rand((1, 3, 608, 608))).shape)


    # dims_out = (128, 256, 512)
    # model = ModelOuts4Resnet(model, dims_out)
    # # fpn = FPN_out3(dims_out, 128)
    # fpn = FPN_out_v2(dims_out, 128, o_ceng=5)
    # outs = fpn(model(torch.rand((4, 3, 416, 416))))
    # for o in outs:
    #     print(o.shape)
    # # fun = Focus(128, 128 * 4)
    # # fun = Focus(128, 128 * 2)
    # fun = SPPv3(128)
    # # fun = SPPv2()
    #
    # print(fun(outs[0]).shape)
