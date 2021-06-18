import torch
from torch import nn
from torchvision import models

from f_pytorch.tools_model.f_layer_get import ModelOuts4Resnet, ModelOut4Resnet18
from f_pytorch.tools_model.f_model_api import FConv2d
from f_pytorch.tools_model.fmodels.model_modules import Focus, SPPv3, SPPv2


# class FPN_out_v2(nn.Module):
#     ''' FPN 单卷积简单化版 '''
#
#     def __init__(self, in_channels_list, feature_size=256, o_ceng=3, act='leaky', is_bias=True):
#
#         super(FPN_out_v2, self).__init__()
#
#         # upsample C5 to get P5 from the FPN paper
#         self.P5_1 = FConv2d(in_channels_list[2], feature_size, k=1, norm='bn', act=act, is_bias=is_bias)
#         self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
#         self.P5_2 = FConv2d(feature_size, feature_size, k=3, norm='bn', act=act, is_bias=is_bias)
#
#         # add P5 elementwise to C4
#         self.P4_1 = FConv2d(in_channels_list[1], feature_size, k=1, norm='bn', act=act, is_bias=is_bias)
#         self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
#         self.P4_2 = FConv2d(feature_size, feature_size, k=3, norm='bn', act=act, is_bias=is_bias)
#
#         # add P4 elementwise to C3
#         self.P3_1 = FConv2d(in_channels_list[0], feature_size, k=1, norm='bn', act=act, is_bias=is_bias)
#         self.P3_2 = FConv2d(feature_size, feature_size, k=3, norm='bn', act=act, is_bias=is_bias)
#
#         self.o_ceng = o_ceng
#         if self.o_ceng == 3:
#             return
#         elif self.o_ceng == 4:
#             # "P6 is obtained via a 3x3 stride-2 conv on C5"
#             self.P6 = FConv2d(in_channels_list[2], feature_size, k=3, s=2, p=1, norm='bn', act=act, is_bias=is_bias)
#         elif self.o_ceng == 5:
#             self.P6 = FConv2d(in_channels_list[2], feature_size, k=3, s=2, p=1, norm='bn', act=act, is_bias=is_bias)
#             # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
#             self.P7 = FConv2d(feature_size, feature_size, k=3, s=2, p=1, norm='bn', act=act, is_bias=is_bias)
#
#     def forward(self, inputs):
#         C3, C4, C5 = inputs
#
#         P5_x = self.P5_1(C5)
#         P5_upsampled_x = self.P5_upsampled(P5_x)
#         P5_x = self.P5_2(P5_x)
#
#         P4_x = self.P4_1(C4)
#         P4_x = P5_upsampled_x + P4_x
#         # self.P4_upsampled(P4_x) 与 F.interpolate 等价
#         P4_upsampled_x = self.P4_upsampled(P4_x)
#         P4_x = self.P4_2(P4_x)
#
#         P3_x = self.P3_1(C3)
#         P3_x = P3_x + P4_upsampled_x
#         P3_x = self.P3_2(P3_x)
#
#         if self.o_ceng == 3:
#             return [P3_x, P4_x, P5_x]
#         elif self.o_ceng == 4:
#             P6_x = self.P6(C5)
#             return [P3_x, P4_x, P5_x, P6_x]
#         elif self.o_ceng == 5:
#             P6_x = self.P6(C5)
#             P7_x = self.P7(P6_x)
#             return [P3_x, P4_x, P5_x, P6_x, P7_x]


class FPN_out_v3(nn.Module):
    ''' FPN 3卷积
    输出不同维 输出可选

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

    def __init__(self, in_channels_list, out_channel=None, o_ceng=3, num_conv=1, act='leaky', is_bias=True):
        '''

        :param in_channels_list: 写死只支持3层
        :param out_channel:
        :param o_ceng:
        :param num_conv: 这个不支持写死的
        :param act:
        :param is_bias:
        '''
        super(FPN_out_v3, self).__init__()
        if out_channel is None:
            c = in_channels_list[2] // 2  # 512//2=256
            self.P5_cbl_set = nn.Sequential(
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
            )
            self.P5_2 = FConv2d(c, c // 2, k=1, act=act, p=0, is_bias=is_bias)
            self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

            # add P5 elementwise to C4
            c = in_channels_list[1] // 2  # 256//2=128
            self.P4_cbl_set = nn.Sequential(
                FConv2d(c * 2 + c, c, k=1, p=0, act=act, is_bias=is_bias),
                FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
            )
            self.P4_2 = FConv2d(c, c // 2, k=1, act=act, p=0, is_bias=is_bias)
            self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

            # add P4 elementwise to C3
            c = in_channels_list[0] // 2  # 128//2=192
            self.P3_cbl_set = nn.Sequential(
                FConv2d(c * 2 + c, c, k=1, p=0, act=act, is_bias=is_bias),
                FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
            )

            self.o_ceng = o_ceng
            if self.o_ceng == 3:
                # 输出维度由小到大
                self.dims_out = [i // 2 for i in in_channels_list]
                return
            elif self.o_ceng == 4:
                c = in_channels_list[2]
                # "P6 is obtained via a 3x3 stride-2 conv on C5"
                self.P6 = FConv2d(c, c * 2, k=3, s=2, p=1, norm='bn', act=act, is_bias=is_bias)
                self.P6_cbl_set = nn.Sequential(
                    FConv2d(c * 2, c, k=1, s=1, p=0, act=act, is_bias=is_bias),
                    FConv2d(c, c * 2, k=3, s=1, p=1, act=act, is_bias=is_bias),
                    FConv2d(c * 2, c, k=1, s=1, p=0, act=act, is_bias=is_bias),
                )
                self.dims_out = [i // 2 for i in in_channels_list]
                self.dims_out.append(self.dims_out[-1] * 2)
            elif self.o_ceng == 5:
                c = in_channels_list[2]
                # "P6 is obtained via a 3x3 stride-2 conv on C5"
                self.P6 = FConv2d(c, c * 2, k=3, s=2, p=1, norm='bn', act=act, is_bias=is_bias)
                self.P6_cbl_set = nn.Sequential(
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                    FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                )
                c = in_channels_list[2] * 2
                # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
                self.P7 = FConv2d(c, c * 2, k=3, s=2, p=1, act=act, is_bias=is_bias)
                self.P7_cbl_set = nn.Sequential(
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                    FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                )
                self.dims_out = [i // 2 for i in in_channels_list]
                self.dims_out.append(self.dims_out[-1] * 2)
                self.dims_out.append(self.dims_out[-1] * 2)
            else:
                raise Exception('self.o_ceng 出错%s ' % self.o_ceng)
        else:
            c = in_channels_list[2] // 2  # 512//2=256
            self.P5_cbl_set = nn.Sequential(
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
            )
            self.P5_2 = FConv2d(c, c // 2, k=1, act=act, p=0, is_bias=is_bias)
            self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

            # add P5 elementwise to C4
            c = in_channels_list[1] // 2  # 256//2=128
            self.P4_cbl_set = nn.Sequential(
                FConv2d(c * 2 + c, c, k=1, p=0, act=act, is_bias=is_bias),
                FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
            )
            self.P4_2 = FConv2d(c, c // 2, k=1, act=act, p=0, is_bias=is_bias)
            self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

            # add P4 elementwise to C3
            c = in_channels_list[0] // 2  # 128//2=192
            self.P3_cbl_set = nn.Sequential(
                FConv2d(c * 2 + c, c, k=1, p=0, act=act, is_bias=is_bias),
                FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
            )

            self.o_ceng = o_ceng
            if self.o_ceng == 3:
                # 输出维度由小到大
                self.dims_out = [i // 2 for i in in_channels_list]
                return
            elif self.o_ceng == 4:
                c = in_channels_list[2]
                # "P6 is obtained via a 3x3 stride-2 conv on C5"
                self.P6 = FConv2d(c, c * 2, k=3, s=2, p=1, norm='bn', act=act, is_bias=is_bias)
                self.P6_cbl_set = nn.Sequential(
                    FConv2d(c * 2, c, k=1, s=1, p=0, act=act, is_bias=is_bias),
                    FConv2d(c, c * 2, k=3, s=1, p=1, act=act, is_bias=is_bias),
                    FConv2d(c * 2, c, k=1, s=1, p=0, act=act, is_bias=is_bias),
                )
                self.dims_out = [i // 2 for i in in_channels_list]
                self.dims_out.append(self.dims_out[-1] * 2)
            elif self.o_ceng == 5:
                c = in_channels_list[2]
                # "P6 is obtained via a 3x3 stride-2 conv on C5"
                self.P6 = FConv2d(c, c * 2, k=3, s=2, p=1, norm='bn', act=act, is_bias=is_bias)
                self.P6_cbl_set = nn.Sequential(
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                    FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                )
                c = in_channels_list[2] * 2
                # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
                self.P7 = FConv2d(c, c * 2, k=3, s=2, p=1, act=act, is_bias=is_bias)
                self.P7_cbl_set = nn.Sequential(
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                    FConv2d(c, c * 2, k=3, p=1, act=act, is_bias=is_bias),
                    FConv2d(c * 2, c, k=1, p=0, act=act, is_bias=is_bias),
                )
                self.dims_out = [i // 2 for i in in_channels_list]
                self.dims_out.append(self.dims_out[-1] * 2)
                self.dims_out.append(self.dims_out[-1] * 2)
            else:
                raise Exception('self.o_ceng 出错%s ' % self.o_ceng)

    def channel_fun(self, cin, cout, num_ceng, act, is_bias):
        '''
        固定 放大 恢复
        :param cin:
        :param cout:
        :param num_ceng:
        :return:
        '''
        convs = nn.ModuleList()
        convs.append(FConv2d(cin, cout, k=1, s=1, p=0, act=act, is_bias=is_bias))
        for i in range(1, num_ceng):
            if i % 2 == 0:  # 偶数
                convs.append(FConv2d(cout, cout * 2, k=3, s=1, p=1, act=act, is_bias=is_bias))
            else:
                convs.append(FConv2d(cout * 2, cout, k=1, s=1, p=0, act=act, is_bias=is_bias))
        return convs

    def forward(self, inputs):
        # C3[1, 128, 40, 40]   C4[1, 256, 20, 20]   C5[1, 512, 10, 10]
        C3, C4, C5 = inputs

        P5_x = self.P5_cbl_set(C5)  # torch.Size([1, 256, 10, 10])
        P5_up = self.P5_upsampled(self.P5_2(P5_x))  # torch.Size([1, 128, 20, 20]

        P4_x = torch.cat([C4, P5_up], dim=1)  # 叠加通道 torch.Size([1, 384, 20, 20])
        P4_x = self.P4_cbl_set(P4_x)
        # self.P4_upsampled(P4_x) 与 F.interpolate 等价
        P4_up = self.P4_upsampled(self.P4_2(P4_x))  # torch.Size([1, 64, 40, 40])

        P3_x = torch.cat([C3, P4_up], dim=1)  # torch.Size([1, 192, 40, 40])
        P3_x = self.P3_cbl_set(P3_x)  # torch.Size([1, 64, 40, 40])

        if self.o_ceng == 3:
            res = [P3_x, P4_x, P5_x]
        elif self.o_ceng == 4:
            P6_x = self.P6(C5)
            P6_x = self.P6_cbl_set(P6_x)
            res = [P3_x, P4_x, P5_x, P6_x]
        elif self.o_ceng == 5:
            P6_x = self.P6(C5)  # torch.Size([1, 1024, 5, 5])
            P7_x = self.P7(P6_x)  # torch.Size([1, 2048, 3, 3])
            P6_x = self.P6_cbl_set(P6_x)  # torch.Size([1, 512, 5, 5])
            P7_x = self.P7_cbl_set(P7_x)  # torch.Size([1, 1024, 3, 3])

            res = [P3_x, P4_x, P5_x, P6_x, P7_x]
        else:
            raise Exception('self.o_ceng 出错 %s ' % self.o_ceng)

        return res


class FPN_out_v4(nn.Module):
    ''' FPN 3卷积
    输出不同维及同维可选 效率较低

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

    def __init__(self, in_channels_list, out_channel=None, o_ceng=3, num_conv=1, act='leaky', is_bias=True):
        '''

        :param in_channels_list: 写死只支持3层
        :param out_channel:
        :param o_ceng:
        :param num_conv: 支持 135 奇数
        :param act:
        :param is_bias:
        '''
        super(FPN_out_v4, self).__init__()
        self.out_channel = out_channel

        cin = in_channels_list[2]  # 512
        cout = in_channels_list[2] // 2 if out_channel is None else out_channel
        self.P5_cbl_set = self.channel_fun(cin, cout, num_conv, act, is_bias)
        self.P5_2 = FConv2d(cout, cout // 2, k=1, act=act, p=0, is_bias=is_bias)  # 维度减半
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P5 elementwise to C4
        if out_channel is None:
            cin = in_channels_list[1] + in_channels_list[1] // 2
        else:
            cin = in_channels_list[1] + out_channel // 2  # 上层输出固定为 out_channel 的一半
        cout = in_channels_list[1] // 2 if out_channel is None else out_channel
        self.P4_cbl_set = self.channel_fun(cin, cout, num_conv, act, is_bias)
        self.P4_2 = FConv2d(cout, cout // 2, k=1, act=act, p=0, is_bias=is_bias)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P4 elementwise to C3
        if out_channel is None:
            cin = in_channels_list[0] + in_channels_list[0] // 2
        else:
            cin = in_channels_list[0] + out_channel // 2
        cout = in_channels_list[0] // 2 if out_channel is None else out_channel
        self.P3_cbl_set = self.channel_fun(cin, cout, num_conv, act, is_bias)

        self.o_ceng = o_ceng
        if self.o_ceng == 3:
            # 输出维度由小到大 缩小2倍
            self.dims_out = [i // 2 for i in in_channels_list]
            return
        elif self.o_ceng == 4:
            # 尺寸减小 升维
            self.P6 = FConv2d(in_channels_list[2], in_channels_list[2] * 2, k=3, s=2, p=1, norm='bn', act=act,
                              is_bias=is_bias)
            cin = in_channels_list[2] * 2
            cout = in_channels_list[2] if out_channel is None else out_channel
            self.P6_cbl_set = self.channel_fun(cin, cout, num_conv, act, is_bias)

            self.dims_out = [i // 2 for i in in_channels_list]
            self.dims_out.append(self.dims_out[-1] * 2)
        elif self.o_ceng == 5:
            self.P6 = FConv2d(in_channels_list[2], in_channels_list[2] * 2, k=3, s=2, p=1, norm='bn', act=act,
                              is_bias=is_bias)
            cin = in_channels_list[2] * 2  # 1024
            cout = in_channels_list[2] if out_channel is None else out_channel
            self.P6_cbl_set = self.channel_fun(cin, cout, num_conv, act, is_bias)

            # 尺寸减小 升维
            self.P7 = FConv2d(in_channels_list[2] * 2, in_channels_list[2] * 4, k=3, s=2, p=1, act=act,
                              is_bias=is_bias)
            cin = in_channels_list[2] * 4  # 2048
            cout = in_channels_list[2] * 2 if out_channel is None else out_channel
            self.P7_cbl_set = self.channel_fun(cin, cout, num_conv, act, is_bias)

            self.dims_out = [i // 2 for i in in_channels_list]
            self.dims_out.append(self.dims_out[-1] * 2)
            self.dims_out.append(self.dims_out[-1] * 2)
        else:
            raise Exception('self.o_ceng 出错%s ' % self.o_ceng)

    def channel_fun(self, cin, cout, num_conv, act, is_bias):
        '''
        固定 放大 恢复
        :param cin:
        :param cout:
        :param num_conv:
        :return:
        '''
        if num_conv % 2 == 0:
            raise Exception('num_conv 不能为偶数 %s ' % num_conv)

        # convs = nn.ModuleList() # 可遍历, 未实现 forward
        convs = []  # 实现 forward
        convs.append(FConv2d(cin, cout, k=1, p=0, act=act, is_bias=is_bias))
        for i in range(1, num_conv):
            # 先维度放大再缩小
            if i % 2 == 0:  # 偶数
                convs.append(FConv2d(cout * 2, cout, k=1, p=0, act=act, is_bias=is_bias))
            else:
                convs.append(FConv2d(cout, cout * 2, k=3, p=1, act=act, is_bias=is_bias))
        return nn.Sequential(*convs)

    def forward(self, inputs):
        # C3[1, 128, 40, 40]   C4[1, 256, 20, 20]   C5[1, 512, 10, 10]
        C3, C4, C5 = inputs

        P5_x = self.P5_cbl_set(C5)  # torch.Size([1, 256, 10, 10])
        P5_up = self.P5_upsampled(self.P5_2(P5_x))  # torch.Size([1, 128, 20, 20]

        P4_x = torch.cat([C4, P5_up], dim=1)  # 叠加通道 torch.Size([1, 384, 20, 20])
        P4_x = self.P4_cbl_set(P4_x)
        # self.P4_upsampled(P4_x) 与 F.interpolate 等价
        P4_up = self.P4_upsampled(self.P4_2(P4_x))  # torch.Size([1, 64, 40, 40])

        P3_x = torch.cat([C3, P4_up], dim=1)  # torch.Size([1, 192, 40, 40])
        P3_x = self.P3_cbl_set(P3_x)  # torch.Size([1, 64, 40, 40])

        if self.o_ceng == 3:
            res = [P3_x, P4_x, P5_x]
        elif self.o_ceng == 4:
            P6_x = self.P6(C5)
            P6_x = self.P6_cbl_set(P6_x)
            res = [P3_x, P4_x, P5_x, P6_x]
        elif self.o_ceng == 5:
            P6_x = self.P6(C5)  # torch.Size([1, 1024, 5, 5])
            P7_x = self.P7(P6_x)  # torch.Size([1, 2048, 3, 3])
            P6_x = self.P6_cbl_set(P6_x)  # torch.Size([1, 512, 5, 5])
            P7_x = self.P7_cbl_set(P7_x)  # torch.Size([1, 1024, 3, 3])

            res = [P3_x, P4_x, P5_x, P6_x, P7_x]
        else:
            raise Exception('self.o_ceng 出错 %s ' % self.o_ceng)

        return res


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

    print(model(torch.rand((1, 3, 416, 416))).shape)

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
