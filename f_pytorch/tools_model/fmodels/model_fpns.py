import torch
from torch import nn

from f_pytorch.tools_model.f_model_api import conv_bn1X1, conv_bn, conv_bn_no_relu
import torch.nn.functional as F


class FPN(nn.Module):
    '''
    三层图像形变大后 相加
    '''

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, inputs):
        # names = list(inputs.keys())
        # inputs = list(inputs.values())

        output1 = self.output1(inputs[0])  # torch.Size([8, 64, 80, 80])
        output2 = self.output2(inputs[1])  # torch.Size([8, 128, 40, 40])
        output3 = self.output3(inputs[2])  # torch.Size([8, 256, 20, 20])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out
