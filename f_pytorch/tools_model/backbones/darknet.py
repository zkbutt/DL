import torch
import torch.nn as nn

from f_pytorch.tools_model.model_look import f_look2


def Conv3x3BNReLU(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU6(inplace=True), # 最大值为6
        nn.LeakyReLU(inplace=True),  # 负x轴给一个固定的斜率
        # nn.RReLU(inplace=True),  # 负x轴给定范围内随机
    )


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU6(inplace=True),
        nn.LeakyReLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, nchannels):
        super(Residual, self).__init__()
        mid_channels = nchannels // 2
        self.conv1x1 = Conv1x1BNReLU(in_channels=nchannels, out_channels=mid_channels)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels, out_channels=nchannels)

    def forward(self, x):
        out = self.conv3x3(self.conv1x1(x))
        return out + x


class Darknet(nn.Module):
    def __init__(self, nums_layer=(1, 2, 8, 8, 4), num_classes=1000):
        super(Darknet, self).__init__()
        self.first_conv = Conv3x3BNReLU(in_channels=3, out_channels=32)

        self.block1 = self._make_layers(in_channels=32, out_channels=64, block_num=nums_layer[0])
        self.block2 = self._make_layers(in_channels=64, out_channels=128, block_num=nums_layer[1])
        self.block3 = self._make_layers(in_channels=128, out_channels=256, block_num=nums_layer[2])
        self.block4 = self._make_layers(in_channels=256, out_channels=512, block_num=nums_layer[3])
        self.block5 = self._make_layers(in_channels=512, out_channels=1024, block_num=nums_layer[4])

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)# F.adaptive_avg_pool2d(x.view(x.size(0), -1), 1)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layers(self, in_channels, out_channels, block_num):
        _layers = []
        _layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=2))
        for _ in range(block_num):
            _layers.append(Residual(nchannels=out_channels))
        return nn.Sequential(*_layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        out = self.softmax(x)
        return out


if __name__ == '__main__':
    model = Darknet(nums_layer=(1, 2, 8, 8, 4))

    print(model)
    data_inputs_list = [1, 3, 416, 416]

    f_look2(model, data_inputs_list[-3:])
    # f_look(model, data_inputs_list)
    input = torch.randn(*data_inputs_list)
    out = model(input)
    print(out.shape)
