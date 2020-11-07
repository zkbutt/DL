from typing import OrderedDict

import torch
from torch import nn
from torchvision import models


class Yolo_v1(nn.Module):
    def __init__(self, backbone, grid, num_classes):
        super(Yolo_v1, self).__init__()
        self.num_classes = num_classes
        self.grid = grid

        layer = OrderedDict()
        for name, module in backbone._modules.items():
            # print(name, module)
            if name == 'classifier':
                break
            layer[name] = module
        self.backbone = nn.Sequential(layer)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_out = nn.Sequential(
            nn.Linear(1280 * grid * grid, 4096),
            # nn.Linear(1280, 1280),
            nn.ReLU(),
            # nn.Linear(1280, 1280),
            # nn.ReLU(),
            nn.Linear(4096, grid * grid * (1 + 4 + num_classes)),  # torch.Size([1, 845])
        )
        '''层权重初始化'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)  # torch.Size([5, 1280, 13, 13])
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        batch = x.size(0)
        x = x.view(batch, -1)
        x = self.layer_out(x)
        # x_out = self.detect(x)
        x = x.view(batch, self.grid, self.grid, 1 + 4 + self.num_classes).contiguous()

        return x


if __name__ == '__main__':
    backbone = models.mobilenet_v2(pretrained=True)
    yolo_v1 = Yolo_v1(backbone, grid=7, num_classes=20)

    # data_inputs_list = [1, 3, 416, 416]
    data_inputs_list = [1, 3, 224, 224]
    import tensorwatch as tw

    args_pd = tw.model_stats(yolo_v1, data_inputs_list)
    args_pd.to_excel('model_log.xlsx')

    # from torchsummary import summary
    # summary = summary(yolo_v1, (3, 416, 416))

    input = torch.empty(data_inputs_list)
    out = yolo_v1(input)
    print(out.shape)
