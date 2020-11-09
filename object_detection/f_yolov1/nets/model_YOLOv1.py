from collections import OrderedDict

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import models


class Yolo_v1(nn.Module):
    def __init__(self, backbone, out_dim, grid, num_classes):
        super(Yolo_v1, self).__init__()
        self.num_classes = num_classes
        self.grid = grid

        # layer = OrderedDict()
        # for name, module in backbone._modules.items():
        #     # print(name, module)
        #     if name == 'classifier':  # mov1
        #         break
        #     elif name == 'avgpool':  # resnet
        #         break
        #     layer[name] = module
        # self.backbone = nn.Sequential(layer)

        self.backbone = backbone  # 去除resnet的最后两层

        # 以下是YOLOv1的最后四个卷积层
        dim_layer = 1024
        self.yolov1_layers = nn.Sequential(
            nn.Conv2d(out_dim, dim_layer, 3, padding=1),
            nn.BatchNorm2d(dim_layer),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(dim_layer, dim_layer, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim_layer),
            nn.LeakyReLU(),
            nn.Conv2d(dim_layer, dim_layer, 3, padding=1),
            nn.BatchNorm2d(dim_layer),
            nn.LeakyReLU(),
            nn.Conv2d(dim_layer, dim_layer, 3, padding=1),
            nn.BatchNorm2d(dim_layer),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.out_layout = nn.Sequential(
            nn.Linear(grid * grid * dim_layer, dim_layer * 4),
            nn.LeakyReLU(),
            nn.Linear(dim_layer * 4, grid * grid * (1 + 4 + self.num_classes)),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )
        '''层权重初始化'''
        # self.init_weight()

    def init_weight(self):
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
        x = self.backbone(x)
        x = self.yolov1_layers(x)
        batch = x.size()[0]
        x = self.out_layout(x.view(batch, -1))  # 拉平
        x = x.view(batch, self.grid, self.grid, 1 + 4 + self.num_classes).contiguous()
        return x


import torch.nn.functional as F

if __name__ == '__main__':
    cuda_idx = 1
    is_mixture_fix = True
    device = torch.device("cuda:%s" % cuda_idx if torch.cuda.is_available() else "cpu")

    # model = models.mobilenet_v2(pretrained=True)
    # in_features = list(model.classifier.children())[-1].in_features
    # model = nn.Sequential(*list(model.children())[:-1])  # 去除resnet的最后两层

    model = models.densenet121(pretrained=True)
    out_dim = model.classifier.in_features
    model = nn.Sequential(*list(model.children())[:-1])  # 去除resnet的最后两层

    model = Yolo_v1(model, out_dim=out_dim, grid=7, num_classes=20)
    model.to(device)
    batch = 2
    input_shape = [batch, 3, 416, 416]
    out_shape = [batch, 7, 7, 25]
    # data_inputs_list = [batch, 3, 224, 224]
    # import tensorwatch as tw

    # args_pd = tw.model_stats(yolo_v1, data_inputs_list)
    # args_pd.to_excel('model_log.xlsx')

    # from torchsummary import summary
    # summary = summary(yolo_v1, (3, 416, 416))
    lr0 = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=5e-4)  # 权重衰减(如L2惩罚)(默认: 0)

    MB = 1024.0 * 1024.0
    scaler = GradScaler(enabled=is_mixture_fix)

    for i in range(999):
        input = torch.rand(input_shape).to(device)
        label = torch.rand(out_shape).to(device)
        if is_mixture_fix:
            with autocast():
                out = model(input)
                print(out.shape)
                loss = F.mse_loss(out, label)
            if i % 10 == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            out = model(input)
            print(out.shape)
            loss = F.mse_loss(out, label.to(out))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('loss', loss)
        print("{}:{}".format(i + 1, torch.cuda.memory_allocated(1)))
