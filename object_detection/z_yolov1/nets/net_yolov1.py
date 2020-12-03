import torch
from torch import nn

from f_tools.fits.f_lossfun import LossYOLOv1


class FPNYolov1(nn.Module):
    def __init__(self, dim_in, dim_layer=1024):
        super(FPNYolov1, self).__init__()
        self.fpn_yolov1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_layer, 3, padding=1),
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

    def forward(self, x):
        x = self.fpn_yolov1(x)
        return x


class Yolo_v1(nn.Module):
    def __init__(self, backbone, dim_in, grid, num_classes, num_bbox):
        super(Yolo_v1, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.grid = grid
        self.num_bbox = num_bbox

        # 以下是YOLOv1的最后四个卷积层
        dim_layer = 1024
        self.fpn_yolov1 = FPNYolov1(dim_in, dim_layer)
        dim_out = 4 * self.num_bbox + 1 + self.num_classes
        self.head_yolov1 = nn.Conv2d(dim_layer, dim_out, kernel_size=(1, 1))
        self.loss = LossYOLOv1()
        self.pred = None  # prediction

        # # 以下是YOLOv1的最后2个全连接层
        # self.out_layout = nn.Sequential(
        #     nn.Linear(grid * grid * dim_layer, dim_layer * 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(dim_layer * 4, grid * grid * (1 + 4 + self.num_bbox * self.num_classes)),
        #     # nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间
        # )

    def forward(self, x, targets=None):
        x = self.backbone(x)  # 输出 torch.Size([1, 1280, 13, 13])
        x = self.fpn_yolov1(x)  # 输出torch.Size([1, 1024, 7, 7])
        x = self.head_yolov1(x)  # 输出torch.Size([1, 490, 7, 7])
        x = x.permute(0, 2, 3, 1).contiguous()  # torch.Size([1, 10, 7, 7])

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.loss(x, targets)
            return loss_total, log_dict
        else:
            self.pred
            pass
        # if torch.jit.is_scripting():  # 这里是生产环境部署

        return x


if __name__ == '__main__':
    pass
