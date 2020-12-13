import torch
from torch import nn

from f_tools.GLOBAL_LOG import flog
from f_tools.f_predictfun import label_nms
from f_tools.fits.f_lossfun import LossYOLOv1
from f_tools.fun_od.f_boxes import offxy2xy, xywh2ltrb
import numpy as np


class PredictYolov1(nn.Module):
    def __init__(self, num_bbox, num_classes, num_grid, threshold_conf=0.5, threshold_nms=0.3, ):
        super(PredictYolov1, self).__init__()
        self.num_bbox = num_bbox
        self.num_classes = num_classes
        self.num_grid = num_grid
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf

    def forward(self, p_yolo_ts4):
        '''
        批量处理 conf + nms
        :param p_yolo_ts: torch.Size([5, 7, 7, 11])
        :return:
            ids_batch2 [nn]
            p_boxes_ltrb2 [nn,4]
            p_labels2 [nn]
            p_scores2 [nn]
        '''
        # 确认一阶段有没有目标
        _is = self.num_bbox * 4
        torch.sigmoid_(p_yolo_ts4[:, :, :, _is:])  # 处理conf 和 label
        mask_box = p_yolo_ts4[:, :, :, _is: _is + 2] > self.threshold_conf  # torch.Size([104, 7, 7, 2])
        if not torch.any(mask_box):  # 如果没有一个对象
            flog.error('没有找到目标')
            return [None] * 4

        device = p_yolo_ts4.device
        # batch = p_yolo_ts4.shape[0]

        '''处理box'''
        # [5, 7, 7, 8] -> [5, 7, 7, 2, 4]
        p_boxes_offxywh = p_yolo_ts4[:, :, :, :_is].view(*p_yolo_ts4.shape[:-1], self.num_bbox, 4)
        # torch.Size([5, 7, 7, 2])


        # [5, 7, 7, 2, 4]^^[5, 7, 7, 2] -> [nn,4] 全正例
        _p_boxes = p_boxes_offxywh[mask_box]
        torch.sigmoid_(_p_boxes[:, :2])
        ids_batch1, ids_row, ids_col, ids_box = torch.where(mask_box)
        grids = torch.tensor([self.num_grid] * 2, device=device, dtype=torch.float)
        colrow_index = torch.cat([ids_col[:, None], ids_row[:, None]], dim=1)
        # 修复 p_boxes_pos
        _p_boxes[:, :2] = offxy2xy(_p_boxes[:, :2], colrow_index, grids)
        p_boxes_ltrb1 = xywh2ltrb(_p_boxes)

        '''处理 label scores'''
        p_labels1 = []
        p_scores1 = []
        for i in range(p_boxes_ltrb1.shape[0]):
            _label_start = self.num_bbox * 4 + self.num_bbox
            _, max_index = p_yolo_ts4[ids_batch1[i], ids_row[i], ids_col[i], _label_start:].max(dim=0)
            p_labels1.append(max_index + 1)  # 类型加1
            _score = p_yolo_ts4[ids_batch1[i], ids_row[i], ids_col[i], self.num_bbox * 4 + ids_box[i]]
            p_scores1.append(_score)  # 类型加1

        # [5, 7, 7, 2] -> [5, 7, 7]
        # mask_yolo = torch.any(mask_box, dim=-1)
        p_labels1 = torch.tensor(p_labels1, device=device, dtype=torch.float)
        p_scores1 = torch.tensor(p_scores1, device=device, dtype=torch.float)

        # 分类 nms
        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    p_boxes_ltrb1,
                                                                    p_labels1,
                                                                    p_scores1,
                                                                    device,
                                                                    self.threshold_nms)

        return ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2


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
    def __init__(self, backbone, dim_in, grid, num_classes, num_bbox, cfg=None):
        super(Yolo_v1, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.grid = grid
        self.num_bbox = num_bbox

        # 以下是YOLOv1的最后四个卷积层
        dim_layer = 1024
        self.fpn_yolov1 = FPNYolov1(dim_in, dim_layer)
        dim_out = self.num_bbox * (4 + 1) + self.num_classes
        self.head_yolov1 = nn.Conv2d(dim_layer, dim_out, kernel_size=(1, 1))
        self.loss = LossYOLOv1(num_cls=num_classes, grid=grid, num_bbox=num_bbox,
                               threshold_box=cfg.THRESHOLD_BOX,
                               threshold_conf_neg=cfg.THRESHOLD_CONF_NEG,
                               )
        self.pred = PredictYolov1(num_bbox=num_bbox,
                                  num_classes=num_classes,
                                  num_grid=cfg.NUM_GRID,
                                  threshold_conf=cfg.THRESHOLD_PREDICT_CONF,
                                  threshold_nms=cfg.THRESHOLD_PREDICT_NMS)  # prediction

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
        x = x.permute(0, 2, 3, 1).contiguous()  # torch.Size([1, 11, 7, 7])

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.loss(x, targets)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_labels, p_scores = self.pred(x)
            return ids_batch, p_boxes_ltrb, p_labels, p_scores
        # if torch.jit.is_scripting():  # 这里是生产环境部署


if __name__ == '__main__':
    pass
