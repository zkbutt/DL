import torch
import torch.nn as nn
from collections import OrderedDict

from f_pytorch.tools_model.f_layer_get import ModelOut4Densenet121
from f_pytorch.tools_model.fmodels.model_modules import SPP
from f_pytorch.tools_model.model_look import f_look_model
from f_tools.GLOBAL_LOG import flog
from f_tools.f_od_gen import f_get_rowcol_index
from f_tools.f_predictfun import label_nms
from f_tools.fits.f_lossfun import LossYOLOv3
import numpy as np

from f_tools.fun_od.f_anc import FAnchors
from f_tools.fun_od.f_boxes import xywh2ltrb
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import show_anc4pil


class PredictYolov3(nn.Module):
    def __init__(self, anc_obj, cfg, threshold_conf=0.5, threshold_nms=0.3, ):
        super(PredictYolov3, self).__init__()
        # self.num_bbox = num_bbox
        # self.num_classes = num_classes
        # self.num_grid = num_grid
        self.cfg = cfg
        self.anc_obj = anc_obj
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf

    def forward(self, p_yolo_ts4, imgs_ts=None):
        '''
        批量处理 conf + nms
        :param p_yolo_ts: torch.Size([7, 10647, 25])
        :return:
            ids_batch2 [nn]
            p_boxes_ltrb2 [nn,4]
            p_labels2 [nn]
            p_scores2 [nn]
        '''
        # 确认一阶段有没有目标
        torch.sigmoid_(p_yolo_ts4[:, :, 4:])  # 处理conf 和 label
        # torch.Size([7, 10647, 25]) -> torch.Size([7, 10647])
        mask_box = p_yolo_ts4[:, :, 4] > self.threshold_conf
        if not torch.any(mask_box):  # 如果没有一个对象
            flog.error('该批次没有找到目标')
            return [None] * 4

        device = p_yolo_ts4.device
        # batch = p_yolo_ts4.shape[0]
        # dim = 4 + 1 + self.cfg.NUM_CLASSES

        # tensor([[52, 52],[26, 26],[13, 13]])
        # feature_sizes = np.array(self.anc_obj.feature_sizes)
        # nums_feature_offset = feature_sizes.prod(axis=1)  # 2704 676 169
        feature_sizes = np.array(self.anc_obj.feature_sizes, dtype=np.float32)
        # 2704 676 169 -> tensor([8112, 2028,  507])
        nums_ceng = (feature_sizes.prod(axis=1) * 3).astype(np.int64)
        # 索引要int
        fsize_p = np.repeat(feature_sizes, nums_ceng, axis=0)
        fsize_p = torch.tensor(fsize_p, device=device)

        # 匹配 rowcol
        # rowcol_index = torch.empty((0, 2), device=device)
        # for s, num_anc in zip(self.anc_obj.feature_sizes, self.cfg.NUMS_ANC):
        #     _rowcol_index = f_get_rowcol_index(*s).repeat_interleave(3, dim=0)
        #     rowcol_index = torch.cat([rowcol_index, _rowcol_index], dim=0)

        '''全量 修复box '''
        p_boxes_xy = p_yolo_ts4[:, :, :2] / fsize_p + self.anc_obj.ancs[:, :2]  # offxy -> xy
        p_boxes_wh = p_yolo_ts4[:, :, 2:4].exp() * self.anc_obj.ancs[:, 2:]  # wh修复
        p_boxes_xywh = torch.cat([p_boxes_xy, p_boxes_wh], dim=-1)

        '''第一阶段'''
        ids_batch1, _ = torch.where(mask_box)
        # torch.Size([7, 10647, 4]) -> (nn,4)
        p_boxes_ltrb1 = xywh2ltrb(p_boxes_xywh[mask_box])
        # torch.Size([7, 10647, 1]) -> (nn)
        p_scores1 = p_yolo_ts4[:, :, 4][mask_box]
        # torch.Size([7, 10647, 20]) -> (nn,20)
        p_labels1_one = p_yolo_ts4[:, :, 5:][mask_box]
        _, p_labels1_index = p_labels1_one.max(dim=1)
        p_labels1 = p_labels1_index + 1

        # if self.cfg.IS_VISUAL:
        #     # 可视化1 原目标图 --- 初始化图片
        #     img_ts = imgs_ts[0]
        #     from torchvision.transforms import functional as transformsF
        #     img_ts = f_recover_normalization4ts(img_ts)
        #     img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
        #     show_anc4pil(img_pil, p_boxes_ltrb1, size=img_pil.size)
        #     # img_pil.save('./1.jpg')

        # 分类 nms
        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    p_boxes_ltrb1,
                                                                    p_labels1,
                                                                    p_scores1,
                                                                    device,
                                                                    self.threshold_nms)

        return ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU()),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    '''
    5次卷积出结果
    :param filters_list:[512, 1024] 交替Conv
    :param in_filters: 1024 输入维度
    :param out_filter: 输出 维度 （20+1+4）*anc数
    :return:
    '''
    m = nn.ModuleList([  # 共7层
        conv2d(in_filters, filters_list[0], 1),  # 1卷积 + 3卷积 交替
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),  # 这里加spp
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),  # 这里输出上采样
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    ])
    return m


class YoloV3SPP(nn.Module):
    def __init__(self, backbone, nums_anc, num_classes, dims_rpn_in, cfg, device, is_spp=False):
        '''
        层属性可以是 nn.Module nn.ModuleList(封装Sequential) nn.Sequential
        '''
        super(YoloV3SPP, self).__init__()
        assert len(dims_rpn_in) == 3, '输出三个维度 的长度必须是3'
        self.cfg = cfg
        #  backbone
        self.backbone = backbone
        self.nums_anc = nums_anc
        if is_spp:
            self.is_spp = is_spp
            self.spp = SPP(int(dims_rpn_in[2] / 2))

        anc_obj = FAnchors(cfg.IMAGE_SIZE, cfg.ANC_SCALE, cfg.FEATURE_MAP_STEPS, cfg.ANCHORS_CLIP, device=device)

        '''根据不同的输出层的anc参数，确定输出结果'''
        final_out_filter1 = nums_anc[0] * (1 + 4 + num_classes)
        self.last_layer1 = make_last_layers([int(dims_rpn_in[0] / 2), dims_rpn_in[0]],
                                            dims_rpn_in[0] + int(dims_rpn_in[1] / 4),  # # 叠加上层的输出
                                            final_out_filter1)

        final_out_filter2 = nums_anc[1] * (1 + 4 + num_classes)
        self.last_layer2 = make_last_layers([int(dims_rpn_in[1] / 2), dims_rpn_in[1]],  # 小大震荡
                                            dims_rpn_in[1] + int(dims_rpn_in[2] / 4),  # 叠加上层的输出
                                            final_out_filter2)
        self.last_layer2_conv = conv2d(int(dims_rpn_in[1] / 2), int(dims_rpn_in[1] / 4), 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        final_out_filter3 = nums_anc[2] * (1 + 4 + num_classes)
        self.last_layer3 = make_last_layers([int(dims_rpn_in[2] / 2), dims_rpn_in[2]],  # 小大震荡 输入的一半 and 还原
                                            dims_rpn_in[2],
                                            final_out_filter3)
        self.last_layer3_conv = conv2d(int(dims_rpn_in[2] / 2), int(dims_rpn_in[2] / 4), 1)  # 决定上采的维度
        # F.interpolate(x,scale_factor=2,mode='nearest')
        self.last_layer3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.loss = LossYOLOv3(anc_obj, cfg)
        # self.sigmoid_out = nn.Sigmoid()
        self.pred = PredictYolov3(anc_obj, cfg, cfg.THRESHOLD_PREDICT_CONF, cfg.THRESHOLD_PREDICT_NMS)

    def forward(self, x, targets=None):
        def _branch(last_layer, layer_in, is_spp=False):
            '''

            :param last_layer: 五层CONV
            :param layer_in: 输入数据
            :return:
                layer_in 输出数据
                out_branch 上采样输入
            '''
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 2 and is_spp:
                    layer_in = self.spp(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        #  backbone ([batch, 512, 52, 52]) ([batch, 1024, 26, 26])  ([batch, 1024, 13, 13])
        backbone_out1, backbone_out2, backbone_out3 = self.backbone(x)
        #  [batch, 1024, 13, 13] -> [batch, 75, 13, 13],[batch, 2048, 13, 13]
        out3, out3_branch = _branch(self.last_layer3, backbone_out3, self.is_spp)
        # [batch, 2048, 13, 13] -> [batch, 1024, 13, 13]
        _out = self.last_layer3_conv(out3_branch)
        _out = self.last_layer3_upsample(_out)
        _out = torch.cat([_out, backbone_out2], 1)  # 叠加 torch.Size([1, 1280, 26, 26])
        out2, out2_branch = _branch(self.last_layer2, _out)  # torch.Size([batch, 75, 26, 26])

        _out = self.last_layer2_conv(out2_branch)  # torch.Size([1, 128, 26, 26])
        _out = self.last_layer2_upsample(_out)  # torch.Size([batch, 128, 26, 26]) -> torch.Size([1, 128, 52, 52])
        _out = torch.cat([_out, backbone_out1], 1)  # 叠加
        out1, _ = _branch(self.last_layer1, _out)  # torch.Size([batch, 75, 52, 52])

        # 自定义数据重装函数 torch.Size([1, 10647, 25])
        outs = self.data_packaging([out1, out2, out3], self.nums_anc)
        '''这里输出每层每格的对应三个anc'''
        # outs[:, :, :2] = self.sigmoid_out(outs[:, :, :2])  # xy归一
        # outs[:, :, 4:] = self.sigmoid_out(outs[:, :, 4:])  # 支持多标签
        '''为每一个特图预测三个尺寸的框,拉平堆叠'''
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.loss(outs, targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_labels, p_scores = self.pred(outs, x)
            return ids_batch, p_boxes_ltrb, p_labels, p_scores
        # if torch.jit.is_scripting():  # 这里是生产环境部署

    def data_packaging(self, outs, nums_anc):
        '''
        3个输入 合成一个输出 与anc进行拉伸
        :param outs: [out1, out2, out3] b,c,h,w
        :param nums_anc: ans数组
        :return: torch.Size([1, 10647, 25])
        '''
        _ts = []
        for out, num_anc in zip(outs, nums_anc):
            batch, o, w, h = out.shape  # torch.Size([5, 75, 52, 52])
            out = out.permute(0, 2, 3, 1)  # torch.Size([5, 52, 52, 75])
            # [5, 52, 52, 75] -> [5, 52*52*3, 25]
            _ts.append(out.reshape(batch, -1, int(o / num_anc)).contiguous())
        # torch.Size([1, 8112, 25])，torch.Size([1, 2028, 25])，torch.Size([1, 507, 25]) -> torch.Size([1, 10647, 25])
        return torch.cat(_ts, dim=1)


if __name__ == '__main__':
    from torchvision import models

    # model = models.densenet121(pretrained=True)
    # # # f_look(model, input=(1, 3, 416, 416))
    # # # conv 可以取 in_channels 不支持数组层
    # dims_out = [512, 1024, 1024]
    # # print(dims_out)
    # ret_name_dict = {'denseblock2': 1, 'denseblock3': 2, 'denseblock4': 3}
    # model = ModelOut4Densenet121(model, 'features', ret_name_dict)
    # f_look_model(model, input=(1, 3, 416, 416))
    #
    # # model = Darknet(nums_layer=(1, 2, 8, 8, 4))
    # # return_layers = {'block3': 1, 'block4': 2, 'block5': 3}
    # # model = Output4Return(model, return_layers)
    # # dims_out = [256, 512, 1024]
    #
    # nums_anc = [3, 3, 3]
    # num_classes = 20
    # model = YoloV3SPP(model, nums_anc, num_classes, dims_out, is_spp=True)
    # f_look_model(model, input=(1, 3, 416, 416))
    # # f_look2(model, input=(3, 416, 416))
    #
    # # torch.save(model, 'yolov3spp.pth')
