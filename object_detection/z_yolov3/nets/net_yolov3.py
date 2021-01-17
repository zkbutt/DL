import torch
import torch.nn as nn
from collections import OrderedDict

from f_pytorch.tools_model.fmodels.model_modules import SPP
from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import labels2onehot4ts
from f_tools.fits.f_predictfun import label_nms
import numpy as np

from f_tools.fun_od.f_anc import FAnchors
from f_tools.fun_od.f_boxes import xywh2ltrb, fix_boxes4yolo3, ltrb2xywh, get_boxes_colrow_index, calc_iou4ts, \
    bbox_iou4one
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import show_anc4pil, f_show_3box4pil
import torch.nn.functional as F


class LossYOLOv3(nn.Module):

    def __init__(self, anc_obj, cfg):
        super(LossYOLOv3, self).__init__()
        self.cfg = cfg
        self.anc_obj = anc_obj
        # self.focal_loss = FocalLoss_v2(alpha=cfg.FOCALLOSS_ALPHA, gamma=cfg.FOCALLOSS_GAMMA, reduction='none')

    def hard_avg(self, cfg, values, mask_ignore, mask_pos):
        '''
        mask_ignore 没有算 反倒比例+难例
        :param values: 25200
        :param mask_ignore:
        :param mask_pos:
        :return:
        '''
        device = values.device
        _values = values.clone().detach()
        _values[torch.logical_or(mask_ignore, mask_pos)] = torch.tensor(0.0, device=device)

        _, max_idx1 = _values.sort(dim=-1, descending=True)  # descending 倒序
        _, max_idx2 = max_idx1.sort(dim=-1)
        num_pos = mask_pos.sum()
        num_neg = num_pos.item() * cfg.NEG_RATIO
        mask_neg = max_idx2 < num_neg

        # mask = mask_pos.float() + mask_neg.float()
        # 正反例可以一起算
        l_conf_neg = (values * mask_neg.float()).sum() / num_pos
        l_conf_pos = (values * mask_pos.float()).sum() / num_pos
        return l_conf_pos, l_conf_neg

    def forward(self, p_yolo_ts, targets, imgs_ts=None):
        ''' 只支持相同的anc数

        :param p_yolo_ts: [5, 10647, 25])
        :param targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
            target['size'] = target['size']
            target['image_id'] = int
        :return:
        '''

        # loss 基本参数
        batch = len(targets)
        device = p_yolo_ts.device
        # 层尺寸   tensor([[52., 52.], [26., 26.], [13., 13.]])
        # feature_sizes = np.array(self.anc_obj.feature_sizes)
        feature_sizes = torch.tensor(self.anc_obj.feature_sizes, dtype=torch.float32, device=device)
        # tensor([8112, 2028,  507], dtype=torch.int32)
        nums_feature_offset = feature_sizes.prod(dim=1) * torch.tensor(self.cfg.NUMS_ANC, device=device)  # 2704 676 169

        # 2704 676 169 -> tensor([8112, 2028,  507])
        nums_ceng = (feature_sizes.prod(axis=1) * 3).type(torch.int)
        # 转为np 用数组索引 tensor([[52., 52.], [26., 26.],[13., 13.]]) -> torch.Size([10647, 2]) 数组索引 只能用np
        fsize_p_anc = np.repeat(feature_sizes.cpu(), nums_ceng.cpu(), axis=0)
        fsize_p_anc = fsize_p_anc.clone().detach().to(device)  # cpu->gpu 安全

        # 匹配完成的数据
        _num_total = sum(nums_feature_offset)  # 10647
        _dim = 4 + 1 + self.cfg.NUM_CLASSES  # 25
        nums_feature_offset[2] = nums_feature_offset[0] + nums_feature_offset[1]
        nums_feature_offset[1] = nums_feature_offset[0]
        nums_feature_offset[0] = 0

        loss_cls_pos = 0
        loss_box_pos = 0  # 只计算匹配的
        loss_conf_pos = 0  # 正反例， 正例*100 取难例
        loss_conf_neg = 0  # 正反例， 正例*100 取难例

        # 分批处理
        for i in range(batch):
            '''------------------负样本选择-------------------'''
            target = targets[i]
            # 只能一个个的处理
            g_boxes_ltrb = target['boxes']  # ltrb
            g_labels = target['labels']
            if g_boxes_ltrb.shape[0] == 0:
                # 没有目标的图片不要
                flog.error('boxes==0 %s', g_boxes_ltrb.shape)
                continue
            if self.cfg.IS_VISUAL:
                # 可视化1 原目标图 --- 初始化图片
                img_ts = imgs_ts[i]
                from torchvision.transforms import functional as transformsF
                img_ts = f_recover_normalization4ts(img_ts)
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                # show_anc4pil(img_pil, g_boxes_ltrb, size=img_pil.size)
                # img_pil.save('./1.jpg')

            '''组装 BOX 对应的 n*9个 匹配9个anc 计算出网格index'''
            g_boxes_xywh = ltrb2xywh(g_boxes_ltrb)
            num_anc = np.array(self.cfg.NUMS_ANC).sum()  # anc总和数 [3,3,3].sum() -> 9
            # 单体复制 每一个box重复9次
            g_boxes_xywh_p = g_boxes_xywh.repeat_interleave(num_anc, dim=0)
            # 每一个bbox对应的9个anc tensor([[52., 52.], [26., 26.], [13., 13.]])
            # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True) torch复制
            # feature_sizes = torch.tensor(feature_sizes.copy(), dtype=torch.float32).to(device)
            # 与多个box匹配 只支持每层相同的anc数(索引不支持数组) [[52., 52.],[52., 52.],[52., 52.], ...[26., 26.]..., [13., 13.]...]
            num_boxes = g_boxes_xywh.shape[0]
            # 这里 fsize_p n*9个 单体 + 整体
            fsize_p_n9 = feature_sizes.repeat_interleave(self.cfg.NUMS_ANC[0], dim=0).repeat(num_boxes, 1)
            # XY对就 col rows
            colrow_index = get_boxes_colrow_index(g_boxes_xywh_p[:, :2], fsize_p_n9)

            '''构造 与输出匹配的 n*9个anc'''
            # 求出anc对应网络的的中心点  在网格左上角
            _ancs_xy = colrow_index / fsize_p_n9  # tensor([[52., 52.], 52., 52.], [52., 52.],[26., 26.],[26., 26.],[26., 26.],[13., 13.],[13., 13.],[13., 13.]])
            # 大特图对应小目标 _ancs_scale 直接作为wh
            _ancs_scale = torch.tensor(self.anc_obj.ancs_scale).to(device)
            _ancs_wh = _ancs_scale.reshape(-1, 2).repeat(num_boxes, 1)  # 拉平后整体复制
            ancs_xywh = torch.cat([_ancs_xy, _ancs_wh], dim=1)

            # --------------可视化调试----------------
            if self.cfg.IS_VISUAL:
                # 显示 boxes 中心点 黄色, 及计算出来的匹配 anc 的位置 3个层的中心点是不一样的 显示的大框 在网格左上角
                # ancs_ltrb = xywh2ltrb(ancs_xywh)
                # f_show_3box4pil(img_pil, g_boxes=g_boxes_ltrb,  # 黄色
                #                 boxes1=ancs_ltrb[:3, :],
                #                 boxes2=ancs_ltrb[3:6, :],
                #                 boxes3=ancs_ltrb[6:, :],
                #                 grids=self.anc_obj.feature_sizes[-1])
                pass

            '''批量找出每一个box对应的anc index'''
            # 主动构建偏移 使每个框的匹配过程独立 tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0')
            ids = torch.arange(num_boxes).to(device)  # 9个anc 0,1,2 只支持每层相同的anc数
            # (n,4) + (n,1)---(n,4) = (n,4)  扩展
            g_boxes_ltrb_offset = g_boxes_ltrb + ids[:, None]  # boxes加上偏移 1,2,3
            # 单体复制 对应anc数 1,2,3 -> 000000000 111111111 222222222
            ids_offset = ids.repeat_interleave(num_anc)
            # 与box 匹配的 anc
            ancs_ltrb = xywh2ltrb(ancs_xywh)  # 转ltrb 用于计算iou
            # 两边都加了同样的偏移
            ancs_ltrb_offset = ancs_ltrb + ids_offset[:, None]
            iou = calc_iou4ts(g_boxes_ltrb_offset, ancs_ltrb_offset, is_ciou=True)  # 这里都等于0
            # iou_sort = torch.argsort(-iou, dim=1) % num_anc  # 9个
            # iou_sort = torch.argsort(-iou, dim=1) % num_anc  # 求余数
            _, max_indexs = iou.max(dim=1)  # box对应anc的 index
            max_indexs = max_indexs % num_anc  # index 偏移值的修正

            '''---- 整体修复box  anc左上角+偏移  ---'''
            # 找出与pbox  与gbox 最大的index
            p_box_xywh = fix_boxes4yolo3(p_yolo_ts[i, :, :4], self.anc_obj.ancs, fsize_p_anc)

            p_box_ltrb = xywh2ltrb(p_box_xywh)
            ious = calc_iou4ts(g_boxes_ltrb, p_box_ltrb, is_ciou=True)  # 全部计算IOU
            # p_box_ltrb 对应的最大的 iou 值
            max_ious, _ = ious.max(dim=0)

            num_feature = len(self.cfg.NUMS_ANC)
            # 计算最大 anc 索引对应在哪个特图层
            match_feature_index = torch.true_divide(max_indexs, num_feature).type(torch.int64)  #
            # tensor([8112., 2028.,  507.]) ->[0., 8112,  2028.+8112.]
            num_feature_offset = nums_feature_offset[match_feature_index]
            colrow_num = colrow_index[torch.arange(num_boxes, device=device) * num_anc + max_indexs]
            _row_index = colrow_num[:, 1]  # [1,2]
            _col_index = colrow_num[:, 0]  # [3,2]
            # 特图层的行列偏移
            offset_colrow = _row_index * feature_sizes[match_feature_index][:, 0] + _col_index
            # 对应物图层的获取anc数
            _nums_anc = torch.tensor(self.cfg.NUMS_ANC, device=device)[match_feature_index]
            offset_total = num_feature_offset + offset_colrow * _nums_anc
            '''这里这个 match_index_pos 有可能已经用了'''
            match_index_pos = (offset_total + max_indexs % num_feature).type(torch.int64)

            '''---------  conf 损失 ---------'''
            # ----------选出难例负例----------
            mask_neg = max_ious > self.cfg.THRESHOLD_CONF_NEG
            mask_ignore = torch.logical_not(mask_neg)
            pconf = p_yolo_ts[i, :, 4]
            # p_yolo_ts[mask_ignore] = 0  # 忽略正反例
            # print(pconf[match_index_pos].sigmoid().tolist())
            gconf = torch.zeros_like(pconf)
            gconf[match_index_pos] = 1  # 25200
            mask_pos = gconf.type(torch.bool)
            _l_conf = F.binary_cross_entropy_with_logits(pconf, gconf, reduction='none')
            _l_conf_pos, _l_conf_neg = self.hard_avg(self.cfg, _l_conf, mask_ignore, mask_pos)

            # l_conf = self.focal_loss(pconf, gconf)
            # match_index_neg = fmatch_OHEM(l_conf, match_index_pos,
            #                               neg_ratio=self.cfg.NEG_RATIO, num_neg=2000, device=device, dim=-1)
            # _l_conf_pos += (l_conf.sum() / num_boxes)

            # focalloss
            # pconf = p_yolo_ts[i, :, 4]
            # # print(pconf[match_index_pos].sigmoid().tolist())
            # gconf = torch.zeros_like(pconf)
            # gconf[match_index_pos] = 1  # 25200
            # # _l_conf += (self.focal_loss(pconf, gconf) / num_boxes)
            # _l_conf += (self.focal_loss(pconf, gconf))

            '''-------- 计算正例 cls 损失 -----------'''
            pcls = p_yolo_ts[i, match_index_pos, 5:]
            # label是从1开始 找出正例
            _t = labels2onehot4ts(g_labels - 1, self.cfg.NUM_CLASSES)
            _l_cls_pos = (F.binary_cross_entropy_with_logits(pcls, _t.type(torch.float), reduction='sum') / num_boxes)
            # _l_cls_pos += (self.focal_loss(pcls, _t.type(torch.float)) / num_boxes)
            # _l_cls_pos += (self.focal_loss(pcls, _t.type(torch.float)))

            '''----- 正例box 损失 -----'''
            ious = bbox_iou4one(g_boxes_ltrb, p_box_ltrb[match_index_pos, :], is_ciou=True)
            # ious = bbox_iou4one(g_boxes_ltrb, p_box_ltrb[match_index_pos, :], is_giou=True)
            # ious = bbox_iou4one(g_boxes_ltrb, p_box_ltrb[match_index_pos, :], is_diou=True)
            w = 2 - g_boxes_xywh[:, 2] * g_boxes_xywh[:, 3]  # 增加小目标的损失权重
            _l_box_pos = torch.mean(w * (1 - ious))  # 每个框的损失
            # _l_box_pos += torch.sum(w * (1 - ious))  # 每个框的损失

            if self.cfg.IS_VISUAL:
                flog.debug('conf neg 损失 %s', _l_conf_neg)
                flog.debug('box 损失 %s', torch.mean(w * (1 - ious)))
                flog.debug('conf pos 损失 %s', _l_conf_pos)
                flog.debug('cls 损失 %s', _l_cls_pos)
                flog.debug('-------------------')
                f_show_3box4pil(img_pil, g_boxes=g_boxes_ltrb,
                                boxes1=p_box_ltrb[match_index_pos, :],
                                boxes2=xywh2ltrb(self.anc_obj.ancs[match_index_pos, :]),
                                grids=self.anc_obj.feature_sizes[-1],  # 网格
                                )

            loss_cls_pos = loss_cls_pos + _l_cls_pos
            loss_box_pos = loss_box_pos + _l_box_pos  # 上面已平均
            loss_conf_pos = loss_conf_pos + _l_conf_pos
            loss_conf_neg = loss_conf_neg + _l_conf_neg

        l_box_p = loss_box_pos / batch * self.cfg.LOSS_WEIGHT[0]
        l_conf_pos = loss_conf_pos / batch * self.cfg.LOSS_WEIGHT[1]
        l_conf_neg = loss_conf_neg / batch * self.cfg.LOSS_WEIGHT[2]
        l_cls_p = loss_cls_pos / batch * self.cfg.LOSS_WEIGHT[3]
        loss_total = l_box_p + l_conf_pos + l_conf_neg + l_cls_p

        # debug
        # _v = p_yolo_ts[:, :, 4].clone().detach().sigmoid()
        # print('min:%s mean:%s max:%s' % (_v.min().item(),
        #                                  _v.mean().item(),
        #                                  _v.max().item(),
        #                                           ))

        log_dict = {}
        log_dict['loss_total'] = loss_total.item()
        log_dict['l_box_p'] = l_box_p.item()
        log_dict['l_conf_p'] = l_conf_pos.item()
        log_dict['l_conf_n'] = l_conf_neg.item()
        log_dict['l_cls_p'] = l_cls_p.item()
        # log_dict['l_conf_min'] = _v.min().item()
        # log_dict['l_conf_mean'] = _v.mean().item()
        # log_dict['l_conf_max'] = _v.max().item()
        # log_dict['l_conf_top100'] = _v.topk(10).values.tolist() # 不支持数组
        # print(_v.topk(20).values.tolist())

        return loss_total, log_dict


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
        # flog.info('conf 最大 %s', p_yolo_ts4[:, :, 4].max())
        mask_box = p_yolo_ts4[:, :, 4] > self.threshold_conf
        if not torch.any(mask_box):  # 如果没有一个对象
            # flog.error('该批次没有找到目标')
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
        p_boxes_xywh = fix_boxes4yolo3(p_yolo_ts4[:, :, :4], self.anc_obj.ancs, fsize_p)
        # p_boxes_xy = p_yolo_ts4[:, :, :2] / fsize_p + self.anc_obj.ancs[:, :2]  # offxy -> xy
        # p_boxes_wh = p_yolo_ts4[:, :, 2:4].exp() * self.anc_obj.ancs[:, 2:]  # wh修复
        # p_boxes_xywh = torch.cat([p_boxes_xy, p_boxes_wh], dim=-1)

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

        if self.cfg.IS_VISUAL:
            # 可视化1 原目标图 --- 初始化图片
            flog.debug('conf后 %s', )
            img_ts = imgs_ts[0]
            from torchvision.transforms import functional as transformsF
            img_ts = f_recover_normalization4ts(img_ts)
            img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
            show_anc4pil(img_pil, p_boxes_ltrb1, size=img_pil.size)
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

        # 自定义数据重装函数 torch.Size([1, 10647, 25])  堆52 堆26 堆13
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
    pass

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
