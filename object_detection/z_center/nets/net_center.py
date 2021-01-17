import torch
import torch.nn as nn

from f_pytorch.tools_model.f_layer_get import ModelOut4Mobilenet_v2
from f_pytorch.tools_model.f_model_api import finit_weights
from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_lossfun import focal_loss4center, focal_loss4center2, cneg_loss, focalloss_v2, GHMC_Loss, \
    FocalLoss_v2
from f_tools.fits.f_match import match4center
from f_tools.fits.f_predictfun import label_nms4keypoints
from f_tools.fun_od.f_boxes import xywh2ltrb, ltrb2xywh, offxy2xy
from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
from f_tools.pic.f_show import show_anc4pil
import torch.nn.functional as F


class LoosCenter(nn.Module):
    def __init__(self, cfg):
        super(LoosCenter, self).__init__()
        self.cfg = cfg
        # self.fun_ghmc = GHMC_Loss(num_bins=10, momentum=0.25, reduction='sum')
        self.fun_ghmc = GHMC_Loss(num_bins=10, momentum=0.25, reduction='mean')
        self.fun_focalloss_v2 = FocalLoss_v2(reduction='sum')

    def forward(self, p_center, targets, imgs_ts=None):
        '''

        :param p_center: 全部已归一化
            photmap: torch.Size([5, 20, 128, 128])  0~1 20个类
            pwh : torch.Size([5, 2, 128, 128])  归一化尺寸
            pxy_offset : torch.Size([5, 2, 128, 128]) 相对网格 同yolo
        :param g_center: torch.Size([5, 128, 128, 24])
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        pheatmap, pwh, pxy_offset, pkeypoint_offset = p_center

        device = pheatmap.device
        fsize = torch.tensor(pheatmap.shape[-2:], device=device)
        batch = len(targets)

        pheatmap = pheatmap.permute(0, 2, 3, 1)  # [5, 20, 128, 128] -> [5, 128, 128,20]

        # num_class + 中间给关键点 + xy偏移 + wh偏移  conf通过高斯生成 热力图层数表示类别索引
        if cfg.NUM_KEYPOINTS > 0:
            dim = cfg.NUM_CLASSES + cfg.NUM_KEYPOINTS * 2 + 4
        else:
            dim = cfg.NUM_CLASSES + 4

        g_center = torch.zeros((batch, fsize[0], fsize[1], dim), dtype=torch.float, device=device)
        for i in range(batch):
            boxes_ltrb = targets[i]['boxes']
            boxes_xywh = ltrb2xywh(boxes_ltrb)
            labels = targets[i]['labels']
            if cfg.NUM_KEYPOINTS > 0:
                keypoints = targets[i]['keypoints']
            else:
                keypoints = None
            # 共享内存加速
            match4center(boxes_xywh, labels, fsize, g_center[i], cfg.NUM_KEYPOINTS, keypoints)

            if cfg.IS_VISUAL:
                # -------------------目标可视化--------------------
                import json
                # json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_widerface_proj.json'), 'r', encoding='utf-8')
                # json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_voc_proj.json'), 'r', encoding='utf-8')
                # ids2classes = json.load(json_file, encoding='utf-8')  # json key是字符
                ids2classes = {1: 'bird', 2: 'cat', 3: 'dog'}

                from torchvision.transforms import functional as transformsF
                from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
                img_ts = imgs_ts[i]
                img_ts = f_recover_normalization4ts(img_ts)
                img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
                print([ids2classes[int(label)] for label in labels])
                print('fsize:', fsize)
                # img_pil.show()

                '''plt画图部分'''
                from matplotlib import pyplot as plt
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
                plt.rcParams['axes.unicode_minus'] = False
                data_hot = torch.zeros_like(g_center[i, :, :, 0])  # 这里是0
                for label in labels.unique():
                    # print(ids2classes[str(int(label))])
                    print(ids2classes[int(label)])
                    # 类别合并输出
                    torch.max(data_hot, g_center[i, :, :, label - 1], out=data_hot)  # 这里是类别合并
                plt.imshow(data_hot.cpu())
                plt.imshow(img_pil.resize(fsize), alpha=0.7)
                plt.colorbar()
                # x,y表示横纵坐标，color表示颜色：'r':红  'b'：蓝色 等，marker:标记，edgecolors:标记边框色'r'、'g'等，s：size大小
                boxes_xywh_cpu = boxes_xywh.cpu()
                fsize_cpu = fsize.cpu()
                xys_f = boxes_xywh_cpu[:, :2] * fsize_cpu
                plt.scatter(xys_f[:, 0], xys_f[:, 1], color='r', s=5)  # 红色

                boxes_ltrb_cpu = boxes_ltrb.cpu()
                boxes_ltrb_f = boxes_ltrb_cpu * fsize_cpu.repeat(2)
                current_axis = plt.gca()
                for i, box_ltrb_f in enumerate(boxes_ltrb_f):
                    l, t, r, b = box_ltrb_f
                    # ltwh
                    current_axis.add_patch(plt.Rectangle((l, t), r - l, b - t, color='green', fill=False, linewidth=2))
                    current_axis.text(l, t - 2, ids2classes[int(labels[i])], size=8, color='white',
                                      bbox={'facecolor': 'green', 'alpha': 0.6})
                plt.show()

        gheatmap = g_center[:, :, :, :cfg.NUM_CLASSES]
        mask_pos = gheatmap.eq(1)
        # torch.Size([40, 128, 128,3 ]) -> [40, 128, 128]
        mask_pos = torch.any(mask_pos, dim=-1)  # 降维运算
        g_center_pos = g_center[mask_pos]  # [40, 128, 128,7 ] -> [58,7 ]
        gxy_offset = g_center_pos[:, -4:-2]  # 取xy
        gwh = g_center_pos[:, -2:]
        num_pos = mask_pos.sum()  # 总正例数
        num_pos.clamp_(min=torch.finfo(torch.float).eps)

        pxy_offset_pos = pxy_offset.permute(0, 2, 3, 1)[mask_pos]  # torch.Size([40, 2, 128, 128]) -> 40, 128, 128, 2
        pwh_pos = pwh.permute(0, 2, 3, 1)[mask_pos]

        if pkeypoint_offset is not None:
            gkeypoint_offset = g_center_pos[:, :, :, cfg.NUM_CLASSES:-4]
            pkeypoint_offset = pkeypoint_offset.permute(0, 2, 3, 1)[mask_pos]
            loss_keypoint = F.l1_loss(pkeypoint_offset, gkeypoint_offset, reduction='sum') / num_pos \
                            * cfg.LOSS_WEIGHT[3]
        else:
            loss_keypoint = 0

        # loss_conf = focal_loss4center2(pheatmap, gheatmap, reduction='sum') / num_pos * cfg.LOSS_WEIGHT[0]
        # loss_conf = F.binary_cross_entropy_with_logits(pheatmap, gheatmap, reduction='sum') / num_pos * cfg.LOSS_WEIGHT[0]
        # loss_conf = focal_loss4center3(pheatmap, gheatmap, reduction='sum', a=2) / num_pos * cfg.LOSS_WEIGHT[0]

        # loss_conf = self.fun_ghmc(pheatmap, gheatmap) / num_pos * cfg.LOSS_WEIGHT[0]
        loss_conf = self.fun_ghmc(pheatmap, gheatmap) * cfg.LOSS_WEIGHT[0]

        # loss_conf = self.fun_focalloss_v2(pheatmap, gheatmap) / num_pos * cfg.LOSS_WEIGHT[0]
        # loss_conf = cneg_loss(pheatmap, gheatmap) * cfg.LOSS_WEIGHT[0]
        # loss_xy = focal_loss4center(pxy_offset, gxy_offset, reduction='sum') / num_pos
        # loss_wh = focal_loss4center(pwh, gwh, reduction='sum') / num_pos
        loss_xy = F.l1_loss(pxy_offset_pos, gxy_offset, reduction='sum') / num_pos * cfg.LOSS_WEIGHT[1]
        loss_wh = F.l1_loss(pwh_pos, gwh, reduction='sum') / num_pos * cfg.LOSS_WEIGHT[2]

        '''---------------动态权重----------------'''

        loss_total = loss_conf + loss_xy + loss_wh + loss_keypoint
        log_dict = {}
        log_dict['loss_total'] = loss_total.item()
        log_dict['l_conf'] = loss_conf.item()
        log_dict['l_xy'] = loss_xy.item()
        log_dict['l_wh'] = loss_wh.item()
        if pkeypoint_offset is not None:
            log_dict['l_kp'] = loss_keypoint.item()

        log_dict['pheatmap_max'] = pheatmap.max().item()
        log_dict['pheatmap_min'] = pheatmap.min().item()
        log_dict['pheatmap_mean'] = pheatmap.mean().item()
        return loss_total, log_dict


class PredictCenter(nn.Module):
    def __init__(self, cfg, threshold_conf=0.5, threshold_nms=0.3, topk=100):
        super(PredictCenter, self).__init__()
        self.cfg = cfg
        self.threshold_nms = threshold_nms
        self.threshold_conf = threshold_conf
        self.topk = topk

    def pool_nms(self, phm, kernel=3):
        '''
        最大池化剔最大值 some 池化
        :param phm:
        :param kernel:
        :return:
        '''
        stride = 1
        pad = (kernel - stride) // 2
        hmax = nn.functional.max_pool2d(phm, (kernel, kernel), stride=stride, padding=pad)
        keep = (hmax == phm).float()  # 同维清零
        return phm * keep

    def forward(self, p_center, imgs_ts=None):
        '''

        :param p_center:
            phm : 归一化 conf b,c,h,w 这个是conf 和 label
            pwh :
            pxy_offset
        :param imgs_ts:
        :return:
        '''
        cfg = self.cfg
        pheatmap, pwh, pxy_offset, pkeypoint_offset = p_center
        pheatmap_nms = self.pool_nms(pheatmap)  # 每层池化 这个包含conf 和label
        # b,c,h,w -> b,h,w,c
        # pheatmap_nms = pheatmap_nms.permute(0, 2, 3, 1)

        k = 70
        device = pheatmap.device
        batch, c, h, w = pheatmap_nms.shape
        nums_grid = torch.tensor([h, w], device=device)

        # 拿到每批 最大的 70 个 (30,70)
        topk_val_b, topk_ids_b = torch.topk(pheatmap_nms.view(batch, -1), k)

        '''----------debug 查看top70 分布-----------'''
        if cfg.IS_TRAIN_DEBUG:
            num_bins = 10
            inds_edges = torch.floor(topk_val_b * (num_bins - 0.0001)).long()
            nums_edges = torch.zeros(num_bins)  # 区间样本数量
            for i in range(num_bins):
                nums_edges[i] = (inds_edges == i).sum().item()
            print('batchsize:%s nums_edges:%s' % (batch, nums_edges))

        _t = h * w  # 16384
        offset = topk_ids_b % _t  # 每个类型偏移id
        rows = (torch.true_divide(offset, w)).int()  # 得到row
        cols = (offset % w).int()  # 得到
        # cls_dis = torch.true_divide(topk_ids_b, _t).int()  # ids属于哪个类型 总索引 /每类的索引最大值
        cls_dis = torch.floor_divide(topk_ids_b, _t)  # ids属于哪个类型 总索引 /每类的索引最大值

        pheatmap_nms = pheatmap_nms.permute(0, 2, 3, 1)
        pwh = pwh.permute(0, 2, 3, 1)
        pxy_offset = pxy_offset.permute(0, 2, 3, 1)

        p_boxes_xywh = torch.empty((0, 4), device=device, dtype=torch.float)
        ids_batch1, p_labels1, p_scores1 = [], [], []
        for b in range(batch):
            for row, col, cls in zip(rows[b], cols[b], cls_dis[b]):
                ids_batch1.append(b)
                p_labels1.append(cls + 1)
                p_scores1.append(pheatmap_nms[b, row, col, cls])
                offset_xy = pxy_offset[b, row, col]
                colrow_index = torch.tensor([col, row], device=device)
                xy = offxy2xy(offset_xy, colrow_index, nums_grid)
                wh = pwh[b, row, col]
                p_boxes_xywh = torch.cat([p_boxes_xywh, torch.cat([xy, wh]).unsqueeze(0)], dim=0)

        p_boxes_ltrb1 = xywh2ltrb(p_boxes_xywh)
        ids_batch1 = torch.tensor(ids_batch1, dtype=torch.float, device=device)
        p_labels1 = torch.tensor(p_labels1, dtype=torch.float, device=device)
        p_scores1 = torch.tensor(p_scores1, dtype=torch.float, device=device)
        ids_batch1 = torch.tensor(ids_batch1, dtype=torch.float, device=device)

        p_keypoints1 = None

        # 分数过滤
        mask = p_scores1 > cfg.THRESHOLD_PREDICT_CONF
        ids_batch1 = ids_batch1[mask]
        p_labels1 = p_labels1[mask]
        p_scores1 = p_scores1[mask]
        p_boxes_ltrb1 = p_boxes_ltrb1[mask]

        ids_batch2, p_boxes_ltrb2, p_keypoints2, p_labels2, p_scores2 = label_nms4keypoints(
            ids_batch1,
            p_boxes_ltrb1,
            p_keypoints1,
            p_labels1,
            p_scores1,
            device,
            self.threshold_nms,
        )

        # _res = ids_batch1, p_boxes_ltrb1, p_keypoints1, p_labels1, p_scores1
        # ids_batch2, p_boxes_ltrb2, p_keypoints2, p_labels2, p_scores2 = _res

        # if self.cfg.IS_VISUAL:
        #     # 可视化1 显示框框 原目标图 --- 初始化图片
        #     flog.debug('conf后 %s 个', p_boxes_ltrb1.shape[0])
        #     img_ts = imgs_ts[0]
        #     from torchvision.transforms import functional as transformsF
        #     img_ts = f_recover_normalization4ts(img_ts)
        #     img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
        #     show_anc4pil(img_pil, p_boxes_ltrb1, size=img_pil.size)

        return ids_batch2, p_boxes_ltrb2, p_keypoints2, p_labels2, p_scores2


class CenterHead(nn.Module):
    def __init__(self, num_classes, in_channels=64, out_channels=64, bn_momentum=0.1, num_keypoints=0):
        '''

        :param num_classes:
        :param out_channels:
        :param bn_momentum:
        '''
        super(CenterHead, self).__init__()
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding=0))
        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding=0))
        if num_keypoints > 0:
            self.keypoint_head = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, num_keypoints * 2, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        phm = self.cls_head(x).sigmoid()
        # phm = self.cls_head(x).softmax(dim=1)
        pwh = self.wh_head(x).sigmoid()
        pxy_offset = self.reg_head(x).sigmoid()
        # pwh = self.wh_head(x)
        # pxy_offset = self.reg_head(x)
        if hasattr(self, 'keypoint_head'):
            pkeypoint_offset = self.keypoint_head(x).sigmoid()
        else:
            pkeypoint_offset = None
        return phm, pwh, pxy_offset, pkeypoint_offset


class CenterUpsample3Conv(nn.Module):
    def __init__(self, dim_in_backbone, bn_momentum=0.1):
        super(CenterUpsample3Conv, self).__init__()
        self.bn_momentum = bn_momentum
        # [256, 512, 1024]
        # self.dim_in_backbone = dim_in_backbone
        self.deconv_layers = self._make_deconv_layer(dim_in_backbone)

    def _make_deconv_layer(self, dim_in_backbone, nums_dim_out=[256, 128, 64], nums_kernel=[4, 4, 4]):
        '''
        反卷积
        :param nums_dim_out:
        :param nums_kernel: 对应核数
        :return:
        '''
        layers = []
        # 16,16,2048 -> 32,32,256
        # 32,32,256 -> 64,64,128
        # 64,64,128 -> 128,128,64

        for dim_out, kernel in zip(nums_dim_out, nums_kernel):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in_backbone,
                    out_channels=dim_out,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(dim_out, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            dim_in_backbone = dim_out
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class CenterNet(nn.Module):
    def __init__(self, cfg, backbone, num_classes, dim_in_backbone):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        # [256, 512, 1024]
        self.upsample3conv = CenterUpsample3Conv(dim_in_backbone)
        self.head = CenterHead(num_classes=num_classes, num_keypoints=cfg.NUM_KEYPOINTS)
        self.preder = PredictCenter(cfg, threshold_conf=cfg.THRESHOLD_PREDICT_CONF)
        self.losser = LoosCenter(cfg)
        finit_weights(self)

    def forward(self, x, targets=None):
        # torch.Size([1, 1280, 13, 13])
        outs = self.backbone(x)
        outs = self.upsample3conv(outs)  # torch.Size([1, 64, 104, 104])
        outs = self.head(outs)
        # 热力图预测 orch.Size([1, 20, 104, 104])
        # 中心点预测 torch.Size([1, 2, 104, 104])
        # 宽高预测 torch.Size([1, 2, 104, 104])
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores


if __name__ == '__main__':
    from torchvision import models
    from f_pytorch.tools_model.model_look import f_look_model

    num_classes = 20
    # model = models.mobilenet_v2(pretrained=True)
    # model = ModelOuts4Mobilenet_v2(model)
    # dims_out = [256, 512, 1024]

    model = models.mobilenet_v2(pretrained=True)
    model = ModelOut4Mobilenet_v2(model)

    model = CenterNet(cfg=None, backbone=model, num_classes=num_classes, dim_in_backbone=model.dim_out)
    model.eval()
    f_look_model(model, input=(1, 3, 416, 416))
