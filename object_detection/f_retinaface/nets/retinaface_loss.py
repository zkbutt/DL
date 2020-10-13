import torch
import torch.nn as nn
import torch.nn.functional as F
from object_detection.retinaface.utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    '''
    类别损失: 计算正反例
    定位损失: 正样本无标签
    '''
    def __init__(self, num_classes, neg_pos, overlap_thresh, variance):
        '''

        :param num_classes: ANCHOR_NUM
        :param neg_pos: 正负样本的比率
        :param overlap_thresh: 重合程度在小于0.35的类别设为0
        :param variance: 控制每一种anc的修复偏移值 [0.1,0.2]
        '''
        super(MultiBoxLoss, self).__init__()
        # 对于retinaface而言num_classes等于2
        self.num_classes = num_classes
        # 重合程度在多少以上认为该先验框可以用来预测
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = variance

    def forward(self, predictions, anchors, targets, device):
        '''

        :param predictions: 16800是各个特图的拉平
            tuple(# 预测器 框4 类别2 关键点10
                torch.Size([batch, 16800, 4]) # 框
                torch.Size([batch, 16800, 2]) # 类别
                torch.Size([batch, 16800, 10]) # 关键点
            )
        :param anchors: default boxes torch.Size([16800, 4])
        :param targets:
            list( # batch个
                target: dict{
                        image_id: int,
                        bboxs: np(num_anns, 4),
                        labels: np(num_anns),
                        keypoints: np(num_anns,10),
                    }
                )
        :param device:
        :return:
        '''
        loc_data, conf_data, landm_data = predictions
        num_batch = loc_data.shape[0]  # 批次
        num_priors = anchors.shape[0]  # 总anc数

        # 创建空的tensor  装类别与目标的差距 和 正负样本分类
        loc_t = torch.Tensor(num_batch, num_priors, 4).to(device)
        landm_t = torch.Tensor(num_batch, num_priors, 10).to(device)
        conf_t = torch.LongTensor(num_batch, num_priors).to(device)

        # 为每一个图 匹配框
        for idx in range(num_batch):  # 遍历每一张图的 标签 GT
            bboxs = targets[idx]['bboxs'].detach()  # 假如有11个框 标签 标签 标签
            labels = targets[idx]['labels'].detach()  # 真实标签
            keypoints = targets[idx]['keypoints'].detach()  # 真实5点定位
            ancs = anchors.detach()  # 不要梯度 安全的
            match(bboxs, labels, keypoints, ancs, idx, loc_t, conf_t, landm_t,
                  self.threshold, self.variance)

        _zeros = torch.tensor(0, device=device)

        # landm Loss (Smooth L1) 只计算正例
        mask_landm_pos = conf_t > _zeros  # torch.Size([batch, 16800])
        num_pos = mask_landm_pos.sum().float()
        n1 = max(num_pos, 1.)  # 确保至少一个 n和n1的区别

        # [batch, 16800] -> batch, 16800,1-> torch.Size([batch, 16800, 10])
        mask_landm_pos = mask_landm_pos.unsqueeze(mask_landm_pos.dim()).expand_as(landm_data)  # 先在后面加1维,扩展成索引
        landm_p = landm_data[mask_landm_pos].view(-1, 10)  # 预测赛选
        landm_t = landm_t[mask_landm_pos].view(-1, 10)  # 标签赛选
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # 将正例全部置1, 前面 已经置0了
        mask_pos = conf_t != _zeros
        conf_t[mask_pos] = 1

        # Localization Loss (Smooth L1)
        mask_pos_loc = mask_pos.unsqueeze(mask_pos.dim()).expand_as(loc_data)
        loc_p = loc_data[mask_pos_loc].view(-1, 4)  # 拉平batch
        loc_t = loc_t[mask_pos_loc].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # 类别交叉熵
        batch_conf = conf_data.view(-1, self.num_classes)  # 拉平batch
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[mask_pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num_batch, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = mask_pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=mask_pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = mask_pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(mask_pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        n = max(num_pos.data.sum().float(), 1)
        loss_loc /= n
        loss_c /= n
        loss_landm /= n1

        return loss_loc, loss_c, loss_landm
