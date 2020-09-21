import torch
import torch.nn as nn
import torch.nn.functional as F
from object_detection.retinaface.utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, neg_pos, overlap_thresh):
        '''

        :param num_classes: ANCHOR_NUM
        :param overlap_thresh: 重合程度在小于0.35的类别设为0
        :param neg_pos: 正负样本的比率
        '''
        super(MultiBoxLoss, self).__init__()
        # 对于retinaface而言num_classes等于2
        self.num_classes = num_classes
        # 重合程度在多少以上认为该先验框可以用来预测
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]

    def forward(self, predictions, anchors, targets, device):
        '''

        :param predictions: 预测box 调整系数
            预测结果tuple(torch.Size([8, 16800, 4]),torch.Size([8, 16800, 2]),torch.Size([8, 16800, 10]))
        :param anchors: default boxes
        :param targets: 标签 ground truth boxes   n,15(4+10+1)
        :return:
        '''
        loc_data, conf_data, landm_data = predictions
        num_batch = loc_data.size(0)
        num_priors = (anchors.size(0))

        # 创建空的tensor
        loc_t = torch.Tensor(num_batch, num_priors, 4)
        landm_t = torch.Tensor(num_batch, num_priors, 10)
        conf_t = torch.LongTensor(num_batch, num_priors)  # 装类别
        for idx in range(num_batch):  # 遍历每一张图的 标签 GT
            truths = targets[idx][:, :4].data  # 假如有11个框 标签 标签 标签
            labels = targets[idx][:, -1].data  # 真实标签
            landms = targets[idx][:, 4:14].data  # 真实5点定位
            defaults = anchors.data  # 复制一个
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        zeros = torch.tensor(0, device=device)
        loc_t = loc_t.to(device)
        conf_t = conf_t.to(device)
        landm_t = landm_t.to(device)

        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros  # 做布尔索引
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)  # 先在后面加1维,扩展成索引
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # conf_t全部置1
        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num_batch, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm

