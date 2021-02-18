from abc import abstractmethod

import torch
import torch.nn as nn

from f_tools.GLOBAL_LOG import flog
from f_tools.floss.f_lossfun import loss_cre_pdata_t1, loss_draw_res, x_bce


class GHM_Loss_Base(nn.Module):

    def __init__(self, num_bins, momentum, reduction='none'):
        super().__init__()
        self.num_bins = num_bins
        self.reduction = reduction
        self.momentum = momentum  # 权重放大的作用 0~1 起放大作用 超过1变负值
        self.nums_bins_last = None

    @abstractmethod
    def i_calc_gradient_length(self, pconf, gconf):
        raise NotImplementedError

    @abstractmethod
    def i_loss_fun(self, pconf, gconf, weight):
        raise NotImplementedError

    def forward(self, pconf, gconf, mask_calc=None, is_debug=False):
        '''
        区间数量越多, 损失值权重越小
        :param pconf:
        :param gconf:
        :param mask_calc: 排除忽略后 需要计算的mask
        :return:
        '''
        float_eps = torch.finfo(torch.float16).eps
        device = pconf.device
        if mask_calc is None:  # 默认全部加入运算
            mask_calc = torch.ones_like(pconf, dtype=torch.bool, device=device)
        num_calc = mask_calc.sum().item()  # 总样本
        g = self.i_calc_gradient_length(pconf, gconf)  # 梯度模长 越大越难 恰好也是反向的梯度

        # 与pconf同维 区间索引
        inds_bins = torch.floor(g * (self.num_bins - float_eps)).long().to(device)  # 数据对应在哪个格子中 格子index
        # 统计各区间样本数量
        nums_bins = torch.zeros(self.num_bins, device=device)
        for i in range(self.num_bins):
            # 样本区间数    no 区间占比---(倒数相加=1) 总样本/样本数
            _num_in_bins = (torch.logical_and(inds_bins == i, mask_calc)).sum().item()
            nums_bins[i] = _num_in_bins
            # if _num_in_bins > 0:
            #     nums_bins[i] = num_calc / _num_in_bins

        if self.nums_bins_last is None:
            self.nums_bins_last = nums_bins
        else:  # 前动量引入
            nums_bins = self.momentum * self.nums_bins_last + (1 - self.momentum) * nums_bins
            self.nums_bins_last = nums_bins

        # 有效区间个安徽
        num_bins_valid = (nums_bins > 0).sum().item()
        nums_bins = nums_bins / num_bins_valid
        # weight_bins = num_calc / (nums_bins * num_bins_valid)
        # weight_bins[torch.isinf(weight_bins)] = 0
        # weight_bins = weight_bins / min(num_calc, 100)

        if is_debug:
            flog.debug('GHM_Loss 梯度模长区间数量:%s', [round(d.item(), 2) for d in nums_bins])  # 区间数量
            flog.debug('GHM_Loss 区间权重:%s', [round(d.item(), 2) for d in weight_bins])  # 权重
            # show_distribution(pconf)  # 在 pconf 分布
            pass

        loss = self.i_loss_fun(pconf, gconf, weight_bins[inds_bins])
        return loss


class GHMC_Loss(GHM_Loss_Base):

    def __init__(self, num_bins=10, momentum=0., reduction='none'):
        super(GHMC_Loss, self).__init__(num_bins=num_bins, momentum=momentum, reduction=reduction)

    def i_calc_gradient_length(self, pconf, gconf):
        return torch.abs(pconf.detach() - gconf)

    def i_loss_fun(self, pconf, gconf, weight):
        # return F.binary_cross_entropy(pconf, gconf, weight=weight, reduction='none')
        return x_bce(pconf, gconf, weight=weight)


class GHMR_Loss(GHM_Loss_Base):
    def __init__(self, mu, num_bins=10, momentum=0., reduction='none'):
        super(GHMR_Loss, self).__init__(num_bins=num_bins, momentum=momentum, reduction=reduction)
        self.mu = mu

    def i_calc_gradient_length(self, pconf, gconf):
        d = pconf - gconf
        mu = self.mu
        return d / torch.sqrt(d * d + mu * mu)

    def i_loss_fun(self, pconf, gconf, weight):
        d = pconf - gconf
        mu = self.mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        return loss * weight


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


# TODO: code refactoring to make it consistent with other losses
class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, mask_calc=None, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            mask_calc (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        device = pred.device
        if mask_calc is None:  # 默认全部加入运算
            mask_calc = torch.ones_like(pred, dtype=torch.bool, device=device)

        # # the target should be binary class label
        # if pred.dim() != target.dim():
        #     target, mask_calc = _expand_binary_labels(
        #         target, mask_calc, pred.size(-1))
        target, mask_calc = target.float(), mask_calc.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        # g = torch.abs(pred.sigmoid().detach() - target)
        g = torch.abs(pred.detach() - target)

        valid = mask_calc > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        # loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / tot
        loss = x_bce(pred, target, weights) / tot
        return loss * self.loss_weight


# TODO: code refactoring to make it consistent with other losses
class GHMR(nn.Module):
    """GHM Regression Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector"
    https://arxiv.org/abs/1811.05181

    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """

    def __init__(self, mu=0.02, bins=10, momentum=0, loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight

    # TODO: support reduction parameter
    def forward(self, pred, target, label_weight, avg_factor=None):
        """Calculate the GHM-R loss.

        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight


def t_损失值():
    p_cls = loss_cre_pdata_t1()
    yy = torch.arange(0, 1, 0.1)

    labels = []
    labels.append('ghmc_obj')
    x1 = []
    y1 = []
    for i, y in enumerate(yy):
        g_cls = torch.full_like(p_cls, y)
        # out = ghmc_obj(p_cls, g_cls, is_debug=True)
        # out = ghmc_obj(p_cls, g_cls)
        out = ghmc_obj(p_cls, g_cls)
        x1.append(y.item())
        _val = out.sum().item()
        y1.append(_val)
        print('ghmc_obj', _val)
    print(y1)
    # _draw_res([x1, x2, x3], [y1, y2, y3], labels=labels, is_xylim=False)
    loss_draw_res([x1], [y1], labels=labels[0], is_xylim=False)


def t_图形():
    xs = []
    ys = []
    labels = []
    p_cls = torch.arange(0, 1, 0.01)
    y = 0.7
    g_cls = torch.full_like(p_cls, y)
    out = ghmc_obj(p_cls, g_cls, is_debug=True)
    xs.append(p_cls)
    ys.append(out)
    labels.append('ghmc_obj')
    loss_draw_res(xs, ys, labels=labels, title=y)


if __name__ == '__main__':
    ghmc_obj = GHMC()

    t_损失值()

    # t_图形()
