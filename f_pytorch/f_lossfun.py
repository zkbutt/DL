import torch
from torch import Tensor

from f_tools.GLOBAL_LOG import flog


class MoreLabelsNumLossFun(torch.nn.Module):
    def __init__(self, a=0.5, b=.5):
        # 这里定义超参 a,b 为系数值
        super().__init__()
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)

    def calculate(self, val):
        _t = torch.pow(val, 2)
        _t = torch.sum(_t,dim=1)
        _t = torch.sqrt(_t)
        ret=torch.mean(_t)
        return ret

    def forward(self, outputs: Tensor, y):
        # 输出一行20列
        # torch.Size([8, 20]) torch.Size([8, 20])
        # flog.debug('forward %s %s', outputs, y)

        y1 = torch.ones_like(outputs)
        y1[y <= 0] = 0  # 取没有目标的值
        y2 = torch.ones_like(outputs)
        y2[y > 0] = 0  # 取没有目标的值

        l1 = (y - outputs) * y1
        l2 = (y - outputs) * y2

        loss1 = self.calculate(l1)  # 已有目标的损失
        loss2 = self.calculate(l2)  # 其它类大于0则损失

        return self.a * loss1 + self.b * loss2
