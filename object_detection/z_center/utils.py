import torch
from torch.nn import functional as F


def gaussian_radius(sizes, min_overlap=0.7):
    '''
    根据长宽 尽量保存能 IOU>0.7 求半径
    :param sizes : bbox torch.Size([nn, 2])
    :param min_overlap: 覆盖面积 1为最大 这个越小半径越大
    :return: 最小的半径，其保证iou>=min_overlap
        [nn] 个最小半径 tensor([1.3666, 0.4733, 0.6581])
    '''
    hh, ww = sizes[:, 0], sizes[:, 1]

    a1 = 1
    b1 = (hh + ww)
    c1 = ww * hh * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (hh + ww)
    c2 = (1 - min_overlap) * ww * hh
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (hh + ww)
    c3 = (min_overlap - 1) * ww * hh
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    val_min, _ = torch.min(torch.cat([r1.unsqueeze(0), r2.unsqueeze(0), r3.unsqueeze(0)], dim=0).T, dim=1)
    return val_min


def cre_grid(rwh, sigma=torch.tensor((1, 2))):
    '''
    根据半径生成
    :param rwh: 这里需转成int 确保中心点为1
    :param sigma:
    :return:
    '''
    w, h = rwh

    # ww = torch.arange(-w, w + 1).true_divide(2 * sigma[0] ** 2).unsqueeze(0)
    # hh = torch.arange(-h, h + 1).true_divide(2 * sigma[1] ** 2).unsqueeze(-1)
    ww = torch.arange(-w, w + 1, device=rwh.device).unsqueeze(0)
    hh = torch.arange(-h, h + 1, device=rwh.device).unsqueeze(-1)

    h = torch.exp(-((ww * ww).true_divide(2 * sigma[1] ** 2) + (hh * hh).true_divide(2 * sigma[1] ** 2)))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, xy, radius_wh, k=1, ):
    '''

    :param heatmap: 原热图
    :param xy: 一个 tensor([30.02572, 30.17100]) for 特图
    :param radius_wh: 一个ts float 半径
    :param k: k=1 核心值为1
    :return:
    '''

    # 半径 -> 直径
    diameter_wh = 2 * radius_wh + 1
    diameter_wh = diameter_wh.type(torch.int32)
    gaussian = cre_grid(diameter_wh, sigma=diameter_wh.true_divide(6.))

    x, y = int(xy[0]), int(xy[1])
    # fsize
    height, width = heatmap.shape[0:2]

    # 这个是目标高期核的框坐标
    left, right = min(x, diameter_wh[0]), min(width - x, diameter_wh[0] + 1)
    top, bottom = min(y, diameter_wh[1]), min(height - y, diameter_wh[1] + 1)
    #  原图的对应数据取出来 x y 反序
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # 超出部份不要
    masked_gaussian = gaussian[diameter_wh[1] - top:diameter_wh[1] + bottom,
                      diameter_wh[0] - left:diameter_wh[0] + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    # heatmap是共享的
    # return heatmap


if __name__ == '__main__':
    sizes_ts = torch.tensor([[5, 5], [5, 1], [3, 2]])
    radiuses = gaussian_radius(sizes_ts, min_overlap=0.7)
    radiuses.clamp_(min=0)
    print(radiuses)

    outs2 = cre_grid(torch.tensor((4, 6)), sigma=torch.tensor((1, 1)))
    print(outs2, )
