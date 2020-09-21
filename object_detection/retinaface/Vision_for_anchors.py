import numpy as np
from math import ceil
from itertools import product as product
from object_detection.retinaface.utils.config import cfg_mnet
import matplotlib.pyplot as plt
import torch


def decode(loc, priors, variances):
    '''
    中心调整 x,y,w,h
    :param loc: 回归偏差
    :param priors: 先验
    :param variances: [0.1, 0.2] 用来作归一化?
    :return:
    '''
    boxes = torch.cat(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],  # 中心调整
         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])),  # 宽高调整
        dim=1)

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    '''
    关键点解码
    :param pre:回归偏差
    :param priors:先验
    :param variances:
    :return:
    '''
    # 对应系数 相乘
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


class Anchors(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 每个网格点2个先验框，都是正方形
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        anchors = np.reshape(anchors, [-1, 4])
        # anchors = anchors[:2] * 224

        # xl,yl,xr,yr 偏移比例 转 w,h,x,y
        output = np.zeros_like(anchors[:, :4])
        output[:, 0] = anchors[:, 0] - anchors[:, 2] / 2  # w
        output[:, 1] = anchors[:, 1] - anchors[:, 3] / 2  # h
        output[:, 2] = anchors[:, 0] + anchors[:, 2] / 2  # x
        output[:, 3] = anchors[:, 1] + anchors[:, 3] / 2  # y

        if self.clip:
            output = np.clip(output, 0, 1)
        return output


if __name__ == '__main__':

    img_dim = cfg_mnet['image_size']
    # img_dim = 224
    # 求出来的是每一个anchors对应原图的偏移比例
    anchors = Anchors(cfg_mnet, image_size=(img_dim, img_dim)).get_anchors()
    anchors = anchors[-800:] * img_dim  # 取后800个 得到真实选框

    # 中心坐标需要重新计算
    center_x = (anchors[:, 0] + anchors[:, 2]) / 2
    center_y = (anchors[:, 1] + anchors[:, 3]) / 2
    # center_x = anchors[:, 2]
    # center_y = anchors[:, 3]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    ax.invert_yaxis()  # y轴反向

    plt.scatter(center_x, center_y)

    box_widths = anchors[0:2, 2] - anchors[0:2, 0]
    box_heights = anchors[0:2, 3] - anchors[0:2, 1]

    for i in [0, 1]:
        rect = plt.Rectangle([anchors[i, 0], anchors[i, 1]], box_widths[i], box_heights[i], color="r", fill=False)
        ax.add_patch(rect)

    # ---------------------------调整---------------------------------
    ax = fig.add_subplot(122)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    ax.invert_yaxis()  # y轴反向

    plt.scatter(center_x, center_y)

    mbox_loc = np.random.randn(800, 4)
    mbox_ldm = np.random.randn(800, 10)

    anchors[:, :2] = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors[:, 2:] = (anchors[:, 2:] - anchors[:, :2]) * 2

    mbox_loc = torch.Tensor(mbox_loc)
    anchors = torch.Tensor(anchors)
    cfg_mnet['variance'] = torch.Tensor(cfg_mnet['variance'])
    decode_bbox = decode(mbox_loc, anchors, cfg_mnet['variance'])

    box_widths = decode_bbox[0:2, 2] - decode_bbox[0:2, 0]
    box_heights = decode_bbox[0:2, 3] - decode_bbox[0:2, 1]

    for i in [0, 1]:
        rect = plt.Rectangle([decode_bbox[i, 0], decode_bbox[i, 1]], box_widths[i], box_heights[i], color="r",
                             fill=False)
        plt.scatter((decode_bbox[i, 2] + decode_bbox[i, 0]) / 2, (decode_bbox[i, 3] + decode_bbox[i, 1]) / 2, color="b")
        ax.add_patch(rect)

    plt.show()
