import torch
from itertools import product as product
import numpy as np
from math import ceil

from f_tools.fun_od.f_boxes import fix_anc4p


class AnchorsFound(object):
    def __init__(self, image_size, anchors_size, feature_map_steps, anchors_clip=False):
        '''
        没有长宽比就是个正方形
        :param image_size:  原图预处理后尺寸
        :param anchors_size:  框尺寸 [[16, 32], [64, 128], [256, 512]] 对应特图
        :param feature_map_steps: [8, 16, 32]  # 特图的步距
        :param anchors_clip: 是否剔除超边界
        '''
        super(AnchorsFound, self).__init__()
        # 定义对应3个特征的尺寸 [[16, 32], [64, 128], [256, 512]] 这里是框正方形
        self.anchors_size = anchors_size  # 这个定义一个特图有多少个Anchors
        # 特征层对应的步距   [8, 16, 32] 原图size/特图size = images/feature_maps = steps
        self.feature_map_steps = feature_map_steps  # 这个定义特图下采样比例
        self.anchors_clip = anchors_clip  # 是否剔除超边界---超参

        self.image_size = image_size
        # 根据预处理后的尺寸及步距 计算每一个特图的尺寸
        # feature_maps: [[80, 80], [40, 40], [20, 20]] = [640,640] / [8, 16, 32]
        # self.feature_maps = np.ceil(self.image_size[None] / self.feature_map_steps[:, None])
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
                             for step in self.feature_map_steps]

    def get_anchors(self):
        '''
        要得到最终的框还需乘原图对应的尺寸
        :return:
            返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数
        '''
        anchors = []
        # 为每一个特图 生成
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.anchors_size[k]  # 取出对应 Anchors 的尺寸数组
            # 每个网格点2个先验框，都是正方形
            for i, j in product(range(f[0]), range(f[1])):  # 取每一个特图的每一个点,多重循环
                # i是行,j是列
                for min_size in min_sizes:  # Anchor尺寸组遍历 [[16, 32], [64, 128], [256, 512]]
                    s_kx = min_size / self.image_size[1]  # 框长度(长宽一样是一个值) / 特图
                    s_ky = min_size / self.image_size[0]  # 得到长度的换算比例
                    # self.steps = [8, 16, 32]
                    # 加0.5是将中间点从左上角调整到格子的中间
                    dense_cx = [x * self.feature_map_steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.feature_map_steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]  # x,y,w,h 特图的每一个点对应原图的比例
            # 最终形成的 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 调整系数
            # 需乘个size才能得到具体的框

        output = torch.Tensor(anchors).view(-1, 4)
        if self.anchors_clip:
            output.clamp_(max=1, min=0)  # 去除超边际的
        return output


if __name__ == '__main__':
    size = [320, 320]
    anchors_size = [[16, 32], [64, 128], [256, 512]]
    feature_map_steps = [8, 16, 32]
    clip = False

    anchors = AnchorsFound(size, anchors_size, feature_map_steps, clip).get_anchors()  # torch.Size([16800, 4])
    index_start = int(anchors.shape[0] / 2)
    # index_start = 0
    anchors = anchors[index_start:index_start + 2]  # 这里选出 anchors

    # --------------anchors 转换画图--------------
    __anchors = anchors.clone()
    __anchors[:, [0, 2]] = __anchors[:, [0, 2]] * size[0]
    __anchors[:, [1, 3]] = __anchors[:, [1, 3]] * size[1]

    # 中心点 --> 左上角 为了plt.Rectangle
    __anchors[:, :2] = __anchors[:, :2] - __anchors[:, 2:] / 2.

    import matplotlib.pyplot as plt

    # 构造点
    w = [i for i in range(size[0])]
    h = [i for i in range(size[1])]
    ww, hh = np.meshgrid(w, h)

    p = 0  # 图片显示padding
    # ----------图形配置---------
    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.ylim(-p, size[0] + p)
    plt.xlim(-p, size[1] + p)
    ax.invert_yaxis()  # y轴反向

    plt.scatter(ww.reshape(-1), hh.reshape(-1))
    for a in __anchors:  # 画矩形框
        rect = plt.Rectangle((a[0], a[1]), a[2], a[3], color="r", fill=False)
        ax.add_patch(rect)

    # -------------下面是画修正点------------
    mbox_loc = torch.rand(2, 4)
    mbox_ldm = torch.rand(2, 10)
    # xs=[0.1, 0.2]

    '''核心 box 修复'''
    fix_anc4p(anchors, mbox_loc, (1, 1))

    # --------------anchors 转换画图--------------
    __anchors = anchors.clone()
    __anchors[:, [0, 2]] = __anchors[:, [0, 2]] * size[0]
    __anchors[:, [1, 3]] = __anchors[:, [1, 3]] * size[1]

    # 中心点 --> 左上角 为了plt.Rectangle
    __anchors[:, :2] = __anchors[:, :2] - __anchors[:, 2:] / 2.

    # ----------图形配置---------
    ax = fig.add_subplot(122)
    plt.ylim(-p, size[0] + p)
    plt.xlim(-p, size[1] + p)
    ax.invert_yaxis()  # y轴反向

    for i, a in enumerate(__anchors):  # 画矩形框和中点
        rect = plt.Rectangle((a[0], a[1]), a[2], a[3], color="r", fill=False)
        plt.scatter(anchors[i, 0] * size[0], anchors[i, 1] * size[1], color="b")
        ax.add_patch(rect)

    plt.show()
