from f_tools.fun_od.f_boxes import ltrb2xywh, xywh2ltrb
from f_tools.yufa.x_calc_adv import f_mershgrid
import itertools
from math import sqrt

import torch
import matplotlib.pyplot as plt
import numpy as np


class FAnchors:
    def __init__(self, img_in_size, anc_scale, feature_size_steps=None, feature_sizes=None, anchors_clip=True,
                 is_xymid=False, is_real_size=False, device=None):
        '''
        用于动态anc
        :param img_in_size:
        :param anc_scale:
            ANCHORS_SIZE = [
                [[0.13, 0.10666667], [0.06, 0.15733333], [0.036, 0.06006006], ],
                [[0.196, 0.51466667], [0.29, 0.28], [0.12, 0.28], ],
                [[0.81786211, 0.872], [0.374, 0.72266667], [0.612, 0.452], ],
            ]
        :param feature_size_steps: 可以输入下采样 算尺寸  也可以直接输入尺寸
        :param anchors_clip:
        :param is_xymid:
        :param is_real_size:
        '''
        self.anchors_clip = anchors_clip  # 是否剔除超边界---超参
        self.ancs_scale = anc_scale
        self.is_real_size = is_real_size  # 以 anchors_size 的真实尺寸输出 非归一化
        self.is_xymid = is_xymid  # 是否将anc中心点移到格子中间
        # 特征层对应的步距   [8, 16, 32] 原图size/特图size = images/feature_maps = steps
        self.feature_size_steps = feature_size_steps  # 这个定义特图下采样比例

        self.img_in_size = img_in_size
        # 根据预处理后的尺寸及步距 计算每一个特图的尺寸
        from math import ceil
        # 根据预处理图片及下采倍数，计算特图尺寸 <class 'list'>: [[52, 52], [26, 26], [13, 13]] (w,h)
        if feature_sizes is not None:
            self.feature_sizes = np.array(feature_sizes)[:, None].repeat(2, axis=1)
        else:
            self.feature_sizes = [[ceil(self.img_in_size[0] / step), ceil(self.img_in_size[1] / step)]
                                  for step in self.feature_size_steps]
        self.nums_level = []  # 每层个数
        self.nums_anc = []  # 每层anc数
        self.device = device
        self.ancs_xywh = self.cre_anchors()

        self.match_grids = None
        self.ancs_xywh_t = None

    def cre_anchors(self):
        '''
        形成的顺序是 每一个特图->每个格子(行优先) 建议从小到大 需与模型匹配 ->每一个尺寸
        :return:
            返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框
            这里是 x,y,w,h 调整系数
        '''
        device = self.device
        ret_ancs = torch.empty((0, 4), device=device)
        for i, (row, col) in enumerate(self.feature_sizes):
            # 求xy 中心点坐标 行列与xy是反的
            rowcol_index = f_mershgrid(row, col, is_rowcol=True, num_repeat=1)
            rowcol_index = rowcol_index.to(device)
            # 这个特图的数量
            num_anc_one = rowcol_index.shape[0]  # 2704
            # 这个特图的尺寸 [[0.025, 0.025], [0.05, 0.05]]
            ancs_wh = self.ancs_scale[i]
            # 单体复制 每一个格子有
            num_anc = len(ancs_wh)
            self.nums_anc.append(num_anc)
            self.nums_level.append(num_anc_one * num_anc)  # self.nums_feature 是全新
            rowcol_index = rowcol_index.repeat_interleave(num_anc, dim=0)  # 单体复制
            rowcol = torch.tensor((row, col), device=device)
            # xy 与 rowcol相反
            ancs_yx = torch.true_divide(rowcol_index, rowcol)  # 特图归一化
            if self.is_xymid:  # 移动中心点到格子中间
                midyx = torch.true_divide(1, rowcol) / 2
                ancs_yx = ancs_yx + midyx
            ancs_wh = torch.tensor(ancs_wh, device=device).repeat(num_anc_one, 1)
            ancs_xy = ancs_yx[:, [1, 0]]
            ans_xywh = torch.cat([ancs_xy, ancs_wh], dim=-1)
            ret_ancs = torch.cat([ret_ancs, ans_xywh], dim=0)
        if self.anchors_clip:  # 对于xywh 来说这个参数 是没有用的
            ret_ancs = xywh2ltrb(ret_ancs)
            ret_ancs.clamp_(min=0, max=1)  # 去除超边际的
            ret_ancs = ltrb2xywh(ret_ancs)
        if self.is_real_size:
            ret_ancs = ret_ancs * torch.tensor(self.img_in_size)[None].repeat(1, 2)
        # __d = 1
        if self.device is not None:
            ret_ancs = ret_ancs.to(self.device)
        return ret_ancs

    def match_anc_grids(self):
        # anc 归一化 -> anc特图
        if self.match_grids is not None:
            return self.match_grids, self.ancs_xywh_t

        nums_ceng_np = np.array(self.nums_level, dtype=np.int32)
        nums_anc_np = np.array(self.nums_anc, dtype=np.float32)
        grids_np = np.sqrt((nums_ceng_np / nums_anc_np))  # [ 52 26 13 7 4]
        match_grids_ts = torch.tensor(np.repeat(grids_np, nums_ceng_np, axis=-1), device=self.ancs_xywh.device,
                                      dtype=torch.float)
        match_grids_ts = match_grids_ts.view(-1, 1)  # torch.Size([32526, 1]) -> [32526, 4]
        self.match_grids = match_grids_ts
        self.ancs_xywh_t = self.ancs_xywh * match_grids_ts
        return self.match_grids, self.ancs_xywh_t


class DefaultBoxes:
    def __init__(self, img_size, feature_size, steps, wh, aspect_ratios, scale_xy=0.1, scale_wh=0.2,
                 device=torch.device('cpu')):
        '''

        :param img_size:  输入图片的尺寸
        :param feature_size: 特图尺寸 [38, 19, 10, 5, 3, 1]
        :param steps: 输入图片的下采样比例 向上取整得 特图尺寸
        :param wh: 每个框的真实尺寸
        :param aspect_ratios: anc比例 2 表示 2:1   [2,3]表示2:1 3:1 两个  1比1 是默认就有
        :param scale_xy:
        :param scale_wh:
        '''
        self.img_size = img_size  # 输入网络的图像大小 300
        # [38, 19, 10, 5, 3, 1]
        self.feature_size = feature_size  # 每个预测层的feature map尺寸

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        # [8, 16, 32, 64, 100, 300]
        self.steps = steps  # 每个特征层上的一个cell在原图上的跨度

        # [21, 45, 99, 153, 207, 261, 315]
        self.wh = wh  # 每个特征层上预测的default box的scale

        fk = img_size / np.array(steps)  # 计算每层特征层的  正常应该=feature_size
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios  # 每个预测特征层上预测的default box的ratios

        self.default_boxes = []
        # size of feature and number of feature
        # 遍历每层特征层，计算default box
        for idx, sfeat in enumerate(self.feature_size):
            sk1 = wh[idx] / img_size  # scale转为相对值[0-1]
            sk2 = wh[idx + 1] / img_size  # scale转为相对值[0-1]
            sk3 = sqrt(sk1 * sk2)
            # 先添加两个1:1比例的default box宽和高
            all_sizes = [(sk1, sk1), (sk3, sk3)]  # 1:1的默认添加

            # 再将剩下不同比例的default box宽和高添加到all_sizes中
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # 计算当前特征层对应原图上的所有default box
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):  # i -> 行（y）， j -> 列（x）
                    # 计算每个default box的中心坐标（范围是在0-1之间）
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        # 将default_boxes转为tensor格式
        self.ancs_xywh = torch.as_tensor(self.default_boxes, dtype=torch.float32, device=device)  # 这里不转类型会报错
        self.ancs_xywh.clamp_(min=0, max=1)  # 将坐标（x, y, w, h）都限制在0-1之间

        # For IoU calculation
        # ltrb is left top coordinate and right bottom coordinate
        # 将(x, y, w, h)转换成(xmin, ymin, xmax, ymax)，方便后续计算IoU(匹配正负样本时)
        self.dboxes_ltrb = self.ancs_xywh.clone()
        self.dboxes_ltrb[:, 0] = self.ancs_xywh[:, 0] - 0.5 * self.ancs_xywh[:, 2]  # xmin
        self.dboxes_ltrb[:, 1] = self.ancs_xywh[:, 1] - 0.5 * self.ancs_xywh[:, 3]  # ymin
        self.dboxes_ltrb[:, 2] = self.ancs_xywh[:, 0] + 0.5 * self.ancs_xywh[:, 2]  # xmax
        self.dboxes_ltrb[:, 3] = self.ancs_xywh[:, 1] + 0.5 * self.ancs_xywh[:, 3]  # ymax

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order='ltrb'):
        # 根据需求返回对应格式的default box
        if order == 'ltrb':
            return self.dboxes_ltrb

        if order == 'xywh':
            return self.ancs_xywh


def cre_ssd_ancs(device=torch.device('cpu')):
    img_size = 300  # 输入网络的图像大小
    feature_size = [38, 19, 10, 5, 3, 1]  # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]  # 每个特征层上的一个cell在原图上的跨度
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]  # 每个特征层上预测的default box的scale
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # 每个预测特征层上预测的default box的ratios

    # fig_size = 4
    # feat_size = [2]
    # steps = [1]
    #
    # # 尺寸为2的 有2比1 3比1种
    # scales = [2, 1]  # 长宽比
    # # 2比1 3比1
    # aspect_ratios = [[2, 3]]

    dboxes = DefaultBoxes(img_size, feature_size, steps, scales, aspect_ratios, device=device)
    return dboxes


def tfanc():
    global size, len
    # __t001()
    # __t_anc4found()
    size = [640, 640]
    # anchors_size = [
    #     [[10, 13], [16, 30], [33, 23]],  # 大特图小目标 52, 52
    #     [[30, 61], [62, 45], [59, 119]],  # 26, 26
    #     [[116, 90], [156, 198], [373, 326]],  # 小特图大目标 13x13
    # ]
    feature_map_steps = [8, 16, 32]
    # anchors = Anchors(size, anchors_size, feature_map_steps,
    #                   is_xymid=False, is_real_size=True, anchors_clip=False).get_anchors()  # torch.Size([10647, 4]
    # anc_scale = [
    #     [[0.13, 0.10666667], [0.06, 0.15733333], [0.036, 0.06006006], ],
    #     [[0.196, 0.51466667], [0.29, 0.28], [0.12, 0.28], ],
    #     [[0.81786211, 0.872], [0.374, 0.72266667], [0.612, 0.452], ],
    # ]
    anc_scale = [
        [[0.025, 0.025], [0.05, 0.05]],
        [[0.1, 0.1], [0.2, 0.2], ],
        [[0.4, 0.4], [0.8, 0.8], ],
    ]
    anchors_xywh = FAnchors(size, anc_scale, feature_map_steps,
                            anchors_clip=False, is_xymid=True, is_real_size=False).cre_anchors()
    # print(anchors)
    # anchors = FAnchors(size, anc_scale, feature_map_steps,
    #                    anchors_clip=True, is_xymid=False, is_real_size=True).cre_anchors()  # torch.Size([10647, 4])
    # [[52, 52], [26, 26], [13, 13]]
    index_start = 0
    # index_start = len(anchors) - 460
    len = 6
    anchors_xywh = anchors_xywh[index_start:index_start + len]  # 这里选出 anchors
    # --------------anchors 转换画图--------------
    # __anchors = anchors.clone()
    # __anchors = xywh2ltrb(__anchors)
    # __anchors[:, ::2] = __anchors[:, ::2] * size[0]
    # __anchors[:, 1::2] = __anchors[:, 1::2] * size[1]
    # xywh --> ltwh 为了plt.Rectangle
    # __anchors[:, :2] = __anchors[:, :2] - __anchors[:, 2:] / 2.
    # img_pil_new = Image.new('RGB', list(np.array(size) * 2), (128, 128, 128))
    # img_pil_new = Image.new('RGB', size, (128, 128, 128))
    # # show_bbox4pil(img_pil_new, xywh2ltrb(__anchors))
    # show_bbox4pil(img_pil_new, xywh2ltrb(anchors_xywh))
    # 构造点
    w = [i for i in range(size[0])]
    h = [i for i in range(size[1])]
    ww, hh = np.meshgrid(w, h)
    p = 100  # 图片显示padding
    # ----------图形配置---------
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-p, size[0] + p)
    plt.xlim(-p, size[1] + p)
    ax.invert_yaxis()  # y轴反向
    plt.scatter(ww.reshape(-1), hh.reshape(-1))
    for a in anchors_xywh:  # 画矩形框
        # xy,w,h
        rect = plt.Rectangle((a[0], a[1]), a[2], a[3],
                             color="r",
                             fill=False)
        ax.add_patch(rect)
    plt.show()
    pass


if __name__ == '__main__':
    # tfanc()

    anc_obj = cre_ssd_ancs()
    anchors_xywh = anc_obj.ancs_xywh
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.invert_yaxis()
    p = 1
    plt.ylim(-p, p)
    plt.xlim(-p, p)
    for a in xywh2ltrb(anchors_xywh):  # 画矩形框
        # ltrb
        rect = plt.Rectangle((a[0], a[1]), a[2], a[3],
                             color="r",
                             fill=False)
        ax.add_patch(rect)
    plt.show()
    print()
