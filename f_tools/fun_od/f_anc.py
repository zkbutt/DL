import torch
from PIL import Image

from f_tools.f_od_gen import f_get_rowcol_index
from f_tools.fun_od.f_boxes import ltrb2xywh, xywh2ltrb
from f_tools.pic.f_show import show_bbox4pil


class FAnchors:
    def __init__(self, img_in_size, anc_scale, feature_size_steps, anchors_clip=True,
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
        :param feature_size_steps:
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
        self.feature_sizes = [[ceil(self.img_in_size[0] / step), ceil(self.img_in_size[1] / step)]
                              for step in self.feature_size_steps]
        self.device = device
        self.ancs = self.cre_anchors()

    def cre_anchors(self):
        '''
        形成的顺序是 每一个特图->每个格子(行优先) 建议从小到大 需与模型匹配 ->每一个尺寸
        :return:
            返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框
            这里是 x,y,w,h 调整系数
        '''
        ret_ancs = torch.empty(0, 4)
        for i, (row, col) in enumerate(self.feature_sizes):
            # 求xy 中心点坐标 行列与xy是反的
            rowcol_index = f_get_rowcol_index(row, col)

            # 这个特图的数量
            num_anc_one = rowcol_index.shape[0]  # 2704
            # 这个特图的尺寸 [[0.025, 0.025], [0.05, 0.05]]
            ancs_wh = self.ancs_scale[i]
            # 单体复制 每一个格子有
            num_anc = len(ancs_wh)
            rowcol_index = rowcol_index.repeat_interleave(num_anc, dim=0)
            rowcol = torch.tensor((row, col))
            # xy 与 rowcol相反
            ancs_yx = torch.true_divide(rowcol_index, rowcol)
            if self.is_xymid:
                midyx = torch.true_divide(1, rowcol) / 2
                ancs_yx = ancs_yx + midyx
            ancs_wh = torch.tensor(ancs_wh).repeat(num_anc_one, 1)
            ancs_xy = ancs_yx.numpy()[:, ::-1]
            ancs_xy = torch.tensor(ancs_xy.copy(), dtype=torch.float)
            ans_xywh = torch.cat([ancs_xy, ancs_wh], dim=-1)
            ret_ancs = torch.cat([ret_ancs, ans_xywh], dim=0)
        if self.anchors_clip:  # 对于xywh 来说这个参数 是没有用的
            xywh2ltrb(ret_ancs, safe=False)
            ret_ancs.clamp_(min=0, max=1)  # 去除超边际的
            ltrb2xywh(ret_ancs, safe=False)
        if self.is_real_size:
            ret_ancs = ret_ancs * torch.tensor(self.img_in_size)[None].repeat(1, 2)
        __d = 1
        if self.device is not None:
            ret_ancs = ret_ancs.to(self.device)
        return ret_ancs


if __name__ == '__main__':
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
    anchors_xywh = FAnchors_v2(size, anc_scale, feature_map_steps,
                          anchors_clip=False, is_xymid=True, is_real_size=True).cre_anchors()
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
    import numpy as np

    # img_pil_new = Image.new('RGB', size, (128, 128, 128))
    # # show_bbox4pil(img_pil_new, xywh2ltrb(__anchors))
    # show_bbox4pil(img_pil_new, xywh2ltrb(anchors_xywh))

    import matplotlib.pyplot as plt

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
