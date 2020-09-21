import itertools

import math
import numpy as np
import torch
from torch import nn

from f_tools.fun_od.f_boxes import tlbr2yxhw, tlbr2tlhw

cor_names = {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    'black': '#000000',
    'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen': '#90EE90',
    'lightgray': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32',
}


def generate_anc_base(base_size=16,
                      ratios=(0.5, 1, 2),
                      anchor_scales=(8, 16, 32)
                      ):
    '''
    生成一套 anchors 模板 AnchorsGenerator
    :param base_size:
    :param ratios:
    :param anchor_scales:
    :return: (len(ratios) * len(anchor_scales), 4) 左上右下
    '''
    py = base_size / 2.
    px = base_size / 2.

    # 以特征图的左上角点为基准产生的9个anchor
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


class AnchorsGenerator4Torch(nn.Module):
    """
    anchors生成器 输出 左上右下 输出list[tensor]
    没有 clamp_
    """

    def __init__(self, scales=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        '''
        这里定义一套 anchors
            aspect_ratios = (0.5, 1, 2)
            anchor_sizes = [[16, 32], [256, 256]]
        :param scales: 这里shape[1]数要对应特图数否则无效  ((32,), (64,), (128,), (256,), (512,))
        :param aspect_ratios: 长宽比 对应anc个数 ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        '''
        super(AnchorsGenerator4Torch, self).__init__()
        if not isinstance(scales[0], (list, tuple)):
            # TODO 元素不是数组就 加一层,统一格式 当用于与多个特征图输入对应
            # ((32,), (64,), (128,), (256,), (512,))
            scales = tuple((s,) for s in scales)
        if not isinstance(aspect_ratios[0], (list, tuple)):  # 自动与前面的括号匹配
            aspect_ratios = (aspect_ratios,) * len(scales)  # 自动与前面的括号匹配
        assert len(scales) == len(aspect_ratios)  # 确保一样长(list两层), 且只有一个元素
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}  # 用于缓存 anchors

    def generate_anchors(self, scale, aspect_ratio, dtype, device):
        '''
        计算 anchors 的尺寸 一次处理一个
        :param scale: (32,)
        :param aspect_ratio: aspect_ratios:(0.5, 1.0, 2.0)
        :param dtype: float32
        :param device: cpu/gpu
        :return:返回 3维(多个特图对应)或2维数组
        '''
        scale = torch.as_tensor(scale, dtype=torch.float, device=device)
        aspect_ratio = torch.as_tensor(aspect_ratio, dtype=torch.float, device=device)
        # 这里是标准算法
        h_ratios = torch.sqrt(aspect_ratio)
        w_ratios = 1.0 / h_ratios
        # 生成x个anchors的中点 多维交叉相乘 3个扩2维 1个扩2维 拉平 np高级
        ws = (w_ratios[:, None] * scale[None, :]).view(-1)
        hs = (h_ratios[:, None] * scale[None, :]).view(-1)

        # 表示以（0, 0）为中心  left-top, right-bottom 两个点
        # 宽高合并2维调为4维 将数组中每一个值按 1 维 进行组合(列组合)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()  # 四舍五入

    def set_cell_anchors(self, dtype, device):
        '''
        新产生的数据保持变量的一致性

        :param dtype:
        :param device:
        :return:
            self.cell_anchors 用于保存一套 anchors
        '''
        # 已有直接返回
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        # sizes ((32,), (64,), (128,), (256,), (512,))
        # aspect_ratios ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
        # anchors模板都是以(0, 0)为中心的anchor
        cell_anchors = [
            self.generate_anchors(scale, aspect_ratio, dtype, device)
            for scale, aspect_ratio in zip(self.scales, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors  # 暂时感觉可以用普通变量,可能后面会用到, 通过类变量保存 每一个特图对应的 anchors 模块[torch.Size([3, 4]),]

    def num_anchors_per_location(self):
        # 计算每个特图对应的 anchors 数量 返回list
        num_anchors = [len(s) * len(a) for s, a in zip(self.scales, self.aspect_ratios)]
        return num_anchors

    def grid_anchors(self, grid_sizes, strides):
        '''
        生成对应原图的所有 anchors 坐标 左上右下
        :param grid_sizes: 每一个特图的尺寸
        :param strides: 对应的步距
        :return:
        '''
        anchors = []
        cell_anchors = self.cell_anchors  # 每个特图对应的一套
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            # --------这里是anc组装------ 三个list一致才能处理
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的列坐标 np高级
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的行坐标
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid 函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width * grid_height, 4]
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # shifts 矩阵的每一个点(每一行偏移)需要叠加9个 anchors...形成 (base anchor, output anchor) ,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))  # 拉平形成 (grid_width * grid_height * len(anchor),4)

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        '''
        缓存生成的 anchors
        :param grid_sizes: 每一个特图的尺寸
        :param strides: 对应的步距
        :return: 输出 List[Tensor] 每一个特图对应的所有 anchors torch.Size([182400, 4]) torch.Size([741, 4])
            特图的越大 anchors 越多
        '''
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)  # 用于缓存 下批有相同的 grid_sizes 和 strides 时复用
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors  # 缓存
        return anchors

    def forward(self, imgs_size, fms_size):
        '''
        np高级 维度运算有点难
        特图尺寸 * anc个数=32*32*3
        :param imgs_size: 一批图片大小是一样的 img_size = torch.tensor([640, 640])
        :param fms_size: feature_maps多个特图  [torch.tensor([32, 32]), torch.tensor([24, 24])]
        :return: List[torch.Tensor[]]
            每一个图对应的所有anchors (每一个图 * 每一图的特图,4)  List[Tensor]-torch.Size([369303, 4])
        '''
        # 取每一个特图 H,W
        grid_sizes = list([f for f in fms_size])

        # 每一个图像图片tensor(最大批) H,W 一批图片大小是一样的
        dtype, device = fms_size[0].dtype, fms_size[0].device  # 取类型和设备

        # 这一批图片与每一个特图的步长  , 计算每一个特图与最大批的步长 最大批尺寸 / 多个特图的尺寸 自动扩展
        strides = [imgs_size.float() / g for g in grid_sizes]

        # 根据提供的 sizes 和 aspect_ratios 生成anchors模板
        self.set_cell_anchors(dtype, device)  # 存到 self.cell_anchors 中

        # -------应用到原图------
        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # List[Tensor] 每一个特图对应的所有 anchors
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        '''-------改造这里支持多特图 直接返回anchors_over_all_feature_maps----------'''
        # anchors = []
        # # List[List[Tensor]]
        # for i in range(imgs_size.dim()):
        #     anchors_in_image = []  # 每一个图对应的所有特征图的 anchors
        #     # 遍历List[Tensor] 每一个特图对应的所有 anchors
        #     for anchors_per_feature_map in anchors_over_all_feature_maps:
        #         anchors_in_image.append(anchors_per_feature_map)
        #     anchors.append(anchors_in_image)
        # # 将每一个图 - 每一图的特图 - 对应anchors  拉平 (每一个图 * 每一图的特图,4)
        # # anchors是个list，每个元素为一张图像的所有anchors信息
        # anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # # Clear the cache in case that memory leaks.
        self._cache.clear()  # 这里要优化 才创建就删除不合理,应该按容量进行操作
        return anchors_over_all_feature_maps


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        '''
        生成 dboxes_ltrb 和 dboxes为所有特图 对应的所有anc 例如  (8000,4) 预测已经组合拉平了的
        :param fig_size: 原图尺寸 输入网络的尺寸 300
        :param feat_size:特图尺寸 [38, 19, 10, 5, 3, 1]
        :param steps: [8, 16, 32, 64, 100, 300] 不等于 原图/特图 这个值怎么来的?
        :param scales: 必须大于feat_size数量 [21, 45, 99, 153, 207, 261, 315]
        :param aspect_ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        :param scale_xy: 用于损失函数修正系数
        :param scale_wh: 用于损失函数修正系数
        '''
        self.fig_size = fig_size  # 输入网络的图像大小 300
        self.feat_size = feat_size  # 每个预测层的feature map尺寸

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        #
        self.steps = steps  # 每个特征层上的一个cell在原图上的跨度

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales  # 每个特征层上预测的default box的scale
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios  # 每个预测特征层上预测的default box的ratios

        fk = fig_size / np.array(steps)  # 计算每层特征层的fk 跨度值用于与调整系数相乘得真实框

        self.default_boxes = []
        # size of feature and number of feature
        # 遍历每层特征层，计算default box
        for idx, sfeat in enumerate(self.feat_size):
            # scales 要比 特图多一层
            sk1 = scales[idx] / fig_size  # scale转为相对值[0-1]
            sk2 = scales[idx + 1] / fig_size  # scale转为相对值[0-1] 下一层的尺寸
            sk3 = math.sqrt(sk1 * sk2)
            # 先添加两个1:1比例的default box宽和高 定制加入1比1
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # 再将剩下不同比例的default box宽和高添加到all_sizes中
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # 计算当前特征层对应原图上的所有default box
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):  # i -> 行（y）， j -> 列（x）
                    # 计算每个default box的中心坐标（范围是在0-1之间）
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        # 将default_boxes转为tensor格式 中心宽高
        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float32)  # 这里不转类型会报错
        self.dboxes.clamp_(min=0, max=1)  # 将坐标（x, y, w, h）都限制在0-1之间

        # For IoU calculation 转左上右下
        # ltrb is left top coordinate and right bottom coordinate
        # 将(x, y, w, h)转换成(xmin, ymin, xmax, ymax)，方便后续计算IoU(匹配正负样本时)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]  # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]  # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]  # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]  # ymax

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
            return self.dboxes


if __name__ == '__main__':
    # print(generate_anc_base())
    # anchor_generator = AnchorsGenerator4Torch(sizes=((32), (64), ((96))),
    #                                           aspect_ratios=((0.5,), (1.0, 2.0), (0.3, 0.8, 1.5)))
    # print(anchor_generator(torch.tensor([640, 640]),
    #                        [torch.tensor([32, 32]), torch.tensor([24, 24])]))
    '''
    16---torch.Size([3072, 4])
    32---torch.Size([3072, 4])
    
    特图尺寸 * anc个数=32*32*3
    
    
    '''
    anchor_scales = [[21], [45], [99], [153], [207], [261, 315]]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    anchor_scales = [[16], [32]]
    aspect_ratios = [[0.5, 1, 2], [2]]
    anchor_generator = AnchorsGenerator4Torch(scales=anchor_scales, aspect_ratios=aspect_ratios)

    img_size = [300, 300]
    fms_size = [
        torch.tensor([38, 38]),
        torch.tensor([19, 19]),
        torch.tensor([10, 10]),
        torch.tensor([5, 5]),
        torch.tensor([3, 3]),
        torch.tensor([1, 1]),
    ]
    anchors = anchor_generator(torch.tensor(img_size), fms_size)

    # -----def测试-------
    feat_size = [38, 19]  # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]  # 每个特征层上的一个cell在原图上的跨度
    _anchor_scales = [i[0] for i in anchor_scales]
    _anchor_scales.append(64)
    dboxes = DefaultBoxes(img_size[0],
                          feat_size,
                          steps,
                          _anchor_scales,
                          aspect_ratios)

    import matplotlib.pyplot as plt

    # 构造点
    space = 10
    ws = [i for i in range(0, img_size[0], space)]
    hs = [i for i in range(0, img_size[1], space)]
    ww, hh = np.meshgrid(ws, hs)
    p = 0  # 图片显示padding
    # ----------图形配置---------
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-p, img_size[0] + p)  # 这里反序wh
    plt.xlim(-p, img_size[1] + p)
    ax.invert_yaxis()  # y轴反向
    plt.scatter(ww.reshape(-1), hh.reshape(-1))

    print(anchors[0].shape[0])
    # index_start = int(anchors[0].shape[0] / 2)
    index_start = 0
    _anchors = anchors[0][index_start:index_start + len(aspect_ratios) * len(anchor_scales)]
    # anchors = anchors[0]
    colors_str = ['k', 'r', 'y', 'g', 'c', 'b', 'm',
                  'gray', 'brown', 'coral', 'gold',
                  'saga']
    for i, (a, cor_key) in enumerate(zip(_anchors, cor_names)):  # 画矩形框
        a = tlbr2tlhw(a[None])
        a = a.reshape(-1)
        # 这里要左上长宽,左上长宽
        rect = plt.Rectangle((a[0], a[1]), a[2], a[3], color=cor_names[cor_key], fill=False)
        # rect = plt.Rectangle((a[0], a[1]), a[2], a[3], color=cor_names['yellowgreen'], fill=False)
        ax.add_patch(rect)

    plt.show()
