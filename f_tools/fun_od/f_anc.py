import itertools

import math
import numpy as np
import torch
from PIL import Image
from torch import nn

from f_tools.fun_od.f_boxes import ltrb2xywh, ltrb2ltwh, fix_bbox, xywh2ltrb
from f_tools.pic.f_show import show_bbox4pil

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


class AnchorsFound(object):
    def __init__(self, image_size, anchors_size, feature_map_steps, anchors_clip=False):
        '''
        没有长宽比就是个正方形
        :param image_size:  原图预处理后尺寸 im_height, im_width
        :param anchors_size:  框尺寸 [[16, 32], [64, 128], [256, 512]] 对应特图
        :param feature_map_steps: [8, 16, 32]  # 特图的步距 可通过尺寸算出来
        :param anchors_clip: 是否剔除超边界
        '''
        super(AnchorsFound, self).__init__()
        # 定义对应3个特征的尺寸 [[16, 32], [64, 128], [256, 512]] 这里是框正方形
        # 这个尺寸是640上面的尺寸
        self.anchors_size = anchors_size  # 这个定义一个特图有多少个Anchors
        # 特征层对应的步距   [8, 16, 32] 原图size/特图size = images/feature_maps = steps
        self.feature_map_steps = feature_map_steps  # 这个定义特图下采样比例
        self.anchors_clip = anchors_clip  # 是否剔除超边界---超参

        self.image_size = image_size
        # 根据预处理后的尺寸及步距 计算每一个特图的尺寸
        from math import ceil
        # 根据预处理图片及下采倍数，计算特图尺寸 <class 'list'>: [[52, 52], [26, 26], [13, 13]] (w,h)
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
                             for step in self.feature_map_steps]

    def get_anchors(self):
        '''
        要得到最终的框还需乘原图对应的尺寸
        :return:
            返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框
            这里是 x,y,w,h 调整系数
        '''
        anchors = []
        # 为每一个特图 生成
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.anchors_size[k]  # 取出对应 Anchors 的尺寸数组
            # 每个网格点2个先验框，都是正方形
            from itertools import product as product
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


class Anchors():
    def __init__(self, image_size, anchors_size, feature_size_steps, anchors_clip=False,
                 is_xymid=True, is_real_size=False):
        '''
        没有长宽比就是个正方形 用于已知图片
        :param image_size:  原图预处理后尺寸 w,h
        :param anchors_size:每一个特图对应的预测的尺寸, 训练集归一化聚9类 * 统一的输入尺寸而得
             ANCHORS_SIZE = [
                [[116, 90], [156, 198], [373, 326]],  # 小特图大目标 13x13
                [[30, 61], [62, 45], [59, 119]], # 26, 26
                [[10, 13], [16, 30], [33, 23]],  # 大特图小目标 52, 52
            ]
        :param feature_size_steps: [8, 16, 32]  # 特图的步距 可通过尺寸算出来
        :param anchors_clip: 是否剔除超边界
        '''
        self.anchors_clip = anchors_clip  # 是否剔除超边界---超参
        self.anchors_size = anchors_size
        self.is_real_size = is_real_size  # 以 anchors_size 的真实尺寸输出 非归一化
        self.is_xymid = is_xymid  # 是否将anc中心点移到格子中间
        # 特征层对应的步距   [8, 16, 32] 原图size/特图size = images/feature_maps = steps
        self.feature_size_steps = feature_size_steps  # 这个定义特图下采样比例

        self.image_size = image_size
        # 根据预处理后的尺寸及步距 计算每一个特图的尺寸
        from math import ceil
        # 根据预处理图片及下采倍数，计算特图尺寸 <class 'list'>: [[52, 52], [26, 26], [13, 13]] (w,h)
        self.feature_sizes = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
                              for step in self.feature_size_steps]

    def get_anchors(self):
        '''
        要得到最终的框还需乘原图对应的尺寸
        :return:
            返回特图 h*w,4 anchors 是每个特图的长宽个 4维整框
            这里是 x,y,w,h 调整系数
        '''
        anchors = []
        # 遍历每一个特图
        for i, feature_size in enumerate(self.feature_sizes):
            # 取出 每一个特图对应的 Anchors 的尺寸组 [[116, 90], [156, 198], [373, 326]]
            ancs_szie = self.anchors_size[i]
            # 每个网格点2个先验框，都是正方形
            from itertools import product as product
            # 取每一个特图的w,h   ,进行笛卡尔积循环 相当于行列坐标   外遍历x  遍历y  anc向下铺?
            for xw, yh in product(range(feature_size[0]), range(feature_size[1])):
                for anc_size in ancs_szie:  # 遍历三个尺寸
                    scale_x = anc_size[0] / self.image_size[0]  # anc -> 特图的映射
                    scale_y = anc_size[1] / self.image_size[1]
                    # self.steps = [8, 16, 32]
                    # 加0.5是将中间点从左上角调整到格子的中间
                    if self.is_xymid:
                        dense_cx = [x * self.feature_size_steps[i] / self.image_size[0] for x in [xw + 0.5]]
                        dense_cy = [y * self.feature_size_steps[i] / self.image_size[1] for y in [yh + 0.5]]
                    else:
                        dense_cx = [x * self.feature_size_steps[i] / self.image_size[0] for x in [xw]]
                        dense_cy = [y * self.feature_size_steps[i] / self.image_size[1] for y in [yh]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, scale_x, scale_y]  # x,y,w,h 特图的每一个点对应原图的比例
            # 最终形成的 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 归一化调整系数 需乘个size才能得到具体的框
            # 形成的顺序是 每一个特图->每个格子(竖向填充)->每一个尺寸

        output = torch.Tensor(anchors).view(-1, 4)
        if self.anchors_clip:
            output.clamp_(max=1, min=0)  # 去除超边际的
        if self.is_real_size:
            output = output * torch.tensor(self.image_size)[None].repeat(1, 2)
        return output  # torch.Size([10647, 4])


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
        anchors = []
        # 遍历每一个特图
        for i, feature_size in enumerate(self.feature_sizes):
            # 取出 每一个特图对应的 Anchors 的尺寸组 [[116, 90], [156, 198], [373, 326]]
            feature_ancs_scale = self.ancs_scale[i]
            # 每个网格点2个先验框，都是正方形
            from itertools import product as product
            # 取每一个特图的w,h   ,进行笛卡尔积循环 相当于行列坐标   外遍历x  遍历y  anc向下铺?
            for dim0, dim1 in product(range(feature_size[1]), range(feature_size[0])):
                # 这里的 row 对y,col对x
                for anc_scale in feature_ancs_scale:  # 遍历三个尺寸
                    scale_x = anc_scale[0]
                    scale_y = anc_scale[1]
                    # self.steps = [8, 16, 32]
                    # 加0.5是将中间点从左上角调整到格子的中间
                    if self.is_xymid:
                        px = [x * self.feature_size_steps[i] / self.img_in_size[0] for x in [dim1 + 0.5]]
                        py = [y * self.feature_size_steps[i] / self.img_in_size[1] for y in [dim0 + 0.5]]
                    else:
                        px = [x * self.feature_size_steps[i] / self.img_in_size[0] for x in [dim1]]
                        py = [y * self.feature_size_steps[i] / self.img_in_size[1] for y in [dim0]]
                    for cy, cx in product(py, px):
                        anchors += [cx, cy, scale_x, scale_y]  # x,y,w,h 特图的每一个点对应原图的比例
            # 最终形成的 anchors 是每个特图的长宽个 4维整框 这里是 x,y,w,h 归一化调整系数 需乘个size才能得到具体的框
            # 形成的顺序是 每一个特图->每个格子(竖向填充) 建议从小到大 需与模型匹配 ->每一个尺寸

        output = torch.Tensor(anchors).view(-1, 4)
        if self.anchors_clip:  # 对于xywh 来说这个参数 是没有用的
            xywh2ltrb(output, safe=False)
            output.clamp_(max=1, min=0)  # 去除超边际的
            ltrb2xywh(output, safe=False)
        if self.is_real_size:
            output = output * torch.tensor(self.img_in_size)[None].repeat(1, 2)
        if self.device is not None:
            output = output.to(self.device)
        return output  # torch.Size([10647, 4])


class FAnchors_v2:
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
            _row_index = torch.arange(row)  # 3行 4列
            _col_index = torch.arange(col)
            y, x = torch.meshgrid(_row_index, _col_index)
            _row_col_index = torch.stack((x, y), dim=2)
            grid_index = _row_col_index.view(-1, 2)
            # 一个anc的数量
            num_anc_one = len(grid_index)  # 2704
            fancs_wh = self.ancs_scale[i]
            # anc 数
            grid_index = grid_index.repeat_interleave(len(fancs_wh), dim=0)
            rowcol = torch.tensor((row, col))
            ancs_xy = torch.true_divide(grid_index, rowcol)
            if self.is_xymid:
                midxy = torch.true_divide(1, rowcol)/2
                ancs_xy = ancs_xy + midxy

            ancs_wh = torch.tensor(fancs_wh).repeat(num_anc_one, 1)
            ans_xywh = torch.cat([ancs_xy, ancs_wh], dim=-1)
            ret_ancs = torch.cat([ret_ancs, ans_xywh], dim=0)
        if self.anchors_clip:  # 对于xywh 来说这个参数 是没有用的
            xywh2ltrb(ret_ancs, safe=False)
            ret_ancs.clamp_(min=0, max=1)  # 去除超边际的
            if self.is_real_size:
                ret_ancs = ret_ancs * torch.tensor(self.img_in_size)[None].repeat(1, 2)
            ltrb2xywh(ret_ancs, safe=False)
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
    anchors = FAnchors_v2(size, anc_scale, feature_map_steps,
                          anchors_clip=True, is_xymid=True, is_real_size=False).cre_anchors()
    # print(anchors)
    # anchors = FAnchors(size, anc_scale, feature_map_steps,
    #                    anchors_clip=True, is_xymid=False, is_real_size=True).cre_anchors()  # torch.Size([10647, 4])

    # [[52, 52], [26, 26], [13, 13]]
    # index_start = int((52**2+26**2)*3+13**2/2+100)
    index_start = len(anchors)-460
    len = 6
    anchors = anchors[index_start:index_start + len]  # 这里选出 anchors
    # --------------anchors 转换画图--------------
    # __anchors = anchors.clone()
    # __anchors = xywh2ltrb(__anchors)
    # __anchors[:, ::2] = __anchors[:, ::2] * size[0]
    # __anchors[:, 1::2] = __anchors[:, 1::2] * size[1]
    # xywh --> ltwh 为了plt.Rectangle
    # __anchors[:, :2] = __anchors[:, :2] - __anchors[:, 2:] / 2.

    # img_pil_new = Image.new('RGB', list(np.array(size) * 2), (128, 128, 128))
    img_pil_new = Image.new('RGB', size, (128, 128, 128))
    # show_bbox4pil(img_pil_new, xywh2ltrb(__anchors))
    show_bbox4pil(img_pil_new, xywh2ltrb(anchors))

    # import matplotlib.pyplot as plt
    #
    # # 构造点
    # w = [i for i in range(size[0])]
    # h = [i for i in range(size[1])]
    # ww, hh = np.meshgrid(w, h)
    # p = 100  # 图片显示padding
    # # ----------图形配置---------
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.ylim(-p, size[0] + p)
    # plt.xlim(-p, size[1] + p)
    # ax.invert_yaxis()  # y轴反向
    # plt.scatter(ww.reshape(-1), hh.reshape(-1))
    # for a in __anchors:  # 画矩形框
    #     rect = plt.Rectangle((a[0], a[1]), a[2], a[3], color="r", fill=False)
    #     ax.add_patch(rect)
    # plt.show()
    pass
