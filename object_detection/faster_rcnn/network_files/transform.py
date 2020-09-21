import torch
from torch import nn, Tensor
import random
import math
from object_detection.faster_rcnn.network_files.image_list import ImageList
from torch.jit.annotations import List, Tuple, Dict, Optional
import torchvision


class GeneralizedRCNNTransform(nn.Module):
    """
    每次只处理一个图片 归一化 , 缩放到指定指定 尺寸范围内 再按批量最大尺寸进行扩展
    """
    def __init__(self, min_size, max_size, image_mean, image_std):
        '''

        :param min_size: 图片最小边长
        :param max_size: 图片最大边长
        :param image_mean: 均值 根据 imgnet
        :param image_std:
        '''
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):  # 用于 resize 时选取没什么用
            min_size = (min_size,)  # 与验证时可以通用 变成list
        self.min_size = min_size  # 指定图像的最小边长范围 数
        self.max_size = max_size  # 指定图像的最大边长范围 数
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值 3个数
        self.image_std = image_std  # 指定图像在标准化处理中的方差 3个数

    def normalize(self, image):
        """标准化处理 根据源设置类型和设备"""
        # 新进的参数需要通过 as_tensor 确保类型和设备的一致性
        dtype, device = image.dtype, image.device  # C,H,W
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # 快速维度调整   [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]  # 返回各通道标准化后的图像

    def torch_choice(self, l):
        '''
        在输入的集合中  随机选择 返回索引
        :param l: 输入支持 集合中 的
        :return: 1~len 中均匀分布中1个
        '''
        # type: (List[int])
        # 通过均匀分布
        index = int(torch.empty(1).uniform_(0., float(len(l))).item())  # 生成1个 指定范围内的数
        return l[index]

    def resize(self, image, target):
        '''
        将图片缩放到指定的大小范围内，并处理对应缩放bboxes信息
        :param image: 一张图片 torch.Size([3, 335, 500])
        :param target: 一个target对象
        :return:
        '''
        # type: (Tensor, Optional[Dict[str, Tensor]])
        # ------确定一个缩放因子 以最大尺寸为优先, 缩放图片到指定的宽高--------
        # image shape is [channel, height, width]
        h, w = image.shape[-2:]  # (800 600) 这个取出来不是tensor
        ts_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(ts_shape))  # 获取高宽中的最小值
        max_size = float(torch.max(ts_shape))  # 获取高宽中的最大值
        if self.training:
            size = float(self.torch_choice(self.min_size))  # 指定的最小值中随机选一个
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])  # 指定输入图片的最小边长,注意是self.min_size不是min_size
        scale_factor = size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

        # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

        # interpolate利用插值的方法缩放图片
        # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
        # bilinear 双线性插值 只支持4D Tensor
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor,
            mode='bilinear', align_corners=False)[0]

        if target is None:  # 测试时不处理bbox
            return image, target

        # --------------------缩放bbox到指定的宽高---------------------
        boxes = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        boxes = resize_boxes(boxes, (h, w), image.shape[-2:])
        target["boxes"] = boxes
        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        '''

        :param the_list: 每一个图片的 c w h
        :return:
        '''
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:  # 只比 hw 感觉可以直接通过矩阵运算完成
            for index, item in enumerate(sublist):  #
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        '''
        统一图像尺寸 用于多图同时处理 没有进行缩入, 根据批量的最大尺寸进行 动态进行 扩充
        :param images: list(tensor 3维)一批图片
        :param size_divisible: # 使尺寸能够尽量对齐32 加速或减小误差 适应神经网络的缩放
        :return: 输出根据现有批量的尺寸确定尺寸
        '''
        # type: (List[Tensor], int)
        if torchvision._is_tracing():  # 训练时不满足,将模型转换为开放模型 可以在各个框架转换 可转换为静态格式
            return self._onnx_batch_images(images, size_divisible)

        # 分别计算一个batch中所有图片中的最大channel, height, width 两个数对应维度取最大 通过矩阵max可直接求
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # max_size = list(max_size)
        # 尺寸扩展成32的倍数值 修正 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width] 这个是形成维度扩展
        batch_shape = [len(images)] + max_size  # [3, 800, 1088]
        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_shape, 0)  # 前面随便带一个生成一个新的tensor 全充0
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # 选中chw值进行替换 修改batched中的数据
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        '''
        对网络最后的预测结果, 就是选框还原到原图的选框
        :param result: 网络的结果 bbox 和类别信息
        :param image_shapes: 预处理后的图片尺寸
        :param original_image_sizes: 原图尺寸
        :return:
        '''
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]])
        if self.training:  # 训练只需要损失
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self, images, targets=None):
        '''
        归一化 , 缩放到指定指定 尺寸范围内 再按批量最大尺寸进行扩展
        :param images: 原始图片
        :param targets: targets对象
        :return:
        '''
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        images = [img for img in images] # 这个感觉 是多余的
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)  # 对图像进行标准化处理
            image, target_index = self.resize(image, target_index)  # 对图像和对应的bboxes缩放到指定范围
            images[i] = image  # 替换img
            if targets is not None and target_index is not None:
                targets[i] = target_index  # 替换target_index

        # 记录resize后的图像尺寸 每张图片的尺寸是不一样的
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)  # 形成动态统一的尺寸
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])  # 类似于java的定义方式

        for image_size in image_sizes:  # 校验维度
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        # 形成对象 并记录原始尺寸返回 用于后期使用
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets


def resize_boxes(boxes, original_size, new_size):
    '''
    用于预处理 和 最后的测试(预测还原)
    :param boxes: 输入多个
    :param original_size: 图像缩放前的尺寸
    :param new_size: 图像缩放后的尺寸
    :return:
    '''
    # type: (Tensor, List[int], List[int]) -> Tensor
    # 输出数组   新尺寸 /旧尺寸 = 对应 h w 的缩放因子
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]

    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(dim=1)  # 分列
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
