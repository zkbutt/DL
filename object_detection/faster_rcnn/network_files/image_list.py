import torch
from torch.jit.annotations import List, Tuple
from torch import Tensor


@torch.jit.script
class ImageList(object):
    '''
    普通的一个类
    '''

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]])
        """
        Arguments:
            tensors (tensor) padding 最大最小 后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors  # 批图片
        self.image_sizes = image_sizes  # 缩放后大小

    def to(self, device):
        '''
        tensor 设备转换
        :param device:
        :return:
        '''
        # type: (Device)
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
