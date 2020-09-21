from __future__ import absolute_import
from __future__ import division
import torch as t
from .voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from . import util
import numpy as np
from ..utils.config import opt


# 反规约化
def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


# 用pytorch权重时需要做的数据规约化
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


# 使用caffe权重时需要做的数据规约化
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


# 图片进行缩放，使得长边小于等于1000，短边小于等于600（至少有一个等于）。
def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)  # 调整到限制尺寸之类
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):  # 预处理类

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        # --- 预算归一 ---
        img = preprocess(img, self.min_size, self.max_size)  # 预处理图片

        _, o_H, o_W = img.shape
        scale = o_H / H
        # 预处理图片中的bbox,对相应的bounding boxes 也进行同等尺度的缩放。
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip水平翻转
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt  # 预设参数
        self.db = VOCBboxDataset(opt.voc_data_dir)  # 用预设参数创建数据集类
        self.tsf = Transform(opt.min_size, opt.max_size)  # 实例化预处理类

    def __getitem__(self, idx):  # 获得idx下标的一个数据
        '''

        :param idx:
        :return:
            img     <class 'tuple'>: (3, 600, 800)
            bbox    <class 'tuple'>: (3, 4)
            label   <class 'tuple'>: (3,)
            scale   1.6
        '''
        ori_img, bbox, label, difficult = self.db.get_example(idx)  # 从数据集中获得一个batch的数据

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))  # 预处理数据
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):  # 获得数据集总长度
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
