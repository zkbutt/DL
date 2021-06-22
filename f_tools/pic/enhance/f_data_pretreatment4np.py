import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

from f_tools.GLOBAL_LOG import flog
from f_tools.pic.enhance.f_data_pretreatment4pil import BasePretreatment
from f_tools.pic.f_show import f_show_od_np4plt
import matplotlib.pyplot as plt

FDEBUG = False

'''
输入：
    原图 uint8 
    boxes 为真实图片的ltrb值
输出：
    正则化变通道后的RGB图 
'''


def cre_transform_resize4np(cfg):
    if cfg.USE_BASE4NP:
        flog.error('使用的是 USE_BASE4NP 模式 %s', cfg.USE_BASE4NP)
        data_transform = {
            "train": Compose([
                ConvertFromInts(),  # image int8转换成float [0,256)
                Resize(cfg.IMAGE_SIZE),
                Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
                ConvertColor(current='BGR', transform='RGB'),
                ToTensor(is_box_oned=False),
            ], cfg)
        }
    else:
        if cfg.KEEP_SIZE:  # 不进行随机缩放 不改变图片的位置和大小
            data_transform = {
                "train": Compose([
                    ConvertFromInts(),  # image int8转换成float [0,256)
                    # ToAbsoluteCoords(), # 恢复真实尺寸
                    PhotometricDistort(),  # 图片处理集合
                    # Expand(cfg.PIC_MEAN),  # 放大缩小图片
                    # RandomSampleCrop(),  # 随机剪切定位
                    RandomMirror(),
                    # ToPercentCoords(),  # boxes 按原图归一化 最后ToTensor 统一归一
                    Resize(cfg.IMAGE_SIZE),  # 定义模型输入尺寸 处理img和boxes
                    Normalize(cfg.PIC_MEAN, cfg.PIC_STD),  # 正则化图片
                    ConvertColor(current='BGR', transform='RGB'),
                    ToTensor(is_box_oned=True),  # img 及 boxes(可选,前面已归一)归一  转tensor
                ], cfg)
            }
        else:
            data_transform = {
                "train": Compose([
                    ConvertFromInts(),  # image int8转换成float [0,256)
                    # ToAbsoluteCoords(),  # 输入已是原图不需要恢复 boxes 恢复原图尺寸
                    PhotometricDistort(),  # 图片处理集合
                    Expand(cfg.PIC_MEAN),  # 放大缩小图片
                    RandomSampleCrop(),  # 随机剪切定位 keypoints
                    RandomMirror(),
                    # ToPercentCoords(),  # boxes 按原图归一化 最后统一归一 最后ToTensor 统一归一
                    Resize(cfg.IMAGE_SIZE),
                    Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
                    ConvertColor(current='BGR', transform='RGB'),
                    ToTensor(is_box_oned=True),
                ], cfg)
            }

    data_transform["val"] = Compose([
        Resize(cfg.IMAGE_SIZE),
        Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
        ConvertColor(current='BGR', transform='RGB'),
        ToTensor(is_box_oned=False),
    ], cfg)

    return data_transform


def cre_transform_base4np(cfg):
    flog.warning('预处理使用 cre_transform_base4np', )
    data_transform = {
        "train": Compose([
            Resize(cfg.IMAGE_SIZE),
            Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
            ConvertColor(current='BGR', transform='RGB'),
            ToTensor(is_box_oned=True),
        ], cfg)
    }

    data_transform["val"] = Compose([
        Resize(cfg.IMAGE_SIZE),
        Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
        ConvertColor(current='BGR', transform='RGB'),
        ToTensor(is_box_oned=False),
    ], cfg)

    return data_transform


def _copy_box(boxes):
    if isinstance(boxes, np.ndarray):
        boxes_ = boxes.copy()
    elif isinstance(boxes, torch.Tensor):
        boxes_ = boxes.clone()
    else:
        raise Exception('类型错误', type(boxes))

    return boxes_


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(BasePretreatment):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, cfg=None):
        super(Compose, self).__init__(cfg)
        self.transforms = transforms

    def __call__(self, img, target):
        # f_plt_show_cv(image,boxes)
        for t in self.transforms:
            img, target = t(img, target)
            if target is not None:
                if len(target['boxes']) != len(target['labels']):
                    flog.warning('!!! 数据有问题 Compose  %s %s %s ', len(target['boxes']), len(target['labels']), t)
        return img, target


class Lambda(object):
    """Applies a lambda as a transform. 这个用于复写方法?"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, target):
        return self.lambd(img, target)


class ConvertFromInts(object):
    def __call__(self, image, target):
        '''cv打开的np 默认是uint8'''
        return image.astype(np.float32), target


class Normalize(object):
    def __init__(self, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, target):
        # bgr
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, target


class ToAbsoluteCoords(object):
    ''' boxes 恢复原图尺寸  归一化尺寸转原图  ToAbsoluteCoords 相反'''

    def __call__(self, image, target):
        '''
        归一化 -> 绝对坐标
        :param image:
        :param boxes:
        :param labels:
        :return:
        '''
        height, width, channels = image.shape
        if target is not None:
            # 这里可以直接改
            target['boxes'][:, 0] *= width
            target['boxes'][:, 2] *= width
            target['boxes'][:, 1] *= height
            target['boxes'][:, 3] *= height
            if 'keypoints' in target:
                target['keypoints'][:, ::2] *= width
                target['keypoints'][:, 1::2] *= height

        return image, target


class ToPercentCoords(object):
    '''原图转归一化尺寸  ToAbsoluteCoords 相反'''

    def __call__(self, image, target):
        height, width, channels = image.shape
        if target is not None:
            target['boxes'][:, 0] /= width
            target['boxes'][:, 2] /= width
            target['boxes'][:, 1] /= height
            target['boxes'][:, 3] /= height

            if 'keypoints' in target:
                target['keypoints'][:, ::2] /= width
                target['keypoints'][:, 1::2] /= height

        return image, target


class Resize(object):
    def __init__(self, size=(320, 320)):
        self.size = size

    def __call__(self, image, target):
        w_ratio, h_ratio = np.array(image.shape[:2][::-1]) / np.array(self.size)
        if FDEBUG:
            plt.imshow(image)
            plt.show()
        if target is not None:
            target['boxes'][:, [0, 2]] = target['boxes'][:, [0, 2]] / w_ratio
            target['boxes'][:, [1, 3]] = target['boxes'][:, [1, 3]] / h_ratio

            if 'keypoints' in target:
                target['keypoints'][:, ::2] /= w_ratio
                target['keypoints'][:, 1::2] /= h_ratio

        image = cv2.resize(image, (self.size[1], self.size[0]))
        return image, target


class RandomSaturation(object):
    '''随机色彩 需要HSV'''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, target


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise(object):
    '''随机颜色打乱'''

    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target


class ConvertColor(object):
    '''bgr -> hsv'''

    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError
        return image, target


class RandomContrast(object):
    '''随机透明度'''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, target):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        # f_plt_show_cv(image)
        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        '''随机亮度增强'''
        if random.randint(2):  # 50%
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        # f_plt_show_cv(image)
        return image, target


class ToCV2Image(object):
    def __call__(self, tensor, target):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), target


class ToTensor(object):
    def __init__(self, is_box_oned=True) -> None:
        super().__init__()
        self.is_box_oned = is_box_oned

    def __call__(self, cvimage, target):
        if target and self.is_box_oned:
            # np整体复制 wh
            whwh = np.tile(cvimage.shape[:2][::-1], 2)
            target['boxes'][:, :] = target['boxes'][:, :] / whwh
            if 'keypoints' in target:
                target['keypoints'][:, ::2] /= whwh[0]
                target['keypoints'][:, 1::2] /= whwh[1]

        if FDEBUG:
            plt.imshow(cvimage)
            plt.show()
        # (h,w,c -> c,h,w) = bgr
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), target


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, target):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, target

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(target['boxes'], rect)

                if 'keypoints' in target:
                    raise Exception('随机剪切不支持 keypoints 请设置cfg.KEEP_SIZE = True')

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (target['boxes'][:, :2] + target['boxes'][:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                # current_boxes = boxes[mask, :].copy()
                current_boxes = _copy_box(target['boxes'][mask, :])

                # take only matching gt labels
                current_labels = target['labels'][mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                target['labels'] = current_labels
                target['boxes'] = current_boxes

                return current_image, target


class Expand(object):
    '''随机扩大'''

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, target):
        if random.randint(2):
            return image, target

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image
        # 原尺寸不要了
        boxes = _copy_box(target['boxes'])

        if isinstance(boxes, np.ndarray):
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))
        elif isinstance(boxes, torch.Tensor):
            boxes[:, :2] += torch.tensor((int(left), int(top)))
            boxes[:, 2:] += torch.tensor((int(left), int(top)))
        else:
            raise Exception('类型错误', type(boxes))

        if 'keypoints' in target:
            raise Exception('扩展不支持 keypoints 请设置cfg.KEEP_SIZE = True')

        target['boxes'] = boxes
        # f_plt_show_cv(image,torch.tensor(boxes))
        return image, target


class RandomMirror(object):
    '''随机水平镜像'''

    def __call__(self, image, target):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            # boxes = boxes.copy()
            boxes = _copy_box(target['boxes'])
            # boxes[:, 0::2] = width - boxes[:, 2::-2]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target['boxes'] = boxes

            if 'keypoints' in target:
                target['keypoints'][:, ::2] = width - target['keypoints'][:, ::2]
        return image, target


class SwapChannels(object):
    """
    随机 RGB 打乱
    Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    '''图片增强'''

    def __init__(self):
        self.pd = [
            RandomContrast(),  # 随机透明度
            ConvertColor(transform='HSV'),  # bgr -> hsv
            RandomSaturation(),  # 随机色彩'
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),  # hsv -> bgr
            RandomContrast()  # 随机透明度
        ]
        self.rand_brightness = RandomBrightness()  # 随机亮度增强
        # self.rand_light_noise = RandomLightingNoise()  # 颜色杂音

    def __call__(self, image, target):
        im = image.copy()
        im, target = self.rand_brightness(im, target)
        if random.randint(2):  # 先转换还是后转换
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, target = distort(im, target)
        return im, target
        # return self.rand_light_noise(im, boxes, labels)


class DisposePicSet4SSD(object):
    '''打包启动程序'''

    def __init__(self, size=416, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(), # 恢复真实尺寸
            PhotometricDistort(),  # 图片集合
            Expand(self.mean),  # 加大图片
            RandomSampleCrop(),
            RandomMirror(),
            # ToPercentCoords(), # 归一化尺寸
            Resize(self.size),
            Normalize(self.mean, self.std),
            ConvertColor(current='BGR', transform='RGB'),
            ToTensor(),
        ])

    def __call__(self, img, target):
        return self.augment(img, target)


class KeyPointCropOne:
    '''
    支持 KeyPoint 随机剪图 + 归一化 图片没有归一化
    用这个 ToTensor is_box_oned=False
    '''

    def __init__(self, size_f, keep_scale_wh=False):
        self.keep_scale_wh = keep_scale_wh  # 这个标签不支持不能用
        self.size_f = size_f
        self.flip_landmarks_dict = {
            0: 32, 1: 31, 2: 30, 3: 29, 4: 28, 5: 27, 6: 26, 7: 25, 8: 24, 9: 23, 10: 22, 11: 21, 12: 20, 13: 19,
            14: 18,
            15: 17,
            16: 16, 17: 15, 18: 14, 19: 13, 20: 12, 21: 11, 22: 10, 23: 9, 24: 8, 25: 7, 26: 6, 27: 5, 28: 4, 29: 3,
            30: 2,
            31: 1, 32: 0,
            33: 46, 34: 45, 35: 44, 36: 43, 37: 42, 38: 50, 39: 49, 40: 48, 41: 47,
            46: 33, 45: 34, 44: 35, 43: 36, 42: 37, 50: 38, 49: 39, 48: 40, 47: 41,
            60: 72, 61: 71, 62: 70, 63: 69, 64: 68, 65: 75, 66: 74, 67: 73,
            72: 60, 71: 61, 70: 62, 69: 63, 68: 64, 75: 65, 74: 66, 73: 67,
            96: 97, 97: 96,
            51: 51, 52: 52, 53: 53, 54: 54,
            55: 59, 56: 58, 57: 57, 58: 56, 59: 55,
            76: 82, 77: 81, 78: 80, 79: 79, 80: 78, 81: 77, 82: 76,
            87: 83, 86: 84, 85: 85, 84: 86, 83: 87,
            88: 92, 89: 91, 90: 90, 91: 89, 92: 88,
            95: 93, 94: 94, 93: 95
        }

    def letterbox(self, img_, img_size=256, mean_rgb=(128, 128, 128)):
        shape_ = img_.shape[:2]  # shape = [height, width]
        ratio = float(img_size) / max(shape_)  # ratio  = old / new
        new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
        dw_ = (img_size - new_shape_[0]) / 2  # width padding
        dh_ = (img_size - new_shape_[1]) / 2  # height padding
        top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
        left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
        # resize img
        img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

        img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT,
                                   value=mean_rgb)  # padded square

        return img_a

    def __call__(self, image, target, site_xy, vis=False):
        '''

        :param image:
        :param target:
        :param site_xy: 旋转锚点
        :param vis:
        :return:
        '''
        cx, cy = site_xy
        pts = target['keypoints']
        angle = random.randint(-36, 36)

        (h, w) = image.shape[:2]
        h = h
        w = w
        # (cx , cy) = (int(0.5 * w) , int(0.5 * h))
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算新图像的bounding
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += int(0.5 * nW) - cx
        M[1, 2] += int(0.5 * nH) - cy

        resize_model = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

        img_rot = cv2.warpAffine(image, M, (nW, nH), flags=resize_model[random.randint(0, 4)])
        # flags : INTER_LINEAR INTER_CUBIC INTER_NEAREST
        # borderMode : BORDER_REFLECT BORDER_TRANSPARENT BORDER_REPLICATE CV_BORDER_WRAP BORDER_CONSTANT

        pts_r = []
        for pt in pts:
            x = float(pt[0])
            y = float(pt[1])

            x_r = (x * M[0][0] + y * M[0][1] + M[0][2])
            y_r = (x * M[1][0] + y * M[1][1] + M[1][2])

            pts_r.append([x_r, y_r])

        x = [pt[0] for pt in pts_r]
        y = [pt[1] for pt in pts_r]

        x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)

        translation_pixels = 60

        scaling = 0.3
        x1 += random.randint(-int(max((x2 - x1) * scaling, translation_pixels)), int((x2 - x1) * 0.25))
        y1 += random.randint(-int(max((y2 - y1) * scaling, translation_pixels)), int((y2 - y1) * 0.25))
        x2 += random.randint(-int((x2 - x1) * 0.15), int(max((x2 - x1) * scaling, translation_pixels)))
        y2 += random.randint(-int((y2 - y1) * 0.15), int(max((y2 - y1) * scaling, translation_pixels)))

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(x2, img_rot.shape[1] - 1))
        y2 = int(min(y2, img_rot.shape[0] - 1))

        crop_rot = img_rot[y1:y2, x1:x2, :]

        crop_pts = []
        width_crop = float(x2 - x1)
        height_crop = float(y2 - y1)
        # 归一化
        for pt in pts_r:
            x = pt[0]
            y = pt[1]
            crop_pts.append([float(x - x1) / width_crop, float(y - y1) / height_crop])

        # 随机镜像 这个存在标签左右眼切换问题 98点可以支持 其它需定制
        if random.random() >= 0.5:
            # print('--------->>> flip')
            crop_rot = cv2.flip(crop_rot, 1)
            crop_pts_flip = []
            for i in range(len(crop_pts)):
                # print( crop_rot.shape[1],crop_pts[flip_landmarks_dict[i]][0])
                x = 1. - crop_pts[self.flip_landmarks_dict[i]][0]
                y = crop_pts[self.flip_landmarks_dict[i]][1]
                # print(i,x,y)
                crop_pts_flip.append([x, y])
            crop_pts = crop_pts_flip

        if vis:
            for pt in crop_pts:
                x = int(pt[0] * width_crop)
                y = int(pt[1] * height_crop)

                cv2.circle(crop_rot, (int(x), int(y)), 2, (255, 0, 255), -1)
            # cv2.imshow('img', crop_rot)
            from matplotlib import pyplot as plt
            plt.imshow(crop_rot)
            plt.show()

        if self.keep_scale_wh:
            # 这个标签不支持不能用
            # crop_rot = self.letterbox(crop_rot, img_size=self.size_f[0], mean_rgb=(128, 128, 128))
            raise Exception('self.keep_scale_wh = %s 不支持 ' % self.keep_scale_wh)
        else:
            crop_rot = cv2.resize(crop_rot, self.size_f, interpolation=resize_model[random.randint(0, 4)])

        if vis:
            for pt in crop_pts:
                x = int(pt[0] * self.size_f[0])
                y = int(pt[1] * self.size_f[1])

                cv2.circle(crop_rot, (int(x), int(y)), 2, (255, 0, 255), -1)
            # cv2.imshow('img', crop_rot)
            from matplotlib import pyplot as plt
            plt.imshow(crop_rot)
            plt.show()
        ''' 图片没有归一化 '''
        return image, target


class RandomGray:
    '''随机水平镜像'''

    def __call__(self, image, target):
        if random.random() > 0.8:
            _img = np.zeros(image.shape, dtype=np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _img[:, :, 0] = gray
            _img[:, :, 1] = gray
            _img[:, :, 2] = gray
            image = _img

        return image, target
