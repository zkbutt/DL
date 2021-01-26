import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

from f_tools.pic.enhance.f_data_pretreatment4pil import BasePretreatment
from f_tools.pic.f_show import f_plt_show_cv


def cre_transform_resize4np(cfg):
    data_transform = {
        "train": Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(), # 恢复真实尺寸
            PhotometricDistort(),  # 图片集合
            Expand(cfg.PIC_MEAN),  # 加大图片
            RandomSampleCrop(),
            RandomMirror(),
            # ToPercentCoords(), # 归一化尺寸
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

    def __call__(self, img, boxes=None, labels=None):
        # f_plt_show_cv(image,boxes)
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        '''默认是uint8'''
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels


class ToAbsoluteCoords(object):
    '''归一化尺寸转原图  ToAbsoluteCoords 相反'''

    def __call__(self, image, boxes=None, labels=None):
        '''
        归一化 -> 绝对坐标
        :param image:
        :param boxes:
        :param labels:
        :return:
        '''
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    '''原图转归一化尺寸  ToAbsoluteCoords 相反'''

    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=(320, 320)):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        w_ratio, h_ratio = np.array(image.shape[:2][::-1]) / np.array(self.size)
        if boxes is not None:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / w_ratio
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / h_ratio
        image = cv2.resize(image, (self.size[1], self.size[0]))
        return image, boxes, labels


class RandomSaturation(object):
    '''随机色彩'''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    '''随机颜色打乱'''

    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    '''bgr -> hsv'''

    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    '''随机透明度'''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        # f_plt_show_cv(image)
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        '''随机亮度增强'''
        if random.randint(2):  # 50%
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        # f_plt_show_cv(image)
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __init__(self, is_box_oned=True) -> None:
        super().__init__()
        self.is_box_oned = is_box_oned

    def __call__(self, cvimage, boxes=None, labels=None):
        if boxes is not None and self.is_box_oned:
            whwh = np.tile(cvimage.shape[:2][::-1], 2)
            boxes[:, :] = boxes[:, :] / whwh
        # (h,w,c -> c,h,w) = bgr
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


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

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

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
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

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
                current_boxes = _copy_box(boxes[mask, :])

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    '''随机扩大'''

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

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
        # boxes = boxes.copy()
        boxes = _copy_box(boxes)

        if isinstance(boxes, np.ndarray):
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))
        elif isinstance(boxes, torch.Tensor):
            boxes[:, :2] += torch.tensor((int(left), int(top)))
            boxes[:, 2:] += torch.tensor((int(left), int(top)))
        else:
            raise Exception('类型错误', type(boxes))

        # f_plt_show_cv(image,torch.tensor(boxes))
        return image, boxes, labels


class RandomMirror(object):
    '''随机水平镜像'''

    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            # boxes = boxes.copy()
            boxes = _copy_box(boxes)
            # boxes[:, 0::2] = width - boxes[:, 2::-2]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return image, boxes, classes


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
            RandomContrast(),
            ConvertColor(transform='HSV'),  # bgr -> hsv
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),  # hsv -> bgr
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        # self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):  # 先转换还是后转换
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels
        # return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
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

    def __call__(self, img, boxes, labels):
        '''

        :param img:
        :param boxes:
        :param labels:
        :return:
        '''
        return self.augment(img, boxes, labels)
