import random
import numpy as np
from PIL import Image
import torchvision.transforms
from torchvision.transforms import functional as F, transforms

from f_tools.GLOBAL_LOG import flog
from f_tools.pic.f_show import show_od_keypoints4pil
from f_tools.pic.size_handler import resize_img_keep_np
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG


def _show(img_ts, target, name):
    flog.debug('%s 后', name)
    concatenate = np.concatenate([target['boxes'], target['keypoints']], axis=1)
    concatenate[:, ::2] = concatenate[:, ::2] * CFG.IMAGE_SIZE[0]
    concatenate[:, 1::2] = concatenate[:, 1::2] * CFG.IMAGE_SIZE[1]
    img_pil = transforms.ToPILImage()(img_ts)
    show_od_keypoints4pil(
        img_pil,
        concatenate[:, :4],
        concatenate[:, 4:14],
        target['labels'])


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ResizeKeep():
    def __init__(self, newsize):
        '''

        :param newsize: (h, w)
        '''
        self.newsize = newsize

    def __call__(self, img_pil, target):
        '''

        :param img_pil: PIL图片
        :param target:
        :return:
        '''
        img_np = np.array(img_pil)
        img_np, ratio, old_size, _ = resize_img_keep_np(img_np, self.newsize)
        img_pil = Image.fromarray(img_np, mode="RGB")

        if target:
            target['boxes'] = target['boxes'] * ratio
            target['keypoints'] = target['keypoints'] * ratio
            if CFG.IS_VISUAL:
                flog.debug('ResizeKeep 后%s')
                show_od_keypoints4pil(
                    img_pil,
                    target['boxes'],
                    target['keypoints'],
                    target['labels'])
        return img_pil, target


class Resize(object):
    """对图像进行 resize 处理 比例要变"""

    def __init__(self, size):
        '''

        :param size: (h, w)
        '''
        self.resize = torchvision.transforms.Resize(size)

    def __call__(self, img_pil, target):
        if CFG.IS_VISUAL:
            flog.debug('显示原图 %s', img_pil.size)
            img_pil.show()
        w, h = img_pil.size  # PIL wh
        h_ratio, w_ratio = np.array([h, w]) / self.resize.size  # hw
        img_pil = self.resize(img_pil)

        if target:
            bbox = target['boxes']
            bbox[:, [0, 2]] = bbox[:, [0, 2]] / w_ratio
            bbox[:, [1, 3]] = bbox[:, [1, 3]] / h_ratio

            if 'keypoints' in target:
                keypoints = target['keypoints']
                keypoints[:, ::2] = keypoints[:, ::2] / w_ratio
                keypoints[:, 1::2] = keypoints[:, 1::2] / h_ratio
                if CFG.IS_VISUAL:
                    flog.debug('缩放后%s', img_pil.size)
                    show_od_keypoints4pil(img_pil, bbox, keypoints, target['labels'])
        return img_pil, target


class RandomHorizontalFlip4TS(object):
    """随机水平翻转图像以及bboxes,该方法应放在ToTensor后"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_ts, target):
        if random.random() < self.prob:
            # height, width = image.shape[-2:]
            img_ts = img_ts.flip(-1)  # 水平翻转图片

            if target:
                bbox = target['boxes']
                bbox[:, [2, 0]] = 1.0 - bbox[:, ::2]  # 翻转对应bbox坐标信息
                target['boxes'] = bbox
                if 'keypoints' in target:
                    keypoints = target['keypoints']
                    _t = np.all(keypoints > 0, axis=1)  # 全有效的行
                    ids = np.nonzero(_t)  # 获取 需更新的行索引      6和8是0
                    # ids = np.arange(0, len(keypoints[0]), 2)
                    # keypoints[ids, ::2] = width - keypoints[ids, ::2]
                    keypoints[ids, ::2] = 1.0 - keypoints[ids, ::2]

                    # if CFG.IS_VISUAL:
                    #     _show(img_ts, target, 'RandomHorizontalFlip4TS')

        return img_ts, target


class RandomHorizontalFlip4PIL(object):
    """
    图片增强---PIL图像
    随机水平翻转图像以及bboxes
    输入单张   dataset的输出  np img
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_pil, target):
        '''

        :param img_pil:
        :param target: 支持ltrb
        :return:
        '''
        if random.random() < self.prob:
            # height, width = image.shape[-2:]
            # image = image.flip(-1)  # 水平翻转图片

            width, height = img_pil.size
            # flog.debug('width,height %s,%s', width, height)
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转图片

            if target:
                bbox = target['boxes']
                # bbox: ltrb
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息

                if 'keypoints' in target:
                    keypoints = target['keypoints']
                    _t = np.all(keypoints > 0, axis=1)  # 全有效的行
                    ids = np.nonzero(_t)  # 获取 需更新的行索引      6和8是0
                    # ids = np.arange(0, len(keypoints[0]), 2)
                    keypoints[ids, ::2] = width - keypoints[ids, ::2]
                    # if CFG.IS_VISUAL:
                    #     flog.debug('RandomHorizontalFlip4PIL 后%s')
                    #     show_od_keypoints4pil(
                    #         img_pil,
                    #         target['boxes'],
                    #         target['keypoints'],
                    #         target['labels'])
        return img_pil, target


class ToTensor(object):
    """
    将PIL图像转为Tensor
    """

    def __call__(self, img_pil, target):
        w, h = img_pil.size  # PIL wh
        img_ts = F.to_tensor(img_pil)  # 将PIL图片hw 转tensor c,h,w 且归一化
        if target:
            bbox = target['boxes']

            bbox[:, [0, 2]] = bbox[:, [0, 2]] / w
            bbox[:, [1, 3]] = bbox[:, [1, 3]] / h

            if 'keypoints' in target:
                keypoints = target['keypoints']
                keypoints[:, ::2] = keypoints[:, ::2] / w
                keypoints[:, 1::2] = keypoints[:, 1::2] / h
                # if CFG.IS_VISUAL:
                #     _show(img_ts, target, 'ToTensor')
        return img_ts, target


class Normalization4TS(object):
    """------------- 根据img net 用于图片调整 输入 tensor --------------"""

    def __init__(self, mean: object = None, std: object = None) -> object:
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, img_ts, target):
        img_ts = self.normalize(img_ts)
        # if CFG.IS_VISUAL:
        #     _show(img_ts, target, 'Normalization4TS')

        return img_ts, target


class ColorJitter(object):
    """对图像颜色信息进行随机调整,该方法应放在ToTensor前"""

    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img_pil, target):
        img_pil = self.trans(img_pil)
        # if CFG.IS_VISUAL:
        #     flog.debug('ColorJitter 后%s', img_pil.size)
        #     img_pil.show()
        return img_pil, target
