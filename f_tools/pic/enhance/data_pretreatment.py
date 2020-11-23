import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms
from torchvision.transforms import functional as F, transforms

from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_boxes import calc_iou4ts
from f_tools.pic.f_show import show_bbox_keypoints4pil, show_bbox4pil
from f_tools.pic.f_size_handler import resize_img_keep_np


def _show(img_ts, target, cfg, name):
    flog.debug('%s 后', name)
    img_pil = transforms.ToPILImage('RGB')(img_ts)
    if 'keypoints' in target:
        concatenate = np.concatenate([target['boxes'], target['keypoints']], axis=1)
        concatenate[:, ::2] = concatenate[:, ::2] * cfg.IMAGE_SIZE[0]
        concatenate[:, 1::2] = concatenate[:, 1::2] * cfg.IMAGE_SIZE[1]
        show_bbox_keypoints4pil(
            img_pil,
            concatenate[:, :4],
            concatenate[:, 4:14],
            target['labels'])
    else:
        boxes = target['boxes']
        if isinstance(boxes, np.ndarray):
            bboxs_ = np.copy(target['boxes'])
        elif isinstance(boxes, torch.Tensor):
            bboxs_ = torch.clone(target['boxes'])
        else:
            raise Exception('类型错误', type(boxes))
        bboxs_[:, ::2] = bboxs_[:, ::2] * cfg.IMAGE_SIZE[0]
        bboxs_[:, 1::2] = bboxs_[:, 1::2] * cfg.IMAGE_SIZE[1]
        show_bbox4pil(img_pil, bboxs_, target['labels'])


class BasePretreatment:

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg


class Compose(BasePretreatment):
    """组合多个transform函数"""

    def __init__(self, transforms, cfg):
        super(Compose, self).__init__(cfg)
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target, self.cfg)
        return image, target


class ResizeKeep():
    def __init__(self, newsize):
        '''

        :param newsize: (h, w)
        '''
        self.newsize = newsize

    def __call__(self, img_pil, target, cfg):
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
            if 'keypoints' in target:
                target['keypoints'] = target['keypoints'] * ratio
                if cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
                    flog.debug('ResizeKeep 后%s')
                    show_bbox_keypoints4pil(
                        img_pil,
                        target['boxes'],
                        target['keypoints'],
                        target['labels'])
            elif cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
                show_bbox4pil(img_pil, target['boxes'], target['labels'])
        return img_pil, target


class Resize(object):
    """对图像进行 resize 处理 比例要变"""

    def __init__(self, size):
        '''

        :param size: (h, w)
        '''
        self.resize = torchvision.transforms.Resize(size)

    def __call__(self, img_pil, target, cfg):
        if cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
            flog.debug('显示原图 %s %s', img_pil.size, target['boxes'].shape)
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
                if cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
                    flog.debug('缩放后%s', img_pil.size)
                    show_bbox_keypoints4pil(img_pil, bbox, keypoints, target['labels'])
            elif cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
                show_bbox4pil(img_pil, bbox, target['labels'])

        return img_pil, target


class RandomHorizontalFlip4TS(object):
    """随机水平翻转图像以及bboxes,该方法应放在ToTensor后"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_ts, target, cfg):
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

                if cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
                    _show(img_ts, target, cfg, 'RandomHorizontalFlip4TS')
        return img_ts, target


class RandomHorizontalFlip4PIL(object):
    """
    图片增强---PIL图像
    随机水平翻转图像以及bboxes
    输入单张   dataset的输出  np img
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_pil, target, cfg):
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


class SSDCroppingPIL(object):
    """
    根据原文，对图像进行裁剪,该方法应放在ToTensor前
    Cropping for SSD, according to original paper
    Choose between following 3 conditions:
    1. Preserve the original image
    2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
    3. Random crop
    Reference to https://github.com/chauhan-utk/ssd.DomainAdaptation
    """

    def __init__(self):
        # 定义随机
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def __call__(self, img_pil, target, cfg):
        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)  # 选一个iou
            if mode is None:  # 不做随机裁剪处理
                return img_pil, target

            htot, wtot = target['height_width']

            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('+inf') if max_iou is None else max_iou

            # Implementation use 5 iteration to find possible candidate
            for _ in range(5):
                # 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w / h < 0.5 or w / h > 2:  # 保证宽高比例不要太大在0.5-2之间
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)  # 均匀分布
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                # boxes的坐标是在0-1之间的
                boxes = target["boxes"]
                size = target['height_width'].numpy()[::-1]  # torch没有反序
                size = torch.tensor(size.copy()).repeat(2)  # repeat 方法与np用法不一致
                boxes_ = boxes / size

                ious = calc_iou4ts(boxes_, torch.tensor([[left, top, right, bottom]]))

                # tailor all the bboxes and return
                # all(): Returns True if all elements in the tensor are True, False otherwise.
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (boxes_[:, 0] + boxes_[:, 2])
                yc = 0.5 * (boxes_[:, 1] + boxes_[:, 3])

                # 查找所有的gt box的中心点有没有在采样patch中的
                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes_, continue searching again
                # 如果所有的gt box的中心点都不在采样的patch中，则重新找
                if not masks.any():
                    continue

                # 修改采样patch中的所有gt box的坐标（防止出现越界的情况）
                boxes_[boxes_[:, 0] < left, 0] = left
                boxes_[boxes_[:, 1] < top, 1] = top
                boxes_[boxes_[:, 2] > right, 2] = right
                boxes_[boxes_[:, 3] > bottom, 3] = bottom

                # 虑除不在采样patch中的gt box
                boxes_ = boxes_[masks, :]
                # 获取在采样patch中的gt box的标签
                labels = target['labels']
                labels = labels[masks]

                # 裁剪patch
                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                img_pil = img_pil.crop((left_idx, top_idx, right_idx, bottom_idx))

                # 调整裁剪后的bboxes_坐标信息
                boxes_[:, 0] = (boxes_[:, 0] - left) / w
                boxes_[:, 1] = (boxes_[:, 1] - top) / h
                boxes_[:, 2] = (boxes_[:, 2] - left) / w
                boxes_[:, 3] = (boxes_[:, 3] - top) / h

                # 更新crop后的gt box坐标信息以及标签信息
                target['boxes'] = boxes_ * size
                target['labels'] = labels
                if cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
                    show_bbox4pil(img_pil, target['boxes'], target['labels'])
                return img_pil, target


class ToTensor(object):
    """
    将PIL图像转为Tensor
    """

    def __call__(self, img_pil, target, cfg):
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
            if cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
                _show(img_ts, target, cfg, 'ToTensor')
        return img_ts, target


class Normalization4TS(object):
    """------------- 根据img net 用于图片调整 输入 tensor --------------"""

    def __init__(self, mean: object = None, std: object = None) -> object:
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, img_ts, target, cfg):
        img_ts = self.normalize(img_ts)
        if cfg.IS_VISUAL and cfg.IS_VISUAL_PRETREATMENT:
            _show(img_ts, target, cfg, 'Normalization4TS')
            # 恢复测试
            # img_ts_show = f_recover_normalization4ts(img_ts)
            # _show(img_ts_show, target, cfg, 'Normalization4TS恢复')
        return img_ts, target


def f_recover_normalization4ts(img_ts):
    '''

    :param img_ts: c,h,w
    :return:
    '''
    img_ts_show = img_ts.permute(1, 2, 0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_ts_show = img_ts_show * std + mean
    img_ts_show = img_ts_show.permute(2, 0, 1)
    return img_ts_show


class ColorJitter(object):
    """对图像颜色信息进行随机调整,该方法应放在ToTensor前"""

    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img_pil, target, cfg):
        img_pil = self.trans(img_pil)
        # if cfg.IS_VISUAL:
        #     flog.debug('ColorJitter 后%s', img_pil.size)
        #     img_pil.show()
        return img_pil, target
