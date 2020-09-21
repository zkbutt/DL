import cv2
import torch
import numpy as np
from f_tools.GLOBAL_LOG import flog


def resize_img_keep_ratio(img, target_size, mode='center', v_fill=(0, 0, 0)):
    '''
    img_new, ratio = resize_img_keep_ratio(path_img, target_size, 'upleft', [125, 21, 23])
    opencv是BGR
    :param path_img: np图片
    :param target_size:
    :param mode: center upleft
    :return:
    '''
    assert isinstance(img, np.ndarray)
    # img = cv2.imread(path_img)
    # if img:
    #     raise Exception('图片加载出错 : ',path_img)
    old_size = img.shape[0:2]
    flog.debug('原尺寸 %s', img.shape)

    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    flog.debug('缩放比例 %s', ratio)

    # 未填充时的尺寸需要保存下来
    h_new, w_new = tuple([int(i * ratio) for i in old_size])
    flog.debug('新尺寸 %s %s', h_new, w_new)
    interpolation = cv2.INTER_LINEAR if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (w_new, h_new), interpolation=interpolation)  # 大坑 W H

    # cv2.imshow('bb', img)
    # cv2.waitKey()

    _pad_w = target_size[1] - w_new
    _pad_h = target_size[0] - h_new

    top, bottom, left, right = 0, 0, 0, 0
    if mode == 'center':
        top, bottom = _pad_h // 2, _pad_h - (_pad_h // 2)
        left, right = _pad_w // 2, _pad_w - (_pad_w // 2)
    elif mode == 'upleft':
        right = _pad_w
        bottom = _pad_h

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, v_fill)
    return img, ratio, (top, bottom, left, right)


def resize_boxes4np(boxes, ratio, adjusts):
    '''
    [xmin, ymin, xmax, ymax]  VS top, bottom, left, right
    :param boxes: np  (n,4)多个选框
    :param ratio:
    :return:
    '''
    boxes = boxes * ratio
    flog.debug('adjusts%s', adjusts)
    boxes[:, 1] += adjusts[0]
    boxes[:, 3] += adjusts[0]
    boxes[:, 0] += adjusts[2]
    boxes[:, 2] += adjusts[2]
    # boxes[:, 0] += 50 # xmin
    # boxes[:, 1] += 50 #  ymin
    # boxes[:, 2] += 100 # xmax
    # boxes[:, 3] += adjusts[2] #  ymax
    # flog.debug("boxes %s", boxes)
    return boxes

def resize_boxes4tensro(boxes, original_size, new_size):
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
