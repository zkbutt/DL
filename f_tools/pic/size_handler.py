import cv2
import torch
import numpy as np

from f_tools.GLOBAL_LOG import flog


def resize_keypoints4ratio(keypoints, ratio):
    print(keypoints)
    return keypoints

def resize_boxes4ratio(boxes, ratio):
    boxes[:, ::2] = boxes[:, ::2] * ratio
    boxes[:, 1::2] = boxes[:, 1::2] * ratio
    return boxes

def resize_boxes4np(boxes, original_size, new_size):
    '''
    用于预处理 和 最后的测试(预测还原) ltrb
    :param boxes: 输入多个
    :param original_size: 图像缩放前的尺寸
    :param new_size: 图像缩放后的尺寸
    :return:
    '''
    # 输出数组   新尺寸 /旧尺寸 = 对应 h w 的缩放因子 处理list
    ratios = [s / s_orig for s, s_orig in zip(new_size, original_size)]
    ratios_height, ratios_width = ratios
    boxes[:, ::2] = boxes[:, ::2] * ratios_width
    boxes[:, 1::2] = boxes[:, 1::2] * ratios_height
    return boxes


def resize_boxes4tensor(boxes, original_size, new_size):
    '''
    用于预处理 和 最后的测试(预测还原) ltrb
    :param boxes: 输入多个
    :param original_size: 图像缩放前的尺寸
    :param new_size: 图像缩放后的尺寸
    :return:
    '''
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


def resize_img_keep_np(img_np, new_size, mode='lt', fill_color=(0, 0, 0)):
    '''
    
    :param path_img: 一张图片 np图片 h,w,c
    :param new_size: [h,w]
    :param mode: center lt
    :return:
        img, old_size list(h,w)
        (top, bottom, left, right) # 填充宽度
    '''
    assert isinstance(img_np, np.ndarray)
    old_size = img_np.shape[0:2]  # hw
    # flog.debug('原尺寸 %s', img.shape)
    # 选择比例小的
    ratio = min(float(new_size[i]) / (old_size[i]) for i in range(len(old_size)))
    # flog.debug('选择的缩放比例 %s', ratio)

    # 未填充时的尺寸需要保存下来
    h_new, w_new = tuple([int(i * ratio) for i in old_size])
    # flog.debug('新尺寸 %s %s', h_new, w_new)
    interpolation = cv2.INTER_LINEAR if ratio > 1.0 else cv2.INTER_AREA
    img_np = cv2.resize(img_np, (w_new, h_new), interpolation=interpolation)  # 大坑 W H

    # cv2.imshow('bb', img)
    # cv2.waitKey()

    # 图片填充
    _pad_w = new_size[1] - w_new
    _pad_h = new_size[0] - h_new

    top, bottom, left, right = 0, 0, 0, 0
    if mode == 'center':
        top, bottom = _pad_h // 2, _pad_h - (_pad_h // 2)
        left, right = _pad_w // 2, _pad_w - (_pad_w // 2)
    elif mode == 'lt':
        right = _pad_w
        bottom = _pad_h

    img_np = cv2.copyMakeBorder(img_np, top, bottom, left, right, cv2.BORDER_CONSTANT, None, fill_color)
    return img_np, ratio, old_size, (top, bottom, left, right)


def resize_img_tensor(image, new_size):
    '''
    将图片缩放到指定的大小范围内，并处理对应缩放bboxes信息
    :param image: 一张已toTensor图片 c,h,w torch.Size([3, 335, 500])
    :param new_size: [h,w]
    :param target: 一个target对象
    :return: tensor c,h,w torch.Size([3, new_size])
    '''
    # ------确定一个缩放因子 以最大尺寸为优先, 缩放图片到指定的宽高--------
    # image shape is [channel, height, width]

    old_size = image.shape[-2:]  # (800 600) 这个取出来不是tensor
    ratio = min(np.array(new_size) / old_size)

    if ratio > 1.0:
        image = torch.nn.functional.interpolate(
            image[None], size=new_size,
            mode='bilinear', align_corners=False)
    else:
        image = torch.nn.functional.interpolate(
            image[None], size=new_size,
            mode='nearest')

    return image.squeeze(0), old_size


if __name__ == '__main__':
    file_img = r'D:\tb\tb\ai_code\DL\_test_pic\2008_000329.jpg'
    # img = cv2.imread(file_img)
    from PIL import Image
    from torchvision.transforms import functional as F

    img_pil = Image.open(file_img).convert('RGB')
    # img_pil = img_pil.resize((300, 600), Image.ANTIALIAS)
    img_pil.thumbnail((800, 600), Image.ANTIALIAS)
    img_pil.show()

    # img_tensor = F.to_tensor(img_pil)  # Image w,h -> c,h,w +归一化
    # img_tensor = resize_img_tensor(img_tensor, (600, 200))
    # img_pil = F.to_pil_image(img_tensor).convert('RGB')  # 这个会还原
    # img_pil.resize()
    # img_pil.show()

    # img = img.repeat(2, 0)  # 2倍 在1维

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # h,w,c(brg)
    # img, ratio, (top, bottom, left, right) = resize_img_keep_ratio_np(img, (300, 400), mode='lt')
    # cv2.imshow('title', img)
    # key = cv2.waitKey(0)
    # print(ratio)
