import cv2
import torch
import numpy as np

from f_tools.GLOBAL_LOG import flog


def resize_img_keep_np(img_np, new_size, mode='lt', fill_color=(114, 114, 114)):
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
    return img_np, ratio, old_size, (left, top, right, bottom)


def resize_img_tensor(image, new_size):
    '''
    将图片缩放到指定的大小范围内，并处理对应缩放bboxes信息
    :param image: 一张已toTensor图片 c,h,w torch.Size([3, 335, 500])
    :param new_size: [w,h]
    :param target: 一个target对象
    :return: tensor c,h,w torch.Size([3, new_size])
    '''
    # ------确定一个缩放因子 以最大尺寸为优先, 缩放图片到指定的宽高--------
    # image shape is [channel, height, width]
    hw = [new_size[1], new_size[0]]
    old_size = image.shape[-2:]  # (800 600) 这个取出来不是tensor
    ratio = min(np.array(hw) / old_size)

    if ratio > 1.0:
        image = torch.nn.functional.interpolate(
            image[None], size=hw,
            mode='bilinear', align_corners=False)
    else:
        image = torch.nn.functional.interpolate(
            image[None], size=hw,
            mode='nearest')

    return image.squeeze(0), old_size


if __name__ == '__main__':
    file_pic = r'D:\tb\tb\ai_code\DL\_test_pic\2007_006046.jpg'

    # img = cv2.imread(file_img)
    from PIL import Image
    from torchvision.transforms import functional as F

    # 直接拉伸
    img_pil = Image.open(file_pic).convert('RGB')
    # interpolation = cv2.INTER_LINEAR if ratio > 1.0 else cv2.INTER_AREA
    # img_pil = img_pil.resize((300, 600), Image.ANTIALIAS)
    # img_pil.show()

    ''' h,w,c(brg) '''
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_np, ratio, old_size, (left, top, right, bottom) = resize_img_keep_np(img_np, (600, 400), mode='center')
    print(ratio, old_size, (left, top, right, bottom))
    # img_np = cv2.imread(file_pic)  # 读取原始图像
    cv2.imshow("img", img_np)  # 显示只支持BGR
    cv2.waitKey(0)

    img_tensor = F.to_tensor(img_pil)  # Image w,h -> c,h,w +归一化
    img_tensor, old_size = resize_img_tensor(img_tensor, (600, 200))
    img_pil = F.to_pil_image(img_tensor).convert('RGB')  # 这个会还原
    img_pil.show()
