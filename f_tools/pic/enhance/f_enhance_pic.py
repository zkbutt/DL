import os
import random

import torch
from PIL import Image, ImageEnhance, ImageDraw
from cv2 import cv2
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

from f_tools.f_general import rand

'''
np处理
    https://blog.csdn.net/caihh2017/article/details/85789443
'''


def pic_generator_keras(path, s_num, batch_size=1):
    datagen = ImageDataGenerator(
        rotation_range=10,  # 旋转范围
        width_shift_range=0.1,  # 水平平移范围
        height_shift_range=0.1,  # 垂直平移范围
        shear_range=0.2,  # 透视变换的范围
        zoom_range=0.1,  # 缩放范围
        horizontal_flip=False,  # 水平反转
        brightness_range=[0.1, 2],  # 图像随机亮度增强，给定一个含两个float值的list，亮度值取自上下限值间
        fill_mode='nearest'  # 输入边界以外的点根据给定的模式填充 ‘constant’，‘nearest’，‘reflect’或‘wrap’
    )

    pic_names = os.listdir(path)  # 读取子目录 和文件

    path_out = os.path.join(path, 'train_out')
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    for index, name in enumerate(pic_names):
        path_pic_in = os.path.join(path, name)
        if os.path.isdir(path_pic_in):
            continue

        print('path_pic_in', path_pic_in)
        img = load_img(path_pic_in)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        for i, pic_np in enumerate(datagen.flow(x, batch_size=batch_size,  # 一次输出几个图
                                                save_to_dir=path_out,
                                                save_prefix=str(index),
                                                save_format='jpg')):
            # print(i, pic_np)
            if i >= s_num:
                break
    print('---------------图片生成完成--------------------')


def Enhance_Brightness_pil(image):
    # 变亮，增强因子为0.0将产生黑色图像,为1.0将保持原始图像。
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = np.random.uniform(0.6, 1.6)
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def Enhance_Color_pil(image):
    # 色度,增强因子为1.0是原始图像
    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = np.random.uniform(0.4, 2.6)
    image_colored = enh_col.enhance(color)
    return image_colored


def Enhance_contrasted_pil(image):
    # 对比度，增强因子为1.0是原始图片
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = np.random.uniform(0.6, 1.6)
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def Enhance_sharped_pil(image):
    # 锐度，增强因子为1.0是原始图片
    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = np.random.uniform(0.4, 4)
    image_sharped = enh_sha.enhance(sharpness)
    return image_sharped


def Add_pepper_salt_pil(image):
    # 增加椒盐噪声
    img = np.array(image)
    rows, cols, _ = img.shape

    random_int = np.random.randint(500, 1000)
    for _ in range(random_int):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        if np.random.randint(0, 2):
            img[x, y, :] = 255
        else:
            img[x, y, :] = 0
    img = Image.fromarray(img)
    return img


def mem_enhance(image_path, change_bri=1, change_color=1, change_contras=1, change_sha=1, add_noise=1):
    # 读取图片
    image = Image.open(image_path)

    if change_bri == 1:
        image = Enhance_Brightness_pil(image)
    if change_color == 1:
        image = Enhance_Color_pil(image)
    if change_contras == 1:
        image = Enhance_contrasted_pil(image)
    if change_sha == 1:
        image = Enhance_sharped_pil(image)
    if add_noise == 1:
        image = Add_pepper_salt_pil(image)
    # image.save("0.jpg")
    return image  # 返回 PIL.Image.Image 类型


def pic_resize_keep_np(img_np, size):
    h, w, _ = img_np.shape
    if h > w:
        padw = (h - w) // 2
        img_np = np.pad(img_np, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
    elif w > h:
        padh = (w - h) // 2
        img_np = np.pad(img_np, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
    img_np = cv2.resize(img_np, (size[0], size[1]))
    return img_np


class Enhance4np:
    def __init__(self) -> None:
        self.prob = random.random() < 0.5

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        '''

        :param bgr:
        :return:
        '''
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        '''
        随机饱和度
        :param bgr:
        :return:
        '''
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        '''
        随机色调
        :param bgr:
        :return:
        '''
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        '''
         随机模糊
        '''
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        '''
         # 固定住高度，以0.6-1.4伸缩宽度，做图像形变
        :param bgr:
        :param boxes:
        :return:
        '''
        if random.random() < 0.5:
            scale = random.uniform(0.6, 1.4)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        '''
        self.mean = (123,117,104)
        :param bgr:
        :param mean:
        :return:
        '''
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        '''
        随机翻转
        '''
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


def c_hsv_np(img_np, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    '''
    随机色域 is_safe
    :param img_np:
    :param h_gain:
    :param s_gain:
    :param v_gain:
    :return:
    '''
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # 0.5~1.5之间
    hue, sat, val = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV))
    dtype = img_np.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img_np)  # no return needed


def random_horizontal_flip_np(img_np, bboxes):
    # 水平
    if random.random() < 0.5:
        _, w, _ = img_np.shape
        # [::-1] 顺序相反操作
        # a = [1, 2, 3, 4, 5]
        # a[::-1]
        # Out[3]: [5, 4, 3, 2, 1]
        img_np = img_np[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return img_np, bboxes


def random_crop_np(img_np, bboxes):
    # 随机剪裁
    if random.random() < 0.5:
        h, w, _ = img_np.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        img_np = img_np[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return img_np, bboxes


def random_translate_np(img_np, bboxes):
    # 旋转
    if random.random() < 0.5:
        h, w, _ = img_np.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        img_np = cv2.warpAffine(img_np, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return img_np, bboxes


if __name__ == '__main__':
    file_pic = r'D:\tb\tb\ai_code\DL\_test_pic\2007_006046.jpg'

    # img_pil = Image.open(file_pic)

    # img_np 测试
    img_np = cv2.imread(file_pic)  # 读取原始图像

    c_hsv_np(img_np)
    img_pil = Image.fromarray(img_np, mode="RGB")
    img_pil.show()
    # cv2.imshow("img", img_np)
    # cv2.waitKey(0)
