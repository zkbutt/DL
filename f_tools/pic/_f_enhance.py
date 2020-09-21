import os

from PIL import Image, ImageEnhance, ImageDraw
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np


def pic_generator(path, s_num, batch_size=1):
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


def Enhance_Brightness(image):
    # 变亮，增强因子为0.0将产生黑色图像,为1.0将保持原始图像。
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = np.random.uniform(0.6, 1.6)
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def Enhance_Color(image):
    # 色度,增强因子为1.0是原始图像
    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = np.random.uniform(0.4, 2.6)
    image_colored = enh_col.enhance(color)
    return image_colored


def Enhance_contrasted(image):
    # 对比度，增强因子为1.0是原始图片
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = np.random.uniform(0.6, 1.6)
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def Enhance_sharped(image):
    # 锐度，增强因子为1.0是原始图片
    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = np.random.uniform(0.4, 4)
    image_sharped = enh_sha.enhance(sharpness)
    return image_sharped


def Add_pepper_salt(image):
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
        image = Enhance_Brightness(image)
    if change_color == 1:
        image = Enhance_Color(image)
    if change_contras == 1:
        image = Enhance_contrasted(image)
    if change_sha == 1:
        image = Enhance_sharped(image)
    if add_noise == 1:
        image = Add_pepper_salt(image)
    # image.save("0.jpg")
    return image  # 返回 PIL.Image.Image 类型


from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()  # 分解成path和区域
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.7, 1.3)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


if __name__ == '__main__':
    # pic_generator(r'E:\datas\t01', 5)
    # print(type(mem_enhance(r'E:\datas\t01\dog.12490.jpg')))
    line = r"E:\datas\t01\dog.12490.jpg 738,279,815,414,0"
    image_data, box_data = get_random_data(line, [416, 416])
    left, top, right, bottom = box_data[0][0:4]
    img = Image.fromarray((image_data * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([left, top, right, bottom])
    img.show()
