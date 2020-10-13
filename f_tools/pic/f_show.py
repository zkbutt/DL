import collections
import random

import cv2
import pylab
import torch
import torchvision
from PIL import ImageDraw, ImageFont, Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from f_tools.GLOBAL_LOG import flog

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def __filter_low_thresh(boxes, scores, classes, category_index,
                        difficult, thresh, box_to_display_str_map,
                        box_to_color_map):
    for i in range(boxes.shape[0]):
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            display_str = '{}: {}%-{}'.format(display_str, int(100 * scores[i]), difficult[i])
            print(display_str)
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = STANDARD_COLORS[
                classes[i] % len(STANDARD_COLORS)]
        else:
            break  # 网络输出概率已经排序过，当遇到一个不满足后面的肯定不满足


def __draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color):
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in box_to_display_str_map[box]]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in box_to_display_str_map[box][::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_box(image, boxes, classes, scores, category_index, difficult=None, thresh=0.5, line_thickness=8):
    flog.debug('最终额 %s 个目标框 ', (scores > thresh).sum())
    if not difficult:
        difficult = np.zeros(boxes.shape[0])
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

    __filter_low_thresh(boxes, scores, classes, category_index, difficult, thresh, box_to_display_str_map,
                        box_to_color_map)

    # Draw all boxes onto image.
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                      ymin * 1, ymax * 1)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=line_thickness, fill=color)
        __draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)


def show_od4dataet(train_data_set, num=5):
    '''
    随机选5张
    :param train_data_set:
    :param num:
    :return:
    '''
    for index in random.sample(range(0, len(train_data_set)), k=num):
        img, target = train_data_set[index]
        img = transforms.ToPILImage()(img)
        draw_box(img,
                 target['boxes'].numpy(),
                 target['labels'].numpy(),
                 [1 for i in range(len(target['labels'].numpy()))],  # 预测概率
                 dict((val, key) for key, val in train_data_set.class_dict.items()),  # 交换
                 target['difficult'].numpy(),
                 thresh=0.5,  # 阈值
                 line_thickness=5,  # 框宽度
                 )
        plt.imshow(img)
        plt.show()


def show_od_keypoints4np(img_np, bboxs, keypoints, scores):
    if isinstance(bboxs, torch.Tensor):
        bboxs = bboxs.numpy()
        keypoints = keypoints.numpy()
        scores = scores.numpy()
    for b, k, s in zip(bboxs, keypoints, scores):
        b = b.astype(np.int)
        k = k.astype(np.int)
        text = "{:.4f}".format(s)
        cv2.rectangle(img_np, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_np, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.circle(img_np, (k[0], k[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_np, (k[2], k[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_np, (k[4], k[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_np, (k[6], k[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_np, (k[8], k[9]), 1, (255, 0, 0), 4)
    # 远程无法显示
    # img_pil = Image.fromarray(img_np, mode="RGB")  # h,w,c
    # img_pil.show()
    show_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("after", show_image)
    cv2.waitKey(0)


def show_od4pil(img_pil, boxs, labels=None):
    '''

    :param img_np: tensor 或 img_pil
    :param boxs: np
    :return:
    '''
    if labels:
        flog.info('show_od4boxs %s', labels)
    # ----------恢复原图-------ToTensor
    draw = ImageDraw.Draw(img_pil)
    im_width, im_height = img_pil.size
    print(im_width, im_height)
    for box in boxs:
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                      ymin * 1, ymax * 1)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=4,
                  fill=STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)])
        # _draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)
    plt.imshow(img_pil)
    plt.show()


def __matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_pics_ts(images, labels=None, one_channel=True):
    '''

    :param images: tensors N,H,W,C
    :param labels:
    :return:
    '''
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    flog.debug('show_pics_ts %s', labels)
    if isinstance(images, (np.ndarray)):
        flog.debug(str(images.shape) + '%s', type(labels))  # N,H,W,C
        col = int(np.ceil(np.sqrt(images.shape[0])))
    elif isinstance(images, (list, tuple)):
        # dataload已处理这里条件不会生效
        if len(images) > 0:
            col = int(np.ceil(np.sqrt(len(images))))
            # 组装tensor list ts -> ts
            _t = images[0][None]
            for i in range(len(images) - 1):
                _t = torch.cat([_t, images[i + 1][None]], dim=0)
            images = _t
        else:
            raise Exception('images为空')
    else:
        col = 4
    # 组合
    img_grid = torchvision.utils.make_grid(images, nrow=col, padding=4)
    __matplotlib_imshow(img_grid, one_channel=one_channel)


def show_pic_ts(image, labels=None, one_channel=True, **kwargs):
    '''
    查看处理后的图片
    :param image:
    :param labels:
    :param one_channel:
    :param kwargs:
    :return:
    '''
    # show_pic_ts(images[0], targets[0], classes=train_data_set.class_dict)
    plt.figure(figsize=(12.80, 7.20))

    classes = kwargs.get('classes', None)
    if classes:
        classes_new = {v: k for k, v in classes.items()}
        s = ', '
        # classes字典是1开始
        label_names = [classes_new[i + 1] + ':' + str(lable.item()) for i, lable in enumerate(labels) if lable > 0]
        s = s.join(label_names)
        plt.title(s)

    flog.debug('show_pic_ts %s', labels)

    __matplotlib_imshow(image, one_channel=one_channel)


def show_pic_np(pic, is_torch=False, is_many=None):
    plt.figure(figsize=(12.80, 7.20))
    if is_torch:
        pic = np.transpose(pic, (1, 2, 0))  # H W C
    if is_many:
        plt.subplot(*is_many)
    plt.imshow(pic)
    plt.colorbar()
    plt.grid(False)


def show_pics_np(pics, is_torch=False):
    plt.figure(figsize=(12.80, 7.20))
    for i, pic in enumerate(pics):
        col = int(np.ceil(np.sqrt(len(pics))))
        row = int(np.ceil(len(pics) / col))
        show_pic_np(pic, is_torch=is_torch, is_many=(row, col, i + 1))
    plt.show()
