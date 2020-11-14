import random

import cv2
import torch
from PIL import ImageDraw, ImageFont
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from f_tools.GLOBAL_LOG import flog

# 126种颜色
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


def show_od4np(img_np, bboxs, scores):
    if isinstance(bboxs, torch.Tensor):
        bboxs = bboxs.numpy()
        scores = scores.numpy()
    for b, s in zip(bboxs, scores):
        b = b.astype(np.int)
        text = "{:.4f}".format(s)
        cv2.rectangle(img_np, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # (l,t),(r,b),颜色.宽度
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_np, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # 远程无法显示
    # img_pil = Image.fromarray(img_np, mode="RGB")  # h,w,c
    # img_pil.show()
    show_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("after", show_image)
    cv2.waitKey(0)


def show_bbox4pil(img_pil, boxs, labels=None):
    '''
    https://blog.csdn.net/qq_36834959/article/details/79921152?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-1-79921152.nonecase&utm_term=imagedraw%E7%94%BB%E7%82%B9%E7%9A%84%E5%A4%A7%E5%B0%8F%20pil&spm=1000.2123.3001.4430
    :param img_np: tensor 或 img_pil
    :param boxs: np ltrb
    :return:
    '''
    if labels is not None:
        flog.info('%s', labels)
    pil_copy = img_pil.copy()
    draw = ImageDraw.Draw(pil_copy)
    # im_width, im_height = pil_copy.size
    # print(im_width, im_height)
    if isinstance(boxs, torch.Tensor):
        boxs = boxs.numpy()
    for i, box in enumerate(boxs):
        l, t, r, b = box.astype(np.int)
        # 创建一个正方形。 [x1,x2,y1,y2]或者[(x1,x2),(y1,y2)]  fill代表的为颜色
        draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=1,
                  # fill=STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)],
                  fill='White',
                  )
        # __draw_text(draw, labels[i], box, l, r, t, b, STANDARD_COLORS)
    pil_copy.show()
    # plt.imshow(img_pil)
    # plt.show()


def show_bbox4ts(img_ts, boxs, labels=None):
    img_pil = transforms.ToPILImage(mode="RGB")(img_ts)
    show_bbox4pil(img_pil, boxs.numpy(), labels)


def show_anc4pil(img_pil, anc, size):
    # _clone = anc[:300, :].clone()
    _clone = anc.clone()
    _clone[:, ::2] = _clone[:, ::2] * size[0]
    _clone[:, 1::2] = _clone[:, 1::2] * size[1]
    draw = ImageDraw.Draw(img_pil)
    for c in _clone:
        if isinstance(c, np.ndarray):
            l, t, r, b = c.astype(np.int)
        elif isinstance(c, torch.Tensor):
            l, t, r, b = c.type(torch.int)
        else:
            raise Exception('类型错误', type(c))
        draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=4,
                  fill=STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)])
    img_pil.show()


def show_anc4ts(img_ts, anc, size):
    img_pil = transforms.ToPILImage()(img_ts)
    show_anc4pil(img_pil, anc, size)


def show_bbox_keypoints4pil(img_pil, bboxs, keypoints, scores=None):
    '''

    :param img_np: tensor 或 img_pil
    :param boxs: np l, t, r, b
    :return:
    '''
    if scores is None:
        flog.error('无分数 %s', scores)
        return
    pil_copy = img_pil.copy()
    draw = ImageDraw.Draw(pil_copy)
    cw = 3
    for bbox, k, s in zip(bboxs, keypoints, scores):
        if isinstance(bboxs, np.ndarray):
            l, t, r, b = bbox.astype(np.int)
        elif isinstance(bboxs, torch.Tensor):
            l, t, r, b = bbox.type(torch.int)
        else:
            raise Exception('类型错误', type(bboxs))

        draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=4,
                  fill=STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)])
        draw.chord((k[0] - cw, k[1] - cw, k[0] + cw, k[1] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        draw.chord((k[2] - cw, k[3] - cw, k[2] + cw, k[3] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        draw.chord((k[4] - cw, k[5] - cw, k[4] + cw, k[5] + cw), 0, 360, fill=(0, 0, 255), outline=(0, 255, 0))
        draw.chord((k[6] - cw, k[7] - cw, k[6] + cw, k[7] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        draw.chord((k[8] - cw, k[9] - cw, k[8] + cw, k[9] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        # draw.text((l, t), "hello", (0, 255, 0))
        # draw.point([(20, 20), (25, 25), (50, 50), (30, 30)], (0, 255, 0))
        # ltrb 角度顺时间 框色 填充色
        # draw.chord((0, 0, 3, 3), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
    pil_copy.show()
    # plt.imshow(img_pil)
    # plt.show()


def show_bbox_keypoints4ts(img_ts, bboxs, keypoints, scores=None):
    img_pil = transforms.ToPILImage()(img_ts)
    show_bbox_keypoints4pil(img_pil, bboxs.numpy(), keypoints.numpy(), scores=scores.numpy())


def show_pic_ts(img_ts, labels=None):
    '''
    查看处理后的图片
    :param img_ts:
    :param labels:
    :return:
    '''
    # show_pic_ts(images[0], targets[0], classes=train_data_set.class_dict)
    # plt.figure(figsize=(12.80, 7.20))
    flog.debug('labels %s', labels)
    np_img = img_ts.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def show_pic_np(pic, is_many=None):
    plt.figure(figsize=(12.80, 7.20))
    if is_many:
        plt.subplot(*is_many)
    plt.imshow(pic)
    plt.colorbar()
    plt.grid(False)


def show_pics_np(pics):
    plt.figure(figsize=(12.80, 7.20))
    for i, pic in enumerate(pics):
        col = int(np.ceil(np.sqrt(len(pics))))
        row = int(np.ceil(len(pics) / col))
        show_pic_np(pic, is_many=(row, col, i + 1))
    plt.show()


def show_pic_label_np(img_np, boxes, labels):
    '''

    :param img_np:
    :param boxes:
    :param labels: labels中文
    :return:
    '''
    h, w = img_np.shape[:2]
    print(w, h)
    for box, label in zip(boxes, labels):
        print(box, label)
        pt1 = (int(box[0] * w - box[2] * w / 2), int(box[1] * h - box[3] * h / 2))
        pt2 = (int(box[0] * w + box[2] * w / 2), int(box[1] * h + box[3] * h / 2))
        cv2.putText(img_np, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.rectangle(img_np, pt1, pt2, (0, 0, 255, 2))
    cv2.imshow("img", img_np)
    cv2.waitKey(0)


def f_show_od4pil(img_pil, boxes_confs, labels, id_to_class=None, font_size=10, text_fill=True):
    '''
    需扩展按conf排序  或conf过滤的功能
    :param img_pil: 一张 pil
    :param boxes_confs: np(9, 5)  前4 bbox  后1 conf
    :param labels: list(int)
    :param id_to_class: dict{id,name}
    :return:
    '''
    img_pil_copy = img_pil.copy()
    try:
        font = ImageFont.truetype('arial.ttf', font_size)  # 参数1：字体文件路径，参数2：字体大小
    except IOError:
        font = ImageFont.load_default()

    # print(len(STANDARD_COLORS))
    # color = random.randint(0, len(STANDARD_COLORS))
    for box, conf, label in zip(boxes_confs[:, :4], boxes_confs[:, 4], labels):
        left, top, right, bottom = box
        if id_to_class:
            show_text = 'type={}:{:.1%}'.format(id_to_class[label], conf)
        else:
            show_text = 'type={}:{:.1%}'.format(label, conf)
        text_width, text_height = font.getsize(show_text)
        margin = np.ceil(0.05 * text_height)
        # 超出屏幕判断
        if top > text_height:
            text_bottom = top
        else:
            text_bottom = bottom + text_height
        color = STANDARD_COLORS[label]

        draw = ImageDraw.Draw(img_pil_copy)
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        if text_fill:
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width + 2 * margin, text_bottom)], fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      show_text, fill='black', font=font)
        else:
            draw.text((left + margin, text_bottom - text_height - margin),
                      show_text, fill=color, font=font)
    img_pil_copy.show()
