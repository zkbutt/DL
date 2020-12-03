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
                  fill=STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)],
                  # fill='White',
                  )
        # __draw_text(draw, labels[i], box, l, r, t, b, STANDARD_COLORS)
    pil_copy.show()
    # plt.imshow(img_pil)
    # plt.show()


def show_bbox4ts(img_ts, boxs, labels=None):
    img_pil = transforms.ToPILImage(mode="RGB")(img_ts)
    show_bbox4pil(img_pil, boxs.numpy(), labels)


def show_anc4pil(img_pil, anc, size=(1, 1)):
    # _clone = anc[:300, :].clone()
    if isinstance(anc, np.ndarray):
        _clone = anc.copy()
    elif isinstance(anc, torch.Tensor):
        _clone = anc.clone()
    else:
        raise Exception('类型错误', type(anc))
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
    '''

    :param img_ts: torch.Size([3, 416, 416])
    :param anc: torch.Size([196, 25]) ltrb
    :param size: (w,h)
    :return:
    '''
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


def f_show_od4pil_yolo(img_pil, p_yolo, id_to_class=None):
    '''

    :param img_pil:
    :param p_yolo:  按实尺寸的框 n,25
    :param id_to_class:
    :return:
    '''
    boxes_confs = p_yolo[:, :5]
    _, labels_ts = p_yolo[:, 5:].max(dim=1)
    labels = list(labels_ts.numpy() + 1)
    f_show_od4pil(img_pil, boxes_confs, labels, id_to_class)


def f_show_od4pil(img_pil, boxes_confs, labels, id_to_class=None, font_size=10, text_fill=True):
    '''
    需扩展按conf排序  或conf过滤的功能
    :param img_pil: 一张 pil
    :param boxes_confs: np(9, 5)  前4 bbox  后1 conf 已按实尺寸的框
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


def f_show_iou4plt(box1, box2):
    '''
    归一化的 BBOX plt画图 归一画图
    :param box1: ltwh
    :param box2: ltwh
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.ylim(-0.5, 1.5)
    plt.ylim(0, 1)  # 量程
    plt.xlim(0, 1)
    ax.invert_yaxis()  # y轴反向 否则是反的
    plt.grid(linestyle='-.')
    # plt.scatter(ww.reshape(-1), hh.reshape(-1))
    for b in box1:
        plt.text(b[0], b[1], 'GT', color='r')
        _rect = plt.Rectangle((b[0], b[1]), b[2], b[3], color='r', fill=False)
        ax.add_patch(_rect)
    for i, b in enumerate(box2):
        plt.text(b[0], b[1], 'P%s' % i, color=STANDARD_COLORS[i])
        _rect = plt.Rectangle((b[0], b[1]), b[2], b[3], color=STANDARD_COLORS[i], fill=False)
        ax.add_patch(_rect)
    plt.show()


def _draw_box4pil(draw, boxes, color=None, width=4):
    from f_tools.fun_od.f_boxes import ltrb2xywh
    boxes_ltrb = ltrb2xywh(boxes)
    if color is not None:
        for c in boxes_ltrb:
            # draw.point(c[:2].numpy().tolist(), fill = color)
            x, y = c[:2]
            r = 4
            # 空心
            # draw.arc((x - r, y - r, x + r, y + r), 0, 360, fill=color)
            # 实心
            draw.chord((x - r, y - r, x + r, y + r), 0, 360, fill=color)

    for c in boxes:
        if isinstance(c, np.ndarray):
            l, t, r, b = c.astype(np.int)
        elif isinstance(c, torch.Tensor):
            l, t, r, b = c.type(torch.int)
        else:
            raise Exception('类型错误', type(c))
        if color is None:
            _color = STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)]
        else:
            _color = color
        draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=width, fill=_color)
    return draw


def f_show_iou4pil(img_pil, g_boxes=None, boxes2=None, grids=None, is_safe=True):
    '''

    :param img_pil:
    :param g_boxes: ltrb 归一化尺寸
    :param boxes2:归一化尺寸
    :return:
    '''
    if is_safe:
        pil_copy = img_pil.copy()
    else:
        pil_copy = img_pil
    draw = ImageDraw.Draw(pil_copy)
    whwh = np.array(img_pil.size).repeat(2, axis=0)  # 单体复制

    if g_boxes is not None:
        _box1 = g_boxes.clone()
        _box1 = _box1 * whwh
        _draw_box4pil(draw, _box1, color='Yellow')
    if boxes2 is not None:
        _box2 = boxes2.clone()
        _box2 = _box2 * whwh
        _draw_box4pil(draw, _box2, width=2)
    if grids is not None:
        _draw_grid4pil(draw, img_pil.size, grids)
    pil_copy.show()


def _draw_grid4pil(draw, size, grids=(7, 7)):
    colors_ = STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)]
    w, h = size
    xys = np.array([w, h]) / grids
    off_x = np.arange(1, grids[0])
    off_y = np.arange(1, grids[1])
    xx = off_x * xys[0]
    yy = off_y * xys[1]
    for x_ in xx:
        # 画列
        draw.line([x_, 0, x_, h], width=2, fill=colors_)
    for y_ in yy:
        # 画列
        draw.line([0, y_, w, y_], width=2, fill=colors_)


def f_show_grid4pil(img_pil, grids=(7, 7)):
    pil_copy = img_pil.copy()
    draw = ImageDraw.Draw(img_pil)
    _draw_grid4pil(draw, img_pil.size, grids)
    pil_copy.show()


if __name__ == '__main__':
    file_pic = r'D:\tb\tb\ai_code\DL\_test_pic\2007_006046.jpg'
    file_pic = r'D:\tb\tb\ai_code\DL\_test_pic\2007_000121.jpg'

    # img = cv2.imread(file_img)
    from PIL import Image
    from torchvision.transforms import functional as F

    # 直接拉伸 pil打开为RGB
    img_pil = Image.open(file_pic).convert('RGB')

    f_show_grid4pil(img_pil, (20, 5))
