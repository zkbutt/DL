from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
import random
import cv2
import torch
from PIL import ImageDraw, ImageFont
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from f_tools.GLOBAL_LOG import flog


# 126种颜色
def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


COLORS_ImageDraw = [
    'AliceBlue',  # 白色
    'Chartreuse',  # 绿色
    'Aqua',
    'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood',  # 橙色
    'CadetBlue', 'AntiqueWhite',
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
    'LimeGreen', 'Linen',  # 白色
    'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink',  # 粉红
    'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

COLORS_plt = {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    'black': '#000000',
    'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen': '#90EE90',
    'lightgray': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'}

COLOR_PBOXES = 'lightcyan'
COLOR_GBOXES = 'red'


def show_od_keypoints4cv(img_np, bboxs, keypoints, scores):
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


def show_anc4pil(img_pil, anc, size=(1, 1)):
    '''
    这个不安全
    :param img_pil:
    :param anc:  n,4
    :param size:
    :return:
    '''
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
                  fill=COLORS_ImageDraw[random.randint(0, len(COLORS_ImageDraw) - 1)])
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


def show_bbox_scores4pil(img_pil, bboxs, scores):
    '''

    :param img_pil:
    :param bboxs:
    :param scores:
    :return:
    '''
    pil_copy = img_pil.copy()
    draw = ImageDraw.Draw(pil_copy)
    cw = 3
    for bbox, k, s in zip(bboxs, scores):
        if isinstance(bboxs, np.ndarray):
            l, t, r, b = bbox.astype(np.int)
        elif isinstance(bboxs, torch.Tensor):
            l, t, r, b = bbox.type(torch.int)
        else:
            raise Exception('类型错误', type(bboxs))

        draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=4,
                  fill=COLORS_ImageDraw[random.randint(0, len(COLORS_ImageDraw) - 1)])
        # draw.text((l, t), "hello", (0, 255, 0))
        # draw.point([(20, 20), (25, 25), (50, 50), (30, 30)], (0, 255, 0))
        # ltrb 角度顺时间 框色 填充色
        # draw.chord((0, 0, 3, 3), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
    pil_copy.show()
    # plt.imshow(img_pil)
    # plt.show()


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
                  fill=COLORS_ImageDraw[random.randint(0, len(COLORS_ImageDraw) - 1)])
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


def f_plot_od4np(img_np, p_boxes, p_keypoints, p_scores):
    '''这个是直接修改 img_np'''
    for b, k, s in zip(p_boxes, p_keypoints, p_scores):
        b = list(b.type(torch.int64).numpy())
        k = list(k.type(torch.int64).numpy())
        text = "{:.4f}".format(s)
        cv2.rectangle(img_np, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # (l,t),(r,b),颜色.宽度
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_np, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.circle(img_np, (k[0], k[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_np, (k[2], k[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_np, (k[4], k[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_np, (k[6], k[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_np, (k[8], k[9]), 1, (255, 0, 0), 4)


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


def f_plot_od4pil(img_pil, boxes_ltrb, scores, labels, id_to_class=None, font_size=10, text_fill=True):
    '''
    显示预测结果
    :param img_pil:
    :param labels:list(int)  torch.tensor
    :param id_to_class: 支持dict + list ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',]
    :param font_size:
    :param text_fill:
    :return:
    '''
    if isinstance(labels, torch.Tensor):
        labels = labels.type(torch.int).tolist()

    boxes_confs = torch.cat([boxes_ltrb, scores[:, None]], dim=1)
    try:
        font = ImageFont.truetype('simhei.ttf', font_size, encoding='utf-8')  # 参数1：字体文件路径，参数2：字体大小
    except IOError:
        font = ImageFont.load_default()

    # print(len(STANDARD_COLORS))
    # color = random.randint(0, len(STANDARD_COLORS))
    for box, conf, label in zip(boxes_confs[:, :4], boxes_confs[:, 4], labels):
        left, top, right, bottom = box
        _s_text = '{}:{:.1%}'
        if id_to_class:
            show_text = _s_text.format(id_to_class[label], conf)
        else:
            show_text = _s_text.format(label, conf)
        flog.debug(show_text)
        text_width, text_height = font.getsize(show_text)
        margin = np.ceil(0.05 * text_height)
        # 超出屏幕判断
        if top > text_height:
            text_bottom = top
        else:
            text_bottom = bottom + text_height
        color = COLORS_ImageDraw[label]

        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        if text_fill:
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width + 2 * margin, text_bottom)], fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      show_text, fill='black', font=font)
        else:
            draw.text((left + margin, text_bottom - text_height - margin),
                      show_text, fill=color, font=font)
        # font = ImageFont.truetype('simhei.ttf', 30, encoding='utf-8')
        # draw.text((100, 100), '优秀, 哈哈', (0, 255, 255), font=font)
    return img_pil


def f_plot_od4pil_keypoints(img_pil, boxes_ltrb, keypoints,
                            scores, labels, id_to_class=None, font_size=10, text_fill=True):
    '''

    :param img_pil:
    :param labels:list(int)  torch.tensor
    :param id_to_class: 支持dict + list ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',]
    :param font_size:
    :param text_fill:
    :return:
    '''
    if isinstance(labels, torch.Tensor):
        labels = labels.type(torch.int).tolist()

    boxes_confs = torch.cat([boxes_ltrb, scores[:, None]], dim=1)
    try:
        font = ImageFont.truetype('simhei.ttf', font_size, encoding='utf-8')  # 参数1：字体文件路径，参数2：字体大小
    except IOError:
        font = ImageFont.load_default()

    # print(len(STANDARD_COLORS))
    # color = random.randint(0, len(STANDARD_COLORS))
    cw = 3
    for box, k, conf, label in zip(boxes_confs[:, :4], keypoints, boxes_confs[:, 4], labels):
        left, top, right, bottom = box
        _s_text = '{}:{:.1%}'
        if id_to_class:
            show_text = _s_text.format(id_to_class[label], conf)
        else:
            show_text = _s_text.format(label, conf)
        flog.debug(show_text)
        text_width, text_height = font.getsize(show_text)
        margin = np.ceil(0.05 * text_height)
        # 超出屏幕判断
        if top > text_height:
            text_bottom = top
        else:
            text_bottom = bottom + text_height
        color = COLORS_ImageDraw[label]

        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        # 画 keypoints
        draw.chord((k[0] - cw, k[1] - cw, k[0] + cw, k[1] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        draw.chord((k[2] - cw, k[3] - cw, k[2] + cw, k[3] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        draw.chord((k[4] - cw, k[5] - cw, k[4] + cw, k[5] + cw), 0, 360, fill=(0, 0, 255), outline=(0, 255, 0))
        draw.chord((k[6] - cw, k[7] - cw, k[6] + cw, k[7] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        draw.chord((k[8] - cw, k[9] - cw, k[8] + cw, k[9] + cw), 0, 360, fill=(255, 0, 0), outline=(0, 255, 0))
        if text_fill:
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width + 2 * margin, text_bottom)], fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      show_text, fill='black', font=font)
        else:
            draw.text((left + margin, text_bottom - text_height - margin),
                      show_text, fill=color, font=font)
        # font = ImageFont.truetype('simhei.ttf', 30, encoding='utf-8')
        # draw.text((100, 100), '优秀, 哈哈', (0, 255, 255), font=font)
    return img_pil


def f_show_od4pil(img_pil, boxes_ltrb, scores, labels, id_to_class=None, font_size=10, text_fill=True):
    '''
    需扩展按conf排序  或conf过滤的功能
    :param img_pil: 一张 pil
    :param boxes_ltrb:
    :param scores: [1. 1. 1. 1. 1. 1. 1. 1.] np
    :param labels: list(int)
    :param id_to_class: dict{id,name}
    :return:
    '''
    img_pil_copy = img_pil.copy()
    img_pil_copy = f_plot_od4pil(img_pil_copy, boxes_ltrb, scores, labels, id_to_class, font_size, text_fill)
    img_pil_copy.show()


def f_show_od4ts(img_ts, boxes_ltrb, scores, labels, class_dict=None):
    img_pil = transforms.ToPILImage()(img_ts)
    # boxes_confs = torch.cat([boxes_ltrb, scores[:, None]], dim=1)
    f_show_od4pil(img_pil, boxes_ltrb, scores,
                  # (labels.type(torch.int)).tolist(),  # 支持数字key  dict
                  labels, id_to_class=class_dict
                  )


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
        plt.text(b[0], b[1], 'P%s' % i, color=COLORS_ImageDraw[i])
        _rect = plt.Rectangle((b[0], b[1]), b[2], b[3], color=COLORS_ImageDraw[i], fill=False)
        ax.add_patch(_rect)
    plt.show()


def _draw_box_circle4pil(draw, gboxes, color=None, width=4, is_draw_circle=False):
    '''

    :param draw:
    :param gboxes:
    :param color: 有颜色画圆心
    :param width:
    :param is_draw_circle:
    :return:
    '''
    from f_tools.fun_od.f_boxes import ltrb2xywh
    boxes_xywh = ltrb2xywh(gboxes)
    if is_draw_circle:
        for c in boxes_xywh:
            if color is None:
                _color = COLORS_ImageDraw[random.randint(0, len(COLORS_ImageDraw) - 1)]
            else:
                _color = color
            # draw.point(c[:2].numpy().tolist(), fill = color)
            x, y = c[:2]
            r = 4
            # 空心
            # draw.arc((x - r, y - r, x + r, y + r), 0, 360, fill=color)
            # 实心
            draw.chord((x - r, y - r, x + r, y + r), 0, 360, fill=_color)

    for c in gboxes:
        if isinstance(c, np.ndarray):
            l, t, r, b = c.astype(np.int)
        elif isinstance(c, torch.Tensor):
            l, t, r, b = c.type(torch.int)
        else:
            raise Exception('类型错误', type(c))
        if color is None:
            _color = COLORS_ImageDraw[random.randint(0, len(COLORS_ImageDraw) - 1)]
        else:
            _color = color
        draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=width, fill=_color)
    return draw


def f_show_3box4pil(img_pil, g_boxes=None, boxes1=None, boxes2=None, boxes3=None, grids=None, is_safe=True,
                    is_oned=True):
    '''
    默认安全
    :param img_pil:
    :param g_boxes: ltrb 归一化尺寸
    :param boxes1: 预测
    :param boxes2: 归一化尺寸
    :param boxes3: 修复
    :return:
    '''
    if is_safe:
        pil_copy = img_pil.copy()
    else:
        pil_copy = img_pil
    draw = ImageDraw.Draw(pil_copy)
    whwh = np.array(img_pil.size).repeat(2, axis=0)  # 单体复制

    if grids is not None:
        _draw_grid4pil(draw, img_pil.size, grids)
    if g_boxes is not None:
        if isinstance(g_boxes, np.ndarray):
            _gbox = g_boxes.copy()
        elif isinstance(g_boxes, torch.Tensor):
            _gbox = g_boxes.clone()
        else:
            raise Exception('类型错误', type(g_boxes))
        if is_oned:
            _gbox = _gbox * whwh
        _draw_box_circle4pil(draw, _gbox, color='Yellow', is_draw_circle=True)  # 黄色
    if boxes1 is not None:
        _box1 = boxes1.clone().detach()
        if is_oned:
            _box1 = _box1 * whwh
        _draw_box_circle4pil(draw, _box1, color='Chartreuse', width=2, is_draw_circle=True)  # 绿色
    if boxes2 is not None:
        _box2 = boxes2.clone()
        if is_oned:
            _box2 = _box2 * whwh
        _draw_box_circle4pil(draw, _box2, color='BurlyWood', width=2, is_draw_circle=True)  # 橙色
    if boxes3 is not None:
        _box3 = boxes3.clone()
        if is_oned:
            _box3 = _box3 * whwh
        _draw_box_circle4pil(draw, _box3, color='Aqua', width=2, is_draw_circle=True)  #
    pil_copy.show()


def _draw_grid4pil(draw, size, grids=(7, 7)):
    # colors_ = STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)]
    colors_ = 'Pink'
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


def f_plt_od_pil(img_pil, boxes_ltrb_f, g_boxes_ltrb=None, ids2classes=None,
                 labels=None, scores=None,
                 is_recover_size=True):
    '''
    f coco使用
    :param img_pil:
    :param boxes_ltrb_f:
    :param g_boxes_ltrb:
    :param ids2classes:
    :param labels:
    :param scores:
    :param is_recover_size:
    :return:
    '''
    whwh = np.array(img_pil.size).repeat(2, axis=0)
    plt.imshow(img_pil, alpha=0.7)
    current_axis = plt.gca()
    if g_boxes_ltrb is not None:
        if is_recover_size:
            g_boxes_ltrb = g_boxes_ltrb * whwh
        for box in g_boxes_ltrb:
            l, t, r, b = box
            plt_rectangle = plt.Rectangle((l, t), r - l, b - t, color='lightcyan', fill=False, linewidth=3)
            current_axis.add_patch(plt_rectangle)
            # x, y = c[:2]
            # r = 4
            # # 空心
            # # draw.arc((x - r, y - r, x + r, y + r), 0, 360, fill=color)
            # # 实心
            # draw.chord((x - r, y - r, x + r, y + r), 0, 360, fill=_color)

    for i, box_ltrb_f in enumerate(boxes_ltrb_f):
        l, t, r, b = box_ltrb_f
        # ltwh
        current_axis.add_patch(plt.Rectangle((l, t), r - l, b - t, color='green', fill=False, linewidth=2))
        if labels is not None:
            # labels : tensor -> int
            show_text = ids2classes[int(labels[i])] + str(round(scores[i], 2))
            current_axis.text(l, t - 2, show_text, size=8, color='white',
                              bbox={'facecolor': 'green', 'alpha': 0.3})
    plt.show()


def f_plt_od_np(img_np, boxes_ltrb_f, g_boxes_ltrb=None, ids2classes=None,
                labels=None, scores=None,
                is_recover_size=True):
    '''
    f coco使用
    :param img_np:
    :param boxes_ltrb_f:
    :param g_boxes_ltrb:
    :param ids2classes:
    :param labels:
    :param scores:
    :param is_recover_size:
    :return:
    '''
    # import matplotlib.pyplot as plt
    plt.title('%s x %s' % (str(img_np.shape[1]), str(img_np.shape[0])))
    # plt.imshow(img_np)
    # plt.show()
    whwh = np.tile(np.array(img_np.shape[:2][::-1]), 2)  # 整体复制
    # plt.figure(whwh)  #要报错
    plt.imshow(img_np, alpha=0.7)
    current_axis = plt.gca()
    if g_boxes_ltrb is not None:
        if is_recover_size:
            g_boxes_ltrb = g_boxes_ltrb * whwh
        for box in g_boxes_ltrb:
            l, t, r, b = box
            plt_rectangle = plt.Rectangle((l, t), r - l, b - t, color='lightcyan', fill=False, linewidth=3)
            current_axis.add_patch(plt_rectangle)
            # x, y = c[:2]
            # r = 4
            # # 空心
            # # draw.arc((x - r, y - r, x + r, y + r), 0, 360, fill=color)
            # # 实心
            # draw.chord((x - r, y - r, x + r, y + r), 0, 360, fill=_color)

    for i, box_ltrb_f in enumerate(boxes_ltrb_f):
        l, t, r, b = box_ltrb_f
        # ltwh
        current_axis.add_patch(plt.Rectangle((l, t), r - l, b - t, color='green', fill=False, linewidth=2))
        if labels is not None:
            # labels : tensor -> int
            show_text = ids2classes[int(labels[i])] + str(round(scores[i], 2))
            current_axis.text(l, t - 2, show_text, size=8, color='white',
                              bbox={'facecolor': 'green', 'alpha': 0.3})
    plt.show()


def f_plt_box2(img_pil, g_boxes_ltrb=None, boxes1_ltrb=None, is_recover_size=True):
    '''
    显示对比
    :param img_pil:
    :param g_boxes_ltrb:
    :param boxes1_ltrb:
    :param is_recover_size:
    :return:
    '''
    whwh = np.array(img_pil.size).repeat(2, axis=0)  # 单体复制
    plt.imshow(img_pil)
    current_axis = plt.gca()

    if boxes1_ltrb is not None:
        if is_recover_size:
            boxes1_ltrb = boxes1_ltrb * whwh
        for box in boxes1_ltrb:
            l, t, r, b = box
            rectangle = plt.Rectangle((l, t), r - l, b - t,
                                      color=COLORS_plt[list(COLORS_plt.keys())[random.randint(0, len(COLORS_plt) - 1)]],
                                      fill=False, linewidth=1)
            current_axis.add_patch(rectangle)

    if g_boxes_ltrb is not None:
        if is_recover_size:
            g_boxes_ltrb = g_boxes_ltrb * whwh
        for box in g_boxes_ltrb:
            l, t, r, b = box
            plt_rectangle = plt.Rectangle((l, t), r - l, b - t, color='lightcyan', fill=False, linewidth=3)
            current_axis.add_patch(plt_rectangle)
    plt.show()


def f_plt_od_f(img_ts, boxes_ltrb):
    '''
    恢复图片 只显示 框
    :param img_ts:
    :param boxes_ltrb:
    :return:
    '''
    from f_tools.pic.enhance.f_data_pretreatment import f_recover_normalization4ts
    from torchvision.transforms import functional as transformsF
    # 特图恢复s
    img_ts = f_recover_normalization4ts(img_ts)
    img_pil = transformsF.to_pil_image(img_ts).convert('RGB')
    _size = img_pil.size
    boxes_ltrb_f = boxes_ltrb.cpu() * torch.tensor(_size).repeat(2)
    f_plt_od_pil(img_pil, boxes_ltrb_f)


def f_plt_show_pil(img_pil):
    plt.imshow(img_pil)
    plt.show()


def _f_draw_box_pil(current_axis, box, color='red', fill=False, linewidth=1, is_show_xy=True):
    l, t, r, b = box
    w = r - l
    h = b - t
    rectangle = plt.Rectangle((l, t), w, h, color=color, fill=fill, linewidth=linewidth)
    current_axis.add_patch(rectangle)

    if is_show_xy:
        x = l + w / 2
        y = t + h / 2
        plt.scatter(x, y, marker='x', color=color, s=40, label='First')


def _f_draw_grid4plt(size, grids):
    '''

    :param size:
    :param grids:
    :return:
    '''
    # colors_ = STANDARD_COLORS[random.randint(0, len(STANDARD_COLORS) - 1)]
    colors_ = 'Pink'
    w, h = size
    xys = np.array([w, h]) / grids
    off_x = np.arange(1, grids[0])
    off_y = np.arange(1, grids[1])
    xx = off_x * xys[0]
    yy = off_y * xys[1]

    for x_ in xx:
        # 画列
        plt.plot([x_, x_], [0, h], color=colors_, linewidth=1., alpha=0.3)
    for y_ in yy:
        # 画列
        plt.plot([0, w], [y_, y_], color=colors_, linewidth=1., alpha=0.3)


def _f_draw_od_np4plt(current_axis, boxes_ltrb, is_show_xy=True, labels_text=None, p_scores_float=None,
                      color='lightcyan', fill=False, linewidth=1):
    '''新版本'''
    # color = 'lightcyan'  # 白色
    # color = COLORS_plt[list(COLORS_plt.keys())[random.randint(0, len(COLORS_plt) - 1)]]
    # color = 'red'  # 白色

    for i, box in enumerate(boxes_ltrb):
        l, t, r, b = box
        _f_draw_box_pil(current_axis, box, color=color, fill=fill, linewidth=linewidth, is_show_xy=is_show_xy)

        if labels_text is not None:
            if p_scores_float is not None:
                text = "{}:{:.3f}".format(labels_text[i], p_scores_float[i])
            else:
                text = labels_text[i]
            current_axis.text(l, t - 2, text, size=8, color='white', bbox={'facecolor': 'green', 'alpha': 0.3})


def f_show_od_np4cv(img_np, boxes_ltrb, scores, plabels_text, is_showing=True):
    if isinstance(boxes_ltrb, torch.Tensor):
        # 转换ts
        boxes_ltrb = boxes_ltrb.numpy()
        # scores = scores.numpy()
    for b, s, t in zip(boxes_ltrb, scores, plabels_text):
        b = b.astype(np.int)
        text = "{}:{:.3f}".format(t, s)
        cv2.rectangle(img_np, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # (l,t),(r,b),颜色.宽度
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_np, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # 远程无法显示
    # img_pil = Image.fromarray(img_np, mode="RGB")  # h,w,c
    # img_pil.show()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    if is_showing:
        cv2.imshow("after", img_np)
        cv2.waitKey(0)
    else:
        return img_np


def f_show_od_np4plt(img_np_bgr, gboxes_ltrb=None, pboxes_ltrb=None, is_recover_size=False,
                     glabels_text=None,
                     plabels_text=None,
                     p_scores_float=None,
                     grids=None):
    '''

    :param img_np_bgr: 转换rgb
    :param gboxes_ltrb: tensors
    :param pboxes_ltrb:
    :param is_recover_size:
    :param labels_text:
    :param grids:
    :return:
    '''
    current_axis = plt.gca()
    img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_np_rgb)

    wh = img_np_rgb.shape[:2][::-1]  # npwh
    plt.title(wh)
    if grids is not None:
        _f_draw_grid4plt(wh, grids)

    if gboxes_ltrb is not None:
        gboxes_ltrb = gboxes_ltrb.cpu()
        if is_recover_size:
            whwh = np.tile(np.array(wh), 2)  # 整体复制 tile
            gboxes_ltrb = gboxes_ltrb * whwh
        _f_draw_od_np4plt(current_axis, gboxes_ltrb, is_show_xy=True, color=COLOR_GBOXES,
                          labels_text=glabels_text, linewidth=3)

    if pboxes_ltrb is not None:
        if is_recover_size:
            whwh = np.tile(np.array(wh), 2)  # 整体复制 tile
            pboxes_ltrb = pboxes_ltrb * whwh
        _f_draw_od_np4plt(current_axis, pboxes_ltrb, is_show_xy=True, color=COLOR_PBOXES,
                          labels_text=plabels_text,
                          p_scores_float=p_scores_float,
                          linewidth=1)

    plt.show()


def f_show_od_ts4plt(img_ts, gboxes_ltrb=None, pboxes_ltrb=None, is_recover_size=False, labels_text=None, grids=None):
    img_np_rgb = img_ts.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
    img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
    f_show_od_np4plt(img_np_bgr, gboxes_ltrb, pboxes_ltrb, is_recover_size, labels_text, grids)


if __name__ == '__main__':
    file_pic = r'D:\tb\tb\ai_code\DL\_test_pic\2007_006046.jpg'
    # file_pic = r'D:\tb\tb\ai_code\DL\_test_pic\2007_000121.jpg'

    # img = cv2.imread(file_img)
    from PIL import Image
    from torchvision.transforms import functional as F

    # # 直接拉伸 pil打开为RGB
    img_pil = Image.open(file_pic).convert('RGB')
    #
    # f_show_grid4pil(img_pil, (20, 5))

    # pil_img = Image.fromarray(cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB))
    # pil_img.show()
    # 生成画笔
    draw = ImageDraw.Draw(img_pil)
    # 第一个参数是字体文件的路径，第二个是字体大小
    font = ImageFont.truetype('simhei.ttf', 30, encoding='utf-8')
    # 第一个参数是文字的起始坐标，第二个需要输出的文字，第三个是字体颜色，第四个是字体类型
    draw.text((100, 100), '优秀, 哈哈', (0, 255, 255), font=font)

    # PIL图片转cv2
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # 变得可以拉伸 winname 必须要一样，且设置可以拉伸在前面
    cv2.namedWindow('w_img', cv2.WINDOW_NORMAL)
    # 显示
    cv2.imshow("w_img", img_np)
    # 等待
    cv2.waitKey(0)
