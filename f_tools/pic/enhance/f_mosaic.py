import torch
from PIL import Image, ImageDraw
import numpy as np

from f_tools.f_general import rand
from f_tools.fun_od.f_boxes import resize_boxes4np
from f_tools.pic.f_show import f_show_od4pil
from f_tools.pic.f_size_handler import resize_img_pil_keep

'''
1、增加数据的多样性
2、增加目标个数
3、BN能一次性统计多张图片的参数
'''


def f_mosaic_pics_ts(imgs, boxs, labels, out_size, is_visual=False, range=(0.4, 0.6),
                     is_keep_wh=True):
    '''

    :param imgs: list(4张图片) list(img_pil)
    :param boxs: list(np) (n,4) 真实的左上右下值 ltrb
    :param out_size: w,h
    :param is_visual: debug
    :param range: 中点随机范围
    :param is_keep_wh: 是否保持wh
    :return:
       img_pil_mosaic_one:已归一化的图片
       boxes_mosaic :拼接缩小后的box
    '''
    '''区域划分---确定4张图片的左上角位置'''
    ow, oh = out_size
    _offset_x_scale = rand(*range)
    _offset_y_scale = rand(*range)
    if is_visual:
        print(_offset_y_scale, _offset_x_scale)
    # 4个图片的偏移起点
    offset_x = int(ow * _offset_x_scale)
    offset_y = int(oh * _offset_y_scale)
    place_x = [0, 0, offset_x, offset_x]  # <class 'list'>: [0, 0, 166, 166]
    place_y = [0, offset_y, offset_y, 0]  # <class 'list'>: [0, 166, 166, 0]
    img_pil_mosaic = Image.new('RGB', (ow, oh), (128, 128, 128))
    boxes_ts = torch.zeros((0, 4))
    labels_ts = torch.zeros(0, dtype=torch.int64)

    index = 0

    for img_pil, box, label in zip(imgs, boxs, labels):
        # 图片的大小
        iw, ih = img_pil.size

        # image.save(str(index)+".jpg")
        '''随机翻转图片'''
        flip = rand() < .5
        if flip and len(box) > 0:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)  # pil图形处理
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        '''图片变成输出尺寸变小,保持输出尺寸(非图片比例)的比例---'''
        if index == 0:
            nw, nh = offset_x, offset_y
        elif index == 1:
            nw, nh = offset_x, oh - offset_y
        elif index == 2:
            nw, nh = ow - offset_x, oh - offset_y
        else:  # index == 3:
            nw, nh = ow - offset_x, offset_y

        if is_keep_wh:
            img_pil, _, nsize = resize_img_pil_keep(img_pil, (nw, nh), is_fill=True)
        else:
            img_pil = img_pil.resize((nw, nh), Image.BICUBIC)

        # 将图片进行放置，分别对应四张分割图片的位置 图片左上点位置
        dx = place_x[index]
        dy = place_y[index]
        # img_pil_mosaic_one = Image.new('RGB', (ow, oh), (128, 128, 128))
        img_pil_mosaic.paste(img_pil, (dx, dy))
        # image_data = np.array(img_pil_mosaic_one) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        if len(box) > 0:
            # 这里可以改进 有些框会变得很小
            if is_keep_wh:
                box = resize_boxes4np(box, np.array([iw, ih]), np.array(nsize))
            else:
                box = resize_boxes4np(box, np.array([iw, ih]), np.array([nw, nh]))

            # torch.tensor(box)
            # wh有一个为0的 过滤
            # box[:, ::2] = box[:, ::2].clip(min=0, max=nw)
            # box[:, 1::2] = box[:, 1::2].clip(min=0, max=nh)
            box[:, ::2].clamp_(min=0, max=nw)
            box[:, 1::2].clamp_(min=0, max=nh)
            wh = box[:, 2:] - box[:, :2]
            _mask = (wh == 0).any(axis=1)  # 降维 np.array([[0,0,0,0],[0,0,3,0],[0,0,0,0],])
            box = box[torch.logical_not(_mask)]
            label = label[torch.logical_not(_mask)]

            # print(wh)
            _dxdy = torch.cat([torch.tensor(dx).reshape(1, -1), torch.tensor(dy).reshape(1, -1)], dim=1)
            # _dxdy = np.concatenate([np.array(dx).reshape(1, -1), np.array(dy).reshape(1, -1)], axis=1)
            # dxdydxdy = np.concatenate([_dxdy, _dxdy], axis=1)
            dxdydxdy = torch.cat([_dxdy, _dxdy], dim=1)
            box = box + dxdydxdy
            boxes_ts = torch.cat([boxes_ts, box], dim=0)
            labels_ts = torch.cat([labels_ts, label], dim=0)
        index = index + 1

        if is_visual:
            '''---可视化---'''
            img_pil_copy = img_pil_mosaic.copy()
            for b in box:
                left, top, right, bottom = b
                draw = ImageDraw.Draw(img_pil_copy)
                draw.rectangle([left, top, right, bottom], outline=(255, 255, 255), width=2)
            img_pil_copy.show()

    return img_pil_mosaic, boxes_ts, labels_ts


if __name__ == "__main__":
    from f_tools.datas.data_factory import VOCDataSet

    '''
    1. 随机读取四张图片
    2. 分别对四张图片预处理
        进行翻转（对原始图片进行左右的翻转）
        缩放（对原始图片进行大小的缩放）
        色域变化（对原始图片的明亮度、饱和度、色调进行改变）等操作
    3. 第一张图片摆放在左上，第二张图片摆放在左下，第三张图片摆放在右下，第四张图片摆放在右上
    4. 利用矩阵的方式将四张图片它固定的区域截取下来，然后将它们拼接起来，拼接成一 张新的图片

    '''
    path = r'M:\AI\datas\VOC2012\trainval'

    dataset_train = VOCDataSet(
        path,
        'train.txt',  # 正式训练要改这里
        transforms=None,
        bbox2one=False,
        isdebug=True
    )

    imgs = []
    boxs = []
    labels4 = []
    i = 1
    for img_pil, target in dataset_train:
        '''
        target["boxes"] = boxes  # 输出左上右下
        target["labels"] = labels
        target["image_id"] = image_id
        target["height_width"] = torch.tensor([image.size[1], image.size[0]])
        target["area"] = area
        target["iscrowd"] = iscrowd
        '''
        # print(img_pil, target)
        imgs.append(img_pil)  # list(img_pil)
        boxs.append(target["boxes"])
        labels4.append(target["labels"])
        if i % 4 == 0:
            img_pil_mosaic, boxes_mosaic, labels = f_mosaic_pics_ts(imgs, boxs, labels4, [550, 550], is_visual=False)
            print(len(boxes_mosaic), len(boxes_mosaic) == len(labels))
            boxes_confs = np.concatenate([boxes_mosaic, np.ones((boxes_mosaic.shape[0], 1))], axis=1)
            f_show_od4pil(img_pil_mosaic, boxes_confs, list(labels.type(torch.int16).numpy()))
            # f_show_od4pil(img_pil_mosaic, boxes_confs, labels, text_fill=False)
            # img_pil_mosaic.save("box_all.jpg")
            imgs.clear()
            boxs.clear()
        i += 1
