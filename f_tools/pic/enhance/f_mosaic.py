from PIL import Image, ImageDraw
import numpy as np

from f_tools.datas.data_factory import VOCDataSet
from f_tools.f_general import rand
from f_tools.fun_od.f_boxes import resize_boxes4np
from f_tools.pic.f_show import f_show_od4pil

'''
1、增加数据的多样性
2、增加目标个数
3、BN能一次性统计多张图片的参数
'''


def mosaic_pic(imgs, boxs, out_size, is_visual=False, range=(0.4, 0.6)):
    '''

    :param imgs: list(4张图片) list(img_pil)
    :param boxs: list(np) (n,4) 真实的左上右下值
    :param out_size: w,h
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
    boxes_np = np.zeros((0, 4))
    index = 0

    for img_pil, box in zip(imgs, boxs):
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
        img_pil = img_pil.resize((nw, nh), Image.BICUBIC)

        # 将图片进行放置，分别对应四张分割图片的位置 图片左上点位置
        dx = place_x[index]
        dy = place_y[index]
        # img_pil_mosaic_one = Image.new('RGB', (ow, oh), (128, 128, 128))
        img_pil_mosaic.paste(img_pil, (dx, dy))
        # image_data = np.array(img_pil_mosaic_one) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        if len(box) > 0:
            box = resize_boxes4np(box, np.array([iw, ih]), np.array([nw, nh]))
            _dxdy = np.concatenate([np.array(dx).reshape(1, -1), np.array(dy).reshape(1, -1)], axis=1)
            dxdydxdy = np.concatenate([_dxdy, _dxdy], axis=1)
            box = box + dxdydxdy
            boxes_np = np.append(boxes_np, box, axis=0)
        index = index + 1

        if is_visual:
            '''---可视化---'''
            img_pil_copy = img_pil_mosaic.copy()
            for b in box:
                left, top, right, bottom = b
                draw = ImageDraw.Draw(img_pil_copy)
                draw.rectangle([left, top, right, bottom], outline=(255, 255, 255), width=2)
            img_pil_copy.show()

    return img_pil_mosaic, boxes_np


if __name__ == "__main__":
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
    labels = []
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
        boxs.append(target["boxes"].numpy())
        labels.extend(list(target["labels"].numpy()))
        if i % 4 == 0:
            img_pil_mosaic, boxes_mosaic = mosaic_pic(imgs, boxs, [550, 550], is_visual=False)
            boxes_confs = np.concatenate([boxes_mosaic, np.ones((boxes_mosaic.shape[0], 1))], axis=1)
            f_show_od4pil(img_pil_mosaic, boxes_confs, labels)
            # f_show_od4pil(img_pil_mosaic, boxes_confs, labels, text_fill=False)
            # img_pil_mosaic.save("box_all.jpg")
            imgs.clear()
            boxs.clear()
        i += 1
