from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from f_tools.datas.data_factory import VOCDataSet


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


def get_mosaic_data(imgs, boxs, out_size, hue=.1, sat=1.5, val=1.5):
    '''

    :param imgs: list(4张图片)
    :param boxs: list(np)
    :param out_size: w,h
    :param hue:
    :param sat:
    :param val:
    :return:
       img_pil_mosaic_one:已归一化的图片
       boxes_mosaic :拼接缩小后的box
    '''
    '''区域划分---确定4张图片的左上角位置'''
    w, h = out_size
    min_offset_x = 0.4
    min_offset_y = 0.4
    # 4个图片的偏移起点
    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]  # <class 'list'>: [0, 0, 166, 166]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]  # <class 'list'>: [0, 166, 166, 0]

    imgs_pil = []
    box_datas = []
    index = 0
    scale_low = 1 - min(min_offset_x, min_offset_y)  # 图片放大比例
    scale_high = scale_low + 0.2  # 0.8

    for img_pil, box in zip(imgs, boxs):
        # 图片的大小
        iw, ih = img_pil.size

        # image.save(str(index)+".jpg")
        '''随机翻转图片'''
        flip = rand() < .5
        if flip and len(box) > 0:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)  # pil图形处理
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        '''图片进行缩放---根据定义的图形点进行拉伸放大'''
        new_ar = w / h  # 目标尺寸长宽比
        scale = rand(scale_low, scale_high)  # 0.6~0.8
        if new_ar < 1:  # w < h 则h最小
            nh = int(scale * h)  # 缩小
            nw = int(nh * new_ar)  # 保持长宽比
        else:  # w > h 缩放w
            nw = int(scale * w)
            nh = int(nw / new_ar)
        img_pil = img_pil.resize((nw, nh), Image.BICUBIC)

        '''进行色域变换'''
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(img_pil) / 255.)  # 归一化
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        img_pil = hsv_to_rgb(x)
        img_pil = Image.fromarray((img_pil * 255).astype(np.uint8))  # 归一化恢复

        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        img_pil_mosaic_one = Image.new('RGB', (w, h), (128, 128, 128))
        img_pil_mosaic_one.paste(img_pil, (dx, dy))
        image_data = np.array(img_pil_mosaic_one) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")

        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            # 边界处理
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            # box_data = np.zeros((len(box), 5))
        box_datas.append(box.copy())

        imgs_pil.append(image_data)

        '''---填充至wh---'''
        img = Image.fromarray((image_data * 255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        index = index + 1
        # img.show() # debug点每张图片的处理

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))
    print(cutx, cuty)

    img_pil_mosaic_one = np.zeros([h, w, 3])
    img_pil_mosaic_one[:cuty, :cutx, :] = imgs_pil[0][:cuty, :cutx, :]
    img_pil_mosaic_one[cuty:, :cutx, :] = imgs_pil[1][cuty:, :cutx, :]
    img_pil_mosaic_one[cuty:, cutx:, :] = imgs_pil[2][cuty:, cutx:, :]
    img_pil_mosaic_one[:cuty, cutx:, :] = imgs_pil[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    boxes_mosaic = merge_bboxes(box_datas, cutx, cuty)

    return img_pil_mosaic_one, boxes_mosaic


def normal_(annotation_line, input_shape):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    iw, ih = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    box[:, [0, 2]] = iw - box[:, [2, 0]]

    return image, box


def show_box(img_pil, boxes):
    for j in range(len(boxes)):
        thickness = 3
        left, top, right, bottom = boxes[j][0:4]
        draw = ImageDraw.Draw(img_pil)
        for t in range(thickness):
            draw.rectangle([left + t, top + t, right - t, bottom - t], outline=(255, 255, 255))
    img_pil.show()


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
        imgs.append(img_pil)
        boxs.append(target["boxes"].numpy())
        if i % 4 == 0:
            img_pil_mosaic_one, boxes_mosaic = get_mosaic_data(imgs, boxs, [550, 550])
            img_pil_mosaic = Image.fromarray((img_pil_mosaic_one * 255).astype(np.uint8))

            show_box(img_pil_mosaic, boxes_mosaic)
            # img.save("box_all.jpg")
            imgs.clear()
            boxs.clear()
        i += 1
