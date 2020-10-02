import cv2
from PIL import Image


def re23size_img_keep_ratio(path_img, target_size, mode='center', v_fill=(0, 0, 0)):
    '''
    img_new, ratio = resize_img_keep_ratio(path_img, target_size, 'upleft', [125, 21, 23])
    :param path_img:
    :param target_size:
    :param mode: center upleft
    :return:
    '''
    img = cv2.imread('t002.py')
    if img:
        raise Exception('图片加载出错 : ', path_img)

    old_size = img.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    print('缩放比例', ratio)
    new_size = tuple([int(i * ratio) for i in old_size])
    interpolation = cv2.INTER_LINEAR if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=interpolation)

    _pad_w = target_size[1] - new_size[1]
    _pad_h = target_size[0] - new_size[0]

    top, bottom, left, right = 0, 0, 0, 0
    if mode == 'center':
        top, bottom = _pad_h // 2, _pad_h - (_pad_h // 2)
        left, right = _pad_w // 2, _pad_w - (_pad_w // 2)
    elif mode == 'upleft':
        right = _pad_w
        bottom = _pad_h

    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, v_fill)
    return img_new, ratio


if __name__ == '__main__':
    # path_img = 'test_99.jpg'
    # img = cv2.imread(path_img)
    # img = cv2.copyMakeBorder(img, 50, 100, 150, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
    # cv2.imshow('new', img)
    # cv2.waitKey(0)
    #
    # image = Image.open(path_img)
    # print(image.size)
    # target_size = [500, 300]
    # img_new, ratio = re23size_img_keep_ratio(path_img, target_size, 'upleft', [125, 21, 23])
    # cv2.imshow('new', img_new)
    # cv2.waitKey(0)
    image = Image.open('../_test_pic/test_99.jpg')
    print(image.size)
