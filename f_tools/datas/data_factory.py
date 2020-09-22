import json
from operator import itemgetter

import cv2
from torch.utils.data import Dataset
import os
import torch
import xmltodict
import numpy as np

# from f_tools.pic.f_show import show_od4boxs
# from object_detection.faster_rcnn.draw_box_utils import draw_box
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from f_tools.GLOBAL_LOG import flog


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, path_data_root, path_file_txt, transforms=None,
                 bbox2one=False):
        '''

        :param path_data_root: voc数据集的根目录
        :param path_file_txt: 提前准备好的xml及jpg对应的文件名
        :param transforms:  自定义的 transforms 支持 boxes 一起处理
        :param bbox2one: 是否需要统一化 bbox2one
        '''
        self.path_data_root = path_data_root
        self.transforms = transforms
        self.bbox2one = bbox2one

        path_txt = os.path.join(path_data_root, path_file_txt)
        _path_xml = os.path.join(path_data_root, 'Annotations')
        with open(path_txt) as read:
            # 读每一行加上路径和扩展名---完整路径
            self.xml_list = [os.path.join(_path_xml, line.strip() + ".xml")
                             for line in read.readlines()]

        try:
            # {"类别1": 1, "类别2":2}
            json_file = open(os.path.join(
                os.path.abspath(os.path.join(path_data_root, "..")), 'classes_voc.json'), 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            flog.error(e)
            exit(-1)

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        '''
        有可能对gt进行归一化
        和img预处理

        :param idx:
        :return:
            image 经预处理后的值 torch.Size([3, 375, 500])
            target对象 全tensor值 bndbox name difficult 输出左上右下
        '''
        # ----------解析xml-----------
        path_xml = self.xml_list[idx]
        doc = self.parse_xml(path_xml)  # 解析xml
        path_img = os.path.join(os.path.join(self.path_data_root, 'JPEGImages'), doc['annotation']['filename'])

        image = Image.open(path_img)  # 这个打开的wh
        if image.format != "JPEG":  # 类型校验 这里打开的是原图
            raise ValueError("Image format not JPEG")

        '''-----------组装 target 结构不一样,且有可能有多个------------'''
        boxes = np.empty(shape=(0, 4), dtype=np.float)
        labels = []
        iscrowd = []  # 是否困难
        _objs = doc['annotation']['object']
        if isinstance(_objs, dict):
            xmin = float(_objs['bndbox']['xmin'])
            ymin = float(_objs['bndbox']['ymin'])
            xmax = float(_objs['bndbox']['xmax'])
            ymax = float(_objs['bndbox']['ymax'])
            boxes = np.concatenate((boxes, np.array([xmin, ymin, xmax, ymax])[None]), axis=0)
            labels.append(self.class_dict[_objs['name']])
            iscrowd.append(int(_objs['difficult']))
        else:
            for obj in _objs:
                # 可能有多个目标
                xmin = float(obj['bndbox']['xmin'])
                ymin = float(obj['bndbox']['ymin'])
                xmax = float(obj['bndbox']['xmax'])
                ymax = float(obj['bndbox']['ymax'])
                boxes = np.concatenate((boxes, np.array([xmin, ymin, xmax, ymax])[None]), axis=0)
                labels.append(self.class_dict[obj['name']])
                iscrowd.append(int(obj['difficult']))

        # show_od4boxs(image,boxes) # 原图测试

        # list(np数组)   转换   为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if self.bbox2one:
            boxes /= torch.tensor(image.size).repeat(2)  # np高级
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])  # 这是第几张图的target

        _h = boxes[:, 3] - boxes[:, 1]
        _w = boxes[:, 2] - boxes[:, 0]
        area = (_h) * (_w)

        target = {}
        target["boxes"] = boxes  # 输出左上右下
        target["labels"] = labels
        target["image_id"] = image_id
        target["height_width"] = torch.tensor([image.size[1], image.size[0]])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:  # 这里是预处理
            image, target = self.transforms(image, target)

        # show_od4boxs(image, target['boxes'], is_tensor=True)  # 预处理图片测试
        return image, target

    def get_height_and_width(self, idx):
        # read xml 多GPU时必须实现这个方法
        # 通过xml获取宽高
        path_xml = self.xml_list[idx]
        doc = self.parse_xml(path_xml)  # 解析xml
        data_height = int(doc['annotation']['size']['height'])
        data_width = int(doc['annotation']['size']['width'])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args：
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def parse_xml(self, path_xml):
        with open(path_xml) as file:
            str_xml = file.read()
        doc = xmltodict.parse(str_xml)
        return doc


def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


class WiderfaceDataSet(Dataset):
    '''
    包含有图片查看的功能
    '''

    def __init__(self, path_file, img_size, mode='train', isdebug=False, look=False):
        '''
        原文件是 lrwh 转 lrtb
        :param path_file: 数据文件的主路径
        :param img_size:  超参 预处理后的尺寸[640,640]
        :param mode:  train val test
        :param isdebug:
        :param look:  是否查看图片
        '''
        self.look = look
        self.img_size = img_size
        if isdebug:
            _file_name = '_label.txt'
        else:
            _file_name = 'label.txt'

        self.txt_path = os.path.join(path_file, mode, _file_name)

        if os.path.exists(self.txt_path):
            f = open(self.txt_path, 'r')
        else:
            raise Exception('标签文件不存在: %s' % self.txt_path)

        self.imgs_path = []  # 每一张图片的全路径
        self.words = []  # 每一张图片对应的15维数据 应该在取的时候再弄出来

        # 这里要重做
        lines = f.readlines()
        is_first = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):  # 删除末尾空格
                if is_first is True:
                    is_first = False
                else:  # 在处理下一个文件时,将上一个labels加入,并清空
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                _path_file = os.path.join(path_file, mode, 'images', line[2:])
                self.imgs_path.append(_path_file)
            else:
                line = line.split(' ')
                label = [int(float(x)) for x in line]  # 这里 lrtb
                _t = list(itemgetter(0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17)(label))
                #lrwh 转 lrtb
                _t[2] += _t[0]
                _t[3] += _t[1]
                if label[4] > 0:
                    _t.append(1)
                else:
                    _t.append(-1)
                labels.append(_t)
        self.words.append(labels)  # 最后还有一个需要处理

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            <class 'tuple'>: (3, 640, 640)
            list(x,15)  ltrb = 4 + 1 +10 归一化后的
        '''
        # 随机打乱 loader有这个功能
        # if index == 0:
        #     shuffle_index = np.arange(len(self.imgs_path))
        #     shuffle(shuffle_index)
        #     self.imgs_path = np.array(self.imgs_path)[shuffle_index]
        #     self.words = np.array(self.words)[shuffle_index]

        img = Image.open(self.imgs_path[index])  # 原图数据
        # 这里是原图
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        labels = self.words[index]

        if self.look:
            self.show_face(img, labels)

        # 加快图片解码 pip install jpeg4py -i https://pypi.tuna.tsinghua.edu.cn/simple
        # import jpeg4py as jpeg
        # img = jpeg.JPEG(self.imgs_path[index]).decode()

        target = np.array(labels)
        # 图形增强处理
        img, target = self.pic_enhance(img, target, self.img_size)
        rgb_mean = (104, 117, 123)  # bgr order

        # 转换为
        img = np.transpose(img - np.array(rgb_mean), (2, 0, 1))
        img = np.array(img, dtype=np.float32)
        return img, target

    def show_face(self, img, labels):
        draw = ImageDraw.Draw(img)
        for box in labels:
            l, t, r, b = box[0:4]
            # left, top, right, bottom = box[0:4]
            draw.rectangle([l, t, r, b], width=2, outline=(255, 255, 255))
            _ww = 2
            if box[-1] > 0:
                _t = 4
                for _ in range(5):
                    ltrb = [int(box[_t]) - _ww, int(box[_t + 1]) - _ww, int(box[_t]) + _ww, int(box[_t + 1]) + _ww]
                    print(ltrb)
                    draw.rectangle(ltrb)
                    _t += 2
        img.show()  # 通过系统的图片软件打开

    def pic_enhance(self, image, targes, input_shape, jitter=(0.9, 1.1), hue=.1, sat=1.5, val=1.5):
        '''
        图片增强
        :param image: 原图数据
        :param targes: 一个图多个框和关键点,15维=4+10+1
        :param input_shape: 预处理后的尺寸
        :param jitter: 原图片的宽高的扭曲比率
        :param hue: hsv色域中三个通道的扭曲
        :param sat: 色调（H），饱和度（S），
        :param val:明度（V）
        :return:
            image 预处理后图片 nparray: (640, 640, 3)
            和 一起处理的选框 并归一化
        '''
        iw, ih = image.size
        h, w = input_shape
        box = targes

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = w / h * _rand(jitter[0], jitter[1]) / _rand(jitter[0], jitter[1])
        scale = _rand(0.75, 1.25)  # 在0.25到2之间缩放
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条 这里调整尺寸
        dx = int(_rand(0, w - nw))
        dy = int(_rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 50% 翻转图像
        flip = _rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域扭曲
        hue = _rand(-hue, hue)
        sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
        val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
        # PIL.Image对象 -> 归一化的np对象 (640, 640, 3)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2, 4, 6, 8, 10, 12]] = box[:, [0, 2, 4, 6, 8, 10, 12]] * nw / iw + dx
            box[:, [1, 3, 5, 7, 9, 11, 13]] = box[:, [1, 3, 5, 7, 9, 11, 13]] * nh / ih + dy
            if flip:
                box[:, [0, 2, 4, 6, 8, 10, 12]] = w - box[:, [2, 0, 6, 4, 8, 12, 10]]
                box[:, [5, 7, 9, 11, 13]] = box[:, [7, 5, 9, 13, 11]]
            box[:, 0:14][box[:, 0:14] < 0] = 0
            box[:, [0, 2, 4, 6, 8, 10, 12]][box[:, [0, 2, 4, 6, 8, 10, 12]] > w] = w
            box[:, [1, 3, 5, 7, 9, 11, 13]][box[:, [1, 3, 5, 7, 9, 11, 13]] > h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        # self.show_face(image, box)

        # -----------归一化--------------
        box[:, 4:-1][box[:, -1] == -1] = 0
        box = box.astype(np.float)
        box[:, [0, 2, 4, 6, 8, 10, 12]] = box[:, [0, 2, 4, 6, 8, 10, 12]] / np.array(w)
        box[:, [1, 3, 5, 7, 9, 11, 13]] = box[:, [1, 3, 5, 7, 9, 11, 13]] / np.array(h)
        box_data = box
        return image_data, box_data


class Data_Prefetcher():
    '''
    nvidia 要求进来的是tensor数据 GPU dataloader加速
    将预处理通过GPU完成
    '''

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


if __name__ == '__main__':
    train_loader = None
    prefetcher = Data_Prefetcher(train_loader)
    data = prefetcher.next()
    i = 0
    while data is not None:
        print(i, len(data))
        i += 1
        data = prefetcher.next()
