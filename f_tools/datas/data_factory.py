import json
from operator import itemgetter

import cv2
from torch.utils.data import Dataset
import os
import torch
import xmltodict
import numpy as np

from PIL import Image, ImageDraw
from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog
from f_tools.pic.enhance.f_mosaic import f_mosaic_pics_ts


def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, path_data_root, path_file_txt, transforms=None,
                 bbox2one=False, isdebug=False, is_mosaic=False, cfg=None):
        '''

        :param path_data_root: voc数据集的根目录
        :param path_file_txt: 提前准备好的xml及jpg对应的文件名
        :param transforms:  自定义的 transforms 支持 boxes 一起处理
        :param bbox2one: bbox是否需要统一化 bbox2one
        '''
        self.path_data_root = path_data_root
        self.transforms = transforms
        self.bbox2one = bbox2one
        self.isdebug = isdebug
        self.is_mosaic = is_mosaic
        self.cfg = cfg

        path_txt = os.path.join(path_data_root, path_file_txt)
        _path_xml = os.path.join(path_data_root, 'Annotations')
        with open(
                path_txt) as read:  # {FileNotFoundError}[Errno 2] No such file or directory: '/home/bak3t/bakls299g/AI/datas/VOC2012/trainval/train.txt'
            # 读每一行加上路径和扩展名---完整路径
            self.xml_list = [os.path.join(_path_xml, line.strip() + ".xml")
                             for line in read.readlines()]

        try:
            # {"类别1": 1, "类别2":2}
            path_json_class = os.path.abspath(os.path.join(path_data_root, ".."))
            json_file = open(os.path.join(path_json_class, 'classes_ids_voc.json'), 'r')
            self.class_to_ids = json.load(json_file)
            self.ids_to_class = dict((val, key) for key, val in self.class_to_ids.items())

            file_json_ids_class = os.path.join(path_json_class, 'ids_classes_voc.json')
            if not os.path.exists(file_json_ids_class):
                json_str = json.dumps(self.ids_to_class, indent=4)
                file_json_ids_class = os.path.join(path_json_class, 'ids_classes_voc.json')
                with open(file_json_ids_class, 'w') as json_file:
                    json_file.write(json_str)


        except Exception as e:
            flog.error(e)
            exit(-1)

    def __len__(self):
        if self.isdebug:
            return 15
        if self.is_mosaic:
            return len(self.xml_list) // 4
        return len(self.xml_list)

    def do_mosaic(self, idx):
        imgs = []
        boxs = []
        labels = []
        for i in range(idx, idx + 4):
            img_pil, target = self.open_img_tar(i)
            imgs.append(img_pil)  # list(img_pil)
            boxs.append(target["boxes"])
            labels.append(target["labels"])
        img_pil_mosaic, boxes_mosaic, labels = f_mosaic_pics_ts(imgs, boxs, labels, self.cfg.IMAGE_SIZE,
                                                                is_visual=False)
        target = {}
        target["boxes"] = boxes_mosaic  # 输出左上右下
        target["labels"] = labels
        return img_pil_mosaic, target

    def __getitem__(self, idx):
        '''
        有可能对gt进行归一化
        和img预处理

        :param idx:
        :return:
            image 经预处理后的值 torch.Size([3, 375, 500])
            target对象 全tensor值 bndbox name difficult 输出左上右下
                box是 ltrb 是否归一化根据参数
        '''

        if self.is_mosaic is True:
            img_ts, target_ts = self.do_mosaic(idx)
        else:
            # ----------解析xml-----------
            # path_xml = self.xml_list[idx + 307 * 16]
            img_ts, target_ts = self.open_img_tar(idx)

        '''这里输出 img_pil'''
        if self.transforms is not None:  # 这里是预处理
            img_ts, target_ts = self.transforms(img_ts, target_ts)  # 这里返回的是匹配后的bbox

        # show_od4boxs(image, target['boxes'], is_tensor=True)  # 预处理图片测试
        return img_ts, target_ts

    def open_img_tar(self, idx):
        path_xml = self.xml_list[idx]
        doc = self.parse_xml(path_xml)  # 解析xml
        path_img = os.path.join(os.path.join(self.path_data_root, 'JPEGImages'), doc['annotation']['filename'])
        # image = Image.open(path_img).convert('RGB')  # 这个打开的wh
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
            labels.append(self.class_to_ids[_objs['name']])
            iscrowd.append(int(_objs['difficult']))
        else:
            for obj in _objs:
                # 可能有多个目标
                xmin = float(obj['bndbox']['xmin'])
                ymin = float(obj['bndbox']['ymin'])
                xmax = float(obj['bndbox']['xmax'])
                ymax = float(obj['bndbox']['ymax'])
                boxes = np.concatenate((boxes, np.array([xmin, ymin, xmax, ymax])[None]), axis=0)
                labels.append(self.class_to_ids[obj['name']])
                iscrowd.append(int(obj['difficult']))
        # show_od4boxs(image,boxes) # 原图测试
        # list(np数组)   转换   为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if self.bbox2one:  # 归一化
            boxes /= torch.tensor(image.size).repeat(2)  # np高级
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])  # 这是第几张图的target
        _h = boxes[:, 3] - boxes[:, 1]  # ltrb -> ltwh
        _w = boxes[:, 2] - boxes[:, 0]
        area = (_h) * (_w)  # 面积也是归一化的
        target = {}
        target["boxes"] = boxes  # 输出左上右下
        target["labels"] = labels
        target["image_id"] = image_id
        target["height_width"] = torch.tensor([image.size[1], image.size[0]])
        target["area"] = area
        target["iscrowd"] = iscrowd
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
                # lrwh 转 lrtb
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


class MapDataSet(Dataset):
    def __init__(self, path_imgs, path_eval_info, ids2classes, transforms=None,
                 is_debug=False, look=False, out='ts', ):
        '''

        :param path_imgs: 图片文件夹
        :param path_eval_info: GT info手动创建的
        :param ids2classes:
        :param transforms:
        :param is_debug:
        :param look:
        :param out:
        '''
        if not os.path.exists(path_eval_info):
            raise Exception('path_eval_info 目录不存在: %s' % path_eval_info)

        path_gt_info = os.path.join(path_eval_info, 'gt_info')
        if not os.path.exists(path_gt_info):
            raise Exception('path_gt_info 目录不存在: %s' % path_gt_info)

        path_dt_info = os.path.join(path_eval_info, 'dt_info')
        if not os.path.exists(path_dt_info):
            os.mkdir(path_dt_info)

        self.path_dt_info = path_dt_info  # 用于保存结果
        self.path_gt_info = path_gt_info  # 用于保存结果

        self.out = out
        self.look = look
        self.is_debug = is_debug
        self.transform_cpu = transforms
        self.path_imgs = path_imgs
        # self.path_gt_info = path_dt_info
        self.ids2classes = ids2classes
        self.classes2ids = {}
        for k, v in ids2classes.items():
            self.classes2ids[v] = k

        self.names_gt_info = os.listdir(self.path_gt_info)
        self.targets = []
        # root 当前目录路径   dirs 当前路径下所有子目录   name 当前路径下所有非目录子文件
        # for root, dirs, files in tqdm(os.walk(self.path_gt_info)):
        for name in self.names_gt_info:
            file = os.path.join(self.path_gt_info, name)
            if os.path.isfile(file):  # 判断是否为文件夹
                # os.makedirs('d:/assist/set') 是否目录
                _labels = []
                _bboxs = []
                with open(file) as f:  # 获取目录文件
                    lines = f.readlines()
                    for line in lines:
                        line = line.rstrip()  # 去换行
                        label, l, t, r, b = line.split(' ')
                        _labels.append(int(self.classes2ids[label]))  # label名
                        _bboxs.append([float(l), float(t), float(r), float(b)])
                # 已提前加载全部的标签
                self.targets.append({'labels': _labels, 'boxes': _bboxs})
        # __d = 1

    def __len__(self):
        if self.is_debug:
            return 10
        return len(self.names_gt_info)

    def __getitem__(self, index):
        '''
        target={labels:[x],bboxs:[x,4],size:(w,h)}
        :param index:
        :return: 返回图片有可能是pil 或 其它格式  需根据transform_cpu 判断
        '''

        names_gt_info = self.names_gt_info[index]
        name_jpg = names_gt_info.split('.')[0] + '.jpg'
        img_pil = Image.open(os.path.join(self.path_imgs, name_jpg))  # 原图数据
        target = self.targets[index]
        target['size'] = img_pil.size
        target['name_txt'] = names_gt_info  # 文件名一致
        target['boxes'] = np.array(target['boxes'], dtype=np.float32)
        img_ts4 = None
        if self.transform_cpu is not None:
            # 预处理输入 PIL img 和 np的target
            img_ts4, target = self.transform_cpu(img_pil, target)
        if self.out == 'ts':
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            target['size'] = torch.tensor(target['size'], dtype=torch.int64)
        # 如果不加 transform_cpu 输出是 img_pil
        if img_ts4 is None:
            return img_pil, target
        return img_ts4, target


def load_data4voc(data_transform, path_data_root, batch_size, bbox2one=False, isdebug=False, data_num_workers=0):
    '''

    :param data_transform:
    :param path_data_root:
    :param batch_size:
    :param bbox2one:  是否gt框进行归一化
    :return:
    '''
    num_workers = data_num_workers
    file_name = ['train.txt', 'val.txt']
    VOC_root = os.path.join(path_data_root, 'trainval')
    # ---------------------data_set生成---------------------------
    train_data_set = VOCDataSet(
        VOC_root,
        file_name[0],  # 正式训练要改这里
        data_transform["train"],
        bbox2one=bbox2one,
        isdebug=isdebug
    )

    # iter(train_data_set).__next__()  # VOC2012DataSet 测试
    class_dict = train_data_set.class_to_ids
    flog.debug('class_dict %s', class_dict)

    # 默认通过 torch.stack() 进行拼接
    '''
    一次两张图片使用3504的显存
    '''
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # windows只能为0
        collate_fn=lambda batch: tuple(zip(*batch)),  # 输出多个时需要处理
        pin_memory=True,
    )

    val_data_set = VOCDataSet(
        VOC_root, file_name[1],
        data_transform["val"],
        bbox2one=bbox2one,
        isdebug=isdebug
    )
    val_data_set_loader = torch.utils.data.DataLoader(
        val_data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: tuple(zip(*batch)),  # 输出多个时需要处理
        pin_memory=True,
    )
    # images, targets = iter(train_data_loader).__next__()
    # show_pic_ts(images[0], targets[0]['labels'], classes=class_dict)

    return train_data_loader, val_data_set_loader


def load_data4widerface(path_data_root, img_size_in, batch_size, mode='train', isdebug=False, look=False):
    '''

    :param path_data_root:
    :param img_size_in:
    :param batch_size:
    :param mode:  train val test 暂时debug只支持 train
    :param isdebug:
    :param look:
    :return:
        images: np(batch,(3,640,640))
        targets: np(batch,(x个选框,15维))  4+1+10
    '''

    def detection_collate(batch):
        '''

        :param batch: list(batch,(tuple((3,640,640),(x个选框,15维))...))
        :return:
            images: <class 'tuple'>: (batch, 3, 640, 640)
            targets: list[batch,(23,15)]
        '''
        images = []
        targets = []
        for img, box in batch:
            if len(box) == 0:
                continue
            images.append(img)
            targets.append(box)
        images = np.array(images)
        targets = np.array(targets)
        return images, targets

    train_dataset = WiderfaceDataSet(path_data_root, img_size_in, mode=mode, isdebug=isdebug, look=look)
    # iter(train_dataset).__next__()
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,  # 不使用虚拟内存
        drop_last=True,  # 除于batch_size余下的数据
        collate_fn=detection_collate,
    )
    # 数据查看
    # iter(data_loader).__next__()
    return data_loader


def fun4dataloader(batch_datas):
    '''

    :param batch_datas: img (3,416,416)      target 字典  boxes不定
    :return:
    '''
    _t = batch_datas[0][0]
    # 包装整个图片数据集 (batch,3,416,416) 转换到显卡
    images = torch.empty((len(batch_datas), *_t.shape)).to(_t)
    targets = []
    for i, (img, target) in enumerate(batch_datas):
        # flog.warning('fun4dataloader测试  %s %s %s ', target, len(target['boxes']), len(target['labels']))
        images[i] = img
        targets.append(target)
    return images, targets



def init_dataloader(cfg, dataset_train, dataset_val, is_mgpu, use_mgpu_eval=True):
    loader_train, loader_val_coco, train_sampler, eval_sampler = [None] * 4
    if cfg.IS_TRAIN:
        # __d = dataset_train[0]  # 调试
        if is_mgpu:  # DataLoader 不一样
            # 给每个rank按显示个数生成定义类 shuffle -> ceil(样本/GPU个数)自动补 -> 间隔分配到GPU
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                            shuffle=True,
                                                                            # seed=20201114,
                                                                            )
            # 按定义为每一个 BATCH_SIZE 生成一批的索引
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, cfg.BATCH_SIZE, drop_last=False)

            loader_train = torch.utils.data.DataLoader(
                dataset_train,
                # batch_size=cfg.BATCH_SIZE,
                batch_sampler=train_batch_sampler,  # 按样本定义加载
                num_workers=cfg.DATA_NUM_WORKERS,
                # shuffle=True,
                pin_memory=True,  # 不使用虚拟内存 GPU要报错
                # drop_last=True,  # 除于batch_size余下的数据
                collate_fn=fun4dataloader,
            )
        else:
            loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=cfg.BATCH_SIZE,
                num_workers=cfg.DATA_NUM_WORKERS,
                shuffle=True,
                pin_memory=True,  # 不使用虚拟内存 GPU要报错
                # drop_last=True,  # 除于batch_size余下的数据
                collate_fn=fun4dataloader,
            )

    if cfg.IS_COCO_EVAL:
        if is_mgpu and use_mgpu_eval:
            # 给每个rank按显示个数生成定义类 shuffle -> ceil(样本/GPU个数)自动补 -> 间隔分配到GPU
            eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                           shuffle=True,
                                                                           seed=20201114,
                                                                           )
            # 按定义为每一个 BATCH_SIZE 生成一批的索引
            eval_batch_sampler = torch.utils.data.BatchSampler(eval_sampler,
                                                               int(cfg.BATCH_SIZE),
                                                               drop_last=False,  # 不要扔掉否则验证不齐
                                                               )

            loader_val_coco = torch.utils.data.DataLoader(
                dataset_val,
                # batch_size=cfg.BATCH_SIZE,
                batch_sampler=eval_batch_sampler,  # 按样本定义加载
                num_workers=cfg.DATA_NUM_WORKERS,
                # shuffle=True,
                pin_memory=True,  # 不使用虚拟内存 GPU要报错
                # drop_last=True,  # 除于batch_size余下的数据
                collate_fn=fun4dataloader,
            )
        else:
            loader_val_coco = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=int(cfg.BATCH_SIZE),
                num_workers=cfg.DATA_NUM_WORKERS,
                # shuffle=True,
                pin_memory=True,  # 不使用虚拟内存 GPU要报错
                # drop_last=True,  # 除于batch_size余下的数据
                collate_fn=fun4dataloader,
            )

    return loader_train, loader_val_coco, train_sampler, eval_sampler


if __name__ == '__main__':
    path_root = r'M:\AI\datas\widerface\val'
    class_to_idx = {'face': 1}
    data_set = MapDataSet(path_root, class_to_idx)
    for ss in data_set:
        print(ss)
