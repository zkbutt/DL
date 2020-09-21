import os
import re
from operator import itemgetter

import torch
import os.path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from random import shuffle
from torch.autograd import Variable
from object_detection.retinaface.utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, neg_pos, overlap_thresh=0.35):
        '''

        :param num_classes: ANCHOR_NUM
        :param overlap_thresh: 重合程度在小于0.35的类别设为0
        :param neg_pos: 正负样本的比率
        '''
        super(MultiBoxLoss, self).__init__()
        # 对于retinaface而言num_classes等于2
        self.num_classes = num_classes
        # 重合程度在多少以上认为该先验框可以用来预测
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]

    def forward(self, predictions, anchors, targets, device):
        '''

        :param predictions: 预测box 调整系数
            预测结果tuple(torch.Size([8, 16800, 4]),torch.Size([8, 16800, 2]),torch.Size([8, 16800, 10]))
        :param anchors: default boxes
        :param targets: 标签 ground truth boxes   n,15(4+10+1)
        :return:
        '''
        loc_data, conf_data, landm_data = predictions
        num_batch = loc_data.size(0)
        num_priors = (anchors.size(0))

        # 创建空的tensor
        loc_t = torch.Tensor(num_batch, num_priors, 4)
        landm_t = torch.Tensor(num_batch, num_priors, 10)
        conf_t = torch.LongTensor(num_batch, num_priors)  # 装类别
        for idx in range(num_batch):  # 遍历每一张图的 标签 GT
            truths = targets[idx][:, :4].data  # 假如有11个框 标签 标签 标签
            labels = targets[idx][:, -1].data  # 真实标签
            landms = targets[idx][:, 4:14].data  # 真实5点定位
            defaults = anchors.data  # 复制一个
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        zeros = torch.tensor(0, device=device)
        loc_t = loc_t.to(device)
        conf_t = conf_t.to(device)
        landm_t = landm_t.to(device)

        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros  # 做布尔索引
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)  # 先在后面加1维,扩展成索引
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # conf_t全部置1
        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num_batch, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


class DataGenerator(data.Dataset):
    '''
    包含有图片查看的功能
    '''

    def __init__(self, path_file, img_size, mode='train', test=False, look=False):
        '''

        :param path_file: 数据文件的主路径
        :param img_size:  预处理后的尺寸[640,640]
        :param mode:  train val test
        '''
        self.look = look
        self.img_size = img_size
        if test:
            _file_name = '_label.txt'
        else:
            _file_name = 'label.txt'

        self.txt_path = os.path.join(path_file, mode, _file_name)

        if os.path.exists(self.txt_path):
            f = open(self.txt_path, 'r')
        else:
            raise Exception('标签文件不存在: %s' % self.txt_path)

        self.imgs_path = [] # 每一张图片的全路径
        self.words = [] # 每一张图片对应的15维数据 应该在取的时候再弄出来

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
                # lrtb 转lrwh
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

        self.show_face(img, labels)

        # 加快图片解码 pip install jpeg4py -i https://pypi.tuna.tsinghua.edu.cn/simple
        # import jpeg4py as jpeg
        # img = jpeg.JPEG(self.imgs_path[index]).decode()

        #
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
        new_ar = w / h * rand(jitter[0], jitter[1]) / rand(jitter[0], jitter[1])
        scale = rand(0.75, 1.25)  # 在0.25到2之间缩放
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 50% 翻转图像
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域扭曲
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
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


def detection_collate(batch):
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
