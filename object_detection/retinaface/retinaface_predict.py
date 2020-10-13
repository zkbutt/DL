import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from object_detection.retinaface.nets.retinaface import RetinaFace
from object_detection.retinaface.utils.config import cfg_mnet, cfg_re50
from object_detection.retinaface.utils.anchorsfound import AnchorsFound
from object_detection.retinaface.utils.box_utils import decode, decode_landm, non_max_suppression


def preprocess_input(image):
    # 根据数据集统一处理
    image -= np.array((104, 117, 123), np.float32)
    return image


class Retinaface(object):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def detect_image(self, image):
        '''
        检测图片
        :param image:
        :return:
        '''
        old_image = image.copy()  # 用于后续 绘制人脸框 使用

        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)

        # 训练归一化参数 各通道减 并转换
        image = preprocess_input(image).transpose(2, 0, 1)
        # 增加batch_size维度
        image = torch.from_numpy(image).unsqueeze(0)  # 最前面增加一维 可用 image[None]
        # 生成 所有比例anchors
        anchors = AnchorsFound(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():

            loc, conf, landms = self.net(image)  # forward pass

            scale = torch.Tensor([im_width, im_height] * 2)
            scale_for_landmarks = torch.Tensor([im_width, im_height] * 5)
            if self.cuda:
                # 它的作用是将归一化后的框坐标转换成原图的大小
                scale = scale.cuda()
                scale_for_landmarks = scale_for_landmarks.cuda()
                image = image.cuda()
                anchors = anchors.cuda()

            # 删除 1维 <class 'tuple'>: (37840, 4)
            _squeeze = loc.data.squeeze(0)
            boxes = decode(_squeeze, anchors, self.cfg['variance'])
            boxes = boxes * scale  # 0-1比例 转换到原图
            boxes = boxes.cpu().numpy()

            # <class 'tuple'>: (37840, 10)
            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            landms = landms * scale_for_landmarks
            landms = landms.cpu().numpy()

            # 取其中index1  得一维数组
            conf = conf.data.squeeze(0)[:, 1:2].cpu().numpy()  # 取出人脸概率  index0为背景 index1为人脸

            # 元素横叠(行方向)
            cat = torch.cat([boxes, conf, landms], dim=-1)
            boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)
            # 同一类型,分数排序, iou大于某阀值的 全部剔除, 然后继续
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return old_image
