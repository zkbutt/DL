import json

from torch.utils.data import Dataset
import os
import torch
import xmltodict
import numpy as np

# from f_tools.pic.f_show import show_od4boxs
# from object_detection.faster_rcnn.draw_box_utils import draw_box
# import matplotlib.pyplot as plt
from PIL import Image


class VOC2012DataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, path_data_root, path_file_txt, path_class_json, transforms=None,
                 bbox2one=False):
        '''

        :param path_data_root: voc数据集的根目录
        :param path_file_txt: 提前准备好的xml及jpg对应的文件名
        :param path_class_json: 提前准备好类别 1 开始
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
            json_file = open(path_class_json, 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        '''

        :param idx:
        :return:
            image 经预处理后的值 torch.Size([3, 375, 500])
            target对象 全tensor值 bndbox name difficult 输出左上右下
        '''
        # ----------解析xml-----------
        path_xml = self.xml_list[idx]
        doc = self.parse_xml(path_xml)  # 解析xml
        path_img = os.path.join(os.path.join(self.path_data_root, 'JPEGImages'), doc['annotation']['filename'])

        image = Image.open(path_img)
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
        target["height_width"] = image.size
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


if __name__ == '__main__':
    pass
