import json
import numpy as np
import pandas as pd
import cv2
import os
import shutil
from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog


class Csv2CocoInstances:

    def __init__(self, path_img, classname_to_id, file_annotations_dict):
        '''

        :param path_img:
        :param file_annotations_dict:
        '''
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.path_img = path_img
        self.file_annotations_dict = file_annotations_dict
        self.classname_to_id = classname_to_id

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco_obj(self, keys):
        self._init_categories()  # 构建类别唯一信息
        for key in tqdm(keys, desc='内存构建标注'):
            self.images.append(self._image(key))
            shapes = self.file_annotations_dict[key]
            for shape in shapes:

                annc = []
                for cor in shape[:-1]:  # 形成bbox
                    annc.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(annc, label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in self.classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        _path = os.path.join(self.path_img, path)
        # flog.debug('_image %s', _path)
        img = cv2.imread(_path)
        if img is None:
            raise Exception('_image目标不存在 %s', _path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, annc, label):
        # label = shape[-1]
        points = annc[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(self.classname_to_id[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0  # 0表示polygon格式  1表示RLE格式
        annotation['area'] = self._get_area(points)

        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        '''
        ltrb -> ltwh
        :param points:
        :return:
        '''
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    # 计算面积
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x + 1) * (max_y - min_y + 1)

    # segmentation
    def _get_seg(self, points):
        '''
        ltrb -> lrwh 选框
        :param points: ltrb
        :return:
        '''
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x, min_y,
                  min_x, min_y + 0.5 * h,
                  min_x, max_y,
                  min_x + 0.5 * w, max_y,
                  max_x, max_y,
                  max_x, max_y - 0.5 * h,
                  max_x, min_y,
                  max_x - 0.5 * w, min_y])
        return a


class Csv2CocoKeypoints(Csv2CocoInstances):

    def __init__(self, path_img, classname_to_id, file_annotations_dict, name_keypoints, skeleton):
        '''

        :param path_img:
        :param file_annotations_dict:
        :param name_keypoints:
            [
                "nose", "left_eye", "right_eye",
                "left_ear", "right_ear",
                "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle"
            ]
        :param skeleton:
            [
                [16, 14], [14, 12], [17, 15], [15, 13],
                [12, 13], [6, 12], [7, 13], [6, 7],
                [6, 8], [7, 9], [8, 10], [9, 11],
                [2, 3], [1, 2], [1, 3], [2, 4],
                [3, 5], [4, 6], [5, 7]
            ]
        '''
        super().__init__(path_img, classname_to_id, file_annotations_dict)
        self.name_keypoints = name_keypoints
        self.skeleton = skeleton

    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k

            # 每一个类型的关键点是否一样
            category['keypoints'] = self.name_keypoints
            category['skeleton'] = self.skeleton
            self.categories.append(category)

    def _annotation(self, annc, label):
        '''

        :param annc:
        :param label:
        :return:
        '''
        annotation = super()._annotation(annc, label)
        # x,y,v  v=0:没有  v=1:标了不可见 v=2标了可见
        num_keypoints = 0
        for a in annc[6::3]:
            if a != 0:
                num_keypoints += 1
        annotation['keypoints'] = annc[4:]
        annotation['num_keypoints'] = num_keypoints
        return annotation


def to_coco(file_csv, classname_to_id, path_img, path_coco_save, mode,
            is_copy=True, is_move=False, type='train2017'):
    '''
    coco ann ltwh
    :param file_csv: csv标注
    :param classname_to_id:
    :param path_img: 图片所在的根目录
    :param path_coco_save: 这个是生成的根
    :param mode:
    :param is_copy: 复制文件
    :param is_move: 是否移动
    :param type:  train2017  val2017
    :return:
    '''
    if is_move:
        fun_copy = shutil.move
    else:
        fun_copy = shutil.copy

    annotations = pd.read_csv(file_csv, header=None).values
    # 重构csv格式标注文件
    file_annotations_dict = {}
    if mode == 'bboxs' and annotations[0].shape[0] != 6:
        raise Exception('加载csv格式出错 mode=%s value=%s' % (mode, annotations[0]))
    if mode == 'keypoints' and annotations[0].shape[0] != 21:
        raise Exception('加载csv格式出错 mode=%s value=%s' % (mode, annotations[0]))
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]  # 取文件名
        value = np.array([annotation[1:]])  # 取bbox

        # assert mode == 'bboxs' and len(value) == 5, '加载csv格式出错 value=%s' % value
        # assert mode == 'keypoints' and len(value) == 16, '加载csv格式出错 value=%s' % value
        if key in file_annotations_dict.keys():
            file_annotations_dict[key] = np.concatenate((file_annotations_dict[key], value), axis=0)
        else:
            file_annotations_dict[key] = value

    file_names = list(file_annotations_dict.keys())
    flog.debug("file_names数量:%s", len(file_names))

    # 创建必须的文件夹
    path_annotations = os.path.join(path_coco_save, 'coco/annotations')
    if not os.path.exists(path_annotations):
        os.makedirs(path_annotations)

    if mode == 'bboxs':
        # 把训练集转化为COCO的json格式
        name_file = 'instances'

        '''
        path_img: 'M:\\AI\\datas\\widerface\\val\\images' #图片所在的根目录
        classname_to_id: {'human_face': 1} 
        file_annotations_dict: dict(文件名:np(框个数,5)...)   5是 bbox4+lable1
        '''
        coco_generate = Csv2CocoInstances(path_img, classname_to_id, file_annotations_dict)
    elif mode == 'keypoints':
        name_file = 'person_keypoints'

        name_keypoints = [
            "left_eye", "right_eye", "nose",
            "left_mouth", "right_mouth",
        ]
        skeleton = [[3, 1], [3, 2], [3, 4], [3, 5]]
        coco_generate = Csv2CocoKeypoints(path_img, classname_to_id, file_annotations_dict, name_keypoints, skeleton)
    else:
        raise Exception('mode 错误', mode)

    # ---------------这里是转换类--------------
    coco_obj = coco_generate.to_coco_obj(file_names)
    path_csv = os.path.join(path_annotations, '%s_%s.json' % (name_file, type))
    coco_generate.save_coco_json(coco_obj, path_csv)
    flog.debug('标注文件制作完成 %s', )

    if is_copy:
        path_save_img = os.path.join(path_coco_save, 'coco/images', type)
        if not os.path.exists(path_save_img):
            os.makedirs(path_save_img)
        with tqdm(total=len(file_names), desc=f'复制文件', postfix=dict, mininterval=0.3) as pbar:
            for file in file_names:
                src = os.path.join(path_img, file)
                dst = path_save_img
                __s = fun_copy(src, dst)
                # flog.debug('复制成功 %s', __s)
                pbar.set_postfix(**{'当前文件': __s})
                pbar.update(1)


if __name__ == '__main__':
    '''
    
    '''
    np.random.seed(20200925)

    path_classname = '../_file/classes_ids_voc.json'  # 0为背景
    file_csv = "../_file/csv_labels_voc.csv"
    type = 'val2017'  # train2017
    # path_img = r'M:\AI\datas\widerface\val\images'  # 真实图片路径
    path_img = r'M:\AI\datas\widerface\val\images'  # 真实图片路径
    mode = 'bboxs'  # 'keypoints'  # 'bboxs':
    path_coco_save = r"M:\AI\datas\widerface"  # 这个是生成的根

    with open(path_classname, 'r', encoding='utf-8') as f:
        classname_to_id = json.load(f)  # 文件转dict 或list

    to_coco(file_csv, classname_to_id, path_img, path_coco_save, mode, is_copy=True, is_move=False, type=type)

    flog.info('数据生成成功 %s', )