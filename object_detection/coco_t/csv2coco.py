import json
import numpy as np
import pandas as pd
import cv2
import os
import shutil
from tqdm import tqdm

from f_tools.GLOBAL_LOG import flog

np.random.seed(41)


class Csv2CoCo:

    def __init__(self, image_dir, total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_instances(self, keys):
        self._init_categories()
        for key in tqdm(keys, desc='构建标注'):
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi, label)
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
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        _path = os.path.join(self.image_dir, path)
        # flog.debug('_image %s', _path)
        img = cv2.imread(_path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
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


def to_coco_instances(csv_file, image_dir, saved_coco_path,
                      is_copy=True, is_move=False,
                      type='train2017'):
    '''
    coco ann ltwh
    :param csv_file: csv标注
    :param saved_coco_path: 这个是生成的根
    :param image_dir: 图片所在的根目录
    :param is_copy: 复制文件
    :param is_move: 是否移动
    :param type:  train2017  val2017
    :return:
    '''
    if is_move:
        fun_copy = shutil.move
    else:
        fun_copy = shutil.copy

    annotations = pd.read_csv(csv_file, header=None).values
    # 重构csv格式标注文件
    file_annotations_dict = {}
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]  # 取文件名
        value = np.array([annotation[1:]])  # 取bbox
        if key in file_annotations_dict.keys():
            file_annotations_dict[key] = np.concatenate((file_annotations_dict[key], value), axis=0)
        else:
            file_annotations_dict[key] = value

    file_names = list(file_annotations_dict.keys())
    flog.debug("file_names数量:%s", len(file_names))

    # 创建必须的文件夹
    path_annotations = '%scoco/annotations/' % saved_coco_path
    if not os.path.exists(path_annotations):
        os.makedirs(path_annotations)

    # 把训练集转化为COCO的json格式
    csv_coco = Csv2CoCo(image_dir=image_dir, total_annos=file_annotations_dict)
    # ---------------这里是转换类--------------
    instance = csv_coco.to_instances(file_names)
    csv_coco.save_coco_json(instance, '%scoco/annotations/instances_%s.json' % (saved_coco_path, type))
    flog.debug('标注文件制作完成 %s', )

    if is_copy:
        path_save_img = '%scoco/images/%s/' % (saved_coco_path, type)
        if not os.path.exists(path_save_img):
            os.makedirs(path_save_img)
        with tqdm(total=len(file_names), desc=f'复制文件', postfix=dict, mininterval=0.3) as pbar:
            for file in file_names:
                src = os.path.join(image_dir, file)
                dst = path_save_img
                __s = fun_copy(src, dst)
                # flog.debug('复制成功 %s', __s)
                pbar.set_postfix(**{'当前文件': __s})
                pbar.update(1)


if __name__ == '__main__':
    # 0为背景
    f = open('classes1.json', 'r', encoding='utf-8')
    # f = open('classes.json', 'r', encoding='utf-8')
    classname_to_id = json.loads(json.load(f, encoding='utf-8'), encoding='utf-8')

    mode = 'val2017'
    # mode = 'train2017'

    csv_file = "./file/csv_labels.csv"
    image_dir = r'D:\down\AI\datas\widerface\val\images'  # 真实图片路径
    saved_coco_path = "d:/t001/"  # 这个是生成的根
    to_coco_instances(csv_file, image_dir, saved_coco_path,
                      is_copy=True, is_move=False, type='val2017')
    flog.info('数据生成成功 %s', )
