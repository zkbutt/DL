import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np

from f_tools.GLOBAL_LOG import flog
from object_detection.coco_t.coco_api import t_coco_pic, show_od4coco


class CocoDataset(Dataset):
    def __init__(self, path_root, mode, data_type, device=torch.device("cpu"), transform=None, out='ts'):
        '''

        :param path_root:
        :param mode: bboxs keypoints
        :param data_type:
        :param device:
        :param transform:
        :param out: 输出格式 ts:tensor np:ndarray
        '''
        # path_save_img = os.path.join(dataset.path_root, 'images', dataset.data_type)
        self.path_root, self.data_type = path_root, data_type
        self.device = device
        self.out = out
        self.transform_cpu = transform
        self.mode = mode
        if self.mode == 'bboxs':
            name_file = 'instances'
        elif self.mode == 'keypoints':
            name_file = 'person_keypoints'
        else:
            raise Exception('mode 错误', self.mode)
        path_file = '{}/annotations/{}_{}.json'.format(path_root, name_file, data_type)
        self.coco = COCO(path_file)
        self.image_ids = self.coco.getImgIds()  # 所有图片的id

        self._load_classes()

    def _load_classes(self):
        '''
        self.classes_ids :  {'Parade': 1}
        self.ids_classes :  {1: 'Parade'}

        :return:
        '''
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])  # 按id升序 [{'id': 1, 'name': 'Parade'}]

        # coco ids is not from 1, and not continue ,make a new index from 0 to 79, continuely
        # 重建index 从1-80
        # classes_ids:   {names:      new_index}
        # coco_ids:  {new_index:  coco_index}
        # coco_ids_inverse: {coco_index: new_index}
        self.classes_ids, self.ids_new_old, self.ids_old_new = {}, {}, {}
        for c in categories:  # 修正从1开始
            self.ids_new_old[len(self.classes_ids) + 1] = c['id']
            self.ids_old_new[c['id']] = len(self.classes_ids) + 1
            self.classes_ids[c['name']] = len(self.classes_ids) + 1  # 这个是新索引 {'Parade': 0}

        # ids_classes: {new_index:  names}
        self.ids_classes = {}
        for k, v in self.classes_ids.items():
            self.ids_classes[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        '''

        :param index:
        :return: tensor or np.array 根据 out: 默认ts or other is np
            img: h,w,c
            target: dict{
                image_id: int,
                bboxs: np(num_anns, 4),
                labels: np(num_anns),
                keypoints: np(num_anns,10),
            }
        '''
        img = self.load_image(index)
        image_id = self.image_ids[index]
        # bboxs, labels,keypoints

        # self.coco.loadImgs(image_id)[0]  # 图片基本信息
        # self.coco.loadAnns(3)[0]  # ann信息

        tars_ = self.load_anns(index)

        # 动态构造target
        target = {}
        l_ = ['bboxs', 'labels', 'keypoints']
        target['image_id'] = image_id
        for i, tar in enumerate(tars_):
            target[l_[i]] = tar

        if self.transform_cpu:
            # 预处理输入 PIL img 和 np的target
            img, target = self.transform_cpu(img, target)

        '''---------------cocoAPI测试 查看图片在归一化前------------------'''
        # cats = self.coco.loadCats(target['labels'])
        # ann_ids = self.coco.getAnnIds(target['image_id'])
        # anns = self.coco.loadAnns(ann_ids)
        # img_info = self.coco.loadImgs(target['image_id'])[0]
        # flog.debug(' %s', img_info)
        # show_od4coco(img, target, self.coco)
        # from f_tools.pic.f_show import show_pics_ts
        # 归一化后用这个查看图
        # show_pics_ts(img[None])

        if self.out == 'ts':
            # target['image_id'] = torch.tensor(image_id)
            img = img.to(self.device)
            for key, val in target.items():
                target[key] = torch.tensor(val, device=self.device).type(torch.float)

        return img, target

    def load_image(self, index):
        '''

        :param index:
        :return:
        '''
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        path_img = os.path.join(self.path_root, 'images', self.data_type,
                                image_info['file_name'])
        img = Image.open(path_img)  # 原图数据
        return img

    def load_anns(self, index):
        '''
        ltwh --> ltrb
        :param index:
        :return:
            bboxs: np(num_anns, 4)
            labels: np(num_anns)
        '''
        # annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        annotation_ids = self.coco.getAnnIds(self.image_ids[index])  # ann的id
        # anns is num_anns x 4, (x1, x2, y1, y2)
        bboxs = np.zeros((0, 4), dtype=np.float32)  # np创建 空数组 高级
        labels = []
        keypoints = np.zeros((0, 10), dtype=np.float32)
        # skip the image without annoations
        if len(annotation_ids) == 0:
            return bboxs, labels

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            labels.append(self.ids_old_new[a['category_id']])

            # skip the annotations with width or height < 1
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            bbox = np.zeros((1, 4), dtype=np.float32)
            bbox[0, :4] = a['bbox']
            bboxs = np.append(bboxs, bbox, axis=0)

            if self.mode == 'keypoints':
                k_ = np.array(a['keypoints'])
                inds = np.arange(2, len(k_), 3)
                ones = np.ones(len(a['keypoints']), dtype=np.float32)
                ones[inds] = 0
                nonzero = np.nonzero(ones)  # 取非零索引
                k_ = k_[nonzero]

                keypoints = np.append(keypoints, k_[None,], axis=0)

        # ltwh --> ltrb
        bboxs[:, 2] += bboxs[:, 0]
        bboxs[:, 3] += bboxs[:, 1]

        if self.mode == 'bboxs':
            return [bboxs, np.array(labels)]
        elif self.mode == 'keypoints':
            return [bboxs, np.array(labels), keypoints]


if __name__ == '__main__':
    path_root = r'd:\t001\coco'  # 自已的数据集
    # data_type = 'val2017'  # 自动会添加 imgages
    data_type = 'train2017'
    mode = 'keypoints'

    from f_tools.datas.data_pretreatment import Compose, RandomHorizontalFlip, ToTensor

    data_transform = {
        "train": Compose([RandomHorizontalFlip(1)]),
        "val": Compose([ToTensor()])
    }

    dataset = CocoDataset(path_root, mode, data_type)
    # dataset = CoCoDataset(path_root, mode, data_type, transform=data_transform['train'])
    # plt.imshow(sample['img'])
    coco = dataset.coco
    flog.debug(coco.loadCats(coco.getCatIds())) # 获取数据集类别数
    path_save_img = os.path.join(dataset.path_root, 'images', dataset.data_type)

    t_coco_pic(coco, path_save_img)
