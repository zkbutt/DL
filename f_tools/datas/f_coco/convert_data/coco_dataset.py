import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.f_coco.coco_api import f_show_coco_pics, f_open_cocoimg
from f_tools.fun_od.f_boxes import ltwh2ltrb
from f_tools.pic.enhance.f_mosaic import f_mosaic_pics_ts
from f_tools.pic.f_show import show_bbox4pil


class CocoDataset(Dataset):

    def __init__(self, path_root, mode, data_type, transform=None,
                 is_mosaic=False, is_debug=False, cfg=None):
        '''

        :param path_root:
        :param mode: bbox keypoints  ['segm', 'bbox', 'keypoints']
            # bbox segm keypoints caption
        :param data_type:
        :param transform:
        :param out_type: 输出格式 ts:tensor np:ndarray
        '''
        # path_save_img = os.path.join(dataset.path_root, 'images', dataset.data_type)
        self.path_root, self.data_type = path_root, data_type
        self.transform = transform
        self.mode = mode
        if self.mode == 'bbox':
            name_file = 'instances'
        elif self.mode == 'keypoints':
            name_file = 'person_keypoints'
        else:
            raise Exception('mode 错误', self.mode)

        file_json = '{}/annotations/{}_{}.json'.format(path_root, name_file, data_type)
        if not os.path.exists(file_json):
            raise Exception('coco标注文件不存在', file_json)
        self.coco = COCO(file_json)

        # debug code
        # path_img = os.path.join(path_root, 'images', data_type)
        # f_show_coco_pics(self.coco, path_img, ids_img=[279])

        self.image_ids = self.coco.getImgIds()  # 所有图片的id 画图数量

        #  self.classes_ids self.classes_ids
        self._init_load_classes()  # 除了coco数据集,其它不管

        self.is_debug = is_debug
        self.is_mosaic = is_mosaic
        self.cfg = cfg

    def __len__(self):
        # flog.debug('__len__ %s', )
        if self.is_debug:
            return 10
        if self.is_mosaic:
            return len(self.image_ids) // 4
        return len(self.image_ids)

    def open_img_tar(self, index):
        img_pil = self.load_image(index)
        image_id = self.image_ids[index]
        # bboxs, labels,keypoints

        tars_ = self.load_anns(index)

        # 动态构造target
        target = {}
        l_ = ['boxes', 'labels', 'keypoints']
        target['image_id'] = image_id
        target['size'] = img_pil.size  # (w,h)
        # 根据标注模式 及字段自动添加 target['boxes', 'labels', 'keypoints']
        for i, tar in enumerate(tars_):
            target[l_[i]] = tar
        return img_pil, target

    def do_mosaic(self, idx):
        imgs = []
        boxs = []
        labels = []
        for i in range(idx, idx + 4):
            img_pil, target = self.open_img_tar(i)
            imgs.append(img_pil)  # list(img_pil)
            boxs.append(target["boxes"])
            labels.append(target["labels"])
        img_pil_mosaic, boxes_mosaic, labels = f_mosaic_pics_ts(imgs, boxs, labels,
                                                                self.cfg.IMAGE_SIZE, is_keep_wh=False)
        target = {}
        target["boxes"] = boxes_mosaic  # 输出左上右下
        target["labels"] = labels
        return img_pil_mosaic, target

    def __getitem__(self, index):
        '''

        :param index:
        :return: tensor or np.array 根据 out: 默认ts or other is np
            img: h,w,c
            target: dict{
                image_id: int,
                bboxs: ts n4 ltrb
                labels: ts n,
                keypoints: ts n,10
                size: wh
            }
        '''
        if self.is_mosaic and self.mode == 'bbox':
            img_pil, target = self.do_mosaic(index)
        else:
            img_pil, target = self.open_img_tar(index)

        '''---------------cocoAPI测试 查看图片在归一化前------------------'''
        if self.cfg.IS_VISUAL_PRETREATMENT:
            show_bbox4pil(img_pil, target['boxes'])
            path_img = os.path.join(self.path_root, 'images', self.data_type)
            # is_mosaic 这个用不起
            f_show_coco_pics(self.coco, path_img, ids_img=[index])

        if target['boxes'].shape[0] == 0:
            # flog.debug('重新加载 %s', index)
            return self.__getitem__(index + 1)

        if self.transform is not None:
            # 预处理输入 PIL img 和 np的target
            img_pil, target = self.transform(img_pil, target)

        # target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float)
        # target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['size'] = torch.tensor(target['size'], dtype=torch.float)  # 用于恢复真实尺寸
        # if self.mode == 'keypoints':
        #     target['keypoints'] = torch.tensor(target['keypoints'], dtype=torch.float)

        if target['boxes'].shape[0] == 0:
            flog.debug('二次检查出错 %s', index)
            return self.__getitem__(index + 1)
        return img_pil, target

    def load_image(self, index):
        '''

        :param index:
        :return:
        '''

        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        path_img = os.path.join(self.path_root, 'images', self.data_type, image_info['file_name'])
        img_pil = Image.open(path_img).convert('RGB')  # 原图数据
        return img_pil

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
            # skip the annotations with width or height < 1
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                # flog.warning('标记框有问题 %s', a)
                continue

            labels.append(self.ids_old_new[a['category_id']])
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
        ltwh2ltrb(bboxs, safe=False)
        if bboxs.shape[0] == 0:
            flog.error('这图标注 不存在 %s', coco_anns)
            # raise Exception('这图标注 不存在 %s', coco_anns)

        if self.mode == 'bbox':
            return [torch.tensor(bboxs, dtype=torch.float), torch.tensor(labels, dtype=torch.int64)]
        elif self.mode == 'keypoints':
            return [torch.tensor(bboxs, dtype=torch.float),
                    torch.tensor(labels, dtype=torch.int64),
                    torch.tensor(keypoints, dtype=torch.float)]

    def _init_load_classes(self):
        '''
        self.classes_ids :  {'Parade': 1}
        self.ids_classes :  {1: 'Parade'}
        self.ids_new_old {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        self.ids_old_new {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        :return:
        '''
        # [{'id': 1, 'name': 'aeroplane'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'bird'}, {'id': 4, 'name': 'boat'}, {'id': 5, 'name': 'bottle'}, {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'car'}, {'id': 8, 'name': 'cat'}, {'id': 9, 'name': 'chair'}, {'id': 10, 'name': 'cow'}, {'id': 11, 'name': 'diningtable'}, {'id': 12, 'name': 'dog'}, {'id': 13, 'name': 'horse'}, {'id': 14, 'name': 'motorbike'}, {'id': 15, 'name': 'person'}, {'id': 16, 'name': 'pottedplant'}, {'id': 17, 'name': 'sheep'}, {'id': 18, 'name': 'sofa'}, {'id': 19, 'name': 'train'}, {'id': 20, 'name': 'tvmonitor'}]
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])  # 按id升序 [{'id': 1, 'name': 'Parade'}]

        # coco ids is not from 1, and not continue ,make a new index from 0 to 79, continuely
        # 重建index 从1-80
        # classes_ids:   {names:      new_index}
        # coco_ids:  {new_index:  coco_index}
        # coco_ids_inverse: {coco_index: new_index}
        self.classes_ids, self.ids_new_old, self.ids_old_new = {}, {}, {}
        # 解决中间有断格的情况
        for c in categories:  # 修正从1开始
            self.ids_new_old[len(self.classes_ids) + 1] = c['id']
            self.ids_old_new[c['id']] = len(self.classes_ids) + 1
            self.classes_ids[c['name']] = len(self.classes_ids) + 1  # 这个是新索引 {'Parade': 0}

        # ids_classes: {new_index:  names}
        self.ids_classes = {}
        for k, v in self.classes_ids.items():
            self.ids_classes[v] = k


if __name__ == '__main__':
    path_root = r'M:\temp\11\voc_coco\coco'  # 自已的数据集
    data_type = 'val2017'  # 自动会添加 imgages
    mode = 'bbox'  # bbox segm keypoints caption
    path_img = os.path.join(path_root, 'images', data_type)

    dataset = CocoDataset(path_root, mode, data_type)
    coco = dataset.coco

    dataset_ = dataset[1]
    img_pil = f_open_cocoimg(path_img, coco, img_id=4)
    # img_pil.show()

    flog.debug(coco.loadCats(coco.getCatIds()))  # 获取数据集类别数
    # f_show_coco_pics(coco, path_img)
