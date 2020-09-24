import os

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np


class SimpleCoCoDataset(Dataset):
    def __init__(self, path_root, data_type='val2017', transform=None):
        self.path_root, self.data_type = path_root, data_type
        self.transform = transform
        path_file = '{}/annotations/instances_{}.json'.format(path_root, data_type)
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
        :return:
            img: h,w,c
            bboxs: np(num_anns, 4)
            labels: np(num_anns)
            image_id: int
        '''
        img = self.load_image(index)
        bboxs, labels = self.load_anns(index)
        image_id = self.image_ids[index]
        # self.coco.loadImgs(image_id)[0]  # 图片基本信息
        # self.coco.loadAnns(3)[0]  # ann信息
        sample = {
            'img': img,
            'bboxs': bboxs,
            'labels': labels,
            'image_id': image_id,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, index):
        '''

        :param index:
        :return:
        '''
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        path_img = os.path.join(self.path_root, 'images', self.data_type,
                                image_info['file_name'])
        img = Image.open(path_img)  # 原图数据
        return np.array(img)

    def load_anns(self, index):
        '''

        :param index:
        :return:
            bboxs: np(num_anns, 4)
            labels: np(num_anns)
        '''
        # annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        annotation_ids = self.coco.getAnnIds(self.image_ids[index])  # ann的id
        # anns is num_anns x 4, (x1, x2, y1, y2)
        bboxs = np.zeros((0, 4))  # np创建 高级
        labels = []
        # skip the image without annoations
        if len(annotation_ids) == 0:
            return bboxs, labels

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            labels.append(self.ids_old_new[a['category_id']])

            # skip the annotations with width or height < 1
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            bbox = np.zeros((1, 4))
            bbox[0, :4] = a['bbox']
            bboxs = np.append(bboxs, bbox, axis=0)

        # ltwh --> ltrb
        bboxs[:, 2] += bboxs[:, 0]
        bboxs[:, 3] += bboxs[:, 1]
        return bboxs, np.array(labels)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path_root = r'd:\t001\coco'  # 自已的数据集
    data_type = 'val2017'  # 自动会添加 imgages
    # dataType = 'train2017'

    dataset = SimpleCoCoDataset(path_root, data_type)
    sample = dataset.__getitem__(2)
    plt.imshow(sample['img'])
    coco = dataset.coco

    id_ann = coco.getAnnIds(sample['image_id'])
    anns = coco.loadAnns(id_ann)
    coco.showAnns(anns)
    plt.show()
    pass
