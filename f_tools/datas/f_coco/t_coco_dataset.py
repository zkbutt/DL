from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import os

from f_tools.GLOBAL_LOG import flog
from f_tools.pic.f_show import f_show_3box4pil


def data_clean(coco_obj, coco_img_ids, catid2clsid, path_img=None):
    '''
    用于加载coco 查看和清理无效GT 后用于训练
    :param coco_obj:
    :param coco_img_ids:
    :param catid2clsid: {1: 0, 2: 1, 3: 2}
    :param path_img:
    :return:
    '''
    records = []
    ct = 0
    for img_id in coco_img_ids:
        img_info = coco_obj.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        im_w = float(img_info['width'])
        im_h = float(img_info['height'])

        ins_anno_ids = coco_obj.getAnnIds(imgIds=img_id, iscrowd=False)  # 读取这张图片所有标注anno的id
        instances = coco_obj.loadAnns(ins_anno_ids)  # 这张图片所有标注anno。每个标注有'segmentation'、'bbox'、...

        coco_targets = []
        anno_id = []  # 注解id
        for inst in instances:
            # print("inst['bbox']", inst['bbox'])
            # 确保 ltwh 点在图片中  coco默认 ltwh
            x, y, box_w, box_h = inst['bbox']  # 读取物体的包围框
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(im_w - 1, x1 + max(0, box_w - 1))
            y2 = min(im_h - 1, y1 + max(0, box_h - 1))  # ltwh -> ltrb
            if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                inst['clean_bbox'] = [x1, y1, x2, y2]  # inst增加一个键值对
                # print("inst['clean_bbox']", inst['clean_bbox'])
                coco_targets.append(inst)  # 这张图片的这个物体标注保留
                anno_id.append(inst['id'])
            else:
                flog.warning(
                    'Found an invalid bbox in annotations: im_id: {}, '
                    'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                        img_id, float(inst['area']), x1, y1, x2, y2))

        num_bbox = len(coco_targets)  # 这张图片的物体数

        # 左上角坐标+右下角坐标+类别id
        gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
        gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_score = np.ones((num_bbox, 1), dtype=np.float32)  # 得分的标注都是1
        is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
        difficult = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_poly = [None] * num_bbox

        for i, target in enumerate(coco_targets):
            catid = target['category_id']
            gt_class[i][0] = catid2clsid[catid]  # id类型顺序转换
            gt_bbox[i, :] = target['clean_bbox']
            is_crowd[i][0] = target['iscrowd']
            if 'segmentation' in target:
                gt_poly[i] = target['segmentation']  # 分割

        if path_img is not None:
            file_img = os.path.join(path_img, file_name)
            img_pil = Image.open(file_img).convert('RGB')  # 原图数据
            f_show_3box4pil(img_pil, gt_bbox, is_oned=False)

        coco_rec = {
            'im_file': file_name,
            'im_id': np.array([img_id]),
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'anno_id': anno_id,
            'gt_bbox': gt_bbox,
            'gt_score': gt_score,
            'gt_poly': gt_poly,
        }

        # 显示文件信息
        # flog.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(file_names, img_id, im_h, im_w))
        records.append(coco_rec)  # 注解文件。
        ct += 1
    flog.info('加载 {} 个图片 .'.format(ct))
    return records


if __name__ == '__main__':
    path_host = 'M:'
    path_root = path_host + r'/AI/datas/VOC2007'
    # file_json = path_root + '/coco/annotations/instances_type3_train_1096.json'
    file_json = path_root + '/coco/annotations/instances_train_5011.json'
    path_img = path_root + '/train/JPEGImages'

    # s_ids_cats = [3, 8, 12]  # 只选部分
    s_ids_cats = [1, 2, 5, 14]  # 只选部分
    # s_ids_cats = []  # 只选部分

    coco_obj = COCO(file_json)
    if not s_ids_cats:
        s_ids_cats = coco_obj.getCatIds()

    coco_img_ids = []
    idcat2idcls = {}  # _catid2clsid = {1: 0, 2: 1, 3: 2}
    classnames = []
    for i, id_cat in enumerate(s_ids_cats):
        cats_info = coco_obj.loadCats(ids=id_cat)[0]
        idcat2idcls[cats_info['id']] = i
        classnames.append(cats_info['name'])
        ids = coco_obj.getImgIds(catIds=id_cat)
        coco_img_ids.extend(ids)
        print(cats_info['name'], len(ids))

    print(classnames)
    coco_img_ids = list(set(coco_img_ids))
    print('共 %s 图' % len(coco_img_ids))

    # {'im_file': 'M:/AI/datas/VOC2007/train/JPEGImages\\008138.jpg', 'im_id': array([0]), 'h': 333.0, 'w': 500.0, 'is_crowd': array([[0]]), 'gt_class': array([[0]]), 'anno_id': [0], 'gt_bbox': array([[ 47.,  44., 389., 243.]], dtype=float32), 'gt_score': array([[1.]], dtype=float32), 'gt_poly': [[[47, 44, 47, 144.0, 47, 244, 218.5, 244, 390, 244, 390, 144.0, 390, 44, 218.5, 44]]]}
    # train_records = data_clean(coco_obj, coco_img_ids, idcat2idcls, path_img)
    print()
