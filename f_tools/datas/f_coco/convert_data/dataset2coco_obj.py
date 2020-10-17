from pycocotools.coco import COCO


def voc2coco_obj(ds):
    '''
    coco_obj没有路径
    :param ds:  from torch.utils.data import Dataset
    :return:
    '''
    coco_gt = COCO()
    # annotation IDs need to start at 1, not 0
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        # img_dict['height'] = img.shape[-2]
        # img_dict['width'] = img.shape[-1]
        img_dict['height'] = targets["height_width"][0]
        img_dict['width'] = targets["height_width"][1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        # 将box的相对坐标信息（0-1）转为绝对值坐标
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_dict["width"]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_dict["height"]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        # 注意这里的boxes area也要进行转换，否则导致(small, medium, large)计算错误
        areas = (targets['area'] * img_dict["width"] * img_dict["height"]).tolist()
        iscrowd = targets['iscrowd'].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    return coco_gt

def widerface2coco_obj(ds):
    coco_gt = COCO()
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        img_np, targets = ds[img_idx]
        img_dict = {}
        img_dict['id'] = img_idx
        # img_dict['height'] = img.shape[-2]
        # img_dict['width'] = img.shape[-1]
        img_dict['height'] = targets["height_width"][0]
        img_dict['width'] = targets["height_width"][1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        # 将box的相对坐标信息（0-1）转为绝对值坐标
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_dict["width"]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_dict["height"]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        # 注意这里的boxes area也要进行转换，否则导致(small, medium, large)计算错误
        areas = (targets['area'] * img_dict["width"] * img_dict["height"]).tolist()
        iscrowd = targets['iscrowd'].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1

    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    return coco_gt