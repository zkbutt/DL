import json
import os

import cv2
import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_loader import cre_transform_resize4pil
from f_tools.datas.f_coco.convert_data.coco_dataset import CustomCocoDataset4cv
from f_tools.fits.fitting.f_fit_eval_base import f_prod_pic4one
# 这里要删除
from f_tools.pic.enhance.f_data_pretreatment4np import cre_transform_resize4np
from object_detection.z_yolov1.CONFIG_YOLOV1 import CFG
from object_detection.z_yolov1.train_yolov1 import init_model, train_eval_set

if __name__ == '__main__':
    '''

    '''
    '''------------------系统配置---------------------'''
    cfg = CFG
    train_eval_set(cfg)
    cfg.PATH_SAVE_WEIGHT = cfg.PATH_HOST + '/AI/weights/feadre'
    cfg.FILE_FIT_WEIGHT = os.path.join(cfg.PATH_SAVE_WEIGHT, cfg.FILE_NAME_WEIGHT)

    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    cfg.THRESHOLD_PREDICT_CONF = 0.4  # 用于预测的阀值
    cfg.THRESHOLD_PREDICT_NMS = 0.5  # 提高 conf 提高召回, 越小框越少
    eval_start = 20
    is_test_dir = True  # 测试dataset 或目录

    # json_file = open(os.path.join(cfg.PATH_DATA_ROOT, 'ids_classes_voc_proj.json'), 'r', encoding='utf-8')
    # ids_classes = json.load(json_file, encoding='utf-8')  # json key是字符

    # 这里是原图
    dataset_test = CustomCocoDataset4cv(
        file_json=cfg.FILE_JSON_TEST,
        path_img=cfg.PATH_IMG_EVAL,
        mode=cfg.MODE_COCO_EVAL,
        transform=None,
        is_mosaic=False,
        is_mosaic_keep_wh=False,
        is_mosaic_fill=False,
        is_debug=cfg.DEBUG,
        cfg=cfg
    )

    data_transform = cre_transform_resize4np(cfg)['val']
    ids_classes = dataset_test.ids_classes
    labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
    labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
    flog.debug('测试类型 %s', labels_lsit)

    '''------------------模型定义---------------------'''
    model, _, _, _ = init_model(cfg, device, id_gpu=None)  # model, optimizer, lr_scheduler, start_epoch
    model.eval()

    if is_test_dir:
        path_img = r'D:\tb\tb\ai_code\DL\_test_pic\dog_cat_bird'
        file_names = os.listdir(path_img)
        for name in file_names:
            file_img = os.path.join(path_img, name)
            img_np = cv2.imread(file_img)
            f_prod_pic4one(img_np=img_np, data_transform=data_transform, model=model,
                           size_ts=torch.tensor(img_np.shape[:2][::-1]),
                           labels_lsit=labels_lsit)
    else:
        # for i in range(start=eval_start, stop=len(dataset_test), step=1):
        for i in range(eval_start, len(dataset_test), 1):
            img_np = dataset_test[i][0]
            target = dataset_test[i][1]
            f_prod_pic4one(img_np, data_transform, model, target['size'], labels_lsit, gboxes_ltrb=target['boxes'],
                           target=target)
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
