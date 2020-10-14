import os

import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import Resize, ResizeKeep
from f_tools.f_general import get_path_root
from object_detection.f_fit_tools import load_weight
from object_detection.ssd import p_transform4ssd
from object_detection.ssd.CONFIG_SSD import NUM_CLASSES, PATH_FIT_WEIGHT, PATH_DATA_ROOT
from object_detection.ssd.draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
from object_detection.ssd.src.ssd_model import SSD300, Backbone


def create_model(num_classes):
    backbone = Backbone()
    # 看 PostProcess
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    flog.info('device %s', device)

    '''---------------数据加载及处理--------------'''
    # original_img = Image.open(os.path.join(get_path_root(), 'object_detection/pic_test/t1.jpg'))
    # original_img = Image.open(os.path.join(get_path_root(), 'object_detection/pic_test/t2.jpg'))
    # original_img = Image.open(os.path.join(get_path_root(), 'object_detection/pic_test/t3.jpg'))
    # original_img = Image.open(os.path.join(get_path_root(), 'object_detection/pic_test/t4.jpg'))

    path = os.path.join(get_path_root(), '_test_pic')
    if os.path.exists(path):
        listdir = os.listdir(path)
    else:
        flog.error('path 不存在 %s', path)
    imgs = []
    for file in listdir:
        img = Image.open(os.path.join(path, file))
        imgs.append(img)

    # from pil image to tensor, do not normalize image
    data_transform = p_transform4ssd.Compose([
        p_transform4ssd.Resize(),
        p_transform4ssd.ToTensor(),
        p_transform4ssd.Normalization(),
    ])

    # 加载类别信息
    category_index = {}
    try:
        json_file = open(os.path.join(PATH_DATA_ROOT, 'classes_ids_voc.json'), 'r')
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)

    '''------------------模型定义---------------------'''
    model = create_model(num_classes=NUM_CLASSES)

    # 加载权重
    start_epoch = load_weight(PATH_FIT_WEIGHT, model)

    model.to(device)

    '''------------------预测开始---------------------'''
    model.eval()

    for img in imgs:
        with torch.no_grad():
            img_t, _ = data_transform(img)
            # expand batch dimension
            img_t = torch.unsqueeze(img_t, dim=0)

            predictions = model(img_t.to(device))[0]  # bboxes_out, labels_out, scores_out
            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()

            # flog.info('共选中 %s 个框', len(predict_boxes))
            if len(predict_boxes) == 0:
                flog.error("没有检测到任何目标!")

            # 这个要加宽一点
            draw_box(img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=5)
            plt.imshow(img)
            # plt.show()
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
