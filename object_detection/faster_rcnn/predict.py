import os

import torch
import torchvision
from torchvision import transforms

from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import get_path_root
from f_tools.f_torch_tools import load_weight
from f_tools.pic.f_show import draw_box
from object_detection.faster_rcnn.CONFIG_FASTER import NUM_CLASSES, PATH_FIT_WEIGHT, PATH_DATA_ROOT
from object_detection.faster_rcnn.backbone.mobilenetv2_model import MobileNetV2
from object_detection.faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN
from object_detection.faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from PIL import Image
import json
import matplotlib.pyplot as plt

from object_detection.faster_rcnn.network_files.rpn_function import AnchorsGenerator


def create_model(num_classes, model=1):
    if model == 1:
        # resNet50+fpn+faster_RCNN
        backbone = resnet50_fpn_backbone()
        model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    else:
        backbone = MobileNetV2().features
        backbone.out_channels = 1280  # 这个参数是必须的 模型创建会检查
        anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))

        # 创建ROI层
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                        output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                        sampling_ratio=2)  # 采样率

        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    return model


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    flog.info('device %s', device)

    '''---------------数据加载及处理--------------'''
    path = os.path.join(get_path_root(), '_test_pic')
    if os.path.exists(path):
        listdir = os.listdir(path)
    else:
        flog.error('path 不存在 %s', path)
    imgs = []
    for file in listdir:
        img = Image.open(os.path.join(path, file))
        imgs.append(img)

    category_index = {}
    try:
        json_file = open(os.path.join(PATH_DATA_ROOT, 'classes_ids_voc.json'), 'r')
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
    except Exception as e:
        flog.error(e)
        exit(-1)

    '''------------------模型定义---------------------'''
    model = create_model(num_classes=NUM_CLASSES, model=2)

    # load train weights
    start_epoch = load_weight(PATH_FIT_WEIGHT, model)

    model.to(device)

    '''------------------测试数据---------------------'''
    data_transform = transforms.Compose([transforms.ToTensor()])

    model.eval()
    for img in imgs:
        img_t = data_transform(img)
        # expand batch dimension
        img_t = torch.unsqueeze(img_t, dim=0)
        with torch.no_grad():
            predictions = model(img_t.to(device))[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            # predict_iscrowd = predictions["iscrowd"].to("cpu").numpy()

            # flog.info('共选中 %s 个框', len(predict_boxes))
            if len(predict_boxes) == 0:
                flog.error("没有检测到任何目标!")

            draw_box(img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=5)
            plt.imshow(img)
            plt.show()
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
