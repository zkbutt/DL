import json
import os

import torch
import numpy as np
from PIL import Image

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.pic.f_show import show_bbox4pil
from object_detection.f_yolov3.CONFIG_YOLO3 import CFG
from object_detection.f_yolov3.process_fun import init_model, DATA_TRANSFORM
from object_detection.f_yolov3.train_eval_fun import PredictHandler


def other(img_pil):
    # img_np = cv2.imread(os.path.join(path_img, file))
    # 打开的是BRG转为RGB
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = np.array(img_pil)
    img_np = img_np.astype(np.float32)
    # 减掉RGB颜色均值  h,w,3 -> 3,h,w
    img_np -= np.array((104, 117, 123), np.float32)
    img_np = img_np.transpose(2, 0, 1)
    # 增加batch_size维度 np->ts
    img_ts = torch.from_numpy(img_np).unsqueeze(0)  # 最前面增加一维 可用 image[None]
    return img_ts


if __name__ == '__main__':
    '''

    '''
    '''------------------系统配置---------------------'''
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    file_class = 'M:\AI\datas\VOC2012/classes_ids_voc.json'
    CFG.FILE_FIT_WEIGHT = 'M:\AI/weights/feadre/train_yolo3_DDP.pydensenet121-35_5.027892589569092.pth'

    idx_to_class = {}
    with open(file_class, 'r') as f:
        class_to_idx = json.load(f)  # 读进来是字符串
        for k, v in class_to_idx.items():
            idx_to_class[v] = k

    '''------------------模型定义---------------------'''
    model, losser, optimizer, lr_scheduler, start_epoch, anc_obj = init_model(CFG, device, id_gpu=None)
    model.eval()

    path_img = r'D:\tb\tb\ai_code\DL\_test_pic'
    files = os.listdir(path_img)
    for file in files:
        '''---------------数据加载及处理--------------'''
        img_pil = Image.open(os.path.join(path_img, file)).convert('RGB')
        w, h = img_pil.size  # 500,335
        # 用于恢复bbox及ke
        # szie_scale4bbox = torch.Tensor([w, h] * 2)
        szie_scale4bbox = torch.Tensor([w, h] * 2)[None]

        '''feadre处理方法'''
        img_ts = DATA_TRANSFORM['val'](img_pil)[0][None]

        '''---------------预测开始--------------'''
        predict_handler = PredictHandler(model, device, anc_obj=anc_obj,
                                         predict_conf_threshold=CFG.PREDICT_CONF_THRESHOLD,
                                         predict_nms_threshold=CFG.PREDICT_NMS_THRESHOLD,
                                         )
        res = predict_handler.predicting4one(img_ts, img_pil, idx_to_class)
        # for img_index, r in res.items():
        #     # img_np = np.array(img_pil)
        #     r_ = np.array(r)
        #     lables = [idx_to_class[i] for i in r_[:, 4]]
        #
        #     show_bbox4pil(img_pil, r_[:, :4], lables)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
