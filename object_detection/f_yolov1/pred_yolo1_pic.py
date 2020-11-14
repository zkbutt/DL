import json
import os

import torch
import numpy as np
from PIL import Image

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.pic.f_show import show_bbox4pil
from object_detection.f_yolov1.CONFIG_YOLO1 import CFG
from object_detection.f_yolov1.utils.process_fun import DATA_TRANSFORM, init_model
from object_detection.f_yolov1.utils.train_eval_fun import PredictHandler


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
    idx_to_class = {}
    with open(file_class, 'r') as f:
        class_to_idx = json.load(f)  # 读进来是字符串
        for k, v in class_to_idx.items():
            idx_to_class[v] = k

    '''------------------模型定义---------------------'''
    model = init_model(CFG)
    model.eval()

    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, device=device)

    path_img = r'D:\tb\tb\ai_code\DL\_test_pic'
    files = os.listdir(path_img)
    for file in files:
        '''---------------数据加载及处理--------------'''
        img_pil = Image.open(os.path.join(path_img, file)).convert('RGB')
        w, h = img_pil.size #500,335
        # 用于恢复bbox及ke
        # szie_scale4bbox = torch.Tensor([w, h] * 2)
        szie_scale4bbox = torch.Tensor([w, h] * 2)[None]

        '''feadre处理方法'''
        img_ts = DATA_TRANSFORM['val'](img_pil)[0][None]

        '''---------------预测开始--------------'''
        # (batch,++特图(w*h)*anc数,4) (batch,++特图(w*h)*anc数,2)  (batch,++特图(w*h)*anc数,10)
        predict_handler = PredictHandler(model, device,
                                         grid=CFG.GRID, num_bbox=CFG.NUM_BBOX, num_cls=CFG.NUM_CLASSES,
                                         threshold_conf=0.5, threshold_nms=0.3)
        res = predict_handler.predicting4one(img_ts, szie_scale4bbox)
        for img_index, r in res.items():
            # img_np = np.array(img_pil)
            r_ = np.array(r)
            lables = [idx_to_class[i] for i in r_[:,4]]

            show_bbox4pil(img_pil, r_[:, :4], lables)
        flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
