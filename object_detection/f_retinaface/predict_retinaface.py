import os

import torch
import numpy as np
from PIL import Image

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fun_od.f_anc import AnchorsFound
from f_tools.pic.f_show import show_od_keypoints4pil
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import *
from object_detection.f_retinaface.utils.process_fun import init_model, DATA_TRANSFORM
from object_detection.f_retinaface.utils.train_eval_fun import PredictHandler


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
    use_y = False  # 是否使用原装验证

    '''------------------模型定义---------------------'''
    model = init_model(CFG)
    model.eval()

    if use_y:
        file_weight = r'D:\tb\tb\ai_code\DL\object_detection\retinaface\file\Retinaface_mobilenet0.25.pth'
        state_dict = torch.load(file_weight, map_location=device)
        model_dict = model.state_dict()
        keys_missing, keys_unexpected = model.load_state_dict(state_dict)
    else:
        start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, device=device)

    path_img = './img'
    files = os.listdir(path_img)
    for file in files:
        '''---------------数据加载及处理--------------'''
        img_pil = Image.open(os.path.join(path_img, file)).convert('RGB')
        w, h = img_pil.size
        # 用于恢复bbox及ke
        szie_scale4bbox = torch.Tensor([w, h] * 2)
        szie_scale4landmarks = torch.Tensor([w, h] * 5)
        # szie_scale4bbox = torch.Tensor(CFG.IMAGE_SIZE * 2)
        # szie_scale4landmarks = torch.Tensor(CFG.IMAGE_SIZE * 5)

        if use_y:
            '''原装预处理方法'''
            img_ts = other(img_pil)
            anc_size = (h, w)
        else:
            '''feadre处理方法'''
            img_ts = DATA_TRANSFORM['val'](img_pil)[0][None]
            anc_size = CFG.IMAGE_SIZE

        # 生成 所有比例anchors
        anchors = AnchorsFound(anc_size, CFG.ANCHORS_SIZE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP).get_anchors()
        # anchors = AnchorsFound((h, w), CFG.ANCHORS_SIZE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP).get_anchors()
        # if CFG.IS_VISUAL:
        #     flog.debug('显示 anc %s', )
        #     show_anc4ts(img_ts.squeeze(0), anchors, CFG.IMAGE_SIZE)
        '''---------------预测开始--------------'''
        # (batch,++特图(w*h)*anc数,4) (batch,++特图(w*h)*anc数,2)  (batch,++特图(w*h)*anc数,10)
        predict_handler = PredictHandler(model, device, anchors,
                                         threshold_conf=0.5, threshold_nms=0.3)
        p_boxes, p_keypoints, p_scores = predict_handler.predicting4one(img_ts)
        if p_boxes is not None:
            # 恢复尺寸
            p_boxes = p_boxes * szie_scale4bbox
            # p_boxes = resize_boxes(p_boxes, (CFG.IMAGE_SIZE), ((w, h)))

            p_keypoints = p_keypoints * szie_scale4landmarks

            # 显示结果
            show_od_keypoints4pil(img_pil, p_boxes, p_keypoints, p_scores)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
