import sys

import torch
import cv2
import time

from PIL import Image

from f_tools.f_torch_tools import load_weight
from f_tools.fun_od.f_anc import AnchorsFound
from object_detection.f_retinaface.utils.process_fun import init_model, DATA_TRANSFORM
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import *

from f_tools.GLOBAL_LOG import flog
from object_detection.f_retinaface.utils.train_eval_fun import PredictHandler


def plot_img(img_np, p_boxes, p_keypoints, p_scores):
    for b, k, s in zip(p_boxes, p_keypoints, p_scores):
        b = list(b.type(torch.int64).numpy())
        k = list(k.type(torch.int64).numpy())
        text = "{:.4f}".format(s)
        cv2.rectangle(img_np, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # (l,t),(r,b),颜色.宽度
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_np, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.circle(img_np, (k[0], k[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_np, (k[2], k[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_np, (k[4], k[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_np, (k[6], k[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_np, (k[8], k[9]), 1, (255, 0, 0), 4)


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)
    # 调用摄像头
    cap = cv2.VideoCapture(0)  # capture=cv2.VideoCapture("1.mp4")
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    '''------------------模型定义---------------------'''
    model = init_model(CFG)
    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, device=device)
    model.eval()

    anchors = AnchorsFound(CFG.IMAGE_SIZE, CFG.ANCHORS_SIZE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP).get_anchors()

    '''---------------预测开始--------------'''
    fps = 0.0
    count = 0
    while True:
        start_time = time.time()
        '''---------------数据加载及处理--------------'''
        ref, img_np = cap.read()  # 读取某一帧 ref是否成功
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
        img_pil = Image.fromarray(img_np, mode="RGB")

        w, h = img_pil.size
        szie_scale4bbox = torch.Tensor([w, h] * 2)
        szie_scale4landmarks = torch.Tensor([w, h] * 5)

        img_ts = DATA_TRANSFORM['val'](img_pil)[0][None]

        '''---------------预测开始--------------'''
        predict_handler = PredictHandler(model, device, anchors,
                                         threshold_conf=0.5, threshold_nms=0.3)
        p_boxes, p_keypoints, p_scores = predict_handler.predicting4one(img_ts)
        if p_boxes is not None:
            # 恢复尺寸
            p_boxes = p_boxes * szie_scale4bbox
            # p_boxes = resize_boxes(p_boxes, (CFG.IMAGE_SIZE), ((w, h)))

            p_keypoints = p_keypoints * szie_scale4landmarks
            plot_img(img_np, p_boxes, p_keypoints, p_scores)
        # RGBtoBGR满足opencv显示格式
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # print("fps= %.2f" % (fps))
        count += 1
        img_np = cv2.putText(img_np, "fps= %.2f count=%s" % (fps, count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             (0, 255, 0), 2)
        fps = (fps + (1. / max(sys.float_info.min, time.time() - start_time))) / 2
        cv2.imshow("video", img_np)

        c = cv2.waitKey(1) & 0xff  # 输入esc退出
        if c == 27:
            cap.release()
            break
