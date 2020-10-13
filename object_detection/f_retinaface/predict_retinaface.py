import os

import cv2
import torch
import numpy as np
from f_tools.GLOBAL_LOG import flog
from f_tools.fun_od.f_anc import AnchorsFound
from f_tools.fun_od.f_boxes import xywh2ltrb, nms, fix_bbox, fix_keypoints
from f_tools.pic.f_show import show_od_keypoints4np
from object_detection.f_retinaface.CONFIG_F_RETINAFACE import *
from object_detection.f_retinaface.utils.process_fun import init_model


def output_res(p_boxes, p_keypoints, p_scores, threshold_conf=0.5, threshold_nms=0.3):
    '''
    已修复的框 点 和对应的分数
        1. 经分数过滤
        2. 经NMS 出最终结果
    :param p_boxes:
    :param p_keypoints:
    :param p_scores:
    :return:
    '''
    mask = p_scores >= threshold_conf
    p_boxes = p_boxes[mask]
    p_scores = p_scores[mask]
    p_keypoints = p_keypoints[mask]

    if p_scores.shape[0] == 0:
        flog.error('threshold_conf 过滤后 没有目标 %s', threshold_conf)
        return None, None, None

    flog.debug('threshold_conf 过滤后有 %s 个', p_scores.shape[0])
    # 2 . 根据得分对框进行从大到小排序。
    keep = nms(p_boxes, p_scores, threshold_nms)
    flog.debug('threshold_nms 过滤后有 %s 个', len(keep))
    p_boxes = p_boxes[keep]
    p_scores = p_scores[keep]
    p_keypoints = p_keypoints[keep]
    return p_boxes, p_keypoints, p_scores


if __name__ == '__main__':
    '''
    分辩录及预处理影响检测
    '''

    '''------------------系统配置---------------------'''
    device = torch.device('cpu')
    flog.info('模型当前设备 %s', device)

    '''------------------模型定义---------------------'''
    model = init_model()
    model.eval()

    # 加载 模型权重
    state_dict = torch.load(PATH_FIT_WEIGHT, map_location=device)
    model_dict = model.state_dict()
    keys_missing, keys_unexpected = model.load_state_dict(state_dict)

    path_img = './img'
    files = os.listdir(path_img)
    for file in files:
        '''---------------数据加载及处理--------------'''
        img_np = cv2.imread(os.path.join(path_img, file))
        # 打开的是BRG转为RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # img_np = cv2.resize(img_np, (1280, 1280))

        img_np_old = img_np.copy()  # 用于后续 绘制人脸框 使用
        h, w, _ = img_np.shape
        # 用于恢复bbox及ke
        szie_scale4bbox = torch.Tensor([w, h] * 2)
        szie_scale4landmarks = torch.Tensor([w, h] * 5)

        # 预处理
        img_np = img_np.astype(np.float32)
        # 减掉RGB颜色均值  h,w,3 -> 3,h,w
        img_np -= np.array((104, 117, 123), np.float32)
        img_np = img_np.transpose(2, 0, 1)
        # 增加batch_size维度 np->ts
        img_ts = torch.from_numpy(img_np).unsqueeze(0)  # 最前面增加一维 可用 image[None]

        # 生成 所有比例anchors
        anchors = AnchorsFound([h, w], ANCHORS_SIZE, FEATURE_MAP_STEPS, ANCHORS_CLIP).get_anchors()

        '''---------------预测开始--------------'''
        # (batch,++特图(w*h)*anc数,4) (batch,++特图(w*h)*anc数,2)  (batch,++特图(w*h)*anc数,10)
        with torch.no_grad():
            p_loc, p_conf, p_landms = model(img_ts)

        p_scores = torch.nn.functional.softmax(p_conf, dim=-1)
        # (batch,++特图(w*h)*anc数,2) -> (batch,++特图(w*h)*anc数,1)
        p_scores = p_scores[:, :, 1]

        # (batch,++特图(w*h)*anc数,2) -> (++特图(w*h)*anc数)
        p_scores = p_scores.data.squeeze(0)  # 取出人脸概率  index0为背景 index1为人脸
        # 只有一张图 (batch,++特图(w*h)*anc数,4) -> (++特图(w*h)*anc数,4)
        p_loc = p_loc.data.squeeze(0)
        p_landms = p_landms.squeeze(0)

        # ---修复----variances = (0.1, 0.2)
        p_boxes = fix_bbox(anchors, p_loc)
        xywh2ltrb(p_boxes)
        p_keypoints = fix_keypoints(anchors, p_landms)

        # 出结果
        p_boxes, p_keypoints, p_scores = output_res(p_boxes, p_keypoints, p_scores)
        if p_boxes is not None:
            # 恢复尺寸
            p_boxes = p_boxes * szie_scale4bbox
            p_keypoints = p_keypoints * szie_scale4landmarks

            # 显示结果
            show_od_keypoints4np(img_np_old, p_boxes, p_keypoints, p_scores)

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
