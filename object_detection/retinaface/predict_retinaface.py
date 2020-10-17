import os

import cv2
import torch
from PIL import Image
import numpy as np
from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_pretreatment import Compose, ResizeKeep, ToTensor, Normalization4TS
from f_tools.fits.f_lossfun import PredictOutput
from f_tools.fun_od.f_boxes import xywh2ltrb
from object_detection.retinaface.CONFIG_RETINAFACE import MOBILENET025, IMAGE_SIZE
from object_detection.retinaface.train_retinaface import model_init
import torch.nn.functional as F


def scale_back_batch(p_bboxs, p_labels, ancs, variance):
    '''
        修正def 并得出分数 softmax
        1）通过预测的 loc_p 回归参数与anc得到最终预测坐标 box
        2）将box格式从 xywh 转换回ltrb
        3）将预测目标 score通过softmax处理
    :param p_bboxs: 预测出的框 偏移量 xywh [N, 4, 8732]
    :param p_labels: 预测所属类别 [N, label_num, 8732]
    :return:  返回 anc+预测偏移 = 修复后anc 的 ltrb 形式
    '''
    # ------------限制预测值------------
    p_bboxs[:, :, :2] = variance[0] * p_bboxs[:, :, :2]  # 预测的x, y回归参数
    p_bboxs[:, :, 2:] = variance[1] * p_bboxs[:, :, 2:]  # 预测的w, h回归参数
    # 将预测的回归参数叠加到default box上得到最终的预测边界框
    p_bboxs[:, :, :2] = p_bboxs[:, :, :2] * ancs[:, :, 2:] + ancs[:, :, :2]
    p_bboxs[:, :, 2:] = p_bboxs[:, :, 2:].exp() * ancs[:, :, 2:]

    # xywh -> ltrb 用于极大nms
    xywh2ltrb(p_bboxs)

    p_scores = F.softmax(p_labels, dim=-1)
    return p_bboxs, p_scores


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    claxx = MOBILENET025  # 这里根据实际情况改
    variance = (0.1, 0.2)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    flog.info('device %s', device)

    '''---------------数据加载及处理--------------'''
    # file_img = './img/street.jpg'
    file_img = './img/timg.jpg'

    img_np = cv2.imread(file_img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_np, mode="RGB")
    # img_pil.show()
    # img_pil = Image.open(file_img).convert('RGB')  # 原图数据
    w, h = img_pil.size

    data_transform = Compose([
        ResizeKeep(IMAGE_SIZE),  # (h,w)
        ToTensor(),
        Normalization4TS(),
    ])
    img_ts, _ = data_transform(img_pil)

    '''------------------模型定义---------------------'''
    model, anchors, losser, optimizer, lr_scheduler, start_epoch = model_init(claxx, device)

    '''------------------预测开始---------------------'''
    model.eval()

    predict_output = PredictOutput(anchors, IMAGE_SIZE)

    with torch.no_grad():
        out = model(img_ts[None])
        p_bboxs = out[0]  # (batch,16800,4)
        p_labels = out[1]  # (batch,16800,10) 这里输出的是分
        p_keypoints = out[2]  # (batch,16800,2)

        # p_bboxs=torch.Size([batch, 16800, 4])  p_scores=torch.Size([batch, 16800, 2])
        # xywh -> ltrb
        p_bboxs, p_scores = scale_back_batch(p_bboxs, p_labels, anchors[None], variance)
        p_bboxs = p_bboxs.reshape(-1, 4)  # [batch, 16800, 4] -> [batch*16800, 4]
        p_scores = p_scores[:, :, 1].reshape(-1)  # [batch, 16800, 2] -> [batch, 16800, 1] ->[batch*16800]

        mask = p_scores >= 0.3  # 分类得分  [batch*16800]
        p_bboxs = p_bboxs[mask]  # [mask个, 4]
        p_scores = p_scores[mask]  # [mask个, 4]

        p_bboxs = p_bboxs.clamp(min=0, max=1)  # 对越界的bbox进行裁剪
        scale = torch.Tensor([w, h] * 2)
        # p_bboxs = p_bboxs * np.array(IMAGE_SIZE * 2)  # 0-1比例 转换到原图

        flog.debug('共有 %s', p_bboxs.shape[0])
        '''
        boxes (Tensor[N, 4])) – bounding boxes坐标. 格式：(x1, y1, x2, y2)
        scores (Tensor[N]) – bounding boxes得分
        iou_threshold (float) – IoU过滤阈值
        '''
        nms_sort = torch.ops.torchvision.nms(p_bboxs, p_scores, 0.5)
        flog.debug('nms后 %s', len(nms_sort))
        # 恢复到特图
        p_bboxs = p_bboxs * np.array(IMAGE_SIZE * 2, dtype=float)[None]
        # img_np = cv2.imread(file_img)
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        for b, p in zip(p_bboxs[nms_sort], p_scores[nms_sort]):
            text = "{:.4f}".format(p)
            b = list(map(int, b))
            # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
            cv2.rectangle(img_np, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_np, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        show_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("after", show_image)
        cv2.waitKey(0)
