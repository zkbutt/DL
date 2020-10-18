# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
import cv2
import torch
import numpy as np
from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fun_od.f_anc import AnchorsFound
from f_tools.fun_od.f_boxes import nms
from object_detection.retinaface.CONFIG_RETINAFACE import MOBILENET025, PATH_FIT_WEIGHT, IMAGE_SIZE, VARIANCE
from object_detection.retinaface.nets.retinaface import RetinaFace
from object_detection.retinaface.utils.box_utils import decode, decode_landm, non_max_suppression

if __name__ == '__main__':

    # 尺寸修正 应该不是原图的
    # _img_info = self.coco.loadImgs(image_id)[0]
    # w, h = _img_info['width'], _img_info['height']
    '''------------------系统配置---------------------'''
    claxx = MOBILENET025  # 这里根据实际情况改

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    flog.info('device %s', device)

    '''---------------数据加载及处理--------------'''
    path_img = './img/street.jpg'
    path_img = './img/timg.jpg'

    image = cv2.imread(path_img)  # cv打开的是 默认是BRG np数组 h,w,c
    if image is None:
        flog.error('Open Error! Try again! %s', path_img)
        exit(-1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 从BRG转为RGB

    old_image = image.copy()
    image = np.array(image, np.float32)
    im_height, im_width, _ = np.shape(image)  # h,w,c

    # 根据数据集统一处理 这个方法感觉值有问题
    image -= np.array((104, 117, 123), np.float32)
    # 并转换通道
    image = image.transpose(2, 0, 1)

    # 增加batch_size维度[batch,3,] 最前面增加一维 可用 image[None]
    image = torch.from_numpy(image).unsqueeze(0)

    '''------------------模型定义---------------------'''
    # retinaface = Retinaface()
    model = RetinaFace(claxx.MODEL_NAME,
                       None,
                       claxx.IN_CHANNELS, claxx.OUT_CHANNEL,
                       claxx.RETURN_LAYERS,
                       anchor_num=claxx.ANCHOR_NUM,
                       num_classes=1
                       )
    # start_epoch = load_weight(PATH_FIT_WEIGHT, model)

    start_epoch = 0
    file_weight = r'D:\tb\tb\ai_code\DL\object_detection\retinaface\file\Retinaface_mobilenet0.25.pth'
    state_dict = torch.load(file_weight, map_location=device)
    model_dict = model.state_dict()
    keys_missing, keys_unexpected = model.load_state_dict(state_dict)
    model.to(device)

    '''------------------预测开始---------------------'''
    model.eval()

    # 生成 所有比例anchors
    anchors = AnchorsFound((im_height, im_width), claxx.ANCHORS_SIZE, claxx.FEATURE_MAP_STEPS,
                           claxx.ANCHORS_CLIP).get_anchors()
    with torch.no_grad():
        #       框 torch.Size([batch, 16800, 4])
        #       类别 torch.Size([batch, 16800, 2])
        #       关键点 torch.Size([batch, 16800, 10])
        loc, conf, landms = model(image)

        # 用于还原归一化的框(使用原图尺寸)
        scale = torch.Tensor([im_width, im_height] * 2)
        scale_for_landmarks = torch.Tensor([im_width, im_height] * 5)

        # 删除 1维 <class 'tuple'>: (37840, 4)
        _squeeze = loc.data.squeeze(0)
        boxes = decode(_squeeze, anchors, VARIANCE)
        boxes = boxes * scale  # 0-1比例 转换到原图
        # boxes = boxes.cpu().numpy()

        # <class 'tuple'>: (37840, 10)
        landms = decode_landm(landms.data.squeeze(0), anchors, VARIANCE)
        landms = landms * scale_for_landmarks
        # landms = landms.cpu().numpy()

        # 取其中index1  得一维数组 取出人脸概率  index0为背景 index1为人脸
        # conf = conf.data.squeeze(0)[:, 1:2]
        # conf = conf.data.squeeze(0)[:, None]  # torch.Size([1, 37840]) -> torch.Size([37840,1 ])
        conf = conf.reshape(-1, 1)  # torch.Size([1, 37840]) -> torch.Size([37840,1 ])

        # 最后一维连接 torch.Size([37840, 4])  torch.Size([1, 37840])  torch.Size([37840, 10])
        # boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)
        boxes_conf_landms = torch.cat([boxes, conf, landms], dim=-1)

        flog.debug('共有 %s', boxes_conf_landms.shape[0])
        mask = boxes_conf_landms[:, 4] >= 0.5  # 分类得分
        boxes_conf_landms = boxes_conf_landms[mask]
        flog.debug('类别分数过滤后 %s', boxes_conf_landms.shape[0])

        nms_sort = nms(boxes_conf_landms[:, :4], boxes_conf_landms[:, 4], 0.3)
        flog.debug('nms后 %s', len(nms_sort))

    for b in boxes_conf_landms:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(old_image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
    show_image = cv2.cvtColor(old_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("after", show_image)
    cv2.waitKey(0)
