import math
import torch
import numpy as np

from f_tools.datas.data_factory import VOCDataSet


def ltrb2xywh(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))

    if dim == 3:
        bboxs[:, :, 2] = bboxs[:, :, 2] - bboxs[:, :, 0]
        bboxs[:, :, 3] = bboxs[:, :, 3] - bboxs[:, :, 1]
        bboxs[:, :, 0] = bboxs[:, :, 0] + 0.5 * bboxs[:, :, 2]
        bboxs[:, :, 1] = bboxs[:, :, 1] + 0.5 * bboxs[:, :, 3]
    elif dim == 2:
        bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
        bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]
        bboxs[:, 0] = bboxs[:, 0] + 0.5 * bboxs[:, 2]
        bboxs[:, 1] = bboxs[:, 1] + 0.5 * bboxs[:, 3]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def ltrb2ltwh(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))

    if dim == 3:
        bboxs[:, :, 2] = bboxs[:, :, 2] - bboxs[:, :, 0]
        bboxs[:, :, 3] = bboxs[:, :, 3] - bboxs[:, :, 1]
    elif dim == 2:
        bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
        bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def ltwh2ltrb(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))

    if dim == 3:
        bboxs[:, :, 2] = bboxs[:, 2] + bboxs[:, :, 0]
        bboxs[:, :, 3] = bboxs[:, 3] + bboxs[:, :, 1]
    elif dim == 2:
        bboxs[:, 2] = bboxs[:, 2] + bboxs[:, 0]
        bboxs[:, 3] = bboxs[:, 3] + bboxs[:, 1]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def xywh2ltrb(bboxs, safe=True):
    dim = len(bboxs.shape)
    if safe:
        if isinstance(bboxs, np.ndarray):
            bboxs = np.copy(bboxs)
        elif isinstance(bboxs, torch.Tensor):
            bboxs = torch.clone(bboxs)
        else:
            raise Exception('类型错误', type(bboxs))
    if dim == 2:
        bboxs[:, :2] -= bboxs[:, 2:] / 2  # 中点移左上
        bboxs[:, 2:] += bboxs[:, :2]
    elif dim == 3:
        bboxs[:, :, :2] -= bboxs[:, :, 2:] / 2  # 中点移左上
        bboxs[:, :, 2:] += bboxs[:, :, :2]
    else:
        raise Exception('维度错误', bboxs.shape)
    return bboxs


def diff_keypoints(anc, g_keypoints, variances=(0.1, 0.2)):
    '''
    用anc同维 和 已匹配的GT 计算差异
    :param anc:
    :param g_keypoints:
    :param variances:
    :return:
    '''
    if len(anc.shape) == 2:
        ancs_xy = anc[:, :2].repeat(1, 5)
        ancs_wh = anc[:, 2:].repeat(1, 5)
        _t = (g_keypoints - ancs_xy) / variances[0] / ancs_wh
    elif len(anc.shape) == 3:
        ancs_xy = anc[:, :, :2].repeat(1, 1, 5)
        ancs_wh = anc[:, :, 2:].repeat(1, 1, 5)
        _t = (g_keypoints - ancs_xy) / variances[0] / ancs_wh
    else:
        raise Exception('维度错误', anc.shape)
    return _t


def diff_bbox(anc, g_bbox, variances=(0.1, 0.2)):
    '''
    用anc同维 和 已匹配的GT 计算差异
    :param anc: xywh  (nn,4) torch.Size([1, 16800, 4])
    :param p_loc: 修正系数 (nn,4)  torch.Size([5, 16800, 4])
    :return: 计算差异值   对应xywh
    '''
    if len(anc.shape) == 2:
        _a = (g_bbox[:, :2] - anc[:, :2]) / variances[0] / anc[:, 2:]
        _b = (g_bbox[:, 2:] / anc[:, 2:]).log() / variances[1]
        _t = torch.cat([_a, _b], dim=1)
    elif len(anc.shape) == 3:
        _a = (g_bbox[:, :, :2] - anc[:, :, :2]) / variances[0] / anc[:, :, 2:]
        _b = (g_bbox[:, :, 2:] / anc[:, :, 2:]).log() / variances[1]
        _t = torch.cat([_a, _b], dim=2)
    else:
        raise Exception('维度错误', anc.shape)
    return _t


def fix_bbox(anc, p_loc, variances=(0.1, 0.2)):
    '''
    用预测的loc 和 anc得到 修复后的框
    :param anc: xywh  (nn,4)
    :param p_loc: 修正系数 (nn,4)
    :return: 修复后的框
    '''
    if len(anc.shape) == 2:
        # 坐标移动
        _a = anc[:, :2] + p_loc[:, :2] * variances[0] * anc[:, 2:]
        # 宽高缩放
        _b = anc[:, 2:] * torch.exp(p_loc[:, 2:] * variances[1])
        _t = torch.cat([_a, _b], dim=1)
    elif len(anc.shape) == 3:
        # 坐标移动
        _a = anc[:, :, :2] + p_loc[:, :, :2] * variances[0] * anc[:, :, 2:]
        # 宽高缩放
        _b = anc[:, :, 2:] * torch.exp(p_loc[:, :, 2:] * variances[1])
        _t = torch.cat([_a, _b], dim=2)
    else:
        raise Exception('维度错误', anc.shape)
    return _t


def fix_keypoints(anc, p_keypoints, variances=(0.1, 0.2)):
    '''
    可以不用返回值
    :param anc:
    :param p_keypoints:
    :param variances:
    :return:
    '''
    if len(anc.shape) == 2:
        # [anc个, 4] ->  [anc个, 2] -> [anc个, 2*5]
        ancs_xy = anc[:, :2].repeat(1, 5)
        ancs_wh = anc[:, 2:].repeat(1, 5)
        _t = ancs_xy + p_keypoints * variances[0] * ancs_wh
    elif len(anc.shape) == 3:
        # [anc个, 4] ->  [anc个, 2] -> [anc个, 2*5]
        ancs_xy = anc[:, :, :2].repeat(1, 1, 5)
        ancs_wh = anc[:, :, 2:].repeat(1, 1, 5)
        _t = ancs_xy + p_keypoints * variances[0] * ancs_wh
    else:
        raise Exception('维度错误', anc.shape)
    return _t


def recover_bbox(bbox, size):
    pass


def recover_keypoints(keypoints, size):
    pass


def bbox_iou4np(bbox_a, bbox_b):
    '''
    求所有bboxs 与所有标定框 的交并比 ltrb
    返回一个数
    :param bbox_a: 多个预测框 (n,4)
    :param bbox_b: 多个标定框 (k,4)
    :return: <class 'tuple'>: (2002, 2) (n,k)
    '''
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    '''
    确定交叉部份的坐标  以下设 --- n = 3 , k = 4 ---
    广播 对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值 输出 (n,k,2)
    左上 右下
    '''
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # (3,1,2) (4,2)--->(3,4,2)
    # 选出前两个最小的 ymin，xmin 左上角的点 后面的通过广播
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    '''
    相交的面积 只有当右下的最小坐标  >>(xy全大于) 左上的最大坐标时才相交 用右下-左上得长宽
    '''
    # (2002,2,2)的每一第三维 降维运算(2002,2)  通过后面的是否相交的 降维运算 (2002,2)赛选
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # (2002,2) axis=1 2->1 行相乘 长*宽 --->降维运算(2002)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    # (2,2) axis=1 2->1 行相乘 长*宽 --->降维运算(2)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    #  (2002,1) +(2) = (2002,2) 每个的二维面积
    _a = area_a[:, None] + area_b
    _area_all = (_a - area_i)  # (2002,2)-(2002,2)
    return area_i / _area_all  # (2002,2)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou4ts(bboxs, ancs):
    '''

    :param bboxs: (Tensor[N, 4])  bboxs  ltrb
    :param ancs: (Tensor[M, 4])  ancs   xywh
    :return: (Tensor[N, M])
    '''
    area1 = box_area(bboxs)
    area2 = box_area(ancs)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(bboxs[:, None, :2], ancs[:, :2])  # left-top [N,M,2]
    rb = torch.min(bboxs[:, None, 2:], ancs[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def calc_ious4ts(box1, box2, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def nms(boxes, scores, iou_threshold):
    ''' IOU大于0.5的抑制掉
         boxes (Tensor[N, 4])) – bounding boxes坐标. 格式：(x1, y1, x2, y2)
         scores (Tensor[N]) – bounding boxes得分
         iou_threshold (float) – IoU过滤阈值
     返回:NMS过滤后的bouding boxes索引（降序排列）
     '''
    # type: (Tensor, Tensor, float) -> Tensor
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    '''

    :param boxes: 拉平所有类别的box重复的 n*20类,4
    :param scores: torch.Size([16766])
    :param idxs:  真实类别index 通过手动创建匹配的 用于表示当前 nms的类别 用于统一偏移 技巧
    :param iou_threshold:float 0.5
    :return:
    '''
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # torchvision.ops.boxes.batched_nms(boxes, scores, lvl, nms_thresh)
    # 根据最大的一个值确定每一类的偏移
    max_coordinate = boxes.max()  # 选出每个框的 坐标最大的一个值
    # idxs 的设备和 boxes 一致 , 真实类别index * (1+最大值) 则确保同类框向 左右平移 实现隔离
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes 加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


@torch.no_grad()
def pos_match(ancs, bboxs, criteria):
    '''
    正样本选取策略
        1. 每一个bboxs框 尽量有一个anc与之对应
        2. 每一个anc iou大于大于阀值的保留
    :param ancs:  anc模板 ltrb
    :param bboxs: 标签框 (xx,4)
    :param criteria: 小于等于0.35的反例
    :return:
        label_neg_mask: 返回反例的布尔索引
        anc_bbox_ind : 通过 g_bbox[anc_bbox_ind]  可选出标签 与anc对应 其它为0
    '''
    # (bboxs个,anc个)
    iou = calc_iou4ts(bboxs, xywh2ltrb(ancs))
    # (1,anc个值)降维  每一个anc对应的bboxs  最大的iou和index
    anc_bbox_iou, anc_bbox_ind = iou.max(dim=0)  # 存的是 bboxs的index

    # (bbox个,1)降维  每一个bbox对应的anc  最大的iou和index
    bbox_anc_iou, bbox_anc_ind = iou.max(dim=1)  # 存的是 anc的index

    # 强制保留 将每一个bbox对应的最大的anc索引 的anc_bbox_iou值强设为2 大于阀值即可
    anc_bbox_iou.index_fill_(0, bbox_anc_ind, 2)  # dim index val

    # 确保保留的 anc 的 gt  index 不会出错 exp:一个anc对应多个bboxs, bbox只有这一个anc时
    _ids = torch.arange(0, bbox_anc_ind.size(0), dtype=torch.int64).to(anc_bbox_ind)
    anc_bbox_ind[bbox_anc_ind[_ids]] = _ids
    # for j in range(bbox_anc_ind.size(0)): # 与上句等价
    #     anc_bbox_ind[bbox_anc_ind[j]] = j

    # ----------正例的index 和 正例对应的bbox索引----------
    # masks = anc_bbox_iou > criteria  # anc 的正例index
    # masks = anc_bbox_ind[masks]  # anc对应最大bboxs的索引 进行iou过滤筛选 形成正例对应的bbox的索引

    # 这里改成反倒置0
    label_neg_mask = anc_bbox_iou <= criteria  # anc 的正例index

    return label_neg_mask, anc_bbox_ind


def resize_boxes4np(boxes, original_size, new_size):
    '''
    用于预处理 和 最后的测试(预测还原)
    :param boxes: 输入多个 np shape(n,4)
    :param original_size: np (w,h)
    :param new_size: np (w,h)
    :return:
    '''
    # 输出数组   新尺寸 /旧尺寸 = 对应 h w 的缩放因子
    ratios = new_size / original_size
    boxes = boxes * np.concatenate([ratios, ratios])
    return boxes


def boxes2yolo(boxes, labels, num_bbox=2, num_class=20, grid=7):
    '''
    将ltrb 转换为 7*7*(2*(4+1)+20) = grid*grid*(num_bbox(4+1)+num_class)
    :param boxes:已原图归一化的bbox ltrb
    :param labels: 1~n值
    :param num_bbox:
    :param num_class:
    :param grid:
    :return:
        torch.Size([7, 7, 25])
    '''
    target = torch.zeros((grid, grid, 5 * num_bbox + num_class))
    # ltrb -> xywh
    wh = boxes[:, 2:] - boxes[:, :2]
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2  # bbox的中心点坐标
    cell_scale = 1. / grid

    for i in range(cxcy.size()[0]):  # 遍历每一个框 cxcy.size()[0]  与shape等价
        '''计算GT落在7x7哪个网格中 同时ij也表示网格的左上角的坐标'''
        cxcy_sample = cxcy[i]  # 每一个框的归一化xy
        # (cxcy_sample / cell_size).type(torch.int) 等价
        lr = (cxcy_sample / cell_scale).ceil() - 1  # cxcy_sample * 7  ceil上取整数 输出 0~6 index
        # xy与矩阵行列是相反的

        xy = lr * cell_scale  # 所在网格的左上角在原图的位置 0~1
        delta_xy = (cxcy_sample - xy) / cell_scale  # GT相对于所在网格左上角的值, 网格的左上角看做原点 需进行放大

        for j in range(num_bbox):
            target[int(lr[1]), int(lr[0]), (j + 1) * 5 - 1] = 1  # conf
            start = j * 5
            target[int(lr[1]), int(lr[0]), start:start + 2] = delta_xy  # 相对于所在网格左上角
            target[int(lr[1]), int(lr[0]), start + 2:start + 4] = wh[i]  # wh值相对于整幅图像的尺寸
        target[int(lr[1]), int(lr[0]), int(labels[i]) + 5 * num_bbox - 1] = 1  # 9指向0~9的最后一位,labels从1开始
    return target


if __name__ == '__main__':
    path = r'M:\AI\datas\VOC2012\trainval'

    dataset_train = VOCDataSet(
        path,
        'train.txt',  # 正式训练要改这里
        transforms=None,
        bbox2one=False,
        isdebug=True
    )
    img_pil, target_tensor = dataset_train[0]
    # bbox = target_tensor['boxes'].numpy()
    # labels = target_tensor["labels"].numpy()

    bbox = target_tensor['boxes']
    hw = target_tensor['height_width']
    wh = hw.numpy()[::-1]
    whwh = torch.tensor(wh.copy()).repeat(2)
    bbox /= whwh

    labels = target_tensor["labels"]
    target = boxes2yolo(bbox, labels)
    print(target.shape)
