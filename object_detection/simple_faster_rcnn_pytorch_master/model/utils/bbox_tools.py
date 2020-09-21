import numpy as np


def xy2xyhw(bboxs):
    '''
    左上右下 -> 中心
    :param bboxs:  yl,xl,yr,xr
    :return:
    '''
    src_height = bboxs[:, 2] - bboxs[:, 0]
    src_width = bboxs[:, 3] - bboxs[:, 1]
    src_ctr_y = bboxs[:, 0] + 0.5 * src_height
    src_ctr_x = bboxs[:, 1] + 0.5 * src_width
    return src_height, src_width, src_ctr_y, src_ctr_x


def xyhw2xy(h, w, ctr_y, ctr_x):
    '''
    中心 -> 左上右下
    :param h, w, ctr_y, ctr_x:
    :return: 多个左上右下
    '''
    dst_bboxs = np.zeros((ctr_y.shape[0], 4), dtype=ctr_y.dtype)
    dst_bboxs[:, 0::4] = ctr_y - 0.5 * h
    dst_bboxs[:, 1::4] = ctr_x - 0.5 * w
    dst_bboxs[:, 2::4] = ctr_y + 0.5 * h
    dst_bboxs[:, 3::4] = ctr_x + 0.5 * w
    return dst_bboxs


def loc2bbox(src_bbox, loc):
    '''
     已知源bbox 和位置偏差dx，dy，dh，dw，求目标框G
    :param src_bbox: (n,4) 表示多个选区 , 左上右下角坐标
    :param loc: 回归修正参数 (n,4)  中心点 (dx,dy,dh,dw)
    :return: 修正后选区
    '''

    if src_bbox.shape[0] == 0:  # 选区不存在
        return np.zeros((0, 4), dtype=loc.dtype)
    # 复制一个
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    # 左上右下 ---> 中心点 (dx,dy,dh,dw)
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    _ = loc[:, 0]  # 这种取出来会降低维度
    _ = loc[:, :1]  # 与这个等效
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # 根据公式 进行选框回归 Gy Gx Gh Gw
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    # 开内存  中心点 (dx,dy,dh,dw) ---> 左上右下
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w
    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    '''
    已知源框和目标框求出其位置偏差
    :param src_bbox:  正选出的正样本
    :param dst_bbox: 左上右下
    :return: <class 'tuple'>: (128, 4)
    '''
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    # 左上右下转中心 hw
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 除法不能为0 log不能为负 取此类型的最小的一个数
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    # 按公式算偏差 , loc2bbox 这个是算偏差
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    '''
    求所有bboxs 与所有标定框 的交并比 第二维（ymin，xmin，ymax，xmax）
    返回一个数
    :param bbox_a: 多个预测框 (n,4)
    :param bbox_b: 多个标定框 (k,4)
    :return: <class 'tuple'>: (2002, 2)
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


def generate_anchor_base(base_size=16, ratios=(0.5, 1, 2),
                         anchor_scales=(8, 16, 32)):
    '''
    生成一套 anchors AnchorsGenerator
    :param base_size:
    :param ratios:
    :param anchor_scales:
    :return: (len(ratios) * len(anchor_scales), 4) 左上右下
    '''
    py = base_size / 2.
    px = base_size / 2.

    # 以特征图的左上角点为基准产生的9个anchor
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


if __name__ == '__main__':
    print(generate_anchor_base())
