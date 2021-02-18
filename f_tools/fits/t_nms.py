import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from f_tools.fits.f_predictfun import batched_nms
from f_tools.fun_od.f_boxes import ltrb2ltwh, bbox_iou4np
from f_tools.pic.f_show import COLORS_ImageDraw

colors = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k',
    # 'w',
]


def show_boxes(boxes, c='k'):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(" nms")


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> suppressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(suppressed) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = nms_cpu.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    """Dispatch to only CPU Soft NMS implementations.

    The input can be either a torch tensor or numpy array.
    The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for Soft NMS.
        method (str): either 'linear' or 'gaussian'
        sigma (float): hyperparameter for gaussian method
        min_score (float): score filter threshold

    Returns:
        tuple: new det bboxes and indice, which is always the same
        data type as the input.

    Example:
        >>> dets = np.array([[4., 3., 5., 3., 0.9],
        >>>                  [4., 3., 5., 4., 0.9],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.4],
        >>>                  [3., 1., 3., 1., 0.0]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> new_dets, inds = soft_nms(dets, iou_thr, sigma=0.5)
        >>> assert len(inds) == len(new_dets) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_t = dets.detach().cpu()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_t = torch.from_numpy(dets)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    results = nms_cpu.soft_nms(dets_t, iou_thr, method_codes[method], sigma,
                               min_score)

    new_dets = results[:, :5]
    inds = results[:, 5]

    if is_tensor:
        return new_dets.to(
            device=dets.device, dtype=dets.dtype), inds.to(
            device=dets.device, dtype=torch.long)
    else:
        return new_dets.numpy().astype(dets.dtype), inds.numpy().astype(
            np.int64)


def py_cpu_nms(dets, thresh):
    '''

    :param dets: dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    :param thresh:
    :return:
    '''
    # 首先数据赋值和计算对应矩形框的面积
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    print('areas  ', areas)
    print('scores ', scores)

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []

    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    print(index)
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        print(index.size)
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)
        print(keep)
        print('x1', x1[i])
        print(x1[index[1:]])

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        print(x11, y11, x22, y22)
        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h
        print('overlaps is', overlaps)

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        print('ious is', ious)

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]
        print(idx)

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1
        print(index)
    return keep


def t_nms1():
    bboxs_list = []
    t1 = np.array([
        [0.1, 0.1, 0.3, 0.3],
        [0.25, 0.25, 0.5, 0.5],
        [0.2, 0.15, 0.6, 0.8]
    ])
    bboxs_list.append(t1)
    t2 = np.array([
        [0.4, 0.1, 0.5, 0.2],
        [0.45, 0.15, 0.6, 0.4],
    ])
    bboxs_list.append(t2)
    lables = [0, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.invert_yaxis()  # y轴反向
    cs = random.sample(colors, k=len(lables))
    for i, (bboxs, l) in enumerate(zip(bboxs_list, lables)):
        # ltrb->ltwh
        bboxs = ltrb2ltwh(bboxs[None])
        bboxs = np.squeeze(bboxs, axis=0)  # 删除一
        for bbox in bboxs:
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], color=cs[i], fill=False)
            ax.add_patch(rect)
    # plt.show()

    show_boxes(boxes)
    bl = np.empty((0, 4))
    for bboxs in bboxs_list:
        bl = np.append(bl, bboxs, axis=0)
        # bl = np.concatenate((bl,bboxs),axis=0)
    bboxs = torch.tensor(bl)
    bboxs = bboxs.reshape((-1, 4))
    lables = torch.tensor([0, 0, 0, 0, 0])
    scores = torch.tensor([0.5, 0.2, 0.3, 0.5, 0.2]).type(torch.float)
    bboxs = torch.tensor(bboxs).type(torch.float)
    keep = batched_nms(bboxs, scores, lables, 0.3)
    print(keep)


def t_nms2():
    global boxes
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]])
    plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax1)
    show_boxes(boxes, 'k')
    print('len(boxes)', len(boxes))
    datas = torch.tensor(boxes)
    # keep = torch.ops.torchvision.nms(datas[:, :4], datas[:, -1], 0.7)
    keep = py_cpu_nms(boxes, thresh=0.7)
    print('len(keep)', len(keep))
    plt.sca(ax2)
    show_boxes(boxes[keep], 'r')
    plt.show()


if __name__ == '__main__':
    # t_nms1()
    # t_nms2()
    iou_thr = 0.7
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                          [49.3, 32.9, 51.0, 35.3, 0.9],
                          [35.3, 11.5, 39.9, 14.5, 0.4],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    # CPU can handle float32 and float64
    dets = base_dets.astype(np.float32)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = torch.FloatTensor(base_dets)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = base_dets.astype(np.float64)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = torch.DoubleTensor(base_dets)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4
