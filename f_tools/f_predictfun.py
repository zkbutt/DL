import torch


def nms(boxes, scores, iou_threshold):
    ''' IOU大于0.5的抑制掉
         boxes (Tensor[N, 4])) – bounding boxes坐标. 格式：(ltrb
         scores (Tensor[N]) – bounding boxes得分
         iou_threshold (float) – IoU过滤阈值
     返回:NMS过滤后的bouding boxes索引（降序排列）
     '''
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    '''

    :param boxes: 拉平所有类别的box重复的 n*20类,4
    :param scores: torch.Size([16766])
    :param idxs:  真实类别index 通过手动创建匹配的 用于表示当前 nms的类别 用于统一偏移 技巧
    :param iou_threshold:float 0.5
    :return:
    '''
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


def label_nms(ids_batch1, p_boxes_ltrb1, p_labels1, p_scores1, device, threshold_nms):
    '''
    分类 nms
    :param ids_batch1: [nn]
    :param p_boxes_ltrb1: [nn,4] float32
    :param p_labels1: [nn]
    :param p_scores1: [nn]
    :param device:
    :param threshold_nms:
    :return:
    '''
    p_labels_unique = p_labels1.unique()  # nn -> n
    ids_batch2 = torch.tensor([], device=device)
    p_scores2 = torch.tensor([], device=device)
    p_labels2 = torch.tensor([], device=device)
    p_boxes_ltrb2 = torch.empty((0, 4), dtype=torch.float, device=device)

    for lu in p_labels_unique:  # 必须每类处理
        # 过滤类别
        _mask = p_labels1 == lu
        _ids_batch = ids_batch1[_mask]
        _p_scores = p_scores1[_mask]
        _p_labels = p_labels1[_mask]
        _p_boxes_ltrb = p_boxes_ltrb1[_mask]
        keep = batched_nms(_p_boxes_ltrb, _p_scores, _ids_batch, threshold_nms)
        # 极大抑制
        ids_batch2 = torch.cat([ids_batch2, _ids_batch[keep]])
        p_scores2 = torch.cat([p_scores2, _p_scores[keep]])
        p_labels2 = torch.cat([p_labels2, _p_labels[keep]])
        p_boxes_ltrb2 = torch.cat([p_boxes_ltrb2, _p_boxes_ltrb[keep]])
        # print('12')
    return ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2


def label_nms4keypoints(ids_batch1, p_boxes_ltrb1, p_keypoints1, p_labels1, p_scores1, device, threshold_nms):
    '''
    分类 nms
    :param ids_batch1: [nn]
    :param p_boxes_ltrb1: [nn,4] float32
    :param p_labels1: [nn]
    :param p_scores1: [nn]
    :param device:
    :param threshold_nms:
    :return:
    '''
    p_labels_unique = p_labels1.unique()  # nn -> n
    ids_batch2 = torch.tensor([], device=device)
    p_scores2 = torch.tensor([], device=device)
    p_labels2 = torch.tensor([], device=device)
    p_boxes_ltrb2 = torch.empty((0, 4), dtype=torch.float, device=device)
    p_keypoints2 = torch.empty((0, 10), dtype=torch.float, device=device)

    for lu in p_labels_unique:  # 必须每类处理
        # 过滤类别
        _mask = p_labels1 == lu
        _ids_batch = ids_batch1[_mask]
        _p_scores = p_scores1[_mask]
        _p_labels = p_labels1[_mask]
        _p_boxes_ltrb = p_boxes_ltrb1[_mask]
        keep = batched_nms(_p_boxes_ltrb, _p_scores, _ids_batch, threshold_nms)
        # 极大抑制
        ids_batch2 = torch.cat([ids_batch2, _ids_batch[keep]])
        p_scores2 = torch.cat([p_scores2, _p_scores[keep]])
        p_labels2 = torch.cat([p_labels2, _p_labels[keep]])
        p_boxes_ltrb2 = torch.cat([p_boxes_ltrb2, _p_boxes_ltrb[keep]])
        if p_keypoints1 is not None:
            _p_keypoints = p_keypoints1[_mask]
            p_keypoints2 = torch.cat([p_keypoints2, _p_keypoints[keep]])
        else:
            p_keypoints2 = None
        # print('12')
    return ids_batch2, p_boxes_ltrb2, p_keypoints2, p_labels2, p_scores2,
