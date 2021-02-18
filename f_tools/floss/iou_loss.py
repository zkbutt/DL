import torch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def iou_loss(pred, target):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=torch.finfo(torch.float16).eps)
    loss = -ious.log()
    return loss


def bounded_iou_loss(pred, target, beta=0.2):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.

    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    eps = torch.finfo(torch.float16).eps
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0] + 1
    pred_h = pred[:, 3] - pred[:, 1] + 1
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps),
        torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps),
        torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w(target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h(target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)
    return loss


def giou_loss(pred, target):
    """
    Generalized Intersection over Union: A Metric and A Loss for
    Bounding Box Regression
    https://arxiv.org/abs/1902.09630

    code refer to:
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    eps = torch.finfo(torch.float16).eps
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt + 1).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    ag = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

    # GIoU
    gious = ious - (enclose_area - union) / enclose_area
    loss = 1 - gious
    return loss
