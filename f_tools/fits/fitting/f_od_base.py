from torch import nn


class ModelBaseOD(nn.Module):

    def __init__(self):
        # 必须 losser preder
        super(ModelBaseOD, self).__init__()

    def model_execute(self, outs, targets, x):
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, x)
            return loss_total, log_dict
        else:
            # with torch.no_grad(): # 这个没用
            ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, x)
            return ids_batch, p_boxes_ltrb, p_labels, p_scores
