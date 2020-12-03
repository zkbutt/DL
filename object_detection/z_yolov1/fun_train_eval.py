import torch

from f_tools.fun_od.f_boxes import fmatch4yolo1


class TEProcess:

    def __init__(self, model, lr_scheduler, device) -> None:
        super().__init__()
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.device = device

    def data_fun(self, batch_data):
        '''
        gpu数据组装
        :param batch_data:
            target['boxes']  torch.Size([nn, 4]) ltrb
            target['labels']  torch.Size([nn])
            target['size']  w h
        :return:
        '''
        cfg = self.model.cfg
        images, targets = batch_data
        images = images.to(self.device)
        dim_out = 4 * cfg.NUM_BBOX + 1 + cfg.NUM_CLASSES
        # 匹配后最终结果
        p_yolos = torch.empty((cfg.BATCH_SIZE, dim_out, cfg.NUM_GRID, cfg.NUM_GRID), device=self.device)

        for i, target in enumerate(targets):
            # 这里是每一个图片
            boxes_one = target['boxes'].to(self.device)
            labels_one = target['labels'].to(self.device)
            p_yolos[i] = fmatch4yolo1(boxes_one, labels_one, cfg.NUM_CLASSES, cfg.NUM_GRID, self.device)
            # target['size'] = target['size'].to(self.device)

        return images, p_yolos

    def ftrain(self, batch_data):
        images, p_yolos = self.data_fun(batch_data)

        log_dict = {}
        log_dict['loss_loc'] = loss_loc.item()
        log_dict['loss_conf'] = loss_conf.item()
        log_dict['loss_cls'] = loss_cls.item()

        return None
