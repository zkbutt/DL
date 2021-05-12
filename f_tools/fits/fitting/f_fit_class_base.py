import os
import random
from abc import abstractmethod

import cv2
import torch.distributed as dist
import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_loader import DataLoader2
from f_tools.datas.f_coco.coco_dataset import CustomCocoDataset
from f_tools.device.f_device import init_video
from f_tools.fits.f_predictfun import label_nms
from f_tools.fits.fitting.f_fit_eval_base import f_prod_pic4one, f_prod_vodeo
from f_tools.fits.fitting.f_fit_fun import init_od_e, base_set_1gpu, show_train_info, train_eval4od
from f_tools.fits.f_gpu.f_gpu_api import mgpu_process0_init, mgpu_init, fis_mgpu, is_main_process
from f_tools.pic.enhance.f_data_pretreatment4np import cre_transform_resize4np
from torch import nn


def fdatas_l2(batch_data, device, cfg, epoch, model):
    '''
    cpu转gpu 输入模型前数据处理方法 定制
    image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor,
            mode='bilinear', align_corners=False)[0]
    images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=False)
    :param batch_data:
    :param device:
    :return:
    '''
    images, targets = batch_data
    if model.training and cfg.IS_MULTI_SCALE:
        if epoch != cfg.tcfg_epoch and epoch % 10 == 0:
            # if epoch % 1 == 0:
            if fis_mgpu():
                randints = torch.randint(low=10, high=19, size=[1], device=device)
                dist.broadcast(randints, src=0)
                randint = randints[0].item()
            else:
                randint = random.randint(*cfg.MULTI_SCALE_VAL)

            cfg.tcfg_size = [randint * 32] * 2  # (320~608) 9个尺寸
            flog.warning("多尺度训练 epoch:{}，尺度{}".format(epoch + 1, cfg.tcfg_size))
            cfg.tcfg_epoch = epoch  # 避免重复判断

        images = torch.nn.functional.interpolate(images, size=cfg.tcfg_size, mode='bilinear', align_corners=False)

    images = images.to(device)

    for target in targets:
        target['boxes'] = target['boxes'].to(device)
        target['labels'] = target['labels'].to(device)
        target['size'] = target['size'].to(device)
        if cfg.NUM_KEYPOINTS > 0:
            target['keypoints'] = target['keypoints'].to(device)

        # for key, val in target.items():
        #     target[key] = val.to(device)
    return images, targets


class Predicting_Base(nn.Module):
    '''
    由 f_fit_eval_base 运行至net模型主Center 再由模型主Center调用
    返回到 f_fit_eval_base
    '''

    def __init__(self, cfg) -> None:
        super(Predicting_Base, self).__init__()
        self.cfg = cfg

    @abstractmethod
    def p_init(self, outs):
        '''
        用于初始化参数 : 后续会用到的 self
        :param outs: 进行调整维度等
        :return:
            outs, device 必须返回
        '''
        # return outs, device

    @abstractmethod
    def get_pscores(self, outs):
        '''

        :param outs: 通过pconf pcls 计算分数
        :return:
            pscores : 这个用于判断 生成 mask_pos
            plabels : 这个用于二阶段
            pconf : 用于显示conf的统计值
        '''

    @abstractmethod
    def get_stage_res(self, outs, mask_pos, pscores, plabels):
        '''

        :param outs: ptxywh -> pltrb , 获取 ids_batch1
        :param mask_pos:
        :param pscores:
        :param plabels:
        :return:
            ids_batch1 : 返回第二阶段
            pboxes_ltrb1 :
            plabels1 :
            pscores1 :
        '''

    def forward(self, outs, imgs_ts=None):
        '''

        :param outs:
        :param imgs_ts: 图片预处理输出
        :return:
        '''
        cfg = self.cfg
        outs, device = self.p_init(outs)

        pscores, plabels, pconf = self.get_pscores(outs)

        mask_pos = pscores > cfg.THRESHOLD_PREDICT_CONF  # b,hw 二维mask
        if not torch.any(mask_pos):  # 如果没有一个对象
            print('该批次没有找到目标 max:{0:.2f} min:{0:.2f} mean:{0:.2f}'.format(pconf.max().item(),
                                                                          pconf.min().item(),
                                                                          pconf.mean().item(),
                                                                          ))
            return [None] * 5

        _i = cfg.PTOPK  # 这个是 topk 阀值
        if pscores.shape[-1] > _i:  # 并行取top100 与mask_pos 进行and操作
            # 最大1000个
            ids_topk = pscores.topk(_i, dim=-1)[1]  # torch.Size([32, 1000])
            mask_topk = torch.zeros_like(mask_pos)
            mask_topk[torch.arange(ids_topk.shape[0])[:, None], ids_topk] = True
            mask_pos = torch.logical_and(mask_pos, mask_topk)

        ids_batch1, pboxes_ltrb1, plabels1, pscores1 = self.get_stage_res(outs, mask_pos, pscores, plabels)

        ids_batch2, p_boxes_ltrb2, p_labels2, p_scores2 = label_nms(ids_batch1,
                                                                    pboxes_ltrb1,
                                                                    plabels1,
                                                                    pscores1,
                                                                    device,
                                                                    cfg.THRESHOLD_PREDICT_NMS)

        return ids_batch2, p_boxes_ltrb2, None, p_labels2, p_scores2,


class FBase:

    def __init__(self, cfg, device) -> None:
        self.cfg = cfg
        self.device = device

    def f_run(self):
        raise NotImplementedError


class Train_Base(FBase):

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, is_mgpu, device, path_project_root, args=None) -> None:
        super(Train_Base, self).__init__(cfg, device)
        '''------------------系统配置---------------------'''
        fun_train_eval_set(cfg)
        init_od_e(cfg)
        cfg.PATH_PROJECT_ROOT = os.path.join(cfg.PATH_HOST, path_project_root)

        '''---------------数据加载及处理--------------'''
        data_loader = DataLoader2(cfg)
        _ret = data_loader.get_train_eval_datas(is_mgpu=is_mgpu)
        self.loader_train, self.loader_val_fmap, self.loader_val_coco, self.train_sampler, self.eval_sampler = _ret
        show_train_info(cfg, self.loader_train, self.loader_val_coco)

        '''------------------模型定义---------------------'''
        self.model, self.optimizer, self.lr_scheduler, self.start_epoch = fun_init_model(cfg, device, id_gpu=args)
        self.tb_writer = None
        if cfg.TB_WRITER and is_main_process():
            self.tb_writer = mgpu_process0_init(args, cfg, self.loader_train, self.loader_val_coco, self.model, device)

    def f_run(self):
        train_eval4od(start_epoch=self.start_epoch, model=self.model, optimizer=self.optimizer,
                      fdatas_l2=fdatas_l2, lr_scheduler=self.lr_scheduler,
                      loader_train=self.loader_train, loader_val_fmap=self.loader_val_fmap,
                      loader_val_coco=self.loader_val_coco,
                      device=self.device, train_sampler=self.train_sampler, eval_sampler=self.eval_sampler,
                      tb_writer=self.tb_writer, maps_def=self.cfg.MAPS_VAL
                      )


class Train_1gpu(Train_Base):

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, path_project_root) -> None:
        self.is_mgpu = False
        device, cfg = base_set_1gpu(cfg, id_gpu=0)
        # fdebug
        # device = torch.device('cpu')
        super(Train_1gpu, self).__init__(cfg, fun_train_eval_set, fun_init_model, self.is_mgpu, device,
                                         path_project_root)


class Train_Mgpu(Train_Base):

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, path_project_root) -> None:
        if torch.cuda.is_available() is False:
            raise EnvironmentError("未发现GPU")
        self.is_mgpu = True
        args, device = mgpu_init()
        super(Train_Mgpu, self).__init__(cfg, fun_train_eval_set, fun_init_model, self.is_mgpu, device,
                                         path_project_root, args)


class Predicted_Base(FBase):

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, device) -> None:
        super(Predicted_Base, self).__init__(cfg, device)
        fun_train_eval_set(cfg)
        cfg.PATH_SAVE_WEIGHT = cfg.PATH_HOST + '/AI/weights/feadre'
        cfg.FILE_FIT_WEIGHT = os.path.join(cfg.PATH_SAVE_WEIGHT, cfg.FILE_NAME_WEIGHT)

        # 这里是原图
        self.dataset_test = CustomCocoDataset(
            file_json=cfg.FILE_JSON_TEST,
            path_img=cfg.PATH_IMG_EVAL,
            mode=cfg.MODE_COCO_EVAL,
            transform=None,
            is_mosaic=False,
            is_mosaic_keep_wh=False,
            is_mosaic_fill=False,
            is_debug=cfg.DEBUG,
            cfg=cfg
        )
        self.data_transform = cre_transform_resize4np(cfg)['val']

        # 初始化 labels
        ids_classes = self.dataset_test.ids_classes
        self.labels_lsit = list(ids_classes.values())  # index 从 1开始 前面随便加一个空
        self.labels_lsit.insert(0, None)  # index 从 1开始 前面随便加一个空
        flog.debug('测试类型 %s', self.labels_lsit)

        '''------------------模型定义---------------------'''
        self.model, _, _, _ = fun_init_model(cfg, device, id_gpu=None)  # model, optimizer, lr_scheduler, start_epoch
        self.model.eval()


class Predicted_Pic(Predicted_Base):

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, device,
                 eval_start=0,
                 is_test_dir=False,
                 path_img=None) -> None:
        super().__init__(cfg, fun_train_eval_set, fun_init_model, device)
        self.is_test_dir = is_test_dir
        self.path_img = path_img
        self.eval_start = eval_start

    def f_run(self):
        if self.is_test_dir:
            file_names = os.listdir(self.path_img)
            for name in file_names:
                file_img = os.path.join(self.path_img, name)
                img_np = cv2.imread(file_img)
                f_prod_pic4one(img_np=img_np, data_transform=self.data_transform, model=self.model,
                               size_ts=torch.tensor(img_np.shape[:2][::-1]),
                               labels_lsit=self.labels_lsit)
        else:
            for i in range(self.eval_start, len(self.dataset_test), 1):
                img_np = self.dataset_test[i][0]
                target = self.dataset_test[i][1]
                f_prod_pic4one(img_np=img_np, data_transform=self.data_transform, model=self.model,
                               size_ts=target['size'], labels_lsit=self.labels_lsit,
                               gboxes_ltrb=target['boxes'], target=target)


class Predicted_Video(Predicted_Base):

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, device) -> None:
        super(Predicted_Video, self).__init__(cfg, fun_train_eval_set, fun_init_model, device)
        # 调用摄像头
        self.cap = init_video()

    def f_run(self):
        f_prod_vodeo(self.cap, self.data_transform, self.model, self.labels_lsit, self.device, is_keeep=False)


if __name__ == '__main__':
    class ttcfg:
        pass


    def fun_train_eval_set():
        pass


    def fun_init_model():
        pass


    path_project_root = 'AI/temp/tmp_pycharm/DL/object_detection/z_yolov1'

    Train_1gpu(ttcfg, fun_train_eval_set, fun_init_model, path_project_root)
    print('debug')
