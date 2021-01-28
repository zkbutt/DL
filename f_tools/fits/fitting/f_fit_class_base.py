import os
import random
import torch.distributed as dist
import torch

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_loader import DataLoader2
from f_tools.fits.fitting.f_fit_fun import init_od_e, base_set_1gpu, show_train_info, train_eval4od
from f_tools.fits.f_gpu.f_gpu_api import mgpu_process0_init, mgpu_init, fis_mgpu, is_main_process


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


class Train_Base:

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, is_mgpu, device, path_project_root, args=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

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
            show_train_info(cfg, self.loader_train, self.loader_val_coco)

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


class Predicted_Pic(Train_Base):

    def __init__(self, cfg, fun_train_eval_set, fun_init_model, is_mgpu, device, path_project_root, args=None) -> None:
        super().__init__(cfg, fun_train_eval_set, fun_init_model, is_mgpu, device, path_project_root, args)

    

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
