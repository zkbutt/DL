import cv2
import matplotlib
import torch
import numpy as np

from f_tools.GLOBAL_LOG import flog
import socket

from f_tools.f_torch_tools import save_weight
from f_tools.fits.fitting.f_fit_eval_base import f_train_one_epoch4, f_evaluate4fmap, f_evaluate4coco3


def init_od_e(cfg):
    # -----------通用系统配置----------------
    torch.set_printoptions(linewidth=320, sci_mode=False, precision=5, profile='long')
    np.set_printoptions(linewidth=320, suppress=True,
                        formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
    matplotlib.rc('font', **{'size': 11})

    # 防止OpenCV进入多线程(使用PyTorch DataLoader)
    cv2.setNumThreads(0)

    if cfg.IS_MULTI_SCALE:
        cfg.tcfg_size = [640, 640]
        flog.warning("多尺度训练 ------ 开 最大 %s" % cfg.tcfg_size)
    else:
        cfg.tcfg_size = cfg.IMAGE_SIZE
        flog.warning("多尺度训练 ------ 关 %s", cfg.IMAGE_SIZE)


def custom_set(cfg):
    flog.info('当前主机: %s 及主数据路径: %s ' % (cfg.host_name, cfg.PATH_HOST))


def base_set_1gpu(cfg, id_gpu=0):
    # cfg.SAVE_FILE_NAME = os.path.basename(__file__)
    if torch.cuda.is_available():
        device = torch.device('cuda:%s' % id_gpu)
    else:
        device = torch.device("cpu")
        cfg.IS_MIXTURE_FIX = False
    flog.info('模型当前设备 %s', device)
    cfg.device = device  # cfg添加属性

    if cfg.DEBUG:
        flog.warning('debug模式')
        # device = torch.device("cpu")
        cfg.PRINT_FREQ = 1
        # cfg.PATH_SAVE_WEIGHT = None
        cfg.BATCH_SIZE = 5
        cfg.DATA_NUM_WORKERS = 0
        pass
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件

    return device, cfg


def fmax_map_save(maps_val, log_dict, cfg, model, optimizer, lr_scheduler, epoch):
    flog.info('map 最大值 %s', maps_val)
    if log_dict is not None:
        l = log_dict['l_total']
    else:
        l = None
    save_weight(
        path_save=cfg.PATH_SAVE_WEIGHT,
        model=model,
        name=cfg.SAVE_FILE_NAME + 'c%s' % cfg.THRESHOLD_PREDICT_CONF,
        loss=l,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=epoch,
        maps_val=maps_val)


def train_eval4od(start_epoch, model, optimizer,
                  fdatas_l2, lr_scheduler=None,
                  loader_train=None, loader_val_fmap=None, loader_val_coco=None,
                  device=torch.device('cpu'), train_sampler=None, eval_sampler=None,
                  tb_writer=None, maps_def=(0., 0),
                  ):
    cfg = model.cfg

    map_val_max_p = maps_def[0]
    map_val_max_r = maps_def[1]
    fun_datas_l2 = fdatas_l2
    for epoch in range(start_epoch, cfg.END_EPOCH):

        if cfg.IS_TRAIN:
            model.train()
            flog.info('训练开始：%s   半精度训练：%s    尺寸：%s' % (epoch + 1, cfg.IS_MIXTURE_FIX, cfg.tcfg_size))
            log_dict = f_train_one_epoch4(
                model=model,
                fun_datas_l2=fun_datas_l2,
                data_loader=loader_train,
                optimizer=optimizer, epoch=epoch,
                lr_scheduler=lr_scheduler,
                train_sampler=train_sampler,
                tb_writer=tb_writer,
                device=device,
            )

            # if lr_scheduler is not None:
            #     flog.warning('更新 lr_scheduler %s', log_dict['loss_total'])
            #     lr_scheduler.step(log_dict['loss_total'])  # 更新学习
        else:
            log_dict = None

        if cfg.IS_COCO_EVAL:
            s_keys = sorted(list(cfg.NUMS_EVAL.keys()), reverse=True)
            for s_key in s_keys:
                _e = (epoch + 1)
                if _e < s_key:
                    continue
                else:  # >=
                    eval_interval = cfg.NUMS_EVAL[s_key]
                    if _e % eval_interval != 0:
                        # 满足epoch 无需验证退出
                        break
                    else:
                        # 开始验证
                        flog.info('COCO 验证开始 %s', epoch + 1)
                        model.eval()
                        # with torch.no_grad():
                        ann_type = 'bbox'
                        res_eval = []

                        maps_val = f_evaluate4coco3(
                            model=model,
                            fun_datas_l2=fun_datas_l2,
                            data_loader=loader_val_coco,
                            epoch=epoch,
                            tb_writer=tb_writer,
                            res_eval=res_eval,
                            ann_type=ann_type,
                            device=device,
                            eval_sampler=eval_sampler,
                            is_keep=cfg.IS_KEEP_SCALE
                        )

                        if maps_val is not None:
                            if maps_val[0] > map_val_max_p:
                                map_val_max_p = maps_val[0]
                                map_val_max_r = max(map_val_max_r, maps_val[1])
                                fmax_map_save(maps_val, log_dict, cfg, model, optimizer, lr_scheduler, epoch)
                            elif maps_val[1] > map_val_max_r:
                                map_val_max_r = maps_val[1]
                                map_val_max_p = max(map_val_max_p, maps_val[0])
                                fmax_map_save(maps_val, log_dict, cfg, model, optimizer, lr_scheduler, epoch)
                        break  # 退出

        # if model.cfg.IS_COCO_EVAL \
        #         and (epoch + 1) > cfg.START_EVAL \
        #         and (epoch + 1) % cfg.EVAL_INTERVAL == 0 \
        #         or (epoch + 1) > cfg.END_EVAL:
        #     flog.info('COCO 验证开始 %s', epoch + 1)
        #     model.eval()
        #     # with torch.no_grad():
        #     ann_type = 'bbox'
        #     res_eval = []
        #
        #     maps_val = f_evaluate4coco3(
        #         model=model,
        #         fun_datas_l2=fun_datas_l2,
        #         data_loader=loader_val_coco,
        #         epoch=epoch,
        #         tb_writer=tb_writer,
        #         res_eval=res_eval,
        #         ann_type=ann_type,
        #         device=device,
        #         eval_sampler=eval_sampler,
        #         is_keep=cfg.IS_KEEP_SCALE
        #     )

        if model.cfg.IS_FMAP_EVAL:
            flog.info('FMAP 验证开始 %s', epoch + 1)
            model.eval()
            f_evaluate4fmap(
                model=model,
                data_loader=loader_val_fmap,
                is_keeep=cfg.IS_KEEP_SCALE,
                cfg=cfg,
            )

            return


def show_train_info(cfg, loader_train, loader_val_coco):
    if loader_train is not None:
        flog.debug('%s dataset_train 数量: %s' % (cfg.PATH_TENSORBOARD, len(loader_train.dataset)))
        flog.debug('loader_train 类型 %s' % loader_train.dataset.ids_classes)
    if loader_val_coco is not None:
        flog.debug('%s dataset_val 数量: %s' % (cfg.PATH_TENSORBOARD, len(loader_val_coco.dataset)))
        flog.debug('loader_val_coco 类型 %s' % loader_val_coco.dataset.ids_classes)
    flog.debug('cfg.BATCH_SIZE---%s' % cfg.BATCH_SIZE)
    flog.warning('cfg.LOSS_WEIGHT--- %s' % cfg.LOSS_WEIGHT)


if __name__ == '__main__':
    host_name = socket.gethostname()
    flog.info('当前主机 %s', host_name)
