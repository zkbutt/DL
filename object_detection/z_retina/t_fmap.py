import os

from f_tools.datas.f_map.convert_data.extra.intersect_gt_and_dr import f_fix_txt, f_recover_gt
from f_tools.datas.f_map.map_go import f_do_fmap
from object_detection.z_retina.CONFIG_RETINAFACE import CFG

if __name__ == '__main__':
    cfg = CFG
    path_gt = os.path.join(cfg.PATH_EVAL_INFO, 'gt_info')
    path_dt = os.path.join(cfg.PATH_EVAL_INFO, 'dt_info')

    f_recover_gt(path_gt)

    # f_fix_txt(path_gt, path_dt)
    # f_do_fmap(path_gt, path_dt, cfg.PATH_EVAL_IMGS,
    #           confidence=cfg.THRESHOLD_PREDICT_CONF,
    #           console_pinter=True,
    #           plot_res=True,
    #           animation=False)


