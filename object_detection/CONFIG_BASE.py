import socket


class CfgBase:
    IS_MIXTURE_FIX = True  # 半精度训练

    # 调参区
    LOSS_WEIGHT = [1., 2., 3., 1.]  # l_box_p,l_conf_p,l_cls_p,l_conf_n
    LR = 1e-3
    # LR = 1e-1

    THRESHOLD_PREDICT_CONF = 0.7  # 用于预测的阀值

    host_name = socket.gethostname()
    if host_name == 'Feadre-NB':
        PATH_HOST = 'M:'
        # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
    elif host_name == 'e2680v2':
        PATH_HOST = ''
        PATH_PROJECT_ROOT = PATH_HOST + '/AI/temp/tmp_pycharm/DL/object_detection/z_center'  # 这个要改

    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_HOST + '/AI/weights/feadre'
    SAVE_FILE_NAME = 'train_center_'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_SAVE_WEIGHT + '/train_center_m2_widerface-59_4.557.pth'
    FILE_FIT_WEIGHT = PATH_SAVE_WEIGHT + '/train_center_r18_raccoon200-1_4662.583.pth'
    # FILE_FIT_WEIGHT = None

    '''Loss参数'''
    # IGNORE_THRESH = 0.225
    FEATURE_MAP_STEP = 4  # 特图的步距 下采倍数
    NUMS_ANC = [3, 3, 3]

    IS_MIXTURE_FIX = True  # 这个不要动 半精度训练

    # import getpass
    # # 获取当前系统用户名
    # user_name = getpass.getuser()
    # # 获取当前系统用户目录
    # user_home = os.path.expanduser('~')
