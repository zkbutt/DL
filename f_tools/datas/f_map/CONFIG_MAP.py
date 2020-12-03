class CFG:
    ANIMATION = False  # 是否动画
    PLOT_RES = False  # 是否日志 和画图
    CONSOLE_PINTER = True# 控制台输出
    IGNORE_CLASSES = []  # ['person','book']# 忽略某类
    IOU_MAP = []  # ['person', '0.7']
    CONFIDENCE = 0.5 # 置信度 default value (defined in the PASCAL VOC2012 challenge)

    PATH_GT = r'M:\AI\datas\widerface\f_map\gt_info'
    PATH_IMG = r'M:\AI\datas\widerface\coco\images\val2017'
    PATH_DT = r'M:\AI\datas\widerface\f_map\dt_info'
