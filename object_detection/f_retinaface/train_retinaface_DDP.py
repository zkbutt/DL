import os

from object_detection.f_retinaface.CONFIG_F_RETINAFACE import CFG

if __name__ == '__main__':
    CFG.SAVE_FILE_NAME = os.path.basename(__file__)
    CFG.DATA_NUM_WORKERS = 5
