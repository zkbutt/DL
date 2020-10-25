import shutil
import sys
import os
import glob
from f_tools.datas.map.CONFIG_MAP import CFG


## This script ensures same number of files in ground-truth and detection-results folder.
## When you encounter file not found error, it's usually because you have
## mismatched numbers of ground-truth and detection-results files.
## You can use this script to move ground-truth and detection-results files that are
## not in the intersection into a backup folder (backup_no_matches_found).
## This will retain only files that have the same name in both folders.


# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)

def backup(src_folder, backup_files, backup_folder):
    # non-intersection files (txt format) will be moved to a backup folder
    if not backup_files:
        print('No backup required for', src_folder)
        return
    os.chdir(src_folder)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in backup_files:
        os.rename(file, backup_folder + '/' + file)


def recover_gt():
    path_gt_bak = os.path.join(GT_PATH, backup_folder)
    if os.path.exists(path_gt_bak):
        os.chdir(path_gt_bak)
        files_gt_bak = glob.glob('*.txt')  # 获取文件名
        for file in files_gt_bak:
            if os.path.exists(os.path.join(GT_PATH, file)):
                continue
            shutil.move(file, GT_PATH)


def fix_txt():
    os.chdir(GT_PATH)
    gt_files = glob.glob('*.txt')  # #获取指定目录下的所有txt
    if len(gt_files) == 0:
        print("Error: no .txt files found in", GT_PATH)
        sys.exit()
    os.chdir(DR_PATH)
    dr_files = glob.glob('*.txt')
    if len(dr_files) == 0:
        print("Error: no .txt files found in", DR_PATH)
        sys.exit()
    gt_files = set(gt_files)
    dr_files = set(dr_files)
    print('total ground-truth files:', len(gt_files))
    print('total detection-results files:', len(dr_files))
    print()
    gt_backup = gt_files - dr_files
    dr_backup = dr_files - gt_files
    backup(GT_PATH, gt_backup, backup_folder)
    backup(DR_PATH, dr_backup, backup_folder)
    if gt_backup:
        print('total ground-truth backup files:', len(gt_backup))
    if dr_backup:
        print('total detection-results backup files:', len(dr_backup))
    intersection = gt_files & dr_files
    print('total intersected files:', len(intersection))
    print("Intersection completed!")


if __name__ == '__main__':
    '''
    41轮 2684/3225   mAP = 29.18%
    11轮 2617/3225   mAP = 36.33%
    resnet 5轮 2904/3225   mAP = 32.47%
    resnext 4轮 2738/3225  mAP = 30.48%
    '''
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 初始化原装测试
    # parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
    # GT_PATH = os.path.join(parent_path, 'input', 'ground-truth')
    # DR_PATH = os.path.join(parent_path, 'input', 'detection-results')
    GT_PATH = CFG.PATH_GT
    DR_PATH = CFG.PATH_DT
    IMG_PATH = CFG.PATH_IMG
    recover = True

    backup_folder = 'backup_no_matches_found'  # must end without slash

    recover_gt()

    fix_txt()
