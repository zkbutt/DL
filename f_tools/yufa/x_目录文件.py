import glob
import os
import shutil

path = r'M:\AI\datas\coco2017\images\train2017'
listdir = os.listdir(path)

for file_str in listdir:
    file_str_ = file_str.replace('.jpg', '.txt')
    if os.path.exists(file_str_):
        pass
print(glob.glob('*.py'))  # 获取文件名

# os.makedirs(path) # 创建目录

fun_copy = shutil.move
fun_copy = shutil.copy
fun_copy(path_src, path_dst)
# shutil.rmtree(path) # 删除目录
