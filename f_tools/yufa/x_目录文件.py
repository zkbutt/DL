import os

path = r'M:\AI\datas\coco2017\images\train2017'
listdir = os.listdir(path)

for file_str in listdir:
    file_str_ = file_str.replace('.jpg', '.txt')
    if os.path.exists(file_str_):
        pass
