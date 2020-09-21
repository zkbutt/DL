# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
from object_detection.retinaface.retinaface import Retinaface
import cv2

if __name__ == '__main__':
    retinaface = Retinaface()

    while True:
        path_img = './img/street.jpg'
        path_img = './img/timg.jpg'

        image = cv2.imread(path_img) # cv打开的是 默认是BRG np数组 h,w,c
        if image is None:
            print('Open Error! Try again!')
            continue
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 从BRG转为RGB
            r_image = retinaface.detect_image(image)
            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("after", r_image)
            cv2.waitKey(0)
