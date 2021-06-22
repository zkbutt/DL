import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np

'''
PIL九种模型为1，L，P，RGB，RGBA，CMYK，YCbCr，I，F
    “1”为二值图像
    “L”为灰色图像
    “P”为8位彩色图像
    “RGBA”为32位彩色图像 红色、绿色和蓝色 透明
    “CMYK”为32位彩色图像 
        C：Cyan = 青色
        M：Magenta = 品红色
        Y：Yellow = 黄色
        K：Key Plate(blacK) = 定位套版色（黑色）
    “YCbCr”为24位彩色图像
        YCbCr其中Y是指亮度分量
        Cb指蓝色色度分量
        而Cr指红色色度分量
    “I”为32位整型灰色图像
    “F”为32位浮点灰色图像
    
'''


def f转换():
    img_np = cv2.imread(file_img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # --------tensor2PIL---------
    # img_tensor = torch.randint(0, 255, (3, 400, 300))  # c,h,w
    # img_tensor = F.to_tensor(img_tensor)
    file_img = r'D:\tb\tb\ai_code\DL\_test_pic\2008_000329.jpg'
    img_pil = Image.open(file_img).convert('RGB')
    img_tensor = F.to_tensor(img_pil)

    img_pil = F.to_pil_image(img_tensor).convert('RGB')  # PIL需加.convert('RGB')
    transform2pic = transforms.ToPILImage(mode="RGB")  # 不会进行归一化还原
    img_pil = transform2pic(img_tensor)

    # img_pil.show()  # 这个是调用系统的图片查看

    transform2tensor = transforms.ToTensor()  # 带归一化,恢复时要用尺寸
    img_tensor = F.to_pil_image(img_pil)
    img_tensor = transform2tensor(img_pil)
    img_pil = transform2pic(img_tensor)

    # --------opencv直接支持np--------
    img_cv = np.transpose(np.uint8(img_tensor.numpy()), (1, 2, 0))  # h,w,c
    cv2.imshow("img", img_cv)  # 显示
    key = cv2.waitKey(0)
    if key == 27:  # 按esc键时，关闭所有窗口
        print(key)
        cv2.destroyAllWindows()

    # ---------numpy2PIL---------
    img_pil = Image.fromarray(img_cv, mode="RGB")  # h,w,c
    img_pil.show()
    print(img_pil.size)  # w,h

    # ---------PIL2numpy---------
    img_np = np.array(img_pil, cv2.COLOR_RGB2BGR)

    # ---------ts2numpy---------
    img_pil = transforms.ToPILImage()(img_ts)
    img_np = Image.fromarray(img_pil.astype('uint8')).convert('RGB')


if __name__ == '__main__':
    # f转换()
    file_img = r'D:\tb\tb\ai_code\DL\_test_pic\2007_000042.jpg'  # 500 335
    ''' pil方式 '''
    # wh 500*335
    # img_pil = Image.open(file_img).convert('RGB')
    # print(img_pil.size)
    # img_pil.show()
    # wh -> c,h,w == c,row,col
    # img_tensor = F.to_tensor(img_pil)

    # (h,w,3)
    img_np = cv2.imread(file_img)  # 这个打开是hwc bgr
    print(img_np.shape)
    # (h,w,c)->(c,h,w)
    img_tensor = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
    print(img_tensor.shape)

    pass
