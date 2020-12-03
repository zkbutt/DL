from torchvision import models


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        print(classname)
        m.inplace = True


if __name__ == '__main__':
    model = models.resnext50_32x4d(pretrained=True)  # 能力 22.38 6.30 ---top3
    model.apply(inplace_relu)
