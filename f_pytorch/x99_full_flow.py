import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms, datasets, utils

from f_tools.datas.data_factory import CreVOCDataset
from f_tools.datas.data_handler import spilt_voc2txt
from f_tools.fits.f_th_fit import f_fit_basics
from f_tools.pic.f_show import show_pics_ts, show_pic_ts


def f数据加载_自带torchvision(path):
    '''

    :param path:
    :return: 返回4个loader
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    # torchvision.datasets.FashionMNIST.resources = [
    #     (r"file:///D:\kaifa\AI\课件\深度学习Code\1.GAN\data\fashion-mnist/train-images-idx3-ubyte.gz", None),
    #     (r"file:///D:\kaifa\AI\课件\深度学习Code\1.GAN\data\fashion-mnist/train-labels-idx1-ubyte.gz", None),
    #     (r"file:///D:\kaifa\AI\课件\深度学习Code\1.GAN\data\fashion-mnist/t10k-images-idx3-ubyte.gz", None),
    #     (r"file:///D:\kaifa\AI\课件\深度学习Code\1.GAN\data\fashion-mnist/t10k-labels-idx1-ubyte.gz", None),
    # ]
    # 返回tensors
    '''
    MNIST 60000 样本的训练集和一个有着 10000 样本的测试集
    Fashion-MNIST 包含 60,000 个训练集图像和 10,000 个测试集图像
    CIFAR-10 数据集由10个类的60000个32x32彩色图像组成，每个类有6000个图像。有50000个训练图像和10000个测试图像
    CIFAR-100 有100个类，每个类包含600个图像。，每类各有500个训练图像和100个测试图像, 100个类被分成20个超类 有层级
    ImageNet 14197122张图像 约 150 GB
    PASCAL VOC VOC2007（430M），VOC2012（1.9G）
    Labelme 
    COCO 33 万张图像、80 个目标类别、每张图像 5 个标题、25 万张带有关键点的人像 约 25 GB（压缩后）
    SUN 
    '''
    fun_data = torchvision.datasets.FashionMNIST
    fun_data = torchvision.datasets.MNIST
    fun_data = torchvision.datasets.CIFAR10
    fun_data = torchvision.datasets.CIFAR100

    train_set = fun_data(path,
                         download=True,
                         train=True,
                         transform=transform)
    val_set = fun_data(path,
                       download=True,
                       train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                               shuffle=True, num_workers=0)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                             shuffle=False, num_workers=0, pin_memory=True)
    print(len(train_set), len(val_set))
    return train_loader, val_loader, train_set, val_set


def f自加载_手写符号图片(path_datas):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    train_set = datasets.ImageFolder(root=os.path.join(path_datas, 'images_background_small1'),
                                     transform=data_transform["train"])
    val_set = datasets.ImageFolder(root=os.path.join(path_datas, 'images_background_small1'),
                                   transform=data_transform["val"])

    flower_list = train_set.class_to_idx
    # 交换并写入---验证时读取 {'Balinese': 0, 'Early_Aramaic': 1, 'Greek': 2, 'Korean': 3, 'Latin': 4})
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 11
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=4, shuffle=True,
                                             num_workers=0)
    return train_loader, val_loader, train_set, val_set


class Net(nn.Module):
    '''f数据加载_自带torchvision'''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    '''f数据加载_手写符号图片'''

    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def f可视化tensorboardX():
    # https://github.com/lanpa/tensorboardX
    from tensorboardX import SummaryWriter
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    with SummaryWriter(comment='residual') as w:  # comment是指定文件名,推荐 默认自动生成的文件名，'runs/Aug20-17-20-33'
        w.add_graph(model, (dummy_input,))  # 画模型计算图 用于训练完成后或运行一次后 只添加一次
        # w.add_scalar('data/scalar1', val, n_iter) # 添加一个迭代的标量, 如loss acc等
        # w.add_scalar('data/scalar1', {'数据1': y1, '数据2': y2}, n_iter)  # 可以是多个
        w.close()  # 或跳出作用域


def f结果保存(model):
    torch.save(model, save_path)  # 保存在当前
    # 加载模型
    model = torch.load(save_path)
    # 只保存参数
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'logs_train': logs_train,
        # 'epoch': epoch,
    }
    torch.save(state, save_path)
    torch.save(model.state_dict(), save_path)  # .ckpt
    # 加载参数
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model'])


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    print('可用gpu数量', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    '''------------------全局变量---------------------'''
    save_path = os.path.join(os.path.abspath('.'), os.path.basename(__file__) + '.pth')  # 当前文件下,当前名称
    PATH_DATAS = r'D:\down\AI\datas'
    # PATH_DATAS = r'M:\datas\omniglot-master\python'
    # PATH_DATAS = r'M:\datas\omniglot-master\python'

    '''------------------数据加载及处理---------------------'''

    # ------------f数据加载_自带torchvision-------------
    train_loader, val_loader, train_set, val_set = f数据加载_自带torchvision(PATH_DATAS)

    # -------------f数据加载_手写符号图片 images_background_small1--------------
    # train_loader, val_loader, train_set, val_set = f数据加载_手写符号图片(PATH_DATAS)
    print(train_set.classes)

    print(train_set.class_to_idx)
    print(len(train_set))
    print(len(train_loader))  # 批次数

    # -------------VOC2007--------------

    # -------------数据取调试-----------
    # images, targets = iter(train_loader).__next__()  # N,H,W,C
    # images, targets = iter(train_loader).next()  # N,H,W,C
    x, y = iter(train_set).__next__()
    print(type(x), type(y))

    '''------------------数据分析---------------------'''
    # 查看一批图片
    # show_pics_ts(images, targets, one_channel=False)
    # show_pic_ts(images[0], targets[0], one_channel=False)

    '''------------------数据处理---------------------'''

    '''------------------模型定义---------------------'''
    # model = torchvision.models.resnet18(pretrained=True)
    # model = torchvision.models.vgg16(pretrained=True)
    model = torchvision.models.MobileNetV2()
    # model = torchvision.models.ShuffleNetV2()
    # model = torchvision.models.SqueezeNet()

    # 迁移修改最后一层
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)

    model = Net()
    # model = AlexNet()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 线性步长策略
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.33)

    # 定义部份训练
    for param in model.parameters():
        # 锁定参数
        param.requires_grad = False
    for name, parameter in model.named_parameters():  # 仅参数+名称的的迭代器
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]  # 优化部份参数

    for parameter in model.parameters():  # 仅参数的迭代器
        pass

    # 查看并测试模型
    # print(net_parameters)
    # summary(model, (1, 28, 28))  # 无需print
    # summary(net, (3, 224, 224))  # 无需print
    # summary(model.cuda(), (3, 400, 400))  # 显卡 model.to(device) 转换到显卡
    # images = torch.randn(64, 3, 224, 224)
    # summary(model, images)
    # outputs = model(images)

    # ------计算量分析----------
    from thop import profile
    from thop import clever_format
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(net, inputs=(input,))
    # print('计算量：%s' % flops, '参数量：%s' % params)
    # flops, params = clever_format([flops, params], "%.3f")
    # print('计算量：%s' % flops, '参数量：%s' % params)

    # f可视化tensorboardX()

    # for l in net.modules():
    #     print(l)

    '''------------------模型训练---------------------'''

    epochs = 4
    # running_loss, best_acc, logs_train = f_fit_basics(model, epochs, train_loader, train_loader,
    #                                                   loss_function, optimizer, lr_scheduler, device,
    #                                                   save_path
    #                                                   )

    # Tensorboard训练
    # f_训练Tensorboard()

    '''------------------结果分析---------------------'''

    '''------------------结果保存---------------------'''
    f结果保存(model)
