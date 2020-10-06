import torchvision
import torch
import os

from f_tools.GLOBAL_LOG import flog

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = " 1,2"  # 全局加载

if __name__ == '__main__':
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = torchvision.models.vgg16(pretrained=True)
    # model = torchvision.models.MobileNetV2()
    # model = torchvision.models.ShuffleNetV2()
    # model = torchvision.models.SqueezeNet()
    model = torchvision.models.resnet50(num_classes=10)

    # model.to(device)
    rnn = model.cuda()
    par = torch.nn.DataParallel(rnn)

    inp = torch.randn(100, 3, 32, 32).cuda()

    par(inp).sum().backward()
    print('完成')
    flog.debug('12 %s', 123)