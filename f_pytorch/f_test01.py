import torchvision

if __name__ == '__main__':
    # model = torchvision.models.vgg16(pretrained=True)
    model = torchvision.models.MobileNetV2()
    # model = torchvision.models.ShuffleNetV2()
    # model = torchvision.models.SqueezeNet()