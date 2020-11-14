import torch.nn as nn
from torchvision.models import _utils
from collections import OrderedDict


class Output4Densenet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        layer1_od = OrderedDict()
        layer2_od = OrderedDict()
        layer3_od = OrderedDict()
        # pool = AvgPool2d(kernel_size=2, stride=2, padding=0)

        for name1, module1 in backbone._modules.items():
            if name1 == 'features':
                _od = layer1_od
                for name2, module2 in module1._modules.items():
                    if name2 == 'transition2':
                        _od = layer2_od
                    elif name2 == 'transition3':
                        _od = layer3_od
                    elif name2 == 'norm5':
                        break
                    _od[name2] = module2
            break

        self.layer1 = nn.Sequential(layer1_od)
        self.layer2 = nn.Sequential(layer2_od)
        self.layer3 = nn.Sequential(layer3_od)

    def forward(self, inputs):
        out1 = self.layer1(inputs)  # torch.Size([1, 512, 52, 52])
        out2 = self.layer2(out1)  # torch.Size([1, 1024, 26, 26])
        out3 = self.layer3(out2)
        return out1, out2, out3


class Output4Return(nn.Module):

    def __init__(self, backbone, return_layers) -> None:
        super().__init__()
        self.layer_out = _utils.IntermediateLayerGetter(backbone, return_layers)

    def forward(self, inputs):
        out = self.layer_out(inputs)
        out = list(out.values())
        return out
