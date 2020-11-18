import torch


class FitBase(torch.nn.Module):
    def __init__(self, model, device):
        super(FitBase, self).__init__()
        self.model = model
        self.device = device
