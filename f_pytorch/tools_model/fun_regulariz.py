import torch


class AffineChannel(torch.nn.Module):
    def __init__(self, num_features):
        super(AffineChannel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_features, ))
        self.bias = torch.nn.Parameter(torch.randn(num_features, ))

    def forward(self, x):
        N = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        transpose_x = x.permute(0, 2, 3, 1)
        flatten_x = transpose_x.reshape(N * H * W, C)
        out = flatten_x * self.weight + self.bias
        out = out.reshape(N, H, W, C)
        out = out.permute(0, 3, 1, 2)
        return out
