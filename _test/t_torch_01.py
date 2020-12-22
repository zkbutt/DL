import torch
from torch.nn import functional as F

torch_max = torch.max(torch.tensor([1, 6, 3, ]), torch.tensor([5, 2, 4, ]), out=torch.tensor([5, 2, 4, ]))
print(torch_max)
