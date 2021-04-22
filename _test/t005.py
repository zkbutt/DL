import torch

torch.manual_seed(1)
a = torch.rand(3, 2)

c1 = torch.tensor([[1, 0], [-1, 0]])
# (3,2) -> (3,1)
b1 = a[..., 1:2]
# (3,2) -> (3,1)
col = torch.where(b1 > 0.5, c1[0], c1[1])
col.unsqueeze_(-1)

c2 = torch.tensor([[0, 1], [0, -1]])
b2 = a[..., :1]
row = torch.where(b2 > 0.5, c2[0], c2[1])
row.unsqueeze_(-1)

bb = torch.cat([row, col], -1)
t1 = torch.zeros_like(bb)[:, 0:1, :]
bb = torch.cat([t1, bb], 1)

print(a)
print(bb)
