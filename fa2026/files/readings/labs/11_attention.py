"""Handout 11: explicit attention, fused heads, and permutation equivariance."""
import math
import torch
from torch import nn

torch.manual_seed(11)
torch.set_num_threads(1)

def attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights

# Hand-computable retrieval: scores [log(2), 0, 0].
q = torch.tensor([[1., 0.]])
k = torch.tensor([[math.sqrt(2) * math.log(2), 0.], [0., 1.], [0., -1.]])
v = torch.tensor([[2., 0.], [0., 2.], [4., 4.]])
y, a = attention(q, k, v)
torch.testing.assert_close(a, torch.tensor([[.5, .25, .25]]))
torch.testing.assert_close(y, torch.tensor([[2., 1.5]]))
print('retrieval:', a.tolist(), y.tolist())

B, T, D, H = 2, 3, 8, 2
Dh = D // H
x = torch.randn(B, T, D)
fused = nn.Linear(D, 3 * D, bias=False)
output = nn.Linear(D, D, bias=False)

def mha(x):
    b, t, d = x.shape
    q, k, v = fused(x).chunk(3, dim=-1)
    q, k, v = [u.reshape(b, t, H, Dh).transpose(1, 2) for u in (q, k, v)]
    y, a = attention(q, k, v)
    y = y.transpose(1, 2).contiguous().reshape(b, t, d)
    return output(y), a

y, a = mha(x)
assert y.shape == (B, T, D) and a.shape == (B, H, T, T)
torch.testing.assert_close(a.sum(-1), torch.ones(B, H, T))
# Fusing the projections changes storage, not the three linear maps.
wq, wk, wv = fused.weight.chunk(3, dim=0)
q, k, v = fused(x).chunk(3, dim=-1)
for actual, weight in zip((q, k, v), (wq, wk, wv)):
    torch.testing.assert_close(actual, x @ weight.T)
# Without positions or a mask, permuting input rows permutes output rows.
perm = torch.tensor([2, 0, 1])
yp, _ = mha(x[:, perm])
torch.testing.assert_close(yp, y[:, perm])
# Change only token 2: token 0 may change under unrestricted attention.
x2 = x.clone()
x2[:, 2] += torch.arange(D, dtype=x.dtype)
y2, _ = mha(x2)
assert not torch.allclose(y2[:, 0], y[:, 0])
print('shape, fusion, row-sum, permutation, and future-dependence checks passed')
