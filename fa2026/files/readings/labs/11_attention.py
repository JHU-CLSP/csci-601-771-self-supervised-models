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

# Section 4.1: separate routing at the same query, then concatenate features.
head_scores = torch.tensor([[math.log(8), 0., 0.], [0., math.log(8), 0.]])
head_weights = head_scores.softmax(-1)
head_values = torch.tensor([[2., 0.], [0., 2.], [4., 4.]])
head_answers = head_weights @ head_values
torch.testing.assert_close(head_weights, torch.tensor([[.8, .1, .1], [.1, .8, .1]]))
torch.testing.assert_close(head_answers, torch.tensor([[2., .6], [.6, 2.]]))
joined = head_answers.reshape(1, 4)
torch.testing.assert_close(joined, torch.tensor([[2., .6, .6, 2.]]))
print('two-head concatenation:', joined.tolist())

# Section 7.4: complete toy forward pass; no norms, biases, or dropout.
# Vocabulary IDs A=0, B=1, C=2; input positions and position IDs start at 0.
embedding = torch.tensor([[1., 0.], [0., .5], [.5, .5]], dtype=torch.float64)
positions = torch.tensor([[0., 0.], [0., .5], [.5, .5]], dtype=torch.float64)
inputs = torch.tensor([0, 1, 2])
targets = torch.tensor([1, 2, 0])
x = embedding[inputs] + positions
q = x @ torch.tensor([[1.], [0.]], dtype=x.dtype)
k = x @ torch.tensor([[math.log(2)], [0.]], dtype=x.dtype)
values = x @ torch.eye(2, dtype=x.dtype)
scores = q @ k.T  # sqrt(d_k) = 1
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
torch.testing.assert_close(weights, torch.tensor([[1., 0., 0.], [.5, .5, 0.], [.4, .2, .4]], dtype=x.dtype))
retrieved = weights @ values
u = x + retrieved
ffn = torch.relu(u) @ (.5 * torch.eye(2, dtype=x.dtype))
hidden = u + ffn
head = torch.tensor([[1., 0., -1.], [0., 1., -1.]], dtype=x.dtype)
logits = hidden @ head
torch.testing.assert_close(logits, torch.tensor([[3., 0., -3.], [.75, 2.25, -3.], [2.7, 2.4, -5.1]], dtype=x.dtype))
losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
probabilities = logits.softmax(-1)[torch.arange(3), targets]
changed_targets = targets.clone()
changed_targets[-1] = 1
changed_losses = torch.nn.functional.cross_entropy(logits, changed_targets, reduction='none')
torch.testing.assert_close(changed_losses.mean() - losses.mean(), torch.tensor(.1, dtype=x.dtype))
print('trace target probabilities:', [round(n, 5) for n in probabilities.tolist()])
print('trace row losses:', [round(n, 5) for n in losses.tolist()])
print('trace mean loss:', round(losses.mean().item(), 5))
