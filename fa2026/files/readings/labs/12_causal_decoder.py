"""Handout 12: a two-layer decoder, causal/padding tests, and per-layer KV cache.
No dropout; learned absolute positions; right padding for the training example.
This is original teaching code, not a high-performance implementation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(12)
torch.set_num_threads(1)

class Block(nn.Module):
    def __init__(self, d=8, heads=2):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x, real, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [u.reshape(b, t, self.heads, self.dh).transpose(1, 2)
                   for u in (q, k, v)]
        past = 0 if cache is None else cache[0].size(-2)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        key_pos = torch.arange(past + t, device=x.device)
        query_pos = past + torch.arange(t, device=x.device)
        allowed = (key_pos[None, :] <= query_pos[:, None])[None, None]
        allowed = allowed & real[:, None, None, :].bool()
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        # Reject all-masked rows explicitly instead of producing NaNs.
        assert allowed.any(-1).all(), 'each query needs an allowed key'
        weights = scores.masked_fill(~allowed, float('-inf')).softmax(-1)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().reshape(b, t, d)
        x = x + self.out(y)
        x = x + self.ff(self.n2(x))
        return x, (k, v)

class TinyDecoder(nn.Module):
    def __init__(self, vocab=12, d=8, max_len=32):
        super().__init__()
        self.token = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d), Block(d)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, ids, real=None, caches=None):
        b, t = ids.shape
        past = 0 if caches is None else caches[0][0].size(-2)
        if real is None:
            real = torch.ones(b, past+t, dtype=torch.bool, device=ids.device)
        positions = past + torch.arange(t, device=ids.device)
        x = self.token(ids) + self.pos(positions)
        new_caches = []
        for i, block in enumerate(self.blocks):
            old = None if caches is None else caches[i]
            x, new = block(x, real, old)
            new_caches.append(new)
        return self.head(self.norm(x)), new_caches

model = TinyDecoder().eval()
ids = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    full, _ = model(ids)
    changed = ids.clone()
    changed[:, 2:] = torch.tensor([[7, 8]])
    altered, _ = model(changed)
    torch.testing.assert_close(full[:, :2], altered[:, :2])
    caches, pieces = None, []
    for t in range(ids.size(1)):
        logits, caches = model(ids[:, t:t+1], caches=caches)
        pieces.append(logits)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-6, rtol=1e-5)
    # Right-padded rows; token 0 is PAD for this example.
    padded = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    real = padded != 0
    batched, _ = model(padded, real=real)
    alone, _ = model(padded[1:2, :2])
    torch.testing.assert_close(batched[1:2, :2], alone)

# Shift first, then mask labels according to whether the TARGET is real.
sequence = torch.tensor([[1, 2, 3, 4, 9], [5, 6, 9, 0, 0]]) # 9 is EOS
inputs, targets = sequence[:, :-1], sequence[:, 1:].clone()
targets[sequence[:, 1:] == 0] = -100
logits, _ = model(inputs, real=inputs != 0)
loss = F.cross_entropy(logits.reshape(-1, 12), targets.reshape(-1), ignore_index=-100)
model.zero_grad(set_to_none=True)
loss.backward()
assert torch.isfinite(loss)
assert model.token.weight.grad is not None
print('causality, full-vs-cached, padding, shifted loss, and backward checks passed')
print('valid targets:', (targets != -100).sum().item(), 'loss:', round(loss.item(), 4))
