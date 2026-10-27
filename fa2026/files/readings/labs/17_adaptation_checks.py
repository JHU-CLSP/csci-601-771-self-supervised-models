"""CPU checks for Handout 17. Run: python3 17_adaptation_checks.py
Requires PyTorch; downloads no models or data. Uses float64 for parity checks.
The tiny random networks check mechanics, not real adaptation quality.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(17)
torch.set_default_dtype(torch.float64)


class LoRALinear(nn.Module):
    """Original teaching implementation; no bias, dropout, or automatic merge."""
    def __init__(self, din, dout, r=2, alpha=8):
        super().__init__()
        self.base = nn.Linear(din, dout, bias=False)
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, din) * 0.1)
        self.B = nn.Parameter(torch.zeros(dout, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def check_response_loss():
    # Toy token IDs, NOT a real tokenizer: BOS, instruction, separator, y1, y2, EOS, PAD.
    ids = torch.tensor([[0, 1, 2, 4, 5, 6, 7]])
    labels = ids.clone()
    labels[:, :3] = -100
    labels[:, 6:] = -100
    logits = torch.zeros(1, 7, 8, requires_grad=True)
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(shifted_logits.view(-1, 8),
                           shifted_labels.view(-1), ignore_index=-100)
    loss.backward()
    assert (shifted_labels != -100).sum().item() == 3
    torch.testing.assert_close(loss.detach(), torch.tensor(math.log(8)))
    # The separator position predicts y1; EOS as a target teaches stopping.
    assert logits.grad[0, 2, 4] < 0
    assert logits.grad[0, 4, 6] < 0
    assert torch.count_nonzero(logits.grad[0, [0, 1, 5, 6]]) == 0
    print(f'response loss: 3 scored targets; mean CE = {loss.item():.6f}')


def check_initialization_and_merge():
    layer = LoRALinear(4, 3)
    x = torch.tensor([[1., 2., -1., 0.5], [-2., 1., 1., 3.]])
    target = torch.ones(2, 3)
    torch.testing.assert_close(layer(x), layer.base(x), rtol=0, atol=0)
    before = layer.base.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    F.mse_loss(layer(x), target).backward()
    assert torch.count_nonzero(layer.A.grad) == 0
    assert layer.B.grad.norm() > 0
    assert layer.base.weight.grad is None
    optimizer.step()
    torch.testing.assert_close(layer.base.weight, before, rtol=0, atol=0)
    assert layer.B.norm() > 0
    with torch.no_grad():
        merged = layer.base.weight + layer.scale * (layer.B @ layer.A)
        reference = layer(x)
        error = (reference - F.linear(x, merged)).abs().max().item()
        torch.testing.assert_close(reference, F.linear(x, merged), rtol=1e-12, atol=1e-12)
        wrong = F.linear(x, layer.base.weight + layer.B @ layer.A)
        assert not torch.allclose(reference, wrong)
    print(f'initialization: A gradient zero, B gradient nonzero; merged max error = {error:.2e}')
    # Both factors zero is a stationary point of the bilinear adapter branch.
    dead = LoRALinear(4, 3)
    with torch.no_grad():
        dead.A.zero_()
    F.mse_loss(dead(x), target).backward()
    assert torch.count_nonzero(dead.A.grad) == 0
    assert torch.count_nonzero(dead.B.grad) == 0
    print('both factors zero: both adapter gradients zero (no weight decay)')


class TinyAdapterChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = LoRALinear(4, 4)
        self.bridge = nn.Linear(4, 4, bias=False)
        self.bridge.requires_grad_(False)
        self.late = LoRALinear(4, 4)
        self.head = nn.Linear(4, 3, bias=False)
        self.head.requires_grad_(False)

    def forward(self, x, cut_bridge=False):
        h = torch.tanh(self.early(x))
        if cut_bridge:
            # Deliberate bug: frozen weights do not justify cutting this graph.
            with torch.no_grad():
                h = torch.tanh(self.bridge(h))
        else:
            h = torch.tanh(self.bridge(h))
        return self.head(torch.tanh(self.late(h)))


def check_frozen_bridge():
    good = TinyAdapterChain()
    broken = copy.deepcopy(good)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 2])
    F.cross_entropy(good(x), y).backward()
    F.cross_entropy(broken(x, cut_bridge=True), y).backward()
    assert good.early.B.grad.norm() > 0
    assert good.late.B.grad.norm() > 0
    assert good.bridge.weight.grad is None
    assert good.head.weight.grad is None
    assert broken.early.B.grad is None
    assert broken.late.B.grad.norm() > 0
    print('frozen bridge: early adapter learns; no_grad bug silently blocks its gradient')


def check_head_only():
    backbone = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4), nn.Tanh())
    backbone.requires_grad_(False)
    head = nn.Linear(4, 8, bias=False)  # Separate parameter: no tied input embedding.
    x = torch.randn(3, 4)
    h = backbone(x)
    assert not h.requires_grad
    F.cross_entropy(head(h), torch.tensor([1, 2, 3])).backward()
    assert head.weight.grad is not None
    assert all(p.grad is None for p in backbone.parameters())
    print('head only: backbone output needs no gradient; head still receives a gradient')


def check_counts():
    d, layers, vocab = 4096, 32, 32000
    head = d * vocab
    counts = [layers * 4 * (2 * d * r) for r in [1, 4, 16]]
    assert head == 131072000
    assert counts == [1048576, 4194304, 16777216]
    rectangular = LoRALinear(6, 10, r=2)
    assert rectangular.A.numel() + rectangular.B.numel() == 32
    print(f'parameter counts: head = {head:,}; QKVO LoRA ranks 1/4/16 = {counts}')


if __name__ == '__main__':
    check_response_loss()
    check_initialization_and_merge()
    check_frozen_bridge()
    check_head_only()
    check_counts()
    print('All adaptation checks passed.')
