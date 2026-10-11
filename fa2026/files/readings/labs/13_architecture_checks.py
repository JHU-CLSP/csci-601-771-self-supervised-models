"""Handout 13: numerical checks for normalization, FFNs, layout, and tying.
Run: python 13_architecture_checks.py (requires PyTorch; CPU, no downloads).
Original teaching code. No timing claims are inferred from these small checks.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # Last axis = features; retain batch/time axes. Accumulate in float32
        # for half types, but preserve float64 for numerical checks.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        unit = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)
        return unit.to(x.dtype) * self.weight.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TinyTiedHead(nn.Module):
    def __init__(self, vocab=5, d=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # Share the Parameter object.

    def forward(self, ids):
        return self.head(self.embed(ids))


def run_checks():
    torch.manual_seed(13)
    torch.set_num_threads(1)
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    ln = nn.LayerNorm(4, eps=1e-6).double()
    rms = RMSNorm(4).double()
    torch.testing.assert_close(ln(x), ln(x + 5))
    assert not torch.allclose(rms(x), rms(x + 5))
    # Positive-scale invariance is approximate when epsilon is nonzero.
    torch.testing.assert_close(rms(3*x), rms(x), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(rms(torch.zeros_like(x))).all()
    print('LN [1,2,3,4]:', [round(v, 6) for v in ln(x).tolist()])
    print('RMS [1,2,3,4]:', [round(v, 6) for v in rms(x).tolist()])
    assert rms(torch.randn(2, 3, 4, dtype=torch.float64)).shape == (2, 3, 4)
    # Changing a different token must not change row 0's normalization.
    rows = torch.randn(2, 3, 4, dtype=torch.float64)
    changed = rows.clone(); changed[:, 1:] += 10
    torch.testing.assert_close(rms(rows)[:, 0], rms(changed)[:, 0])
    low = RMSNorm(4)(torch.tensor([[1000., 2000., 3000., 4000.]], dtype=torch.float16))
    assert low.dtype == torch.float16 and torch.isfinite(low).all()
    high = RMSNorm(4)(torch.tensor([[1000., 2000., 3000., 4000.]])).half()
    torch.testing.assert_close(low, high, atol=1e-3, rtol=1e-3)

    # Registered gains must receive gradients and be updated by an optimizer.
    assert dict(rms.named_parameters())['weight'] is rms.weight
    rms(x).sum().backward()
    assert rms.weight.grad is not None and torch.isfinite(rms.weight.grad).all()
    gain_before = rms.weight.detach().clone()
    gain_grad = rms.weight.grad.detach().clone()
    torch.optim.SGD(rms.parameters(), lr=0.1).step()
    torch.testing.assert_close(rms.weight, gain_before - 0.1 * gain_grad)

    # Zero branch: pre-norm preserves x and its identity Jacobian;
    # post-norm normalizes x, removing the common-shift direction.
    pre = lambda z: z + 0 * ln(z)
    post = lambda z: ln(z + 0 * z)
    eye = torch.eye(4, dtype=torch.float64)
    jpre = torch.autograd.functional.jacobian(pre, x)
    jpost = torch.autograd.functional.jacobian(post, x)
    torch.testing.assert_close(jpre, eye)
    torch.testing.assert_close(jpost @ torch.ones_like(x), torch.zeros_like(x))

    # Fixed numerical gate trace and parameter-matched FFNs.
    gate = torch.tensor([-1., 0., 1.])
    up = torch.tensor([2., -3., 4.])
    gated = F.silu(gate) * up
    torch.testing.assert_close(gated, torch.tensor([-0.53788284, 0., 2.92423439]))
    print('SwiGLU intermediate:', [round(v, 6) for v in gated.tolist()])
    assert F.gelu(torch.tensor(-1.)).item() < 0  # Deterministic, not a random gate.
    assert F.silu(torch.tensor(2.)).item() > 1  # Not a probability.
    d, dense_hidden, gated_hidden = 12, 48, 32
    dense = nn.Sequential(nn.Linear(d, dense_hidden, bias=False), nn.GELU(),
                          nn.Linear(dense_hidden, d, bias=False))
    glu = SwiGLU(d, gated_hidden)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(dense) == count(glu) == 1152
    seq = torch.randn(2, 3, d)
    assert glu(seq).shape == seq.shape
    glu(seq).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in glu.parameters())
    # Fusing up/gate projections changes scheduling, not the mathematical gate.
    merged = F.linear(seq, torch.cat([glu.gate.weight, glu.up.weight], dim=0))
    g, u = merged.chunk(2, dim=-1)
    torch.testing.assert_close(glu(seq), glu.down(F.silu(g) * u))

    # Toy branches isolate dependencies (not attention or a full decoder).
    # A(N(x)) changes x non-uniformly; M(z)=z. Same N for both branches.
    norm = nn.LayerNorm(4, elementwise_affine=False, eps=1e-6).double()
    a = lambda z: z * torch.tensor([1., 0., 0., 0.], dtype=z.dtype)
    m = lambda z: z
    h = x + a(norm(x))
    serial = h + m(norm(h))
    parallel = x + a(norm(x)) + m(norm(x))
    # Current printed slide 29 uses h as the parallel residual, doubling A.
    wrong = h + m(norm(x)) + a(norm(x))
    assert not torch.allclose(serial, parallel)
    torch.testing.assert_close(wrong - parallel, a(norm(x)))
    torch.testing.assert_close(serial, torch.tensor(
        [-1.897815, 1.897802, 3.518725, 5.139648], dtype=x.dtype), atol=1e-6, rtol=0)
    torch.testing.assert_close(parallel, torch.tensor(
        [-1.683280, 1.552787, 3.447213, 5.341640], dtype=x.dtype), atol=1e-6, rtol=0)
    print('Serial toy:', [round(v, 6) for v in serial.tolist()])
    print('Parallel toy:', [round(v, 6) for v in parallel.tolist()])

    # A copied value is not a tied parameter.
    model = TinyTiedHead().double()
    assert model.head.weight is model.embed.weight
    assert count(model) == 15
    ids, targets = torch.tensor([0, 0]), torch.tensor([1, 2])
    base = model.embed.weight.detach().clone()
    embed_w = base.clone().requires_grad_()
    head_w = base.clone().requires_grad_()
    loss = F.cross_entropy(F.embedding(ids, embed_w) @ head_w.T, targets)
    loss.backward()
    # Independently derive both routes. CE uses mean reduction across examples.
    hidden = F.embedding(ids, base)
    prob = (hidden @ base.T).softmax(-1)
    delta = (prob - F.one_hot(targets, num_classes=5).to(prob.dtype)) / len(ids)
    manual_head = delta.T @ hidden
    manual_embed = torch.zeros_like(base)
    manual_embed.index_add_(0, ids, delta @ base)
    torch.testing.assert_close(head_w.grad, manual_head)
    torch.testing.assert_close(embed_w.grad, manual_embed)
    assert torch.count_nonzero(embed_w.grad[4]) == 0
    assert torch.count_nonzero(head_w.grad[4]) > 0
    F.cross_entropy(model(ids), targets).backward()
    torch.testing.assert_close(model.embed.weight.grad, embed_w.grad + head_w.grad)
    # Break tying with a clone, then verify that the identity check detects it.
    broken = TinyTiedHead().double()
    broken.head.weight = nn.Parameter(broken.embed.weight.detach().clone())
    assert torch.equal(broken.embed.weight, broken.head.weight)
    assert broken.head.weight is not broken.embed.weight and count(broken) == 30
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.embed.weight.detach().clone()
    gradient = model.embed.weight.grad.detach().clone()
    opt.step()
    torch.testing.assert_close(model.embed.weight, before - 0.1 * gradient)
    assert model.head.weight is model.embed.weight
    print('Tying: shared gradient = input gradient + output gradient; unique params 15 vs 30.')
    print('All architecture checks passed.')


if __name__ == '__main__':
    run_checks()
