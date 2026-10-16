"""Handout 14: gated FFNs, block dependencies, shared gradients, and dimensions.
Run: python 14_architecture_checks.py (PyTorch, CPU, no downloads).
Original teaching code; component checks do not establish training quality or speed.
"""
import torch
from torch import nn
from torch.nn import functional as F


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


class FlexibleAttention(nn.Module):
    """Ordinary causal MHA: attention width is independent of residual width."""
    def __init__(self, d=10, heads=3, head_dim=4):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.attention_width = heads * head_dim
        self.qkv = nn.Linear(d, 3 * self.attention_width, bias=False)
        self.out = nn.Linear(self.attention_width, d, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for z in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        allowed = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~allowed, float('-inf')).softmax(-1)
        mixed = weights @ v
        merged = mixed.transpose(1, 2).contiguous().reshape(b, t, self.attention_width)
        return self.out(merged)


def run_checks():
    torch.manual_seed(14)
    torch.set_num_threads(1)
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    # Fixed numerical gate trace and parameter-matched FFNs.
    gate = torch.tensor([-1., 0., 1.])
    up = torch.tensor([2., -3., 4.])
    gated = F.silu(gate) * up
    torch.testing.assert_close(gated, torch.tensor([-0.53788284, 0., 2.92423439]))
    print('SwiGLU intermediate:', [round(v, 6) for v in gated.tolist()])
    assert F.gelu(torch.tensor(-1.)).item() < 0  # Deterministic, not a random gate.
    assert F.silu(torch.tensor(2.)).item() > 1  # Not a probability.
    # Optional product-rule exercise: zero gate output need not mean zero gate grad.
    gate_leaf = gate.clone().requires_grad_()
    up_leaf = up.clone().requires_grad_()
    (F.silu(gate_leaf) * up_leaf).sum().backward()
    sig = gate.sigmoid()
    expected_gate = up * (sig + gate * sig * (1 - sig))
    torch.testing.assert_close(gate_leaf.grad, expected_gate)
    torch.testing.assert_close(up_leaf.grad, F.silu(gate))
    torch.testing.assert_close(gate_leaf.grad[1], torch.tensor(-1.5))
    assert up_leaf.grad[1] == 0
    detached_gate = gate.clone().requires_grad_()
    connected_up = up.clone().requires_grad_()
    detached_output = F.silu(detached_gate).detach() * connected_up
    torch.testing.assert_close(detached_output, gated)
    detached_output.sum().backward()
    assert detached_gate.grad is None
    torch.testing.assert_close(connected_up.grad, up_leaf.grad)
    print('Gate gradients at g=0, u=-3: gate=-1.5, content=0; detach hides the gate path.')
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
    # A hypothetical bug: using the serial h as parallel residual doubles A.
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
    # D=10 is not divisible by H=3. The attention width A=12 is legal.
    attention = FlexibleAttention(d=10, heads=3, head_dim=4)
    assert count(attention) == 4 * 10 * 12 == 480
    seq = torch.randn(2, 4, 10)
    full = attention(seq)
    assert full.shape == seq.shape
    changed = seq.clone(); changed[:, 2:] += torch.randn_like(changed[:, 2:])
    torch.testing.assert_close(full[:, :2], attention(changed)[:, :2])
    # Compare the manual attention with the library primitive, including scale/mask.
    q, k, v = attention.qkv(seq).chunk(3, -1)
    q, k, v = [z.reshape(2, 4, 3, 4).transpose(1, 2) for z in (q, k, v)]
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference = reference.transpose(1, 2).contiguous().reshape(2, 4, 12)
    torch.testing.assert_close(full, attention.out(reference))
    full.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in attention.parameters())
    try:
        # A=12 cannot be split using the residual-derived width 10//3=3.
        attention.qkv(seq).chunk(3, -1)[0].reshape(2, 4, 3, 10 // 3)
    except RuntimeError:
        pass
    else:
        raise AssertionError('the incorrect residual-derived reshape must fail')
    # Rounding and equal block counts: these are exact matrix counts, not timings.
    assert 2 * 1024 * 4096 == 8_388_608
    assert 3 * 1024 * 2816 == 8_650_752
    assert (3 * 1024 * 2816) / (2 * 1024 * 4096) == 1.03125
    blocks = lambda layers, width: 12 * layers * width ** 2
    assert blocks(12, 768) == blocks(48, 384) == 84_934_656
    assert 48 * 384 == 2 * 12 * 768  # MHA cache/score-value arithmetic ratio.
    print('Independent attention width: D=10, H=3, head_dim=4; 480 parameters.')
    # Optional tying fractions: always identify the untied baseline denominator.
    removed_fraction = lambda vocab, layers, width: vocab / (12 * layers * width + 2 * vocab)
    small = removed_fraction(50_000, 12, 768)
    large = removed_fraction(50_000, 48, 1536)
    assert abs(small * 100 - 23.7425923112) < 1e-8
    assert abs(large * 100 - 5.0775030059) < 1e-8
    assert 50_000 * 768 == 38_400_000 and 50_000 * 1536 == 76_800_000
    assert large < small < 0.5
    assert abs(38 / 162 * 100 - 23.4567901235) < 1e-8
    assert abs(38 / 124 * 100 - 30.6451612903) < 1e-8
    print('Tying fractions from untied baselines:', round(100 * small, 2), round(100 * large, 2))
    # Warm-up answers: PyTorch orientation, object aliasing, and scalar dataflow.
    linear = nn.Linear(6, 12, bias=False)
    assert linear(torch.zeros(2, 3, 6)).shape == (2, 3, 12)
    assert linear(torch.zeros(4, 3, 6)).shape == (4, 3, 12)
    assert linear.weight.shape == (12, 6) and count(linear) == 72
    warm_a = nn.Parameter(torch.tensor([1., 2.]))
    warm_b = nn.Parameter(warm_a.detach().clone())
    warm_c = warm_a
    with torch.no_grad():
        warm_a.add_(1)
    assert warm_a is warm_c and warm_a is not warm_b
    assert len({id(w) for w in (warm_a, warm_b, warm_c)}) == 2
    torch.testing.assert_close(warm_a, torch.tensor([2., 3.]))
    torch.testing.assert_close(warm_c, warm_a)
    torch.testing.assert_close(warm_b, torch.tensor([1., 2.]))
    warm_x = 1
    warm_h = warm_x + 2 * warm_x
    assert warm_h + 3 * warm_h == 12
    assert warm_x + 2 * warm_x + 3 * warm_x == 6
    print('Warm-up: 72 weights; two Parameter objects; serial 12 versus parallel 6.')
    # Theory: one multiplicative hidden coordinate is a quadratic form.
    a = torch.tensor([1., 0.], dtype=torch.float64)
    b = torch.tensor([0., 1.], dtype=torch.float64)
    quadratic = (torch.outer(a, b) + torch.outer(b, a)) / 2
    for point in [[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., -3.]]:
        point = torch.tensor(point, dtype=torch.float64)
        torch.testing.assert_close((a @ point) * (b @ point), point @ quadratic @ point)
    # Matching the first three points forces a zero affine function, failing at (1,1).
    assert (a @ torch.ones(2, dtype=torch.float64)) * (b @ torch.ones(2, dtype=torch.float64)) == 1
    direction = torch.tensor([2., -3.], dtype=torch.float64)
    leading = 0.5 * (a @ direction) * (b @ direction)
    local_errors = []
    for radius in [1e-2, 1e-3, 1e-4]:
        point = radius * direction
        product = F.silu(a @ point) * (b @ point)
        local_errors.append(abs((product / radius**2 - leading).item()))
    assert local_errors[2] < local_errors[1] < local_errors[0]
    assert local_errors[-1] < 0.001
    # Theory: tying restricts the feasible set and can sum opposing gradients.
    shared = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    role_a, role_b = (shared - 1)**2, (shared + 1)**2
    grad_a = torch.autograd.grad(role_a, shared, retain_graph=True)[0]
    grad_b = torch.autograd.grad(role_b, shared, retain_graph=True)[0]
    total_grad = torch.autograd.grad(role_a + role_b, shared)[0]
    assert grad_a.item() == -2 and grad_b.item() == 2 and total_grad.item() == 0
    assert (role_a + role_b).item() == 2
    assert (1 - 1)**2 + (-1 + 1)**2 == 0  # Untied optimum.
    for candidate in [-3., -1., 0., 1., 3.]:
        assert (candidate - 1)**2 + (candidate + 1)**2 == 2 * candidate**2 + 2
    print('Theory: quadratic gates, local SwiGLU limit, and conflicting tied gradients pass.')
    print('All architecture checks passed.')


if __name__ == '__main__':
    run_checks()
