"""Handout 13: normalization axes, precision, learned gains, and residual paths.
Run: python 13_architecture_checks.py (PyTorch, CPU, no downloads).
Original teaching code.
"""
import torch
from torch import nn


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

    # Pre-norm normalizes branch input, not the stream being incremented.
    update = torch.ones_like(x)
    stream = x.clone()
    residual_rms = []
    for k in range(4):
        torch.testing.assert_close(stream, x + k * update)
        torch.testing.assert_close(ln(stream), ln(x))
        residual_rms.append(stream.square().mean().sqrt().item())
        stream = stream + update  # Toy S(N(stream)) returns the fixed update.
    torch.testing.assert_close(torch.tensor([residual_rms[i] for i in [0, 1, 3]]),
                               torch.tensor([2.7386128, 3.6742346, 5.6124861]))
    assert all(a < b for a, b in zip(residual_rms, residual_rms[1:]))
    print('Residual stream grows under constant updates; LN branch input is unchanged.')
    # Warm-up answers: normalized scale does not constrain a later readout.
    normalized = torch.tensor([1., -1.])
    assert normalized.mean() == 0 and normalized.var(unbiased=False) == 1
    assert torch.tensor([100., 0.]) @ normalized == 100
    centered = []
    for values in ([1., 3.], [101., 103.]):
        vector = torch.tensor(values)
        centered.append((vector - vector.mean()) / vector.std(unbiased=False))
    torch.testing.assert_close(centered[0], torch.tensor([-1., 1.]))
    torch.testing.assert_close(centered[0], centered[1])
    assert 100 / 100 == 1 and 100 / 10 == 10  # FLOPs per transferred byte.
    print('Warm-up: unequal readout influence, lost common offset, intensities 1 and 10.')
    # Theory: radial/tangent sensitivities and the LN null directions.
    theory_x = torch.tensor([1., 2., 4.], dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    for epsilon in [0., 1e-3]:
        normalize = lambda z: z / (z.square().mean() + epsilon).sqrt()
        scale = (theory_x.square().mean() + epsilon).sqrt()
        analytic = identity / scale - torch.outer(theory_x, theory_x) / (3 * scale**3)
        jacobian = torch.autograd.functional.jacobian(normalize, theory_x)
        torch.testing.assert_close(jacobian, analytic)
        torch.testing.assert_close(jacobian @ theory_x, epsilon / scale**3 * theory_x)
        tangent = torch.tensor([2., -1., 0.], dtype=torch.float64)
        torch.testing.assert_close(jacobian @ tangent, tangent / scale)
    zero = torch.zeros(3, dtype=torch.float64)
    epsilon = 1e-6
    at_zero = torch.autograd.functional.jacobian(
        lambda z: z / (z.square().mean() + epsilon).sqrt(), zero)
    torch.testing.assert_close(at_zero, identity / epsilon**0.5)
    ln_exact = lambda z: (z - z.mean()) / z.var(unbiased=False).sqrt()
    ln_jacobian = torch.autograd.functional.jacobian(ln_exact, theory_x)
    torch.testing.assert_close(ln_jacobian @ torch.ones_like(theory_x), zero)
    torch.testing.assert_close(ln_jacobian @ (theory_x - theory_x.mean()), zero)
    assert torch.linalg.matrix_rank(ln_jacobian).item() == 1  # D - 2
    # Theory: fixed orthogonal update, unchanged RMS-normalized branch input.
    angles = []
    for magnitude in [1., 10.]:
        residual = torch.tensor([magnitude, 0.], dtype=torch.float64)
        normalized = residual / residual.square().mean().sqrt()
        torch.testing.assert_close(normalized, torch.tensor([2**0.5, 0.], dtype=torch.float64))
        updated = residual + torch.tensor([0., 1.], dtype=torch.float64)
        angle = torch.acos(torch.dot(residual, updated) / (residual.norm() * updated.norm()))
        torch.testing.assert_close(angle, torch.atan(torch.tensor(1 / magnitude, dtype=torch.float64)))
        angles.append(torch.rad2deg(angle).item())
    assert abs(angles[0] - 45) < 1e-10 and abs(angles[1] - 5.7105931375) < 1e-9
    print('Theory: normalization Jacobians/null directions and residual angles pass.')
    print('All normalization checks passed.')


if __name__ == '__main__':
    run_checks()
