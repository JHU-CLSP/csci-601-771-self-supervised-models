"""Handout 18: reward-model mechanics on CPU (PyTorch only).
Run: python3 18_reward_model_checks.py
No models/data downloaded. Tiny scores/features test mechanics, not LM quality.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def pair_loss(chosen, rejected):
    # One scalar score per sequence; both tensors must have identical shapes.
    if chosen.shape != rejected.shape:
        raise ValueError('chosen/rejected score shapes must match')
    return -F.logsigmoid(chosen - rejected).mean()


def last_valid(hidden, mask):
    """Gather the last non-padding hidden state for either padding direction."""
    mask = mask.bool()
    if hidden.shape[:2] != mask.shape or not mask.any(dim=1).all():
        raise ValueError('match [batch, time] shapes and retain a token in every row')
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    index = positions.masked_fill(~mask, -1).amax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]


def check_loss_and_gradients():
    scores = torch.tensor([1.2, 0.8], dtype=torch.float64, requires_grad=True)
    delta = scores[0] - scores[1]
    loss = pair_loss(scores[:1], scores[1:])
    bce = F.binary_cross_entropy_with_logits(delta, torch.ones_like(delta))
    torch.testing.assert_close(loss, bce)
    torch.testing.assert_close(loss, F.softplus(-delta))
    torch.testing.assert_close(loss, torch.tensor(0.5130152524, dtype=torch.float64))
    loss.backward()
    expected = torch.tensor([-0.4013123399, 0.4013123399], dtype=torch.float64)
    torch.testing.assert_close(scores.grad, expected)
    updated = scores.detach() - 0.1 * scores.grad
    assert updated[0] - updated[1] > delta
    torch.testing.assert_close(updated.mean(), scores.detach().mean())
    print(f'pair: P(chosen wins)={delta.sigmoid().item():.6f}; loss={loss.item():.6f}')


def check_order_offset_and_scale():
    a = torch.tensor([1.2, -0.2, 2.0], dtype=torch.float64)
    b = torch.tensor([0.8, 0.4, -1.0], dtype=torch.float64)
    labels = torch.tensor([1., 0., 1.], dtype=torch.float64)  # 1 iff A wins.
    delta = a - b
    loss = F.binary_cross_entropy_with_logits(delta, labels)
    flipped = F.binary_cross_entropy_with_logits(-delta, 1 - labels)
    torch.testing.assert_close(loss, flipped)
    torch.testing.assert_close(pair_loss(a, b), pair_loss(a + 5, b + 5))
    assert not torch.allclose(delta.sigmoid(), (10 * delta).sigmoid())
    # Broadcasting can silently manufacture all cross-pairs with a manual loss.
    wrong = a[:, None] - b
    assert wrong.shape == (3, 3)
    try:
        pair_loss(a[:, None], b)
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched score shapes were not rejected')
    print('order swap + label flip agrees; common offset cancels; scaling does not')


def check_numerical_stability():
    delta = torch.tensor([-1000., 1000.], dtype=torch.float64, requires_grad=True)
    losses = F.softplus(-delta)
    assert torch.isfinite(losses).all()
    torch.testing.assert_close(losses, torch.tensor([1000., 0.], dtype=torch.float64))
    losses.sum().backward()
    torch.testing.assert_close(delta.grad, torch.tensor([-1., 0.], dtype=torch.float64))
    naive = -torch.log(torch.sigmoid(delta.detach()))
    assert not torch.isfinite(naive[0])
    d = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    correct = F.binary_cross_entropy_with_logits(d, torch.ones_like(d))
    wrong = F.binary_cross_entropy_with_logits(d.sigmoid(), torch.ones_like(d))
    correct_grad, = torch.autograd.grad(correct, d, retain_graph=True)
    wrong_grad, = torch.autograd.grad(wrong, d)
    assert not torch.allclose(correct, wrong)
    assert not torch.allclose(correct_grad, wrong_grad)
    print(f'double sigmoid: loss {wrong.item():.6f} vs correct {correct.item():.6f}; '
          f'gradient {wrong_grad.item():.6f} vs {correct_grad.item():.6f}')


def check_disagreement():
    # Seven A wins and three B wins for one pair.
    labels = torch.tensor([1.] * 7 + [0.] * 3, dtype=torch.float64)
    d = torch.tensor(math.log(7 / 3), dtype=torch.float64, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(d.expand_as(labels), labels)
    loss.backward()
    assert abs(d.grad.item()) < 1e-12
    torch.testing.assert_close(d.detach().sigmoid(), torch.tensor(0.7, dtype=torch.float64))
    assert loss.item() > 0
    print(f'disagreement: optimum P(A wins)=0.7; nonzero mean loss={loss.item():.6f}')


def check_padding():
    # These supplied states stand in for a contextual LM's output; 999 is padding.
    h = torch.tensor([[[1.], [2.], [999.]], [[999.], [3.], [4.]]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    pooled = last_valid(h, mask)
    torch.testing.assert_close(pooled, torch.tensor([[2.], [4.]]))
    assert not torch.equal(h[:, -1], pooled)
    # sum(mask)-1 only works for nonempty RIGHT-padded contiguous sequences.
    wrong_index = mask.sum(dim=1) - 1
    wrong = h[torch.arange(2), wrong_index]
    assert not torch.equal(wrong, pooled)
    try:
        last_valid(h, torch.zeros_like(mask))
    except ValueError:
        pass
    else:
        raise AssertionError('empty row was not rejected')
    print('pooling: last valid states are 2 and 4; both padding shortcuts fail')


def run_pairs(model, loader, optimizer=None):
    """One CPU pass; optimizer supplied = train, otherwise evaluate."""
    training = optimizer is not None
    model.train(training)  # Controls dropout/normalization behavior.
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):  # Separately controls autograd.
        for chosen, rejected in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            chosen_scores = model(chosen).squeeze(-1)
            rejected_scores = model(rejected).squeeze(-1)
            expected = (chosen.shape[0],)
            if chosen_scores.shape != expected or rejected_scores.shape != expected:
                raise ValueError('the scorer must return one scalar per sequence')
            loss = pair_loss(chosen_scores, rejected_scores)
            if training:
                loss.backward()
                optimizer.step()
            n = chosen_scores.numel()
            loss_sum += loss.detach().item() * n
            correct += (chosen_scores.detach() > rejected_scores.detach()).sum().item()
            total += n
    if total == 0:
        raise ValueError('the loader must contain at least one comparison')
    return {'loss': loss_sum / total, 'accuracy': correct / total, 'pairs': total}


def check_shared_scorer():
    torch.manual_seed(18)
    scorer = nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        scorer.weight.zero_()
        scorer.bias.zero_()
    chosen = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    rejected = torch.zeros_like(chosen)
    # Three pairs produce batches of sizes two and one: do not average means equally.
    loader = DataLoader(TensorDataset(chosen, rejected), batch_size=2, shuffle=False)
    initial = run_pairs(scorer, loader)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=0.2)
    for _ in range(30):
        training_metrics = run_pairs(scorer, loader, optimizer)
        assert scorer.training
        torch.testing.assert_close(scorer.bias.grad, torch.zeros_like(scorer.bias))
    assert training_metrics['pairs'] == 3
    before = {name: value.clone() for name, value in scorer.state_dict().items()}
    before_grads = {name: p.grad.clone() for name, p in scorer.named_parameters()}
    final = run_pairs(scorer, loader)
    assert not scorer.training
    assert final['loss'] < initial['loss'] and final['accuracy'] == 1.0
    for name, value in scorer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for name, p in scorer.named_parameters():
        torch.testing.assert_close(p.grad, before_grads[name], rtol=0, atol=0)
    assert torch.is_grad_enabled()  # The evaluation context restores the outer setting.
    with torch.no_grad():
        per_pair = F.softplus(-(scorer(chosen) - scorer(rejected)).squeeze(-1))
        torch.testing.assert_close(torch.tensor(final['loss'], dtype=torch.float64), per_pair.mean())
        wrong_mean = (per_pair[:2].mean() + per_pair[2:].mean()) / 2
        assert not torch.allclose(wrong_mean, per_pair.mean())
    # New synthetic feature pairs test evaluation plumbing, not real task transfer.
    held_chosen = torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64)
    held_rejected = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    held_loader = DataLoader(TensorDataset(held_chosen, held_rejected), batch_size=1)
    held = run_pairs(scorer, held_loader)
    assert held['accuracy'] == 1.0
    try:
        run_pairs(nn.Linear(2, 2, dtype=torch.float64), loader)
    except ValueError:
        pass
    else:
        raise AssertionError('a multi-output reward head was not rejected')
    print(f'shared scorer + DataLoader: loss {initial["loss"]:.6f} -> {final["loss"]:.6f}; '
          'unequal-batch aggregation and unchanged evaluation state/gradients verified')


if __name__ == '__main__':
    check_loss_and_gradients()
    check_order_offset_and_scale()
    check_numerical_stability()
    check_disagreement()
    check_padding()
    check_shared_scorer()
    print('All reward-model checks passed.')
