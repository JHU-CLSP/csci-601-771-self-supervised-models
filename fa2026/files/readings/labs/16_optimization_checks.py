"""Handout #16 CPU checks. Run: python3 16_optimization_checks.py

Requires PyTorch; no dataset or network. Tested with PyTorch 2.12.0.
The get_lr function is verbatim from nanoGPT/train.py, revision
3adf61e154c3fe3fca428ad6bc3818b27a3b8291 (MIT; see nanoGPT-LICENSE.txt).
Other examples are original scalar-regression, next-token LM, and packing demonstrations.
"""
import math
import torch


def full_gradient():
    weight = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    ((weight * x) ** 2).mean().backward()
    return weight.grad.item()


def equal_micro_gradient():
    weight = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    accumulation_steps = 2
    for x in (torch.tensor([1.0, 2.0], dtype=torch.float64),
              torch.tensor([3.0, 4.0], dtype=torch.float64)):
        loss = ((weight * x) ** 2).mean()
        (loss / accumulation_steps).backward()  # Try replacing / with *.
    return weight.grad.item()


def unequal_micro_gradient(token_weighted=True):
    weight = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    micro_batches = (torch.tensor([1.0], dtype=torch.float64),
                     torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    total_targets = sum(x.numel() for x in micro_batches)
    for x in micro_batches:
        squared_errors = (weight * x) ** 2
        if token_weighted:
            loss = squared_errors.sum() / total_targets
        else:
            loss = squared_errors.mean() / len(micro_batches)
        loss.backward()
    return weight.grad.item()


# Verbatim nanoGPT get_lr; only module-level configuration below is local.
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 2
lr_decay_iters = 10


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def check_gradients():
    full = full_gradient()
    equal = equal_micro_gradient()
    weighted = unequal_micro_gradient()
    wrong = unequal_micro_gradient(token_weighted=False)
    assert math.isclose(full, 15.0)
    assert math.isclose(equal, full), (equal, full)
    assert math.isclose(weighted, full), (weighted, full)
    assert math.isclose(wrong, 32 / 3)
    print(f"Full/equal/weighted gradients: {full:.6f}, {equal:.6f}, {weighted:.6f}")
    print(f"Naive unequal-mean gradient: {wrong:.6f}")
    print(f"One SGD step at 0.01: correct={1-0.01*full:.6f}, naive={1-0.01*wrong:.6f}")


def check_adamw():
    weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    opt = torch.optim.AdamW([weight], lr=0.1, weight_decay=0.2)
    values = []
    for _ in range(2):
        weight.grad = torch.zeros_like(weight)  # Zero, not None.
        opt.step()
        values.append(weight.item())
    assert math.isclose(values[0], 1.96)
    assert math.isclose(values[1], 1.9208)
    state = opt.state[weight]
    assert state['exp_avg'].numel() == weight.numel()
    assert state['exp_avg_sq'].numel() == weight.numel()
    assert torch.count_nonzero(state['exp_avg']).item() == 0
    assert torch.count_nonzero(state['exp_avg_sq']).item() == 0
    opt.zero_grad(set_to_none=True)
    before = weight.detach().clone()
    opt.step()  # PyTorch skips this parameter because grad is None.
    assert torch.equal(weight, before)
    print(f"AdamW with zero gradients: {values[0]:.4f}, {values[1]:.4f}; None gradient skips update")


def check_schedule():
    assert lr_decay_iters > warmup_iters >= 0  # Guard the configuration.
    expected = {0: 1e-3 / 3, 1: 2e-3 / 3, 2: 1e-3,
                6: 5.5e-4, 10: 1e-4, 11: 1e-4}
    for step, want in expected.items():
        assert math.isclose(get_lr(step), want), (step, get_lr(step), want)
    for step in range(warmup_iters, lr_decay_iters):
        assert get_lr(step) >= get_lr(step + 1)
    print("Cosine schedule at steps 0, 2, 6, 10:",
          ", ".join(f"{get_lr(step):.9f}" for step in (0, 2, 6, 10)))


def check_budget_examples():
    assert 2 * 8 * 4 * 2048 == 131_072
    assert 1_310_720_000 // 131_072 == 10_000
    assert 2 * 1024 * 8 * 2 == 32_768
    assert 327_680_000 // 32_768 == 10_000
    assert math.isclose((2 + 3 * 4) / 4, 3.5)


def lm_examples():
    # Vocabulary: 0=BOS, 1=a, 2=b, 3=c, 4=EOS.
    # Short row has two valid predictions and two ignored filler positions.
    inputs = torch.tensor([[0, 1, 4, 4], [0, 2, 3, 1]])
    targets = torch.tensor([[1, 4, -100, -100], [2, 3, 1, 4]])
    return inputs, targets


def lm_gradient(mode, initial_logits=None):
    inputs, targets = lm_examples()
    initial = torch.zeros((5, 5), dtype=torch.float64) if initial_logits is None else initial_logits
    table = initial.detach().clone().requires_grad_(True)
    total_targets = (targets != -100).sum().item()
    if mode == 'full':
        logits = table[inputs]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 5), targets.reshape(-1),
            ignore_index=-100, reduction='sum') / total_targets
        loss.backward()
        reported_loss = loss.item()
    elif mode in ('weighted', 'document_mean'):
        reported_loss = 0.0
        for x, y in zip(inputs, targets):
            logits = table[x]
            loss_sum = torch.nn.functional.cross_entropy(
                logits, y, ignore_index=-100, reduction='sum')
            if mode == 'weighted':
                loss = loss_sum / total_targets
            else:
                # Wrong for unequal valid counts: each document gets half the weight.
                loss = loss_sum / ((y != -100).sum() * len(inputs))
            loss.backward()
            reported_loss += loss.item()
    else:
        raise ValueError(mode)
    return table.grad.detach().clone(), reported_loss


def check_lm_accumulation():
    _, targets = lm_examples()
    assert (targets != -100).sum().item() == 6
    full, full_loss = lm_gradient('full')
    weighted, weighted_loss = lm_gradient('weighted')
    wrong, wrong_loss = lm_gradient('document_mean')
    torch.testing.assert_close(full, weighted, rtol=1e-12, atol=1e-12)
    # Check the next parameter vector as well as the gradient.
    initial = torch.zeros_like(full)
    torch.testing.assert_close(initial - 0.1 * full, initial - 0.1 * weighted)
    assert not torch.allclose(full, wrong)
    nonuniform = torch.arange(25, dtype=torch.float64).sin().reshape(5, 5)
    reference, _ = lm_gradient('full', nonuniform)
    accumulated, _ = lm_gradient('weighted', nonuniform)
    torch.testing.assert_close(reference, accumulated, rtol=1e-12, atol=1e-12)
    assert math.isclose(full_loss, math.log(5))
    assert math.isclose(full_loss, weighted_loss)
    assert math.isclose(full_loss, wrong_loss)  # Same scalar can hide different gradients.
    # EOS appears only in ignored input positions here: no gradient on that row.
    torch.testing.assert_close(full[4], torch.zeros(5, dtype=torch.float64))
    print(f"Masked next-token LM: 6 targets; loss={full_loss:.6f}; weighted gradient/update match")
    print(f"Naive document means: same initial loss, max gradient error={(full-wrong).abs().max().item():.6f}")


def check_packing():
    # One allocation: BOS_A, a, EOS_A, BOS_B, b, EOS_B.
    # BOS/EOS share vocabulary IDs across documents; membership is separate metadata.
    tokens = torch.tensor([0, 1, 4, 0, 2, 4])
    document_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2])
    targets = torch.cat((tokens[1:], torch.tensor([-100])))
    targets[:-1][document_ids[:-1] != document_ids[1:]] = -100
    assert targets.tolist() == [1, 4, -100, 2, 4, -100]
    assert (targets != -100).sum().item() == 4
    assert position_ids.tolist() == [0, 1, 2, 0, 1, 2]
    causal = torch.ones((6, 6), dtype=torch.bool).tril()
    same_document = document_ids[:, None] == document_ids[None, :]
    allowed = causal & same_document  # True means allowed in this demonstration.
    assert allowed.int().tolist() == [
        [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]]
    # Uniform scores isolate what the mask permits. Changing A cannot affect B.
    scores = torch.zeros((6, 6), dtype=torch.float64)
    attention = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
    values = torch.arange(6, dtype=torch.float64).reshape(6, 1)
    before = attention @ values
    changed = values.clone()
    changed[:3] += 100
    after = attention @ changed
    torch.testing.assert_close(before[3:], after[3:])
    # Under ordinary causal stream attention, B can see A, so the same change matters.
    stream = scores.masked_fill(~causal, -torch.inf).softmax(dim=-1)
    assert not torch.allclose((stream @ values)[3:], (stream @ changed)[3:])
    print("Packing: two causal blocks; 4 valid targets; B is isolated from changes to A")


if __name__ == '__main__':
    check_gradients()
    check_adamw()
    check_schedule()
    check_budget_examples()
    check_lm_accumulation()
    check_packing()
    print(f"All checks passed (PyTorch {torch.__version__}; CPU float64).")
