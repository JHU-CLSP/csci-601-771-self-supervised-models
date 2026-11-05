#!/usr/bin/env python3
"""Lecture 19: exact finite-model checks for slides 24–36.
Run: python3 19_policy_gradient_checks.py (PyTorch, CPU; no downloads).
Original teaching code except the explicitly credited upstream function string.
"""
import math
import torch
from torch.distributions import Categorical

DTYPE = torch.float64


def close(actual, expected, atol=1e-10):
    torch.testing.assert_close(torch.as_tensor(actual, dtype=DTYPE),
                               torch.as_tensor(expected, dtype=DTYPE),
                               atol=atol, rtol=1e-9)


def binary(theta):
    p = theta.sigmoid()
    return torch.stack((p, 1 - p))


def policy_loss(logits, targets, response_mask, rewards):
    """Aligned [B,T,V] logits, [B,T] targets/mask, [B] rewards."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("expected aligned [B,T,V] logits and [B,T] targets")
    if response_mask.shape != targets.shape or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean [B,T]")
    if rewards.shape != (logits.shape[0],):
        raise ValueError("expected one reward per response")
    if not response_mask.any(dim=1).all():
        raise ValueError("each row needs at least one response target")
    token_logp = logits.log_softmax(-1).gather(
        -1, targets.unsqueeze(-1)).squeeze(-1)
    sequence_logp = token_logp.masked_fill(~response_mask, 0).sum(-1)
    return -(rewards.detach() * sequence_logp).mean()


def check_score_function():
    theta = torch.tensor(math.log(1 / 3), dtype=DTYPE, requires_grad=True)
    probs = binary(theta)
    rewards = torch.tensor([3., 1.], dtype=DTYPE)
    exact = torch.autograd.grad((probs * rewards).sum(), theta, retain_graph=True)[0]
    possible = torch.stack([
        rewards[i] * torch.autograd.grad(probs[i].log(), theta, retain_graph=True)[0]
        for i in range(2)])
    close(possible, [2.25, -0.25])
    close((probs.detach() * possible).sum(), exact)
    close(exact, .375)
    close((probs * rewards).sum(), 1.5)
    for baseline in [0., 1., -10.]:
        coefficients = rewards - baseline
        grads = torch.stack([
            coefficients[i] * torch.autograd.grad(probs[i].log(), theta, retain_graph=True)[0]
            for i in range(2)])
        close((probs.detach() * grads).sum(), exact)
        variance = (probs.detach() * (grads - exact).square()).sum()
        if baseline == 0:
            close(variance, 1.171875)
        if baseline == 1:
            close(variance, .421875)
    score_mean = sum(probs[i].detach() * torch.autograd.grad(
        probs[i].log(), theta, retain_graph=True)[0] for i in range(2))
    close(score_mean, 0.)  # Any constant reward has zero exact policy gradient.
    close(torch.sigmoid(theta.detach() + .1 * possible[0]), .29450321618895675)
    close(torch.sigmoid(theta.detach() + .1 * possible[1]), .2453418560090823)
    close(torch.sigmoid(theta.detach() - .1 * possible[0]), .2102179305740014)


def check_sequence_loss():
    # t=0 predicts a prompt token. Response EOS is at t=2 / t=3.
    logits = torch.zeros((2, 5, 3), dtype=DTYPE, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 0, 2, 0]])
    mask = torch.tensor([[False, True, True, False, False],
                         [False, True, True, True, False]])
    rewards = torch.tensor([3., -1.], dtype=DTYPE, requires_grad=True)
    loss = policy_loss(logits, targets, mask, rewards)
    close(loss, 1.5 * math.log(3))  # (3*2 - 1*3) * log(3) / 2
    loss.backward()
    assert rewards.grad is None, 'reward coefficients must be detached'
    close(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    expected = torch.zeros_like(logits)
    for b, t in mask.nonzero().tolist():
        expected[b, t] = rewards.detach()[b] / 2 * torch.full((3,), 1/3, dtype=DTYPE)
        expected[b, t, targets[b, t]] -= rewards.detach()[b] / 2
    close(logits.grad, expected)
    # Explicitly verify ascent for a positively rewarded selected response.
    new_loss = policy_loss(logits.detach().clone().requires_grad_(), targets, mask,
                           torch.tensor([3., 0.], dtype=DTYPE))
    assert new_loss.item() > 0, 'positive reward needs negative log-probability loss'
    changed = targets.clone()
    changed[~mask] = 2
    close(policy_loss(logits.detach(), changed, mask, rewards.detach()), loss.detach())
    # Masking out EOS changes the sequence sum, not merely a display annotation.
    no_eos = mask.clone()
    no_eos[0, 2] = False
    no_eos[1, 3] = False
    close(policy_loss(logits.detach(), targets, no_eos, rewards.detach()), .5*math.log(3))
    # Shape mistakes must not silently broadcast coefficients across examples.
    for bad_rewards in [rewards[:, None], rewards[:1]]:
        try:
            policy_loss(logits, targets, mask, bad_rewards)
        except ValueError:
            pass
        else:
            raise AssertionError('bad reward shape was accepted')
    try:
        policy_loss(logits, targets, torch.zeros_like(mask), rewards)
    except ValueError:
        pass
    else:
        raise AssertionError('empty response accepted')


def check_parallel_response_scoring():
    """One causal pass equals separate prefix conditionals; no timing claim."""
    torch.manual_seed(19)
    width, vocab = 8, 7
    embedding = torch.nn.Embedding(vocab, width).double()
    positions = torch.nn.Embedding(16, width).double()
    block = torch.nn.TransformerEncoderLayer(
        d_model=width, nhead=2, dim_feedforward=16,
        dropout=0., batch_first=True).double().eval()
    head = torch.nn.Linear(width, vocab).double()
    calls = []

    def forward(ids):
        calls.append(ids.shape[1])
        n = ids.shape[1]
        h = embedding(ids) + positions(torch.arange(n))
        causal = torch.ones(n, n, dtype=torch.bool).triu(1)
        return head(block(h, src_mask=causal))

    prompt = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5, 6, 0]])  # Final 0 is EOS.
    # Last response token is a target, never needed as an input to score itself.
    inputs = torch.cat((prompt, response[:, :-1]), dim=1)
    logits = forward(inputs)
    n_prompt, n_response = prompt.shape[1], response.shape[1]
    selected_logits = logits[:, n_prompt - 1:]
    bulk = selected_logits.log_softmax(-1).gather(
        -1, response.unsqueeze(-1)).squeeze(-1)
    assert calls == [6] and bulk.shape == (1, 4)
    calls.clear()
    separate = []
    with torch.no_grad():
        for t in range(n_response):
            prefix = inputs[:, :n_prompt + t]
            logp = forward(prefix)[:, -1].log_softmax(-1)
            separate.append(logp[0, response[0, t]])
        close(bulk.detach()[0], torch.stack(separate))
        close(bulk.detach().sum(), torch.stack(separate).sum())
        assert calls == [3, 4, 5, 6]
        # A later supplied input must not leak into earlier predictions.
        changed = inputs.clone()
        changed[0, -1] = 1
        close(forward(changed)[:, :-1], logits.detach()[:, :-1])
    # Teacher-forced scoring also retains the policy gradient when requested.
    (-bulk.sum()).backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    # The sequential values above deliberately recompute prefixes, rather than
    # implementing a KV cache. This verifies causal probability parity, not speed.


def check_kl_gradient():
    theta = torch.tensor(math.log(4), dtype=DTYPE, requires_grad=True)
    probs = binary(theta)
    ref = torch.tensor([.5, .5], dtype=DTYPE)
    rewards = torch.tensor([3., 1.], dtype=DTYPE)
    logratio = probs.log() - ref.log()
    kl = (probs * logratio).sum()
    objective = (probs * rewards).sum() - kl
    exact = torch.autograd.grad(objective, theta, retain_graph=True)[0]
    # Enumerating samples supplies their fixed current probabilities as weights.
    # Both the shaped coefficient and sampling weights are detached.
    shaped = (rewards - logratio).detach()
    surrogate = -(probs.detach() * shaped * probs.log()).sum()
    estimate = -torch.autograd.grad(surrogate, theta, retain_graph=True)[0]
    close(estimate, exact)
    close(kl, .1927447570217575)
    close(objective, 2.4072552429782426)
    close(exact, .09819290222081749)
    assert logratio[1] < 0 and kl > 0
    # Naively differentiate only fixed-sample shaped rewards: expected gradient 0.
    naive = (probs.detach() * (rewards - logratio)).sum()
    naive_gradient = torch.autograd.grad(naive, theta)[0]
    close(naive_gradient, 0.)
    assert abs((naive_gradient - exact).item()) > .09
    close(.16*(2 - 2*math.log(4)), -.123614195558365)
    # At the reference, KL and its first derivative vanish.
    zero = torch.tensor(0., dtype=DTYPE, requires_grad=True)
    pp = binary(zero)
    kk = (pp * (pp.log() - ref.log())).sum()
    close(kk, 0.)
    close(torch.autograd.grad(kk, zero)[0], 0.)


def check_practice_and_hacking():
    close(.6*2 + .4*(-1), .8)
    close((2-(-1))*.6*.4, .72)
    close(.6*.8 + .4*.6, .72)
    p = torch.tensor([.25, .75], dtype=DTYPE)
    q = torch.tensor([.5, .5], dtype=DTYPE)
    kl = (p * (p/q).log()).sum()
    close(kl, .13081203594113697)
    close(1 - .5*kl, .9345939820294315)
    initial = torch.tensor([.7, .2, .1], dtype=DTYPE)
    updated = torch.tensor([.2, .1, .7], dtype=DTYPE)
    rewards = torch.tensor([2., 1., 5.], dtype=DTYPE)
    quality = torch.tensor([3., 1., -2.], dtype=DTYPE)
    close(initial @ rewards, 2.1)
    close(updated @ rewards, 4.)
    close(initial @ quality, 2.1)
    close(updated @ quality, -.7)


# Verbatim from pytorch/examples, reinforcement_learning/reinforce.py, lines 54–60
# Revision acc295dc7b90714f1bf47f06004fc19a7fe235c4.
# Copyright (c) 2017, Pytorch contributors. BSD 3-Clause:
# see PYTORCH-EXAMPLES-LICENSE.txt beside this file.
UPSTREAM_SELECT_ACTION = 'def select_action(state):\n    state = torch.from_numpy(state).float().unsqueeze(0)\n    probs = policy(state)\n    m = Categorical(probs)\n    action = m.sample()\n    policy.saved_log_probs.append(m.log_prob(action))\n    return action.item()\n'


def check_upstream_path():
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(math.log(1/3)))
            self.saved_log_probs = []

        def forward(self, state):
            return binary(self.theta).unsqueeze(0)

    for detached in [False, True]:
        torch.manual_seed(19)
        policy = TinyPolicy()
        namespace = {'torch': torch, 'Categorical': Categorical, 'policy': policy}
        source = UPSTREAM_SELECT_ACTION
        if detached:
            source = source.replace('m.log_prob(action)', 'm.log_prob(action).detach()')
        exec(compile(source, '<pinned select_action>', 'exec'), namespace)
        action = namespace['select_action'](torch.zeros(4).numpy())
        assert action in (0, 1)
        loss = -3 * policy.saved_log_probs[0].sum()
        if detached:
            try:
                loss.backward()
            except RuntimeError:
                pass
            else:
                raise AssertionError('detached log-probability unexpectedly backpropagated')
        else:
            loss.backward()
            assert policy.theta.grad is not None and policy.theta.grad.abs() > 0
    # Controlled one-action test of the actual signed expression in finish_episode.
    for wrong_sign, expected in [(False, .29450321618895675), (True, .2102179305740014)]:
        theta = torch.tensor(math.log(1/3), dtype=DTYPE, requires_grad=True)
        log_prob, R = theta.sigmoid().log(), 3.
        expression = 'log_prob * R' if wrong_sign else '-log_prob * R'
        loss = eval(expression, {'log_prob': log_prob, 'R': R})
        gradient = torch.autograd.grad(loss, theta)[0]
        close(torch.sigmoid(theta - .1*gradient), expected)


def main():
    for check in [check_score_function, check_sequence_loss,
                  check_parallel_response_scoring, check_kl_gradient,
                  check_practice_and_hacking, check_upstream_path]:
        check()
        print('PASS', check.__name__)
    print('All lecture 19 checks passed on PyTorch', torch.__version__, '(CPU).')


if __name__ == '__main__':
    main()
