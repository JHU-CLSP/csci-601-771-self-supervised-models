#!/usr/bin/env python3
"""Lecture 20: finite-model advantage, importance, and PPO checks.
Requires PyTorch on CPU. No network, pretrained model, or Gym dependency.
Original teaching code except the explicitly credited upstream fragment.
"""
import math
import re
import textwrap
from types import SimpleNamespace
import torch

DTYPE = torch.float64


def close(actual, expected, atol=1e-10):
    torch.testing.assert_close(torch.as_tensor(actual, dtype=DTYPE),
                               torch.as_tensor(expected, dtype=DTYPE),
                               atol=atol, rtol=1e-9)


def clipped_policy_loss(new_logp, old_logp, advantages, epsilon=0.2):
    if new_logp.ndim != 1 or new_logp.numel() == 0:
        raise ValueError("expected a nonempty vector of valid action log-probabilities")
    if old_logp.shape != new_logp.shape or advantages.shape != new_logp.shape:
        raise ValueError("new/old log-probabilities and advantages must match")
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0, 1)")
    ratio = (new_logp - old_logp.detach()).exp()
    if not torch.isfinite(ratio).all() or not torch.isfinite(advantages).all():
        raise ValueError("ratios and advantages must be finite")
    a = advantages.detach()
    raw = ratio * a
    bounded = ratio.clamp(1 - epsilon, 1 + epsilon) * a
    return -torch.minimum(raw, bounded).mean()


def check_verifier_coverage():
    cases = [(1, 3), (7, 2)]
    wrong = lambda a, b: max(abs(a), abs(b))
    assert all(wrong(a, b) == max(a, b) for a, b in cases)
    assert wrong(-2, -5) != max(-2, -5)
    parse_last = lambda text: int(re.findall(r"-?\d+", text)[-1])
    assert parse_last("Final answer: 9") == 9
    assert parse_last("Final answer: 9; confidence: 80") != 9
    close(sum([.2, .4]), .6)
    close(sum([.2, .4])/2, .3)
    close(sum([.2]*4), .8)
    close(sum([.2]*4)/4, .2)


def check_advantages():
    rewards = torch.tensor([2., 5., 8.], dtype=DTYPE)
    equal = torch.full((3,), 1/3, dtype=DTYPE)
    close(equal @ rewards, 5.)
    close(rewards - equal @ rewards, [-3., 0., 3.])
    changed = torch.tensor([.6, .3, .1], dtype=DTYPE)
    close(changed @ rewards, 3.5)
    close(rewards - changed @ rewards, [-1.5, 1.5, 4.5])
    theta = torch.tensor(0., dtype=DTYPE, requires_grad=True)
    p = torch.stack((theta.sigmoid(), 1-theta.sigmoid()))
    r = torch.tensor([2., 0.], dtype=DTYPE)
    baseline = p @ r
    advantage = r - baseline
    close(p @ advantage, 0.)
    moving_zero = torch.autograd.grad(p @ advantage, theta, retain_graph=True)[0]
    close(moving_zero, 0.)
    loss = -(p.detach() * advantage.detach() * p.log()).sum()
    close(-torch.autograd.grad(loss, theta)[0], .5)
    # A prompt-only baseline keeps the expected score-function gradient.
    theta = torch.tensor(-.7, dtype=DTYPE, requires_grad=True)
    p = torch.stack((theta.sigmoid(), 1-theta.sigmoid()))
    exact = torch.autograd.grad(p @ r, theta, retain_graph=True)[0]
    for b in [-1., 0., 1., 4.]:
        surrogate = -(p.detach() * (r-b) * p.log()).sum()
        close(-torch.autograd.grad(surrogate, theta, retain_graph=True)[0], exact)


def check_importance():
    old = torch.tensor([.25, .75], dtype=DTYPE)
    current = torch.tensor([.6, .4], dtype=DTYPE)
    f = torch.tensor([4., -1.], dtype=DTYPE)
    ratio = current/old
    close(ratio, [2.4, .4/.75])
    close(old @ f, .25)
    close(current @ f, 2.)
    close(old @ (ratio*f), current @ f)
    close(old @ ratio, 1.)
    # Reference log-ratios estimate current-policy KL under current sampling.
    reference = torch.full_like(current, .5)
    reference_logratio = (current/reference).log()
    current_kl = current @ reference_logratio
    assert current_kl > 0 and old @ reference_logratio < 0
    close(old @ (ratio*reference_logratio), current_kl)
    theta = torch.tensor(math.log(.6/.4), dtype=DTYPE, requires_grad=True)
    p = torch.stack((theta.sigmoid(), 1-theta.sigmoid()))
    direct = torch.autograd.grad(p @ f, theta, retain_graph=True)[0]
    # The old distribution weights an exact enumeration of all sampled actions.
    surrogate = -(old * (p/old) * f).sum()
    close(-torch.autograd.grad(surrogate, theta)[0], direct)
    close(direct, 1.2)
    # Exact conditional token ratios multiply into the whole-sequence ratio.
    new_token = torch.tensor([.6, .7], dtype=DTYPE)
    old_token = torch.tensor([.3, .5], dtype=DTYPE)
    per_token = new_token/old_token
    close(per_token, [2., 1.4])
    close(per_token.prod(), new_token.prod()/old_token.prod())
    close(per_token.prod(), 2.8)
    close(per_token.mean(), 1.7)
    # A missing old action cannot be recovered from observed actions' ratios.
    missing_old = torch.tensor([1., 0.], dtype=DTYPE)
    target = torch.tensor([.5, .5], dtype=DTYPE)
    assert ((missing_old == 0) & (target > 0)).any()


def check_clipping():
    ratios = torch.tensor([.6, 1.5, .6, 1.5], dtype=DTYPE)
    advantages = torch.tensor([2., 2., -2., -2.], dtype=DTYPE, requires_grad=True)
    old = torch.full((4,), math.log(.2), dtype=DTYPE, requires_grad=True)
    new = (old.detach() + ratios.log()).requires_grad_()
    expected_terms = torch.tensor([1.2, 2.4, -1.6, -3.], dtype=DTYPE)
    loss = clipped_policy_loss(new, old, advantages)
    close(loss, -expected_terms.mean())
    loss.backward()
    close(new.grad, [-.3, 0., 0., .75])
    assert old.grad is None and advantages.grad is None
    for r in [.2, .6, .9, 1., 1.1, 1.5, 2.]:
        for a in [-3., -1., 0., 1., 3.]:
            for epsilon in [.1, .2, .4]:
                if abs(r-(1-epsilon)) < 1e-8 or abs(r-(1+epsilon)) < 1e-8:
                    continue  # Treat kinks separately; ordinary derivatives differ.
                value = torch.tensor([math.log(.1*r)], dtype=DTYPE, requires_grad=True)
                fixed = torch.tensor([math.log(.1)], dtype=DTYPE)
                adv = torch.tensor([a], dtype=DTYPE)
                loss = clipped_policy_loss(value, fixed, adv, epsilon)
                flat = (a>0 and r>1+epsilon) or (a<0 and r<1-epsilon) or a==0
                expected = a*min(r,1+epsilon) if a>0 else a*max(r,1-epsilon)
                close(loss, -expected)
                close(torch.autograd.grad(loss, value)[0], [0. if flat else -r*a])
                h=1e-6
                fd=(clipped_policy_loss(value.detach()+h,fixed,adv,epsilon)-
                    clipped_policy_loss(value.detach()-h,fixed,adv,epsilon))/(2*h)
                close(fd, 0. if flat else -r*a, atol=1e-7)
    for a,r in [(2.,1.2),(-2.,.8),(2.,1.),(-2.,1.)]:
        epsilon = .2 if r != 1 else 0.
        new = torch.tensor([math.log(.1*r)],dtype=DTYPE,requires_grad=True)
        loss = clipped_policy_loss(new,torch.tensor([math.log(.1)],dtype=DTYPE),
                                   torch.tensor([a],dtype=DTYPE),epsilon)
        assert torch.isfinite(torch.autograd.grad(loss,new)[0]).all()
    # Deliberate mutation: use max instead of min inside the maximized surrogate.
    r = torch.tensor([1.5], dtype=DTYPE, requires_grad=True)
    wrong = -torch.maximum(r*2, r.clamp(.8,1.2)*2).mean()
    close(wrong, -3.)
    close(torch.autograd.grad(wrong,r)[0], [-2.])
    # Deliberate mutation: clamp alone flattens a harmful probability decrease.
    r = torch.tensor([.6], dtype=DTYPE, requires_grad=True)
    wrong = -(r.clamp(.8,1.2)*2).mean()
    close(torch.autograd.grad(wrong,r)[0], [0.])
    for eps in [-.1, 1.]:
        try:
            clipped_policy_loss(new, new.detach(), torch.ones_like(new), eps)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid epsilon accepted')
    try:
        clipped_policy_loss(torch.tensor([0.]), torch.tensor([-1000.]), torch.tensor([1.]))
    except ValueError:
        pass
    else:
        raise AssertionError('overflowing ratio silently accepted')


def check_frozen_batch_update():
    """An actual plain-SGD step, then re-scoring with a frozen denominator."""
    theta = torch.nn.Parameter(torch.tensor(0., dtype=DTYPE))
    optimizer = torch.optim.SGD([theta], lr=1., momentum=0.)
    actions = torch.tensor([0, 0, 1, 1])
    rewards = torch.tensor([2., 2., 0., 0.], dtype=DTYPE)
    advantages = rewards - 1.  # Known, action-independent baseline fixed beforehand.

    def selected_logp():
        probs = torch.stack((theta.sigmoid(), 1-theta.sigmoid()))
        return probs.log()[actions]

    old_logp = selected_logp().detach().clone()
    saved_old = old_logp.clone()
    initial = clipped_policy_loss(selected_logp(), old_logp, advantages)
    close(initial, 0.)
    initial.backward()
    close(theta.grad, -.5)
    optimizer.step()
    close(theta, .5)
    close(theta.sigmoid(), .6224593312018546)
    optimizer.zero_grad(set_to_none=True)
    new_logp = selected_logp()
    ratios = (new_logp-old_logp).exp()
    close(ratios, [1.2449186624037092]*2 + [.7550813375962908]*2)
    contributions = torch.minimum(ratios*advantages,
                                  ratios.clamp(.8,1.2)*advantages)
    close(contributions, [1.2,1.2,-.8,-.8])
    reused = clipped_policy_loss(new_logp, old_logp, advantages)
    close(reused, -.2)
    reused.backward()
    close(theta.grad, 0.)
    optimizer.step()
    close(theta, .5)
    assert torch.equal(old_logp, saved_old)
    # Deliberately rewriting old probabilities resets ratios without new sampling.
    optimizer.zero_grad(set_to_none=True)
    new_logp = selected_logp()
    wrong = clipped_policy_loss(new_logp, new_logp.detach(), advantages)
    close(wrong, 0.)
    wrong.backward()
    close(theta.grad, -.5)  # Spurious renewed incentive on the same stale samples.


def check_practice():
    close(torch.tensor([1.,4.,7.],dtype=DTYPE).mean(),4.)
    old=torch.tensor([.4,.6],dtype=DTYPE)
    new=torch.tensor([.7,.3],dtype=DTYPE)
    f=torch.tensor([3.,-1.],dtype=DTYPE)
    close(old@f,.6); close(new@f,1.8)
    close(new/old,[1.75,.5]);close(old@((new/old)*f),1.8)
    r=torch.tensor([1.4,.6,.6,1.4],dtype=DTYPE)
    a=torch.tensor([3.,3.,-3.,-3.],dtype=DTYPE)
    terms=torch.minimum(r*a,r.clamp(.8,1.2)*a)
    close(terms,[3.6,1.8,-2.4,-4.2]);close(terms.mean(),-.3)
    logp=(torch.full((4,),math.log(.2),dtype=DTYPE)+r.log()).requires_grad_()
    loss=clipped_policy_loss(logp,torch.full((4,),math.log(.2),dtype=DTYPE),a)
    close(loss,.3);close(torch.autograd.grad(loss,logp)[0],[0.,-.45,0.,1.05])
    close(min(1.6*(-4),1.2*(-4)),-6.4)
    close(.15/.1,1.5);close(math.log(.15/.05),math.log(3))
    p=.4;b=.8
    close(p*(2-b)*(1-p)+(1-p)*(0-b)*(-p),.48)


# Byte-verbatim from CleanRL cleanrl/ppo.py, lines 264–267, revision
# fe8d8a03c41a7ef5b523e2e354bd01c363e786bb. Copyright (c) 2019 CleanRL
# developers. MIT license; see CLEANRL-LICENSE.txt beside this file.
UPSTREAM_POLICY_FRAGMENT = '                # Policy loss\n                pg_loss1 = -mb_advantages * ratio\n                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)\n                pg_loss = torch.max(pg_loss1, pg_loss2).mean()\n'


def check_upstream_fragment():
    for mutation, expected, gradient in [(False,-2.4,0.),(True,-3.,-3.)]:
        new = torch.tensor([math.log(.3)],dtype=DTYPE,requires_grad=True)
        ratio=(new-math.log(.2)).exp()
        fragment=UPSTREAM_POLICY_FRAGMENT
        if mutation:
            fragment=fragment.replace('torch.max(', 'torch.min(')
        ns={'torch':torch,'ratio':ratio,'mb_advantages':torch.tensor([2.],dtype=DTYPE),
            'args':SimpleNamespace(clip_coef=.2)}
        exec(textwrap.dedent(fragment),ns)
        close(ns['pg_loss'],expected)
        close(torch.autograd.grad(ns['pg_loss'],new)[0],[gradient])
    # A reported outside-range ratio is not necessarily in a flat objective region.
    r=torch.tensor([.6],dtype=DTYPE,requires_grad=True)
    assert ((r-1).abs()>.2).item()
    correct=-torch.minimum(r*2,r.clamp(.8,1.2)*2).mean()
    close(torch.autograd.grad(correct,r)[0],[-2.])


def main():
    for check in [check_verifier_coverage,check_advantages,check_importance,
                  check_clipping,check_frozen_batch_update,check_practice,check_upstream_fragment]:
        check(); print('PASS',check.__name__)
    print('All lecture 20 checks passed on PyTorch',torch.__version__,'(CPU).')


if __name__ == '__main__':
    main()
