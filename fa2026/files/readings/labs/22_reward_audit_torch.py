"""Handout 22: reuse the helpers from #20/#21 to audit a reversed reward.
Requires CPU PyTorch and the adjacent 20_reasoning_policy_checks.py and
21_grpo_checks.py files. No network, model downloads, or random rollouts.
"""
import importlib
import math
import torch

ppo = importlib.import_module("20_reasoning_policy_checks")
grpo = importlib.import_module("21_grpo_checks")
DTYPE = torch.float64


def run_case(rewards, beta=0.0):
    # B is false: p(B)=sigmoid(theta). Start from a balanced reference.
    theta = torch.nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
    optimizer = torch.optim.SGD([theta], lr=1.0, momentum=0.0)
    actions = torch.tensor([0, 0, 1, 1])  # A,A,B,B: one whole-response action each
    old_logp = torch.full((4,), math.log(.5), dtype=DTYPE, requires_grad=True)
    advantages = torch.tensor(grpo.group_advantages(rewards),
                              dtype=DTYPE, requires_grad=True)
    reference = torch.full((2,), .5, dtype=DTYPE)
    trace = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        p_b = theta.sigmoid()
        probs = torch.stack((1-p_b, p_b))
        task_loss = ppo.clipped_policy_loss(probs.log()[actions], old_logp,
                                            advantages, epsilon=.2)
        exact_kl = (probs * (probs.log()-reference.log())).sum()
        loss = task_loss + beta*exact_kl
        loss.backward()
        assert old_logp.grad is None and advantages.grad is None
        gradient = theta.grad.item()
        optimizer.step()
        with torch.no_grad():
            p = theta.sigmoid().item()
            kl = p*math.log(p/.5)+(1-p)*math.log((1-p)/.5)
            trace.append(dict(theta=theta.item(), gradient=gradient, false=p,
                              proxy=1+p, correctness=1-p, kl=kl))
    return trace


def main():
    cases = {
        "Correctness reward": run_case([1, 1, 0, 0]),
        "Flawed proxy": run_case([1, 1, 2, 2]),
        "Flawed proxy + KL": run_case([1, 1, 2, 2], beta=1.0),
    }
    truth, proxy, anchored = cases.values()
    # Hand calculation for step 1: normalized coefficients are +/-1/(1+2*delta).
    step = .5/(1+2e-8)
    expected_false = 1/(1+math.exp(-step))
    assert abs(proxy[0]['false']-expected_false)<1e-12
    assert abs(truth[0]['false']-(1-expected_false))<1e-12
    # Initially current==reference, so exact KL has zero first derivative.
    assert abs(proxy[0]['false']-anchored[0]['false'])<1e-12
    # Both task contributions hit their favorable flat regions after step 1.
    assert abs(proxy[1]['gradient'])<1e-12
    assert abs(truth[1]['gradient'])<1e-12
    # KL alone moves step 2 back toward reference; compare its analytic logit slope.
    expected_second_theta = step-step*expected_false*(1-expected_false)
    assert abs(anchored[1]['theta']-expected_second_theta)<1e-12
    assert .5<anchored[1]['false']<proxy[1]['false']
    print('After two SGD steps on the same frozen batch:')
    print(f'{"Reward rule":24} {"Proxy":>8} {"Correct":>8} {"KL":>8}')
    for label, trace in cases.items():
        m = trace[-1]
        print(f'{label:24} {m["proxy"]:8.4f} {m["correctness"]:8.4f} {m["kl"]:8.4f}')
    print('PASS: update directions; stored-data detachment; clipping plateaus; exact KL gradient')


if __name__ == '__main__':
    main()
