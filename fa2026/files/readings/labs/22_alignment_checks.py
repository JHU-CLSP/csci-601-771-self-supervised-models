"""Handout #22: standard-library arithmetic and evaluator-coverage checks."""
import math
from itertools import product

def false_selected(p0, n):
    if not math.isfinite(p0) or not 0 <= p0 <= 1:
        raise ValueError("p0 must be a probability")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return 1 - (1 - p0)**n


def kl_optimum(p0, beta):
    if not 0 < p0 < 1 or not math.isfinite(beta) or beta <= 0:
        raise ValueError("need interior reference and finite positive beta")
    z = math.log(p0) - math.log1p(-p0) + 1/beta
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def bernoulli_kl(p, q):
    return sum(a*math.log(a/b) if a else 0 for a,b in [(p,q),(1-p,1-q)])


def close(a,b,tol=1e-10):
    assert math.isclose(a,b,abs_tol=tol,rel_tol=tol),(a,b)


def run():
    # Enumerate all candidate pools, independently of the closed-form function.
    for p0 in [.01,.1,.2,.49]:
        for n in range(1,9):
            total=0
            for pool in product([0,1],repeat=n):
                prob=math.prod(p0 if is_b else 1-p0 for is_b in pool)
                total+=prob*int(any(pool))
            close(false_selected(p0,n),total)
    for n in [1,4,16,64]:
        assert false_selected(.1,n) >= .1-1e-12
    close(false_selected(.1,4),.3439)
    close(false_selected(.2,3),.488)
    close(.1**4,.0001)  # repaired ranking: B only if all candidates are B
    for p0,n in [(-.1,2),(1.1,2),(float('nan'),2),(.1,0),(.1,2.5),(.1,True)]:
        try: false_selected(p0,n)
        except ValueError: pass
        else: raise AssertionError((p0,n))
    # Independent first derivative and grid search checks of the strict optimum.
    for p0 in [.1,.2,.49]:
        for beta in [.25,1,5]:
            p=kl_optimum(p0,beta)
            deriv=1-beta*(math.log(p/(1-p))-math.log(p0/(1-p0)))
            close(deriv,0,tol=1e-9)
            objective=lambda t: 1+t-beta*bernoulli_kl(t,p0)
            for i in range(1001): assert objective(p)>=objective(i/1000)-1e-12
            assert p>p0
    close(kl_optimum(.1,1),.23196931668407395)
    close(bernoulli_kl(.1,.1),0)
    for p in [0,.1,.5,.9,1]: assert bernoulli_kl(p,.1)>=0
    # Aggregate gains can conceal a regression on a fixed subset.
    close((162+16)/200,.89)
    close((171+8)/200,.895)
    close(16/20,.8)
    close(8/20,.4)
    close(.99**200,.13397967485796172)
    # Costs, score weighting, expected harm: quiz and in-section fixtures.
    assert 200*16*256==819200 and 200*16==3200
    assert 100*8*200==160000 and 100*8==800
    assert 9+4<6+9 and 3*9+4>3*6+9
    close(.1*5,.5)
    buggy=lambda a,b:abs(max(a,b))
    visible=[(1,2),(5,3),(4,9)]
    assert all(buggy(a,b)==max(a,b) for a,b in visible)
    assert buggy(-5,-2)!=max(-5,-2)
    print("PASS: 32 enumerated candidate-pool cases; best-of-N limits and fixtures;")
    print("invalid inputs; KL optimum/derivative/grid; cost and score arithmetic; verifier counterexample")


if __name__ == '__main__':
    run()
