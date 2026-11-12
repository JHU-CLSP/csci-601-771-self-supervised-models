"""CPU arithmetic checks for handout #21. Python standard library only."""
import math

def group_advantages(rewards, delta=1e-8):
    if len(rewards) < 2 or delta <= 0:
        raise ValueError("need a group of >=2 and positive delta")
    if not math.isfinite(delta) or not all(map(math.isfinite, rewards)):
        raise ValueError("inputs must be finite")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean)**2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + delta) for r in rewards]


def term(r, a, low=.2, high=.28):
    return min(r*a, max(1-low, min(r, 1+high))*a)


def close(x, y):
    assert math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-7), (x, y)


def run():
    for rs, expected in [([1,1,0,0], [1,1,-1,-1]),
                         ([1,0,0,0], [math.sqrt(3)]+[-1/math.sqrt(3)]*3),
                         ([1,1,1,0], [1/math.sqrt(3)]*3+[-math.sqrt(3)]),
                         ([1]*4, [0]*4), ([0]*4, [0]*4)]:
        for a,b in zip(group_advantages(rs), expected): close(a,b)
    for r,a,v in [(1.2,3,3.6),(1.4,3,3.84),(.7,-3,-2.4),(1.1,-3,-3.3)]:
        close(term(r,a),v)
    # Nine interactive group states, verified against binary closed forms.
    for k in range(9):
        aa=group_advantages([1]*k+[0]*(8-k))
        expected=([math.sqrt((8-k)/k)]*k+[-math.sqrt(k/(8-k))]*(8-k)) if 0<k<8 else [0]*8
        for a,b in zip(aa,expected): close(a,b)
    for bad in [[], [1], [1,float('nan')], [0,float('inf')]]:
        try: group_advantages(bad)
        except ValueError: pass
        else: raise AssertionError(bad)
    for d in [0,-1,float('nan'),float('inf')]:
        try: group_advantages([0,1],d)
        except ValueError: pass
        else: raise AssertionError(d)
    # Values and independent finite-difference slopes, away from kinks.
    for r,a,v,slope in [(1.25,2,2.5,2),(1.30,2,2.56,0),(.75,-2,-1.6,0),(1.25,-2,-2.5,-2)]:
        close(term(r,a),v)
        close((term(r+1e-6,a)-term(r-1e-6,a))/2e-6,slope)
    rows=[(1,[1.3,1.1]),(1,[.9,1]),(-1,[.7,.9]),(-1,[1.1,.8])]
    close(sum(sum(term(r,a,high=.2) for r in rr)/2 for a,rr in rows)/4,.075)
    # Length reductions; quiz and in-section fixtures differ.
    for short,long,expected in [(1,-1,(0,-.6,-.375)),(2,-1,(.5,-.4,-.25))]:
        got=((short+long)/2,(2*short+8*long)/10,(2*short+8*long)/16)
        for a,b in zip(got,expected): close(a,b)
    close(1-.5**4-.5**4,.875)
    close(1-.99**4-.01**4,.03940398)
    close((.12/.10)*(.15/.12),1.5)
    close((.12/.10)*term(.15/.12,1,high=.2),1.44)
    close(1/6/(1/8),4/3)
    # Conditional KL estimator: expected values agree under p, not old.
    p,q,old=[.7,.3],[.5,.5],[.4,.6]
    kk=[qv/pv-math.log(qv/pv)-1 for pv,qv in zip(p,q)]
    exact=sum(pv*math.log(pv/qv) for pv,qv in zip(p,q))
    close(sum(pv*k for pv,k in zip(p,kk)),exact)
    close(sum(ov*(pv/ov)*k for ov,pv,k in zip(old,p,kk)),exact)
    assert not math.isclose(sum(ov*k for ov,k in zip(old,kk)),exact)
    assert all(k>=0 for k in kk)
    # ESS extremes and branch-only mutation.
    ess=lambda w:sum(w)**2/sum(v*v for v in w)
    close(ess([1]*4),4)
    close(ess([7,1,1,1]),25/13)
    close(ess([1,0,0,0]),1)
    close(term(.6,2,high=.2),1.2)
    assert max(.8,min(.6,1.2))*2 != term(.6,2,high=.2)
    print("PASS: group statistics; 9 widget states; invalid inputs; clipping values/slopes;")
    print("worked batch; reductions; mixed-group rate; stale correction; throughput; KL; ESS; mutation")


if __name__ == '__main__':
    run()
