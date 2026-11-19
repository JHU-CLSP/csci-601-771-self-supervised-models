"""Handout 23: deterministic core checks; optional bonus checks via --bonus.
Python standard library only. No trained model, GPU, network, or random data.
Adjacent-pair convention, with fixed positional frequencies during cached use.
"""
import argparse
import math

def rope(x, position, base=10000.0, scale=1.0):
    if not x or len(x) % 2 or base <= 1 or scale <= 0:
        raise ValueError("need nonempty even width, base>1, scale>0")
    if not all(map(math.isfinite, [*x, position, base, scale])):
        raise ValueError("inputs must be finite")
    d, out = len(x), []
    for i in range(d // 2):
        angle = (position / scale) * base**(-2*i/d)
        c, s = math.cos(angle), math.sin(angle)
        a, b = x[2*i:2*i+2]
        out.extend((a*c-b*s, a*s+b*c))
    return out


def dot(x,y):
    return sum(a*b for a,b in zip(x,y))


def matvec(matrix,x):
    return [dot(row,x) for row in matrix]


def dense_rotation(d,position,base=10000.,scale=1.):
    # Independent explicit block construction for comparison with the O(d) helper.
    result=[[0.]*d for _ in range(d)]
    for j in range(0,d,2):
        phi=position*base**(-j/d)/scale
        c,s=math.cos(phi),math.sin(phi)
        result[j][j],result[j][j+1]=c,-s
        result[j+1][j],result[j+1][j+1]=s,c
    return result


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def vector_close(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):close(x,y)


def core_checks():
    # Same supplied NLLs, different target masks: never count BOS/padding.
    for answer in [.1,.2,3.]:
        losses=[2.,3.,1.,answer,.3]
        close(sum(losses)/len(losses),(6.3+answer)/5)
        close(sum(losses[-2:])/2,(answer+.3)/2)
    # Response-only loss still differentiates through a prompt representation u.
    response_loss=lambda u:math.log1p(math.exp(-u))
    h=1e-5
    close((response_loss(h)-response_loss(-h))/(2*h),-.5)
    vectors=[[1.,2.],[-.5,2.,1.,-3.],[1.,-2.,3.,4.,-.5,2.,1.,-1.]]
    for x in vectors:
        for m in [0,1,5,17,64]:
            for scale in [1.,4.]:
                actual=rope(x,m,base=100.,scale=scale)
                vector_close(actual,matvec(dense_rotation(len(x),m,100.,scale),x))
                close(dot(actual,actual),dot(x,x))
    q=[1.,2.,-1.,.5];k=[-2.,1.,3.,1.]
    for m,n in [(5,2),(17,4),(4,4)]:
        actual=dot(rope(q,m,100.),rope(k,n,100.))
        close(actual,dot(q,matvec(dense_rotation(4,n-m,100.),k)))
        for shift in [0,7,19]:
            close(actual,dot(rope(q,m+shift,100.),rope(k,n+shift,100.)))
    vector_close(rope([1.,0.],3,100.),[math.cos(3),math.sin(3)])
    simple=[1.,0.,1.,0.]
    close(dot(rope(simple,5,100.),rope(simple,2,100.)),math.cos(3)+math.cos(.3))
    close(dot(rope(simple,5,100.,4.),rope(simple,2,100.,4.)),math.cos(.75)+math.cos(.075))
    # Query/key offsets stay valid during cached scoring: compare full recomputation.
    contents=[[1.,0.,.5,1.],[0.,1.,-1.,2.],[1.,1.,0.,.5],[2.,-.5,1.,0.]]
    cache=[rope(v,n,100.) for n,v in enumerate(contents)]
    query=[.25,1.,-1.,2.];m=3
    cached_scores=[dot(rope(query,m,100.),key)/2 for key in cache]
    full_scores=[dot(matvec(dense_rotation(4,m,100.),query),
                     matvec(dense_rotation(4,n,100.),key))/2
                 for n,key in enumerate(contents)]
    vector_close(cached_scores,full_scores)
    original=[1.,2.,3.,4.]
    rerotated=rope(rope(original,10,100.),11,100.)
    vector_close(rerotated,rope(original,21,100.))
    assert any(abs(a-b)>1e-6 for a,b in zip(rerotated,rope(original,10,100.)))
    # Independent closed forms for core frequencies and costs.
    vector_close([100.**(-2*i/4) for i in range(2)],[1.,.1])
    vector_close([100.**(-2*i/4)/4 for i in range(2)],[.25,.025])
    vector_close([10000.**(-2*i/4) for i in range(2)],[1.,.01])
    assert 12000/4==3000
    assert 4**2==16 and (4/16)*4**2==4
    assert 65536//4096==16 and 65536//16384==4
    cache_bytes=lambda n:2*32*n*8*128*2
    assert cache_bytes(4096)==2**29 and cache_bytes(16384)==2**31
    close(.6*2+.4*1,1.6)
    close(math.exp(1.6),4.953032424395115)
    close(.6*math.exp(2)+.4*math.exp(1),5.520746390742007)
    close((.95+.4+.9)/3,.75)
    assert 20*.6==12 and 20*(1-.6)==8 and 8/2==4
    for x,pos,base,scale in [([],1,100,1),([1],1,100,1),([1,2],1,1,1),
                             ([1,2],1,100,0),([1,float('nan')],1,100,1),
                             ([1,2],float('inf'),100,1)]:
        try:rope(x,pos,base,scale)
        except ValueError:pass
        else:raise AssertionError((x,pos,base,scale))
    # A wrong second coordinate breaks norm preservation; zero alone cannot catch it.
    a,b,phi=1.,2.,.7;c,s=math.cos(phi),math.sin(phi)
    mutated=[a*c-b*s,a*s+b*s]
    assert not math.isclose(dot(mutated,mutated),a*a+b*b)
    print('PASS core: dense/pairwise parity; norms; relative offsets; shifts; PI; cache; costs; mixtures; evaluation; invalid inputs; mutation')


def bonus_checks():
    # Only run on explicit request; none are prerequisites for the core exercises.
    wavelength=2*math.pi/(100**(-2/4))
    close(wavelength,20*math.pi)
    assert abs(wavelength-2*math.pi*100)>500
    for d in [8,16,64,128]:
        for factor in [2,4,8]:
            for i in range(d//2):
                old=2*math.pi*10000**(2*i/d)
                pi=old*factor
                changed=2*math.pi*(10000*factor)**(2*i/d)
                close(pi/old,factor)
                close(changed/old,factor**(2*i/d))
            close((10000*factor)**0,1.)
    close(20/1000,.02)
    close(20/(1000+20),1/51)
    assert (3-8)==(14-19)
    print('PASS optional bonus: wavelength correction; stage-budget denominators; relative bias offset')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bonus',action='store_true',help='also run explicitly optional bonus checks')
    args=parser.parse_args()
    core_checks()
    if args.bonus:bonus_checks()
