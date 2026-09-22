"""Small Gaussian rules for a known truncated exponential measure.

Cheap scalar quadrature constructs the measure; expensive basis functions are
evaluated only at the returned Gaussian nodes. Full reorthogonalized Lanczos
avoids a monomial Hankel solve. No physical tail or part of the domain is cut off.
"""
from functools import lru_cache
import numpy as np
import scipy.linalg as la
from scipy.special import roots_legendre


@lru_cache(None)
def _seed_rule(order=24):
    x,w=roots_legendre(order)
    return (x+1)/2,w/2


def truncated_exponential_gauss(upper,order=12):
    """Positive nodes/weights for integral_0^upper exp(-z)*f(z) dz."""
    if not np.isfinite(upper) or upper<=0 or not isinstance(order,int) or order<1:
        raise ValueError('Require positive finite upper and positive integer order')
    x,w=_seed_rule(max(24,order+4))
    cuts=[0.]+[a for a in (1.,2.,4.,8.,16.,32.,64.,128.,256.) if a<upper]+[upper]
    z=np.concatenate([lo+(hi-lo)*x for lo,hi in zip(cuts[:-1],cuts[1:])])
    seed=np.concatenate([(hi-lo)*w for lo,hi in zip(cuts[:-1],cuts[1:])])*np.exp(-z)
    active=seed>0;z=z[active];seed=seed[active]
    mass=-np.expm1(-upper)
    # Scale small intervals to avoid absolute small-number thresholds.
    scale=min(1.,upper);z=z/scale
    v=np.sqrt(seed/seed.sum());vectors=[];diagonal=[];off=[]
    previous=np.zeros_like(v);beta=0.
    for k in range(order):
        vectors.append(v)
        candidate=z*v-beta*previous
        alpha=float(v@candidate);diagonal.append(alpha)
        candidate-=alpha*v
        # Twice-reorthogonalized recurrence remains reliable at small orders.
        Q=np.column_stack(vectors)
        for _ in range(2):candidate-=Q@(Q.T@candidate)
        if k+1<order:
            beta=float(np.linalg.norm(candidate))
            if not np.isfinite(beta) or beta<=0:raise ArithmeticError('Degenerate exponential quadrature')
            off.append(beta);previous,v=v,candidate/beta
    nodes,eigenvectors=la.eigh_tridiagonal(diagonal,off)
    weights=mass*eigenvectors[0]**2
    nodes=nodes*scale
    if not (np.all(nodes>0) and np.all(nodes<upper) and np.all(weights>0)):
        raise ArithmeticError('Invalid exponential Gaussian rule')
    return nodes,weights


def exponential_gauss(length,upper,order=12):
    """Positive rule for integral_0^upper exp(-n/length)*f(n) dn."""
    if not np.isfinite(length) or length<=0:raise ValueError('length must be positive')
    z,w=truncated_exponential_gauss(upper/length,order)
    return length*z,length*w
