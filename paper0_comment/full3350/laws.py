"""Complete-support positive-law scores with shared empirical support.
Existing mathematical scoring identities, not a new scoring rule. W1 is
integrated in physical speed intervals; inverse CDF needed only at crossings.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.special import gammaln,gammainc,gammaincc,gammaincinv,ndtr,ndtri

@dataclass(frozen=True)
class Law:
    family:str
    mu:np.ndarray
    sd:np.ndarray
    p:np.ndarray
    q:np.ndarray
    def subset(self,sl):return Law(self.family,self.mu[sl],self.sd[sl],self.p[sl],self.q[sl])

def make(family,mu,sd,weibull_parameters):
    m=np.asarray(mu,dtype=np.float64);s=np.asarray(sd,dtype=np.float64)
    if m.shape!=s.shape or m.ndim!=1 or not np.isfinite(m).all() or not np.isfinite(s).all() or np.any(m<=0) or np.any(s<=0):raise ValueError('positive finite moment vectors')
    if family=='weibull':p,q=weibull_parameters(m,s)
    elif family=='gamma':
        p=(m/s)**2;q=s*s/m
        if np.any(p<1e-4) or np.any(p>1e6):raise ValueError('Gamma shape outside predeclared domain')
    elif family=='lognormal':
        p=np.sqrt(np.log1p((s/m)**2));q=np.log(m)-p*p/2
        if np.any(p<=0) or np.any(p>8):raise ValueError('lognormal log-SD outside domain')
    else:raise ValueError(family)
    return Law(family,m,s,p,q)

def cdf_partial(law,y):
    """CDF and truncated first raw moment at each nonnegative speed."""
    p=law.p[:,None];q=law.q[:,None];m=law.mu[:,None]
    with np.errstate(divide='ignore',over='ignore',under='ignore'):
        if law.family=='weibull':
            z=np.exp(p*(np.log(y)-q));f=-np.expm1(-z);h=m*gammainc(1+1/p,z)
        elif law.family=='gamma':
            z=y/q;f=gammainc(p,z);h=m*gammainc(p+1,z)
        else:
            z=(np.log(y)-q)/p;f=ndtr(z);h=m*ndtr(z-p)
    return f,h

def partial_at_prob(law,ii,prob):
    p=law.p[ii];q=law.q[ii];m=law.mu[ii]
    if law.family=='weibull':return m*gammainc(1+1/p,-np.log1p(-prob))
    if law.family=='gamma':return m*gammainc(p+1,gammaincinv(p,prob))
    return m*ndtr(ndtri(prob)-p)

def summary(law,threshold):
    p,q,m,s=law.p,law.q,law.mu,law.sd
    if law.family=='weibull':
        r3=np.exp(3*q+gammaln(1+3/p));q95=np.exp(q+math.log(-math.log(.05))/p);q99=np.exp(q+math.log(-math.log(.01))/p)
        with np.errstate(over='ignore',under='ignore'):ex=np.exp(-np.exp(p*(math.log(threshold)-q)))
        half=m*(1-np.exp2(-1/p))
    elif law.family=='gamma':
        r3=m**3+3*m*s*s+2*s**4/m;q95=q*gammaincinv(p,.95);q99=q*gammaincinv(p,.99);ex=gammaincc(p,threshold/q)
        half=np.exp(np.log(q)+gammaln(p+.5)-gammaln(p)-.5*math.log(math.pi))
    else:
        r3=m**3*np.exp(3*p*p);q95=np.exp(q+p*ndtri(.95));q99=np.exp(q+p*ndtri(.99));ex=ndtr((q-math.log(threshold))/p)
        half=m*(2*ndtr(p/math.sqrt(2))-1)
    return dict(mu=m,S=s,param1=p,param2=q,R3=r3,q95=q95,q99=q99,exceedance=ex,half_gini=half)

def empirical(raw,reference_mu=None,reference_sd=None,threshold=6.348013401031494):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=729 or not np.isfinite(raw).all() or (raw<0).any():raise ValueError('Full729 nonnegative original target')
    y=np.sort(raw,axis=1);n=729
    return dict(y=y,raw_mu=raw.mean(1),raw_S=raw.std(1,ddof=0),mu=raw.mean(1) if reference_mu is None else reference_mu,S=raw.std(1,ddof=0) if reference_sd is None else reference_sd,R3=(raw**3).mean(1),q95=y[:,692],q99=y[:,721],exceedance=(raw>threshold).mean(1),selfy=(y*(2*np.arange(1,n+1)-n-1)).sum(1)/(n*n))

def score(law,ref,threshold=6.348013401031494):
    y=ref['y'];n=y.shape[1];f,h=cdf_partial(law,y);ss=summary(law,threshold)
    expected_abs=y*(2*f-1)+law.mu[:,None]-2*h
    crps=expected_abs.mean(1)-ss.pop('half_gini');D=crps-ref['selfy']
    # Integral of F on [0,y] is A(y)=yF(y)-E[X 1(X<=y)].
    ant=y*f-h;left=np.concatenate((np.zeros((len(y),1)),y[:,:-1]),axis=1)
    al=np.concatenate((np.zeros((len(y),1)),ant[:,:-1]),axis=1)
    fl=np.concatenate((np.zeros((len(y),1)),f[:,:-1]),axis=1);prob=np.arange(n)/n
    integ=ant-al-prob*(y-left);w=np.abs(integ)
    cross=(fl<prob)&(f>prob)&(y>left)
    ii,jj=np.nonzero(cross)
    if len(ii):
        # At F(q)=prob, qF(q)-prob*q cancels; only first partial moment remains.
        mp=partial_at_prob(law,ii,prob[jj])
        w[ii,jj]=integ[ii,jj]+2*(mp+al[ii,jj]-prob[jj]*left[ii,jj])
    tail=law.mu-h[:,-1]-y[:,-1]*(1-f[:,-1]);W=w.sum(1)+tail
    ss.update(CDF_D=D,expected_CRPS=crps,W1=W,reference_self=ref['selfy'],S_squared_error=(law.sd-ref['S'])**2)
    for name in ('mu','S','q95','q99','R3','exceedance'):ss[name+'_error']=ss[name]-ref[name]
    if min(float(D.min()),float(W.min()))<-2e-8:raise ArithmeticError('Negative score beyond fixed tolerance')
    if not all(np.isfinite(v).all() for v in ss.values()):raise ArithmeticError('Nonfinite final score')
    return ss
