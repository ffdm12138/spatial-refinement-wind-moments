"""Experimental frozen exchange predictor on P08-prepared coarse inputs.
No reference winds, labels, case IDs, training, or filesystem access in predict().
The caller supplies Phi/SPATIAL2 transformed with model['preprocessing'].
This is NOT a raw-WRF ingest pipeline or a certified unseen-weather model.
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln,expit

def _moments(k: np.ndarray, logscale: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    m=np.exp(logscale+gammaln(1+1/k));s=m*np.sqrt(np.expm1(gammaln(1+2/k)-2*gammaln(1+1/k)))
    return m,s

def _historic_law(mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    # Reproduce the actual P08 law, including its inherited CV interpolation.
    ks=np.geomspace(.15,1000.,15000)
    cv=np.sqrt(np.expm1(gammaln(1+2/ks)-2*gammaln(1+1/ks)))
    k=np.interp(sd/np.maximum(mu,1e-12),cv[::-1],ks[::-1])
    ls=np.log(np.maximum(mu/np.exp(gammaln(1+1/k)),1e-10))
    return _moments(k,ls)

def _final_law(mu: np.ndarray, sd: np.ndarray) -> dict[str,np.ndarray]:
    if np.any(mu<=0) or np.any(sd<=0) or not np.isfinite(mu).all() or not np.isfinite(sd).all():raise ValueError('Positive finite requested moments required')
    target=np.log1p((sd/mu)**2)
    g=lambda k:gammaln(1+2/k)-2*gammaln(1+1/k)
    if np.any(target>g(.1)) or np.any(target<g(256.)):raise ValueError('Requested shape outside qualified domain; no clipping')
    lo=np.full(mu.shape,np.log(.1));hi=np.full(mu.shape,np.log(256.))
    for _ in range(46):
        mid=(lo+hi)/2;mask=g(np.exp(mid))>target;lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
    k=np.exp((lo+hi)/2);ls=np.log(mu)-gammaln(1+1/k);m,s=_moments(k,ls)
    if not np.allclose(m,mu,rtol=2e-8,atol=1e-10) or not np.allclose(s,sd,rtol=2e-8,atol=1e-10):raise ArithmeticError('Moment inversion parity')
    out=dict(mu=m,S=s,k=k,logscale=ls,R3=np.exp(3*ls+gammaln(1+3/k)))
    for p,n in ((.95,'q95'),(.99,'q99')):out[n]=np.exp(ls+np.log(-np.log1p(-p))/k)
    return out

def predict(model: dict, Phi: np.ndarray, SPATIAL2: np.ndarray, native_U: np.ndarray) -> dict[str,np.ndarray]:
    """The same frozen JS-mean/QA-spread law, with unused JS-spread head removed."""
    if model.get('schema')!='npj.p0.experimental_compact_component_predictor/v1':raise ValueError('Compact model schema')
    x=np.asarray(Phi,dtype=np.float64);e=np.asarray(SPATIAL2,dtype=np.float64);U=np.asarray(native_U,dtype=np.float64)
    d=int(model['preprocessing']['base_dimension'])
    if U.ndim!=1 or x.shape!=(len(U),d) or e.shape!=(len(U),2) or not len(U):raise ValueError('Input shape')
    if not all(np.isfinite(a).all() for a in (x,e,U)) or np.any(U<0):raise ValueError('Finite inputs and nonnegative speed required')
    q=model['QA'];j=model['JS_mean'];pre=model['preprocessing']
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        latent_j=(x@np.asarray(j['base_coefficients'])+e@np.asarray(j['extra_coefficients']))*pre['JS']['latent_sd'][0]+pre['JS']['latent_mean'][0]
        mu=U+np.exp(latent_j)
        latent_q=(x@np.asarray(q['base_coefficients'])+e@np.asarray(q['extra_coefficients']))*np.asarray(pre['QA']['latent_sd'])+np.asarray(pre['QA']['latent_mean'])
        Q=np.exp(latent_q[:,0]);a=expit(latent_q[:,1]);J=a*Q/(np.sqrt(U*U+a*Q)+U);S=np.sqrt((1-a)*Q)
        _,S0=_historic_law(U+J,S)
        cal=model['spread_calibration'];z=(np.log1p(U)-cal['normalizer']['Umean'])/cal['normalizer']['Usd']
        sd=S0*np.exp(cal['theta'][0]+cal['theta'][1]*z)
        out=_final_law(mu,sd)
    if np.any(out['mu']<U-1e-10):raise ArithmeticError('Native mean compatibility')
    Qn=out['S']**2+out['mu']**2-U**2
    if np.any(Qn<=0):raise ArithmeticError('Native variance must be positive')
    out.update(native_J=out['mu']-U,Q_native=Qn,a_native=(out['mu']**2-U**2)/Qn)
    if not all(np.isfinite(a).all() for a in out.values()):raise ArithmeticError('Nonfinite prediction')
    return out
