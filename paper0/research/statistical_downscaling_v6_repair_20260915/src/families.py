"""Distribution parameters/CDFs with vectorized all-sample sufficient statistics."""
import numpy as np
from scipy.special import digamma,polygamma,gammaln,gammainc,gammaincinv,ndtr,ndtri,ndtri_exp,log_ndtr,erfcx


def trunc_lookup(cv):
    alpha=np.linspace(-50,35,25001)
    with np.errstate(over='ignore'): hazard=np.sqrt(2/np.pi)/erfcx(alpha/np.sqrt(2))
    mean=hazard-alpha; variance=1+alpha*hazard-hazard*hazard
    ratio=np.sqrt(np.maximum(variance,1e-12))/mean
    # In the exponential boundary limit CV=1; samples with CV>=1 have no interior moment solution.
    values=np.asarray(cv,float); a=np.interp(values,ratio,alpha)
    boundary=values>=ratio[-1]
    return a,boundary


def fit_rows(samples,family,mode='mle'):
    y=np.asarray(samples,float); mu=y.mean(axis=1); var=y.var(axis=1); sigma=np.sqrt(var); safe=np.maximum(y,1e-10); logy=np.log(safe); lmean=logy.mean(axis=1)
    if family=='normal': return {'family':family,'loc':mu,'scale':np.maximum(sigma,1e-8)}
    if family=='gamma':
        if mode=='moments': shape=np.maximum(mu*mu/np.maximum(var,1e-12),.01)
        else:
            gap=np.maximum(np.log(np.maximum(mu,1e-10))-lmean,1e-10)
            shape=(3-gap+np.sqrt((gap-3)**2+24*gap))/(12*gap)
            for _ in range(12):
                f=np.log(shape)-digamma(shape)-gap; d=1/shape-polygamma(1,shape)
                shape=np.maximum(shape-np.clip(f/d,-.5*shape,.5*shape),.01)
        return {'family':family,'shape':shape,'scale':np.maximum(mu/shape,1e-10)}
    if family=='lognormal':
        if mode=='moments': s2=np.log1p(var/np.maximum(mu*mu,1e-20)); loc=np.log(np.maximum(mu,1e-10))-.5*s2; s=np.sqrt(s2)
        else: loc=lmean; s=logy.std(axis=1)
        return {'family':family,'loc':loc,'scale':np.maximum(s,1e-8)}
    if family=='weibull':
        ks=np.geomspace(.15,1000,15000); ratios=np.sqrt(np.expm1(gammaln(1+2/ks)-2*gammaln(1+1/ks)))
        k=np.interp(sigma/np.maximum(mu,1e-10),ratios[::-1],ks[::-1])
        if mode=='mle':
            centered=logy-logy.max(axis=1)[:,None]
            for _ in range(15):
                weights=np.exp(k[:,None]*centered); sw=weights.sum(axis=1); mean=(weights*logy).sum(axis=1)/sw; variance=(weights*(logy-mean[:,None])**2).sum(axis=1)/sw
                score=1/k+lmean-mean; derivative=-1/(k*k)-variance
                k=np.maximum(k-np.clip(score/derivative,-.5*k,.5*k),.15)
            logscale=logy.max(axis=1)+np.log(np.mean(np.exp(k[:,None]*centered),axis=1))/k
            scale=np.exp(logscale)
        else: scale=mu/np.exp(gammaln(1+1/k))
        return {'family':family,'shape':k,'scale':np.maximum(scale,1e-10)}
    if family=='truncnormal':
        alpha,boundary=trunc_lookup(sigma/np.maximum(mu,1e-10))
        with np.errstate(over='ignore'): hazard=np.sqrt(2/np.pi)/erfcx(alpha/np.sqrt(2))
        scale=mu/(hazard-alpha); loc=-alpha*scale
        return {'family':family,'loc':loc,'scale':np.maximum(scale,1e-8),'boundary':boundary,'boundary_mean':mu}
    raise ValueError(family)


def ppf(params,levels):
    p=np.asarray(levels,float); f=params['family']; scale=np.asarray(params['scale'])[:,None]
    if f=='normal': return np.asarray(params['loc'])[:,None]+scale*ndtri(p)
    if f=='gamma': return scale*gammaincinv(np.asarray(params['shape'])[:,None],p)
    if f=='weibull': return scale*np.power(-np.log1p(-p),1/np.asarray(params['shape'])[:,None])
    if f=='lognormal': return np.exp(np.asarray(params['loc'])[:,None]+scale*ndtri(p))
    if f=='truncnormal':
        loc=np.asarray(params['loc'])[:,None]; alpha=-loc/scale
        result=loc-scale*ndtri_exp(log_ndtr(-alpha)+np.log1p(-p))
        if 'boundary' in params: result=np.where(np.asarray(params['boundary'])[:,None],-np.asarray(params['boundary_mean'])[:,None]*np.log1p(-p),result)
        return np.maximum(result,0)
    raise ValueError(f)


def cdf(params,grid):
    x=np.asarray(grid,float); f=params['family']; scale=np.asarray(params['scale'])[:,None]
    if f=='normal': return ndtr((x-np.asarray(params['loc'])[:,None])/scale)
    if f=='gamma': return gammainc(np.asarray(params['shape'])[:,None],np.maximum(x,0)/scale)
    if f=='weibull': return -np.expm1(-np.power(np.maximum(x,0)/scale,np.asarray(params['shape'])[:,None]))
    if f=='lognormal': return np.where(x>0,ndtr((np.log(np.maximum(x,1e-300))-np.asarray(params['loc'])[:,None])/scale),0)
    if f=='truncnormal':
        loc=np.asarray(params['loc'])[:,None]; alpha=-loc/scale; result=-np.expm1(np.minimum(log_ndtr(-(x-loc)/scale)-log_ndtr(-alpha),0)); result=np.where(x>=0,result,0)
        if 'boundary' in params: result=np.where(np.asarray(params['boundary'])[:,None],-np.expm1(-np.maximum(x,0)/np.asarray(params['boundary_mean'])[:,None]),result)
        return result
    raise ValueError(f)


def cross_absolute(y,forecast_nodes):
    y=np.sort(np.asarray(y,float),axis=1); q=np.asarray(forecast_nodes,float); rows,n=y.shape
    # Row offset transforms many sorted search problems into one; offsets separated by full ranges.
    span=max(float(np.max(y)),float(np.max(q)))-min(float(np.min(y)),float(np.min(q)))+1
    offset=np.arange(rows)[:,None]*span
    pos=np.searchsorted((y+offset).ravel(),(q+offset).ravel(),side='right').reshape(q.shape)-np.arange(rows)[:,None]*n
    prefix=np.column_stack([np.zeros(rows),np.cumsum(y,axis=1)])
    return (((2*pos-n)*q+prefix[:,-1,None]-2*np.take_along_axis(prefix,pos,axis=1))/n).mean(axis=1)


def conditional_scores(y,params,k=129):
    levels=(np.arange(k)+.5)/k; q=ppf(params,levels)
    cross=cross_absolute(y,q); halfpair=(q*(2*np.arange(k)-k+1)).sum(axis=1)/(k*k)
    realq=np.quantile(y,levels,axis=1).T
    # W1 is a documented quantile quadrature here; final mixture evaluator is separate.
    tails=ppf(params,np.array([.05,.25,.5,.75,.95])); rt=np.quantile(y,[.05,.25,.5,.75,.95],axis=1).T
    return {'crps':cross-halfpair,'w1_quantile_quadrature':np.abs(realq-q).mean(axis=1),**{f'q{p:02d}_error':tails[:,i]-rt[:,i] for i,p in enumerate([5,25,50,75,95])}}
