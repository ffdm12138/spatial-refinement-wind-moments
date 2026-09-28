"""Finite instantaneous d02 + terrain models; fit routines receive A-only arrays."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
# Conda's CUDA 12.1 runtime is installed under env/bin, outside Windows' default
# dependent-DLL search. Keep the directory handle alive for NVRTC builtins.
_runtime_bin=Path(sys.prefix)/'bin'
_dll_handle=os.add_dll_directory(str(_runtime_bin)) if os.name=='nt' and _runtime_bin.is_dir() else None
if os.name=='nt' and _runtime_bin.is_dir(): os.environ['PATH']=str(_runtime_bin)+os.pathsep+os.environ.get('PATH','')
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,gammaln,erfcx
from sklearn.preprocessing import SplineTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans
import torch


def design_fit(features,columns,names):
    idx=[columns.index(n) for n in names]; x=np.asarray(features[:,idx],float)
    means=x.mean(axis=0); sd=x.std(axis=0); keep=sd>1e-9
    z=(x[:,keep]-means[keep])/sd[keep]
    # One nonlinear wind-speed basis; static and directional descriptors remain explicit.
    ui=columns.index('log1p_U'); spline=SplineTransformer(n_knots=4,degree=3,include_bias=False,knots='quantile')
    spl=spline.fit_transform(np.asarray(features[:,ui],float).reshape(-1,1)); sm=spl.mean(axis=0); ss=np.maximum(spl.std(axis=0),1e-9)
    matrix=np.column_stack([np.ones(len(x)),z,(spl-sm)/ss])
    return matrix,{'indices':idx,'mean':means,'sd':np.maximum(sd,1e-9),'keep':keep,'spline':spline,'spline_index':ui,'spline_mean':sm,'spline_sd':ss,'feature_names':names}


def design_apply(spec,features):
    x=np.asarray(features[:,spec['indices']],float)
    z=((x-spec['mean'])/spec['sd'])[:,spec['keep']]
    spl=spec['spline'].transform(np.asarray(features[:,spec['spline_index']],float).reshape(-1,1))
    return np.column_stack([np.ones(len(x)),z,(spl-spec['spline_mean'])/spec['spline_sd']])


def inverse_softplus(x):
    x=np.asarray(x,float); return x+np.log(-np.expm1(-np.maximum(x,1e-12)))


def optimize_torch(initial,objective,maxiter=500):
    device='cuda' if torch.cuda.is_available() else 'cpu'; shape=initial.shape
    def fg(value):
        beta=torch.tensor(value.reshape(shape),dtype=torch.float64,device=device,requires_grad=True)
        loss=objective(beta)
        if not bool(torch.isfinite(loss)): return 1e100,np.zeros_like(value)
        loss.backward(); grad=beta.grad.detach().cpu().numpy().ravel()
        if not np.isfinite(grad).all(): return 1e100,np.zeros_like(value)
        return float(loss.detach().cpu()),grad
    result=minimize(fg,initial.ravel(),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-5,'maxls':40})
    value,gradient=fg(result.x)
    accepted=bool(np.isfinite(value) and value<1e90 and (result.success or np.max(np.abs(gradient))<2e-4))
    if not accepted: raise RuntimeError(f'Optimizer failed: {result.message}; gradient={np.max(np.abs(gradient))}')
    return result.x.reshape(shape),{'success':bool(result.success),'accepted_stationary':accepted,'message':str(result.message),'objective':value,'gradient_max':float(np.max(np.abs(gradient))),'iterations':int(result.nit),'device':device}


def fit(spec,features,columns,names,labels,samples):
    x,ds=design_fit(features,columns,names); n,p=x.shape
    route=spec['route']; alpha=float(spec.get('alpha',.01)); device='cuda' if torch.cuda.is_available() else 'cpu'
    model={'spec':spec,'design':ds,'training_rows':n,'parameter_dimension':p,'U_index':columns.index('U')}
    if route=='QA' and spec['engine']=='HGB':
        q=labels.Q.to_numpy(float); a=np.clip(labels.a.to_numpy(float),1e-8,1-1e-8)
        kw=dict(max_leaf_nodes=int(spec['leaves']),max_iter=150,learning_rate=.05,min_samples_leaf=128,l2_regularization=10.,early_stopping=False,random_state=20260915)
        qm=HistGradientBoostingRegressor(**kw).fit(x[:,1:],q)
        am=HistGradientBoostingRegressor(**kw).fit(x[:,1:],a,sample_weight=q/(q+1))
        model.update(q_model=qm,a_model=am,fit_receipt={'engine':'HGB','trees':150,'leaves':int(spec['leaves'])}); return model
    tx=torch.as_tensor(x,dtype=torch.float64,device=device)
    if route=='QA':
        q=labels.Q.to_numpy(float); a=np.clip(labels.a.to_numpy(float),0,1); qs=max(float(q.std()),1e-6)
        tq=torch.as_tensor(q/qs,dtype=torch.float64,device=device); ta=torch.as_tensor(a,dtype=torch.float64,device=device); weight=tq/(tq+1/qs)
        init=np.zeros((p,2)); init[0,0]=inverse_softplus(q.mean()/qs); av=np.clip(np.average(a,weights=q/(q+1)),1e-5,1-1e-5); init[0,1]=np.log(av/(1-av))
        def objective(beta):
            z=tx@beta; qhat=torch.nn.functional.softplus(z[:,0]); aq=torch.nn.functional.binary_cross_entropy_with_logits(z[:,1],ta,reduction='none')
            return ((qhat-tq)**2).mean()+(weight*aq).sum()/weight.sum()+alpha*(beta[1:]**2).sum()
        coef,receipt=optimize_torch(init,objective); model.update(coef=coef,qscale=qs,fit_receipt=receipt)
    elif route=='DIRECT':
        y=np.asarray(samples,float); logy=np.log(np.maximum(y,1e-10)); mu=y.mean(axis=1); logs=logy.mean(axis=1); logsq=(logy*logy).mean(axis=1); second=(y*y).mean(axis=1)
        U=np.asarray(features[:,columns.index('U')],float); jscale=max(float(labels.J.std()),.05)
        tU=torch.as_tensor(U,dtype=torch.float64,device=device); tmu=torch.as_tensor(mu,dtype=torch.float64,device=device); tl=torch.as_tensor(logs,dtype=torch.float64,device=device); tls=torch.as_tensor(logsq,dtype=torch.float64,device=device); tsecond=torch.as_tensor(second,dtype=torch.float64,device=device)
        family=spec['family']; tylog=torch.as_tensor(logy,dtype=torch.float64,device=device) if family=='weibull' else None
        init=np.zeros((p,2)); init[0,0]=inverse_softplus(max(float(labels.J.mean()),.01)/jscale); cv=float(np.median(y.std(axis=1)/np.maximum(mu,1e-8)))
        init[0,1]=np.log(max(1/cv**2,.2)) if family=='gamma' else (np.log(max(cv**-1.086,.2)) if family=='weibull' else (np.log(max(np.sqrt(np.log1p(cv*cv)),.01)) if family=='lognormal' else -1/max(cv,.1)))
        def objective(beta):
            z=tx@beta; mean=tU+jscale*torch.nn.functional.softplus(z[:,0]); mean=torch.clamp(mean,min=1e-8)
            if family in ['gamma','weibull']:
                shape=torch.exp(torch.clamp(z[:,1],-3,8))
                if family=='gamma':
                    scale=mean/shape; nll=shape*torch.log(scale)+torch.lgamma(shape)-(shape-1)*tl+tmu/scale
                else:
                    logscale=torch.log(mean)-torch.lgamma(1+1/shape)
                    power=torch.exp(shape[:,None]*(tylog-logscale[:,None])).mean(axis=1)
                    nll=-torch.log(shape)+shape*logscale-(shape-1)*tl+power
            elif family=='lognormal':
                scale=torch.exp(torch.clamp(z[:,1],-5,2)); loc=torch.log(mean)-.5*scale*scale
                nll=tl+torch.log(scale)+.5*np.log(2*np.pi)+.5*(tls-2*loc*tl+loc*loc)/(scale*scale)
            else:
                a=torch.clamp(z[:,1],-100,20); hazard=torch.exp(-.5*a*a-.5*np.log(2*np.pi)-torch.special.log_ndtr(-a)); scale=mean/(hazard-a); loc=-a*scale
                nll=torch.log(scale)+.5*np.log(2*np.pi)+.5*(tsecond-2*loc*tmu+loc*loc)/(scale*scale)+torch.special.log_ndtr(-a)
            return nll.mean()+alpha*(beta[1:]**2).sum()
        coef,receipt=optimize_torch(init,objective); model.update(coef=coef,jscale=jscale,fit_receipt=receipt)
    elif route=='QUANTILE':
        levels=np.array([.05,.1,.25,.5,.75,.9,.95]); y=np.sort(np.asarray(samples,float),axis=1); prefix=np.column_stack([np.zeros(n),np.cumsum(y,axis=1)])
        ty=torch.as_tensor(y,dtype=torch.float64,device=device).contiguous(); tp=torch.as_tensor(prefix,dtype=torch.float64,device=device); tau=torch.as_tensor(levels,dtype=torch.float64,device=device); mean=ty.mean(axis=1)
        init=np.zeros((p,7)); init[0]=inverse_softplus(np.quantile(y,levels))
        def objective(beta):
            q=torch.nn.functional.softplus(tx@beta); pos=torch.searchsorted(ty,q.contiguous(),right=True); left=tp.gather(1,pos)
            expected=tau*(mean[:,None]-q)+(pos*q-left)/ty.shape[1]
            return expected.mean()+alpha*(beta[1:]**2).sum()
        coef,receipt=optimize_torch(init,objective,maxiter=700); model.update(coef=coef,levels=levels,fit_receipt=receipt)
    else: raise ValueError(route)
    return model


def qa_values(model,features):
    x=design_apply(model['design'],features)
    if model['spec']['engine']=='HGB': q=np.maximum(model['q_model'].predict(x[:,1:]),0); a=np.clip(model['a_model'].predict(x[:,1:]),0,1)
    else:
        z=x@model['coef']; q=model['qscale']*np.logaddexp(0,z[:,0]); a=expit(z[:,1])
    return q,a


def fit_joint_residual(model,cal_features,cal_labels):
    q,a=qa_values(model,cal_features); yq=cal_labels.Q.to_numpy(float); ya=np.clip(cal_labels.a.to_numpy(float),1e-5,1-1e-5); a=np.clip(a,1e-5,1-1e-5)
    residual=np.column_stack([np.log1p(yq)-np.log1p(q),np.log(ya/(1-ya))-np.log(a/(1-a))])
    scale=np.maximum(residual.std(axis=0),1e-5)
    km=KMeans(n_clusters=16,n_init=1,random_state=20260915,max_iter=100).fit(residual/scale)
    counts=np.bincount(km.labels_,minlength=16); model['joint_residual']={'centers':km.cluster_centers_*scale,'weights':counts/counts.sum(),'fit_rows':len(residual),'meaning':'discrete joint residual law preserving Q/a pairing, 16 fixed clusters'}


def weibull_from_moments(mean,spread):
    ks=np.geomspace(.15,1000,15000); cv=np.sqrt(np.expm1(gammaln(1+2/ks)-2*gammaln(1+1/ks)))
    shape=np.interp(np.asarray(spread)/np.maximum(mean,1e-12),cv[::-1],ks[::-1]); scale=mean/np.exp(gammaln(1+1/shape))
    return {'family':'weibull','shape':shape,'scale':np.maximum(scale,1e-10)}


def predict_parameters(model,features,probabilistic=False):
    U=np.asarray(features[:,model['U_index']],float); route=model['spec']['route']
    if route=='QA':
        q,a=qa_values(model,features)
        if probabilistic:
            r=model['joint_residual']; centers=r['centers']; q=np.maximum(np.expm1(np.log1p(q)[:,None]+centers[None,:,0]),0); ac=np.clip(a,1e-8,1-1e-8); a=expit(np.log(ac/(1-ac))[:,None]+centers[None,:,1]); weights=r['weights']
        else: q=q[:,None]; a=a[:,None]; weights=np.ones(1)
        mean=np.sqrt(U[:,None]**2+a*q); J=np.divide(a*q,mean+U[:,None],out=np.zeros_like(mean),where=(mean+U[:,None])>0); spread=np.sqrt(np.maximum((1-a)*q,0)); params=weibull_from_moments(mean.ravel(),spread.ravel())
        params.update(component_count=q.shape[1],weights=weights,J_components=J,mean_components=mean)
        return params
    x=design_apply(model['design'],features); z=x@model['coef']
    if route=='DIRECT':
        mean=U+model['jscale']*np.logaddexp(0,z[:,0]); f=model['spec']['family']
        if f=='gamma': shape=np.exp(np.clip(z[:,1],-3,8)); params={'family':f,'shape':shape,'scale':mean/shape}
        elif f=='weibull': shape=np.exp(np.clip(z[:,1],-3,8)); params={'family':f,'shape':shape,'scale':mean/np.exp(gammaln(1+1/shape))}
        elif f=='lognormal': scale=np.exp(np.clip(z[:,1],-5,2)); params={'family':f,'loc':np.log(mean)-.5*scale*scale,'scale':scale}
        else:
            alpha=np.clip(z[:,1],-100,20)
            with np.errstate(over='ignore'): hazard=np.sqrt(2/np.pi)/erfcx(alpha/np.sqrt(2))
            scale=mean/(hazard-alpha); params={'family':f,'loc':-alpha*scale,'scale':scale}
        params.update(component_count=1,weights=np.ones(1),J_components=(mean-U)[:,None],mean_components=mean[:,None]); return params
    q=np.sort(np.logaddexp(0,z),axis=1); levels=np.asarray(model['levels']); knots=np.column_stack([np.zeros(len(q)),q]); p=np.r_[0,levels]
    tail=(1-levels[-1])*(q[:,-1]-q[:,-2])/(levels[-1]-levels[-2]); mean=np.sum(np.diff(p)*(knots[:,1:]+knots[:,:-1])/2,axis=1)+(1-levels[-1])*(q[:,-1]+tail)
    shift=np.maximum(U-mean,0); mean+=shift
    return {'family':'quantile','quantiles':q,'levels':levels,'tail_scale':tail,'shift':shift,'component_count':1,'weights':np.ones(1),'J_components':(mean-U)[:,None],'mean_components':mean[:,None]}
