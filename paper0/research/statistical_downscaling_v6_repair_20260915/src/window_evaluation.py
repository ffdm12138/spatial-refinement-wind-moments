"""Full-time conditional and window CDF evaluation, with no time subsampling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from families import cdf
from scientific_models import predict_parameters


def speed_grid(refinement=1):
    return np.unique(np.r_[np.linspace(0,64,2048*refinement+1),np.geomspace(64,4096,256*refinement+1)])


def j_grid(refinement=1):
    return np.unique(np.r_[np.linspace(0,8,1024*refinement+1),np.geomspace(8,1024,256*refinement+1)])


def empirical_rows(samples,grid):
    sorted_rows=np.sort(np.asarray(samples),axis=1)
    return np.stack([np.searchsorted(row,grid,side='right')/len(row) for row in sorted_rows]).astype(np.float32)


def parameter_cdf(params,grid):
    n=len(params['J_components']); components=params['component_count']
    if params['family']!='quantile':
        values=cdf(params,grid).reshape(n,components,len(grid))
        return np.einsum('nkg,k->ng',values,params['weights'])
    q=np.column_stack([np.zeros(n),params['quantiles']]); p=np.r_[0,params['levels']]; x=grid[None,:]-params['shift'][:,None]; out=np.zeros_like(x)
    for i,weight in enumerate(np.diff(p)):
        width=q[:,i+1]-q[:,i]; positive=width>1e-12
        fraction=np.clip((x-q[:,i,None])/np.maximum(width[:,None],1e-12),0,1)
        out+=weight*np.where(positive[:,None],fraction,(x>=q[:,i,None]))
    tail=params['tail_scale']; residual=np.maximum(x-q[:,-1,None],0)
    out+=(1-p[-1])*np.where(tail[:,None]>1e-12,-np.expm1(-residual/np.maximum(tail[:,None],1e-12)),x>=q[:,-1,None])
    return np.clip(out,0,1)


def make_truth(samples,labels,sg=None,jg=None):
    sg=speed_grid() if sg is None else sg; jg=j_grid() if jg is None else jg
    n=len(labels); nt=n//36
    if n%36 or not np.array_equal(labels.parent_id,np.tile(np.arange(36),nt)): raise ValueError('Expected complete chronological ALL36 rows')
    G=empirical_rows(samples,sg).reshape(nt,36,len(sg)); Gj=(labels.J.to_numpy()[:,None]<=jg).astype(np.float32).reshape(nt,36,len(jg))
    return {'speed_grid':sg,'j_grid':jg,'speed_rows':G,'j_rows':Gj,'speed_window':G.mean(axis=0,dtype=np.float64),'j_window':Gj.mean(axis=0,dtype=np.float64),'labels':labels}


def window_quantiles(F,grid,levels=(.05,.25,.5,.75,.95)):
    return np.stack([np.interp(levels,row,grid) for row in F])


def summarize(F,Fj,truth,jpoint,meanpoint,model_id,retain=False):
    nt=truth['speed_rows'].shape[0]; sg=truth['speed_grid']; jg=truth['j_grid']; G=truth['speed_rows']; Gj=truth['j_rows']; labels=truth['labels']
    F=np.asarray(F).reshape(nt,36,len(sg)); Fj=np.asarray(Fj).reshape(nt,36,len(jg))
    if not np.isfinite(F).all() or not np.isfinite(Fj).all() or np.min(F)<-1e-8 or np.max(F)>1+1e-8: raise ValueError('Invalid predictive CDF')
    fw=F.mean(axis=0,dtype=np.float64); gw=truth['speed_window']; fjw=Fj.mean(axis=0,dtype=np.float64); gjw=truth['j_window']
    tail=float(max(np.max(1-F[:,:,-1]),np.max(1-Fj[:,:,-1])))
    speed_excess=np.trapezoid((fw-gw)**2,sg,axis=1); speed_crps=speed_excess+np.trapezoid(gw*(1-gw),sg,axis=1)
    jex=np.trapezoid((fjw-gjw)**2,jg,axis=1); jcrps=jex+np.trapezoid(gjw*(1-gjw),jg,axis=1)
    pq=window_quantiles(fw,sg); rq=window_quantiles(gw,sg)
    jp=np.asarray(jpoint).reshape(nt,36); mp=np.asarray(meanpoint).reshape(nt,36); jy=labels.J.to_numpy().reshape(nt,36)
    parent=pd.DataFrame({'model_id':model_id,'parent_id':np.arange(36),'speed_w1':np.trapezoid(np.abs(fw-gw),sg,axis=1),'speed_crps':speed_crps,'speed_excess_crps':speed_excess,'J_w1':np.trapezoid(np.abs(fjw-gjw),jg,axis=1),'J_crps':jcrps,'J_rmse':np.sqrt(np.mean((jp-jy)**2,axis=0)),'mean_speed_bias':np.mean(mp-labels.mu.to_numpy().reshape(nt,36),axis=0),'conditional_speed_w1':np.mean(np.trapezoid(np.abs(F-G),sg,axis=2),axis=0),'conditional_speed_crps':np.mean(np.trapezoid((F-G)**2+G*(1-G),sg,axis=2),axis=0)})
    for i,p in enumerate([5,25,50,75,95]): parent[f'q{p:02d}_error']=pq[:,i]-rq[:,i]
    domainF=fw.mean(axis=0); domainG=gw.mean(axis=0)
    summary={'model_id':model_id,'speed_w1':float(parent.speed_w1.mean()),'speed_crps':float(parent.speed_crps.mean()),'speed_excess_crps':float(parent.speed_excess_crps.mean()),'J_w1':float(parent.J_w1.mean()),'J_crps':float(parent.J_crps.mean()),'J_rmse':float(np.sqrt(np.mean((jp-jy)**2))),'q05_mae':float(parent.q05_error.abs().mean()),'q95_mae':float(parent.q95_error.abs().mean()),'domain_speed_w1':float(np.trapezoid(np.abs(domainF-domainG),sg)),'conditional_speed_w1':float(parent.conditional_speed_w1.mean()),'conditional_speed_crps':float(parent.conditional_speed_crps.mean()),'tail_mass_at_upper':tail,'valid_cdf':bool(tail<1e-6),'time_count':nt,'sample_count':nt*36*729}
    result={'summary':summary,'parents':parent,'speed_window':fw,'J_window':fjw,'J_point':jp,'mean_point':mp}
    if retain: result.update(speed_rows=F,j_rows=Fj)
    return result


def evaluate_model(model,features,truth,model_id,probabilistic=False,retain=False):
    n=len(features); sg=truth['speed_grid']; jg=truth['j_grid']; F=np.empty((n,len(sg)),np.float32); Fj=np.empty((n,len(jg)),np.float32); J=np.empty(n); mean=np.empty(n)
    for start in range(0,n,256):
        stop=min(n,start+256); params=predict_parameters(model,features[start:stop],probabilistic)
        F[start:stop]=parameter_cdf(params,sg)
        Fj[start:stop]=np.einsum('nkg,k->ng',(params['J_components'][:,:,None]<=jg).astype(float),params['weights'])
        J[start:stop]=params['J_components']@params['weights']; mean[start:stop]=params['mean_components']@params['weights']
    return summarize(F,Fj,truth,J,mean,model_id,retain)


def baseline_curves(train_speed,train_labels,truth,model_id):
    sg=truth['speed_grid']; jg=truth['j_grid']; nt=truth['speed_rows'].shape[0]; y=np.asarray(train_speed).reshape(-1,729)
    if model_id=='V6_B0':
        s=np.sort(y.ravel()); j=np.sort(train_labels.J.to_numpy()); F=np.broadcast_to(np.searchsorted(s,sg,side='right')/len(s),(nt,36,len(sg))); Fj=np.broadcast_to(np.searchsorted(j,jg,side='right')/len(j),(nt,36,len(jg))); jp=np.full(nt*36,j.mean()); mp=np.full(nt*36,s.mean(dtype=np.float64))
    elif model_id=='V6_B1':
        fs=[]; fj=[]; jm=[]; sm=[]
        for p in range(36):
            s=np.sort(y[p::36].ravel()); j=np.sort(train_labels.J.to_numpy()[p::36]); fs.append(np.searchsorted(s,sg,side='right')/len(s)); fj.append(np.searchsorted(j,jg,side='right')/len(j)); jm.append(j.mean()); sm.append(s.mean(dtype=np.float64))
        F=np.broadcast_to(fs,(nt,36,len(sg))); Fj=np.broadcast_to(fj,(nt,36,len(jg))); jp=np.tile(jm,nt); mp=np.tile(sm,nt)
    elif model_id=='V6_B2':
        U=train_labels.U.to_numpy(); bounds=np.unique(np.quantile(U,np.linspace(0,1,9)))[1:-1]; bins=np.searchsorted(bounds,U); evalbins=np.searchsorted(bounds,truth['labels'].U.to_numpy()); fs=[]; fj=[]; jm=[]; sm=[]
        for b in range(len(bounds)+1):
            mask=bins==b; s=np.sort(y[mask].ravel()); j=np.sort(train_labels.J.to_numpy()[mask]); fs.append(np.searchsorted(s,sg,side='right')/len(s)); fj.append(np.searchsorted(j,jg,side='right')/len(j)); jm.append(j.mean()); sm.append(s.mean(dtype=np.float64))
        F=np.asarray(fs)[evalbins]; Fj=np.asarray(fj)[evalbins]; jp=np.asarray(jm)[evalbins]; mp=np.asarray(sm)[evalbins]
    else: raise ValueError(model_id)
    return summarize(F,Fj,truth,jp,mp,model_id)
