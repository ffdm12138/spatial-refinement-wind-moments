"""Numerical integration repair; model, label and sampling contracts unchanged.

J marginals integrate the conditional Gaussian analytically.  Speed retains
the original moment-matched Weibull family and uses explicit-order Gaussian
quadrature.  No fit or model selection occurs in this module.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, ndtr

KS = np.geomspace(.15, 1000., 15000)
CV = np.sqrt(np.expm1(gammaln(1 + 2 / KS) - 2 * gammaln(1 + 1 / KS)))
SPEED_QA_ORDER = 32
SPEED_JS_ORDER = (96, 24)
GAP_ORDER = 48


def normal_rule(order):
    z, w = np.polynomial.hermite.hermgauss(order)
    return z * np.sqrt(2.), w / np.sqrt(np.pi)


def validate_inputs(pred, u, calibration):
    u = np.asarray(u, float)
    if u.ndim != 1 or not np.isfinite(u).all() or (u < 0).any():
        raise ValueError('U must be a finite nonnegative vector')
    for key in ('logQ', 'logita', 'logJ', 'logS'):
        a = np.asarray(pred[key])
        if a.shape != u.shape or not np.isfinite(a).all():
            raise ValueError(f'invalid latent location {key}')
    for key in ('std_logQ', 'std_logita', 'std_logJ', 'std_logS'):
        if not np.isfinite(calibration[key]) or calibration[key] <= 0:
            raise ValueError(f'invalid residual scale {key}')


def gap_cdf(pred, u, calibration, model, grid, order=48):
    """J CDF: exact lognormal JS/constant-QA; smooth 1-D conditional QA.

    Given eta=logit(a), J<=j iff log(Q)<=log(j*(2*U+j))-log(a).
    For M3, log(Q)|eta is Gaussian with its unchanged frozen covariance.
    Conditioning removes the discontinuous indicator that broke GH3.
    """
    u = np.asarray(u, float)
    grid = np.asarray(grid, float)
    if grid.ndim != 1 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
        raise ValueError('invalid integration grid')
    positive = grid > 0
    out = np.zeros((len(u), len(grid)), dtype=float)
    g = grid[positive]
    if model.startswith('Prob_JS'):
        out[:, positive] = ndtr((np.log(g)[None, :] - pred['logJ'][:, None]) / calibration['std_logJ'])
        return out
    logthreshold = np.log(g[None, :] * (2 * u[:, None] + g[None, :]))
    if model == 'QA_const':
        a = calibration['a_const']
        if not 0 < a <= 1:
            raise ValueError('invalid a_const')
        out[:, positive] = ndtr((logthreshold - np.log(a) - pred['logQ'][:, None]) / calibration['std_logQ'])
        return out
    if model not in ('QA_dyn', 'QA_m3'):
        raise ValueError(model)
    vq = calibration['std_logQ'] ** 2
    va = calibration['std_logita'] ** 2
    cov = float(calibration['covariance_QA'][0][1]) if model == 'QA_m3' else 0.
    conditional_var = vq - cov * cov / va
    if conditional_var <= 0:
        raise ValueError('non-positive conditional covariance')
    sigma = np.sqrt(conditional_var)
    nodes, weights = normal_rule(order)
    for z, w in zip(nodes, weights):
        deta = calibration['std_logita'] * z
        eta = pred['logita'] + deta
        loga = -np.logaddexp(0., -eta)
        meanq = pred['logQ'] + cov / va * deta
        out[:, positive] += w * ndtr((logthreshold - loga[:, None] - meanq[:, None]) / sigma)
    return np.clip(out, 0., 1.)


def moment_nodes(pred, u, calibration, model, order):
    """Original Gaussian law and deployment; only quadrature order varies."""
    n1, n2 = (order, order) if np.isscalar(order) else order
    z, w = normal_rule(n1)
    if model == 'QA_const':
        za, zb, weights = z, np.zeros_like(z), w
    else:
        z2, w2 = normal_rule(n2)
        aa, bb = np.meshgrid(z, z2)
        wa, wb = np.meshgrid(w, w2)
        za, zb, weights = aa.ravel(), bb.ravel(), (wa * wb).ravel()
    if model.startswith('QA'):
        if model == 'QA_m3':
            chol = np.asarray(calibration['cholesky_QA'])
            dq, da = chol[0,0]*za+chol[0,1]*zb, chol[1,0]*za+chol[1,1]*zb
        else:
            dq, da = calibration['std_logQ']*za, calibration['std_logita']*zb
        q = np.exp(pred['logQ'][None, :] + dq[:, None])
        a = calibration['a_const'] if model == 'QA_const' else expit(pred['logita'][None, :] + da[:, None])
        mu = np.sqrt(u[None, :]**2 + a*q)
        spread = np.sqrt((1-a)*q)
        gap = (a*q)/(mu+u[None, :])
    else:
        if model == 'Prob_JS_joint':
            chol = np.asarray(calibration['cholesky_JS'])
            dj, ds = chol[0,0]*za+chol[0,1]*zb, chol[1,0]*za+chol[1,1]*zb
        elif model == 'Prob_JS_ind':
            dj, ds = calibration['std_logJ']*za, calibration['std_logS']*zb
        else:
            raise ValueError(model)
        gap = np.exp(pred['logJ'][None, :] + dj[:, None])
        mu = u[None, :] + gap
        spread = np.exp(pred['logS'][None, :] + ds[:, None])
    return mu, gap, np.broadcast_to(spread, mu.shape), weights


def gap_cdf_gpu(pred,u,calibration,model,grid,order=48):
    """Accelerate only the smooth conditional QA integral; exact CPU marginals otherwise."""
    if model not in ('QA_dyn','QA_m3'):
        return gap_cdf(pred,u,calibration,model,grid,order)
    import torch
    if not torch.cuda.is_available():return gap_cdf(pred,u,calibration,model,grid,order)
    grid=np.asarray(grid);positive=grid>0;g=grid[positive];u=np.asarray(u)
    vq=calibration['std_logQ']**2;va=calibration['std_logita']**2
    cov=float(calibration['covariance_QA'][0][1]) if model=='QA_m3' else 0.
    variance=vq-cov*cov/va
    if variance<=0:raise ValueError('non-positive conditional covariance')
    threshold=np.log(g[None,:]*(2*u[:,None]+g[None,:]))
    nodes,weights=normal_rule(order)
    offsets=[]
    for z in nodes:
        deta=calibration['std_logita']*z
        loga=-np.logaddexp(0.,-(pred['logita']+deta))
        offsets.append(loga+pred['logQ']+cov/va*deta)
    with torch.no_grad():
        tt=torch.as_tensor(threshold,dtype=torch.float32,device='cuda')
        total=torch.zeros_like(tt,dtype=torch.float64)
        for k in range(0,order,8):
            off=torch.as_tensor(np.asarray(offsets[k:k+8]),dtype=torch.float32,device='cuda')
            ww=torch.as_tensor(weights[k:k+8],dtype=torch.float32,device='cuda')
            # Survival formulation also removes upper-tail accumulation drift.
            sf=torch.special.ndtr((off[:,:,None]-tt[None,:,:])/np.sqrt(variance))
            total+=(sf*ww[:,None,None]).sum(dim=0,dtype=torch.float64)
        out=np.zeros((len(u),len(grid)))
        out[:,positive]=1-total.cpu().numpy()
    return np.clip(out,0,1)


def speed_cdf(pred, u, calibration, model, grid, order=24, device='cuda'):
    """Weibull mixture on the ORIGINAL scoring grid (float64 CPU fallback)."""
    mu, gap, spread, weights = moment_nodes(pred, u, calibration, model, order)
    expectations = {k: np.average(a,axis=0,weights=weights) for k,a in [('mu',mu),('J',gap),('S',spread)]}
    # Omit only the smallest nodes with TOTAL mass <= 1e-12. Since each
    # conditional CDF lies in [0,1], the uniform CDF error is <= 1e-12.
    # Expectations use all nodes: their integrands are not bounded by one.
    ranked=np.argsort(weights);discard=int(np.searchsorted(np.cumsum(weights[ranked]),1e-12,side='right'))
    keep=np.sort(ranked[discard:])
    mu,gap,spread,weights=mu[keep],gap[keep],spread[keep],weights[keep]
    cv = spread / np.maximum(mu, 1e-12)
    shape = np.interp(cv, CV[::-1], KS[::-1])
    logscale = np.log(np.maximum(mu / np.exp(gammaln(1+1/shape)), 1e-10))
    grid = np.asarray(grid, float)
    if device == 'cpu':
        out = np.zeros((len(u), len(grid)), dtype=float)
        for k, w in enumerate(weights):
            q = shape[k,:,None] * (np.log(np.maximum(grid,1e-300))[None,:] - logscale[k,:,None])
            out += w * -np.expm1(-np.exp(np.minimum(q, 80)))
    else:
        import torch
        with torch.no_grad():
            gx = torch.as_tensor(np.log(np.maximum(grid,1e-300)),dtype=torch.float32,device=device)
            total = torch.zeros((len(u), len(grid)),dtype=torch.float64,device=device)
            for k in range(0,len(weights),16):
                sh = torch.as_tensor(shape[k:k+16],dtype=torch.float32,device=device)
                ls = torch.as_tensor(logscale[k:k+16],dtype=torch.float32,device=device)
                ww = torch.as_tensor(weights[k:k+16],dtype=torch.float32,device=device)
                q = sh[:,:,None] * (gx[None,None,:]-ls[:,:,None])
                # Accumulate survival: saturated upper tails are exactly one
                # after complementing, avoiding long-grid W1 bias from weight rounding.
                c = torch.exp(-torch.exp(torch.clamp(q,max=80)))
                total += torch.sum(c * ww[:,None,None],dim=0,dtype=torch.float64)
            out = 1. - total.cpu().numpy()
    out[:,grid<=0] = 0.
    if not np.isfinite(out).all():
        raise FloatingPointError('non-finite predictive CDF')
    return np.clip(out,0.,1.), expectations


def write_streams(models, calibration, X, u, truth, out_dir, include_det=False, climatology=None):
    """Canonical corrected entry point; retains the existing model contract."""
    from v9_6_common import predict_scalar, qa_deploy_v96
    from v9_6_run_nested import write_cdfs_to_disk
    try:
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device='cpu'
    pred={k:predict_scalar(models[k],X) for k in ('logQ','logita','logJ','logS')}
    validate_inputs(pred,u,calibration)
    out_dir.mkdir(parents=True,exist_ok=True)
    nt,np_=truth['j_rows'].shape[:2];n=len(u);paths={};expectations={}
    for model in ('QA_dyn','QA_m3','QA_const','Prob_JS_ind','Prob_JS_joint'):
        order=SPEED_JS_ORDER if model.startswith('Prob_JS') else SPEED_QA_ORDER
        paths[model]={kind:out_dir/f'{model}_{kind}.npy' for kind in ('speed','J')}
        maps={kind:np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(nt,np_,len(truth['speed_grid' if kind=='speed' else 'j_grid']))).reshape(n,-1) for kind,path in paths[model].items()}
        ex={k:np.empty(n) for k in ('mu','J','S')}
        for start in range(0,n,128):
            end=min(n,start+128);p={k:v[start:end] for k,v in pred.items()};uu=u[start:end]
            maps['speed'][start:end],ee=speed_cdf(p,uu,calibration,model,truth['speed_grid'],order,device)
            maps['J'][start:end]=gap_cdf_gpu(p,uu,calibration,model,truth['j_grid'],GAP_ORDER)
            for k in ex:ex[k][start:end]=ee[k]
        for m in maps.values():m.flush()
        expectations[model]=ex
    if include_det:
        q=np.maximum(predict_scalar(models['rawQ'],X),1e-4)
        a=np.clip(predict_scalar(models['rawa'],X),1e-4,1-1e-4)
        mu,j,s=qa_deploy_v96(q,a,u)
        paths['Det_QA']=write_cdfs_to_disk([j],[mu],[s],[1.],truth,out_dir,'Det_QA')
        expectations['Det_QA']=dict(mu=mu,J=j,S=s)
    if climatology is not None:
        ex={k:np.full(n,climatology[k]) for k in ('mu','J','S')}
        paths['Climatology']=write_cdfs_to_disk([ex['J']],[ex['mu']],[ex['S']],[1.],truth,out_dir,'Climatology')
        expectations['Climatology']=ex
    return paths,expectations,pred


def constant_qa_score(logq,u,calibration,truth):
    """Corrected constant-partition falsification on its original fixed subgrid."""
    n=len(u);nt,np_=truth['j_rows'].shape[:2]
    avg={kind:np.zeros((np_,len(truth[g]))) for kind,g in [('speed','speed_grid'),('J','j_grid')]}
    for t in range(0,nt,16):
        sl=slice(t*np_,min(nt,t+16)*np_);uu=u[sl];q=logq[sl]
        p=dict(logQ=q,logita=np.zeros(len(q)),logJ=np.zeros(len(q)),logS=np.zeros(len(q)))
        fs,_=speed_cdf(p,uu,calibration,'QA_const',truth['speed_grid'],order=SPEED_QA_ORDER)
        fj=gap_cdf(p,uu,calibration,'QA_const',truth['j_grid'])
        for kind,f in [('speed',fs),('J',fj)]:avg[kind]+=f.reshape(-1,np_,f.shape[-1]).sum(axis=0)/nt
    result={}
    for kind,g,rows in [('speed','speed_grid','speed_rows'),('J','j_grid','j_rows')]:
        gw=np.asarray(truth[rows]).mean(axis=0,dtype=float);fw=avg[kind];grid=truth[g]
        result[f'{kind}_crps']=float(np.trapezoid((fw-gw)**2+gw*(1-gw),grid,axis=1).mean())
        result[f'{kind}_w1']=float(np.trapezoid(abs(fw-gw),grid,axis=1).mean())
    return result


def constant_qa_pair_scores(pred,u,targets,cal,truth_s,truth_j,sg,jg,batch_size=256):
    pred=np.asarray(pred);u=np.asarray(u);targets=np.asarray(targets);count,nt=pred.shape
    out_s=np.empty(count);out_j=np.empty(count)
    # Each pair remains independent; aggregate every original time row.
    for i in range(count):
        means=[np.zeros(len(sg)),np.zeros(len(jg))]
        for t in range(0,nt,256):
            q=pred[i,t:t+256];uu=u[i,t:t+256];p=dict(logQ=q,logita=np.zeros(len(q)),logJ=np.zeros(len(q)),logS=np.zeros(len(q)))
            fs,_=speed_cdf(p,uu,cal,'QA_const',sg,order=SPEED_QA_ORDER)
            fj=gap_cdf(p,uu,cal,'QA_const',jg)
            means[0]+=fs.sum(axis=0)/nt;means[1]+=fj.sum(axis=0)/nt
        for dest,fw,gw,grid in [(out_s,means[0],truth_s[targets[i]],sg),(out_j,means[1],truth_j[targets[i]],jg)]:
            dest[i]=np.trapezoid((fw-gw)**2+gw*(1-gw),grid)
    return out_s,out_j
