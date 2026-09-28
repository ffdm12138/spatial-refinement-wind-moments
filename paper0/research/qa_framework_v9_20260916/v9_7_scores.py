"""Memory-bounded scores and exact-Nt circular bootstrap for repaired streams."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sys
from v9_7_cache import SCHEMA, inputs_identity, binding, bind, mismatch
from v9_7_repair import OUT, HERE, MODELS, write_json
from window_evaluation import speed_grid,j_grid


def get_stream(meta,model,kind):
    p=OUT/meta['key']/'streams'/f'{model}_{kind}.npy'
    if not p.exists() and model in ('Det_QA','Climatology'):
        p=Path(meta['legacy'])/meta['stream']/f'{model}_{kind}.npy'
    return np.load(p,mmap_mode='r')


def truth_stream(meta,kind):
    return np.load(Path(meta['legacy'])/'truth'/f'truth_{kind}.npy',mmap_mode='r')


def scores(meta,models):
    out={model:{} for model in models}
    for kind,grid in [('speed',speed_grid()),('J',j_grid())]:
        truth=truth_stream(meta,kind);gw=truth.mean(axis=0,dtype=float)
        for model in models:
            a=get_stream(meta,model,kind);fw=a.mean(axis=0,dtype=float)
            if a.shape!=truth.shape or not np.isfinite(fw).all() or fw.min()<-1e-7 or fw.max()>1+1e-7:
                raise ValueError(('invalid CDF stream',meta['key'],model,kind,a.shape))
            out[model][f'{kind}_crps']=float(np.trapezoid((fw-gw)**2+gw*(1-gw),grid,axis=1).mean())
            out[model][f'{kind}_w1']=float(np.trapezoid(abs(fw-gw),grid,axis=1).mean())
    return out


def bootstrap(meta,candidate,baseline,seed,length=120,nboot=1000, *, expected=None):
    """Prefix sums in float64, batched resampling on CUDA; CPU cross-check."""
    import torch
    if expected is None:
        expected=binding(inputs_identity(sys.modules[__name__],meta,[candidate,baseline]),candidate,baseline,seed,length,nboot)
    nt=meta['time_count'];np_=len(meta['parents']);length=min(length,nt)
    count=(nt+length-1)//length;rem=nt-(count-1)*length
    starts=np.random.default_rng(seed).integers(0,nt,(nboot,count),dtype=np.int64)
    # The sampled block starts are the bootstrap mapping.  Retain a stable
    # digest plus the exact generation recipe so a reviewer can regenerate
    # the mapping from the recorded seed without treating a draw index as a
    # seed.  Draw-level deltas below are retained for the same reason: CI
    # summaries alone cannot establish which resamples produced them.
    starts_bytes=np.ascontiguousarray(starts,dtype=np.int64).tobytes()
    starts_sha256=hashlib.sha256(starts_bytes).hexdigest()
    lengths=np.full(count,length);lengths[-1]=rem;ends=starts+lengths
    result={};max_error=0.;draws={}
    for kind,grid in [('speed',speed_grid()),('J',j_grid())]:
        truth=truth_stream(meta,kind);cand=get_stream(meta,candidate,kind);base=get_stream(meta,baseline,kind)
        weights=np.empty(len(grid));weights[0]=(grid[1]-grid[0])/2;weights[-1]=(grid[-1]-grid[-2])/2;weights[1:-1]=(grid[2:]-grid[:-2])/2
        wt=torch.as_tensor(weights,dtype=torch.float64,device='cuda')
        st=torch.as_tensor(starts,device='cuda');en=torch.as_tensor(ends,device='cuda')
        delta=np.zeros(nboot);point=0.
        for parent in range(np_):
            for data,sign in [(base,1),(cand,-1)]:
                difference=np.asarray(data[:,parent],float)-np.asarray(truth[:,parent],float)
                point+=sign*np.trapezoid(difference.mean(axis=0)**2,grid)/np_
                prefix=np.vstack([np.zeros((1,len(grid))),np.cumsum(np.vstack([difference,difference[:length]]),axis=0)])
                # float64 subtraction avoids cancellation of neighboring prefix sums.
                blocks=prefix[np.arange(nt)+length]-prefix[np.arange(nt)]
                tails=prefix[np.arange(nt)+rem]-prefix[np.arange(nt)]
                bt=torch.as_tensor(blocks,dtype=torch.float32,device='cuda');tt=torch.as_tensor(tails,dtype=torch.float32,device='cuda')
                for b in range(0,nboot,50):
                    s=st[b:b+50]
                    avg=(bt[s[:,:-1]].sum(dim=1)+tt[s[:,-1]])/nt if count>1 else tt[s[:,-1]]/nt
                    vals=(avg.double().square()*wt).sum(dim=1).cpu().numpy()
                    delta[b:b+50]+=sign*vals/np_
                    if b==0:
                        direct=(prefix[ends[0]]-prefix[starts[0]]).sum(axis=0)/nt
                        exact=np.trapezoid(direct**2,grid)
                        max_error=max(max_error,abs(exact-vals[0]))
                del bt,tt
        draws[kind]=np.asarray(delta,dtype=np.float64)
        result.update({f'delta_{kind}_point':float(point),f'delta_{kind}_mean':float(delta.mean()),f'delta_{kind}_ci_low':float(np.quantile(delta,.025)),f'delta_{kind}_ci_high':float(np.quantile(delta,.975)),f'p_candidate_{kind}':float(np.mean(delta>0))})
    if max_error>2e-7:raise AssertionError(('bootstrap CUDA precision',max_error))
    result.update(
        block_len=length,block_count=count,dropped_tail_rows=0,wrapped_tail_rows=0,
        bootstrap_mode='exact_nt_circular_moving_block_bootstrap',seed=seed,
        n_boot=nboot,float64_crosscheck_max_abs=max_error,
        mapping_kind='circular_block_start_indices',
        mapping_rng='numpy.random.default_rng',
        mapping_dtype='int64',mapping_shape=list(starts.shape),
        mapping_sha256=starts_sha256,
        mapping_reproduction=(
            'np.random.default_rng(seed).integers(0, time_count, '
            '(n_boot, block_count), dtype=np.int64)'
        ),
        draw_delta_speed=draws['speed'].tolist(),
        draw_delta_J=draws['J'].tolist(),
        draw_delta_speed_sha256=hashlib.sha256(draws['speed'].tobytes()).hexdigest(),
        draw_delta_J_sha256=hashlib.sha256(draws['J'].tobytes()).hexdigest(),
        provenance_version=SCHEMA,
    )
    return bind(result,expected)


def provenance_complete(payload, expected=None):
    return mismatch(payload,expected) is None


def cached_bootstrap(path,meta,candidate,baseline,seed,length=120,nboot=1000,*,inputs=None):
    if inputs is None:inputs=inputs_identity(sys.modules[__name__],meta,[candidate,baseline])
    expected=binding(inputs,candidate,baseline,seed,length,nboot)
    if path.exists():
        try:old=json.loads(path.read_text())
        except (ValueError,OSError):old={}
        reason=mismatch(old,expected)
        if reason is None:return old
        print('STALE CACHE',path,reason,flush=True)
        # Retain exact stale bytes before replacing a cache.
        import shutil,hashlib
        archived=path.parent/'stale_cache'/f'{path.stem}_{hashlib.sha256(path.read_bytes()).hexdigest()}.json'
        archived.parent.mkdir(exist_ok=True)
        if not archived.exists():shutil.copyfile(path,archived)
    result=bootstrap(meta,candidate,baseline,seed,length,nboot,expected=expected)
    write_json(path,result)
    return result


def evaluate():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for i,meta in enumerate(manifest):
        d=OUT/meta['key'];destination=d/'evaluation.json'
        if not all((d/'streams'/f'{m}_receipt.json').exists() for m in MODELS):
            print('not ready:',meta['key'],flush=True);continue
        models=list(MODELS)
        if meta['key'] in ('B','C','A_670'):models+=['Det_QA','Climatology']
        sc=scores(meta,models);records=[]
        if meta['key'].startswith(('A_T_','A_S_','A_ST_')):
            pairs=[('Prob_JS_ind','Prob_JS_joint',4101+i),('QA_dyn','QA_const',4201+i),('QA_dyn','QA_m3',4301+i),('QA_dyn','Prob_JS_joint',5401+i),('QA_const','Prob_JS_joint',5501+i)]
        else:
            base_seed={'B':6200,'C':6300,'A_670':7670}[meta['key']]
            pairs=[(c,b,base_seed+(17*k if meta['key']!='A_670' else 0)) for k,(c,b) in enumerate([('QA_dyn','Prob_JS_joint'),('QA_const','Prob_JS_joint'),('Det_QA','Prob_JS_joint'),('QA_dyn','Climatology')])]
        inputs=inputs_identity(sys.modules[__name__],meta,models)
        for cand,base,seed in pairs:
            cache=d/f'bootstrap_{cand}_{base}.json'
            boot=cached_bootstrap(cache,meta,cand,base,seed,inputs=inputs)
            records.append(dict(candidate=cand,baseline=base,**boot))
            for kind in ('speed','J'):
                expected=sc[base][f'{kind}_crps']-sc[cand][f'{kind}_crps']
                if abs(expected-boot[f'delta_{kind}_point'])>2e-12:
                    raise AssertionError(('score/bootstrap point mismatch',meta['key'],cand,base,kind))
            print(meta['key'],cand,base,boot['delta_speed_point'],boot['delta_J_point'],flush=True)
        if meta['key']!='A_670':
            seed=next(seed for c,b,seed in pairs if c=='QA_dyn' and b=='Prob_JS_joint')
            for length in (60,240):
                cache=d/f'bootstrap_sensitivity_{length}.json'
                cached_bootstrap(cache,meta,'QA_dyn','Prob_JS_joint',seed,length,inputs=inputs)
        write_json(destination,dict(scores=sc,bootstrap=records))


if __name__=='__main__':evaluate()
