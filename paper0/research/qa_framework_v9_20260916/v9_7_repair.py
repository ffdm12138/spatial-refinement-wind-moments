"""Replay the existing predictions and repair numerical integration only.

All new outputs are staged; promotion is a separate validated operation.
The original feature/label caches, calibrations, partitions and model families
are read-only. Missing outer fit caches are replayed with the original code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from v9_6_common import (load_case, level_segments, fit_models_on_rows,
    feature_block_v96, predict_scalar, D02_T4_NAMES)
from v9_7_numerics import gap_cdf, gap_cdf_gpu, speed_cdf, moment_nodes, validate_inputs
from window_evaluation import speed_grid, j_grid

HERE = Path(__file__).resolve().parent
OUT = HERE / 'numerical_repair_20260921'
MODELS = ('QA_dyn','QA_m3','QA_const','Prob_JS_ind','Prob_JS_joint')


def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def prepare():
    OUT.mkdir(exist_ok=True)
    y, x, _ = load_case('A')
    specs=[]
    for level in ('A_T','A_S','A_ST'):
        for fold,tr,case,te,parents,info in level_segments(level):
            key=f'{level}_{fold}'; d=OUT/key;d.mkdir(exist_ok=True)
            calpath=HERE/'OUTER_FOLD_CALIBRATION_RECEIPTS'/f'OUTER_CALIBRATION_{key}.json'
            cal=json.loads(calpath.read_text())['parameters']
            if (d/'models.pkl.gz').exists():
                models=pd.read_pickle(d/'models.pkl.gz')
            else:
                models=fit_models_on_rows(y,x,tr)
                pd.to_pickle(models,d/'models.pkl.gz')
            specs.append(dict(key=key,case='A',parents=parents,train_rows=len(tr),
                rows=te,models=models,calibration=cal,legacy=str(HERE/'v9_6_a_miyeemrw'/key),stream='streams',level=level,fold=fold))
            print('prepared original models',key,flush=True)
    models={k:pd.read_pickle(HERE/'V9_4_CANONICAL_MODELS'/f'm_{k}.pkl.gz') for k in ['logQ','logita','logJ','logS','rawQ','rawa']}
    cal=json.loads((HERE/'V9_6_TRANSFER_CALIBRATION.json').read_text())['calibration']
    for case in ('B','C','A'):
        start=3650 if case=='A' else 970;key='A_670' if case=='A' else case
        legacy=HERE/('v9_6_confirmatory_670_u261cluv' if case=='A' else f'v9_6_bc_streams/{case}')
        specs.append(dict(key=key,case=case,parents=list(range(36)),train_rows=96480,
            rows=np.arange(start*36,4320*36),models=models,calibration=cal,legacy=str(legacy),stream='streams_prim',level=key,fold='full'))
    manifest=[]
    for s in specs:
        d=OUT/s['key'];d.mkdir(exist_ok=True)
        labels,features,_=load_case(s['case']);rows=s['rows'];X=feature_block_v96(features,rows,D02_T4_NAMES)
        pred={k:predict_scalar(m,X) for k,m in s['models'].items()}
        u=labels.iloc[rows].U.to_numpy(float)
        validate_inputs(pred,u,s['calibration'])
        np.savez_compressed(d/'predictions.npz',**pred,U=u,rows=rows)
        if not (d/'models.pkl.gz').exists():pd.to_pickle(s['models'],d/'models.pkl.gz')
        write_json(d/'calibration.json',s['calibration'])
        # Compare the replay to actual legacy streams at deterministic full-parent slices.
        old=Path(s['legacy']); nt=len(rows)//len(s['parents']);times=np.unique(np.linspace(0,nt-1,13,dtype=int))
        ix=(times[:,None]*len(s['parents'])+np.arange(len(s['parents']))).ravel()
        pp={k:v[ix] for k,v in pred.items()};uu=u[ix];checks=[]
        if s['key']!='A_670':
            for model in MODELS:
                order=5 if model=='QA_const' else 3
                mu,j,ss,w=moment_nodes(pp,uu,s['calibration'],model,order)
                fj=np.zeros((len(ix),len(j_grid())),np.float32)
                for atom,weight in zip(j,w):fj+=(weight*(atom[:,None]<=j_grid())).astype(np.float32)
                ref=np.load(old/s['stream']/f'{model}_J.npy',mmap_mode='r')[times].reshape(fj.shape)
                fs,_=speed_cdf(pp,uu,s['calibration'],model,speed_grid(),order=order,device='cpu')
                refs=np.load(old/s['stream']/f'{model}_speed.npy',mmap_mode='r')[times].reshape(fs.shape)
                ej=float(abs(fj-ref).max());es=float(abs(fs-refs).max())
                if ej>2e-6 or es>2e-6:raise AssertionError((s['key'],model,'legacy replay mismatch',ej,es))
                checks.append(dict(model=model,J_max_abs=ej,speed_max_abs=es))
        meta={k:v for k,v in s.items() if k not in ('models','rows','calibration')}
        meta.update(row_count=len(rows),time_count=nt,legacy_checks=checks)
        write_json(d/'metadata.json',meta);manifest.append(meta)
        print('legacy replay verified',s['key'],max((a['speed_max_abs'] for a in checks),default=0),flush=True)
    write_json(OUT/'manifest.json',manifest)


def probe():
    results=[]
    for meta in json.loads((OUT/'manifest.json').read_text()):
        d=OUT/meta['key'];z=np.load(d/'predictions.npz');cal=json.loads((d/'calibration.json').read_text())
        # Deterministic spread through every fold and the full U range.
        ids=np.unique(np.r_[np.linspace(0,len(z['U'])-1,64,dtype=int),np.argsort(z['U'])[np.linspace(0,len(z['U'])-1,32,dtype=int)]])
        p={k:z[k][ids] for k in ['logQ','logita','logJ','logS']};u=z['U'][ids]
        for model in MODELS:
            previous=None
            for order in (8,16,24,40,64):
                t=time.time();fs,_=speed_cdf(p,u,cal,model,speed_grid(),order=order)
                fj=gap_cdf(p,u,cal,model,j_grid(),order=order)
                if previous is not None:
                    ds=np.trapezoid(abs(fs-previous[0]),speed_grid(),axis=1)
                    dj=np.trapezoid(abs(fj-previous[1]),j_grid(),axis=1)
                    row=dict(key=meta['key'],model=model,order=order,speed_mean_L1=float(ds.mean()),speed_max_L1=float(ds.max()),J_mean_L1=float(dj.mean()),J_max_L1=float(dj.max()),seconds=time.time()-t)
                    results.append(row);print(row,flush=True)
                previous=(fs,fj)
            cpu,_=speed_cdf({k:v[:3] for k,v in p.items()},u[:3],cal,model,speed_grid(),order=16,device='cpu')
            gpu,_=speed_cdf({k:v[:3] for k,v in p.items()},u[:3],cal,model,speed_grid(),order=16)
            if np.max(abs(cpu-gpu))>2e-6:raise AssertionError('GPU precision')
        write_json(OUT/'probe.json',results)


def window_probe():
    results=[]
    for key in ('A_T_fold1','A_S_S1','B','C','A_670'):
        d=OUT/key; meta=json.loads((d/'metadata.json').read_text())
        z=np.load(d/'predictions.npz');cal=json.loads((d/'calibration.json').read_text())
        np_=len(meta['parents']); nt=meta['time_count']
        times=np.unique(np.linspace(0,nt-1,24,dtype=int))
        ix=(times[:,None]*np_+np.arange(np_)).ravel()
        p={k:z[k][ix] for k in ['logQ','logita','logJ','logS']};u=z['U'][ix]
        grid=speed_grid()[::4]
        truth=np.load(Path(meta['legacy'])/'truth/truth_speed.npy',mmap_mode='r')[times,:,::4].mean(axis=0,dtype=float)
        for model in ('QA_dyn','QA_m3','QA_const','Prob_JS_ind','Prob_JS_joint'):
            orders=(16,24,40) if model.startswith('QA') else ((24,12),(48,16),(96,24),(160,32))
            previous=None
            for order in orders:
                t=time.time();parts=[]
                for start in range(0,len(u),128):
                    fs,_=speed_cdf({k:v[start:start+128] for k,v in p.items()},u[start:start+128],cal,model,grid,order=order)
                    parts.append(fs)
                avg=np.concatenate(parts).reshape(len(times),np_,len(grid)).mean(axis=0)
                score=np.trapezoid((avg-truth)**2,grid,axis=1).mean()
                item=dict(key=key,model=model,order=order,score=float(score),seconds=time.time()-t)
                if previous is not None:
                    item.update(score_change=float(score-previous[0]),mean_L1=float(np.trapezoid(abs(avg-previous[1]),grid,axis=1).mean()))
                results.append(item); print(item,flush=True);previous=score,avg
                write_json(OUT/'window_probe.json',results)


def generate():
    """Persist repaired CDFs on every original time/parent/grid coordinate."""
    for meta in json.loads((OUT/'manifest.json').read_text()):
        d=OUT/meta['key']; z=np.load(d/'predictions.npz');cal=json.loads((d/'calibration.json').read_text())
        p={k:z[k] for k in ['logQ','logita','logJ','logS']}; u=z['U'];n=len(u)
        for model in MODELS:
            stream=d/'streams';stream.mkdir(exist_ok=True)
            done=stream/f'{model}_receipt.json'
            if done.exists():continue
            order=(96,24) if model.startswith('Prob_JS') else 32
            outputs={}
            for kind,grid in [('speed',speed_grid()),('J',j_grid())]:
                outputs[kind]=np.lib.format.open_memmap(stream/f'{model}_{kind}.npy',mode='w+',dtype=np.float32,shape=(meta['time_count'],len(meta['parents']),len(grid))).reshape(n,len(grid))
            expectation={k:np.empty(n) for k in ['mu','J','S']};t=time.time()
            for start in range(0,n,128):
                end=min(n,start+128);pp={k:v[start:end] for k,v in p.items()};uu=u[start:end]
                fs,ex=speed_cdf(pp,uu,cal,model,speed_grid(),order=order)
                fj=gap_cdf_gpu(pp,uu,cal,model,j_grid(),order=48)
                outputs['speed'][start:end]=fs;outputs['J'][start:end]=fj
                for k in expectation:expectation[k][start:end]=ex[k]
            for a in outputs.values():a.flush()
            np.savez_compressed(stream/f'{model}_expectations.npz',**expectation)
            write_json(done,dict(status='GENERATED_PENDING_FULL_VALIDATION',rows=n,speed_order=order,J_order=48,seconds=time.time()-t))
            print(meta['key'],model,'generated',round(time.time()-t,1),'seconds',flush=True)


def terrain():
    from v9_6_run_nested import run_within_test_falsification
    labels,features,_=load_case('A')
    for level in ('A_T','A_S','A_ST'):
        for fold,tr,case,te,parents,info in level_segments(level):
            d=OUT/f'{level}_{fold}';dest=d/'terrain.json'
            if dest.exists():continue
            meta=json.loads((d/'metadata.json').read_text());cal=json.loads((d/'calibration.json').read_text())
            models=pd.read_pickle(d/'models.pkl.gz');old=Path(meta['legacy'])/'truth'
            truth=dict(speed_grid=speed_grid(),j_grid=j_grid(),speed_path=old/'truth_speed.npy',j_path=old/'truth_J.npy')
            truth.update(speed_rows=np.load(truth['speed_path'],mmap_mode='r'),j_rows=np.load(truth['j_path'],mmap_mode='r'))
            x=feature_block_v96(features,te,D02_T4_NAMES);u=labels.iloc[te].U.to_numpy(float)
            t=time.time();result=run_within_test_falsification(labels,features,level,fold,tr,te,parents,models,cal,truth,x,u)
            write_json(dest,dict(zip(['falsification','sanity','hierarchy','permutations','marginal'],result)))
            print('terrain',level,fold,round(time.time()-t,1),flush=True)


def controls():
    from v9_6_run_nested import write_cdfs_to_disk
    from v9_6_common import qa_deploy_v96
    d=OUT/'A_670';meta=json.loads((d/'metadata.json').read_text());z=np.load(d/'predictions.npz');old=Path(meta['legacy'])/'truth'
    truth=dict(speed_grid=speed_grid(),j_grid=j_grid(),j_rows=np.load(old/'truth_J.npy',mmap_mode='r'))
    q=np.maximum(z['rawQ'],1e-4);a=np.clip(z['rawa'],1e-4,1-1e-4);mu,j,s=qa_deploy_v96(q,a,z['U'])
    stream=d/'streams';stream.mkdir(exist_ok=True)
    write_cdfs_to_disk([j],[mu],[s],[1.],truth,stream,'Det_QA')
    np.savez_compressed(stream/'Det_QA_expectations.npz',mu=mu,J=j,S=s)
    labels,_,_=load_case('A');tr=labels.iloc[970*36:3650*36]
    ex={k:np.full(len(q),float(tr[c].mean())) for k,c in [('mu','U'),('J','J'),('S','S')]}
    write_cdfs_to_disk([ex['J']],[ex['mu']],[ex['S']],[1.],truth,stream,'Climatology')
    np.savez_compressed(stream/'Climatology_expectations.npz',**ex)
    print('A670 unchanged deterministic and fixed-moment controls restored',flush=True)


def sensitivity():
    bundle=json.loads((HERE/'V9_4_FROZEN_BUNDLE.json').read_text())
    cal=dict(a_const=bundle['constant_partition']['a_const_opt'])
    for tag,keys in [('QA',['logQ','logita']),('JS',['logJ','logS'])]:
        src=bundle[f'oof_covariance_{tag}'];cal[f'covariance_{tag}']=src['cov_matrix'];cal[f'cholesky_{tag}']=src['cholesky_L']
        for key in keys:cal[f'std_{key}']=src[f'std_{key}']
    for key in ('B','C'):
        d=OUT/key;meta=json.loads((d/'metadata.json').read_text());z=np.load(d/'predictions.npz');p={k:z[k] for k in ['logQ','logita','logJ','logS']};u=z['U'];nt=meta['time_count'];np_=len(meta['parents'])
        from v9_7_scores import truth_stream
        for model in ('QA_dyn','QA_const','Prob_JS_joint'):
            dest=d/f'sensitivity_{model}.json'
            if dest.exists():continue
            means={kind:np.zeros((np_,len(g))) for kind,g in [('speed',speed_grid()),('J',j_grid())]};t=time.time()
            for start in range(0,nt,4):
                a=start*np_;b=min(nt,start+4)*np_;pp={k:v[a:b] for k,v in p.items()};uu=u[a:b]
                fs,_=speed_cdf(pp,uu,cal,model,speed_grid(),order=(96,24) if model.startswith('Prob_JS') else 32)
                fj=gap_cdf_gpu(pp,uu,cal,model,j_grid())
                for kind,f in [('speed',fs),('J',fj)]:means[kind]+=f.reshape(-1,np_,f.shape[-1]).sum(axis=0)/nt
            score={}
            for kind,grid in [('speed',speed_grid()),('J',j_grid())]:
                gw=truth_stream(meta,kind).mean(axis=0,dtype=float);fw=means[kind]
                score[f'{kind}_crps']=float(np.trapezoid((fw-gw)**2+gw*(1-gw),grid,axis=1).mean())
                score[f'{kind}_w1']=float(np.trapezoid(abs(fw-gw),grid,axis=1).mean())
            write_json(dest,dict(scores=score,calibration=cal,rows=len(u),seconds=time.time()-t))
            print('sensitivity',key,model,round(time.time()-t,1),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','probe','window_probe','generate','terrain','controls','sensitivity']);args=ap.parse_args()
    globals()[args.action]()
