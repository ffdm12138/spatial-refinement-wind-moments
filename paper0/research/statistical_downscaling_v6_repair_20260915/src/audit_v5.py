"""Recompute historical V5 evaluation without training or changing V5 artifacts."""
from __future__ import annotations
import ast, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import empirical_crps, paired_gain, block_indices
from contracts import config, aggregate_cv, validate_keys

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'statistical_downscaling_v5_qprob_20260915'
RUN=OLD/'runs/v5_qprob_01'
OUT=ROOT/'experiments/00_metric_repair'
SOURCES={}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def record(p):
    p=Path(p)
    SOURCES[str(p.resolve())]={'sha256':sha(p),'bytes':p.stat().st_size}
    return p


def readp(relative):
    return pd.read_pickle(record(RUN/relative),compression='gzip')


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def matrix(frame,value,times,parents):
    if frame.duplicated(['time_index','parent_id']).any(): raise ValueError('Duplicate matrix keys')
    result=frame.pivot(index='time_index',columns='parent_id',values=value).reindex(index=times,columns=parents)
    if result.isna().any().any(): raise ValueError('Missing/nonfinite matrix key')
    return result.to_numpy(float)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    record(OLD/'src/run_v5.py')
    for p in sorted((ROOT/'src').glob('*.py')): record(p)
    record(ROOT/'configs/experiment.json')
    result=subprocess.run([sys.executable,str(ROOT/'tests/test_repair.py')],text=True,capture_output=True)
    (OUT/'TEST_OUTPUT.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    write_json(OUT/'KERNEL_TEST_RECEIPT.json',{'exit_code':result.returncode,'scope':'17 kernel/contract tests, not full pipeline acceptance','test_source_sha256':sha(ROOT/'tests/test_repair.py')})
    if result.returncode: raise RuntimeError('Kernel tests failed; no evaluation')
    cfg=config(); parents=cfg['core_parent_ids']; bootstrap=cfg['bootstrap']
    atrain=readp('labels/A_labels.pkl.gz')
    atrain=atrain[atrain.split=='train'].copy(); validate_keys(atrain,'A','train',parents)
    trainj=atrain.J.to_numpy(float); sample=np.sort(trainj); n=len(sample)
    half_pair=np.sum((2*np.arange(n)-n+1)*sample)/(n*n)
    baseline_rows=[]; vector_rows=[]; index_rows=[]; ci_rows=[]; selection=None
    for case,split in [('A','validation'),('B','test'),('C','test')]:
        lab=readp(f'labels/{case}_labels.pkl.gz'); lab=lab[lab.split==split].copy()
        validate_keys(lab,case,split,parents)
        for name,values in [('vector_gap',lab.evec),('J_identity_error',np.abs(lab.J-(lab.mu-lab.V))),('Q_identity_error',np.abs(lab.Q-(lab.S**2+2*lab.V*lab.J+lab.J**2)))]:
            vector_rows.append(dict(case=case,split=split,support='CORE16',quantity=name,n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)),p99=float(np.quantile(values,.99)),maximum=float(np.max(values))))
        corrected=empirical_crps(lab.J.to_numpy(float),trainj)
        baseline_rows.append(dict(case=case,split=split,baseline='V5_B1_GLOBAL_J_EMPIRICAL',n=len(lab),historical_formula_crps=float(corrected.mean()+half_pair/2),corrected_crps=float(corrected.mean()),overcount=float(half_pair/2)))
        export=lab[['case','split','time_index','parent_id','J']].copy(); export['corrected_baseline_crps']=corrected
        export.to_csv(OUT/f'J_BASELINE_ROWS_{case}.csv.gz',index=False,compression='gzip')
        if case=='A':
            sel=pd.read_csv(record(RUN/'tables/A_SELECTION_V5.csv'))
            sel['corrected_j_ratio']=sel.j_crps/corrected.mean()
            sel['corrected_eligible']=(sel.speed_crps_vs_B1<=1)&(sel.corrected_j_ratio<=1)
            sel['corrected_score']=.5*(sel.speed_crps_vs_B1+sel.corrected_j_ratio)
            sel.to_csv(OUT/'A_SELECTION_BASELINE_ONLY_CORRECTION.csv',index=False)
            pool=sel[sel.corrected_eligible].copy()
            if pool.empty: pool=sel.copy()
            pool=pool.sort_values(['corrected_score','category_order','moment_id','calibrator_id'])
            selection={'old_eligible':int(sel.eligible_b1.sum()),'corrected_eligible':int(sel.corrected_eligible.sum()),'winner_baseline_correction_only':str(pool.iloc[0].moment_id),'calibrator':str(pool.iloc[0].calibrator_id),'scope':'Arithmetic audit of stored A scores only; calibration/CV still invalid, no model refit or new selection certification'}
            continue
        pred=readp(f'logs/T12_PRED_{case}.pkl.gz'); pred=pred[pred.split==split].copy()
        if pred.duplicated(['case','split','model_id','time_index','parent_id']).any(): raise ValueError('Duplicate model prediction keys')
        times=np.sort(lab.time_index.unique())
        groups=dict(tuple(pred.groupby('model_id',sort=True)))
        for mid,frame in groups.items(): validate_keys(frame,case,split,parents)
        indices={}
        for block in bootstrap['blocks']:
            seed=bootstrap['seed']+1009*len(times)+9176*block+sum((i+1)*ord(ch) for i,ch in enumerate(case))
            idx=block_indices(len(times),block,bootstrap['replicates'],seed); indices[block]=idx
            index_rows.append(dict(case=case,block=block,seed=seed,n_times=len(times),reps=len(idx),sha256=hashlib.sha256(idx.tobytes()).hexdigest(),support='CORE16',method='noncircular moving block; all 16 parents together'))
        for target in ['J','S']:
            truth=matrix(lab,target,times,parents); baseline=matrix(groups['B4'],target+'_hat',times,parents)
            candidates={mid:matrix(frame,target+'_hat',times,parents) for mid,frame in groups.items()}
            candidates['B0']=np.zeros_like(truth); candidates['B1']=np.full_like(truth,atrain[target].mean())
            for mid,p in candidates.items():
                for block,idx in indices.items():
                    metric=paired_gain(p,baseline,truth,idx)
                    ci_rows.append(dict(case=case,split=split,support='CORE16',target=target,model_id=mid,baseline='V5_B4',block=block,comparison_role='BASELINE' if mid=='B4' else 'RETROSPECTIVE_COMPARISON',**metric))
            print(f'{case} {target}: corrected {len(candidates)} models x 3 blocks',flush=True)
        del pred,groups,candidates,indices
    pd.DataFrame(ci_rows).to_csv(OUT/'MOMENT_GAIN_CI_CORRECTED.csv',index=False)
    pd.DataFrame(index_rows).to_csv(OUT/'BOOTSTRAP_INDEX_MANIFEST.csv',index=False)
    pd.DataFrame(baseline_rows).to_csv(OUT/'EMPIRICAL_J_BASELINE_CORRECTED.csv',index=False)
    pd.DataFrame(vector_rows).to_csv(OUT/'CORE16_IDENTITY_AUDIT.csv',index=False)
    cv=pd.read_csv(record(RUN/'tables/NEW_HEAD_CV.csv'))
    rank=aggregate_cv(cv); rank.to_csv(OUT/'NEW_HEAD_CV_THREE_FOLD_MEANS.csv',index=False)
    change=[]
    for head,g in rank.groupby('head_id'):
        best=g[g.eligible].sort_values(['mean_score','config_id']).iloc[0]
        old=cv[(cv.head_id==head)&cv.success].sort_values(['score','config_id']).iloc[0]
        change.append(dict(head_id=head,old_config_id=int(old.config_id),correct_config_id=int(best.config_id),correct_mean_score=float(best.mean_score),requires_refit=bool(best.config_id!=old.config_id)))
    pd.DataFrame(change).to_csv(OUT/'CV_SELECTION_CHANGES.csv',index=False)
    write_json(OUT/'A_SELECTION_AUDIT.json',selection)
    # Claims are deliberately narrow. No old overall PASS/FAIL is regenerated.
    write_json(OUT/'REPAIR_RECEIPT.json',{'status':'HISTORICAL_METRIC_RECOMPUTATION_COMPLETE_NOT_NEW_MODEL_ACCEPTANCE','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'ci_rows':len(ci_rows),'scope':'CORE16 cached predictions; J/S errors use labels; empirical baseline corrected; new-head CV means','not_completed':['full raw ALL36 audit','correct CV refits','forward calibrator refits','saved-model label-shield subprocess acceptance','full-window-CDF evaluation','new-model training and validation'],'git_commit':None,'git_reason':'workspace has no .git; source SHA256 used','source_manifest':'SOURCE_MANIFEST.json'})
    write_json(OUT/'SOURCE_MANIFEST.json',SOURCES)
    print('Historical metric correction complete; new experiment remains pending.',flush=True)


if __name__=='__main__': main()
