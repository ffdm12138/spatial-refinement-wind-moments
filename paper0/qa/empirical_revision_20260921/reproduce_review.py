"""Historical score-level replay of an existing review archive (numpy/pandas).

This script replays receipts for the already-created historical bundle; it does
not build or refresh any package. Run from an extracted archive with
``python qa/empirical_revision_20260921/reproduce_review.py --quick`` to check
full-grid scores and empirical training reconstruction. Without ``--quick`` it
also regenerates all draws for empirical-versus-QA primary (L=120) bootstrap comparisons on CPU, with the
exact frozen maps. No raw WRF, GPU or model refit is required.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'research/empirical_revision_20260921';V=R/'research/qa_framework_v9_20260916';OLD=V/'numerical_repair_20260921'
def integral_weights(g):
 return np.r_[(g[1]-g[0])/2,(g[2:]-g[:-2])/2,(g[-1]-g[-2])/2]
def legacy(meta):return V/str(meta['legacy']).replace('\\','/').split('qa_framework_v9_20260916/')[-1]
def main():
 quick='--quick' in sys.argv;manifest=json.loads((OLD/'manifest.json').read_text());score=pd.read_csv(O/'EMPIRICAL_BASELINE_SCORES.csv');records=[]
 # Disjoint A_S truth shards are a complete training reference, not test borrowing.
 train={kind:{} for kind in ('speed','J')}
 for m in manifest:
  if m['level']=='A_S':
   for kind in train:
    a=np.load(legacy(m)/'truth'/f'truth_{kind}.npy',mmap_mode='r')
    for k,p in enumerate(m['parents']):train[kind][p]=a[:,k]
 for meta in manifest:
  key=meta['key'];d=O/key;rows=np.load(d/'row_indices.npz');tr=rows['train_rows'];nt=meta['time_count'];np_=len(meta['parents']);curves=np.load(d/'baseline_cdfs.npz')
  assert tr.min()>=970*36 and tr.max()<3650*36
  if meta['case']=='A':assert not np.intersect1d(tr,rows['test_rows']).size
  maxscore=maxbase=maxci=0.
  for kind in ('speed','J'):
   grid=curves[kind+'_grid'];weights=integral_weights(grid);truth=np.load(legacy(meta)/'truth'/f'truth_{kind}.npy',mmap_mode='r');gw=truth.mean(axis=0,dtype=float)
   per={p:np.asarray(train[kind][p][tr[tr%36==p]//36-970],float).mean(axis=0) for p in np.unique(tr%36)}
   e0=np.average(np.stack(list(per.values())),weights=[(tr%36==p).sum() for p in per],axis=0)
   maxbase=max(maxbase,float(abs(curves['E0_'+kind]-e0).max()))
   if 'E1_'+kind in curves.files:maxbase=max(maxbase,float(abs(curves['E1_'+kind]-np.stack([per[p] for p in meta['parents']])).max()))
   assert maxbase<1e-12
   streams={m:np.load(OLD/key/'streams'/f'{m}_{kind}.npy',mmap_mode='r') for m in ('QA_dyn','Prob_JS_joint')}
   models=list(streams)+['E0']+(['E1'] if 'E1_'+kind in curves.files else [])
   for model in models:
    fw=curves[model+'_'+kind] if model.startswith('E') else streams[model].mean(axis=0,dtype=float)
    val=((fw*fw-2*fw*gw+gw)*weights).sum(axis=1).mean();reference=score[(score.key==key)&(score.model==model)&(score.target==kind)].iloc[0].crps
    maxscore=max(maxscore,abs(val-reference));assert maxscore<2e-11
   if not quick:
    for base in models[2:]:
     boot=json.loads((d/f'{base}_L120.json').read_text());L=min(120,nt);K=(nt+L-1)//L;rem=nt-(K-1)*L
     starts=np.random.default_rng(boot['seed']).integers(0,nt,(1000,K),dtype=np.int64);lengths=np.full(K,L);lengths[-1]=rem
     assert hashlib.sha256(starts.tobytes()).hexdigest()==boot['mapping_sha256']
     delta=np.zeros(1000)
     for p in range(np_):
      for data,sign in [(curves[base+'_'+kind][p][None,:]-np.asarray(truth[:,p],float),1),(np.asarray(streams['QA_dyn'][:,p],float)-truth[:,p],-1)]:
       prefix=np.vstack([np.zeros((1,len(grid))),np.cumsum(np.vstack([data,data[:L]]),axis=0)])
       for b in range(0,1000,20):
        ss=starts[b:b+20];avg=(prefix[ss+lengths]-prefix[ss]).sum(axis=1)/nt
        delta[b:b+20]+=sign*((avg*avg)*weights).sum(axis=1)/np_
     error=float(abs(delta-np.asarray(boot['draw_delta_'+kind])).max());maxci=max(maxci,error);assert error<2e-11
  records.append(dict(key=key,score_max_error=maxscore,empirical_rebuild_max_error=maxbase,draw_max_error=None if quick else maxci));print(records[-1],flush=True)
 output=R/'review_reproduction.json';output.write_text(json.dumps(dict(status='PASS',mode='full_scores_and_empirical_rebuild' if quick else 'full_scores_empirical_rebuild_and_all_empirical_primary_draws',records=records),indent=2)+'\n')
if __name__=='__main__':main()
