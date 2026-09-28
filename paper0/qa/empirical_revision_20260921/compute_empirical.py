"""Fixed-contract empirical controls using existing reference/prediction streams."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
V=R/'research/qa_framework_v9_20260916';sys.path.insert(0,str(V))
import v9_7_scores as scoring
from v9_7_repair import OUT,write_json
from v9_6_common import level_segments
from window_evaluation import speed_grid,j_grid
from bounded_bootstrap import bootstrap as bounded_bootstrap
from cache_binding import inputs_identity, fingerprint, load_cache, bind_cache
O=R/'research/empirical_revision_20260921';O.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def run():
 manifest=json.loads((OUT/'manifest.json').read_text());bykey={m['key']:m for m in manifest}
 splits={}
 for level in ('A_T','A_S','A_ST'):
  for fold,tr,case,te,parents,info in level_segments(level):splits[level+'_'+fold]=(tr,te)
 # Existing spatial-fold truth shards partition ALL A training parents.
 truth_shards=[m for m in manifest if m['level']=='A_S']
 train_maps={k:{} for k in ('speed','J')}
 for kind in train_maps:
  for m in truth_shards:
   arr=scoring.truth_stream(m,kind)
   assert arr.shape[0]==2680
   for j,p in enumerate(m['parents']):train_maps[kind][p]=arr[:,j]
 assert all(len(x)==36 for x in train_maps.values())
 all_scores=[];all_ci=[];audits=[]
 original_get=scoring.get_stream
 for idx,meta in enumerate(manifest):
  key=meta['key'];dest=O/key;dest.mkdir(exist_ok=True)
  z=np.load(OUT/key/'predictions.npz');te=z['rows'];nt=meta['time_count'];parents=meta['parents'];np_=len(parents)
  tr=splits[key][0] if key in splits else np.arange(970*36,3650*36)
  if key in splits:assert np.array_equal(te,splits[key][1])
  assert tr.min()>=970*36 and tr.max()<3650*36
  if meta['case']=='A':assert not np.intersect1d(tr,te).size
  e1=meta['case']!='C' and all(p in set(tr%36) for p in parents)
  models=['QA_dyn','Prob_JS_joint','E0']+(['E1'] if e1 else [])
  np.savez_compressed(dest/'row_indices.npz',train_rows=tr,test_rows=te,parents=parents)
  curves={};units={'time_index':te//36,'parent_id':te%36};windows={}
  for kind,grid in [('speed',speed_grid()),('J',j_grid())]:
   local={p:np.asarray(train_maps[kind][int(p)][tr[tr%36==p]//36-970],float).mean(axis=0) for p in np.unique(tr%36)}
   sizes=np.array([(tr%36==p).sum() for p in local])
   global_cdf=np.average(np.stack(list(local.values())),weights=sizes,axis=0)
   curves['E0',kind]=np.tile(global_cdf,(np_,1))
   if e1:curves['E1',kind]=np.stack([local[p] for p in parents])
   truth=scoring.truth_stream(meta,kind);gw=truth.mean(axis=0,dtype=float)
   windows['truth_'+kind]=gw
   for model in models:
    fw=curves[model,kind] if model.startswith('E') else original_get(meta,model,kind).mean(axis=0,dtype=float)
    score=np.trapezoid((fw-gw)**2+gw*(1-gw),grid,axis=1)
    row=np.empty((nt,np_))
    for p in range(np_):
     for t in range(0,nt,128):
      g=np.asarray(truth[t:t+128,p],float)
      row[t:t+128,p]=np.trapezoid(fw[p]**2-2*fw[p]*g+g,grid,axis=1)
    assert abs(row.mean()-score.mean())<1e-10
    units[model+'_'+kind]=row.ravel();windows[model+'_'+kind]=fw
    all_scores.append(dict(key=key,case=meta['case'],fold=meta['fold'],model=model,target=kind,crps=score.mean(),time_count=nt,parent_count=np_,row_count=len(te)))
   # Verify frozen QA/JS score consistency before any inference.
   ref=json.loads((OUT/key/'evaluation.json').read_text())['scores']
   for model in models[:2]:
    actual=next(x['crps'] for x in reversed(all_scores) if x['model']==model and x['target']==kind)
    assert abs(actual-ref[model][kind+'_crps'])<1e-12
  np.savez_compressed(dest/'baseline_cdfs.npz',**{m+'_'+k:a for (m,k),a in curves.items()},speed_grid=speed_grid(),J_grid=j_grid())
  np.savez_compressed(dest/'window_cdfs.npz',**windows)
  pd.DataFrame(units).to_csv(dest/'unit_scores.csv.gz',index=False)
  ev=json.loads((OUT/key/'evaluation.json').read_text())
  seed=next(b['seed'] for b in ev['bootstrap'] if b['candidate']=='QA_dyn' and b['baseline']=='Prob_JS_joint')
  def get(m,model,kind):
   if model.startswith('E'):return np.broadcast_to(curves[model,kind],(nt,np_,curves[model,kind].shape[1]))
   return original_get(m,model,kind)
  scoring.get_stream=get
  binding_inputs=inputs_identity(scoring,meta,tr,te,models[2:])
  for baseline in models[2:]:
   for length in (120,60,240):
    cache=dest/f'{baseline}_L{length}.json'
    binding=fingerprint(binding_inputs,baseline,seed,length)
    boot=load_cache(cache,binding)
    if boot is None:
     boot=bind_cache(bounded_bootstrap(scoring,meta,'QA_dyn',baseline,seed,length),binding);write_json(cache,boot)
    for kind in ('speed','J'):
     expected=next(x['crps'] for x in reversed(all_scores) if x['model']==baseline and x['target']==kind)-next(x['crps'] for x in reversed(all_scores) if x['model']=='QA_dyn' and x['target']==kind)
     assert abs(expected-boot[f'delta_{kind}_point'])<2e-12
     lo=boot[f'delta_{kind}_ci_low'];hi=boot[f'delta_{kind}_ci_high']
     all_ci.append(dict(key=key,case=meta['case'],fold=meta['fold'],baseline=baseline,target=kind,delta=expected,ci_low=lo,ci_high=hi,status='QA_LOWER' if lo>0 else 'EMPIRICAL_LOWER' if hi<0 else 'UNRESOLVED',block_length=length,n_boot=1000,time_count=nt,parent_count=np_,row_count=len(te),seed=seed,mapping_sha256=boot['mapping_sha256']))
    print(key,baseline,length,boot['delta_speed_point'],boot['delta_J_point'],flush=True)
  audits.append(dict(key=key,train_rows=len(tr),test_rows=len(te),E1_applicable=e1,E1_reason='same_parent_training_available' if e1 else 'no_training_parent_correspondence',no_test_leakage=True,baseline_sha256=sha(dest/'baseline_cdfs.npz'),row_indices_sha256=sha(dest/'row_indices.npz')))
  pd.DataFrame(all_scores).to_csv(O/'EMPIRICAL_BASELINE_SCORES.csv',index=False)
  pd.DataFrame(all_ci).to_csv(O/'EMPIRICAL_BASELINE_PAIRED_CI.csv',index=False)
  write_json(O/'EMPIRICAL_BASELINE_AUDIT.json',dict(status='RUNNING',spec_sha256=sha(D/'EMPIRICAL_BASELINE_SPEC.md'),configurations=audits))
 scoring.get_stream=original_get
 write_json(O/'EMPIRICAL_BASELINE_AUDIT.json',dict(status='PASS',spec_sha256=sha(D/'EMPIRICAL_BASELINE_SPEC.md'),configurations=audits,configuration_count=len(audits),models_refitted=0))
if __name__=='__main__':run()
