"""Resume persisted empirical curves without importing the model/GPU runtime."""
from pathlib import Path
import json,hashlib,gc
import numpy as np
import pandas as pd
from bounded_bootstrap import bootstrap
from cache_binding import inputs_identity, fingerprint, load_cache, bind_cache
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;V=R/'research/qa_framework_v9_20260916';OLD=V/'numerical_repair_20260921';O=R/'research/empirical_revision_20260921'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
all_scores=[];all_ci=[];audits=[]
for meta in json.loads((OLD/'manifest.json').read_text()):
 key=meta['key'];d=O/key;d.mkdir(exist_ok=True);nt=meta['time_count'];parents=meta['parents'];npar=len(parents)
 if not (d/'baseline_cdfs.npz').exists():
  assert key=='A_670'
  source=np.load(O/'B/baseline_cdfs.npz');np.savez_compressed(d/'baseline_cdfs.npz',**{k:source[k] for k in source.files})
  z=np.load(OLD/key/'predictions.npz');np.savez_compressed(d/'row_indices.npz',train_rows=np.arange(970*36,3650*36),test_rows=z['rows'],parents=parents)
 curves=np.load(d/'baseline_cdfs.npz');rows=np.load(d/'row_indices.npz');tr=rows['train_rows'];te=rows['test_rows'];e1='E1_J' in curves.files
 legacy=V/str(meta['legacy']).replace('\\','/').split('qa_framework_v9_20260916/')[-1]
 class Adapter:
  def speed_grid(self):return curves['speed_grid']
  def j_grid(self):return curves['J_grid']
  def truth_stream(self,m,kind):return np.load(legacy/'truth'/f'truth_{kind}.npy',mmap_mode='r')
  def get_stream(self,m,model,kind):
   if model.startswith('E'):return np.broadcast_to(curves[model+'_'+kind],(nt,npar,len(curves[kind+'_grid'])))
   return np.load(OLD/key/'streams'/f'{model}_{kind}.npy',mmap_mode='r')
 adapter=Adapter();models=['QA_dyn','Prob_JS_joint','E0']+(['E1'] if e1 else []);windows={};units=dict(time_index=te//36,parent_id=te%36)
 for kind in ('speed','J'):
  grid=curves[kind+'_grid'];truth=adapter.truth_stream(meta,kind);gw=truth.mean(axis=0,dtype=float);windows['truth_'+kind]=gw
  for model in models:
   fw=curves[model+'_'+kind] if model.startswith('E') else adapter.get_stream(meta,model,kind).mean(axis=0,dtype=float)
   value=float(np.trapezoid(fw*fw-2*fw*gw+gw,grid,axis=1).mean());windows[model+'_'+kind]=fw
   all_scores.append(dict(key=key,case=meta['case'],fold=meta['fold'],model=model,target=kind,crps=value,time_count=nt,parent_count=npar,row_count=len(te)))
   if not (d/'unit_scores.csv.gz').exists():
    vals=np.empty((nt,npar))
    for p in range(npar):
     for t in range(0,nt,16):
      g=np.asarray(truth[t:t+16,p],float);vals[t:t+16,p]=np.trapezoid(fw[p]**2-2*fw[p]*g+g,grid,axis=1)
    units[model+'_'+kind]=vals.ravel()
 if not (d/'unit_scores.csv.gz').exists():pd.DataFrame(units).to_csv(d/'unit_scores.csv.gz',index=False)
 np.savez_compressed(d/'window_cdfs.npz',**windows)
 ev=json.loads((OLD/key/'evaluation.json').read_text());seed=next(b['seed'] for b in ev['bootstrap'] if b['candidate']=='QA_dyn' and b['baseline']=='Prob_JS_joint')
 binding_inputs=inputs_identity(adapter,meta,tr,te,models[2:])
 for base in models[2:]:
  for L in (120,60,240):
   p=d/f'{base}_L{L}.json'
   binding=fingerprint(binding_inputs,base,seed,L)
   boot=load_cache(p,binding)
   if boot is None:
    print('computing',key,base,L,flush=True);boot=bind_cache(bootstrap(adapter,meta,'QA_dyn',base,seed,L,cpu=True),binding);dump(p,boot)
   for kind in ('speed','J'):
    point=next(x['crps'] for x in reversed(all_scores) if x['model']==base and x['target']==kind)-next(x['crps'] for x in reversed(all_scores) if x['model']=='QA_dyn' and x['target']==kind)
    assert abs(point-boot['delta_'+kind+'_point'])<2e-11
    lo=boot['delta_'+kind+'_ci_low'];hi=boot['delta_'+kind+'_ci_high']
    all_ci.append(dict(key=key,case=meta['case'],fold=meta['fold'],baseline=base,target=kind,delta=point,ci_low=lo,ci_high=hi,status='QA_LOWER' if lo>0 else 'EMPIRICAL_LOWER' if hi<0 else 'UNRESOLVED',block_length=L,n_boot=1000,time_count=nt,parent_count=npar,row_count=len(te),seed=seed,mapping_sha256=boot['mapping_sha256']))
 audits.append(dict(key=key,train_rows=len(tr),test_rows=len(te),E1_applicable=e1,no_test_leakage=bool(meta['case']!='A' or not np.intersect1d(tr,te).size),baseline_sha256=sha(d/'baseline_cdfs.npz'),row_indices_sha256=sha(d/'row_indices.npz')))
 pd.DataFrame(all_scores).to_csv(O/'EMPIRICAL_BASELINE_SCORES.csv',index=False);pd.DataFrame(all_ci).to_csv(O/'EMPIRICAL_BASELINE_PAIRED_CI.csv',index=False)
 print('finished',key,flush=True);gc.collect()
dump(O/'EMPIRICAL_BASELINE_AUDIT.json',dict(status='PASS',spec_sha256=sha(D/'EMPIRICAL_BASELINE_SPEC.md'),configurations=audits,configuration_count=len(audits),models_refitted=0))
