"""Independent score identities, direct resamples and raw-CDF spot checks."""
from compute_empirical import *
from v9_6_common import load_case
from window_evaluation import empirical_rows
def empirical_crps(train,target):
 x=np.sort(np.asarray(train,float));y=np.asarray(target,float);n=len(x)
 cs=np.r_[0,np.cumsum(x)];ix=np.searchsorted(x,y,side='right')
 absolute=(y*ix-cs[ix]+cs[-1]-cs[ix]-y*(n-ix))/n
 halfpair=np.dot(2*np.arange(n)-n+1,x)/(n*n)
 return float(absolute.mean()-halfpair)
def verify():
 audit=json.loads((O/'EMPIRICAL_BASELINE_AUDIT.json').read_text());assert audit['status']=='PASS'
 scores=pd.read_csv(O/'EMPIRICAL_BASELINE_SCORES.csv');ci=pd.read_csv(O/'EMPIRICAL_BASELINE_PAIRED_CI.csv')
 manifest=json.loads((OUT/'manifest.json').read_text());records=[];source=[]
 for meta in manifest:
  key=meta['key'];d=O/key;z=np.load(d/'row_indices.npz');tr=z['train_rows'];te=z['test_rows'];parents=z['parents'];nt=meta['time_count'];npar=len(parents)
  base=np.load(d/'baseline_cdfs.npz');win=np.load(d/'window_cdfs.npz');units=pd.read_csv(d/'unit_scores.csv.gz')
  assert np.array_equal(units.time_index.to_numpy(),te//36) and np.array_equal(units.parent_id.to_numpy(),te%36)
  if meta['case']=='A':assert not np.intersect1d(tr,te).size
  labels,_,speed=load_case(meta['case'])
  refs={k:scoring.truth_stream(meta,k) for k in ('speed','J')}
  ts=np.unique([0,nt//2,nt-1]);ps=np.unique([0,npar//2,npar-1]);selected=(ts[:,None]*npar+ps).ravel()
  actual=empirical_rows(speed.reshape(-1,729)[te[selected]],speed_grid())
  stored=refs['speed'].reshape(-1,len(speed_grid()))[selected]
  assert np.array_equal(actual,stored)
  assert np.array_equal((labels.iloc[te[selected]].J.to_numpy()[:,None]<=j_grid()).astype(np.float32),refs['J'].reshape(-1,len(j_grid()))[selected])
  maxdirect=0
  for baseline in ('E0','E1'):
   if baseline+'_J' not in base.files:continue
   boot=json.loads((d/f'{baseline}_L120.json').read_text());L=min(120,nt);K=(nt+L-1)//L
   starts=np.random.default_rng(boot['seed']).integers(0,nt,(1000,K),dtype=np.int64)
   assert hashlib.sha256(starts.tobytes()).hexdigest()==boot['mapping_sha256']
   for target,grid in [('speed',speed_grid()),('J',j_grid())]:
    for model in ('QA_dyn','Prob_JS_joint',baseline):
     f=win[model+'_'+target];g=win['truth_'+target]
     value=np.trapezoid(f*f-2*f*g+g,grid,axis=1).mean()
     tab=scores[(scores.key==key)&(scores.model==model)&(scores.target==target)].iloc[0].crps
     assert abs(value-tab)<2e-12 and abs(units[model+'_'+target].mean()-tab)<2e-12
    qa=scoring.get_stream(meta,'QA_dyn',target);truth=refs[target];fixed=base[baseline+'_'+target]
    for b in (0,499,999):
     ix=((starts[b,:,None]+np.arange(L))%nt).ravel()[:nt]
     delta=0.
     for p in range(npar):
      f=np.asarray(qa[ix,p],float).mean(axis=0);g=np.asarray(truth[ix,p],float).mean(axis=0)
      delta+=np.trapezoid((fixed[p]-g)**2-(f-g)**2,grid)/npar
     err=abs(delta-boot['draw_delta_'+target][b]);maxdirect=max(maxdirect,err);assert err<2e-11,(key,target,b,err)
   if key in ('A_670','B','C'):
    a,_,_=load_case('A');vals=[]
    for p in parents:
     training=a.iloc[tr if baseline=='E0' else tr[tr%36==p]].J.to_numpy()
     target=labels.iloc[te[te%36==p]].J.to_numpy();vals.append(empirical_crps(training,target))
    continuous=float(np.mean(vals));discrete=scores[(scores.key==key)&(scores.model==baseline)&(scores.target=='J')].iloc[0].crps
    source.append(dict(key=key,baseline=baseline,continuous_J_crps=continuous,production_grid_J_crps=discrete,grid_minus_continuous=discrete-continuous))
  records.append(dict(key=key,raw_reference_spotcheck_rows=len(selected),direct_draw_max_error=maxdirect,score_and_row_identity='PASS'))
  print('verified',key,maxdirect,flush=True)
 pd.DataFrame(source).to_csv(O/'EMPIRICAL_CONTINUOUS_CROSSCHECK.csv',index=False)
 write_json(D/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',configurations=records,continuous_crosscheck=source,scope='All stored primary score identities; 3 independent direct bootstrap draws per metric/baseline/configuration; raw-reference deterministic spot checks. Full bootstrap maps and draws retained.'))
if __name__=='__main__':verify()
