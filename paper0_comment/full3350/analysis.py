"""Reproduce full3350 WRF--LES Comment diagnostics from raw released PT tensors.
No fitting, GPU, pickle execution, or modification of ancestral scientific files.
Commands run one case or one bounded scoring shard, not a background worker.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,zipfile,struct,hashlib,importlib.util,time,ast,gc,math,shutil
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import gammaln,gammainc
from scipy.stats import spearmanr
SOURCE=Path(__file__).resolve().parent
D=Path(os.environ.get('COMMENT_WORKDIR', str(SOURCE/'generated_run'))).resolve()
S=SOURCE
QA=D/'qa'
RAW_ROOT=Path(os.environ.get('COMMENT_RAW_ROOT', str(SOURCE/'raw'))).resolve()
SPATIAL_CACHE=os.environ.get('COMMENT_SPATIAL_CACHE', '')
CASES={'A':('yakou20150924','case_A_yakou0924'),'B':('yakou20151201','case_B_yakou1201'),'C':('ebao20151201','case_C_ebao1201')}
T=3350;C=36;N=729;START=970;STOP=4320

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def jsonable(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False,default=jsonable)
def csvout(p,rows):
 with Path(p).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mod(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def blocks(a,b=27):
 n=162//b;return a.reshape(-1,n,b,n,b).transpose(0,1,3,2,4).reshape(-1,n*n,b*b)
def tensor(case,field):
 suffix=CASES[case][0]; name=field+'_'+suffix+'.pt'
 choices=[RAW_ROOT/name, RAW_ROOT/('1min_'+suffix)/name, RAW_ROOT/case/name]
 found=[p for p in choices if p.is_file()]
 if not found:raise FileNotFoundError('Released PT asset '+name+' not under '+str(RAW_ROOT))
 p=found[0]
 z=zipfile.ZipFile(p);entries=[i for i in z.infolist() if i.filename.endswith('/data/0')];assert len(entries)==1
 i=entries[0];shape=(162,162) if field=='HGT_d05' else (4320,162,162)
 assert i.compress_type==0 and i.file_size==np.prod(shape)*4
 byteorder=[n for n in z.namelist() if n.endswith('/byteorder')];assert len(byteorder)==1 and z.read(byteorder[0])==b'little'
 with p.open('rb') as f:
  f.seek(i.header_offset);h=f.read(30);assert h[:4]==b'PK\x03\x04';nl,xl=struct.unpack_from('<HH',h,26);offset=i.header_offset+30+nl+xl
 return np.memmap(p,dtype='<f4',mode='r',offset=offset,shape=shape),p

def moments(u,v):
 mu=u.mean(-1);mv=v.mean(-1);sv=np.hypot(mu,mv);s=np.hypot(u,v);ss=s.mean(-1);j=ss-sv
 du=u-mu[...,None];dv=v-mv[...,None];cu=(du*du).mean(-1);cv=(dv*dv).mean(-1);cuv=(du*dv).mean(-1);q=cu+cv
 ds=s-ss[...,None];var=(ds*ds).mean(-1);m3=(ds**3).mean(-1);r3=(s**3).mean(-1)
 valid=sv>=1;ex=np.divide(mu,sv,out=np.zeros_like(sv),where=sv>0);ey=np.divide(mv,sv,out=np.zeros_like(sv),where=sv>0)
 vp=ex*ex*cu+2*ex*ey*cuv+ey*ey*cv;vt=q-vp
 assert np.all(q>1e-12) and np.all(r3>1e-12)
 assert np.allclose(q,var+2*sv*j+j*j,rtol=1e-10,atol=1e-10)
 assert np.allclose(r3,ss**3+3*ss*var+m3,rtol=1e-10,atol=1e-10)
 assert np.min(j)>-1e-12
 a=dict(u_mean=mu,v_mean=mv,S_v=sv,S_s=ss,J=j,sigma_s2=var,sigma_s=np.sqrt(var),Q=q,C_uu=cu,C_vv=cv,C_uv=cuv,mu3=m3,R3=r3,var_parallel=vp,var_perp=vt)
 return a

def weight(mode):
 # Same half-pixel coordinates and cubic convolution a=-0.75 as align_corners=False.
 w=np.zeros((162,6));q=(np.arange(162)+.5)/27-.5
 for i,x in enumerate(q):
  k=int(np.floor(x))
  if mode=='bilinear':idx=[k,k+1];ww=[1-(x-k),x-k]
  else:
   idx=list(range(k-1,k+3));r=np.abs(x-np.array(idx));a=-.75
   ww=np.where(r<=1,(a+2)*r**3-(a+3)*r*r+1,np.where(r<2,a*r**3-5*a*r*r+8*a*r-4*a,0))
  for j,z in zip(idx,ww):w[i,min(5,max(0,j))]+=z
 assert np.allclose(w.sum(1),1,atol=1e-14)
 return w

def init():
 """Create an empty workdir after environment/input checks; no training."""
 if D.exists():raise FileExistsError('Output already exists: '+str(D))
 D.mkdir(parents=True);QA.mkdir()
 for m in ['bilinear','bicubic']:
  w=weight(m)
  assert np.allclose(w.sum(1),1,atol=1e-14)
 sample=np.array([[[4.,6.]],[[2.,8.]]]); v=np.zeros_like(sample); mm=moments(sample,v)
 assert np.allclose(mm['J'],0) and np.allclose(mm['sigma_s'].ravel(),[1,3])
 source_files=[]
 for name in ['diagnostic_model.json','diagnostic_predictor.py','affine_index36.py','laws.py']:
  p=S/name;assert p.is_file(),p
  source_files.append(dict(path=p.name,sha256=digest(p),bytes=p.stat().st_size))
 save(D/'PROTOCOL.json',dict(schema='npj.comment.full3350.public_reproduction/v1',
  case_order=list(CASES),window=[START,STOP],times=T,cells=C,fine_points=N,
  conditional_group='4 < native coarse speed < 6 m/s',raw_directory=str(RAW_ROOT),
  frozen_predictor='MU_JS_SD_QA; A-only fitted, B/C retrospective',
  original_data_release='raw-data-multicase-v1',new_fits=0,source_files=source_files))
 save(QA/'QUALIFICATION.json',{'status':'PASS','numerical_identity_and_interpolation':True,'no_fit':True})
 print('READY',D)

def reference(case):
 assert (QA/'QUALIFICATION.json').exists();out=D/(case+'_reference.npz');assert not out.exists()
 t0=time.monotonic();u,up=tensor(case,'U10_d05');v,vp=tensor(case,'V10_d05');cu,cp=tensor(case,'U10_d02');cv,dp=tensor(case,'V10_d02');h,hp=tensor(case,'HGT_d05')
 # Bind entire physical sources and existing moments; streaming hashes do not load full arrays.
 sources=[{'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size} for p in [up,vp,cp,dp,hp]]
 arrays={};interp={};budgets={b:np.empty((T,3)) for b in [3,9,27]};wts={m:weight(m) for m in ['bilinear','bicubic']};repmax=0.;identitymax=0.
 for a in range(START,STOP,24):
  b=min(a+24,STOP);sl=slice(a-START,b-START);uu=np.asarray(u[a:b],dtype=float);vv=np.asarray(v[a:b],dtype=float)
  for x in [uu,vv]:assert np.isfinite(x).all()
  ub=blocks(uu);vb=blocks(vv);m=moments(ub,vb)
  utc=blocks(np.asarray(cu[a:b],dtype=float));vtc=blocks(np.asarray(cv[a:b],dtype=float));ur=utc[:,:,0];vr=vtc[:,:,0]
  repmax=max(repmax,float(np.abs(utc-ur[...,None]).max()),float(np.abs(vtc-vr[...,None]).max()));assert repmax<=1e-6
  m.update(u02p=ur,v02p=vr,S_c=np.hypot(ur,vr))
  m['delta_mag']=m['S_v']-m['S_c'];m['E_vec']=np.hypot(m['u_mean']-ur,m['v_mean']-vr)
  arct=np.arctan2(m['v_mean'],m['u_mean'])-np.arctan2(vr,ur);m['direction_difference']=np.abs((arct+np.pi)%(2*np.pi)-np.pi)*180/np.pi
  m['D2_x']=((np.diff(ub.reshape(-1,36,27,27),axis=-1))**2+(np.diff(vb.reshape(-1,36,27,27),axis=-1))**2).mean((-2,-1))
  m['D2_y']=((np.diff(ub.reshape(-1,36,27,27),axis=-2))**2+(np.diff(vb.reshape(-1,36,27,27),axis=-2))**2).mean((-2,-1))
  for k,value in m.items():arrays.setdefault(k,np.empty((T,36)))[sl]=value
  total=uu.var((1,2))+vv.var((1,2))
  for scale in [3,9,27]:
   us=blocks(uu,scale);vs=blocks(vv,scale);um=us.mean(-1);vm=vs.mean(-1)
   within=(us.var(-1)+vs.var(-1)).mean(-1);between=um.var(-1)+vm.var(-1)
   assert np.allclose(within+between,total,atol=1e-10,rtol=1e-10)
   budgets[scale][sl]=np.column_stack([total,within,between])
  for mode,w in wts.items():
   ui=w@ur.reshape(-1,6,6)@w.T;vi=w@vr.reshape(-1,6,6)@w.T
   ui=blocks(ui);vi=blocks(vi);ubar=ui.mean(-1);vbar=vi.mean(-1)
   for corrected in [False,True]:
    ut=ui+(ur-ubar)[...,None] if corrected else ui;vt=vi+(vr-vbar)[...,None] if corrected else vi
    # Total interpolation Q can be zero on flat input; moments evaluated directly.
    mmu=ut.mean(-1);mmv=vt.mean(-1);speed=np.hypot(ut,vt);mean=speed.mean(-1);sv=np.hypot(mmu,mmv);sd=speed.std(-1)
    tag=mode+('_M' if corrected else '')
    for k,value in dict(S_s=mean,J=mean-sv,sigma_s=sd,Q=ut.var(-1)+vt.var(-1),component_residual=np.maximum(np.abs(mmu-ur),np.abs(mmv-vr)),vector_residual=np.hypot(mmu-ur,mmv-vr)).items():
     interp.setdefault(tag+'_'+k,np.empty((T,36)))[sl]=value
   assert np.allclose(interp[mode+'_Q'][sl],interp[mode+'_M_Q'][sl],atol=1e-10,rtol=1e-10)
  identitymax=max(identitymax,float(np.max(np.abs(m['R3']-(m['S_s']**3+3*m['S_s']*m['sigma_s2']+m['mu3'])))))
 # Independent existing float64 archive moments parity; these files never change.
 # Optional comparison with a previously deposited moments file. No external
 # project-relative archive is needed to run the public pipeline.
 errors={}
 np.savez_compressed(out,time_index=np.arange(START,STOP),cell_id=np.arange(36),**arrays,**{'budget_'+str(k):val for k,val in budgets.items()})
 np.savez_compressed(D/(case+'_interpolation.npz'),time_index=np.arange(START,STOP),cell_id=np.arange(36),**interp)
 hb=blocks(np.asarray(h,dtype=float))[0];np.savez(D/(case+'_terrain.npz'),hgt=hb)
 save(D/(case+'_RAW_CHECK.json'),{'status':'PASS','source_files':sources,'source_moment_path':None,'source_moment_sha256':None,'full3350_reductions':True,'raw_replication_max':repmax,'original_moment_max_abs':errors,'cubic_identity_max_abs':identitymax,'times':T,'cell_times':T*C,'seconds':time.monotonic()-t0,'reference_sha256':digest(out)})
 print('REFERENCE',case,T,'moments',errors,'time',round(time.monotonic()-t0,2))

def inference(case):
 p=D/(case+'_prediction.npz');assert not p.exists();m=np.load(D/(case+'_reference.npz'),allow_pickle=False)
 model=json.loads((S/'diagnostic_model.json').read_text(encoding='utf-8'))
 pred=mod(S/'diagnostic_predictor.py','frozen_compact')
 geo=mod(S/'affine_index36.py','frozen_geo')
 rows=list(csv.DictReader((S/f'{case}_PARENT_STATIC.csv').open(encoding='utf-8')))
 assert [int(row['parent_id']) for row in rows]==list(range(36))
 names=['height_sd','relief_27_mean','slope_mean','tpi_9_mean'];terrain=np.array([[float(row[k]) for k in names] for row in rows])
 assert np.allclose(terrain[:,0],np.load(D/(case+'_terrain.npz'))['hgt'].std(-1),atol=2e-5,rtol=2e-6)
 u=m['u02p'];v=m['v02p'];U=np.hypot(u,v);valid=U>=1.;x=np.stack([u,v,U,np.log1p(U),np.divide(u,U,out=np.zeros_like(U),where=valid),np.divide(v,U,out=np.zeros_like(U),where=valid),valid.astype(float)]+[np.broadcast_to(terrain[:,i],u.shape) for i in range(4)],axis=-1).reshape(-1,11).astype(np.float32).astype(float)
 pre=model['preprocessing'];spec=pre['design'];bs=BSpline(np.asarray(spec['spline_t']),np.asarray(spec['spline_c']),spec['spline_k'])
 spl=bs(np.clip(x[:,spec['spline_index_in_selected']],bs.t[bs.k],bs.t[-bs.k-1]))[:,:-1]
 phi=np.column_stack([np.ones(len(x)),((x-np.array(spec['mean']))/np.array(spec['sd']))[:,np.array(spec['keep'],bool)],(spl-np.array(spec['spline_mean']))/np.array(spec['spline_sd'])])
 result={};extra=np.empty((len(x),2))
 for a in range(0,T,48):
  b=min(a+48,T);sl=slice(a*36,b*36);g=geo.apply_operator(x[sl,:2].reshape(-1,36,2));raw=np.column_stack([np.log1p(np.trace(g['affine_explained_covariance'],axis1=-2,axis2=-1).ravel()),np.log1p(g['residual_covariance_trace'].ravel())]);e=(raw-np.array(pre['SPATIAL2']['mean']))/np.array(pre['SPATIAL2']['sd']);extra[sl]=e
  ans=pred.predict(model,phi[sl],e,U.ravel()[sl])
  for k,vv in ans.items():result.setdefault(k,np.empty(T*36))[sl]=vv
 # The historical source-fit summary is not an input to prediction.
 # The independent value check is implemented in verify.py.
 errors={}
 np.savez_compressed(p,time_index=np.arange(START,STOP),cell_id=np.arange(36),**{k:v.reshape(T,36) for k,v in result.items()})
 picks=np.array([0,1674,3349]);idx=(picks[:,None]*36+np.arange(36)).ravel()
 np.savez_compressed(D/(case+'_prediction_fixture.npz'),row_ids=idx+START*36,original11=x[idx],Phi=phi[idx],SPATIAL2=extra[idx],native_U=U.ravel()[idx],**{k:v[idx] for k,v in result.items()})
 save(D/(case+'_PREDICTION_CHECK.json'),{'status':'PASS','model':'MU_JS_SD_QA','model_sha256':digest(S/'diagnostic_model.json'),'rows':T*36,'no_refit':True,'historical_full_mean_differences':errors,'source_feature_mean_max_error':float(np.max(np.abs(x.mean(0)-np.array(spec['mean'])))) if case=='A' else None,'prediction_sha256':digest(p)})
 print('PREDICTION',case,'mean',float(result['mu'].mean()),'sd',float(result['S'].mean()),'parity',errors)

def score(case,start,end):
 assert 0<=start<end<=T;out=D/'scores'/f'{case}_{start:04d}_{end:04d}.npz';assert not out.exists();out.parent.mkdir(exist_ok=True)
 predictions=np.load(D/(case+'_prediction.npz'),allow_pickle=False);reference=np.load(D/(case+'_reference.npz'),allow_pickle=False)
 laws=mod(S/'laws.py','frozen_laws');u,_=tensor(case,'U10_d05');v,_=tensor(case,'V10_d05');values={};t0=time.monotonic()
 for a in range(start,end,8):
  b=min(end,a+8);sl=slice(a,b);uu=blocks(np.asarray(u[START+a:START+b],dtype=float));vv=blocks(np.asarray(v[START+a:START+b],dtype=float));raw=np.hypot(uu,vv).astype(np.float32).astype(float).reshape(-1,729)
  re=laws.empirical(raw);re['mu']=reference['S_s'][sl].ravel();re['S']=reference['sigma_s'][sl].ravel()
  law=laws.Law('weibull',predictions['mu'][sl].ravel(),predictions['S'][sl].ravel(),predictions['k'][sl].ravel(),predictions['logscale'][sl].ravel())
  val=laws.score(law,re)
  for k in ['CDF_D','W1','mu_error','S_error','q99_error','R3_error']:
   values.setdefault(k,np.empty((end-start,36)))[a-start:b-start]=val[k].reshape(-1,36)
 np.savez_compressed(out,time_index=np.arange(START+start,START+end),cell_id=np.arange(36),**values)
 print('SCORED',case,start,end,'D',values['CDF_D'].mean(),'seconds',round(time.monotonic()-t0,2))

def describe(a):
 a=np.asarray(a);assert np.isfinite(a).all();return dict(mean=float(a.mean()),mean_abs=float(np.abs(a).mean()),rms=float(np.sqrt(np.mean(a*a))),min=float(a.min()),max=float(a.max()),p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)))

def permutations(case):
 """20 fixed whole-vector-pair shuffles at each cell, across all 3350 times.

 Same permutation within each cell is held fixed over time. Arrays are generated
 from original d05 PT fields; no randomized weather selection or fitting.
 """
 out=D/f'spatial_targets_cell_time_{case}.npz'
 assert not out.exists()
 u,_=tensor(case,'U10_d05');v,_=tensor(case,'V10_d05')
 h=np.load(D/f'{case}_terrain.npz',allow_pickle=False)['hgt']
 lo=np.quantile(h,.25,axis=1);hi=np.quantile(h,.75,axis=1)
 low=h<=lo[:,None];high=h>=hi[:,None]
 assert np.all(low.sum(-1)>0) and np.all(high.sum(-1)>0)
 assert not np.any(low & high)
 seeds=np.r_[-1,np.arange(20261003,20261023,dtype=np.int64)]
 maps=np.arange(N)[None,None,:].repeat(C,axis=1)
 maps=maps.repeat(len(seeds),axis=0)
 for j,seed in enumerate(seeds[1:],1):
  rng=np.random.default_rng(int(seed))
  maps[j]=np.asarray([rng.permutation(N) for _ in range(C)])
 d2x=np.empty((len(seeds),T,C));d2y=np.empty_like(d2x);dh=np.empty_like(d2x)
 for first in range(START,STOP,16):
  last=min(STOP,first+16);sl=slice(first-START,last-START)
  uu=blocks(np.asarray(u[first:last],dtype=float))
  vv=blocks(np.asarray(v[first:last],dtype=float))
  for j in range(len(seeds)):
   uv=np.take_along_axis(uu,maps[j,None,:,:],axis=-1).reshape(-1,C,27,27)
   vv2=np.take_along_axis(vv,maps[j,None,:,:],axis=-1).reshape(-1,C,27,27)
   d2x[j,sl]=((np.diff(uv,axis=-1))**2+(np.diff(vv2,axis=-1))**2).mean((-2,-1))
   d2y[j,sl]=((np.diff(uv,axis=-2))**2+(np.diff(vv2,axis=-2))**2).mean((-2,-1))
   speeds=np.hypot(uv,vv2).reshape(-1,C,N)
   dh[j,sl]=(speeds*high).sum(-1)/high.sum(-1)-(speeds*low).sum(-1)/low.sum(-1)
 np.savez_compressed(out,seeds=seeds,time_index=np.arange(START,STOP),cell_id=np.arange(C),
                     D2_x=d2x,D2_y=d2y,D_h=dh)
 print('PERMUTATIONS',case,'complete_seed_count',len(seeds)-1,'sha256',digest(out))

def summarize(case):
 out=D/(case+'_SUMMARY.json');assert not out.exists();m=np.load(D/(case+'_reference.npz'),allow_pickle=False);it=np.load(D/(case+'_interpolation.npz'),allow_pickle=False);pr=np.load(D/(case+'_prediction.npz'),allow_pickle=False)
 summary={'case':case,'window':[970,4320],'times':3350,'cell_times':120600,'scales':{},'states':{},'interpolation':{},'groups':{},'transfer':{},'spatial':{},'checks':{}}
 for scale in [3,9,27]:
  a=m['budget_'+str(scale)];summary['scales'][str(scale)]={'total_variance_mean':float(a[:,0].mean()),'within_fraction_mean':float(np.mean(a[:,1]/a[:,0])),'within_ratio_of_means':float(a[:,1].mean()/a[:,0].mean()),'n_times':len(a)}
 for k in ['S_c','S_v','S_s','J','sigma_s','delta_mag','E_vec']:summary['states'][k]=describe(m[k])
 ss=m['S_s'];j=m['J'];sd=m['sigma_s'];q=m['Q'];r3=m['R3'];m3=m['mu3']
 summary['moments']={'speed_variance_share':float(np.mean(sd*sd/q)),'L3':float(np.mean(np.abs(m3)/r3)),'G3':float(abs(m3.mean())/r3.mean()),'R3_mean':float(r3.mean()),'mu3_mean':float(m3.mean()),'R3_second_mean':float((ss**3+3*ss*sd*sd).mean()),'mu3_negative_fraction':float(np.mean(m3<0))}
 mask=(m['S_v']>=1);ap=m['var_perp'][mask]/(2*m['S_v'][mask]);truth=j[mask];summary['small_fluctuation']={'count':int(mask.sum()),'spearman':float(spearmanr(ap,truth).statistic),'median_relative_error':float(np.median(np.abs(ap-truth)/np.maximum(truth,1e-12)))}
 mask=(m['S_v']>=1)&(m['S_c']>=1);summary['direction']={'count':int(mask.sum()),**describe(m['direction_difference'][mask])}
 for mode in ['bilinear','bicubic','bilinear_M','bicubic_M']:
  summary['interpolation'][mode]={'max_component_residual':float(it[mode+'_component_residual'].max()),'vector_residual_rms':float(np.sqrt(np.mean(it[mode+'_vector_residual']**2))),'J_ratio':float(it[mode+'_J'].mean()/j.mean()),'SD_ratio':float(it[mode+'_sigma_s'].mean()/sd.mean()),'J_mae':float(np.abs(it[mode+'_J']-j).mean()),'SD_mae':float(np.abs(it[mode+'_sigma_s']-sd).mean())}
  if mode.endswith('_M'):assert summary['interpolation'][mode]['max_component_residual']<1e-6
 # The original 20-seed permutation archive was internal; this public
 # reproduction accepts an explicitly supplied cache OR recomputes it from raw PT.
 spatial_path=D/f'spatial_targets_cell_time_{case}.npz'
 if not spatial_path.exists():
  if SPATIAL_CACHE:
   candidate=Path(SPATIAL_CACHE)/spatial_path.name
   if candidate.is_file():shutil.copy2(candidate,spatial_path)
  if not spatial_path.exists():permutations(case)
 sp=np.load(spatial_path,allow_pickle=False)
 assert np.array_equal(sp['time_index'],np.arange(970,4320)) and np.array_equal(sp['cell_id'],np.arange(36))
 assert np.array_equal(sp['seeds'],np.r_[-1,np.arange(20261003,20261023)])
 for k in ['D2_x','D2_y','D_h']:
  a=sp[k];assert a.shape==(21,T,36) and np.isfinite(a).all()
  means=a.mean((1,2));rms=np.sqrt((a*a).mean(1)).mean(1)
  if k.startswith('D2'):assert np.allclose(a[0],m[k],atol=1e-10,rtol=1e-10)
  summary['spatial'][k]={'original_mean':float(means[0]),'seed_means':means[1:].tolist(),'permutation_median_mean':float(np.median(means[1:])),'mean_amplification':float(np.median(means[1:])/means[0]) if abs(means[0])>1e-12 else None,'original_cell_RMS_mean':float(rms[0]),'permuted_cell_RMS_mean_median':float(np.median(rms[1:]))}
 summary['spatial']['source_sha256']=digest(spatial_path);summary['spatial']['expected_D2']=float(2*729*q.mean()/728)
 del sp;gc.collect()
 scor={};ids=[]
 for p in sorted((D/'scores').glob(case+'_*.npz')):
  with np.load(p,allow_pickle=False) as z:
   ids.append(z['time_index']);
   for k in ['CDF_D','W1','mu_error','S_error','q99_error','R3_error']:scor.setdefault(k,[]).append(z[k])
 assert ids and np.array_equal(np.concatenate(ids),np.arange(970,4320)),'incomplete scoring'
 scor={k:np.concatenate(v) for k,v in scor.items()}
 for k,val in scor.items():summary['transfer'][k]=describe(val)
 summary['transfer'].update(SD_reference_mean=float(sd.mean()),SD_prediction_mean=float(pr['S'].mean()),SD_amplitude_ratio=float(pr['S'].mean()/sd.mean()),mu_reference_mean=float(ss.mean()),mu_prediction_mean=float(pr['mu'].mean()),new_fits=0)
 for tag,mask in [('all',np.ones((T,36),bool)),('U4to6',(m['S_c']>4)&(m['S_c']<6))]:
  row={'n':int(mask.sum())}
  for k in ['S_c','S_v','S_s','J','sigma_s','delta_mag','E_vec','var_parallel','var_perp']:row[k]=describe(m[k][mask])
  row.update(predicted_SD=float(pr['S'][mask].mean()),predicted_mean=float(pr['mu'][mask].mean()),CDF_D=float(scor['CDF_D'][mask].mean()))
  summary['groups'][tag]=row
 # Hourly non-overlapping summaries preserve time evolution, including final short bin.
 time_rows=[]
 for a in range(0,T,60):
  b=min(T,a+60);row=dict(case=case,start_index=a+970,stop_index_exclusive=b+970,elapsed_hour=(a+(b-a-1)/2)/60,n_times=b-a)
  for k in ['S_c','S_v','S_s','J','sigma_s','E_vec','R3']:row[k]=float(m[k][a:b].mean())
  row['SD_pred']=float(pr['S'][a:b].mean());row['mu_pred']=float(pr['mu'][a:b].mean());row['CDF_D']=float(scor['CDF_D'][a:b].mean());time_rows.append(row)
 csvout(D/(case+'_hourly.csv'),time_rows)
 save(out,summary);print('SUMMARY',case,'scales',[summary['scales'][str(k)]['within_fraction_mean'] for k in [3,9,27]],'L3/G3',summary['moments']['L3'],summary['moments']['G3'],'D',summary['transfer']['CDF_D']['mean'],'SD_RATIO',summary['transfer']['SD_amplitude_ratio'])

def finish():
 summaries=[json.loads((D/(c+'_SUMMARY.json')).read_text(encoding='utf-8')) for c in CASES]
 save(D/'RESULTS.json',{'status':'COMPLETED_ALL3350','cases':summaries,'new_fits':0,'model_identity':'MU_JS_SD_QA','analysis_cell_times':361800})
 rows=[]
 for s in summaries:
  rows.append(dict(case=s['case'],n_times=T,cell_times=T*C,**{k:s['states'][k]['mean'] for k in ['S_c','S_v','S_s','J','sigma_s','delta_mag','E_vec']},L3=s['moments']['L3'],G3=s['moments']['G3'],CDF_D=s['transfer']['CDF_D']['mean'],SD_ratio=s['transfer']['SD_amplitude_ratio'],SD_MAE=s['transfer']['S_error']['mean_abs'],mu_MAE=s['transfer']['mu_error']['mean_abs'],q99_MAE=s['transfer']['q99_error']['mean_abs'],R3_MAE=s['transfer']['R3_error']['mean_abs']))
 csvout(D/'MAIN_NUMBERS.csv',rows);print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':
 command=sys.argv[1];start=time.monotonic()
 if command=='init':init()
 elif command=='reference':reference(sys.argv[2])
 elif command=='predict':inference(sys.argv[2])
 elif command=='score':score(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
 elif command=='summary':summarize(sys.argv[2])
 elif command=='finish':finish()
 else:raise ValueError(command)
 print('DONE',command,round(time.monotonic()-start,3),flush=True)
