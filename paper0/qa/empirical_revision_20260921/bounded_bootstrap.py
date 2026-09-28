"""Same P0 circular draw contract, float64 and bounded CDF-grid chunks."""
import hashlib
import numpy as np
def bootstrap(scoring,meta,candidate,baseline,seed,length=120,nboot=1000,cpu=False):
 if not cpu:import torch
 nt=meta['time_count'];np_=len(meta['parents']);length=min(length,nt);count=(nt+length-1)//length;rem=nt-(count-1)*length
 starts=np.random.default_rng(seed).integers(0,nt,(nboot,count),dtype=np.int64);lens=np.full(count,length);lens[-1]=rem
 result={};draws={};maxerror=0.
 for kind,grid in [('speed',scoring.speed_grid()),('J',scoring.j_grid())]:
  truth=scoring.truth_stream(meta,kind);cand=scoring.get_stream(meta,candidate,kind);base=scoring.get_stream(meta,baseline,kind)
  weights=np.r_[(grid[1]-grid[0])/2,(grid[2:]-grid[:-2])/2,(grid[-1]-grid[-2])/2];delta=np.zeros(nboot);point=0.
  st=starts if cpu else torch.as_tensor(starts,device='cuda')
  for p in range(np_):
   for data,sign in [(base,1),(cand,-1)]:
    for g in range(0,len(grid),128):
     sl=slice(g,g+128);ww=weights[sl];difference=np.asarray(data[:,p,sl],float)-np.asarray(truth[:,p,sl],float)
     mean=difference.mean(axis=0);point+=sign*np.dot(mean*mean,ww)/np_
     prefix=np.zeros((nt+length+1,len(ww)));prefix[1:nt+1]=difference;prefix[nt+1:]=difference[:length];np.cumsum(prefix,axis=0,out=prefix)
     ids=np.arange(nt);blocks=prefix[ids+length]-prefix[ids];tails=prefix[ids+rem]-prefix[ids]
     bt=blocks if cpu else torch.as_tensor(blocks,dtype=torch.float64,device='cuda');tt=tails if cpu else torch.as_tensor(tails,dtype=torch.float64,device='cuda');wt=ww if cpu else torch.as_tensor(ww,device='cuda')
     for b in range(0,nboot,25):
      s=st[b:b+25];summed=bt[s[:,:-1]].sum(axis=1) if cpu else bt[s[:,:-1]].sum(dim=1)
      avg=(summed+tt[s[:,-1]])/nt if count>1 else tt[s[:,-1]]/nt
      vals=((avg*avg)*wt).sum(axis=1) if cpu else (avg.square()*wt).sum(dim=1).cpu().numpy();delta[b:b+25]+=sign*vals/np_
      if b==0:
       direct=(prefix[starts[0]+lens]-prefix[starts[0]]).sum(axis=0)/nt;maxerror=max(maxerror,abs(np.dot(direct*direct,ww)-vals[0]))
     del bt,tt,wt,prefix,blocks,tails,difference
  draws[kind]=delta
  result.update({f'delta_{kind}_point':float(point),f'delta_{kind}_mean':float(delta.mean()),f'delta_{kind}_ci_low':float(np.quantile(delta,.025)),f'delta_{kind}_ci_high':float(np.quantile(delta,.975)),f'p_candidate_{kind}':float(np.mean(delta>0))})
 assert maxerror<2e-11,maxerror
 result.update(block_len=length,block_count=count,n_boot=nboot,seed=seed,bootstrap_mode='exact_nt_circular_moving_block_bootstrap',accumulation_dtype='float64',grid_chunk_size=128,dropped_tail_rows=0,wrapped_tail_rows=0,float64_crosscheck_max_abs=maxerror,mapping_sha256=hashlib.sha256(starts.tobytes()).hexdigest(),mapping_shape=list(starts.shape),mapping_dtype='int64',mapping_rng='numpy.random.default_rng',mapping_reproduction='np.random.default_rng(seed).integers(0,time_count,(n_boot,block_count),dtype=np.int64)',provenance_version='V9.7_BOOTSTRAP_PROVENANCE_1')
 for kind,arr in draws.items():result['draw_delta_'+kind]=arr.tolist();result['draw_delta_'+kind+'_sha256']=hashlib.sha256(arr.tobytes()).hexdigest()
 return result
