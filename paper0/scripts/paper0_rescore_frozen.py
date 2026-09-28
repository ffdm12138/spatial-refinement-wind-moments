"""Recompute window scores from frozen CDFs, or regenerate CDFs from predictions.

No fitting, promotion, bootstrap rewrite or canonical output writes occur.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research/qa_framework_v9_20260916'
sys.path.insert(0,str(HERE))
from v9_7_scores import scores, OUT, MODELS
from v9_7_numerics import speed_cdf,gap_cdf,SPEED_QA_ORDER,SPEED_JS_ORDER,GAP_ORDER
from window_evaluation import speed_grid,j_grid

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--config',default='A_670');ap.add_argument('--out',required=True,type=Path)
 ap.add_argument('--from-predictions',action='store_true',help='CPU float64 regeneration on full scoring grids; expensive; never refits')
 a=ap.parse_args();a.out=a.out.resolve()
 if a.out==OUT.resolve() or OUT.resolve() in a.out.parents or (ROOT/'paper0_frozen_evidence').resolve() in (a.out,*a.out.parents):raise ValueError('output must not overwrite frozen/staged numerical evidence')
 if a.out.exists() and any(a.out.iterdir()):raise ValueError('output directory must be empty')
 a.out.mkdir(parents=True,exist_ok=True)
 metas=json.loads((OUT/'manifest.json').read_text());meta=next(m for m in metas if m['key']==a.config)
 recorded=json.loads((OUT/a.config/'evaluation.json').read_text())['scores']
 if not a.from_predictions:
  result=scores(meta,MODELS)
 else:
  z=np.load(OUT/a.config/'predictions.npz');cal=json.loads((OUT/a.config/'calibration.json').read_text())
  n=len(z['U']);result={}
  from v9_7_scores import truth_stream
  for model in MODELS:
   result[model]={}
   for kind,grid in [('speed',speed_grid()),('J',j_grid())]:
    truth=truth_stream(meta,kind);shape=truth.shape
    arr=np.lib.format.open_memmap(a.out/f'{model}_{kind}.npy',mode='w+',dtype=np.float64,shape=shape)
    flat=arr.reshape(n,len(grid))
    for start in range(0,n,32):
     stop=min(n,start+32);pred={k:z[k][start:stop] for k in ('logQ','logita','logJ','logS')};u=z['U'][start:stop]
     flat[start:stop]=(speed_cdf(pred,u,cal,model,grid,order=SPEED_QA_ORDER if model.startswith('QA') else SPEED_JS_ORDER,device='cpu')[0] if kind=='speed' else gap_cdf(pred,u,cal,model,grid,GAP_ORDER))
    arr.flush();fw=arr.mean(axis=0,dtype=float);gw=truth.mean(axis=0,dtype=float)
    result[model][f'{kind}_crps']=float(np.trapezoid((fw-gw)**2+gw*(1-gw),grid,axis=1).mean())
    result[model][f'{kind}_w1']=float(np.trapezoid(abs(fw-gw),grid,axis=1).mean())
 report={'config':a.config,'mode':'CPU_prediction_regeneration' if a.from_predictions else 'stored_CDF_rescore','scores':result,'delta_from_recorded':{m:{k:v-recorded[m][k] for k,v in vals.items()} for m,vals in result.items()},'limits':'CPU regeneration may differ from original GPU rounding; no automatic promotion; bootstrap not recomputed'}
 (a.out/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'config':a.config,'output':str(a.out/'scores.json'),'max_abs_difference':max(abs(v) for m in report['delta_from_recorded'].values() for v in m.values())}))
if __name__=='__main__':main()
