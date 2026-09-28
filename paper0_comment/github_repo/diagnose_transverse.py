"""Supplemental approximation diagnostics; no changes to the released arrays."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
def main():
 rows=[]
 for case in 'ABC':
  z=np.load(ROOT/'RELEASE'/case/'moments_cell_time.npz')
  mask=z['U_vec'][970:]>=1;r=np.sqrt(z['trC'][970:])[mask]/z['U_vec'][970:][mask]
  e=abs(z['J2nd'][970:][mask]-z['J'][970:][mask])/np.maximum(z['J'][970:][mask],1e-12)
  for lo,hi,name in [(0,.25,'r<0.25'),(.25,.5,'0.25<=r<0.5'),(.5,1,'0.5<=r<1'),(1,np.inf,'r>=1'),(0,np.inf,'all')]:
   keep=(r>=lo)&(r<hi);x=e[keep]
   rows.append(dict(case=case,bin=name,count=len(x),fraction=keep.mean(),median_relative_error=float(np.median(x)) if len(x) else None,p90_relative_error=float(np.quantile(x,.9)) if len(x) else None,epsilon=1e-12,Sv_min=1,window='[970,4320)'))
 out=ROOT/'reproduced';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'COMMENT_TRANSVERSE_ERROR.csv',index=False)
 print('Wrote supplemental approximation errors to',out/'COMMENT_TRANSVERSE_ERROR.csv')
if __name__=='__main__':main()
