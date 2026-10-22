"""Reproduce the Comment's mean-anchor and spatial-heterogeneity checks.

Input: A/B/C_reference.npz and A/B/C_prediction.npz from the deposited full3350
post-processing pipeline (commit 40ac8ba02c84). No fitting or raw-WRF modification.
Example:
  python summarize_revision_diagnostics.py --input /path/to/run3350 --output /path/to/new_checks
The output directory must be new. Input filenames, hashes and array conventions
are recorded; private machine paths are not written into the portable results.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

WINDOW=(970,4320)
SHAPE=(3350,36)
BANDS=((2,4),(3,5),(4,6),(5,7),(6,8))

def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def save_csv(p:Path,rows:list[dict])->None:
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(input_dir:Path,output:Path)->dict:
    if not input_dir.is_dir():raise FileNotFoundError(input_dir)
    if output.exists():raise FileExistsError('Use a new output directory: '+str(output))
    ids=np.arange(36);rr,cc=np.divmod(ids,6)
    inner=(rr>0)&(rr<5)&(cc>0)&(cc<5)
    assert int(inner.sum())==16 and int((~inner).sum())==20
    cases={};sources={};cell_rows=[];band_rows=[]
    for case in 'ABC':
        refpath=input_dir/f'{case}_reference.npz';predpath=input_dir/f'{case}_prediction.npz'
        with np.load(refpath,allow_pickle=False) as ref,np.load(predpath,allow_pickle=False) as pred:
            sc=np.array(ref['S_c'],dtype=np.float64);ss=np.array(ref['S_s'],dtype=np.float64)
            sd=np.array(ref['sigma_s'],dtype=np.float64);hat=np.array(pred['S'],dtype=np.float64)
        if any(x.shape!=SHAPE or not np.isfinite(x).all() for x in (sc,ss,sd,hat)):
            raise ValueError(case+': require complete finite 3350 x 36 arrays')
        if np.any(sd<=0) or np.any(hat<=0) or np.any(sc<0):raise ValueError(case+': invalid speed/SD')
        for p in (refpath,predpath):sources[p.name]={'sha256':digest(p),'bytes':p.stat().st_size}
        mean_sd=sd.mean(axis=0);mean_hat=hat.mean(axis=0);ratios=mean_hat/mean_sd
        mae=np.abs(hat-sd).mean(axis=0)
        if mae.sum()<=0:raise ValueError('Error concentration undefined for zero total error')
        rank=np.lexsort((ids,-mae));top=rank[:6]
        weights=mean_sd/mean_sd.sum();pooled=float(hat.mean()/sd.mean())
        if not np.isclose(pooled,float(weights@ratios),rtol=1e-12,atol=1e-12):
            raise ArithmeticError('Pooled ratio must equal reference-amplitude-weighted cell ratio')
        violations=int((ss<sc).sum());floor=float(np.maximum(sc-ss,0).mean())
        cases[case]={
          'cell_times':ss.size,
          'native_anchor':{'Ss_below_Sc_count':violations,'fraction':violations/ss.size,
                           'min_Ss_minus_Sc_mps':float((ss-sc).min()),'MAE_lower_bound_mps':floor},
          'spread':{'pooled_ratio':pooled,'median_cell_ratio':float(np.median(ratios)),
                    'min_cell_ratio':float(ratios.min()),'max_cell_ratio':float(ratios.max()),
                    'inner16_pooled_ratio':float(hat[:,inner].mean()/sd[:,inner].mean()),
                    'perimeter20_pooled_ratio':float(hat[:,~inner].mean()/sd[:,~inner].mean()),
                    'highest_error6_share':float(mae[top].sum()/mae.sum()),
                    'highest_error6_cells':top.tolist(),
                    'highest_error6_perimeter_count':int((~inner[top]).sum()),
                    'SD_MAE_mps':float(mae.mean())}}
        for cell in ids:
            cell_rows.append({'case':case,'cell':int(cell),'row':int(rr[cell]),'column':int(cc[cell]),
              'array_perimeter':int(not inner[cell]),'reference_time_mean_SD_mps':float(mean_sd[cell]),
              'prediction_time_mean_SD_mps':float(mean_hat[cell]),'cell_SD_ratio':float(ratios[cell]),
              'time_mean_SD_absolute_error_mps':float(mae[cell]),
              'share_of_case_SD_absolute_error':float(mae[cell]/mae.sum()),
              'top6_by_SD_absolute_error':int(cell in top)})
        for low,high in BANDS:
            mask=(sc>low)&(sc<high);n=int(mask.sum())
            if n==0:raise ValueError('No samples in prelisted descriptive band')
            band_rows.append({'case':case,'lower_exclusive_mps':low,'upper_exclusive_mps':high,
              'cell_times':n,'pooled_SD_ratio':float(hat[mask].mean()/sd[mask].mean())})
    report={'schema':'npj.comment.revision_diagnostics/v1','window':list(WINDOW),
      'times_per_case':3350,'cells_per_case':36,'cell_order':'c=6*row+column; zero-based',
      'native_anchor_bound':'mean(abs(mu_hat-Ss)) >= mean(max(Sc-Ss,0)) for any mu_hat>=Sc; not an error attribution',
      'cell_ratio':'mean_t(predicted_SD)/mean_t(reference_SD)',
      'pooled_ratio':'mean_tc(predicted_SD)/mean_tc(reference_SD); reference-amplitude-weighted cell ratios',
      'top6_definition':'Six largest time-mean absolute SD errors, separately selected within each case; descriptive, not a holdout',
      'array_partition':'16 central cells vs 20 perimeter cells in the 6x6 analysis array; not an authenticated WRF boundary',
      'bands':'Strict overlapping bands fixed in the preceding review; post-hoc descriptive sensitivity, no independent inference',
      'new_model_fits':0,'input_sources':sources,'cases':cases}
    output.mkdir(parents=True)
    (output/'REVISION_DIAGNOSTICS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    save_csv(output/'CELLWISE_SPREAD_DIAGNOSTICS.csv',cell_rows)
    save_csv(output/'SPEED_BAND_DIAGNOSTICS.csv',band_rows)
    print(json.dumps({'status':'COMPUTED','cases':cases,'cell_rows':len(cell_rows),'band_rows':len(band_rows)},ensure_ascii=False))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.input.resolve(),a.output.resolve())
