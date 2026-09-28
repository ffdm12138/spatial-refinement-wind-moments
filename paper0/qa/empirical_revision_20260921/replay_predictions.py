"""Replay all frozen coefficients on preserved features; never refit."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];V=R/'research/qa_framework_v9_20260916';sys.path.insert(0,str(V))
from v9_6_common import predict_scalar,feature_block_v96,D02_T4_NAMES
O=V/'numerical_repair_20260921';records=[]
for meta in json.loads((O/'manifest.json').read_text()):
 d=O/meta['key'];z=np.load(d/'predictions.npz');models=pd.read_pickle(d/'models.pkl.gz');features=np.load(R/f'research/statistical_downscaling_v6_repair_20260915/cache/{meta["case"]}_features.npy',mmap_mode='r');X=feature_block_v96(features,z['rows'],D02_T4_NAMES)
 errors={k:float(abs(predict_scalar(models[k],X)-z[k]).max()) for k in ('logQ','logita','logJ','logS')}
 assert max(errors.values())<1e-10,errors
 records.append(dict(key=meta['key'],rows=len(z['rows']),max_absolute_errors=errors));print(meta['key'],errors,flush=True)
(R/'qa/empirical_revision_20260921/PREDICTION_REPLAY.json').write_text(json.dumps(dict(status='PASS',models_refitted=0,records=records),indent=2)+'\n')
