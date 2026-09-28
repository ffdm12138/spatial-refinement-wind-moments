"""Ordered-input identity and fail-closed validation for all original pairs."""
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA = 'V9.7_BOOTSTRAP_PROVENANCE_2'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def array_id(value):
    a=np.asarray(value)
    if a.dtype.hasobject: raise TypeError('Object array cannot identify input')
    h=hashlib.sha256()
    for i in range(0,len(a),16):h.update(np.ascontiguousarray(a[i:i+16]).tobytes())
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=h.hexdigest())

def file_id(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def inputs_identity(scoring,meta,models):
    """Re-read actual ordered data; never memoize by path, mtime or point score."""
    d=scoring.OUT/meta['key']
    with np.load(d/'predictions.npz') as z: rows=z['rows']
    from v9_6_common import level_segments
    if meta['key'].startswith(('A_T_','A_S_','A_ST_')):
        match=next(s for s in level_segments(meta['level']) if s[0]==meta['fold'])
        train,test=match[1],match[3]
        if not np.array_equal(rows,test):raise ValueError('evaluation mapping differs from split contract')
    else:train=np.arange(970*36,3650*36)
    if len(rows)!=meta['time_count']*len(meta['parents']):raise ValueError('row shape')
    parent_rows=rows.reshape(meta['time_count'],-1)
    if not np.all(parent_rows%36==np.asarray(meta['parents'])):raise ValueError('parent order')
    if not np.all(parent_rows//36==(parent_rows[:,0]//36)[:,None]):raise ValueError('time alignment')
    sources=[HERE/n for n in ('v9_7_cache.py','v9_7_scores.py','v9_7_numerics.py','v9_7_repair.py','v9_6_common.py')]
    sources += [Path(__import__('window_evaluation').__file__), ROOT/'state/CASE_CONTRACT.json',d/'calibration.json']
    return dict(configuration=meta,mapping_schema='row=time_index*36+parent_id;time-major;v1',
        train_rows=array_id(train),evaluation_rows=array_id(rows),time_sequence=array_id(rows//36),
        parent_sequence=array_id(rows%36),parents=array_id(np.asarray(meta['parents'])),
        sources={p.relative_to(ROOT).as_posix():file_id(p) for p in sources},
        grids={k:array_id(g) for k,g in [('speed',scoring.speed_grid()),('J',scoring.j_grid())]},
        reference={k:array_id(scoring.truth_stream(meta,k)) for k in ('speed','J')},
        predictions={m:{k:array_id(scoring.get_stream(meta,m,k)) for k in ('speed','J')} for m in models},
        numpy_version=np.__version__,torch_version=__import__('torch').__version__,
        numerics='float64 prefix; float32 CUDA block sums; float64 squared CDF integration')

def binding(inputs,candidate,baseline,seed,length=120,nboot=1000):
    nt=inputs['configuration']['time_count'];actual=min(length,nt)
    if min(nt,length,nboot)<=0:raise ValueError('positive sampling parameters required')
    count=(nt+actual-1)//actual
    starts=np.random.default_rng(seed).integers(0,nt,(nboot,count),dtype=np.int64)
    selected=dict(inputs,predictions={m:inputs['predictions'][m] for m in (candidate,baseline)})
    meta=dict(schema=SCHEMA,inputs=selected,candidate=candidate,baseline=baseline,seed=int(seed),
        requested_block_length=int(length),block_len=int(actual),n_boot=int(nboot),block_count=count,
        targets=['speed','J'],score='parentwise-window-marginal CRPS baseline minus candidate',
        ci='numpy.quantile linear; equal-tailed [0.025,0.975]',
        sampling='exact_nt_circular_moving_block_bootstrap',mapping=array_id(starts))
    return dict(metadata=meta,sha256=digest(meta))

def payload_digest(payload):
    return digest({k:v for k,v in payload.items() if k not in ('cache_binding','cache_payload_sha256')})

def bind(payload,expected):
    return dict(payload,cache_binding=expected,cache_payload_sha256=payload_digest(payload))

def mismatch(payload,expected):
    """An expected identity is mandatory, including on sensitivity paths."""
    if expected is None:return 'CURRENT_INPUT_IDENTITY_REQUIRED'
    try:
        if payload.get('provenance_version')!=SCHEMA:return 'UNSUPPORTED_SCHEMA'
        if expected['sha256']!=digest(expected['metadata']):return 'INVALID_EXPECTED_IDENTITY'
        if payload.get('cache_binding')!=expected:return 'ORDERED_INPUT_OR_SETTINGS_MISMATCH'
        m=expected['metadata']
        for k in ('seed','block_len','n_boot','block_count'):
            if payload.get(k)!=m[k]:return k.upper()+'_MISMATCH'
        if payload.get('bootstrap_mode')!=m['sampling']:return 'SAMPLING_MISMATCH'
        if payload.get('mapping_sha256')!=m['mapping']['sha256']:return 'MAPPING_MISMATCH'
        if payload.get('mapping_shape')!=m['mapping']['shape']:return 'MAPPING_SHAPE_MISMATCH'
        if payload.get('mapping_dtype')!='int64' or payload.get('mapping_rng')!='numpy.random.default_rng':return 'MAPPING_RECIPE_MISMATCH'
        for k in ('speed','J'):
            a=np.asarray(payload['draw_delta_'+k],dtype=np.float64)
            if a.shape!=(m['n_boot'],) or not np.isfinite(a).all():return 'INVALID_DRAWS'
            if array_id(a)['sha256']!=payload['draw_delta_'+k+'_sha256']:return 'DRAW_DIGEST_MISMATCH'
            if payload['delta_'+k+'_ci_low']!=float(np.quantile(a,.025)) or payload['delta_'+k+'_ci_high']!=float(np.quantile(a,.975)):return 'CI_MISMATCH'
        if payload.get('cache_payload_sha256')!=payload_digest(payload):return 'PAYLOAD_MISMATCH'
    except (KeyError,TypeError,ValueError,IndexError):return 'MALFORMED_CACHE'
    return None
