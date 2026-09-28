"""Fail-closed cache identity for ordered statistical inputs, not point scores."""
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA = 'EMPIRICAL_BOOTSTRAP_CACHE_V1'
R = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def array_identity(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise TypeError('Object arrays cannot define cache identity')
    h = hashlib.sha256()
    if a.ndim == 0:
        h.update(a.tobytes())
    else:
        for start in range(0, len(a), 16):
            h.update(np.ascontiguousarray(a[start:start+16]).tobytes())
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=h.hexdigest())

def source_identity():
    paths = [D / name for name in ('cache_binding.py', 'bounded_bootstrap.py',
             'compute_empirical.py', 'finish_lowmemory.py', 'EMPIRICAL_BASELINE_SPEC.md')]
    paths += [R/'research/qa_framework_v9_20260916'/name for name in
              ('v9_7_scores.py', 'v9_7_numerics.py', 'v9_6_common.py')]
    paths += [R/'research/statistical_downscaling_v6_repair_20260915/src/window_evaluation.py',
              R/'state/CASE_CONTRACT.json']
    return {p.relative_to(R).as_posix(): sha(p) for p in paths}

def inputs_identity(scoring, meta, train_rows, test_rows, baselines):
    """Hash the true ordered streams once per configuration; no file mtime cache."""
    return dict(configuration=meta, numpy_version=np.__version__, sources=source_identity(),
        train_rows=array_identity(train_rows), test_rows=array_identity(test_rows),
        time_index=array_identity(np.asarray(test_rows)//36),
        parent_index=array_identity(np.asarray(test_rows)%36),
        parents=array_identity(np.asarray(meta['parents'])),
        grids={k:array_identity(g) for k,g in [('speed',scoring.speed_grid()),('J',scoring.j_grid())]},
        reference={k:array_identity(scoring.truth_stream(meta,k)) for k in ('speed','J')},
        prediction={k:array_identity(scoring.get_stream(meta,'QA_dyn',k)) for k in ('speed','J')},
        # Empirical streams are constant in time; hash their full parent-by-grid values.
        empirical={b:{k:array_identity(scoring.get_stream(meta,b,k)[0]) for k in ('speed','J')}
                   for b in baselines})

def fingerprint(inputs, baseline, seed, length, nboot=1000):
    metadata = dict(schema_version=SCHEMA, inputs=inputs, candidate='QA_dyn',
                    baseline=baseline, targets=['speed','J'], seed=int(seed),
                    requested_block_length=int(length), n_bootstrap=int(nboot),
                    sampling='exact_nt_circular_moving_block_bootstrap')
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(metadata=metadata, sha256=digest)

def load_cache(path, expected):
    """Legacy or mismatching caches are errors; never overwrite official results."""
    path=Path(path)
    if not path.exists():
        return None
    value=json.loads(path.read_text(encoding='utf-8'))
    if 'cache_binding' not in value:
        raise RuntimeError(f'CACHE_SCHEMA_LEGACY: {path}; independently audit before migration')
    if value['cache_binding'] != expected:
        raise RuntimeError(f'CACHE_INPUT_MISMATCH: {path}; preserve cache and audit changed inputs')
    if value.get('cache_payload_sha256') != payload_identity(value):
        raise RuntimeError(f'CACHE_PAYLOAD_MISMATCH: {path}')
    return value

def payload_identity(value):
    value={k:v for k,v in value.items() if k not in ('cache_binding','cache_payload_sha256','cache_migration')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def bind_cache(value, binding):
    return dict(value,cache_binding=binding,cache_payload_sha256=payload_identity(value))
