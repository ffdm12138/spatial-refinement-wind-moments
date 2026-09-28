"""V7 common: subset truth/evaluation and transferable empirical baselines.

Generalizes src/window_evaluation.py (hardcoded 36 parents) to arbitrary parent
subsets so spatial and space-time CV can score held-out parents. Formulas are
byte-identical to the V6 originals; only the parent count P is parameterized.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window_evaluation import speed_grid, j_grid, empirical_rows, parameter_cdf, window_quantiles  # noqa: E402
from scientific_models import predict_parameters  # noqa: E402

PARENT_COUNT = 36
CORE16 = [7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28]


def subset_index(times, parents):
    """Row indices (time-major, subset-parent-minor) into flat (T*36) arrays."""
    parents = np.asarray(parents, dtype=np.int64)
    return np.concatenate([t * PARENT_COUNT + parents for t in np.asarray(times, dtype=np.int64)])


def renumber(labels_subset, parents):
    """Relabel parent_id -> position 0..P-1 for subset labels (copy)."""
    out = labels_subset.copy()
    out['parent_id'] = np.tile(np.arange(len(parents)), len(out) // len(parents))
    return out


def make_truth_p(samples, labels, parents, sg=None, jg=None):
    """samples (nt,P,729); labels (nt*P) already renumbered; parents kept for provenance."""
    sg = speed_grid() if sg is None else sg
    jg = j_grid() if jg is None else jg
    n = len(labels)
    nt = n // len(parents)
    if n % len(parents) or not np.array_equal(labels.parent_id, np.tile(np.arange(len(parents)), nt)):
        raise ValueError('Expected complete chronological subset rows')
    G = empirical_rows(samples.reshape(-1, samples.shape[-1]), sg)  # rows x grid
    G = G.reshape(nt, len(parents), len(sg))
    Gj = (labels.J.to_numpy()[:, None] <= jg).astype(np.float32).reshape(nt, len(parents), len(jg))
    return {'speed_grid': sg, 'j_grid': jg, 'speed_rows': G, 'j_rows': Gj,
            'speed_window': G.mean(axis=0, dtype=np.float64),
            'j_window': Gj.mean(axis=0, dtype=np.float64),
            'labels': labels, 'parents': list(parents)}


def summarize_p(F, Fj, truth, jpoint, meanpoint, model_id, retain=False):
    nt = truth['speed_rows'].shape[0]
    P = truth['speed_rows'].shape[1]
    sg = truth['speed_grid']; jg = truth['j_grid']; G = truth['speed_rows']; Gj = truth['j_rows']
    labels = truth['labels']
    F = np.asarray(F).reshape(nt, P, len(sg))
    Fj = np.asarray(Fj).reshape(nt, P, len(jg))
    if not np.isfinite(F).all() or not np.isfinite(Fj).all() or np.min(F) < -1e-8 or np.max(F) > 1 + 1e-8:
        raise ValueError('Invalid predictive CDF')
    fw = F.mean(axis=0, dtype=np.float64); gw = truth['speed_window']
    fjw = Fj.mean(axis=0, dtype=np.float64); gjw = truth['j_window']
    tail = float(max(np.max(1 - F[:, :, -1]), np.max(1 - Fj[:, :, -1])))
    speed_excess = np.trapezoid((fw - gw) ** 2, sg, axis=1)
    speed_crps = speed_excess + np.trapezoid(gw * (1 - gw), sg, axis=1)
    jex = np.trapezoid((fjw - gjw) ** 2, jg, axis=1)
    jcrps = jex + np.trapezoid(gjw * (1 - gjw), jg, axis=1)
    pq = window_quantiles(fw, sg); rq = window_quantiles(gw, sg)
    jp = np.asarray(jpoint).reshape(nt, P); mp = np.asarray(meanpoint).reshape(nt, P)
    jy = labels.J.to_numpy().reshape(nt, P)
    parent = pd.DataFrame({'model_id': model_id, 'parent_id': np.asarray(truth['parents'], dtype=int),
                           'speed_w1': np.trapezoid(np.abs(fw - gw), sg, axis=1),
                           'speed_crps': speed_crps, 'speed_excess_crps': speed_excess,
                           'J_w1': np.trapezoid(np.abs(fjw - gjw), jg, axis=1), 'J_crps': jcrps,
                           'J_rmse': np.sqrt(np.mean((jp - jy) ** 2, axis=0)),
                           'mean_speed_bias': np.mean(mp - labels.mu.to_numpy().reshape(nt, P), axis=0),
                           'conditional_speed_w1': np.mean(np.trapezoid(np.abs(F - G), sg, axis=2), axis=0),
                           'conditional_speed_crps': np.mean(np.trapezoid((F - G) ** 2 + G * (1 - G), sg, axis=2), axis=0)})
    for i, p in enumerate([5, 25, 50, 75, 95]):
        parent[f'q{p:02d}_error'] = pq[:, i] - rq[:, i]
    domainF = fw.mean(axis=0); domainG = gw.mean(axis=0)
    summary = {'model_id': model_id, 'speed_w1': float(parent.speed_w1.mean()),
               'speed_crps': float(parent.speed_crps.mean()),
               'speed_excess_crps': float(parent.speed_excess_crps.mean()),
               'J_w1': float(parent.J_w1.mean()), 'J_crps': float(parent.J_crps.mean()),
               'J_rmse': float(np.sqrt(np.mean((jp - jy) ** 2))),
               'q05_mae': float(parent.q05_error.abs().mean()), 'q95_mae': float(parent.q95_error.abs().mean()),
               'domain_speed_w1': float(np.trapezoid(np.abs(domainF - domainG), sg)),
               'conditional_speed_w1': float(parent.conditional_speed_w1.mean()),
               'conditional_speed_crps': float(parent.conditional_speed_crps.mean()),
               'tail_mass_at_upper': tail, 'valid_cdf': bool(tail < 1e-6),
               'time_count': nt, 'parent_count': P, 'sample_count': nt * P * 729}
    result = {'summary': summary, 'parents': parent, 'speed_window': fw, 'J_window': fjw,
              'J_point': jp, 'mean_point': mp}
    if retain:
        result.update(speed_rows=F, j_rows=Fj)
    return result


def evaluate_model_p(model, features, truth, model_id, probabilistic=False, retain=False):
    n = len(features); sg = truth['speed_grid']; jg = truth['j_grid']
    F = np.empty((n, len(sg)), np.float32); Fj = np.empty((n, len(jg)), np.float32)
    J = np.empty(n); mean = np.empty(n)
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        params = predict_parameters(model, features[start:stop], probabilistic)
        F[start:stop] = parameter_cdf(params, sg)
        Fj[start:stop] = np.einsum('nkg,k->ng', (params['J_components'][:, :, None] <= jg).astype(float), params['weights'])
        J[start:stop] = params['J_components'] @ params['weights']
        mean[start:stop] = params['mean_components'] @ params['weights']
    return summarize_p(F, Fj, truth, J, mean, model_id, retain)


def transferable_baselines(train_speed, train_labels, truth, model_id):
    """V6_B0 / V6_B2 generalized to subsets. train_speed (m,729) rows, train_labels (m) with parent_id/U/J.

    V6_B1 (parent-specific history) is intentionally absent: it is not defined for
    unseen parents and would leak the spatial CV target.
    """
    sg = truth['speed_grid']; jg = truth['j_grid']
    nt = truth['speed_rows'].shape[0]; P = len(truth['parents'])
    y = np.asarray(train_speed).reshape(-1, 729)
    if model_id == 'V6_B0':
        s = np.sort(y.ravel()); j = np.sort(train_labels.J.to_numpy())
        F = np.broadcast_to(np.searchsorted(s, sg, side='right') / len(s), (nt, P, len(sg)))
        Fj = np.broadcast_to(np.searchsorted(j, jg, side='right') / len(j), (nt, P, len(jg)))
        jp = np.full(nt * P, j.mean()); mp = np.full(nt * P, s.mean(dtype=np.float64))
    elif model_id == 'V6_B2':
        U = train_labels.U.to_numpy()
        bounds = np.unique(np.quantile(U, np.linspace(0, 1, 9)))[1:-1]
        bins = np.searchsorted(bounds, U)
        evalbins = np.searchsorted(bounds, truth['labels'].U.to_numpy())
        fs = []; fj = []; jm = []; sm = []
        for b in range(len(bounds) + 1):
            mask = bins == b
            s = np.sort(y[mask].ravel()); j = np.sort(train_labels.J.to_numpy()[mask])
            fs.append(np.searchsorted(s, sg, side='right') / len(s))
            fj.append(np.searchsorted(j, jg, side='right') / len(j))
            jm.append(j.mean()); sm.append(s.mean(dtype=np.float64))
        F = np.asarray(fs)[evalbins]; Fj = np.asarray(fj)[evalbins]
        jp = np.asarray(jm)[evalbins]; mp = np.asarray(sm)[evalbins]
    else:
        raise ValueError(model_id)
    return summarize_p(F, Fj, truth, jp, mp, model_id)
