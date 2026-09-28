"""V8 common: data interface (V6 fork cache, read-only), scalar GAM engine,
level iterators, window evaluation from (J, mean, S) triples, receipts.

All fits in V8 use the SAME design as V6 QA heads: scaled features + 4-knot
cubic speed spline + ridge alpha (default 0.01), torch L-BFGS. Truth labels are
exact V-based quantities from the V6 label caches (Q, a, J, S, mu, V, U).
"""
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FORK = HERE.parent / 'statistical_downscaling_v6_repair_20260915'
ROOT = FORK
for p in (str(FORK / 'src'), str(FORK / 'experiments/V7_terrain_identifiability')):
    if p not in sys.path:
        sys.path.insert(0, p)

from audit_v5 import sha, write_json  # noqa: E402
from scientific_models import design_fit, design_apply, optimize_torch, inverse_softplus, weibull_from_moments  # noqa: E402
from window_evaluation import parameter_cdf  # noqa: E402
from v7_common import (PARENT_COUNT, CORE16, make_truth_p, summarize_p,  # noqa: E402
                       subset_index, renumber)

import torch  # noqa: E402

CFG = json.loads((HERE / 'V8_MODELS.json').read_text(encoding='utf-8'))
CONTRACT_SHA = None  # set on first use
TERRAIN4 = CFG['terrain4']
SEED = CFG['engine']['seed']
ALL_COLUMNS = json.loads((FORK / 'experiments/04_terrain_features/FEATURE_SCHEMA.json').read_text(encoding='utf-8'))['all_columns']
BASE_NAMES = ['u02p', 'v02p', 'U', 'log1p_U', 'unit_u', 'unit_v', 'direction_valid']
D02_T4_NAMES = BASE_NAMES + TERRAIN4


def contract_sha():
    global CONTRACT_SHA
    if CONTRACT_SHA is None:
        CONTRACT_SHA = sha(HERE / 'V8_CONTRACT.md')
    return CONTRACT_SHA


def cache_sha(case):
    return {'labels': sha(FORK / f'cache/{case}_labels.csv.gz'),
            'features': sha(FORK / f'cache/{case}_features.npy'),
            'speed': sha(FORK / f'cache/{case}_speed.npy')}


def load_case(case):
    labels = pd.read_csv(FORK / f'cache/{case}_labels.csv.gz')
    features = np.load(FORK / f'cache/{case}_features.npy', mmap_mode='r')
    speed = np.load(FORK / f'cache/{case}_speed.npy', mmap_mode='r')
    return labels, features, speed


def feature_block(features, rows, names):
    idx = [ALL_COLUMNS.index(n) for n in names]
    return np.asarray(features[rows][:, idx], dtype=np.float32)


def level_segments(level):
    """Yield (fold_name, train_rows(A), test_case, test_rows, test_parents, context).

    train_rows index into case-A flat arrays; test_rows index into the case's
    flat arrays. For B/C, train = full A-train and case = B/C.
    """
    a_labels, _, _ = load_case('A')
    if level == 'A_T':
        start = 970
        for k, (fit_end, vs, ve) in enumerate(CFG['levels']['A_T']['folds'], 1):
            base_end = fit_end - CFG['levels']['A_T']['cal_tail']
            train_rows = np.arange(start * PARENT_COUNT, base_end * PARENT_COUNT)
            test_rows = np.arange(vs * PARENT_COUNT, ve * PARENT_COUNT)
            yield f'fold{k}', train_rows, 'A', test_rows, list(range(PARENT_COUNT)), {
                'level': level, 'fold': f'fold{k}', 'base_fit_end': base_end,
                'val_start': vs, 'val_end': ve}
    elif level in ('A_S', 'A_ST'):
        train_times = np.arange(970, 3650 if level == 'A_S' else CFG['levels']['A_ST']['train_end'])
        test_times = np.arange(970 if level == 'A_S' else CFG['levels']['A_ST']['train_end'], 3650)
        for name, test_parents in CFG['levels']['A_S']['folds'].items():
            test_parents = list(test_parents)
            train_parents = [p for p in range(PARENT_COUNT) if p not in test_parents]
            train_rows = subset_index(train_times, train_parents)
            test_rows = subset_index(test_times, test_parents)
            yield name, train_rows, 'A', test_rows, test_parents, {
                'level': level, 'fold': name, 'train_time_end': int(train_times[-1]) + 1,
                'test_time_start': int(test_times[0]), 'test_time_end': int(test_times[-1]) + 1}
    elif level in ('B', 'C'):
        train_rows = np.arange(970 * PARENT_COUNT, 3650 * PARENT_COUNT)
        case = level
        test_rows = np.arange(970 * PARENT_COUNT, 4320 * PARENT_COUNT)
        yield 'full', train_rows, case, test_rows, list(range(PARENT_COUNT)), {
            'level': level, 'fold': 'full', 'fit_interval': [970, 3650]}
    else:
        raise ValueError(level)


def train_frame(train_rows):
    a_labels, _, _ = load_case('A')
    return a_labels.iloc[train_rows].reset_index(drop=True)


def eval_truth(case, test_rows, parents):
    labels, _, speed = load_case(case)
    yv = renumber(labels.iloc[test_rows].reset_index(drop=True), parents)
    flat = speed.reshape(-1, 729)
    times = len(test_rows) // len(parents)
    svs = np.take(flat, test_rows, axis=0).reshape(times, len(parents), 729)
    return make_truth_p(svs, yv, parents)


def fit_scalar(features, columns, names, y, alpha=0.01, head='softplus', scale=None, weights=None):
    """Single-target GAM. head: softplus (positive, /scale), expit (0..1), linear."""
    x, ds = design_fit(features, columns, names)
    n, p = x.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tx = torch.as_tensor(x, dtype=torch.float64, device=device)
    init = np.zeros(p)
    if head == 'softplus':
        assert scale is not None
        ys = np.asarray(y, float) / scale
        ty = torch.as_tensor(ys, dtype=torch.float64, device=device)
        init[0] = inverse_softplus(max(float(ys.mean()), 1e-8))
    elif head == 'expit':
        ty = torch.as_tensor(np.asarray(y, dtype=np.float64), device=device)
        m = float(np.average(y, weights=weights)) if weights is not None else float(np.mean(y))
        m = min(max(m, 1e-5), 1 - 1e-5)
        init[0] = np.log(m / (1 - m))
    else:
        ty = torch.as_tensor(np.asarray(y, dtype=np.float64), device=device)
        init[0] = float(np.mean(y))
    if weights is not None:
        tw = torch.as_tensor(np.asarray(weights, dtype=np.float64), device=device)
        tw = tw / tw.sum()
    else:
        tw = None

    def objective(beta):
        z = tx @ beta
        if head == 'softplus':
            pred = torch.nn.functional.softplus(z)
            loss = ((pred - ty) ** 2).mean()
        elif head == 'expit':
            pred = torch.nn.functional.binary_cross_entropy_with_logits(z, ty, reduction='none')
            loss = (tw * pred).sum() if tw is not None else pred.mean()
        else:
            loss = ((z - ty) ** 2).mean()
        return loss + alpha * (beta[1:] ** 2).sum()

    coef, receipt = optimize_torch(init, objective)
    return {'design': ds, 'coef': coef, 'head': head, 'scale': scale, 'alpha': alpha,
            'feature_names': list(names), 'training_rows': n, 'parameter_dimension': p,
            'fit_receipt': receipt}


def predict_scalar(model, features):
    x = design_apply(model['design'], features)
    z = x @ model['coef']
    if model['head'] == 'softplus':
        out = np.logaddexp(0.0, z) * model['scale']
    elif model['head'] == 'expit':
        from scipy.special import expit
        out = expit(z)
    else:
        out = z
    return out


def qa_deploy(Q_hat, a_hat, U):
    """Deployment identities (U-approx): returns mu_hat, J_hat, S_hat."""
    Q_hat = np.maximum(np.asarray(Q_hat, float), 0.0)
    a_hat = np.clip(np.asarray(a_hat, float), 0.0, 1.0)
    U = np.asarray(U, float)
    mu = np.sqrt(U ** 2 + a_hat * Q_hat)
    J = np.divide(a_hat * Q_hat, mu + U, out=np.zeros_like(mu), where=(mu + U) > 0)
    S = np.sqrt(np.maximum((1 - a_hat) * Q_hat, 0.0))
    return mu, J, S


def weibull_params_from_moments(mu, S, J):
    params = weibull_from_moments(np.asarray(mu, float), np.asarray(S, float))
    params.update(component_count=1, weights=np.ones(1),
                  J_components=np.asarray(J, float)[:, None],
                  mean_components=np.asarray(mu, float)[:, None])
    return params


def window_eval_from_triples(J, mean, S, truth, model_id):
    """Streaming twin of v7_common.summarize_p with identical formulas.

    Accumulates per-parent window CDFs and conditional metric sums in time
    blocks (float32) so C-sized segments never materialize the (nt,P,grid)
    monolith. Returns {'summary','parents'} with the same fields as summarize_p.
    """
    from window_evaluation import j_grid, window_quantiles
    nt, P = truth['j_rows'].shape[0], len(truth['parents'])
    sg = truth['speed_grid']; jg = truth['j_grid']
    G = truth['speed_rows']; Gj = truth['j_rows']
    labels = truth['labels']
    n_rows = nt * P
    J = np.asarray(J, float).reshape(n_rows)
    mean = np.asarray(mean, float).reshape(n_rows)
    S = np.asarray(S, float).reshape(n_rows)
    jy = labels.J.to_numpy(float).reshape(n_rows)
    muy = labels.mu.to_numpy(float).reshape(n_rows)

    fw_acc = np.zeros((P, len(sg)), np.float64)
    fjw_acc = np.zeros((P, len(jg)), np.float64)
    cw1 = np.zeros(P, np.float64); ccrps = np.zeros(P, np.float64)
    j_sq = np.zeros(P, np.float64); bias_acc = np.zeros(P, np.float64)
    tail_max = 0.0
    for t0 in range(0, nt, 24):
        t1 = min(nt, t0 + 24)
        sl = np.arange(t0 * P, t1 * P)
        params = weibull_params_from_moments(mean[sl], S[sl], J[sl])
        Fb = parameter_cdf(params, sg).astype(np.float32)
        Gb = G[t0:t1].reshape(-1, len(sg))
        diff = np.abs(Fb - Gb)
        cw1 += np.trapezoid(diff.reshape(t1 - t0, P, len(sg)), sg, axis=2).sum(axis=0)
        cc = (Fb - Gb) ** 2 + Gb * (1 - Gb)
        ccrps += np.trapezoid(cc.reshape(t1 - t0, P, len(sg)), sg, axis=2).sum(axis=0)
        tail_max = max(tail_max, float(np.max(1 - Fb[:, -1])))
        fw_acc += Fb.reshape(t1 - t0, P, len(sg)).sum(axis=0, dtype=np.float64)
        Fjb = (params['J_components'][:, 0][:, None] <= jg[None, :]).astype(np.float32)
        Gjb = Gj[t0:t1].reshape(-1, len(jg))
        tail_max = max(tail_max, float(np.max(1 - Fjb[:, -1])))
        fjw_acc += Fjb.reshape(t1 - t0, P, len(jg)).sum(axis=0, dtype=np.float64)
        jb = J[sl].reshape(t1 - t0, P); jyb = jy[sl].reshape(t1 - t0, P)
        j_sq += ((jb - jyb) ** 2).sum(axis=0)
        bias_acc += (mean[sl].reshape(t1 - t0, P) - muy[sl].reshape(t1 - t0, P)).sum(axis=0)
        del Fb, Gb, diff, cc, Fjb, Gjb
    fw = fw_acc / nt; gw = truth['speed_window']
    fjw = fjw_acc / nt; gjw = truth['j_window']
    if not np.isfinite(fw).all() or not np.isfinite(fjw).all() or fw.min() < -1e-8 or fw.max() > 1 + 1e-8:
        raise ValueError('Invalid predictive CDF')
    speed_excess = np.trapezoid((fw - gw) ** 2, sg, axis=1)
    speed_crps = speed_excess + np.trapezoid(gw * (1 - gw), sg, axis=1)
    jex = np.trapezoid((fjw - gjw) ** 2, jg, axis=1)
    jcrps = jex + np.trapezoid(gjw * (1 - gjw), jg, axis=1)
    pq = window_quantiles(fw, sg); rq = window_quantiles(gw, sg)
    parent = pd.DataFrame({'model_id': model_id, 'parent_id': np.asarray(truth['parents'], dtype=int),
                           'speed_w1': np.trapezoid(np.abs(fw - gw), sg, axis=1),
                           'speed_crps': speed_crps, 'speed_excess_crps': speed_excess,
                           'J_w1': np.trapezoid(np.abs(fjw - gjw), jg, axis=1), 'J_crps': jcrps,
                           'J_rmse': np.sqrt(j_sq / nt),
                           'mean_speed_bias': bias_acc / nt,
                           'conditional_speed_w1': cw1 / nt, 'conditional_speed_crps': ccrps / nt})
    for i, p in enumerate([5, 25, 50, 75, 95]):
        parent[f'q{p:02d}_error'] = pq[:, i] - rq[:, i]
    domainF = fw.mean(axis=0); domainG = gw.mean(axis=0)
    summary = {'model_id': model_id, 'speed_w1': float(parent.speed_w1.mean()),
               'speed_crps': float(parent.speed_crps.mean()),
               'speed_excess_crps': float(parent.speed_excess_crps.mean()),
               'J_w1': float(parent.J_w1.mean()), 'J_crps': float(parent.J_crps.mean()),
               'J_rmse': float(np.sqrt(j_sq.sum() / n_rows)),
               'q05_mae': float(parent.q05_error.abs().mean()),
               'q95_mae': float(parent.q95_error.abs().mean()),
               'domain_speed_w1': float(np.trapezoid(np.abs(domainF - domainG), sg)),
               'conditional_speed_w1': float(parent.conditional_speed_w1.mean()),
               'conditional_speed_crps': float(parent.conditional_speed_crps.mean()),
               'tail_mass_at_upper': tail_max, 'valid_cdf': bool(tail_max < 1e-6),
               'time_count': nt, 'parent_count': P, 'sample_count': nt * P * 729}
    return {'summary': summary, 'parents': parent}


def point_metrics(y_true, y_pred, clim):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = float(1 - np.sum(err ** 2) / max(np.sum((y_true - y_true.mean()) ** 2), 1e-30))
    clim = float(np.sqrt(np.mean((clim - y_true) ** 2)))
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'skill_vs_clim': 1 - rmse / max(clim, 1e-30)}


def save_parents(result, path):
    result['parents'].to_csv(path, index=False)


def stage_receipt(stage, payload):
    payload = dict(payload)
    payload.setdefault('stage', stage)
    payload.setdefault('contract_sha256', contract_sha())
    payload.setdefault('registry_sha256', sha(HERE / 'V8_MODELS.json'))
    payload.setdefault('utc', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(HERE / stage / 'RECEIPT.json', payload)


def log_failure(stage, name, exc):
    d = HERE / stage / 'logs'
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / f'{name}_FAILURE.json', {'error': str(exc)[:400],
                                            'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    (d / f'{name}_TRACE.txt').write_text(traceback.format_exc(), encoding='utf-8')
