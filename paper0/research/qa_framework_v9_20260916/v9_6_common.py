"""Shared, fail-closed primitives for the V9.6 Final Representation Seal.

Governed by V9_6_CONTRACT.md.
Imports read-only data interfaces and the GAM engine from V8.
Implements:
  1. Prob-JS independent and joint covariance formulations under mu = U + J
  2. Within-test-parent terrain derangements strictly preserving test terrain multisets
  3. Each-row-once 5-fold cross-fitting across Case A-train [970, 3650) (96,480 rows)
  4. Memory-bounded paired contiguous 120-minute block bootstrap
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln, logit

V96 = Path(__file__).resolve().parent
ROOT = V96.parent.parent
V8 = ROOT / "research" / "qa_framework_v8_20260916"
if V8.exists() and str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

try:
    from v8_common import (  # noqa: E402
        D02_T4_NAMES as _V8_D02_T4_NAMES,
        PARENT_COUNT,
        eval_truth,
        feature_block as _v8_feature_block,
        fit_scalar,
        level_segments,
        load_case,
        predict_scalar,
        train_frame,
    )
    from window_evaluation import parameter_cdf  # noqa: E402
except ImportError:
    _V8_D02_T4_NAMES = None
    PARENT_COUNT = 36
    eval_truth = None
    _v8_feature_block = None
    fit_scalar = None
    level_segments = None
    load_case = None
    predict_scalar = None
    train_frame = None
    parameter_cdf = None

if (V96 / "FEATURE_SCHEMA.json").exists():
    SCHEMA_SOURCE = V96 / "FEATURE_SCHEMA.json"
else:
    DATA_ROOT = ROOT / "research" / "statistical_downscaling_v6_repair_20260915"
    SCHEMA_SOURCE = DATA_ROOT / "experiments" / "04_terrain_features" / "FEATURE_SCHEMA.json"
SCHEMA_SOURCE_SHA256 = hashlib.sha256(SCHEMA_SOURCE.read_bytes()).hexdigest()
SOURCE_SCHEMA = json.loads(SCHEMA_SOURCE.read_text(encoding="utf-8"))
ALL_COLUMNS = list(SOURCE_SCHEMA["all_columns"])

BASE_NAMES = ["u02p", "v02p", "U", "log1p_U", "unit_u", "unit_v", "direction_valid"]
TERRAIN_FEATURE_NAMES = [
    "terrain_height_sd",
    "terrain_relief_27_mean",
    "terrain_slope_mean",
    "terrain_tpi_9_mean",
]
DYNAMIC_FEATURE_NAMES = list(BASE_NAMES)
D02_T4_NAMES = list(BASE_NAMES) + list(TERRAIN_FEATURE_NAMES)
if _V8_D02_T4_NAMES is not None:
    assert D02_T4_NAMES == list(_V8_D02_T4_NAMES)
TERRAIN_COLUMN_INDICES = [ALL_COLUMNS.index(name) for name in TERRAIN_FEATURE_NAMES]
BASE_COLUMN_INDICES = [ALL_COLUMNS.index(name) for name in BASE_NAMES]

M_PERMUTATIONS = 999
BLOCK_MINUTES = 120
N_BOOT = 1000
ALPHA = 0.01
EPS = 1e-12

_WEIBULL_KS = np.geomspace(0.15, 1000.0, 15000)
_WEIBULL_CV = np.sqrt(np.expm1(gammaln(1.0 + 2.0 / _WEIBULL_KS) - 2.0 * gammaln(1.0 + 1.0 / _WEIBULL_KS)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_rows(rows: np.ndarray) -> str:
    arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    return sha256_bytes(np.sort(arr).tobytes())


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return sha256_bytes(arr.tobytes())


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def feature_block_v96(features: np.ndarray, rows: np.ndarray, names: list[str]) -> np.ndarray:
    """Extract named columns from the canonical full matrix by name only."""
    indices = [ALL_COLUMNS.index(name) for name in names]
    return np.asarray(features[np.asarray(rows, dtype=np.int64)][:, indices], dtype=np.float32)


def qa_deploy_v96(Q_hat: np.ndarray, a_hat: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical Q-a deployment: mu=sqrt(U^2+aQ), J=mu-U, S=sqrt((1-a)Q)."""
    q = np.maximum(np.asarray(Q_hat, dtype=float), 0.0)
    a = np.clip(np.asarray(a_hat, dtype=float), 0.0, 1.0)
    u = np.asarray(U, dtype=float)
    mu = np.sqrt(np.maximum(u * u + a * q, 0.0))
    j = mu - u
    s = np.sqrt(np.maximum((1.0 - a) * q, 0.0))
    return mu, j, s


def js_deploy_v96(U: np.ndarray, J_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical JS deployment: mu = U + J."""
    u = np.asarray(U, dtype=float)
    j = np.maximum(np.asarray(J_hat, dtype=float), 0.0)
    mu = u + j
    return mu, j


def get_1d_quadrature() -> Tuple[np.ndarray, np.ndarray]:
    z, w = np.polynomial.hermite.hermgauss(5)
    return z * np.sqrt(2.0), w / np.sqrt(np.pi)


def get_2d_quadrature() -> Tuple[np.ndarray, np.ndarray]:
    z, w = np.polynomial.hermite.hermgauss(3)
    z = z * np.sqrt(2.0)
    w = w / np.sqrt(np.pi)
    z1, z2 = np.meshgrid(z, z)
    w1, w2 = np.meshgrid(w, w)
    return np.stack([z1.ravel(), z2.ravel()], axis=1), (w1 * w2).ravel()


def weibull_params_from_moments(mu: np.ndarray, S: np.ndarray, J: np.ndarray) -> dict:
    mean = np.asarray(mu, float)
    spread = np.asarray(S, float)
    cv = spread / np.maximum(mean, 1e-12)
    shape = np.interp(cv, _WEIBULL_CV[::-1], _WEIBULL_KS[::-1])
    scale = mean / np.exp(gammaln(1.0 + 1.0 / shape))
    params = {"family": "weibull", "shape": shape, "scale": np.maximum(scale, 1e-10)}
    params.update(
        component_count=1,
        weights=np.ones(1),
        J_components=np.asarray(J, float)[:, None],
        mean_components=np.asarray(mu, float)[:, None],
    )
    return params


def compute_time_series_cdfs(
    J_nodes: list[np.ndarray],
    mu_nodes: list[np.ndarray],
    S_nodes: list[np.ndarray],
    weights: np.ndarray,
    truth: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate one representation predictive CDFs without unbounded memory allocations."""
    K = len(weights)
    nt, parents = truth["j_rows"].shape[:2]
    sg, jg = truth["speed_grid"], truth["j_grid"]
    fs = np.zeros((nt, parents, len(sg)), dtype=np.float32)
    fj = np.zeros((nt, parents, len(jg)), dtype=np.float32)
    for k in range(K):
        wk = float(weights[k])
        jk = np.asarray(J_nodes[k], dtype=float)
        muk = np.asarray(mu_nodes[k], dtype=float)
        sk = np.asarray(S_nodes[k], dtype=float)
        for t0 in range(0, nt, 24):
            t1 = min(nt, t0 + 24)
            sl = np.arange(t0 * parents, t1 * parents)
            params = weibull_params_from_moments(muk[sl], sk[sl], jk[sl])
            fb = parameter_cdf(params, sg)
            fs[t0:t1] += (wk * fb.reshape(t1 - t0, parents, len(sg))).astype(np.float32)
            fjb = (jk[sl, None] <= jg[None, :]).astype(np.float32)
            fj[t0:t1] += (wk * fjb.reshape(t1 - t0, parents, len(jg))).astype(np.float32)
    return fs, fj


def window_scores(F_s: np.ndarray, F_j: np.ndarray, truth: dict) -> dict:
    fs = np.asarray(F_s)
    fj = np.asarray(F_j)
    gs = truth["speed_rows"]
    gj = truth["j_rows"]
    sg = truth["speed_grid"]
    jg = truth["j_grid"]
    fw_s = fs.mean(axis=0, dtype=np.float64)
    fw_j = fj.mean(axis=0, dtype=np.float64)
    gw_s = gs.mean(axis=0, dtype=np.float64)
    gw_j = gj.mean(axis=0, dtype=np.float64)
    speed_excess = np.trapezoid((fw_s - gw_s) ** 2, sg, axis=1)
    speed_crps = speed_excess + np.trapezoid(gw_s * (1.0 - gw_s), sg, axis=1)
    j_excess = np.trapezoid((fw_j - gw_j) ** 2, jg, axis=1)
    j_crps = j_excess + np.trapezoid(gw_j * (1.0 - gw_j), jg, axis=1)
    return {
        "speed_crps": float(np.mean(speed_crps)),
        "J_crps": float(np.mean(j_crps)),
        "speed_w1": float(np.trapezoid(np.abs(fw_s - gw_s), sg, axis=1).mean()),
        "J_w1": float(np.trapezoid(np.abs(fw_j - gw_j), jg, axis=1).mean()),
        "speed_excess_crps": float(np.mean(speed_excess)),
        "J_excess_crps": float(np.mean(j_excess)),
    }


def _trapezoidal_weights(grid: np.ndarray) -> np.ndarray:
    w = np.zeros_like(grid, dtype=np.float64)
    w[0] = 0.5 * (grid[1] - grid[0])
    w[-1] = 0.5 * (grid[-1] - grid[-2])
    w[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return w


def circular_block_sums(D: np.ndarray, L: int) -> np.ndarray:
    """Precompute circular moving-block sums of length L for all start indices in [0, nt)."""
    nt = D.shape[0]
    if L >= nt:
        return np.repeat(D.sum(axis=0, keepdims=True), nt, axis=0)
    D_circ = np.concatenate([D, D[:L]], axis=0)
    cumsum = np.cumsum(D_circ, axis=0, dtype=np.float64)
    zero_pad = np.zeros((1,) + D.shape[1:], dtype=np.float64)
    cs = np.concatenate([zero_pad, cumsum], axis=0)
    return cs[L : nt + L] - cs[:nt]


def paired_contiguous_block_bootstrap(
    F_cand_s: np.ndarray,
    F_base_s: np.ndarray,
    G_s: np.ndarray,
    grid_s: np.ndarray,
    F_cand_j: np.ndarray,
    F_base_j: np.ndarray,
    G_j: np.ndarray,
    grid_j: np.ndarray,
    block_len: int = BLOCK_MINUTES,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    """Exact-Nt circular moving-block bootstrap (Politis & Romano 1992).

    Samples K = ceil(nt / L) circular block start indices s_k ~ Uniform(0, nt-1).
    Concatenates circular blocks ((s_k + ell) % nt) and truncates to exactly nt points,
    giving equal marginal inclusion probabilities without wrapped tail padding distortion.
    """
    nt, parents, _ = F_cand_s.shape
    L = min(int(block_len), nt)
    K = int(np.ceil(nt / L))
    L_rem = nt - (K - 1) * L

    D_cand_s = np.asarray(F_cand_s, dtype=np.float64) - np.asarray(G_s, dtype=np.float64)
    D_base_s = np.asarray(F_base_s, dtype=np.float64) - np.asarray(G_s, dtype=np.float64)
    D_cand_j = np.asarray(F_cand_j, dtype=np.float64) - np.asarray(G_j, dtype=np.float64)
    D_base_j = np.asarray(F_base_j, dtype=np.float64) - np.asarray(G_j, dtype=np.float64)

    w_s = _trapezoidal_weights(np.asarray(grid_s, dtype=np.float64))
    w_j = _trapezoidal_weights(np.asarray(grid_j, dtype=np.float64))

    b_sums_L_cs = circular_block_sums(D_cand_s, L)
    b_sums_L_bs = circular_block_sums(D_base_s, L)
    b_sums_rem_cs = circular_block_sums(D_cand_s, L_rem)
    b_sums_rem_bs = circular_block_sums(D_base_s, L_rem)

    b_sums_L_cj = circular_block_sums(D_cand_j, L)
    b_sums_L_bj = circular_block_sums(D_base_j, L)
    b_sums_rem_cj = circular_block_sums(D_cand_j, L_rem)
    b_sums_rem_bj = circular_block_sums(D_base_j, L_rem)

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, nt, size=(n_boot, K))

    speed = np.empty(n_boot, dtype=np.float64)
    gap = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        s = starts[b]
        if K > 1:
            cs_b = (b_sums_L_cs[s[:-1]].sum(axis=0) + b_sums_rem_cs[s[-1]]) / nt
            bs_b = (b_sums_L_bs[s[:-1]].sum(axis=0) + b_sums_rem_bs[s[-1]]) / nt
            cj_b = (b_sums_L_cj[s[:-1]].sum(axis=0) + b_sums_rem_cj[s[-1]]) / nt
            bj_b = (b_sums_L_bj[s[:-1]].sum(axis=0) + b_sums_rem_bj[s[-1]]) / nt
        else:
            cs_b = b_sums_rem_cs[s[-1]] / nt
            bs_b = b_sums_rem_bs[s[-1]] / nt
            cj_b = b_sums_rem_cj[s[-1]] / nt
            bj_b = b_sums_rem_bj[s[-1]] / nt
        speed[b] = ((bs_b ** 2) @ w_s).mean() - ((cs_b ** 2) @ w_s).mean()
        gap[b] = ((bj_b ** 2) @ w_j).mean() - ((cj_b ** 2) @ w_j).mean()

    return {
        "delta_speed_mean": float(speed.mean()),
        "delta_speed_ci_low": float(np.percentile(speed, 2.5)),
        "delta_speed_ci_high": float(np.percentile(speed, 97.5)),
        "p_candidate_speed": float(np.mean(speed > 0.0)),
        "delta_J_mean": float(gap.mean()),
        "delta_J_ci_low": float(np.percentile(gap, 2.5)),
        "delta_J_ci_high": float(np.percentile(gap, 97.5)),
        "p_candidate_J": float(np.mean(gap > 0.0)),
        "block_len": int(L),
        "block_count": int(K),
        "dropped_tail_rows": 0,
        "wrapped_tail_rows": 0,
        "bootstrap_mode": "exact_nt_circular_moving_block_bootstrap",
    }


def determine_pareto_status(delta_s_low: float, delta_s_high: float, delta_j_low: float, delta_j_high: float) -> str:
    """Use the current evidence-table contract, never infer noninferiority."""
    from v9_7_reporting_gate import ci_relation
    return ci_relation(delta_s_low, delta_s_high, delta_j_low, delta_j_high)


def fit_models_on_rows(labels_A: pd.DataFrame, features_A: np.ndarray, rows: np.ndarray, names: list[str] = D02_T4_NAMES) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    frame = labels_A.iloc[rows].reset_index(drop=True)
    if not np.isfinite(frame.Q.to_numpy(float)).all() or (frame.Q.to_numpy(float) <= 1e-12).any():
        raise ValueError('Q <= 1e-12 or nonfinite: undefined allocation target; explicit degenerate-state policy required')
    X = feature_block_v96(features_A, rows, names)
    y_q = np.log(np.maximum(frame.Q.to_numpy(float), 1e-4))
    y_a = logit(np.clip(frame.a.to_numpy(float), 1e-4, 1.0 - 1e-4))
    y_j = np.log(np.maximum(frame.J.to_numpy(float), 1e-4))
    y_s = np.log(np.maximum(frame.S.to_numpy(float), 1e-4))
    return {
        "logQ": fit_scalar(X, names, names, y_q, head=None, alpha=ALPHA),
        "logita": fit_scalar(X, names, names, y_a, head=None, alpha=ALPHA),
        "logJ": fit_scalar(X, names, names, y_j, head=None, alpha=ALPHA),
        "logS": fit_scalar(X, names, names, y_s, head=None, alpha=ALPHA),
    }


def optimize_a_const_from_oof(q_hat: np.ndarray, u: np.ndarray, j_true: np.ndarray, s_true: np.ndarray) -> Tuple[float, float]:
    q_hat = np.maximum(np.asarray(q_hat, float), 1e-8)
    u = np.asarray(u, float)
    j_true = np.asarray(j_true, float)
    s_true = np.asarray(s_true, float)
    scale_j = max(float(np.std(j_true)), 1e-6)
    scale_s = max(float(np.std(s_true)), 1e-6)

    def loss(a: float) -> float:
        _, j_hat, s_hat = qa_deploy_v96(q_hat, np.full_like(q_hat, a), u)
        return float(0.5 * np.mean(np.abs(j_hat - j_true)) / scale_j + 0.5 * np.mean(np.abs(s_hat - s_true)) / scale_s)

    result = minimize_scalar(loss, bounds=(0.01, 0.99), method="bounded", options={"xatol": 0.002})
    return float(result.x), float(result.fun)


def inner_folds_for_outer(level: str, tr_rows: np.ndarray, info: dict, n_inner: int = 3) -> list[dict]:
    """Deterministic inner cross-fitting partitions wholly inside outer-train."""
    rows = np.asarray(tr_rows, dtype=np.int64)
    times = rows // PARENT_COUNT
    parents = rows % PARENT_COUNT
    if level == "A_T":
        groups = [np.asarray(x, dtype=np.int64) for x in np.array_split(np.unique(times), n_inner)]
        group_kind = "time"
        group_values = groups
        is_val = lambda values: np.isin(times, values)
    else:
        groups = [np.asarray(x, dtype=np.int64) for x in np.array_split(np.unique(parents), n_inner)]
        group_kind = "parent"
        group_values = groups
        is_val = lambda values: np.isin(parents, values)
    result = []
    for k, values in enumerate(group_values, 1):
        val_mask = is_val(values)
        val_rows = rows[val_mask]
        fit_rows = rows[~val_mask]
        if len(fit_rows) == 0 or len(val_rows) == 0:
            raise AssertionError(f"empty inner fold {level}/{k}")
        result.append({
            "inner_fold": f"inner{k}",
            "group_kind": group_kind,
            "validation_groups": values.tolist(),
            "fit_rows": fit_rows,
            "validation_rows": val_rows,
        })
    return result


def calibrate_outer_fold(
    labels_A: pd.DataFrame,
    features_A: np.ndarray,
    level: str,
    fold_name: str,
    tr_rows: np.ndarray,
    te_rows: np.ndarray,
    info: dict,
    receipt_path: Path,
) -> Tuple[dict, dict]:
    """Fit inner OOF residual calibration using outer-train rows only."""
    tr_rows = np.asarray(tr_rows, dtype=np.int64)
    te_rows = np.asarray(te_rows, dtype=np.int64)
    if np.intersect1d(tr_rows, te_rows).size:
        raise AssertionError(f"outer train/test overlap in {level}/{fold_name}")
    inner = inner_folds_for_outer(level, tr_rows, info)
    residuals = {key: [] for key in ("logQ", "logita", "logJ", "logS")}
    inner_receipts = []
    for spec in inner:
        models = fit_models_on_rows(labels_A, features_A, spec["fit_rows"])
        X_val = feature_block_v96(features_A, spec["validation_rows"], D02_T4_NAMES)
        frame = labels_A.iloc[spec["validation_rows"]].reset_index(drop=True)
        truth = {
            "logQ": np.log(np.maximum(frame.Q.to_numpy(float), 1e-4)),
            "logita": logit(np.clip(frame.a.to_numpy(float), 1e-4, 1.0 - 1e-4)),
            "logJ": np.log(np.maximum(frame.J.to_numpy(float), 1e-4)),
            "logS": np.log(np.maximum(frame.S.to_numpy(float), 1e-4)),
        }
        predictions = {key: predict_scalar(models[key], X_val) for key in residuals}
        for key in residuals:
            residuals[key].append(truth[key] - predictions[key])
        inner_receipts.append({
            "inner_fold": spec["inner_fold"],
            "group_kind": spec["group_kind"],
            "validation_groups": spec["validation_groups"],
            "fit_row_count": int(len(spec["fit_rows"])),
            "validation_row_count": int(len(spec["validation_rows"])),
            "fit_rows_sha256": sha256_rows(spec["fit_rows"]),
            "validation_rows_sha256": sha256_rows(spec["validation_rows"]),
        })

    res = {key: np.concatenate(value) for key, value in residuals.items()}
    calibration_rows = np.concatenate([x["validation_rows"] for x in inner])
    if set(calibration_rows.tolist()) != set(tr_rows.tolist()):
        raise AssertionError(f"inner OOF rows do not cover outer train for {level}/{fold_name}")
    if np.intersect1d(calibration_rows, te_rows).size:
        raise AssertionError(f"calibration/test overlap in {level}/{fold_name}")

    qa_matrix = np.cov(np.column_stack([res["logQ"], res["logita"]]), rowvar=False, ddof=0)
    js_matrix = np.cov(np.column_stack([res["logJ"], res["logS"]]), rowvar=False, ddof=0)
    qa_matrix = np.asarray(qa_matrix, dtype=float) + np.eye(2) * 1e-12
    js_matrix = np.asarray(js_matrix, dtype=float) + np.eye(2) * 1e-12

    qhat_oof = np.exp(np.log(np.maximum(labels_A.iloc[calibration_rows].Q.to_numpy(float), 1e-4)) - res["logQ"])
    oof_frame = labels_A.iloc[calibration_rows].reset_index(drop=True)
    a_const, a_loss = optimize_a_const_from_oof(
        qhat_oof,
        oof_frame.U.to_numpy(float),
        oof_frame.J.to_numpy(float),
        oof_frame.S.to_numpy(float),
    )
    calibration = {
        "std_logQ": float(np.sqrt(qa_matrix[0, 0])),
        "std_logita": float(np.sqrt(qa_matrix[1, 1])),
        "covariance_QA": qa_matrix.tolist(),
        "cholesky_QA": np.linalg.cholesky(qa_matrix).tolist(),
        "std_logJ": float(np.sqrt(js_matrix[0, 0])),
        "std_logS": float(np.sqrt(js_matrix[1, 1])),
        "covariance_JS": js_matrix.tolist(),
        "cholesky_JS": np.linalg.cholesky(js_matrix).tolist(),
        "a_const": a_const,
        "a_const_oof_loss": a_loss,
        "sample_count": int(len(calibration_rows)),
    }
    receipt = {
        "receipt_id": f"OUTER_CALIBRATION_{level}_{fold_name}",
        "status": "AUTHORITATIVE_V9_6",
        "calibration_protocol": "outer-train -> inner OOF -> outer-test",
        "level": level,
        "fold": fold_name,
        "outer_train": {
            "row_count": int(len(tr_rows)),
            "min_time": int((tr_rows // PARENT_COUNT).min()),
            "max_time_exclusive": int((tr_rows // PARENT_COUNT).max() + 1),
            "parent_ids": sorted(set((tr_rows % PARENT_COUNT).tolist())),
            "rows_sha256": sha256_rows(tr_rows),
        },
        "outer_test": {
            "row_count": int(len(te_rows)),
            "min_time": int((te_rows // PARENT_COUNT).min()),
            "max_time_exclusive": int((te_rows // PARENT_COUNT).max() + 1),
            "parent_ids": sorted(set((te_rows % PARENT_COUNT).tolist())),
            "rows_sha256": sha256_rows(te_rows),
        },
        "inner_folds": inner_receipts,
        "calibration_rows": {
            "row_count": int(len(calibration_rows)),
            "rows_sha256": sha256_rows(calibration_rows),
            "outer_test_intersection_count": int(np.intersect1d(calibration_rows, te_rows).size),
            "equals_outer_train": bool(set(calibration_rows.tolist()) == set(tr_rows.tolist())),
        },
        "parameters": calibration,
        "source": {
            "data_contract": "V9_4_DATA_CONTRACT.json (read-only, unchanged)",
            "feature_schema_source_sha256": SCHEMA_SOURCE_SHA256,
            "fit_engine": "research/qa_framework_v8_20260916/v8_common.py",
            "v9_6_common_sha256": sha256_file(V96 / "v9_6_common.py"),
            "forbidden_full_a_calibration": True,
        },
    }
    write_json(receipt_path, receipt)
    return calibration, receipt


def make_within_test_permutation_maps(test_parents: list[int] | np.ndarray, m_permutations: int = M_PERMUTATIONS, seed_base: int = 202609190000) -> list[np.ndarray]:
    """Generate M derangements strictly within the held-out test parents.

    For each derangement pi on test_parents:
      1. pi is a bijection from test_parents to test_parents.
      2. pi(p) != p for all p in test_parents (zero fixed points).
      3. The multiset of terrain features {T_{pi(p)}} is bitwise identical to {T_p}.
    """
    p_arr = np.asarray(test_parents, dtype=np.int64)
    n_parents = len(p_arr)
    if n_parents < 2:
        raise ValueError(f"Cannot derange set of size {n_parents}")

    maps = []
    seen_hashes = set()
    for index in range(m_permutations):
        rng = np.random.default_rng(seed_base + index)
        while True:
            perm_indices = rng.permutation(n_parents)
            # Check derangement condition (no fixed points)
            if not np.any(perm_indices == np.arange(n_parents)):
                mapping = p_arr[perm_indices]
                h = sha256_array(mapping)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    maps.append(mapping)
                    break
    assert len(maps) == m_permutations
    return maps


def make_5fold_each_row_once_splits(time_start: int = 970, time_end: int = 3650, parent_count: int = PARENT_COUNT, k_folds: int = 5) -> list[dict]:
    """Partition A-train [970, 3650) into K contiguous time blocks.

    Guarantees:
      - Union of validation rows equals exactly the full [970, 3650) row set (96,480 rows).
      - Intersection between any two validation sets is strictly empty.
      - Every row appears in validation EXACTLY ONCE.
    """
    total_times = time_end - time_start  # 2680
    time_splits = np.array_split(np.arange(time_start, time_end), k_folds)
    splits = []
    all_val_rows = []
    for k, val_times in enumerate(time_splits):
        val_t_min, val_t_max = int(val_times[0]), int(val_times[-1]) + 1
        val_rows = np.concatenate([np.arange(t * parent_count, (t + 1) * parent_count) for t in val_times])
        all_val_rows.append(val_rows)
        # Train times are all times except val_times
        train_times = np.setdiff1d(np.arange(time_start, time_end), val_times)
        train_rows = np.concatenate([np.arange(t * parent_count, (t + 1) * parent_count) for t in train_times])
        splits.append({
            "fold_index": k,
            "val_time_start": val_t_min,
            "val_time_end": val_t_max,
            "val_time_count": len(val_times),
            "val_rows": val_rows,
            "train_rows": train_rows,
        })
    concatenated_val = np.concatenate(all_val_rows)
    expected_rows = np.arange(time_start * parent_count, time_end * parent_count)
    assert len(concatenated_val) == len(expected_rows) == 96480
    assert np.array_equal(np.sort(concatenated_val), expected_rows)
    return splits
