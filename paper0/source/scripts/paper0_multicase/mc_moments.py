"""Primary numeric implementations for the multicase extension.

Frozen estimand definitions (CLAUDE.md / calibration_evaluation_contract):
  |U_p|      parent vector magnitude (native d02 value of the parent cell)
  |u_bar_p|  magnitude of the mean fine-cell vector (equal-weight over 729)
  s_bar_p    mean fine-cell scalar speed
  J          = s_bar_p - |u_bar_p|  >= 0
  delta_dyn  = |u_bar_p| - |U_p|
  delta_scalar = s_bar_p - |U_p| = delta_dyn + J   (exact identity)
  S          population SD of the 729 fine-cell speeds (ddof=0)

The independent audit implementations live in audit_independent.py and must
NOT import numeric helpers from this module.
"""
from __future__ import annotations

import warnings

import numpy as np

import mc_config as C


# ---------------------------------------------------------------------------
# Block reduction (reshape path; float64 accumulation)
# ---------------------------------------------------------------------------
def _to64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def parent_block_means(field: np.ndarray, block: int = C.BLOCK) -> np.ndarray:
    """(n, 162, 162) -> (n, 36) equal-weight block means, row-major cells."""
    n = field.shape[0]
    g = field.shape[1] // block
    r = field.reshape(n, g, block, g, block).mean(axis=(2, 4))
    return r.reshape(n, g * g)


def fine_moments(u: np.ndarray, v: np.ndarray, start: int, stop: int,
                 chunk: int = 512) -> dict[str, np.ndarray]:
    """Per-state per-parent moments of the fine (d05) vector field.

    Returns dict of (n, 36) float64 arrays:
      s_bar, u_bar, v_bar, U_vec, J, S, var_par, var_perp, trC, J2nd
    """
    n = stop - start
    keys = ["s_bar", "u_bar", "v_bar", "U_vec", "J", "S",
            "var_par", "var_perp", "trC", "J2nd"]
    out = {k: np.zeros((n, C.N_PARENT)) for k in keys}
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        uu = _to64(u[start + c0:start + c1])
        vv = _to64(v[start + c0:start + c1])
        sp = np.hypot(uu, vv)
        ub = parent_block_means(uu)
        vb = parent_block_means(vv)
        sb = parent_block_means(sp)
        s2 = parent_block_means(sp * sp)
        suu = parent_block_means(uu * uu) - ub * ub
        svv = parent_block_means(vv * vv) - vb * vb
        suv = parent_block_means(uu * vv) - ub * vb
        uvec = np.hypot(ub, vb)
        sd = np.sqrt(np.maximum(s2 - sb * sb, 0.0))       # population SD, ddof=0
        trc = suu + svv
        with np.errstate(divide="ignore", invalid="ignore"):
            vpar = (ub * ub * suu + 2.0 * ub * vb * suv + vb * vb * svv) / (uvec * uvec)
        vpar = np.where(uvec > 1e-9, vpar, trc)
        with np.errstate(divide="ignore", invalid="ignore"):
            j2 = (trc - vpar) / (2.0 * uvec)
        j2 = np.where(uvec > 1e-9, j2, 0.0)
        sl = slice(c0, c1)
        out["s_bar"][sl] = sb
        out["u_bar"][sl] = ub
        out["v_bar"][sl] = vb
        out["U_vec"][sl] = uvec
        out["S"][sl] = sd
        out["var_par"][sl] = vpar
        out["var_perp"][sl] = trc - vpar
        out["trC"][sl] = trc
        out["J2nd"][sl] = j2
    out["J"] = out["s_bar"] - out["U_vec"]
    return out


def parent_vectors_d02(u02: np.ndarray, v02: np.ndarray, start: int, stop: int,
                       chunk: int = 1024) -> dict[str, np.ndarray]:
    """Native d02 parent vectors per state per cell -> u02p, v02p, mag (n, 36).

    The archived d02 field is a cellwise 27x27 replication of the native 6x6
    values, so block means recover the parent vectors exactly (verified by the
    Phase-0 tiling audit).
    """
    n = stop - start
    out = {k: np.zeros((n, C.N_PARENT)) for k in ["u02p", "v02p", "mag"]}
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        ub = parent_block_means(_to64(u02[start + c0:start + c1]))
        vb = parent_block_means(_to64(v02[start + c0:start + c1]))
        out["u02p"][c0:c1] = ub
        out["v02p"][c0:c1] = vb
        out["mag"][c0:c1] = np.hypot(ub, vb)
    return out


def scale_moments(u: np.ndarray, v: np.ndarray, start: int, stop: int,
                  k: int, chunk: int = 512) -> dict[str, np.ndarray]:
    """Scale-l moments: vector-average U/V over k x k blocks FIRST, then form
    scalar quantities within each 27x27 parent (frozen order of operations).

    At k = 27 there is exactly one block per parent, so J(l) = S(l) = 0 holds
    exactly by construction.
    Returns (n, 36) arrays: s_bar_l, U_vec (parent mean vector magnitude,
    scale-independent), J_l, S_l.
    """
    n = stop - start
    sub = C.BLOCK // k                       # sub-blocks per parent side
    out = {kk: np.zeros((n, C.N_PARENT)) for kk in ["s_bar_l", "U_vec", "J_l", "S_l"]}
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        uu = _to64(u[start + c0:start + c1])
        vv = _to64(v[start + c0:start + c1])
        m = uu.shape[0]
        g = C.GRID // k
        uk = uu.reshape(m, g, k, g, k).mean(axis=(2, 4))
        vk = vv.reshape(m, g, k, g, k).mean(axis=(2, 4))
        spk = np.hypot(uk, vk)               # block speeds at scale l
        # within-parent stats over the sub x sub block speeds
        sbl = spk.reshape(m, 6, sub, 6, sub).mean(axis=(2, 4)).reshape(m, 36)
        s2l = (spk * spk).reshape(m, 6, sub, 6, sub).mean(axis=(2, 4)).reshape(m, 36)
        ubp = uk.reshape(m, 6, sub, 6, sub).mean(axis=(2, 4)).reshape(m, 36)
        vbp = vk.reshape(m, 6, sub, 6, sub).mean(axis=(2, 4)).reshape(m, 36)
        uvec = np.hypot(ubp, vbp)
        sl = slice(c0, c1)
        out["s_bar_l"][sl] = sbl
        out["U_vec"][sl] = uvec
        out["J_l"][sl] = sbl - uvec
        out["S_l"][sl] = np.sqrt(np.maximum(s2l - sbl * sbl, 0.0))
    return out


# ---------------------------------------------------------------------------
# Metrics (frozen conventions from a4_error_decomposition.metrics)
# ---------------------------------------------------------------------------
def cell_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """RMSE / bias / centred RMSE / Pearson r / OLS slope on cell means."""
    p = np.asarray(pred, float)
    o = np.asarray(obs, float)
    if p.shape != o.shape:
        raise ValueError(f"metric shape mismatch: prediction {p.shape} vs reference {o.shape}")
    if not np.isfinite(p).all() or not np.isfinite(o).all():
        raise ValueError("metric inputs must be finite")
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    bias = float(np.mean(p - o))
    pc, oc = p - p.mean(), o - o.mean()
    crmse = float(np.sqrt(np.mean((pc - oc) ** 2)))
    p_var = float(np.var(p))
    o_var = float(np.var(o))
    p_constant = bool(np.all(p == p.flat[0]))
    o_constant = bool(np.all(o == o.flat[0]))
    if p_constant or o_constant or p_var <= 0.0 or o_var <= 0.0:
        # A constant vector has no rank/covariance information.  In
        # particular, never report the apparent slope of a constant predictor
        # against a non-constant reference, nor a slope against a constant
        # reference.  This is the fail-safe contract for all baselines.
        r = float("nan")
        slope = float("nan")
    else:
        denom = np.sqrt((pc * pc).sum() * (oc * oc).sum())
        r = float((pc * oc).sum() / denom)
        slope = float(np.polyfit(o, p, 1)[0])
    return {"rmse": rmse, "bias": bias, "crmse": crmse, "r": r, "slope": slope}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Constant-safe Pearson correlation used by every multicase metric."""
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError(f"correlation shape mismatch: {x.shape} vs {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be finite")
    if np.all(x == x.flat[0]) or np.all(y == y.flat[0]):
        return float("nan")
    if np.var(x) <= 0.0 or np.var(y) <= 0.0:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    return float((xc * yc).sum() /
                 np.sqrt((xc * xc).sum() * (yc * yc).sum()))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks and constant-vector fail-safe.

    scipy.stats.spearmanr is the reference implementation for ties.  The
    explicit variance guard also avoids scipy's warning/implementation detail
    for constant predictors and makes the contract obvious to callers.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError(f"correlation shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 1:
        x, y = x.reshape(-1), y.reshape(-1)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be finite")
    if (x.size < 2 or np.all(x == x.flat[0]) or np.all(y == y.flat[0]) or
            np.var(x) <= 0.0 or np.var(y) <= 0.0):
        return float("nan")
    from scipy.stats import spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = spearmanr(x, y)
    return float(result.statistic)


def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS with intercept; returns coefficients, R2, adjusted R2, RMSE and
    approximate slope intervals (frozen 1.984 * SE convention)."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y, float)
    n, k = X.shape
    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / (tss + 1e-30)
    dof = max(n - k - 1, 1)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    m = C.OLS_INTERVAL_MULTIPLIER
    return {
        "intercept": float(beta[0]),
        "coefficients": [float(b) for b in beta[1:]],
        "se": [float(s) for s in se[1:]],
        "ci_low": [float(b - m * s) for b, s in zip(beta[1:], se[1:])],
        "ci_high": [float(b + m * s) for b, s in zip(beta[1:], se[1:])],
        "r2": float(r2), "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "fitted": fitted, "n": int(n),
    }


def meteorological_from(u: np.ndarray, v: np.ndarray):
    """Frozen direction convention: (270 - degrees(atan2(v, u))) mod 360."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    speed = np.hypot(u, v)
    calm = speed <= C.CALM_CUTOFF_MPS
    flow_to = np.degrees(np.arctan2(v, u)) % 360.0
    frm = (270.0 - flow_to) % 360.0
    return (np.where(calm, np.nan, flow_to), np.where(calm, np.nan, frm), calm)


def sector_index(direction_from_deg: np.ndarray) -> np.ndarray:
    deg = np.asarray(direction_from_deg, float)
    out = np.full(deg.shape, -1, dtype=int)
    ok = np.isfinite(deg)
    out[ok] = (np.floor(((deg[ok] + 22.5) % 360.0) / 45.0).astype(int) % 8)
    return out


def block_resample(rng: np.random.Generator, n: int, length: int) -> np.ndarray:
    """Overlapping moving-block resampling (frozen legacy implementation)."""
    n_blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n - length + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + length) for s in starts])[:n]


def cell_rmse(obs_mean: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs_mean - pred) ** 2)))


# ---------------------------------------------------------------------------
# Terrain descriptors (frozen algorithms)
# ---------------------------------------------------------------------------
_SQRT2_INV = 1.0 / np.sqrt(2.0)
_DIRS = ((1, 0, 1.0), (1, 1, _SQRT2_INV), (0, 1, 1.0), (-1, 1, _SQRT2_INV),
         (-1, 0, 1.0), (-1, -1, _SQRT2_INV), (0, -1, 1.0), (1, -1, _SQRT2_INV))


def lambda_iso_block(z: np.ndarray, dx: float = C.DX_M, L: float | None = None) -> float:
    """Frozen isotropic frontal exposure: eight directions, non-wrapping,
    physical direction, cardinal weight 1, diagonal weight 1/sqrt(2),
    normalization dx / (L*L), average over the eight directions.

    Vectorized positive-increment sum: for direction d the union of all
    profile steps equals the set of in-block neighbour pairs (p, p+d), each
    visited exactly once by the frozen per-profile walk.
    """
    n = z.shape[0]
    if n < 2:
        return 0.0
    if L is None:
        L = n * dx
    scale = dx / (L * L)
    acc = 0.0
    for di, dj, w in _DIRS:
        src_i = slice(max(0, -di), n - max(0, di))
        src_j = slice(max(0, -dj), n - max(0, dj))
        dst_i = slice(max(0, di), n - max(0, -di))
        dst_j = slice(max(0, dj), n - max(0, -dj))
        inc = z[dst_i, dst_j] - z[src_i, src_j]
        acc += scale * w * float(np.sum(inc[inc > 0.0]))
    return acc / 8.0


def terrain_descriptors(hgt: np.ndarray) -> list[dict]:
    """Per-parent frozen descriptors from a (162,162) float64 height field."""
    rows = []
    for cid in range(C.N_PARENT):
        r, c = divmod(cid, 6)
        z = hgt[r * C.BLOCK:(r + 1) * C.BLOCK, c * C.BLOCK:(c + 1) * C.BLOCK]
        sigma = float(z.std(ddof=0))
        q25, q75 = np.percentile(z, [25.0, 75.0])
        rows.append({
            "cell_id": cid, "row": r, "col": c,
            "mean_elevation": float(z.mean()),
            "min_elevation": float(z.min()),
            "max_elevation": float(z.max()),
            "elevation_range": float(z.max() - z.min()),
            "sigma_h": sigma,
            "r_sigma": sigma / C.L_M,
            "terrain_iqr": float(q75 - q25),
            "r_iqr": float(q75 - q25) / C.L_M,
            "lambda_iso": lambda_iso_block(z),
        })
    return rows
