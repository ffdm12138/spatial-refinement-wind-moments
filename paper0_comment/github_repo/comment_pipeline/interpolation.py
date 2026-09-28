"""Bilinear / bicubic interpolation controls, with and without the
coarse-vector conservation constraint.

    bilinear, bicubic                      ordinary geometric refinement
    bilinear_constrained, bicubic_constrained
                                           + uniform coarse-block component
                                             increment restoring the native
                                             coarse block-mean vector

The constraint is the controlled baseline: it makes coarse-scale conservation
exact, and is used here to test whether conservation implies recovery of the
subgrid statistics J and sigma_s.  It is not a recommended algorithm.
"""
from __future__ import annotations

import numpy as np

from . import contract as C
from .states import block_means


# Keys that must be present, and the arrays that must each have the full
# (4320, 36) per-cell-time shape.  A collapsed or broadcast archive is a
# contract violation, not a smaller-but-valid input.
REQUIRED_KEYS = {
    "U02_mag",
    *(f"{m}_{sfx}{k}"
       for m in C.MODES
       for sfx in ("", "constrained_")
       for k in ("mean_scalar_speed", "spread", "J", "delta_scalar", "delta_dyn")),
}
PER_CELLTIME_KEYS = {k for k in REQUIRED_KEYS if k != "U02_mag"}


def validate_archive(iv: dict, case: str) -> list:
    """Structural problems of one interpolation archive (empty list == valid).

    Checks the key set, the (RAW_N, N_COARSE) shape of every per-cell-time
    field, finiteness, and the algebraic relation the archive stores directly:
    ``delta_scalar == delta_dyn + J``.  It also checks non-negative recovery
    moments and their audited mean-level bands.  The archive does not store the
    interpolated vector aggregate, so ``J == mean_scalar_speed - U02_mag`` is
    not a valid archive-level identity and is intentionally not asserted here.
    """
    bad = []
    missing = sorted(REQUIRED_KEYS - set(iv))
    if missing:
        return [f"missing keys: {missing}"]

    u02 = np.asarray(iv["U02_mag"], dtype=np.float64)
    if u02.shape != (C.RAW_N, C.N_COARSE):
        bad.append(f"U02_mag shape {u02.shape} != {(C.RAW_N, C.N_COARSE)}")
    if not np.isfinite(u02).all():
        bad.append("U02_mag not finite")
    if (u02 < 0).any():
        bad.append("U02_mag negative")

    for k in sorted(PER_CELLTIME_KEYS):
        a = np.asarray(iv[k], dtype=np.float64)
        if a.shape != (C.RAW_N, C.N_COARSE):
            bad.append(f"{k} shape {a.shape} != {(C.RAW_N, C.N_COARSE)}")
            continue
        if not np.isfinite(a).all():
            bad.append(f"{k} not finite")
    if bad:
        return bad

    for m in C.MODES:
        for sfx in ("", "constrained_"):
            pre = f"{m}_{sfx}"
            J = np.asarray(iv[pre + "J"], dtype=np.float64)
            dsc = np.asarray(iv[pre + "delta_scalar"], dtype=np.float64)
            ddy = np.asarray(iv[pre + "delta_dyn"], dtype=np.float64)
            e1 = float(np.abs(dsc - (ddy + J)).max())
            if e1 > 1e-9:
                bad.append(f"{pre}delta_scalar != delta_dyn + J (max {e1:.2e})")
            # J = S_s - S_v >= 0, where S_v is the magnitude of the mean vector
            # after refinement.  The archive does not store S_v directly, so the
            # check that matters here is non-negativity plus a floor on the
            # mean amplitude: a 10x rescale keeps every ratio internally
            # consistent but moves the absolute level out of the audited band.
            if (J < -1e-9).any():
                bad.append(f"{pre}negative J (min {float(J.min()):.3e})")
            lvl = float(J.mean())
            lo, hi = C.ANCHORS_J_LEVEL[m]
            if not (lo <= lvl <= hi):
                bad.append(f"{pre}mean J {lvl:.5f} outside [{lo}, {hi}]")
            sp = np.asarray(iv[pre + "spread"], dtype=np.float64)
            if (sp < -1e-9).any():
                bad.append(f"{pre}negative spread")
            spl = float(sp.mean())
            slo, shi = C.ANCHORS_SPREAD_LEVEL[m]
            if not (slo <= spl <= shi):
                bad.append(f"{pre}mean spread {spl:.5f} outside [{slo}, {shi}]")
    return bad


def load(data_root, case: str) -> dict:
    d = data_root / C.CASE_META[case]["dir"]
    z = np.load(d / "interpolation_cell_time_full.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def constrained_correction(fine: np.ndarray, native: np.ndarray) -> np.ndarray:
    """Add a uniform per-block component increment so that block means match.

    `fine`  : (n, 162, 162) interpolated component
    `native`: (n, 36)       native coarse component
    """
    n = fine.shape[0]
    ub = block_means(fine)
    off = (native - ub).reshape(n, C.COARSE_EDGE, C.COARSE_EDGE)
    off = np.repeat(np.repeat(off, C.BLOCK, axis=1), C.BLOCK, axis=2)
    return fine + off


def conservation_residual(u_pred: np.ndarray, v_pred: np.ndarray,
                          u_native: np.ndarray, v_native: np.ndarray):
    """Maximum component residual and the vector RMSE of the block means.

    RMSE = sqrt(<|| blockmean(pred) - native ||^2>) and is the common metric
    plotted in Fig.3(a) for all four controls.
    """
    du = block_means(u_pred) - u_native
    dv = block_means(v_pred) - v_native
    max_comp = float(max(np.abs(du).max(), np.abs(dv).max()))
    rmse = float(np.sqrt((du ** 2 + dv ** 2).mean()))
    return max_comp, rmse


def recovery_ratios(pred: dict, ref: dict) -> dict:
    return {k: float(pred[k].mean() / ref[k].mean())
            for k in ("delta_scalar", "J", "spread")}


def reference(m: dict, window) -> dict:
    sl = slice(*window)
    return {"delta_scalar": m["delta_scalar"][sl],
            "J": m["J"][sl],
            "spread": m["S"][sl]}


def predicted_states(u: np.ndarray, v: np.ndarray, native_mag: np.ndarray) -> dict:
    """Block moments of a predicted fine component field."""
    ub, vb = block_means(u), block_means(v)
    sp = np.hypot(u, v)
    sb = block_means(sp)
    s2 = block_means(sp * sp)
    return {
        "u_bar": ub, "v_bar": vb, "s_bar": sb,
        "sd": np.sqrt(np.maximum(s2 - sb ** 2, 0.0)),
        "u_vec": np.hypot(ub, vb),
        "delta_scalar": sb - native_mag,
        "J": sb - np.hypot(ub, vb),
        "spread": np.sqrt(np.maximum(s2 - sb ** 2, 0.0)),
    }
