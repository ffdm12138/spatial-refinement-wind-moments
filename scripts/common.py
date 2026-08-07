"""Shared preprocessing for the 2026-08-07 refactor analyses.

Reads the raw archived 1-min tensors (sha256-verified against
outputs_prepared_sealed_holdout_manifest.json) and computes per-state,
per-parent-cell moment statistics used by a1..a8.

Analysis window: tensor indices [970, 4320) = 3350 states.
Training: [970, 3650) = 2680 states. Evaluation: [3650, 4320) = 670 states.
Parent cells: 6x6 grid, row-major cell_id, each a 27x27 block of d05 cells.

The raw tensors are archived at the National Tibetan Plateau / Third Pole
Environment Data Center (Feng and Li, 2026; doi:10.11888/atmos.tpdc.303495);
download them and point NPJ_RAW_DIR at the directory containing the one-minute
tensors (the sha256 of {U10,V10}_d0{2,5}.pt match the sealed manifest
raw_sha256). This file intentionally contains no machine-specific paths.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

# internal layout: source/scripts/refactor_20260807/common.py -> parents[3] is
# the workspace root; release layout: scripts/common.py (flat) -> parents[1] is
# the package root (which carries source_data/ in the release layout)
_PROJ = Path(__file__).resolve().parents
PROJECT_ROOT = str(_PROJ[1] if (_PROJ[1] / "source_data").is_dir() else _PROJ[3])
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "source", "evidence", "refactor_20260807")
RAW_DIR = os.environ.get("NPJ_RAW_DIR", "")
TRAIN_SLICE = (970, 3650)
EVAL_SLICE = (3650, 4320)
N_CELLS = 36
BLOCK = 27  # 27x27 d05 cells per parent cell

_cached: dict | None = None


def load_tensors() -> dict:
    """Load U10/V10 for d02/d05 once; cache."""
    global _cached
    if _cached is not None:
        return _cached
    if not RAW_DIR:
        raise RuntimeError(
            "NPJ_RAW_DIR is not set: download the archived WRF dataset "
            "(doi:10.11888/atmos.tpdc.303495) and point NPJ_RAW_DIR at the "
            "directory containing the one-minute tensors")
    out = {}
    for name in ["U10_d02", "U10_d05", "V10_d02", "V10_d05"]:
        path = os.path.join(RAW_DIR, f"{name}.pt")
        import torch
        t = torch.load(path, map_location="cpu", weights_only=True)
        a = np.asarray(t.numpy(), dtype=np.float32)  # (4320, 162, 162)
        assert a.shape == (4320, 162, 162), a.shape
        out[name] = a
        del t
    _cached = out
    return out


def cell_blocks() -> list[tuple[int, int]]:
    """Return [(r0, c0), ...] top-left corners of the 36 parent cells, row-major."""
    return [(r * BLOCK, c * BLOCK) for r in range(6) for c in range(6)]


def per_state_parent_stats(u: np.ndarray, v: np.ndarray,
                           start: int, stop: int) -> dict:
    """For a window [start, stop), compute per-state per-cell statistics of the
    fine-grid vector field u, v (both 162x162 fields, time-major).

    Returns dict of (n_states, 36) float64 arrays:
      U_avg    : block-mean scalar speed
      U_vec    : magnitude of block-mean vector
      J        : U_avg - U_vec  (exact Jensen gap)
      sd       : block std of scalar speed
      var_par  : sigma_par^2 along the mean-wind direction
      var_perp : sigma_perp^2 across the mean-wind direction
      trC      : trace of the vector covariance
      J2nd     : (trC - sigma_par^2) / (2 U_vec) = sigma_perp^2 / (2 U_vec)
    """
    n = stop - start
    res = {k: np.zeros((n, N_CELLS), dtype=np.float64)
           for k in ["U_avg", "U_vec", "J", "sd", "var_par", "var_perp",
                     "trC", "J2nd", "u_bar", "v_bar"]}
    sp = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
    for i, (r0, c0) in enumerate(cell_blocks()):
        sl = (slice(start, stop), slice(r0, r0 + BLOCK), slice(c0, c0 + BLOCK))
        ub = u[sl].mean(axis=(1, 2))
        vb = v[sl].mean(axis=(1, 2))
        sa = sp[sl]
        res["U_avg"][:, i] = sa.mean(axis=(1, 2))
        res["sd"][:, i] = sa.std(axis=(1, 2))
        Uv = np.sqrt(ub ** 2 + vb ** 2)
        res["U_vec"][:, i] = Uv
        res["u_bar"][:, i] = ub
        res["v_bar"][:, i] = vb
        # covariance terms
        du = u[sl] - ub[:, None, None]
        dv = v[sl] - vb[:, None, None]
        suu = (du * du).mean(axis=(1, 2))
        suv = (du * dv).mean(axis=(1, 2))
        svv = (dv * dv).mean(axis=(1, 2))
        trC = suu + svv
        with np.errstate(divide="ignore", invalid="ignore"):
            var_par = (ub ** 2 * suu + 2 * ub * vb * suv + vb ** 2 * svv) / (Uv ** 2)
            var_par = np.where(Uv > 1e-9, var_par, trC)
            J2 = (trC - var_par) / (2 * Uv)
            J2 = np.where(Uv > 1e-9, J2, 0.0)
        res["var_par"][:, i] = var_par
        res["var_perp"][:, i] = trC - var_par
        res["trC"][:, i] = trC
        res["J2nd"][:, i] = J2
    res["J"][:] = res["U_avg"] - res["U_vec"]
    return res


def parent_d02_vector(u02: np.ndarray, v02: np.ndarray,
                      start: int, stop: int) -> np.ndarray:
    """Block-mean d02 vectors per state per cell -> (n_states, 36, 2).
    The archived d02 field is a cellwise (27x27) replication of the native
    6x6 values, so block means recover the native parent vectors."""
    n = stop - start
    out = np.zeros((n, N_CELLS, 2), dtype=np.float64)
    for i, (r0, c0) in enumerate(cell_blocks()):
        sl = (slice(start, stop), slice(r0, r0 + BLOCK), slice(c0, c0 + BLOCK))
        out[:, i, 0] = u02[sl].mean(axis=(1, 2))
        out[:, i, 1] = v02[sl].mean(axis=(1, 2))
    return out


def state_arrays(win: str = "train") -> dict:
    """Full per-state per-cell arrays for the training or evaluation window.

    Returns dict with:
      delta_scalar (n,36) = U_avg_d05 - |vbar_d02|   (scalar increment target)
      delta_dyn    (n,36) = |vbar_d02| - U_vec_d05   (mean-vector-magnitude term)
      jensen       (n,36) = J (exact, from d05)
      spread       (n,36) = block sd of scalar speed (within-parent spread target)
      speed        (n,36) = U_vec_d05 (parent-cell mean vector magnitude)
      ...
    """
    start, stop = TRAIN_SLICE if win == "train" else EVAL_SLICE
    T = load_tensors()
    d5 = per_state_parent_stats(T["U10_d05"], T["V10_d05"], start, stop)
    v02 = parent_d02_vector(T["U10_d02"], T["V10_d02"], start, stop)
    u02mag = np.sqrt(v02[..., 0] ** 2 + v02[..., 1] ** 2)
    out = dict(d5)
    out["delta_scalar"] = d5["U_avg"] - u02mag
    # signed mean-vector-magnitude difference; sign convention follows the
    # manuscript (locked domain mean -0.004 m s-1): |vbar_d05| - |vbar_d02|
    out["delta_dyn"] = d5["U_vec"] - u02mag
    out["spread"] = d5["sd"]
    out["speed"] = d5["U_vec"]
    out["vbar_d02"] = u02mag
    out["wind_dir"] = np.degrees(np.arctan2(v02[..., 1], v02[..., 0])) % 360.0
    return out


def domain_mean_speed(T, start, stop):
    """Per-state domain-mean scalar wind speed (m s-1) over the d05 grid."""
    u = T["U10_d05"][start:stop].astype(np.float64)
    v = T["V10_d05"][start:stop].astype(np.float64)
    return np.sqrt(u ** 2 + v ** 2).mean(axis=(1, 2))


def domain_mean_vector(T, start, stop):
    """Per-state domain-mean wind vector (u, v) over the d05 grid."""
    u = T["U10_d05"][start:stop].astype(np.float64).mean(axis=(1, 2))
    v = T["V10_d05"][start:stop].astype(np.float64).mean(axis=(1, 2))
    return u, v
