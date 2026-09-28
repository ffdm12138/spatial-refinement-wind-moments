"""Three-state accounting (S_c, S_v, S_s) and resolved-vector consistency.

State definitions (exact, no assumption that S_c = S_v):

    S_c = |U_c|                  native coarse vector magnitude
    S_v = |mean(u_f)|            fine-grid vector aggregate magnitude
    S_s = mean(|u_f|)            fine-grid scalar mean

    Delta_mag = S_v - S_c        signed resolved magnitude mismatch
    J         = S_s - S_v >= 0   subgrid aggregation gap
    Delta_total = Delta_mag + J = S_s - S_c
    E_vec     = |mean(u_f) - U_c|   full first-order resolved vector mismatch
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import contract as C


def load(data_root, case: str) -> dict:
    """Load the released per-case moment archive."""
    d = data_root / C.CASE_META[case]["dir"]
    z = np.load(d / "moments_cell_time.npz")
    return {k: z[k].astype(np.float64) for k in z.files if not k.startswith("_")}


REQUIRED_KEYS = ("u_bar", "v_bar", "s_bar", "U_vec", "u02p", "v02p",
                 "U02_mag", "S", "var_par", "var_perp", "trC", "J",
                 "delta_dyn", "delta_scalar")


def validate_archive(m: dict, case: str = "") -> list[str]:
    """Structural and internal-consistency checks on a loaded moment archive.

    These are the checks whose absence let an injected 100 m/s error in the
    native vector, or a negative recovery moment, pass unnoticed.  Returns a
    list of problem strings; an empty list means the archive is well formed.

    Every assertion below is a *definition* the Comment relies on, not a
    tolerance on the reported numbers:

      * the fourteen required variables exist with shape ``(4320, 36)``;
      * every variable is finite;
      * the aggregation gap is non-negative, ``J >= 0``;
      * the scalar spread is non-negative, ``S >= 0``;
      * ``J = s_bar - U_vec`` and ``delta_scalar = s_bar - U02_mag`` hold to
        floating-point precision (a stored column must agree with the stored
        primaries it is defined from).
    """
    tag = f"{case}: " if case else ""
    problems: list[str] = []

    for k in REQUIRED_KEYS:
        if k not in m:
            problems.append(f"{tag}missing variable {k!r}")
    if problems:
        return problems

    for k in REQUIRED_KEYS:
        v = m[k]
        if v.ndim != 2 or v.shape != (C.ARCHIVE_STATES, C.N_COARSE):
            problems.append(f"{tag}{k} shape {v.shape} != "
                            f"({C.ARCHIVE_STATES}, {C.N_COARSE})")
        elif not np.isfinite(v).all():
            problems.append(f"{tag}{k} has {int((~np.isfinite(v)).sum())} "
                            f"non-finite entries")

    if any("shape" in p or "non-finite" in p for p in problems):
        return problems

    if float(m["J"].min()) < -1e-6:
        problems.append(f"{tag}J minimum {float(m['J'].min()):.3e} < 0")
    if float(m["S"].min()) < -1e-6:
        problems.append(f"{tag}spread minimum {float(m['S'].min()):.3e} < 0")
    if float(m["var_perp"].min()) < -1e-6:
        problems.append(f"{tag}var_perp minimum "
                        f"{float(m['var_perp'].min()):.3e} < 0")
    if float(m["var_par"].min()) < -1e-6:
        problems.append(f"{tag}var_par minimum "
                        f"{float(m['var_par'].min()):.3e} < 0")

    nonneg = {
        "U02_mag": m["U02_mag"], "U_vec": m["U_vec"], "s_bar": m["s_bar"],
    }
    for k, v in nonneg.items():
        if float(v.min()) < 0:
            problems.append(f"{tag}{k} minimum {float(v.min()):.3e} < 0")

    # Magnitudes must agree with the components they are derived from.  This is
    # the check that catches a corrupt native vector.
    mag_err = float(np.abs(np.hypot(m["u02p"], m["v02p"]) - m["U02_mag"]).max())
    if mag_err > 1e-6:
        problems.append(f"{tag}S_c != |(u02p, v02p)| by {mag_err:.3e}")
    vec_err = float(np.abs(np.hypot(m["u_bar"], m["v_bar"]) - m["U_vec"]).max())
    if vec_err > 1e-6:
        problems.append(f"{tag}S_v != |(u_bar, v_bar)| by {vec_err:.3e}")

    # Stored derived columns must equal their definitions.
    j_def = float(np.abs((m["s_bar"] - m["U_vec"]) - m["J"]).max())
    if j_def > 1e-6:
        problems.append(f"{tag}J != s_bar - U_vec by {j_def:.3e}")
    dt_def = float(np.abs((m["s_bar"] - m["U02_mag"]) - m["delta_scalar"]).max())
    if dt_def > 1e-6:
        problems.append(f"{tag}delta_scalar != s_bar - U02_mag by {dt_def:.3e}")
    dm_def = float(np.abs((m["U_vec"] - m["U02_mag"]) - m["delta_dyn"]).max())
    if dm_def > 1e-6:
        problems.append(f"{tag}delta_dyn != U_vec - U02_mag by {dm_def:.3e}")

    # Variance decomposition.  For the 729 fine vectors of a cell,
    #   var_par + var_perp = tr(C) = E|u|^2 - |E u|^2 = S^2 + s_bar^2 - S_v^2,
    # because E|u|^2 = S^2 + s_bar^2 and |E u|^2 = S_v^2.  This identity ties the
    # two stored variance columns to the stored S, s_bar and U_vec, so a
    # multiplicative corruption of var_perp alone (which leaves every rank
    # correlation unchanged) cannot survive it.
    trc_def = float(np.abs(m["trC"] - (m["var_par"] + m["var_perp"])).max())
    if trc_def > 1e-6:
        problems.append(f"{tag}trC != var_par + var_perp by {trc_def:.3e}")
    var_ident = float(np.abs(
        (m["var_par"] + m["var_perp"])
        - (m["S"] ** 2 + m["s_bar"] ** 2 - m["U_vec"] ** 2)).max())
    if var_ident > 1e-9:
        problems.append(f"{tag}var_par + var_perp != S^2 + s_bar^2 - U_vec^2 "
                        f"by {var_ident:.3e}")

    return problems


def block_means(field: np.ndarray) -> np.ndarray:
    """(n, 162, 162) -> (n, 36) equal-weight block means, row-major.

    A coarse cell is the arithmetic mean of its 27x27 fine cells; two-dimensional
    block topology is preserved.  This is the frozen aggregation operator.
    """
    n = field.shape[0]
    r = field.reshape(n, C.COARSE_EDGE, C.BLOCK, C.COARSE_EDGE, C.BLOCK).mean(axis=(2, 4))
    return r.reshape(n, C.N_COARSE)


def ang_diff_deg(u1, v1, u2, v2) -> np.ndarray:
    d = np.arctan2(v1, u1) - np.arctan2(v2, u2)
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return np.abs(np.degrees(d))


def accounting(m: dict) -> dict:
    """Domain means over the analysis window plus held-out consistency."""
    sc, sv, ss = m["U02_mag"], m["U_vec"], m["s_bar"]
    dt, dmag, J = m["delta_scalar"], m["delta_dyn"], m["J"]
    du, dv = m["u_bar"] - m["u02p"], m["v_bar"] - m["v02p"]
    e_vec = np.hypot(du, dv)

    a, b = C.ANALYSIS
    sl = slice(a, b)
    ident = float(np.abs(dt - (dmag + J)).max())
    assert ident < 5e-3, ("delta_total = delta_mag + J identity", ident)
    assert float(J.min()) > -1e-6, "J must be non-negative"

    ev = slice(*C.HELDOUT)
    mask = np.minimum(sv, sc) >= C.SPEED_MASK
    jm, dtm = J[sl].mean(), dt[sl].mean()
    return {
        "n_states": b - a,
        "dt_mean": float(dtm),
        "dmag_mean": float(dmag[sl].mean()),
        "J_mean": float(jm),
        "J_abs_mean": float(np.abs(Jm := J[sl]).mean()),
        "J_over_dt": float(jm / dtm),
        "abs_dmag_max": float(np.abs(dmag[sl]).max()),
        "identity_max_abs": ident,
        "E_vec_mean": float(e_vec[ev].mean()),
        "E_vec_over_J": float(e_vec[ev].mean() / J[ev].mean()),
        # Evaluation-window counterparts of the analysis-window means above.
        # Both windows must be available explicitly, otherwise a panel that
        # mixes J_mean (analysis) with E_vec_mean (held-out) silently reports a
        # cross-window ratio instead of the concurrent one.
        "J_mean_eval": float(J[ev].mean()),
        "dt_mean_eval": float(dt[ev].mean()),
        "dmag_mean_eval": float(dmag[ev].mean()),
        "E_vec_over_J_eval_percent": float(
            100 * e_vec[ev].mean() / J[ev].mean()),
        "dtheta_median_masked": float(np.median(
            ang_diff_deg(m["u_bar"], m["v_bar"], m["u02p"], m["v02p"])[ev][mask[ev]])),
        "mask_frac": float(mask[ev].mean()),
    }


def correlations(m: dict, masked: bool = True) -> dict:
    """Spearman rank correlations of J against the transverse approximation.

    Computed on the 3,350-state retained descriptive sample ``[970, 4320)``,
    restricted to the samples the Comment's Methods declare: cell-times whose
    fine-grid mean vector speed satisfies ``S_v >= SPEED_MASK``.  The approximation
    ``J ~ sigma_perp^2 / (2 S_v)`` divides by ``S_v``, so the low-speed tail is
    exactly where it is undefined; the mask is part of the diagnostic, not a
    cosmetic filter.

    Set ``masked=False`` to obtain the unmasked values for comparison only; the
    reported figures and text use the masked ones.  Fig.2(b) draws a 650-point
    fixed-seed subsample of the masked sample for display.
    """
    a = C.ANALYSIS[0]
    sl = slice(a, None)
    J, S, sv = m["J"][sl], m["S"][sl], m["U_vec"][sl]
    perp = m["var_perp"][sl] / (2 * sv)
    par = m["var_par"][sl] / (2 * sv)

    keep = (sv >= C.SPEED_MASK) if masked else np.ones_like(sv, dtype=bool)

    def corr(x, y, method):
        xa, ya = x[keep].ravel(), y[keep].ravel()
        return float(pd.Series(xa).corr(pd.Series(ya), method=method))

    return {
        "sp_J_perp": corr(J, perp, "spearman"),
        "sp_J_par": corr(J, par, "spearman"),
        "sp_J_S": corr(J, S, "spearman"),
        "pe_J_perp": corr(J, perp, "pearson"),
        "masked": bool(masked),
        "n_samples": int(keep.sum()),
        "n_total": int(keep.size),
    }


def summary_csv(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_csv(index=False)
