"""Frozen analysis contract for the Comment.

Every constant below is a frozen definition that the manuscript reports.  No
value here is fitted inside this repository: the pipeline is a fixed, audited
computation over the released archives.
"""
from __future__ import annotations

# --- case identities (permanent; do not rename) -----------------------------
CASES = ("A", "B", "C")
CASE_META = {
    "A": {"site": "Yakou", "event": "2015-09-24 06:00 to 2015-09-27 06:00 UTC", "dir": "A",
          "tag": "yakou20150924"},
    "B": {"site": "Yakou", "event": "2015-12-01 06:00 to 2015-12-04 06:00 UTC", "dir": "B",
          "tag": "yakou20151201"},
    "C": {"site": "Ebao",  "event": "2015-12-01 06:00 to 2015-12-04 06:00 UTC", "dir": "C",
          "tag": "ebao20151201"},
}

# --- archive and window contract --------------------------------------------
ARCHIVE_STATES = 4320          # one-minute archived states
ARCHIVE_START_UTC = "06:00"    # on the first event day; archive ends at 06:00 on the fourth day
EXCLUDED = (0, 970)            # pre-existing analysis mask, NOT the 6 h spin-up
ANALYSIS = (970, 3650)         # 2,680 states
HELDOUT = (3650, 4320)         # 670 states
assert ANALYSIS[1] - ANALYSIS[0] == 2680
assert HELDOUT[1] - HELDOUT[0] == 670
assert ANALYSIS[1] == HELDOUT[0]

# --- spatial contract -------------------------------------------------------
COARSE_EDGE = 6                # d02 cells retained per case
N_COARSE = COARSE_EDGE ** 2    # 36 contiguous coarse cells
BLOCK = 27                     # d05 cells per coarse cell along each axis
FINEDGE = COARSE_EDGE * BLOCK  # 162 fine cells along each axis
N_FINE_PER_COARSE = BLOCK ** 2  # 729

D02_KM = 3.0                   # coarse representation
D05_KM = 0.11111111111111111   # fine reference

# Full archived length and the derived reconstruction grid.  RAW_N is the
# tensor extent (0..4320); the analysis/held-out windows are sub-slices of it.
RAW_N = 4320
GRID = FINEDGE                 # 162 fine cells along each axis

# Refinement modes evaluated in the two-level comparison.
MODES = ("bilinear", "bicubic")

# Native coarse tensor edge (the released d02 field is a 27x27 replication of
# these 6x6 values, so a block mean recovers the parent vector exactly).
COARSE_EDGE_NATIVE = 6

# --- statistical contract ---------------------------------------------------
DDof = 0                       # population variance for sigma_s
SPEED_MASK = 1.0               # m/s; direction statistics need both speeds >= this
L1_TOLERANCE = 1e-6            # m/s; maximum-component coarse-vector tolerance

# --- interpolation archive level anchors -------------------------------------
# Mean aggregation gap and mean spread of the refined (predicted) fields, per
# mode, pooled over the three cases.  Bands are deliberately wider than the
# case-to-case spread so they never reject a legitimate archive, but a
# multiplicative corruption of the predicted moments (e.g. x0.1 or x100) moves
# the mean clear of them.  Observed pooled means: J ~0.024-0.046 (bilinear) and
# ~0.041-0.052 (bicubic); spread ~0.31-0.63.
ANCHORS_J_LEVEL = {
    "bilinear": (0.005, 0.15),
    "bicubic": (0.010, 0.20),
}
ANCHORS_SPREAD_LEVEL = {
    "bilinear": (0.08, 1.5),
    "bicubic": (0.10, 1.9),
}
L2_HALF = 0.5                  # "recovers less than half" threshold

# --- plotting contract ------------------------------------------------------
PLOT_SUBSAMPLE = 650           # Fig.2(b) display subsample, not the correlation
PLOT_SEED = 20260906           # fixed; the reported Spearman uses the full sample

# --- reproduction anchors quoted in the Comment -----------------------------
# Per-case narrow bands (not one shared band): a shared band cannot detect a
# single case drifting, which is exactly the failure mode this guards against.
ANCHORS_DT = {
    "A": (0.269, 0.273),   # archived 0.27129
    "B": (0.223, 0.227),   # archived 0.22525
    "C": (0.227, 0.231),   # archived 0.22900
}
ANCHORS_EVEC = (0.005, 0.015)

# --- per-case value anchors quoted in the Comment ---------------------------
# These make the previously print-only diagnostics into real assertions.
ANCHORS_ACCOUNTING = {
    # case: (dt_total, J_over_dt, E_vec, E_vec_over_J_percent, dtheta_median)
    "A": (0.2713, 1.014, 0.00832, 2.58, 0.0147),
    "B": (0.2253, 1.011, 0.01506, 6.34, 0.0026),
    "C": (0.2290, 1.019, 0.00481, 1.38, 0.0170),
}
ANCHORS_ACCOUNTING_TOL = {
    "dt_total": 0.0015, "J_over_dt": 0.006, "E_vec": 0.0015,
    "E_vec_over_J_percent": 0.15, "dtheta_median": 0.0015,
}
# Rank correlations with J, on the speed-masked sample the Methods declare.
# Masked values differ materially from the unmasked ones for scalar spread
# (0.42 -> 0.48, 0.36 -> 0.56, 0.20 -> 0.44), which is why the mask is part of
# the definition rather than a display choice.
ANCHORS_SPEARMAN = {
    # case: (J~transverse, J~sigma_s)
    "A": (0.95, 0.48),
    "B": (0.91, 0.56),
    "C": (0.84, 0.44),
}
ANCHORS_SPEARMAN_TOL = 0.005

# Level-1 conservation error, as quoted in the Comment.
ANCHORS_CONSERVATION = {
    "ordinary_lo": 0.2,          # m/s; archived ordinary vector RMSE >= 0.25
    "ordinary_hi": 0.7,          # m/s; archived ordinary vector RMSE <= 0.64
    "constrained_hi": 1e-13,     # m/s; archived constrained vector RMSE ~1e-15
}


def as_dict() -> dict:
    return {
        "cases": list(CASES),
        "archive_states": ARCHIVE_STATES,
        "excluded": list(EXCLUDED),
        "analysis": list(ANALYSIS),
        "heldout": list(HELDOUT),
        "coarse_cells": N_COARSE,
        "fine_cells_per_coarse": N_FINE_PER_COARSE,
        "d02_km": D02_KM,
        "d05_km": D05_KM,
        "ddof": DDof,
        "speed_mask": SPEED_MASK,
        "l1_tolerance": L1_TOLERANCE,
        "anchors_conservation": ANCHORS_CONSERVATION,
        "anchors_accounting": ANCHORS_ACCOUNTING,
        "anchors_spearman": ANCHORS_SPEARMAN,
        "plot_subsample": PLOT_SUBSAMPLE,
        "plot_seed": PLOT_SEED,
    }
