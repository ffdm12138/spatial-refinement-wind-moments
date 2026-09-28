"""Phases 7–8 — terrain descriptors and terrain–moment organization
(PLAN §12–13).

Descriptors are computed independently per case from that case's HGT_d05
with the FROZEN algorithms (r_sigma, r_IQR, lambda_iso: eight directions,
non-wrapping, physical direction, cardinal/diagonal normalization).  No new
descriptor screening is performed.

Relationships are calibration-only descriptive fits of the frozen descriptor
identities:
    delta_scalar ~ r_sigma      J ~ r_sigma
    S ~ lambda_iso              S ~ r_IQR + lambda_iso
with approximate-OLS intervals (frozen 1.984*SE convention) plus a
moving-block bootstrap CI (per-generation master seed, block 120, 2000
replicates).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import mc_moments as M  # noqa: E402
import phase01_moments as P1  # noqa: E402

RELATIONS = [
    ("delta_scalar", ("r_sigma",), "delta_vs_r_sigma"),
    ("J", ("r_sigma",), "J_vs_r_sigma"),
    ("S", ("lambda_iso",), "S_vs_lambda_iso"),
    ("S", ("r_iqr", "lambda_iso"), "S_vs_r_iqr_plus_lambda_iso"),
    ("S", ("r_iqr",), "S_vs_r_iqr"),
]


def descriptors(case: str) -> pd.DataFrame:
    return pd.read_csv(C.case_dir(case) / "terrain_descriptors.csv")


def run_descriptors(case: str) -> pd.DataFrame:
    print(f"[phase07] case {case}: terrain descriptors")
    hgt = C.load_case_hgt(case)
    rows = M.terrain_descriptors(hgt)
    df = pd.DataFrame(rows)
    C.write_df(C.case_dir(case) / "terrain_descriptors.csv", df.round(12), case=case)
    qs = [5, 25, 50, 75, 95]
    summary = {
        "domain": {
            "mean_elevation": float(hgt.mean()),
            "min_elevation": float(hgt.min()),
            "max_elevation": float(hgt.max()),
            "elevation_range": float(hgt.max() - hgt.min()),
            "terrain_sd": float(hgt.std(ddof=0)),
            "terrain_iqr": float(np.percentile(hgt, 75) - np.percentile(hgt, 25)),
        },
        "descriptor_distributions": {
            d: {"min": float(df[d].min()), "max": float(df[d].max()),
                "mean": float(df[d].mean()),
                **{f"p{q:02d}": float(np.percentile(df[d], q)) for q in qs}}
            for d in ("r_sigma", "r_iqr", "lambda_iso", "sigma_h")},
        "algorithm_contract": {
            "lambda_iso": ("8 directions, non-wrapping, physical direction, "
                           "cardinal weight 1 / diagonal 1/sqrt(2), "
                           "normalization dx/(L*L), average over directions"),
            "r_sigma": "population SD of tile height / L (ddof=0), L = 3000 m",
            "r_iqr": "(p75 - p25) of tile height / L",
        },
    }
    C.write_json(C.case_dir(case) / "terrain_summary.json", summary, case=case)
    return df


def regime_comparison(cases: list[str]) -> None:
    """cross_case/terrain_regime_comparison.csv + A-range overlap flags."""
    a_rng = C.frozen_a_calibration_ranges()
    rows = []
    descs = {c: descriptors(c) for c in cases}
    for case in cases:
        df = descs[case]
        row = {"case": case, "label": C.CASES[case]["label"]}
        for d in ("r_sigma", "r_iqr", "lambda_iso"):
            row[f"{d}_min"] = df[d].min()
            row[f"{d}_max"] = df[d].max()
            row[f"{d}_mean"] = df[d].mean()
            key = {"r_sigma": "r_sigma", "r_iqr": "r_iqr",
                   "lambda_iso": "lambda_iso"}[d]
            lo, hi = a_rng[key]
            inside = ((df[d] >= lo) & (df[d] <= hi)).sum()
            row[f"{d}_cells_in_A_calibration_range"] = int(inside)
            row[f"{d}_cells_extrapolating"] = int(36 - inside)
        row["mean_elevation"] = df["mean_elevation"].mean()
        row["elevation_range_domain"] = df["max_elevation"].max() - df["min_elevation"].min()
        rows.append(row)
    out = pd.DataFrame(rows)
    # explicit regime verdict columns for B vs C (same wording as the plan)
    if {"B", "C"}.issubset(set(cases)):
        b, c = descs["B"], descs["C"]
        for d in ("r_sigma", "lambda_iso"):
            lob, hib = float(b[d].min()), float(b[d].max())
            loc, hic = float(c[d].min()), float(c[d].max())
            overlap = (min(hib, hic) - max(lob, loc)) / (
                max(hib, hic) - min(lob, loc) + 1e-30)
            out[f"BC_{d}_range_overlap_frac"] = float(max(overlap, 0.0))
    C.write_df(C.ext_dir("cross_case") / "terrain_regime_comparison.csv",
               out.round(10), status_value=C.combined_status(*cases))


def run_relationships(case: str) -> None:
    print(f"[phase08] case {case}: terrain relationships (calibration only)")
    cdir = C.case_dir(case)
    df = descriptors(case)
    tab = P1.load_moments(case)
    a, b = C.CAL_SLICE
    targets = {"delta_scalar": tab["delta_scalar"][a:b],
               "J": tab["J"][a:b], "S": tab["S"][a:b]}
    cm = {k: v.mean(axis=0) for k, v in targets.items()}
    X = {d: df[d].to_numpy(float) for d in ("r_sigma", "r_iqr", "lambda_iso")}

    rng = C.rng_for(f"relationship_boot_{case}")
    n_cal = b - a
    boot_idx = [M.block_resample(rng, n_cal, C.PRIMARY_BLOCK)
                for _ in range(C.N_REPLICATES)]

    rows = []
    metrics_json = {}
    for target, feats, name in RELATIONS:
        Xmat = np.column_stack([X[f] for f in feats])
        fit = M.ols_fit(Xmat, cm[target])
        pear = (M.pearson(X[feats[0]], cm[target])
                if len(feats) == 1 else
                M.pearson(fit["fitted"], cm[target]))
        # moving-block bootstrap CI for every coefficient
        boots = np.empty((C.N_REPLICATES, len(feats)))
        for i, idx in enumerate(boot_idx):
            ym = targets[target][idx].mean(axis=0)
            beta, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(36), Xmat]), ym, rcond=None)
            boots[i] = beta[1:]
        blo = np.percentile(boots, 2.5, axis=0)
        bhi = np.percentile(boots, 97.5, axis=0)
        for k, f in enumerate(feats):
            rows.append({
                "relation": name, "target": target, "predictor": f,
                "n_predictors": len(feats), "window": "calibration",
                "slope": fit["coefficients"][k],
                "intercept": fit["intercept"],
                "slope_se": fit["se"][k],
                "slope_ci_low_ols": fit["ci_low"][k],
                "slope_ci_high_ols": fit["ci_high"][k],
                "slope_ci_low_boot": float(blo[k]),
                "slope_ci_high_boot": float(bhi[k]),
                "r2": fit["r2"], "adj_r2": fit["adj_r2"],
                "rmse": fit["rmse"], "correlation": pear,
                "n_cells": 36,
                "inference_note": ("within-domain descriptive summary; 36 "
                                   "contiguous parents are not independent "
                                   "spatial replicates"),
            })
        metrics_json[name] = {
            "target": target, "predictors": list(feats),
            "intercept": fit["intercept"],
            "coefficients": fit["coefficients"],
            "r2": fit["r2"], "adj_r2": fit["adj_r2"], "rmse": fit["rmse"],
            "correlation": pear,
            "bootstrap": {"seed_master": C.MASTER_SEED,
                          "spawn_key": list(C.SEED_SPAWN[f"relationship_boot_{case}"]),
                          "replicates": C.N_REPLICATES,
                          "block": C.PRIMARY_BLOCK,
                          "ci_low": [float(x) for x in blo],
                          "ci_high": [float(x) for x in bhi]},
        }
    C.write_df(cdir / "terrain_relationships.csv",
               pd.DataFrame(rows).round(9), case=case)
    C.write_json(cdir / "terrain_relationship_metrics.json",
                 {"relations": metrics_json,
                  "fit_window": "calibration [970,3650) only",
                  "descriptor_screening": "none (frozen identities only)"},
                 case=case)


def main(case: str) -> None:
    run_descriptors(case)
    run_relationships(case)
    C.run_receipt(f"phase07_terrain_case_{case}",
                  {"case": case, "data_status": C.data_status(case)})


if __name__ == "__main__":
    args = sys.argv[1:] or ["A"]
    for cs in args:
        main(cs)
