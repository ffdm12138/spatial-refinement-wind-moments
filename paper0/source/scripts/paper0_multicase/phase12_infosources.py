"""Phase 12 — within-case information-source comparison (PLAN §18).

Scalar-increment sources: zero correction, global calibration mean, bilinear,
bicubic, terrain roughness (own-calibration fit of the frozen form),
same-cell historical mean (own calibration).
Spread sources: global calibration mean, bilinear, bicubic, roughness-only,
exposure-only, joint terrain, same-cell historical mean.

Metrics (frozen conventions): RMSE, centred RMSE, bias, correlation, slope on
the 36 evaluation cell means; all-record RMSE (24,120 cell-time records of
parent-cell mean scalar speed) for the scalar group.

Public naming lock: the method key is "same_cell_historical_mean"; the frozen
internal key cell_mean_persistence is carried only in the internal_key column.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import mc_moments as M  # noqa: E402
import phase01_moments as P1  # noqa: E402
import phase05_interpolation as P5  # noqa: E402

INFO_CLASS = {
    "zero_correction": "no fine-scale information",
    "global_calibration_mean": "no fine-scale information",
    "bilinear": "geometric interpolation",
    "bicubic": "geometric interpolation",
    "terrain_roughness_equation": "static terrain",
    "roughness_only": "static terrain",
    "exposure_only": "static terrain",
    "joint_terrain_equation": "static terrain",
    C.PUBLIC_HISTORY_NAME: "fine-grid history",
}
INFO_USED = {
    "zero_correction": "none",
    "global_calibration_mean": "one calibration-period domain mean",
    "bilinear": "instantaneous d02 U and V, no fitted parameters",
    "bicubic": "instantaneous d02 U and V, no fitted parameters",
    "terrain_roughness_equation": "static subgrid terrain and two fitted coefficients",
    "roughness_only": "static subgrid terrain and two fitted coefficients",
    "exposure_only": "static subgrid terrain and two fitted coefficients",
    "joint_terrain_equation": "static subgrid terrain and fitted coefficients",
    C.PUBLIC_HISTORY_NAME: "36 calibration-period d05 cell moments",
}
INTERNAL_KEY = {C.PUBLIC_HISTORY_NAME: C.INTERNAL_HISTORY_KEY}
SCALAR_ORDER = ["zero_correction", "global_calibration_mean", "bilinear",
                "bicubic", "terrain_roughness_equation", C.PUBLIC_HISTORY_NAME]
SPREAD_ORDER = ["global_calibration_mean", "bilinear", "bicubic",
                "roughness_only", "exposure_only", "joint_terrain_equation",
                C.PUBLIC_HISTORY_NAME]


def own_terrain_predictions(case: str) -> dict[str, np.ndarray]:
    """Per-cell predictions from the case's own calibration-only fits."""
    rel = json.loads((C.case_dir(case) / "terrain_relationship_metrics.json")
                     .read_text(encoding="utf-8"))["relations"]
    desc = pd.read_csv(C.case_dir(case) / "terrain_descriptors.csv")
    r_sigma = desc["r_sigma"].to_numpy(float)
    r_iqr = desc["r_iqr"].to_numpy(float)
    lam = desc["lambda_iso"].to_numpy(float)

    def lin(rname, *feats):
        m = rel[rname]
        return m["intercept"] + sum(c * f for c, f in zip(m["coefficients"], feats))
    return {
        "terrain_roughness_equation": lin("delta_vs_r_sigma", r_sigma),
        "roughness_only": lin("S_vs_r_iqr", r_iqr),
        "exposure_only": lin("S_vs_lambda_iso", lam),
        "joint_terrain_equation": lin("S_vs_r_iqr_plus_lambda_iso", r_iqr, lam),
    }


def build_predictions(case: str) -> dict:
    """Static per-cell (36,) or time-varying (670,36) evaluation predictions."""
    tab = P1.load_moments(case)
    a, b = C.CAL_SLICE
    ea, eb = C.EVAL_SLICE
    terr = own_terrain_predictions(case)
    with np.load(P5.interp_npz_path(case)) as z:
        interp = {(g, m): z[f"{m}_{col}"][ea:eb]
                  for m in ("bilinear", "bicubic")
                  for g, col in [("scalar", "delta_scalar"), ("spread", "spread")]}
        interp_speed = {m: z[f"{m}_mean_scalar_speed"][ea:eb]
                        for m in ("bilinear", "bicubic")}
    pred = {
        ("scalar", "zero_correction"): np.zeros(36),
        ("scalar", "global_calibration_mean"):
            np.full(36, tab["delta_scalar"][a:b].mean()),
        ("scalar", "bilinear"): interp[("scalar", "bilinear")],
        ("scalar", "bicubic"): interp[("scalar", "bicubic")],
        ("scalar", "terrain_roughness_equation"): terr["terrain_roughness_equation"],
        ("scalar", C.PUBLIC_HISTORY_NAME): tab["delta_scalar"][a:b].mean(axis=0),
        ("spread", "global_calibration_mean"): np.full(36, tab["S"][a:b].mean()),
        ("spread", "bilinear"): interp[("spread", "bilinear")],
        ("spread", "bicubic"): interp[("spread", "bicubic")],
        ("spread", "roughness_only"): terr["roughness_only"],
        ("spread", "exposure_only"): terr["exposure_only"],
        ("spread", "joint_terrain_equation"): terr["joint_terrain_equation"],
        ("spread", C.PUBLIC_HISTORY_NAME): tab["S"][a:b].mean(axis=0),
    }
    obs = {"scalar": tab["delta_scalar"][ea:eb], "spread": tab["S"][ea:eb],
           "speed_ref": tab["s_bar"][ea:eb], "U02_mag": tab["U02_mag"][ea:eb]}
    return {"pred": pred, "obs": obs, "interp_speed": interp_speed}


def main(case: str) -> None:
    print(f"[phase12] case {case}: information-source comparison")
    P = build_predictions(case)
    pred, obs = P["pred"], P["obs"]
    rows = []
    for group, order in [("scalar_increment", SCALAR_ORDER), ("spread", SPREAD_ORDER)]:
        gkey = "scalar" if group == "scalar_increment" else "spread"
        obs_cm = obs[gkey].mean(axis=0)
        for m in order:
            p = pred[(gkey, m)]
            pred_cm = p.mean(axis=0) if p.ndim == 2 else p
            met = M.cell_metrics(pred_cm, obs_cm)
            if group == "scalar_increment":
                if p.ndim == 2:
                    s_hat = P["interp_speed"][m]
                else:
                    s_hat = obs["U02_mag"] + p[None, :]
                ar = float(np.sqrt(np.mean((s_hat - obs["speed_ref"]) ** 2)))
            else:
                ar = np.nan
            rows.append({
                "group": group, "method": m,
                "internal_key": INTERNAL_KEY.get(m, ""),
                "info_class": INFO_CLASS[m],
                "information_used": INFO_USED[m],
                **met, "all_record_rmse": ar,
                "pred_mean": float(np.asarray(pred_cm).mean()),
                "obs_mean": float(obs_cm.mean()),
                "n_cells": 36,
                "evaluation_segment": "final 670 held-out states",
            })
    df = pd.DataFrame(rows)
    C.write_df(C.case_dir(case) / "information_sources.csv", df.round(9), case=case)
    C.run_receipt(f"phase12_infosources_case_{case}",
                  {"case": case, "data_status": C.data_status(case)})


def ranking(cases: list[str]) -> None:
    """cross_case/information_source_ranking.csv (rank by RMSE per group)."""
    frames = []
    for case in cases:
        df = pd.read_csv(C.case_dir(case) / "information_sources.csv")
        df.insert(0, "case", case)
        frames.append(df)
    alldf = pd.concat(frames, ignore_index=True)
    alldf["rank_rmse"] = alldf.groupby(["case", "group"])["rmse"].rank(method="min")
    piv = alldf.pivot_table(index=["group", "method", "info_class"],
                            columns="case", values="rank_rmse").reset_index()
    piv.columns = [f"rank_case_{c}" if c in cases else c for c in piv.columns]
    rank_cols = [f"rank_case_{c}" for c in cases]
    piv["rank_stable_across_cases"] = piv[rank_cols].nunique(axis=1) == 1
    out = alldf.merge(piv, on=["group", "method", "info_class"], how="left")
    C.write_df(C.ext_dir("cross_case") / "information_source_ranking.csv",
               out.round(9))
    C.run_receipt("phase12_ranking", {"cases": cases})


if __name__ == "__main__":
    cases = sys.argv[1:] or ["A"]
    for cs in cases:
        main(cs)
    if len(cases) > 1:
        ranking(cases)
