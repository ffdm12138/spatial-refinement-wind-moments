"""Phases 10–11 — wind-state characterization and state-stratified robustness
(PLAN §16–17), with the FROZEN strata definitions:

  * speed terciles of the calibration-period d05 domain-mean scalar speed
    (frozen 1/3, 2/3 quantile rule; cut VALUES are case-specific because the
    definition, not the numbers, is frozen)
  * meteorological-from direction sectors (frozen (270 - atan2) convention,
    8 x 45 deg, calm cutoff 1e-12) with the frozen >= 300-state
    well-sampled threshold
  * inner 4x4 boundary refit and leave-cell / leave-row / leave-column
    spatial cross-validation on calibration cell means (frozen protocol)

B vs C wording contract: the two December cases share the same synoptic time
window (contemporaneous episode); the local wind states are NOT identical and
are quantified here.
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
import phase07_terrain as P7  # noqa: E402

WINDOWS = {"calibration": C.CAL_SLICE, "evaluation": C.EVAL_SLICE}


def circular_mean_deg(deg: np.ndarray) -> float:
    v = np.asarray(deg, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    rad = np.deg2rad(v)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def circular_sd_deg(deg: np.ndarray) -> float:
    v = np.asarray(deg, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    rad = np.deg2rad(v)
    r = float(np.hypot(np.sin(rad).mean(), np.cos(rad).mean()))
    r = min(max(r, np.finfo(float).tiny), 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def wind_state(case: str) -> dict:
    print(f"[phase10] case {case}: wind-state characterization")
    cdir = C.case_dir(case)
    tab = P1.load_moments(case)
    rows = []
    summary = {}
    for wname, (a, b) in WINDOWS.items():
        speed = tab["d05_domain_speed"][a:b]
        du = tab["d05_domain_u"][a:b]
        dv = tab["d05_domain_v"][a:b]
        flow_to, frm, calm = M.meteorological_from(du, dv)
        sect = M.sector_index(frm)
        # d02 parent-based domain mean for reference (representative-state rule)
        d02_dm = tab["U02_mag"][a:b].mean(axis=1)
        for i in range(b - a):
            rows.append({
                "period": wname, "state_abs": a + i,
                "speed": speed[i],
                "direction_from_deg": frm[i],
                "flow_to_math_deg": flow_to[i],
                "sector": int(sect[i]),
                "sector_label": (C.SECTOR_LABELS[sect[i]] if sect[i] >= 0 else "CALM"),
                "is_calm": bool(calm[i]),
                "d02_domain_mean_parent_speed": d02_dm[i],
            })
        counts = [int((sect == k).sum()) for k in range(8)]
        total = int((~calm).sum())
        freq = [c / total if total else 0.0 for c in counts]
        dom = int(np.argmax(counts)) if total else -1
        summary[wname] = {
            "n_states": b - a,
            "n_directional_states": total,
            "n_calm_excluded": int(calm.sum()),
            "speed_mean": float(speed.mean()),
            "speed_median": float(np.median(speed)),
            "speed_q10": float(np.quantile(speed, 0.1)),
            "speed_q90": float(np.quantile(speed, 0.9)),
            "domain_mean_vector_u": float(du.mean()),
            "domain_mean_vector_v": float(dv.mean()),
            "direction_circular_mean_from_deg": circular_mean_deg(frm),
            "direction_circular_sd_deg": circular_sd_deg(frm),
            "sector_counts": counts,
            "sector_frequencies": freq,
            "dominant_sector": dom,
            "dominant_sector_label": (C.SECTOR_LABELS[dom] if dom >= 0 else "CALM"),
            "dominant_sector_frac": (freq[dom] if dom >= 0 else 0.0),
        }
    df = pd.DataFrame(rows).round(8)
    C.write_df(cdir / "wind_state_timeseries.csv", df, case=case)
    summary["calib_minus_eval"] = {
        "speed_mean_diff": summary["calibration"]["speed_mean"]
        - summary["evaluation"]["speed_mean"],
        "sector_freq_diff": [a - b for a, b in zip(
            summary["calibration"]["sector_frequencies"],
            summary["evaluation"]["sector_frequencies"])],
    }
    summary["metadata"] = {
        "direction_convention": "meteorological_from",
        "conversion": "(270 - degrees(atan2(v,u))) mod 360",
        "sector_labels": list(C.SECTOR_LABELS),
        "sector_width_deg": 45.0,
        "calm_cutoff_mps": C.CALM_CUTOFF_MPS,
        "speed_variable": "d05 domain-mean scalar speed",
        "direction_variable": "d05 domain-mean vector",
    }
    C.write_json(cdir / "wind_state_summary.json", summary, case=case)
    return summary


def wind_state_comparison(cases: list[str], summaries: dict) -> None:
    rows = []
    for case in cases:
        for wname in ("calibration", "evaluation"):
            s = summaries[case][wname]
            rows.append({
                "case": case, "label": C.CASES[case]["label"], "window": wname,
                "speed_mean": s["speed_mean"], "speed_median": s["speed_median"],
                "speed_q10": s["speed_q10"], "speed_q90": s["speed_q90"],
                "direction_circular_mean_from_deg": s["direction_circular_mean_from_deg"],
                "direction_circular_sd_deg": s["direction_circular_sd_deg"],
                "dominant_sector_label": s["dominant_sector_label"],
                "dominant_sector_frac": s["dominant_sector_frac"],
                **{f"freq_{lab}": s["sector_frequencies"][k]
                   for k, lab in enumerate(C.SECTOR_LABELS)},
            })
    df = pd.DataFrame(rows).round(6)
    df["BC_wording"] = ("B and C share the same synoptic time window "
                        "(contemporaneous episode); local wind states are "
                        "not identical")
    C.write_df(C.ext_dir("cross_case") / "wind_state_comparison.csv", df)


def stratified(case: str) -> None:
    print(f"[phase11] case {case}: state-stratified robustness")
    cdir = C.case_dir(case)
    tab = P1.load_moments(case)
    desc = P7.descriptors(case)
    rough = desc["r_sigma"].to_numpy(float)
    lam = desc["lambda_iso"].to_numpy(float)
    riqr = desc["r_iqr"].to_numpy(float)
    a, b = C.CAL_SLICE
    speed = tab["d05_domain_speed"][a:b]
    _, frm, calm = M.meteorological_from(tab["d05_domain_u"][a:b],
                                         tab["d05_domain_v"][a:b])
    sect = M.sector_index(frm)
    delta = tab["delta_scalar"][a:b]
    spread = tab["S"][a:b]

    def fit_row(stratum, level, mask, sector_label="", sector_deg=None):
        dcm = delta[mask].mean(axis=0)
        scm = spread[mask].mean(axis=0)
        fd = M.ols_fit(rough, dcm)
        fs = M.ols_fit(lam, scm)
        return {
            "stratum": stratum, "level": level, "sector_label": sector_label,
            "sector_center_deg": sector_deg, "n_states": int(mask.sum()),
            "speed_range": f"[{speed[mask].min():.6f}, {speed[mask].max():.6f}]",
            "slope_delta_vs_rough": fd["coefficients"][0],
            "ci_low_delta": fd["ci_low"][0], "ci_high_delta": fd["ci_high"][0],
            "r2_delta": fd["r2"],
            "slope_spread_vs_lambda": fs["coefficients"][0],
            "ci_low_spread": fs["ci_low"][0], "ci_high_spread": fs["ci_high"][0],
            "r2_spread": fs["r2"],
            "interval_type": "approximate_ols",
            "interval_multiplier": C.OLS_INTERVAL_MULTIPLIER,
            "interval_unit": "36 cell means",
        }

    rows = []
    cuts = np.quantile(speed, C.TERCILE_QUANTILES)
    masks = [("low", speed <= cuts[0]),
             ("medium", (speed > cuts[0]) & (speed <= cuts[1])),
             ("high", speed > cuts[1])]
    for level, m in masks:
        rows.append(fit_row("speed_tercile", level, m))
    n_sectors_used = 0
    for k in range(8):
        m = (sect == k) & (~calm)
        if int(m.sum()) < C.MIN_STATES_PER_SECTOR:
            continue
        n_sectors_used += 1
        rows.append(fit_row("direction_sector", k, m,
                            C.SECTOR_LABELS[k], C.SECTOR_CENTRES_DEG[k]))
    df = pd.DataFrame(rows)
    C.write_df(cdir / "state_stratified_terrain.csv", df.round(9), case=case)
    C.write_json(cdir / "state_stratified_terrain.json", {
        "metadata": {
            "speed_tercile_variable": "calibration-period d05 domain-mean scalar speed",
            "speed_tercile_cut_values_mps": [float(x) for x in cuts],
            "minimum_states_per_direction_sector": C.MIN_STATES_PER_SECTOR,
            "n_direction_sectors_well_sampled": n_sectors_used,
            "strata_definitions": "frozen from Case A; cut values case-specific",
        },
        "rows": df.to_dict("records")}, case=case)

    # ---- spatial CV: leave-cell / leave-row / leave-column (frozen protocol)
    dcm = delta.mean(axis=0)
    scm = spread.mean(axis=0)
    models = {
        "mean_roughness": (rough.reshape(-1, 1), dcm),
        "spread_joint_reported": (np.column_stack([riqr, lam]), scm),
        "spread_exposure_sensitivity": (lam.reshape(-1, 1), scm),
    }
    rows_cv = []
    rows_ids = np.arange(36) // 6
    cols_ids = np.arange(36) % 6
    for mname, (Xm, y) in models.items():
        for split, groups in [("cell", np.arange(36)), ("row", rows_ids),
                              ("col", cols_ids)]:
            pred = np.zeros(36)
            for gid in np.unique(groups):
                tr = groups != gid
                te = ~tr
                design = np.column_stack([np.ones(tr.sum()), Xm[tr]])
                beta, *_ = np.linalg.lstsq(design, y[tr], rcond=None)
                pred[te] = np.column_stack([np.ones(te.sum()), Xm[te]]) @ beta
            rows_cv.append({
                "model": mname, "split": split,
                "protocol": "calibration_cell_means_only", "n": 36,
                "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                "mae": float(np.mean(np.abs(pred - y))),
                "bias": float(np.mean(pred - y)),
                "correlation": M.pearson(pred, y),
            })
    C.write_df(cdir / "spatial_cv.csv", pd.DataFrame(rows_cv).round(10), case=case)

    # ---- boundary refit: all 36 vs inner 4x4 (frozen columns) ----
    ea, eb = C.EVAL_SLICE
    delta_ev = tab["delta_scalar"][ea:eb].mean(axis=0)
    spread_ev = tab["S"][ea:eb].mean(axis=0)
    dyn_cal = tab["delta_dyn"][a:b].mean(axis=0)
    j_cal = tab["J"][a:b].mean(axis=0)
    bs_rows = []
    for subset, ids in [("all36", np.arange(36)), ("inner16", np.array(C.INNER16_IDS))]:
        fd = M.ols_fit(rough[ids], dcm[ids])
        fs = M.ols_fit(lam[ids], scm[ids])
        pd_ev = fd["intercept"] + fd["coefficients"][0] * rough[ids]
        ps_ev = fs["intercept"] + fs["coefficients"][0] * lam[ids]
        bs_rows.append({
            "subset": subset, "n_cells": len(ids),
            "delta_dyn_mean": float(dyn_cal[ids].mean()),
            "jensen_mean": float(j_cal[ids].mean()),
            "roughness_slope": fd["coefficients"][0],
            "exposure_slope": fs["coefficients"][0],
            "calibration_r2_delta": fd["r2"],
            "calibration_r2_spread": fs["r2"],
            "eval_rmse_delta": float(np.sqrt(np.mean((pd_ev - delta_ev[ids]) ** 2))),
            "eval_rmse_spread": float(np.sqrt(np.mean((ps_ev - spread_ev[ids]) ** 2))),
            "obs_delta_mean": float(delta_ev[ids].mean()),
            "obs_spread_mean": float(spread_ev[ids].mean()),
        })
    C.write_df(cdir / "boundary_sensitivity.csv",
               pd.DataFrame(bs_rows).round(10), case=case)


def main(case: str) -> dict:
    s = wind_state(case)
    stratified(case)
    C.run_receipt(f"phase10_windstate_case_{case}",
                  {"case": case, "data_status": C.data_status(case)})
    return s


if __name__ == "__main__":
    summaries = {}
    cases = sys.argv[1:] or ["A"]
    for cs in cases:
        summaries[cs] = main(cs)
    if len(cases) > 1:
        wind_state_comparison(cases, summaries)
