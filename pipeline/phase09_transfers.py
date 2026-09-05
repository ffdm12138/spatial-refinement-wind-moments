"""Phase 9 — the three classes of transfer tests (PLAN §14–15).

T1  A -> B  external temporal transfer: A's FROZEN terrain equations
            (calibration-2680 coefficients, read from the frozen route lock)
            applied unchanged to B (same terrain, new event).
T2  A -> B  same-cell history external transfer: A calibration per-cell
            historical moments predict B directly (legal: same grid), compared
            against A-terrain->B and A-global-mean->B.
T3  B -> C  equation-transfer stress test: frozen descriptor FORMS fitted on
            B calibration ONLY, applied unchanged to C.  Generation 20260905
            reads this as a CONTEMPORANEOUS CROSS-SITE SPATIAL TRANSFER
            (same chronological/synoptic window, new geography and terrain
            regime, potentially different local wind realization) -- never a
            pure terrain transfer, never a causal terrain effect.  Secondary
            evidence; not a headline gate.
T4  A -> C  strictest transfer (new event + new site + new terrain regime +
            potentially different local wind realization = combined external
            stress test): A frozen equations applied to C.  Kept regardless
            of outcome.

History boundary (PLAN §15): there is NO A/B -> C same-cell history; C only
gets within-site C-calibration -> C-evaluation persistence (that baseline
lives in phase12 information sources).  Every A->C / B->C terrain application
carries per-descriptor in_range / extrapolation flags (no silent
extrapolation).
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
import phase07_terrain as P7  # noqa: E402

SEGMENTS = {"evaluation_670": C.EVAL_SLICE, "retained_3350": C.RETAINED_SLICE}


def _apply_equation(coefs: list[float], feats: list[str], desc: pd.DataFrame) -> np.ndarray:
    colmap = {"r_sigma": "r_sigma", "terrain_iqr_ratio": "r_iqr", "r_iqr": "r_iqr",
              "lambda": "lambda_iso", "lambda_iso": "lambda_iso"}
    pred = np.full(36, float(coefs[0]))
    for k, f in enumerate([f for f in feats if f != "intercept"]):
        pred += float(coefs[k + 1]) * desc[colmap[f]].to_numpy(float)
    return pred


def _range_flags(desc: pd.DataFrame, ranges: dict, feats: list[str]) -> tuple[np.ndarray, dict]:
    colmap = {"r_sigma": "r_sigma", "terrain_iqr_ratio": "r_iqr", "r_iqr": "r_iqr",
              "lambda": "lambda_iso", "lambda_iso": "lambda_iso"}
    used = [colmap[f] for f in feats if f != "intercept"]
    in_range = np.ones(36, dtype=bool)
    detail = {}
    for d in used:
        lo, hi = ranges[d]
        ok = (desc[d].to_numpy(float) >= lo) & (desc[d].to_numpy(float) <= hi)
        in_range &= ok
        detail[d] = {"range": [lo, hi], "n_out": int((~ok).sum())}
    return in_range, detail


def _score(pred: np.ndarray, target_case: str, moment: str,
           in_range: np.ndarray) -> list[dict]:
    tab = P1.load_moments(target_case)
    key = {"delta_scalar": "delta_scalar", "S": "S"}[moment]
    rows = []
    for seg, (a, b) in SEGMENTS.items():
        obs = tab[key][a:b].mean(axis=0)
        met = M.cell_metrics(pred, obs)
        rows.append({
            "segment": seg, "n_cells": 36, **met,
            "spearman": M.spearman(pred, obs),
            "pred_mean": float(pred.mean()), "obs_mean": float(obs.mean()),
            "amplitude_ratio_pred_over_obs": (float(pred.mean() / obs.mean())
                                              if abs(obs.mean()) > 1e-12 else np.nan),
            "n_cells_extrapolating": int((~in_range).sum()),
            "extrapolation_flag": bool((~in_range).any()),
        })
    return rows


def _cell_table(pred: np.ndarray, target_case: str, moment: str,
                in_range: np.ndarray) -> pd.DataFrame:
    tab = P1.load_moments(target_case)
    key = {"delta_scalar": "delta_scalar", "S": "S"}[moment]
    a, b = C.EVAL_SLICE
    obs = tab[key][a:b].mean(axis=0)
    return pd.DataFrame({
        "cell_id": np.arange(36), "prediction": pred, "observed_eval_mean": obs,
        "error": pred - obs, "in_range": in_range})


def frozen_a_models() -> dict:
    lock = C.frozen_route_lock()
    flat = lock["models"]
    out = {
        "A_terrain_delta": {"coefs": flat["mean"]["coefficients"],
                            "feats": flat["mean"]["feature_order"],
                            "moment": "delta_scalar"},
        "A_terrain_spread_joint": {"coefs": flat["spread"]["coefficients"],
                                   "feats": flat["spread"]["feature_order"],
                                   "moment": "S"},
    }
    # exposure-only / roughness-only companions from the baseline blocks
    def _walk(d):
        found = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "baseline_role" in v and "coefficients" in v:
                    found[v["baseline_role"]] = v
                else:
                    found.update(_walk(v))
        return found
    roles = _walk(lock)
    if "exposure_only" in roles:
        out["A_terrain_spread_exposure_only"] = {
            "coefs": roles["exposure_only"]["coefficients"],
            "feats": roles["exposure_only"]["feature_order"], "moment": "S"}
    if "roughness_only" in roles:
        out["A_terrain_spread_roughness_only"] = {
            "coefs": roles["roughness_only"]["coefficients"],
            "feats": roles["roughness_only"]["feature_order"], "moment": "S"}
    return out


def b_calibration_models() -> dict:
    """Frozen FORMS refit on B calibration only (phase08 output for case B)."""
    p = C.case_dir("B") / "terrain_relationship_metrics.json"
    rel = json.loads(p.read_text(encoding="utf-8"))["relations"]
    return {
        "B_cal_delta_vs_r_sigma": {
            "coefs": [rel["delta_vs_r_sigma"]["intercept"],
                      *rel["delta_vs_r_sigma"]["coefficients"]],
            "feats": ["intercept", "r_sigma"], "moment": "delta_scalar"},
        "B_cal_S_vs_lambda_iso": {
            "coefs": [rel["S_vs_lambda_iso"]["intercept"],
                      *rel["S_vs_lambda_iso"]["coefficients"]],
            "feats": ["intercept", "lambda_iso"], "moment": "S"},
        "B_cal_S_vs_r_iqr_plus_lambda_iso": {
            "coefs": [rel["S_vs_r_iqr_plus_lambda_iso"]["intercept"],
                      *rel["S_vs_r_iqr_plus_lambda_iso"]["coefficients"]],
            "feats": ["intercept", "r_iqr", "lambda_iso"], "moment": "S"},
    }


def main(cases_ok: set[str]) -> None:
    cross = C.ext_dir("cross_case")
    a_models = frozen_a_models()
    a_ranges = C.frozen_a_calibration_ranges()

    # ---------------- T1 + T2: A -> B ----------------
    if "B" in cases_ok:
        print("[phase09] T1/T2: A -> B transfers")
        desc_b = P7.descriptors("B")
        rows, cells = [], []
        # T1: frozen terrain equations
        for name, m in a_models.items():
            pred = _apply_equation(m["coefs"], m["feats"], desc_b)
            in_r, detail = _range_flags(desc_b, a_ranges, m["feats"])
            for r in _score(pred, "B", m["moment"], in_r):
                rows.append({"test": "T1", "model": name,
                             "information_class": "static terrain (A frozen)",
                             "target": m["moment"], **r,
                             "range_detail": json.dumps(detail)})
            ct = _cell_table(pred, "B", m["moment"], in_r)
            ct.insert(0, "model", name)
            ct.insert(0, "test", "T1")
            cells.append(ct)
        C.write_df(cross / "transfer_T1_A_to_B.csv",
                   pd.DataFrame([r for r in rows if r["test"] == "T1"]).round(8),
                   status_value=C.combined_status("A", "B"))

        # T2: A history / A terrain / A global -> B
        hist = C.frozen_a_history()
        t2_models = {
            "A_same_cell_historical_mean_scalar": ("delta_scalar", hist["scalar"],
                                                   "fine-grid history (A calibration)"),
            "A_same_cell_historical_mean_spread": ("S", hist["spread"],
                                                   "fine-grid history (A calibration)"),
            "A_terrain_delta": ("delta_scalar", hist["terrain_scalar"],
                                "static terrain (A frozen)"),
            "A_terrain_spread_joint": ("S", hist["terrain_spread_joint"],
                                       "static terrain (A frozen)"),
            "A_global_calibration_mean_scalar": ("delta_scalar",
                                                 np.full(36, hist["global_scalar"]),
                                                 "no fine-scale information (A)"),
            "A_global_calibration_mean_spread": ("S",
                                                 np.full(36, hist["global_spread"]),
                                                 "no fine-scale information (A)"),
        }
        t2_rows = []
        in_r_all = np.ones(36, dtype=bool)   # same grid: history is in-range
        for name, (moment, pred, iclass) in t2_models.items():
            for r in _score(np.asarray(pred, float), "B", moment, in_r_all):
                t2_rows.append({"test": "T2", "model": name,
                                "information_class": iclass,
                                "target": moment, **r,
                                "history_boundary": ("A->B same-cell history is "
                                                     "legal: identical grid")})
            ct = _cell_table(np.asarray(pred, float), "B", moment, in_r_all)
            ct.insert(0, "model", name)
            ct.insert(0, "test", "T2")
            cells.append(ct)
        C.write_df(cross / "transfer_T2_history_A_to_B.csv",
                   pd.DataFrame(t2_rows).round(8),
                   status_value=C.combined_status("A", "B"))
        C.write_df(cross / "transfer_A_to_B_cell_means.csv",
                   pd.concat(cells, ignore_index=True).round(8),
                   status_value=C.combined_status("A", "B"))

    # ---------------- T3: B -> C ----------------
    if {"B", "C"}.issubset(cases_ok):
        print("[phase09] T3: B -> C contemporaneous cross-site transfer"
              if C.GEN.get("bc_synchronized_bootstrap") else
              "[phase09] T3: B -> C external transfer under a joint event+site shift")
        desc_b = P7.descriptors("B")
        desc_c = P7.descriptors("C")
        b_ranges = {d: [float(desc_b[d].min()), float(desc_b[d].max())]
                    for d in ("r_sigma", "r_iqr", "lambda_iso")}
        rows, cells = [], []
        for name, m in b_calibration_models().items():
            pred = _apply_equation(m["coefs"], m["feats"], desc_c)
            in_r, detail = _range_flags(desc_c, b_ranges, m["feats"])
            for r in _score(pred, "C", m["moment"], in_r):
                rows.append({"test": "T3", "model": name,
                             "label": "new multicase secondary experiment",
                             "information_class": "static terrain (B-calibration fit)",
                             "target": m["moment"], **r,
                             "range_detail": json.dumps(detail)})
            ct = _cell_table(pred, "C", m["moment"], in_r)
            ct.insert(0, "model", name)
            ct.insert(0, "test", "T3")
            cells.append(ct)
        C.write_df(cross / "transfer_T3_B_to_C.csv",
                   pd.DataFrame(rows).round(8),
                   status_value=C.combined_status("B", "C"))
        C.write_df(cross / "transfer_B_to_C_cell_means.csv",
                   pd.concat(cells, ignore_index=True).round(8),
                   status_value=C.combined_status("B", "C"))

    # ---------------- T4: A -> C ----------------
    if "C" in cases_ok:
        print("[phase09] T4: A -> C full transfer")
        desc_c = P7.descriptors("C")
        rows, cells = [], []
        for name, m in a_models.items():
            pred = _apply_equation(m["coefs"], m["feats"], desc_c)
            in_r, detail = _range_flags(desc_c, a_ranges, m["feats"])
            for r in _score(pred, "C", m["moment"], in_r):
                rows.append({"test": "T4", "model": name,
                             "information_class": "static terrain (A frozen)",
                             "target": m["moment"], **r,
                             "range_detail": json.dumps(detail),
                             "history_boundary": ("NO A->C same-cell history: "
                                                  "different geography, cell "
                                                  "identities do not map")})
            ct = _cell_table(pred, "C", m["moment"], in_r)
            ct.insert(0, "model", name)
            ct.insert(0, "test", "T4")
            cells.append(ct)
        C.write_df(cross / "transfer_T4_A_to_C.csv",
                   pd.DataFrame(rows).round(8),
                   status_value=C.combined_status("A", "C"))
        C.write_df(cross / "transfer_A_to_C_cell_means.csv",
                   pd.concat(cells, ignore_index=True).round(8),
                   status_value=C.combined_status("A", "C"))

    C.run_receipt("phase09_transfers", {"cases_ok": sorted(cases_ok)})


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"A", "B", "C"})
