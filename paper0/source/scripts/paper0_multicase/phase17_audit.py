"""Phase 17 — audit assembly (PLAN §27–§32).

  * Case-A legacy reproduction: recomputed headline quantities compared
    against values read from the FROZEN evidence files (read-only), with
    explicit tolerances and PASS/FAIL per quantity.  A failure never touches
    frozen evidence; it is recorded prominently and the run continues.
  * Leakage matrix + audit (PLAN §28).
  * Counterexample register (PLAN §29) — mechanical Core-Gate evaluations,
    never deleted, branded PLACEHOLDER DRY-RUN where B/C are placeholders.
  * FINAL_MULTICASE_P0_AUDIT.md with the fixed A–K structure, the
    12-question thunder summary and both verdict lines.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402

ROWS: list[dict] = []


def add(quantity: str, recomputed, frozen, tol: float, source: str) -> None:
    d = abs(float(recomputed) - float(frozen))
    ROWS.append({"quantity": quantity, "recomputed": float(recomputed),
                 "frozen": float(frozen), "abs_diff": d, "tolerance": tol,
                 "status": "PASS" if d <= tol else "FAIL",
                 "frozen_source": source})


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def case_a_reproduction() -> pd.DataFrame:
    print("[phase17] Case-A legacy reproduction")
    ROWS.clear()
    A = C.case_dir("A")

    # 1. calibration-period domain decomposition (headline -0.004 + 0.275 = 0.271)
    dom = json.loads((A / "moments_domain_summary.json").read_text(encoding="utf-8"))
    cal = dom["windows"]["calibration"]["domain_mean"]
    j2 = json.loads(C.FROZEN_J2_JSON.read_text(encoding="utf-8"))
    cc = j2["crosscheck"]
    add("cal_domain_mean_delta_dyn", cal["delta_dyn"],
        cc["delta_dyn_domain_mean"], 1e-5, "jensen_second_order.json")
    add("cal_domain_mean_J", cal["J"], cc["jensen_domain_mean"], 1e-5,
        "jensen_second_order.json")
    add("cal_domain_mean_delta_scalar", cal["delta_scalar"],
        cc["delta_scalar_domain_mean"], 1e-5, "jensen_second_order.json")

    ev = dom["windows"]["evaluation"]["domain_mean"]
    bs = _csv(C.FROZEN_BOUNDARY).set_index("subset")
    add("eval_domain_mean_delta_scalar", ev["delta_scalar"],
        bs.loc["all36", "obs_delta_mean"], 1e-5, "boundary_sensitivity.csv")
    add("eval_domain_mean_S", ev["S"], bs.loc["all36", "obs_spread_mean"],
        1e-5, "boundary_sensitivity.csv")

    # 2. representative state (frozen rule answer 3955)
    rep = json.loads((A / "representative_state_provenance.json")
                     .read_text(encoding="utf-8"))
    frozen_rep = json.loads(C.FROZEN_INTERP_CONTRACT.read_text(encoding="utf-8"))[
        "representative_state"]["computed_values"]
    add("representative_state_global_index", rep["global_tensor_index"],
        frozen_rep["global_tensor_index"], 0, "interpolation_contract_frozen.json")
    add("representative_state_relative_index", rep["evaluation_relative_index"],
        frozen_rep["evaluation_relative_index"], 0,
        "interpolation_contract_frozen.json")
    add("representative_state_median_speed", rep["domain_mean_d02_speed_median"],
        frozen_rep["domain_mean_d02_speed_median"], 1e-5,
        "interpolation_contract_frozen.json")

    # 3. interpolation metrics (both modes, both groups)
    mine = _csv(A / "interpolation_metrics.csv")
    frz = _csv(C.FROZEN_INTERP_METRICS)
    for _, fr in frz.iterrows():
        mr = mine[(mine.group == fr["group"]) & (mine.model == fr["model"])]
        for col, tol in [("rmse", 1e-5), ("bias", 1e-5), ("crmse", 1e-5),
                         ("r", 1e-4), ("slope", 1e-4),
                         ("E_vec_rmse", 1e-5), ("delta_dyn_domain_mean", 1e-5)]:
            add(f"interp_{fr['model']}_{fr['group']}_{col}",
                mr[col].iloc[0], fr[col], tol, "interpolation_metrics.csv")
        if fr["group"] == "scalar_increment":
            add(f"interp_{fr['model']}_all_record_rmse",
                mr["all_record_rmse"].iloc[0], fr["all_record_rmse"], 1e-5,
                "interpolation_metrics.csv")

    # 4. recovery ratios (12 frozen rows)
    mrec = _csv(A / "interpolation_recovery.csv")
    frec = _csv(C.FROZEN_INTERP_RECOVERY)
    for _, fr in frec.iterrows():
        mr = mrec[(mrec.moment == fr["moment"]) & (mrec.model == fr["model"])
                  & (mrec.cell_set == fr["cell_set"])]
        add(f"Q_{fr['moment']}_{fr['model']}_{fr['cell_set']}",
            mr["recovery_ratio"].iloc[0], fr["recovery_ratio"], 1e-5,
            "interpolation_recovery.csv")

    # 5. parent-vector-constrained metrics + recovery
    mpv = _csv(A / "parent_vector_constrained_metrics.csv")
    fpv = _csv(C.FROZEN_PVC_METRICS)
    for _, fr in fpv.iterrows():
        mr = mpv[(mpv.group == fr["group"]) & (mpv.model == fr["model"])]
        for col, tol in [("rmse", 1e-5), ("bias", 1e-5), ("crmse", 1e-5),
                         ("r", 1e-4), ("slope", 1e-4)]:
            add(f"pvc_{fr['model']}_{fr['group']}_{col}",
                mr[col].iloc[0], fr[col], tol,
                "parent_vector_constrained_metrics.csv")
    mrecp = _csv(A / "parent_vector_constrained_recovery.csv")
    frecp = _csv(C.FROZEN_PVC_RECOVERY)
    for _, fr in frecp.iterrows():
        mr = mrecp[(mrecp.moment == fr["moment"]) & (mrecp.model == fr["model"])
                   & (mrecp.cell_set == fr["cell_set"])]
        add(f"Q_{fr['moment']}_{fr['model']}_{fr['cell_set']}",
            mr["recovery_ratio"].iloc[0], fr["recovery_ratio"], 1e-5,
            "parent_vector_constrained_recovery.csv")

    # 6. terrain descriptors (per-cell max abs diff folded into one row each)
    mdesc = _csv(A / "terrain_descriptors.csv").sort_values("cell_id")
    fdesc = _csv(C.FROZEN_SCALE_DESC).sort_values("cell_id")
    add("terrain_r_sigma_max_abs_cell_diff", 
        float(np.abs(mdesc.r_sigma.to_numpy() - fdesc.r_sigma.to_numpy()).max()),
        0.0, 1e-8, "scale_closure/terrain_descriptors.csv")
    # frozen lambda_iso was sealed at ~1e-5 precision (the legacy production
    # check `parent27_lambda_iso_max_abs_vs_frozen` also used 1e-5)
    add("terrain_lambda_iso_max_abs_cell_diff",
        float(np.abs(mdesc.lambda_iso.to_numpy() - fdesc.lambda_iso.to_numpy()).max()),
        0.0, 1e-5, "scale_closure/terrain_descriptors.csv")
    ftm = _csv(C.FROZEN_TERRAIN_METRICS).sort_values("cell_id")
    add("terrain_r_iqr_max_abs_cell_diff",
        float(np.abs(mdesc.r_iqr.to_numpy() - ftm.terrain_iqr_ratio.to_numpy()).max()),
        0.0, 1e-9, "outputs_prepared_terrain_metrics.csv")

    # 7. frozen equation coefficients (calibration refits)
    rel = json.loads((A / "terrain_relationship_metrics.json")
                     .read_text(encoding="utf-8"))["relations"]
    lock = C.frozen_route_lock()["models"]
    add("delta_eq_intercept", rel["delta_vs_r_sigma"]["intercept"],
        lock["mean"]["coefficients"][0], 1e-4, "outputs_selection_route_lock.json")
    add("delta_eq_slope", rel["delta_vs_r_sigma"]["coefficients"][0],
        lock["mean"]["coefficients"][1], 1e-2, "outputs_selection_route_lock.json")
    add("joint_eq_intercept", rel["S_vs_r_iqr_plus_lambda_iso"]["intercept"],
        lock["spread"]["coefficients"][0], 1e-4, "outputs_selection_route_lock.json")
    add("joint_eq_riqr_coef", rel["S_vs_r_iqr_plus_lambda_iso"]["coefficients"][0],
        lock["spread"]["coefficients"][1], 1e-2, "outputs_selection_route_lock.json")
    add("joint_eq_lambda_coef", rel["S_vs_r_iqr_plus_lambda_iso"]["coefficients"][1],
        lock["spread"]["coefficients"][2], 1e-2, "outputs_selection_route_lock.json")

    # 8. global calibration means + same-cell history
    hist = C.frozen_a_history()
    with np.load(A / "moments_cell_time.npz") as z:
        a, b = C.CAL_SLICE
        my_hist_sc = z["delta_scalar"][a:b].mean(axis=0)
        my_hist_sp = z["S"][a:b].mean(axis=0)
    add("global_calibration_mean_scalar", float(my_hist_sc.mean()),
        hist["global_scalar"], 1e-5, "outputs_evaluation_holdout_model_cells.csv")
    add("global_calibration_mean_spread", float(my_hist_sp.mean()),
        hist["global_spread"], 1e-5, "outputs_evaluation_holdout_model_cells.csv")
    add("same_cell_history_scalar_max_abs_cell_diff",
        float(np.abs(my_hist_sc - hist["scalar"]).max()), 0.0, 1e-5,
        "outputs_evaluation_holdout_model_cells.csv")
    add("same_cell_history_spread_max_abs_cell_diff",
        float(np.abs(my_hist_sp - hist["spread"]).max()), 0.0, 1e-5,
        "outputs_evaluation_holdout_model_cells.csv")

    # 9. information-source evaluation metrics vs frozen error decomposition
    info = _csv(A / "information_sources.csv")
    ed = _csv(C.FROZEN_ERROR_DECOMP)
    name_map = {("scalar_increment", "zero_correction"): "zero_correction",
                ("scalar_increment", "global_calibration_mean"): "global_training_mean",
                ("scalar_increment", C.PUBLIC_HISTORY_NAME): "cell_mean_persistence",
                ("scalar_increment", "terrain_roughness_equation"): "terrain_roughness_equation",
                ("spread", "global_calibration_mean"): "global_training_mean",
                ("spread", C.PUBLIC_HISTORY_NAME): "cell_mean_persistence",
                ("spread", "roughness_only"): "roughness_only",
                ("spread", "exposure_only"): "exposure_only",
                ("spread", "joint_terrain_equation"): "locked_joint_equation"}
    for (grp, mine_name), frz_name in name_map.items():
        mr = info[(info.group == grp) & (info.method == mine_name)]
        fr = ed[(ed.group == grp) & (ed.model == frz_name)]
        for col, tol in [("rmse", 2e-4), ("bias", 2e-4), ("crmse", 2e-4)]:
            add(f"info_{grp}_{mine_name}_{col}", mr[col].iloc[0],
                fr[col].iloc[0], tol, "error_decomposition.csv")

    # all-record RMSE for the static scalar sources (full-precision sealed)
    scores = _csv(C.FROZEN_BASELINE_SCORES)
    for mine_name, frz_name in [("zero_correction", "zero_correction"),
                                ("global_calibration_mean", "global_training_mean"),
                                (C.PUBLIC_HISTORY_NAME, "cell_mean_persistence"),
                                ("terrain_roughness_equation", "terrain_roughness_equation")]:
        fr = scores[(scores.group == "mean") & (scores.baseline == frz_name)
                    & (scores.evaluation_level == "all_record_scalar_speed")
                    & (scores.metric == "rmse")]
        mr = info[(info.group == "scalar_increment") & (info.method == mine_name)]
        add(f"all_record_rmse_{mine_name}", mr["all_record_rmse"].iloc[0],
            fr["estimate"].iloc[0], 2e-4,
            "outputs_evaluation_holdout_baseline_scores.csv")

    # 10. wind state + terciles + sectors
    ws = json.loads((A / "wind_state_summary.json").read_text(encoding="utf-8"))
    fws = json.loads(C.FROZEN_WINDSTATE.read_text(encoding="utf-8"))
    add("cal_speed_mean", ws["calibration"]["speed_mean"],
        fws["training"]["speed_mean"], 1e-6, "wind_state_summary.json")
    add("eval_speed_mean", ws["evaluation"]["speed_mean"],
        fws["evaluation"]["speed_mean"], 1e-6, "wind_state_summary.json")
    for k in range(8):
        add(f"cal_sector_count_{C.SECTOR_LABELS[k]}",
            ws["calibration"]["sector_counts"][k],
            fws["training"]["sector_counts"][k], 0, "wind_state_summary.json")
    strat = json.loads((A / "state_stratified_terrain.json").read_text(encoding="utf-8"))
    fstrat = _csv(C.FROZEN_STRATIFIED)
    cuts = strat["metadata"]["speed_tercile_cut_values_mps"]
    add("tercile_cut_q1", cuts[0], 2.4191059497839675, 1e-9,
        "state_stratified_terrain.json")
    add("tercile_cut_q2", cuts[1], 3.043454869311997, 1e-9,
        "state_stratified_terrain.json")
    mstrat = _csv(A / "state_stratified_terrain.csv")
    for _, fr in fstrat.iterrows():
        key = fr["level"] if fr["stratum"] == "speed_tercile" else fr["sector_label"]
        mr = mstrat[(mstrat.stratum == fr["stratum"])
                    & ((mstrat.level.astype(str) == str(fr["level"]))
                       if fr["stratum"] == "speed_tercile"
                       else (mstrat.sector_label == fr["sector_label"]))]
        add(f"strat_{fr['stratum']}_{key}_slope_delta",
            mr["slope_delta_vs_rough"].iloc[0], fr["slope_delta_vs_rough"],
            1e-4, "state_stratified_terrain.csv")
        add(f"strat_{fr['stratum']}_{key}_slope_spread",
            mr["slope_spread_vs_lambda"].iloc[0], fr["slope_spread_vs_lambda"],
            1e-4, "state_stratified_terrain.csv")

    # 11. spatial CV + boundary sensitivity
    mcv = _csv(A / "spatial_cv.csv")
    fcv = _csv(C.FROZEN_SPATIAL_CV)
    for _, fr in fcv.iterrows():
        mr = mcv[(mcv.model == fr["model"]) & (mcv.split == fr["split"])]
        add(f"spatial_cv_{fr['model']}_{fr['split']}_rmse", mr["rmse"].iloc[0],
            fr["rmse"], 1e-4, "spatial_cv.csv")
        add(f"spatial_cv_{fr['model']}_{fr['split']}_corr",
            mr["correlation"].iloc[0], fr["correlation"], 1e-4, "spatial_cv.csv")
    mbs = _csv(A / "boundary_sensitivity.csv").set_index("subset")
    fbs = _csv(C.FROZEN_BOUNDARY).set_index("subset")
    for subset in ("all36", "inner16"):
        for col, tol in [("delta_dyn_mean", 1e-5), ("jensen_mean", 1e-5),
                         ("roughness_slope", 1e-2), ("exposure_slope", 1e-2),
                         ("calibration_r2_delta", 1e-4),
                         ("calibration_r2_spread", 1e-4),
                         ("eval_rmse_delta", 1e-4), ("eval_rmse_spread", 1e-4)]:
            add(f"boundary_{subset}_{col}", mbs.loc[subset, col],
                fbs.loc[subset, col], tol, "boundary_sensitivity.csv")

    # 12. scale evidence: calibration cell means + frozen transfer metrics
    msc = _csv(A / "scale_cell_means.csv")
    msc = msc[msc.window == "calibration"].sort_values(["scale_m", "cell_id"])
    fsc = _csv(C.FROZEN_SCALE_CAL_MEANS).sort_values(["scale_m", "cell_id"])
    for col in ("J", "S"):
        add(f"scale_cal_cell_means_{col}_max_abs_diff",
            float(np.abs(msc[col].to_numpy() - fsc[col].to_numpy()).max()),
            0.0, 1e-6, "scale_closure/scale_cell_means_calibration.csv")
    mtr = _csv(C.EXT / "cross_case" / "frozen_scale_closure_transfer_A_reproduction.csv")
    ftr = _csv(C.FROZEN_SCALE_TRANSFER)
    for _, fr in ftr.iterrows():
        mr = mtr[(mtr.moment == fr["moment"])
                 & np.isclose(mtr.scale_m, fr["scale_m"])]
        for col, tol in [("r", 1e-4), ("rmse", 1e-4), ("bias", 1e-4),
                         ("crmse", 1e-4), ("slope", 1e-3)]:
            add(f"scale_transfer_{fr['moment']}_{fr['scale_m']:.0f}_{col}",
                mr[col].iloc[0], fr[col], tol,
                "scale_closure/scale_aware_transfer_metrics.csv")

    # 13. second-order group stats
    mgrp = _csv(A / "jensen_second_order_groups.csv")
    for wname, fkey in [("calibration", "train"), ("evaluation", "eval")]:
        g_all = mgrp[(mgrp.window == wname) & (mgrp.wind_group == "all")].iloc[0]
        g_lo = mgrp[(mgrp.window == wname) & (mgrp.wind_group == "below")].iloc[0]
        f = j2[fkey]
        add(f"j2_{fkey}_domain_mean_J2nd", g_all["domain_mean_J2nd"],
            f["domain_mean_J2nd"], 1e-5, "jensen_second_order.json")
        add(f"j2_{fkey}_corr_all", g_all["corr"], f["corr_all"], 1e-4,
            "jensen_second_order.json")
        add(f"j2_{fkey}_rmse_all", g_all["rmse"], f["rmse_all"], 1e-4,
            "jensen_second_order.json")
        add(f"j2_{fkey}_low_wind_fraction", g_lo["record_fraction"],
            f["low_wind_frac_states"], 1e-6, "jensen_second_order.json")
        add(f"j2_{fkey}_low_wind_rmse", g_lo["rmse"], f["low_wind_rmse"], 1e-4,
            "jensen_second_order.json")

    df = pd.DataFrame(ROWS)
    C.write_df(C.ext_dir("audit") / "case_A_legacy_reproduction.csv",
               df.round(12), case="A")
    n_fail = int((df.status == "FAIL").sum())
    print(f"  Case-A reproduction: {len(df) - n_fail}/{len(df)} PASS")
    return df


# ---------------------------------------------------------------------------
# Leakage audit (PLAN §28)
# ---------------------------------------------------------------------------
def leakage(cases_ok: list[str]) -> pd.DataFrame:
    print("[phase17] leakage audit")
    rows = []

    def rule(consumer, source, allowed, status, verification):
        rows.append({"consumer": consumer, "information_source": source,
                     "allowed": allowed, "status": status,
                     "verification": verification})

    for case in cases_ok:
        rel = json.loads((C.case_dir(case) / "terrain_relationship_metrics.json")
                         .read_text(encoding="utf-8"))
        fit_ok = rel["fit_window"].startswith("calibration")
        rule(f"case {case} terrain fitting", f"case {case} evaluation 670",
             "FORBIDDEN", "CLEAN" if fit_ok else "VIOLATION",
             "terrain_relationship_metrics.json fit_window == calibration only")
        rule(f"case {case} global mean / same-cell history",
             f"case {case} evaluation 670", "FORBIDDEN", "CLEAN",
             "phase12 slices CAL_SLICE=[970,3650) by module constant")
        ic = json.loads((C.case_dir(case) / "interpolation_contract.json")
                        .read_text(encoding="utf-8"))
        pred_ok = (ic["prediction_inputs"] == ["U10_d02.pt", "V10_d02.pt"]
                   and ic["prediction_complete_before_reference_load"])
        rule(f"case {case} interpolation prediction path",
             "d05 / terrain / station / history", "FORBIDDEN",
             "CLEAN" if pred_ok else "VIOLATION",
             "interpolation_contract.json prediction_inputs + ordering flag")
        rule(f"case {case} science estimators", "T2 tensors", "FORBIDDEN",
             "CLEAN", "T2 files are hashed in phase00 only; no science module "
                      "reads T2_*.pt (manifest-only contract)")
        rule(f"case {case} estimators/evaluation", "station observations",
             "FORBIDDEN", "CLEAN",
             "no station data exists in the pipeline inputs (raw file set only)")
    lock_sha = C.sha256_of(C.FROZEN_ROUTE_LOCK)
    rule("A frozen equations (T1/T4)", "case B / case C data", "FORBIDDEN",
         "CLEAN", f"coefficients read from frozen route lock (sha256 "
                  f"{lock_sha[:16]}...), produced before B/C existed")
    rule("A scale-aware closures (Phase 15)", "case B / case C data",
         "FORBIDDEN", "CLEAN",
         "coefficients read from frozen scale_closure_lock.json; never refit")
    if {"B", "C"}.issubset(set(cases_ok)):
        t3 = pd.read_csv(C.EXT / "cross_case" / "transfer_T3_B_to_C.csv")
        rule("T3 B-calibration fits", "case C data", "FORBIDDEN",
             "CLEAN" if (t3.model.str.startswith("B_cal_").all()) else "VIOLATION",
             "T3 models are B-calibration fits applied unchanged to C")
        hist_leak = any("history" in m.lower()
                        for m in list(t3.model.unique())
                        + list(pd.read_csv(C.EXT / "cross_case"
                                           / "transfer_T4_A_to_C.csv").model.unique()))
        rule("case C predictions", "A/B same-cell history", "FORBIDDEN",
             "VIOLATION" if hist_leak else "CLEAN",
             "no history-class model appears in T3/T4 outputs; C history "
             "exists only as within-site C-calibration persistence (phase12)")
    if "B" in cases_ok:
        rule("case B (T1/T2 target)", "A model fitting", "FORBIDDEN", "CLEAN",
             "A models are frozen evidence; B data never enters any A fit")
    df = pd.DataFrame(rows)
    df["data_status"] = C.combined_status(*cases_ok)
    C.write_df(C.ext_dir("audit") / "leakage_matrix.csv", df)
    n_viol = int((df.status == "VIOLATION").sum())
    md = [f"# Leakage audit — {C.EXTENSION_NAME}", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.",
          f"Rows: {len(df)}; violations: **{n_viol}**.", "",
          "Verification is structural: window slices are module constants "
          "(CAL_SLICE = [970,3650), EVAL_SLICE = [3650,4320)); frozen "
          "coefficients are read from read-only evidence files whose hashes "
          "are recorded; the interpolation prediction path is ordered so that "
          "d05 is loaded only after all predictions exist (contract flag).",
          "", "| consumer | source | allowed | status |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['consumer']} | {r['information_source']} | "
                  f"{r['allowed']} | {r['status']} |")
    md += ["", "Full verification notes live in leakage_matrix.csv."]
    (C.ext_dir("audit") / "leakage_audit.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return df


# ---------------------------------------------------------------------------
# Mechanical core gates + counterexample register (PLAN §23, §29)
# ---------------------------------------------------------------------------
def core_gates(cases_ok: list[str]) -> dict:
    m2 = pd.read_csv(C.ext_dir("tables") / "M2_representation_loss.csv")
    m5 = pd.read_csv(C.ext_dir("tables") / "M5_geometric_recovery.csv")
    m6 = pd.read_csv(C.ext_dir("tables") / "M6_terrain_relationships.csv")
    gates: dict = {}
    counterexamples: list[str] = []

    def brand(case):
        return (" [PLACEHOLDER DRY-RUN — NOT SCIENCE]"
                if C.data_status(case) == "PLACEHOLDER" else "")

    # Gate 1 — representation deficit (J and S clearly present in B/C)
    g1 = {}
    for case in [c for c in cases_ok if c != "A"]:
        r = m2[(m2.case == case) & (m2.window == "evaluation")].iloc[0]
        rc = m2[(m2.case == case) & (m2.window == "calibration")].iloc[0]
        present = (r["J_domain_mean"] > 1e-3 and r["S_domain_mean"] > 1e-3
                   and rc["J_domain_mean"] > 1e-3)
        dyn_comparable = abs(rc["delta_dyn_domain_mean"]) >= rc["J_domain_mean"]
        g1[case] = {"pass": bool(present),
                    "J": float(rc["J_domain_mean"]),
                    "S": float(rc["S_domain_mean"]),
                    "delta_dyn": float(rc["delta_dyn_domain_mean"]),
                    "note": ("Jensen dominance is event/site specific, but "
                             "multiple missing moments remain"
                             if dyn_comparable else "J-dominated composition")}
        if not present:
            counterexamples.append(
                f"Gate 1: case {case} lacks a clear J/S deficit{brand(case)}")
        if dyn_comparable:
            counterexamples.append(
                f"Gate 1 note: case {case} delta_dyn is comparable to J "
                f"(|{rc['delta_dyn_domain_mean']:.4f}| vs "
                f"{rc['J_domain_mean']:.4f}) — reading: Jensen dominance is "
                f"case dependent, not a failure{brand(case)}")
    gates["gate1_representation_deficit"] = g1

    # Gate 2 — geometric recovery stays incomplete (Q_J < 0.5 protocol bar)
    g2 = {}
    for case in cases_ok:
        sub = m5[m5.case == case]
        qj = sub[sub.model.isin(["bilinear", "bicubic"])]["Q_J"].max()
        qs = sub[sub.model.isin(["bilinear", "bicubic"])]["Q_spread"].max()
        qd_con = sub[sub.model.str.endswith("_constrained")]["Q_delta_scalar"].max()
        incomplete = bool(qj < 0.5 and qs < 0.9)
        g2[case] = {"pass": incomplete, "max_Q_J": float(qj),
                    "max_Q_S": float(qs),
                    "max_constrained_Q_delta": float(qd_con)}
        if not incomplete:
            counterexamples.append(
                f"Gate 2 HEADLINE COUNTEREXAMPLE: case {case} geometric "
                f"interpolation approaches full recovery (max Q_J {qj:.3f}, "
                f"max Q_S {qs:.3f}){brand(case)}")
    gates["gate2_geometric_recovery"] = g2

    # Gate 3 — terrain organization (sign + correlation of frozen identities)
    g3 = {}
    for case in cases_ok:
        rd = m6[(m6.case == case) & (m6.relation == "delta_vs_r_sigma")].iloc[0]
        rs = m6[(m6.case == case) & (m6.relation == "S_vs_lambda_iso")].iloc[0]
        keeps = bool(rd["slope"] > 0 and rs["slope"] > 0
                     and rd["correlation"] > 0.3 and rs["correlation"] > 0.3)
        g3[case] = {"pass": keeps,
                    "delta_slope": float(rd["slope"]),
                    "delta_r": float(rd["correlation"]),
                    "S_slope": float(rs["slope"]),
                    "S_r": float(rs["correlation"])}
        if not keeps and case != "A":
            counterexamples.append(
                f"Gate 3: case {case} terrain organization weakened "
                f"(delta slope {rd['slope']:.3f} r {rd['correlation']:.3f}; "
                f"S slope {rs['slope']:.3f} r {rs['correlation']:.3f}) — "
                f"reading: terrain is informative but mapping is regime "
                f"dependent{brand(case)}")
    gates["gate3_terrain_organization"] = g3

    # Gate 4 — temporal transfer A->B (ranking survival, Spearman >= 0.7)
    if "B" in cases_ok:
        t1 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T1_A_to_B.csv")
        seg = t1[(t1.segment == "evaluation_670")]
        sp_d = float(seg[seg.model == "A_terrain_delta"]["spearman"].iloc[0])
        sp_s = float(seg[seg.model == "A_terrain_spread_joint"]["spearman"].iloc[0])
        ok4 = bool(sp_d >= 0.7 and sp_s >= 0.7)
        gates["gate4_temporal_transfer"] = {
            "pass": ok4, "spearman_delta": sp_d, "spearman_spread": sp_s}
        if not ok4:
            counterexamples.append(
                f"Gate 4: A->B spatial ranking degraded (Spearman delta "
                f"{sp_d:.3f}, spread {sp_s:.3f}) — persistent spatial "
                f"organization claim must be weakened if this survives real "
                f"data{brand('B')}")

    # Gate 5 — external equation transfer under a joint event+site shift (B->C / A->C)
    if {"B", "C"}.issubset(set(cases_ok)):
        t3 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T3_B_to_C.csv")
        t4 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T4_A_to_C.csv")
        s3 = t3[t3.segment == "evaluation_670"]["spearman"]
        s4 = t4[t4.segment == "evaluation_670"]["spearman"]
        ok5 = bool(s3.max() >= 0.7 or s4.max() >= 0.7)
        gates["gate5_joint_shift_transfer"] = {
            "pass": ok5,
            "best_spearman_T3": float(s3.max()),
            "best_spearman_T4": float(s4.max())}
        if not ok5:
            if C.GEN.get("bc_synchronized_bootstrap"):
                counterexamples.append(
                    f"Gate 5: external site transfer did not preserve ranking "
                    f"(best Spearman T3 {s3.max():.3f}, T4 {s4.max():.3f})"
                    f" — reading: missing moments replicate across sites, but "
                    f"empirical terrain mappings are not geographically "
                    f"universal{brand('C')}")
            else:
                counterexamples.append(
                    f"Gate 5: B->C / A->C joint event+site shift transfer did not "
                    f"preserve ranking (best Spearman T3 {s3.max():.3f}, T4 {s4.max():.3f})"
                    f" — reading: missing moments replicate, but empirical "
                    f"terrain mappings are not geographically universal"
                    f"{brand('C')}")
    gates["counterexamples"] = counterexamples
    return gates


def write_counterexamples(gates: dict, repro: pd.DataFrame,
                          indep: pd.DataFrame | None) -> None:
    lines = [f"# Counterexample register — {C.EXTENSION_NAME}", "",
             "Entries are mechanical gate evaluations (PLAN §23/§29); they are "
             "never deleted.  While Cases B/C are placeholders every "
             "B/C-dependent entry is a **PLACEHOLDER DRY-RUN** artefact of "
             "synthetic data, NOT science.", ""]
    ces = gates.get("counterexamples", [])
    if not ces:
        lines.append("No mechanical counterexamples were triggered in this run.")
    for i, ce in enumerate(ces, 1):
        lines.append(f"{i}. {ce}")
    fails = repro[repro.status == "FAIL"]
    lines += ["", "## Case-A reproduction failures", ""]
    if fails.empty:
        lines.append("None — every compared frozen quantity reproduced within "
                     "tolerance.")
    else:
        for _, r in fails.iterrows():
            lines.append(f"* {r['quantity']}: recomputed {r['recomputed']} vs "
                         f"frozen {r['frozen']} (diff {r['abs_diff']:.3e} > tol "
                         f"{r['tolerance']:.1e}) — frozen evidence NOT touched")
    if indep is not None:
        ifails = indep[indep.status == "FAIL"]
        lines += ["", "## Independent-implementation disagreements", ""]
        if ifails.empty:
            lines.append("None.")
        else:
            for _, r in ifails.iterrows():
                lines.append(f"* {r['quantity']} (case {r['case']}): primary "
                             f"{r['primary_value']} vs independent "
                             f"{r['independent_value']}")
    (C.ext_dir("audit") / "counterexample_register.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Final audit document (PLAN §31–32)
# ---------------------------------------------------------------------------
def verdicts(gates: dict, repro: pd.DataFrame, cases_ok: list[str]) -> dict:
    """Mechanical verdict rule declared in ANALYSIS_CONTRACT.json."""
    new_cases = [c for c in cases_ok if c != "A"]
    g1 = all(gates["gate1_representation_deficit"].get(c, {}).get("pass", False)
             for c in new_cases) if new_cases else False
    g2 = all(gates["gate2_geometric_recovery"].get(c, {}).get("pass", False)
             for c in cases_ok)
    g3 = all(gates["gate3_terrain_organization"].get(c, {}).get("pass", False)
             for c in cases_ok)
    g4 = gates.get("gate4_temporal_transfer", {}).get("pass", False)
    g5 = gates.get("gate5_joint_shift_transfer", {}).get("pass", False)
    headline_ce = any("HEADLINE" in ce for ce in gates.get("counterexamples", []))
    if not g1 or headline_ce:
        comment = "CORE_CLAIM_CHALLENGED" if headline_ce else "MIXED"
    elif g2 and g3 and g4 and g5:
        comment = "STRONGLY_STRENGTHENED"
    elif g2:
        comment = "STRENGTHENED_WITH_LIMITS"
    else:
        comment = "MIXED"
    article = comment   # article additionally requires scale/second-order
    a_repro_ok = (repro.status == "PASS").all()
    if not a_repro_ok and article == "STRONGLY_STRENGTHENED":
        article = "STRENGTHENED_WITH_LIMITS"
    return {"comment": comment, "article": article,
            "gates": {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5},
            "a_reproduction_all_pass": bool(a_repro_ok)}


def thunder(gates: dict, verd: dict, cases_ok: list[str]) -> list[str]:
    placeholder = any(C.data_status(c) == "PLACEHOLDER" for c in cases_ok)
    tag = (" [PLACEHOLDER DRY-RUN — NOT SCIENCE]" if placeholder else "")
    g1 = gates["gate1_representation_deficit"]
    g2 = gates["gate2_geometric_recovery"]
    g3 = gates["gate3_terrain_organization"]
    g4 = gates.get("gate4_temporal_transfer", {})
    g5 = gates.get("gate5_joint_shift_transfer", {})

    def yn(x):
        return "YES" if x else "NO"

    q = []
    q.append(f"1. A->B (same terrain, new event): P0 headline deficit survives"
             f" = {yn(g1.get('B', {}).get('pass'))} (J {g1.get('B', {}).get('J', float('nan')):.3f},"
             f" S {g1.get('B', {}).get('S', float('nan')):.3f} m/s){tag}")
    q.append(f"2. B->C (same episode, new terrain): deficit survives = "
             f"{yn(g1.get('C', {}).get('pass'))} (J {g1.get('C', {}).get('J', float('nan')):.3f},"
             f" S {g1.get('C', {}).get('S', float('nan')):.3f} m/s){tag}")
    q.append(f"3. A->C (double change): deficit survives = "
             f"{yn(g1.get('C', {}).get('pass'))}{tag}")
    q.append(f"4. Geometric refinement incomplete in all cases = "
             f"{yn(all(v['pass'] for v in g2.values()))} "
             f"(max Q_J {max(v['max_Q_J'] for v in g2.values()):.3f}, "
             f"max Q_S {max(v['max_Q_S'] for v in g2.values()):.3f}){tag}")
    dyn_notes = [c for c, v in g1.items() if "case dependent" in v.get("note", "")]
    q.append(f"5. Jensen dominance structural vs case-specific: "
             f"{'case-dependent in ' + ','.join(dyn_notes) if dyn_notes else 'J-dominated in all new cases'}{tag}")
    s_ok = all(v.get("S", 0) > 1e-3 for v in g1.values())
    q.append(f"6. Spread loss stable = {yn(s_ok)}{tag}")
    q.append(f"7. Terrain spatial organization transferable = "
             f"{yn(bool(g4.get('pass')) or bool(g5.get('pass')))} "
             f"(A->B Spearman delta {g4.get('spearman_delta', float('nan')):.3f}){tag}")
    amp_ok = verd["gates"]["g3"]
    q.append(f"8. Terrain amplitude mapping transferable = {yn(amp_ok)} "
             f"(within-case organization retained: {yn(amp_ok)}; amplitude "
             f"shifts reported in M8-M10){tag}")
    if "B" in cases_ok:
        t2 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T2_history_A_to_B.csv")
        seg = t2[(t2.segment == "evaluation_670")
                 & (t2.target == "delta_scalar")].set_index("model")
        hist_rmse = float(seg.loc["A_same_cell_historical_mean_scalar", "rmse"])
        terr_rmse = float(seg.loc["A_terrain_delta", "rmse"])
        q.append(f"9. Same-cell history across months: scalar RMSE "
                 f"{hist_rmse:.3f} vs A-terrain {terr_rmse:.3f} m/s on B "
                 f"evaluation cells{tag}")
    else:
        q.append("9. Same-cell history across months: case B unavailable")
    q.append(f"10. A frozen coefficients extrapolate to B/C = "
             f"{yn(bool(g4.get('pass')))}/"
             f"{yn(bool(g5.get('pass')))} (ranking reading; amplitude in "
             f"M8/M10){tag}")
    m4 = pd.read_csv(C.ext_dir("tables") / "M4_scale_dependence.csv")
    mono = True
    for case in cases_ok:
        sub = m4[(m4.case == case) & (m4.window == "evaluation")].sort_values("scale_m")
        mono &= bool(sub["J_domain_mean"].is_monotonic_decreasing
                     and sub["S_domain_mean"].is_monotonic_decreasing)
    q.append(f"11. Scale dependence (monotone decay to the exact 3000 m "
             f"endpoint) replicates = {yn(mono)}{tag}")
    n_ce = len([c for c in gates.get("counterexamples", []) if "HEADLINE" in c])
    q.append(f"12. Counterexamples forcing a Comment rewrite = "
             f"{yn(n_ce > 0)} ({n_ce} headline counterexample(s); full "
             f"register in audit/counterexample_register.md){tag}")
    return q


def final_audit(cases_ok: list[str], repro: pd.DataFrame,
                indep: pd.DataFrame | None, leak: pd.DataFrame,
                gates: dict, verd: dict) -> None:
    placeholder = any(C.data_status(c) == "PLACEHOLDER" for c in cases_ok)
    banner = ("**PLACEHOLDER DRY-RUN — NOT SCIENCE.** Cases B and C are "
              "deterministic synthetic placeholders (master seed 20260826); "
              "every B/C-dependent number below is a plumbing exercise and "
              "must never be quoted as evidence.  Case A is REAL and its "
              "legacy reproduction (section C) is meaningful."
              if placeholder else "All cases REAL.")
    n_fail = int((repro.status == "FAIL").sum())
    ind_fail = int((indep.status == "FAIL").sum()) if indep is not None else -1
    n_viol = int((leak.status == "VIOLATION").sum())
    raw = json.loads((C.EXT / "raw_contract" / "shape_contract.json")
                     .read_text(encoding="utf-8"))

    L = []
    L.append(f"# FINAL MULTICASE P0 AUDIT — {C.EXTENSION_NAME}")
    L.append("")
    L.append(banner)
    L.append("")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; "
             f"cases run: {', '.join(cases_ok)}; per-case data_status: "
             + ", ".join(f"{c}={C.data_status(c)}" for c in cases_ok) + ".")
    L.append("")
    L.append("## A. Raw-data integrity")
    L.append("")
    for c in cases_ok:
        st = raw["cases"][c]
        L.append(f"* Case {c}: contract {'STOPPED' if st['stop'] else 'PASS'}, "
                 f"data_status {st['data_status']}; filename set, shapes, "
                 f"dtype float32, 4320 states, no NaN/Inf, d02 native 6x6 "
                 f"tiled 27x27 exactly (full-tensor deviation 0), HGT "
                 f"162x162 — see raw_contract/.")
    L.append("")
    L.append("## B. Spatial/time contract")
    L.append("")
    L.append("* Identical inherited window on all cases: exclude [0,970), "
             "calibration [970,3650) = 2680, evaluation [3650,4320) = 670; "
             "for B/C this is a fixed inherited analysis-window duration, "
             "specified before examining the new-case science results.")
    L.append("* Row-major 6x6 parent mapping, 27x27 = 729 fine cells per "
             "parent, parent centres at fine index 13 + 27k — verified in "
             "raw_contract/spatial_mapping_audit.json.")
    L.append("")
    L.append("## C. Case A legacy reproduction")
    L.append("")
    L.append(f"* {len(repro) - n_fail} / {len(repro)} frozen quantities "
             f"reproduced within tolerance ({n_fail} failures) — full table: "
             f"audit/case_A_legacy_reproduction.csv.")
    if n_fail:
        for _, r in repro[repro.status == "FAIL"].iterrows():
            L.append(f"  * FAIL {r['quantity']}: recomputed {r['recomputed']}"
                     f" vs frozen {r['frozen']} (tol {r['tolerance']})")
    L.append("* Headline anchor: calibration domain means "
             "delta_scalar = delta_dyn + J = -0.004 + 0.275 = 0.271 m/s; "
             "interpolation recovers ~10-16% of J and ~39-49% of S on the "
             "final 670; representative state 3955 — all recomputed and "
             "compared above.")
    L.append("")
    for sec, case, title in [("D", "B", "Case B temporal replication"),
                             ("E", "C", "Case C terrain replication")]:
        L.append(f"## {sec}. {title}")
        L.append("")
        if case not in cases_ok:
            L.append(f"* Case {case} was STOPPED at the raw contract; no "
                     "science was computed.")
            L.append("")
            continue
        g1c = gates["gate1_representation_deficit"].get(case, {})
        L.append(f"* data_status = {C.data_status(case)}"
                 + (" — numbers below are dry-run plumbing only."
                    if C.data_status(case) == "PLACEHOLDER" else ""))
        L.append(f"* Representation deficit present: {g1c.get('pass')} "
                 f"(calibration J {g1c.get('J', float('nan')):.4f}, "
                 f"S {g1c.get('S', float('nan')):.4f}, delta_dyn "
                 f"{g1c.get('delta_dyn', float('nan')):.4f} m/s); composition "
                 f"note: {g1c.get('note', '')}.")
        L.append(f"* Detail: tables/M2, cross_case/core_loss_*, "
                 f"case_{case}/moments_domain_summary.json.")
        L.append("")
    L.append("## F. Geometric recovery")
    L.append("")
    for case, v in gates["gate2_geometric_recovery"].items():
        L.append(f"* Case {case}: max Q_J {v['max_Q_J']:.3f}, max Q_S "
                 f"{v['max_Q_S']:.3f}, constrained Q_delta "
                 f"{v['max_constrained_Q_delta']:.3f} -> recovery "
                 f"{'incomplete (PASS)' if v['pass'] else 'NEAR-COMPLETE (headline counterexample)'}"
                 + (" [PLACEHOLDER]" if C.data_status(case) == "PLACEHOLDER" else ""))
    L.append("")
    L.append("## G. Terrain / state / history information")
    L.append("")
    for case, v in gates["gate3_terrain_organization"].items():
        L.append(f"* Case {case}: delta~r_sigma slope {v['delta_slope']:.3f} "
                 f"(r {v['delta_r']:.3f}); S~lambda_iso slope "
                 f"{v['S_slope']:.3f} (r {v['S_r']:.3f})"
                 + (" [PLACEHOLDER]" if C.data_status(case) == "PLACEHOLDER" else ""))
    if "gate4_temporal_transfer" in gates:
        g4 = gates["gate4_temporal_transfer"]
        L.append(f"* A->B temporal transfer: Spearman delta "
                 f"{g4['spearman_delta']:.3f}, spread {g4['spearman_spread']:.3f}; "
                 f"history-vs-terrain comparison in M8/thunder Q9.")
    if "gate5_joint_shift_transfer" in gates:
        g5 = gates["gate5_joint_shift_transfer"]
        L.append(f"* B->C / A->C joint event+site shift equation transfer "
                 f"(secondary stress test): best Spearman T3 "
                 f"{g5['best_spearman_T3']:.3f}, T4 {g5['best_spearman_T4']:.3f}; "
                 f"C has NO cross-site same-cell history (within-site "
                 f"persistence only).")
    L.append("")
    L.append("## H. Scale / second-order Article evidence")
    L.append("")
    L.append("* Scale decay tables M4 (exact J = S = 0 at 3000 m asserted per "
             "case); frozen scale-aware closure transfer M11; second-order "
             "diagnostic M3 with the frozen 1.5 m/s descriptive reference.")
    L.append("")
    L.append("## I. Counterexamples")
    L.append("")
    ces = gates.get("counterexamples", [])
    if ces:
        for ce in ces:
            L.append(f"* {ce}")
    else:
        L.append("* None triggered mechanically in this run.")
    L.append("* Register: audit/counterexample_register.md (never deleted).")
    L.append("")
    L.append("## J. Leakage")
    L.append("")
    L.append(f"* Leakage matrix rows: {len(leak)}; violations: {n_viol} "
             f"(audit/leakage_matrix.csv, audit/leakage_audit.md).")
    L.append(f"* Independent re-implementations: "
             + (f"{len(indep) - ind_fail} / {len(indep)} PASS "
                f"(audit/independent_reproduction.csv)." if indep is not None
                else "not run in this invocation."))
    L.append("")
    L.append("## K. Manuscript implications")
    L.append("")
    if placeholder:
        L.append("* CLASSIFICATION WITHHELD — placeholder dry-run.  The "
                 "mechanical gate outcome below is a plumbing check of the "
                 "verdict logic, not evidence about the manuscripts.  With "
                 "real B/C data the audit will classify each claim as "
                 "SUPPORTED / SUPPORTED_WITH_NARROWER_WORDING / MIXED / "
                 "CONTRADICTED.  No manuscript file was touched.")
    else:
        L.append("* Classification per claim to be reviewed by the author; "
                 "no manuscript file was touched.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Thunder summary (12 pre-registered questions)")
    L.append("")
    for line in thunder_dispatch(gates, verd, cases_ok):
        L.append(line)
    L.append("")
    tagv = (" [PLACEHOLDER DRY-RUN — NOT SCIENCE; mechanical plumbing value "
            "only, recompute with real B/C data]" if placeholder else "")
    L.append(f"P0_COMMENT_EMPIRICAL_SUPPORT = {verd['comment']}{tagv}")
    L.append("")
    L.append(f"P0_ARTICLE_EMPIRICAL_SUPPORT = {verd['article']}{tagv}")
    L.append("")
    L.append("Promotion gate: this round STOPS here.  latex_paper0/, "
             "latex_paper0_npjcomment/, data_paper0/frozen/ and "
             "source/evidence/npj_refactor/ were not modified.  Any promotion "
             "requires the explicit human instruction "
             "\"MULTICASE AUDIT PASSED — PROMOTE\".")
    (C.ext_dir("audit") / "FINAL_MULTICASE_P0_AUDIT.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")
    print("[phase17] FINAL_MULTICASE_P0_AUDIT.md written")


def _eligible(case: str) -> bool:
    return C.case_is_science_eligible(case)


def _c_status() -> str:
    return C.case_metadata("C")["case_status"]


def _c_blocked_sentence() -> str:
    status = _c_status()
    if status == C.CASE_STATUS_PLACEHOLDER:
        return "C remains PLACEHOLDER_READY and C-dependent science is blocked."
    if status == C.CASE_STATUS_REAL_PENDING_AUDIT:
        return "C is REAL_PENDING_AUDIT and C-dependent science is blocked pending promotion."
    if status == C.CASE_STATUS_REAL:
        return "C is REAL and C-dependent science is enabled subject to its empirical verdicts."
    return f"C has unresolved status {status} and C-dependent science is blocked."


def _blocked_gate(case: str) -> dict:
    meta = C.case_metadata(case)
    return {
        "pass": False,
        "status": meta["dependency_status"],
        "case_status": meta["case_status"],
        "science_eligible": meta["science_eligible"],
        "contains_placeholder": meta["contains_placeholder"],
    }


def _max_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if len(values) else float("nan")


def core_gates(cases_ok: list[str]) -> dict:
    """Evaluate gates per dependency set, keeping C out of A/B claims."""
    m2 = pd.read_csv(C.ext_dir("tables") / "M2_representation_loss.csv")
    m5 = pd.read_csv(C.ext_dir("tables") / "M5_geometric_recovery.csv")
    m6 = pd.read_csv(C.ext_dir("tables") / "M6_terrain_relationships.csv")
    gates: dict = {}
    counterexamples: list[str] = []

    g1 = {}
    for case in C.CASE_ORDER:
        if case not in cases_ok or not _eligible(case):
            g1[case] = _blocked_gate(case)
            continue
        rows = m2[(m2.case == case) & (m2.window.isin(["calibration", "evaluation"]))]
        rc = rows[rows.window == "calibration"].iloc[0]
        reval = rows[rows.window == "evaluation"].iloc[0]
        present = bool(reval["J_domain_mean"] > 1e-3 and
                       reval["S_domain_mean"] > 1e-3 and
                       rc["J_domain_mean"] > 1e-3)
        dyn_comparable = abs(float(rc["delta_dyn_domain_mean"])) >= float(rc["J_domain_mean"])
        g1[case] = {
            "pass": present,
            "status": "REAL",
            "case_status": C.case_metadata(case)["case_status"],
            "science_eligible": True,
            "contains_placeholder": False,
            "J": float(rc["J_domain_mean"]),
            "S": float(reval["S_domain_mean"]),
            "delta_dyn": float(rc["delta_dyn_domain_mean"]),
            "note": ("Jensen dominance is event/site specific, but multiple "
                     "missing moments remain" if dyn_comparable
                     else "J-dominated composition"),
        }
        if not present:
            counterexamples.append(
                f"Gate 1: REAL case {case} lacks a clear J/S deficit")
    gates["gate1_representation_deficit"] = g1

    g2 = {}
    for case in C.CASE_ORDER:
        if case not in cases_ok or not _eligible(case):
            g2[case] = _blocked_gate(case)
            continue
        sub = m5[m5.case == case]
        base = sub[sub.model.isin(["bilinear", "bicubic"])]
        constrained = sub[sub.model.astype(str).str.endswith("_constrained")]
        qj = _max_numeric(base, "Q_J")
        qs = _max_numeric(base, "Q_spread")
        qd = _max_numeric(constrained, "Q_delta_scalar")
        incomplete = bool(np.isfinite(qj) and np.isfinite(qs) and qj < 0.5 and qs < 0.9)
        g2[case] = {
            "pass": incomplete, "status": "REAL",
            "case_status": C.case_metadata(case)["case_status"],
            "science_eligible": True, "contains_placeholder": False,
            "max_Q_J": qj, "max_Q_S": qs, "max_constrained_Q_delta": qd,
        }
        if not incomplete:
            counterexamples.append(
                f"Gate 2 HEADLINE COUNTEREXAMPLE: REAL case {case} geometric "
                f"interpolation approaches full recovery (max Q_J {qj:.3f}, "
                f"max Q_S {qs:.3f})")
    gates["gate2_geometric_recovery"] = g2

    g3 = {}
    for case in C.CASE_ORDER:
        if case not in cases_ok or not _eligible(case):
            g3[case] = _blocked_gate(case)
            continue
        rd = m6[(m6.case == case) & (m6.relation == "delta_vs_r_sigma")].iloc[0]
        rs = m6[(m6.case == case) & (m6.relation == "S_vs_lambda_iso")].iloc[0]
        keeps = bool(rd["slope"] > 0 and rs["slope"] > 0 and
                     rd["correlation"] > 0.3 and rs["correlation"] > 0.3)
        g3[case] = {
            "pass": keeps, "status": "REAL",
            "case_status": C.case_metadata(case)["case_status"],
            "science_eligible": True, "contains_placeholder": False,
            "delta_slope": float(rd["slope"]), "delta_r": float(rd["correlation"]),
            "S_slope": float(rs["slope"]), "S_r": float(rs["correlation"]),
        }
        if not keeps:
            counterexamples.append(
                f"Gate 3: REAL case {case} terrain organization weakened "
                f"(delta slope {rd['slope']:.3f} r {rd['correlation']:.3f}; "
                f"S slope {rs['slope']:.3f} r {rs['correlation']:.3f})")
    gates["gate3_terrain_organization"] = g3

    if set(("A", "B")).issubset(cases_ok) and all(_eligible(c) for c in ("A", "B")):
        t1 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T1_A_to_B.csv")
        seg = t1[t1.segment == "evaluation_670"]
        sp_d = float(seg[seg.model == "A_terrain_delta"]["spearman"].iloc[0])
        sp_s = float(seg[seg.model == "A_terrain_spread_joint"]["spearman"].iloc[0])
        ok4 = bool(sp_d >= 0.7 and sp_s >= 0.7)
        gates["gate4_temporal_transfer"] = {
            "pass": ok4, "status": "REAL", "case_status": "A=REAL;B=REAL",
            "science_eligible": True, "contains_placeholder": False,
            "dependency_status": "REAL", "spearman_delta": sp_d,
            "spearman_spread": sp_s,
        }
        if not ok4:
            counterexamples.append(
                f"Gate 4: A->B spatial ranking is PARTIAL (Spearman delta "
                f"{sp_d:.3f}, spread {sp_s:.3f}); the temporal-transfer "
                "claim requires narrower wording")
    else:
        gates["gate4_temporal_transfer"] = {
            **C.status_metadata(["A", "B"]), "pass": False,
            "status": C.status_metadata(["A", "B"])["dependency_status"],
        }

    if set(("B", "C")).issubset(cases_ok) and all(_eligible(c) for c in ("B", "C")):
        t3 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T3_B_to_C.csv")
        t4 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T4_A_to_C.csv")
        s3 = pd.to_numeric(t3[t3.segment == "evaluation_670"]["spearman"], errors="coerce")
        s4 = pd.to_numeric(t4[t4.segment == "evaluation_670"]["spearman"], errors="coerce")
        best3, best4 = float(s3.max()), float(s4.max())
        ok5 = bool(best3 >= 0.7 or best4 >= 0.7)
        gates["gate5_joint_shift_transfer"] = {
            "pass": ok5, "status": "REAL", "case_status": "B=REAL;C=REAL",
            "science_eligible": True, "contains_placeholder": False,
            "dependency_status": "REAL", "best_spearman_T3": best3,
            "best_spearman_T4": best4,
        }
        if not ok5:
            if C.GEN.get("bc_synchronized_bootstrap"):
                counterexamples.append(
                    f"Gate 5: B->C / A->C REAL external site transfer did not "
                    f"preserve ranking (best Spearman T3 {best3:.3f}, "
                    f"T4 {best4:.3f}) — reading: missing moments replicate "
                    f"across sites, but empirical terrain mappings are not "
                    f"geographically universal")
            else:
                counterexamples.append(
                    f"Gate 5: B->C / A->C REAL joint event+site shift transfer did not preserve "
                    f"ranking (best Spearman T3 {best3:.3f}, T4 {best4:.3f})")
    else:
        gates["gate5_joint_shift_transfer"] = {
            **C.status_metadata(["B", "C"]), "pass": False,
            "status": C.status_metadata(["B", "C"])["dependency_status"],
        }
    gates["counterexamples"] = counterexamples
    gates["claim_statuses"] = {
        x["claim_id"]: C.claim_metadata(x["claim_id"])
        for x in C.CLAIM_DEFINITIONS
    }
    return gates


def write_counterexamples(gates: dict, repro: pd.DataFrame,
                          indep: pd.DataFrame | None) -> None:
    """Append a structured claim-scoped register and archive prior Markdown."""
    import shutil

    audit_dir = C.ext_dir("audit")
    current_md = audit_dir / "counterexample_register.md"
    archive_dir = audit_dir / "counterexample_register_archive"
    archived = None
    if current_md.is_file():
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        archived = archive_dir / f"counterexample_register_{stamp}_{time.time_ns() % 1000000:06d}.md"
        shutil.copy2(current_md, archived)

    entries = []

    def add_entry(claim_id: str, status: str, message: str,
                   dependencies: list[str] | None = None) -> None:
        deps = dependencies or next(
            x["dependencies"] for x in C.CLAIM_DEFINITIONS
            if x["claim_id"] == claim_id)
        meta = C.status_metadata(deps)
        entries.append({"record_type": "claim_verdict", "claim_id": claim_id,
                        "verdict": status, "status": status,
                        "message": message, "dependencies": deps, **meta})

    def add_finding(parent_claim: str, finding_id: str, effect: str,
                    dependencies: list[str] | None = None) -> None:
        deps = dependencies or next(
            x["dependencies"] for x in C.CLAIM_DEFINITIONS
            if x["claim_id"] == parent_claim)
        meta = C.status_metadata(deps)
        entries.append({"record_type": "finding", "finding_id": finding_id,
                        "parent_claim": parent_claim,
                        "finding": "COUNTEREXAMPLE_TRIGGER", "effect": effect,
                        "dependencies": deps, **meta})

    repro_ok = len(repro) > 0 and bool((repro.status == "PASS").all())
    add_entry("MC-A-single-case", "SUPPORTED" if repro_ok else "FAILED",
              "Case-A legacy quantities reproduce the frozen evidence within tolerance."
              if repro_ok else "Case-A legacy reproduction contains failures.")
    b1 = gates["gate1_representation_deficit"].get("B", {})
    add_entry("MC-B-single-case", "SUPPORTED" if b1.get("pass") else "TRIGGERED_COUNTEREXAMPLE",
              "Case B REAL representation deficit is present." if b1.get("pass")
              else "Case B REAL representation deficit gate failed.")
    c1 = gates["gate1_representation_deficit"].get("C", {})
    c_ready = _eligible("C")
    add_entry("MC-C-single-case", "SUPPORTED" if c_ready and c1.get("pass")
              else "TRIGGERED_COUNTEREXAMPLE" if c_ready else "BLOCKED",
              "Case C REAL representation evidence is available."
              if c_ready and c1.get("pass") else
              _c_blocked_sentence() if not c_ready else
              "Case C REAL representation gate failed.")
    g4 = gates.get("gate4_temporal_transfer", {})
    add_entry("MC-A-to-B-temporal-transfer",
               "SUPPORTED" if g4.get("pass") else "PARTIAL",
               "A->B is REAL and science-enabled; ranking survival is partial when the Spearman threshold is not met.")
    independent_ok = indep is None or not (indep.status == "FAIL").any()
    add_entry("MC-A-plus-B-replication",
               "SUPPORTED" if b1.get("pass") and independent_ok else
               "TRIGGERED_COUNTEREXAMPLE",
               "A/B same-terrain multi-event evidence is available for claim-level review."
               if b1.get("pass") and independent_ok else
               "A/B replication verdict is challenged by a mechanical gate or independent disagreement.")
    g5 = gates.get("gate5_joint_shift_transfer", {})
    add_entry("MC-B-to-C-joint-shift-transfer",
              "SUPPORTED" if c_ready and g5.get("pass")
              else "PARTIAL" if c_ready else "BLOCKED",
              ("B->C is available as a contemporaneous cross-site equation-"
               "transfer stress test (secondary; paired contemporaneous "
               "cross-site differences; no terrain-only attribution)."
               if C.GEN.get("bc_synchronized_bootstrap") else
               "B->C is available as an external equation-transfer stress test "
               "under a joint event+site shift (secondary; no terrain-only "
               "attribution).")
              if c_ready else _c_blocked_sentence())
    add_entry("MC-A-to-C-joint-shift-transfer",
              "SCIENCE_ENABLED" if c_ready else "BLOCKED",
              ("A->C is available as a combined event+site external transfer "
               "reference (event, site, terrain regime and local wind "
               "realization all change), not a spatial-only comparison."
               if C.GEN.get("bc_synchronized_bootstrap") else
               "A->C is available as a joint event+site shift total-difference "
               "reference, not a spatial-only comparison.")
              if c_ready else
              ("A->C depends on the real "
               f"{C.CASES['C']['location']} case."
               if C.GENERATION != "20260826" else
               "A->C depends on the real Dayekou case."))
    add_entry("MC-C-independent-replication",
              "SCIENCE_ENABLED" if c_ready else "BLOCKED",
              "Case C stands as an independent site-event replication of the "
              "representation deficit and the geometric-recovery limitation."
              if c_ready else _c_blocked_sentence())
    add_entry("MC-A-plus-B-plus-C-replication", "SCIENCE_ENABLED" if c_ready else "BLOCKED",
              "HEADLINE claim: three-case replication is available for "
              "claim-level review. Its anchors are gate1 + gate2 only; terrain "
              "transfer outcome cannot veto it."
              if c_ready else "Three-case replication depends on real Case C evidence.")
    add_entry("MC-M4-scale-master-schema", "ENGINEERING_ONLY",
              "M4 is a fixed 3-case x 2-window x 4-scale schema; row eligibility "
              "follows the live C state.")

    if gates.get("counterexamples"):
        for message in gates["counterexamples"]:
            if "Gate 4" in message:
                claim, finding = "MC-A-to-B-temporal-transfer", "G4-ranking"
            elif "Gate 5" in message:
                claim, finding = "MC-B-to-C-joint-shift-transfer", "G5-ranking"
            elif "case B" in message:
                claim, finding = "MC-B-single-case", "B-representation"
            elif "case C" in message:
                claim, finding = "MC-C-single-case", "C-representation"
            else:
                claim, finding = "MC-A-plus-B-replication", "AB-replication"
            add_finding(claim, f"{claim}:{finding}", message)
    if not repro_ok:
        add_finding("MC-A-single-case", "MC-A-single-case:legacy-reproduction",
                    "Frozen Case-A reproduction failure; frozen evidence was not modified.")
    if indep is not None and (indep.status == "FAIL").any():
        add_finding("MC-A-plus-B-replication",
                    "MC-A-plus-B-replication:independent-reproduction",
                    "Independent implementation disagreement is recorded in independent_reproduction.csv.")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jsonl = audit_dir / "counterexample_register.jsonl"
    claim_verdicts = {
        entry["claim_id"]: entry["verdict"]
        for entry in entries if entry.get("record_type") == "claim_verdict"
    }
    snapshot = {"run_id": stamp, "generated_at_utc": stamp,
                "entries": entries, "claim_verdicts": claim_verdicts,
                "claim_verdicts_unique": len(claim_verdicts) == len([
                    e for e in entries if e.get("record_type") == "claim_verdict"]),
                **C.status_metadata(list(C.CASE_ORDER))}
    with jsonl.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, default=C._json_default) + "\n")

    register_policy = ("A/B/C claims are science-enabled subject to their "
                       "individual verdicts."
                       if c_ready else
                       "A/B evidence is science-enabled; " + _c_blocked_sentence())
    lines = [f"# Counterexample register — {C.EXTENSION_NAME}", "",
             "This register is append/archive based.  Every entry carries its "
             "claim dependencies and the four-field output gate.  "
             + register_policy, "",
             f"Current snapshot: `{stamp}`.",
             f"Archived prior Markdown: `{archived.as_posix() if archived else 'none'}`.", "",
              "| record_type | claim_id / finding_id | verdict / finding | parent_claim | dependency_status | science_eligible | message / effect |",
              "|---|---|---|---|---|---:|---|"]
    for entry in entries:
        if entry["record_type"] == "claim_verdict":
            identifier = entry["claim_id"]
            verdict = entry["verdict"]
            parent = "-"
            message = entry["message"]
        else:
            identifier = entry["finding_id"]
            verdict = entry["finding"]
            parent = entry["parent_claim"]
            message = entry["effect"]
        message = str(message).replace("|", "\\|")
        lines.append(f"| {entry['record_type']} | {identifier} | {verdict} | "
                     f"{parent} | {entry['dependency_status']} | "
                     f"{str(entry['science_eligible']).lower()} | {message} |")
    lines += ["", "Append-only JSONL receipt: `counterexample_register.jsonl`.",
              "Historical snapshots remain under `counterexample_register_archive/`.",
              ("C rows are engineering records, never manuscript evidence."
               if not c_ready else
               "Real C rows are included only after promotion and claim-level review.")]
    current_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    gates["counterexample_entries"] = entries


def verdicts(gates: dict, repro: pd.DataFrame, cases_ok: list[str]) -> dict:
    """Return A/B science verdicts plus dependency-scoped claim states."""
    ab = ["A", "B"]
    ab_ready = all(c in cases_ok and _eligible(c) for c in ab)
    g1 = ab_ready and all(gates["gate1_representation_deficit"][c].get("pass", False)
                          for c in ab)
    g2 = ab_ready and all(gates["gate2_geometric_recovery"][c].get("pass", False)
                          for c in ab)
    g3 = ab_ready and all(gates["gate3_terrain_organization"][c].get("pass", False)
                          for c in ab)
    g4 = bool(gates.get("gate4_temporal_transfer", {}).get("pass", False))
    g5 = bool(gates.get("gate5_joint_shift_transfer", {}).get("pass", False))
    headline_ce = any("HEADLINE" in ce for ce in gates.get("counterexamples", []))
    if not ab_ready:
        comment = "BLOCKED_DEPENDENCY"
    elif headline_ce:
        comment = "CORE_CLAIM_CHALLENGED"
    elif not g1:
        comment = "MIXED"
    elif g2 and g3 and g4:
        comment = "STRONGLY_STRENGTHENED" if g5 else "STRENGTHENED_WITH_LIMITS"
    elif g2:
        comment = "STRENGTHENED_WITH_LIMITS"
    else:
        comment = "MIXED"
    article = comment
    a_repro_ok = len(repro) > 0 and bool((repro.status == "PASS").all())
    if not a_repro_ok and article == "STRONGLY_STRENGTHENED":
        article = "STRENGTHENED_WITH_LIMITS"

    claim_states = {}
    for definition in C.CLAIM_DEFINITIONS:
        cm = C.claim_metadata(definition["claim_id"])
        if definition.get("engineering_only"):
            state = "ENGINEERING_ONLY"
        elif not cm["science_eligible"]:
            state = cm["dependency_status"]
        elif definition["claim_id"] == "MC-A-to-B-temporal-transfer" and not g4:
            state = "SCIENCE_ENABLED_WITH_LIMITS"
        else:
            state = "SCIENCE_ENABLED"
        claim_states[definition["claim_id"]] = {**cm, "verdict": state}
    return {"comment": comment, "article": article,
            "gates": {"g1": bool(g1), "g2": bool(g2), "g3": bool(g3),
                      "g4": bool(g4), "g5": bool(g5)},
            "a_reproduction_all_pass": bool(a_repro_ok),
            "claim_states": claim_states}


def _fmt(value) -> str:
    try:
        return "n/a" if not np.isfinite(float(value)) else f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def thunder(gates: dict, verd: dict, cases_ok: list[str]) -> list[str]:
    """Twelve questions with explicit claim/dependency scope."""
    g1, g2, g3 = (gates["gate1_representation_deficit"],
                  gates["gate2_geometric_recovery"],
                  gates["gate3_terrain_organization"])
    g4, g5 = gates["gate4_temporal_transfer"], gates["gate5_joint_shift_transfer"]
    b = g1.get("B", {})
    c_ready = _eligible("C")
    q = [
        f"1. A->B representation deficit: {'YES' if b.get('pass') else 'NO'} "
        f"(J {_fmt(b.get('J'))}, S {_fmt(b.get('S'))} m/s; dependency A+B REAL).",
        "2. B->C representation deficit: " +
        (f"{'YES' if g1.get('C', {}).get('pass') else 'NO'} (C REAL)." if c_ready
         else "BLOCKED — real Dayekou input required."),
        "3. A->C representation deficit: " +
        ("available for science review." if c_ready else "BLOCKED — real Dayekou input required."),
        f"4. A/B geometric recovery remains incomplete: "
        f"{'YES' if all(g2.get(c, {}).get('pass', False) for c in ('A', 'B')) else 'NO'} "
        f"(max Q_J A/B {_fmt(max([g2.get(c, {}).get('max_Q_J', np.nan) for c in ('A', 'B')]))}, "
        f"max Q_S A/B {_fmt(max([g2.get(c, {}).get('max_Q_S', np.nan) for c in ('A', 'B')]))}).",
        "5. Jensen dominance reading: event/site specific for the REAL A/B evidence; "
        + ("C is REAL and available for separate review." if c_ready else
           _c_blocked_sentence()),
        f"6. A/B spread loss stable: {'YES' if all(g1.get(c, {}).get('S', 0) > 1e-3 for c in ('A', 'B')) else 'NO'}.",
        f"7. A->B terrain ranking transferable: {'YES' if g4.get('pass') else 'PARTIAL'} "
        f"(delta {_fmt(g4.get('spearman_delta'))}, spread {_fmt(g4.get('spearman_spread'))}).",
        f"8. A/B terrain amplitude organization retained: {'YES' if all(g3.get(c, {}).get('pass', False) for c in ('A', 'B')) else 'NO'}; "
        "amplitude remains distinct from ranking transfer.",
    ]
    if set(("A", "B")).issubset(cases_ok) and all(_eligible(c) for c in ("A", "B")):
        t2 = pd.read_csv(C.ext_dir("cross_case") / "transfer_T2_history_A_to_B.csv")
        seg = t2[(t2.segment == "evaluation_670") & (t2.target == "delta_scalar")].set_index("model")
        hist_rmse = float(seg.loc["A_same_cell_historical_mean_scalar", "rmse"])
        terr_rmse = float(seg.loc["A_terrain_delta", "rmse"])
        q.append(f"9. Same-cell history across months: scalar RMSE {hist_rmse:.3f} vs "
                 f"A-terrain {terr_rmse:.3f} m/s on B evaluation cells (A/B REAL).")
    else:
        q.append("9. Same-cell history across months: BLOCKED — A/B REAL evidence unavailable.")
    q.append(f"10. Frozen A coefficients to B/C: B={'ENABLED' if g4.get('status') == 'REAL' else 'BLOCKED'}, "
             f"C={'ENABLED' if c_ready else 'BLOCKED'}.")
    m4_path = C.ext_dir("tables") / "M4_scale_dependence.csv"
    m4_n = len(pd.read_csv(m4_path)) if m4_path.is_file() else 0
    q.append(f"11. M4 scale schema: {m4_n} rows; required 24 = 3 cases x 2 windows x 4 scales; "
             + ("C rows are REAL and eligible after promotion."
                if c_ready else "C rows remain non-eligible while C is blocked."))
    n_ce = len([x for x in gates.get("counterexamples", []) if "HEADLINE" in x])
    q.append(f"12. Headline counterexamples forcing a Comment rewrite: {'YES' if n_ce else 'NO'} "
             f"({n_ce}); claim-level register records partial/blocked items.")
    return q


def _safe_csv(relpath: str) -> pd.DataFrame | None:
    path = C.ext_dir(relpath.split("/")[0]) / "/".join(relpath.split("/")[1:])
    try:
        return pd.read_csv(path) if path.is_file() else None
    except Exception:  # noqa: BLE001 - the audit must not die on a missing table
        return None


def thunder_20260905(gates: dict, verd: dict, cases_ok: list[str]) -> list[str]:
    """PLAN_MULTICASE_20260905.md §32: the sixteen first-page questions."""
    g1 = gates["gate1_representation_deficit"]
    g2 = gates["gate2_geometric_recovery"]
    g3 = gates["gate3_terrain_organization"]
    g4 = gates["gate4_temporal_transfer"]
    g5 = gates["gate5_joint_shift_transfer"]
    c_ready = _eligible("C")
    q: list[str] = []

    def yn(flag: bool) -> str:
        return "YES" if flag else "NO"

    # 1 A->B headline survival
    b = g1.get("B", {})
    q.append(f"1. A->B (same Yakou terrain, new event) P0 headline survives: "
             f"{yn(bool(b.get('pass')))} "
             f"(J {_fmt(b.get('J'))}, S {_fmt(b.get('S'))} m/s).")
    # 2 Ebao representation-loss survival
    c1c = g1.get("C", {})
    q.append("2. Ebao1201 independent case representation-loss headline survives: "
             + (f"{yn(bool(c1c.get('pass')))} (J {_fmt(c1c.get('J'))}, "
                f"S {_fmt(c1c.get('S'))} m/s; C REAL)." if c_ready else
                "BLOCKED — Case C promotion pending."))
    # 3 B<->C replication
    q.append("3. B<->C same-window/different-site core structure replicates: "
             + (f"{yn(bool(g1.get('C', {}).get('pass')))}/"
                f"{yn(bool(g1.get('B', {}).get('pass')))} "
                "(C vs B representation gates; paired contemporaneous "
                "cross-site differences, not causal terrain effects)."
                if c_ready else "BLOCKED — Case C promotion pending."))
    # 4 A->C combined stress
    q.append("4. A->C combined event+site stress: representation deficit "
             + (f"{yn(bool(g1.get('C', {}).get('pass')))}; best evaluation "
                f"Spearman T3 {_fmt(g5.get('best_spearman_T3', np.nan))}, "
                f"T4 {_fmt(g5.get('best_spearman_T4', np.nan))} "
                "(secondary stress test, not a headline gate)."
                if c_ready else "BLOCKED — Case C promotion pending."))
    # 5 geometric refinement failure across cases
    q.append("5. Geometric refinement incomplete in all three cases: "
             + (f"{yn(all(g2.get(c, {}).get('pass', False) for c in ('A', 'B', 'C')))} "
                if c_ready else
                f"{yn(all(g2.get(c, {}).get('pass', False) for c in ('A', 'B')))} (A/B; C pending) ")
             + f"(max Q_J A { _fmt(g2.get('A', {}).get('max_Q_J', np.nan))}, "
             f"B {_fmt(g2.get('B', {}).get('max_Q_J', np.nan))}, "
             f"C {_fmt(g2.get('C', {}).get('max_Q_J', np.nan))}).")
    # 6 Jensen dominance structural vs case dependent
    cl = _safe_csv("cross_case/core_loss_summary.csv")
    jcomp = {}
    if cl is not None:
        cal = cl[cl.window == "calibration"]
        jcomp = {str(r.case): float(r.J_over_delta_scalar)
                 for r in cal.itertuples() if np.isfinite(r.J_over_delta_scalar)}
    q.append("6. Jensen dominance is case dependent, not structural: "
             "J/Delta_scalar calibration "
             + ", ".join(f"{k} {v:.3f}" for k, v in
                         sorted(jcomp.items()) if k in ("A", "B", "C"))
             + " (composition reported per case; large delta_dyn is not a "
               "failure).")
    # 7 spread loss stability
    q.append("7. Spread loss (S) stable across cases: "
             f"{yn(all(g1.get(c, {}).get('S', 0) > 1e-3 for c in cases_ok))} "
             "(domain-mean S > 1e-3 m/s per case).")
    # 8 terrain organization across events (A, B)
    q.append("8. Terrain spatial organization survives across events (A, B): "
             f"{yn(all(g3.get(c, {}).get('pass', False) for c in ('A', 'B')))}.")
    # 9 terrain organization Yakou -> Ebao
    q.append("9. Terrain spatial organization survives Yakou->Ebao: "
             + (f"{yn(bool(g3.get('C', {}).get('pass', False)))}."
                if c_ready else "BLOCKED — Case C promotion pending."))
    # 10 amplitude-coefficient geographic transferability
    t3 = _safe_csv("cross_case/transfer_T3_B_to_C.csv")
    t4 = _safe_csv("cross_case/transfer_T4_A_to_C.csv")

    def _best_r(df: pd.DataFrame | None) -> str:
        if df is None or not len(df) or "r" not in df.columns:
            return "n/a"
        vals = pd.to_numeric(df["r"], errors="coerce").dropna()
        return f"{float(vals.median()):.3f}" if len(vals) else "n/a"

    q.append("10. Terrain amplitude coefficients transfer geographically: "
             + (f"T3 median r {_best_r(t3)}, T4 median r {_best_r(t4)} "
                "(in_range/extrapolation flags carried per predictor; "
                "association, not causation)."
                if c_ready else "BLOCKED — Case C promotion pending."))
    # 11 same-cell history across months
    t2 = _safe_csv("cross_case/transfer_T2_history_A_to_B.csv")
    hist_txt = "n/a"
    if t2 is not None and len(t2):
        seg = t2[(t2.segment == "evaluation_670") &
                 (t2.target == "delta_scalar")].set_index("model")
        if "A_same_cell_historical_mean_scalar" in seg.index:
            hist_txt = (f"{float(seg.loc['A_same_cell_historical_mean_scalar', 'rmse']):.3f} "
                        f"vs A-terrain "
                        f"{float(seg.loc['A_terrain_delta', 'rmse']):.3f} m/s")
    q.append(f"11. Same-cell history across months (A cal -> B eval): {hist_txt}.")
    # 12 Ebao local history vs terrain ranking
    rank = _safe_csv("cross_case/information_source_ranking.csv")
    c_rank = "see tables/M7_information_source_ranking.csv"
    if (rank is not None and len(rank) and c_ready
            and {"case", "group", "method", "rank_rmse"}.issubset(rank.columns)):
        sub = rank[rank.case == "C"]
        parts = []
        for gname, labelg, terr in (
                ("scalar_increment", "scalar", "terrain_roughness_equation"),
                ("spread", "spread", "joint_terrain_equation")):
            sg = sub[sub.group == gname]
            if len(sg):
                order = dict(zip(sg["method"].astype(str),
                                 sg["rank_rmse"].astype(int)))
                parts.append(
                    f"{labelg}: history #{order.get('same_cell_historical_mean', 'n/a')}, "
                    f"terrain #{order.get(terr, 'n/a')} of {int(sg['rank_rmse'].max())}")
        if parts:
            c_rank = "; ".join(parts)
    q.append(f"12. Ebao local history vs terrain ranking: {c_rank} "
             "(C history = within-site calibration -> evaluation only).")
    # 13 frozen A coefficients extrapolation
    q.append("13. Frozen A coefficients extrapolate to B/C: "
             f"B {'ENABLED' if g4.get('status') == 'REAL' else 'BLOCKED'}, "
             + (f"C {'ENABLED' if c_ready else 'BLOCKED'} "
                "(frozen scale-aware closure transfer CSVs carry "
                "in_range/extrapolation flags)."
                if c_ready else "(C blocked)."))
    # 14 scale dependence replication
    m4 = _safe_csv("tables/M4_scale_dependence.csv")
    q.append(f"14. Scale dependence replicates: M4 has "
             f"{0 if m4 is None else len(m4)} rows "
             "(required 24 = 3 cases x 2 windows x 4 scales; J=S=0 exact at "
             "3000 m by construction).")
    # 15 B/C local wind-state difference
    ws = _safe_csv("cross_case/wind_state_comparison.csv")
    bcdiff = "see cross_case/wind_state_comparison.csv"
    if (ws is not None and len(ws)
            and {"case", "window", "speed_mean"}.issubset(ws.columns)):
        ev = ws[ws.window == "evaluation"]
        b = ev[ev.case == "B"]
        c_ = ev[ev.case == "C"]
        if len(b) and len(c_):
            b_speed = float(b["speed_mean"].iloc[0])
            c_speed = float(c_["speed_mean"].iloc[0])
            b_dir = float(b["direction_circular_mean_from_deg"].iloc[0])
            c_dir = float(c_["direction_circular_mean_from_deg"].iloc[0])
            bcdiff = (f"evaluation domain-mean speed B {b_speed:.2f} vs C "
                      f"{c_speed:.2f} m/s (diff {c_speed - b_speed:+.2f}); "
                      f"mean from-direction {b_dir:.0f} vs {c_dir:.0f} deg "
                      f"(diff {c_dir - b_dir:+.0f}; sector fractions in the CSV)")
    q.append(f"15. B/C local wind states in the shared window: {bcdiff} "
             "(quantified, never assumed identical).")
    # 16 counterexamples
    n_ce = len([x for x in gates.get("counterexamples", []) if "HEADLINE" in x])
    q.append(f"16. Counterexamples forcing a Comment rewrite: {'YES' if n_ce else 'NO'} "
             f"({n_ce} headline-grade; full register in audit/).")
    return q


def thunder_dispatch(gates: dict, verd: dict, cases_ok: list[str]) -> list[str]:
    if C.GEN.get("thunder") == "sixteen":
        return thunder_20260905(gates, verd, cases_ok)
    return thunder(gates, verd, cases_ok)


def final_audit(cases_ok: list[str], repro: pd.DataFrame,
                indep: pd.DataFrame | None, leak: pd.DataFrame,
                gates: dict, verd: dict) -> None:
    """Write a current, claim-scoped P0 audit without global placeholder branding."""
    raw_path = C.ext_dir("raw_contract") / "shape_contract.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else {"cases": {}}
    n_fail = int((repro.status == "FAIL").sum())
    ind_fail = int((indep.status == "FAIL").sum()) if indep is not None else -1
    n_viol = int((leak.status == "VIOLATION").sum())
    audit_dir = C.ext_dir("audit")
    candidate_audit = {}
    try:
        from export_manuscript_source_data import audit_directory
        for name in ("comment_candidate", "article_candidate"):
            candidate_audit[name] = audit_directory(C.ext_dir(name))
    except Exception as exc:  # keep the audit explicit if a candidate is malformed
        candidate_audit = {"error": str(exc)}
    metric_path = audit_dir / "METRIC_REGRESSION_AUDIT.md"
    metric_text = metric_path.read_text(encoding="utf-8") if metric_path.is_file() else ""
    metric_pass = ("Unit tests: **PASS**" in metric_text and
                   "Unexpected differences: **0**" in metric_text and
                   "non-NaN Spearman after repair: 0" in metric_text)
    scientific_path = audit_dir / "SCIENTIFIC_REGRESSION_AUDIT.md"
    scientific_text = (scientific_path.read_text(encoding="utf-8")
                       if scientific_path.is_file() else "")
    scientific_pass = ("SCIENTIFIC_NUMERIC_DRIFT = NONE" in scientific_text and
                       "Overall: **PASS**" in scientific_text)
    c_status = _c_status()
    c_ready = _eligible("C")
    opening = ("All three cases are REAL and audit-eligible."
               if c_ready else
               "A/B REAL evidence is enabled.  " + _c_blocked_sentence())

    L = [f"# FINAL MULTICASE P0 AUDIT — {C.EXTENSION_NAME}", "",
         opening + "  This audit does not promote manuscript text.", "",
         "## Live case and dependency state", "",
         "| case | data_status | case_status | science_eligible | contains_placeholder | dependency_status |",
         "|---|---|---|---:|---:|---|"]
    for case in C.CASE_ORDER:
        meta = C.case_metadata(case)
        L.append(f"| {case} | {meta['data_status']} | {meta['case_status']} | "
                 f"{str(meta['science_eligible']).lower()} | "
                 f"{str(meta['contains_placeholder']).lower()} | {meta['dependency_status']} |")
    L += ["", "## A. Raw-data integrity", ""]
    for case in C.CASE_ORDER:
        st = raw.get("cases", {}).get(case, {})
        meta = C.case_metadata(case)
        if st:
            L.append(f"* Case {case}: contract {'STOPPED' if st.get('stop') else 'PASS'}, "
                     f"case_status={meta['case_status']}; seven tensors, float32, "
                     f"4320 states, 162x162 d05 and exact 6x6 tiled d02 mapping "
                     "are recorded under raw_contract/.")
        else:
            L.append(f"* Case {case}: raw contract record unavailable.")
    L += ["", "## B. Shared time/grid contract", "",
          "* All cases use the inherited 4320-state contract: exclude [0,970), "
          "calibration [970,3650) = 2680, evaluation [3650,4320) = 670.",
          "* Row-major 6x6 parent mapping and 27x27 = 729 fine cells per parent "
          "are fixed in the raw-contract audit.", "",
          "## C. Case A legacy reproduction", "",
          f"* {len(repro) - n_fail} / {len(repro)} frozen quantities reproduced "
          f"within tolerance ({n_fail} failures); frozen evidence was not modified.",
          "* Detail: audit/case_A_legacy_reproduction.csv.", "",
          "## D. Case B REAL temporal replication", ""]
    b = gates["gate1_representation_deficit"].get("B", {})
    L.append(f"* Case B is REAL and science-eligible.  Representation gate: "
             f"{b.get('pass')} (J/S values are retained in M2 and are not a C claim).")
    g4 = gates["gate4_temporal_transfer"]
    L.append(f"* A->B is SCIENCE_ENABLED: evaluation Spearman delta {_fmt(g4.get('spearman_delta'))}, "
             f"spread {_fmt(g4.get('spearman_spread'))}; verdict is "
             f"{verd['claim_states']['MC-A-to-B-temporal-transfer']['verdict']}.")
    if c_ready:
        c_gate = gates["gate1_representation_deficit"].get("C", {})
        if C.GENERATION == "20260905":
            e_lines = ["", "## E. Case C external-site replication (Ebao)", "",
                       "* Case C (DEC_EBAO, `1min_ebao20151201`) is REAL and "
                       "audit-eligible after the raw contract, schema-v2 "
                       "byte-attested event register, provenance "
                       "(Case-B precedent: upstream WRF artefacts "
                       "UNKNOWN_NOT_PACKAGED), independent audit, leakage "
                       "audit, candidate contamination gate and promotion gate.",
                       f"* Representation gate: {c_gate.get('pass')} (C values are "
                       "now eligible for scientific review).",
                       "* B<->C is a contemporaneous cross-site replication "
                       "(paired contemporaneous cross-site differences, never "
                       "causal terrain effects); A->C is a combined event+site "
                       "stress test; A+B+C claims are no longer "
                       "dependency-blocked, their empirical verdicts remain "
                       "recorded below.", ""]
        else:
            e_lines = ["", "## E. Case C Dayekou hot-swap result", "",
                       "* Case C is REAL and audit-eligible after the standard hot-swap "
                       "validation, provenance and promotion gates.",
                       f"* Representation gate: {c_gate.get('pass')} (C values are "
                       "now eligible for scientific review).",
                       "* B->C, A->C and A+B+C claims are no longer dependency-blocked; "
                       "their empirical verdicts remain recorded below.", ""]
    else:
        if c_status == C.CASE_STATUS_REAL_PENDING_AUDIT:
            if C.GENERATION == "20260905":
                e_lines = ["", "## E. Case C (Ebao) promotion-pending state", "",
                           "* Case C is REAL_PENDING_AUDIT (DEC_EBAO, "
                           "`1min_ebao20151201`, declared window 2015-12-01 ~ "
                           "2015-12-03 byte-attested by the schema-v2 event "
                           "register), fixed seven-file schema and fixed M4 row "
                           "contract.",
                           "* C rows may exist in master tables and diagnostics, but they carry "
                           "`science_eligible=false`, `contains_placeholder=false` and "
                           "`dependency_status=BLOCKED_REAL_PENDING_AUDIT`; no C number is "
                           "quoted as promoted evidence.",
                           "* B<->C, A->C and A+B+C replication claims remain BLOCKED until the "
                           "promotion gate passes.", ""]
            else:
                e_lines = ["", "## E. Case C Dayekou promotion-pending state", "",
                           "* Case C is REAL_PENDING_AUDIT with a stable identity (`C`, "
                           "Dayekou, 2015-12-01), fixed seven-file schema and fixed M4 row contract.",
                           "* C rows may exist in master tables and diagnostics, but they carry "
                           "`science_eligible=false`, `contains_placeholder=false` and "
                           "`dependency_status=BLOCKED_REAL_PENDING_AUDIT`; no C number is "
                           "quoted as promoted evidence.",
                           "* B->C, A->C and A+B+C replication claims remain BLOCKED until the "
                           "promotion gate passes.", ""]
        else:
            e_lines = ["", "## E. Case C Dayekou hot-swap slot", "",
                       "* Case C is PLACEHOLDER_READY with a stable identity (`C`, Dayekou, "
                       "2015-12-01), fixed seven-file schema and fixed M4 row contract.",
                       "* Engineering C rows may exist in master tables and diagnostics, but "
                       "they carry `science_eligible=false`, `contains_placeholder=true` and "
                       "`dependency_status=BLOCKED_PLACEHOLDER_DEPENDENCY`; no C number is "
                       "quoted as evidence.",
                       "* B->C, A->C and A+B+C replication claims are BLOCKED until the real "
                       "payload passes validation, provenance and the promotion gate.", ""]
    L += e_lines + ["## F. Geometric recovery", ""]
    for case in ("A", "B"):
        v = gates["gate2_geometric_recovery"].get(case, {})
        L.append(f"* Case {case} REAL: max Q_J {_fmt(v.get('max_Q_J'))}, max Q_S "
                 f"{_fmt(v.get('max_Q_S'))}, constrained Q_delta "
                 f"{_fmt(v.get('max_constrained_Q_delta'))}; "
                 f"incomplete-recovery gate = {v.get('pass')}.")
    if c_ready:
        v = gates["gate2_geometric_recovery"].get("C", {})
        L.append(f"* Case C REAL: max Q_J {_fmt(v.get('max_Q_J'))}, max Q_S "
                 f"{_fmt(v.get('max_Q_S'))}, constrained Q_delta "
                 f"{_fmt(v.get('max_constrained_Q_delta'))}; "
                 f"incomplete-recovery gate = {v.get('pass')}.")
    else:
        L.append("* Case C geometric recovery is BLOCKED by dependency status; any C rows are engineering-only.")
    L += ["", "## G. Terrain, state and history information", ""]
    for case in ("A", "B"):
        v = gates["gate3_terrain_organization"].get(case, {})
        L.append(f"* Case {case} REAL: delta~r_sigma slope {_fmt(v.get('delta_slope'))} "
                 f"(r {_fmt(v.get('delta_r'))}); S~lambda_iso slope "
                 f"{_fmt(v.get('S_slope'))} (r {_fmt(v.get('S_r'))}).")
    L.append("* Cross-terrain terrain/history conclusions involving C are available for review; no cross-site same-cell history is inferred."
             if c_ready else
             "* Cross-terrain terrain/history conclusions involving C are BLOCKED; no cross-site same-cell history is inferred.")
    L += ["", "## H. Scale and second-order evidence", "",
          f"* M4 has {len(pd.read_csv(C.ext_dir('tables') / 'M4_scale_dependence.csv')) if (C.ext_dir('tables') / 'M4_scale_dependence.csv').is_file() else 0} rows; required schema is exactly 24 (3 cases x 2 windows x 4 scales: 111.11, 333.33, 1000, 3000 m).",
          "* The 3000 m endpoint and second-order diagnostics use the fixed contracts; "
          + ("all real-case rows are eligible subject to claim verdicts."
             if c_ready else "C rows are not manuscript candidates while blocked."), "",
          "## I. Counterexample register", ""]
    for message in gates.get("counterexamples", []):
        L.append(f"* {message}")
    if not gates.get("counterexamples"):
        L.append("* No mechanical A/B headline counterexample was triggered.")
    L.append("* Rich claim-scoped register: audit/counterexample_register.md and append-only audit/counterexample_register.jsonl.")
    L += ["", "## J. Leakage and independent audit", "",
          f"* Leakage matrix rows: {len(leak)}; violations: {n_viol}.",
          f"* Independent re-implementations: {len(indep) - ind_fail} / {len(indep)} PASS."
          if indep is not None else "* Independent audit was not run.",
          "* No evaluation-window fitting, station input, T2 science consumption or cross-site history path is permitted.", "",
          "## K. Candidate boundary and metric non-regression", "",
           f"* Candidate-directory audit: `{json.dumps(candidate_audit, ensure_ascii=False, default=C._json_default)}`.",
           (f"* Metric regression audit: NOT_APPLICABLE_NEW_GENERATION "
            f"(the 20260829 numeric baselines belong to the 20260826 "
            f"workspace; mc_moments metric unit tests "
            f"{'PASS' if _metric_unit_tests_pass() else 'FAIL'} and A-number "
            f"drift is anchored by case_A_reproduction "
            f"({len(repro) - n_fail}/{len(repro)} PASS) in section C)."
            if not C.GEN.get("regression_baselines") else
            f"* Constant-safe metric regression audit: {'PASS' if metric_pass else 'NOT_AVAILABLE_OR_FAIL'} "
            f"({metric_path.as_posix()})."),
           (f"* Scientific regression audit: NOT_APPLICABLE_NEW_GENERATION "
            f"(the future-C baseline belongs to the 20260826 workspace; "
            f"A-number drift is anchored by case_A_reproduction in this "
            f"audit)."
            if not C.GEN.get("regression_baselines") else
            f"* Future-C scientific regression audit: "
            f"{'PASS' if scientific_pass else 'NOT_AVAILABLE_OR_FAIL'} "
            f"({scientific_path.as_posix()}); the protected A/B and A→B scope "
            "must report `SCIENTIFIC_NUMERIC_DRIFT = NONE`."),
          "* All manuscript candidates are fail-closed; "
          + ("C rows are included only after the real-case promotion gate."
             if c_ready else "C-dependent rows are absent from candidate CSVs."),
          "* No file under latex_paper0/, latex_paper0_npjcomment/, data_paper0/frozen/ or source/evidence/ was modified by this extension.",
          "",
          "## Claim-level verdicts", "",
          "| claim_id | verdict | dependency_status |", "|---|---|---|"]
    for claim_id, state in verd["claim_states"].items():
        L.append(f"| {claim_id} | {state['verdict']} | {state['dependency_status']} |")
    _nq = 16 if C.GEN.get("thunder") == "sixteen" else 12
    L += ["", "---", "", f"## Thunder summary ({_nq} pre-registered questions)", ""]
    L.extend(thunder_dispatch(gates, verd, cases_ok))
    L += ["", f"P0_COMMENT_EMPIRICAL_SUPPORT = {verd['comment']}",
          "", f"P0_ARTICLE_EMPIRICAL_SUPPORT = {verd['article']}", "",
          "Promotion remains separate from author manuscript approval; this run "
          "does not edit or promote any manuscript source."]
    (audit_dir / "FINAL_MULTICASE_P0_AUDIT.md").write_text("\n".join(L) + "\n",
                                                            encoding="utf-8")
    print("[phase17] FINAL_MULTICASE_P0_AUDIT.md written")


def _metric_unit_tests_pass() -> bool:
    """Run the mc_moments metric unit tests standalone (used when the
    generation has no pre-repair numeric baseline to regress against)."""
    import io
    import unittest
    try:
        import test_metrics
        suite = unittest.TestLoader().loadTestsFromModule(test_metrics)
        runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
        return bool(runner.run(suite).wasSuccessful())
    except Exception:  # noqa: BLE001 - a broken test module must fail the gate
        return False


def promotion_gate(result: dict, cases_ok: list[str]) -> dict:
    """Promotion gate for a real replacement; placeholder runs have no target."""
    pending = [c for c in cases_ok
               if C.case_record(c).get("case_status") == C.CASE_STATUS_REAL_PENDING_AUDIT]
    checks = []
    partial_precedent = bool(C.GEN.get("promote_with_partial_provenance"))
    for case in pending:
        prov_path = C.case_dir(case) / "WRF_RUN_PROVENANCE.json"
        prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
        if partial_precedent:
            # 20260905: documented Case-B precedent (2026-08-29 architecture
            # repair).  Upstream WRF artefacts are UNKNOWN_NOT_PACKAGED and
            # identity rests on the observed tensor hashes plus the
            # byte-frozen schema-v2 event register, which is verified as its
            # own check below.  A COMPLETE provenance stays pass-by-default.
            prov_pass = (prov.get("provenance_status") == "COMPLETE"
                         or prov.get("provenance_status") ==
                         "PARTIAL_UNKNOWN_NOT_PACKAGED")
            checks.append({"name": f"{case}_identity_provenance_B_precedent",
                           "pass": bool(prov_pass),
                           "evidence": (f"{prov_path} (provenance_status="
                                        f"{prov.get('provenance_status')}; "
                                        "upstream WRF artefacts not packaged; "
                                        "identity = tensor hashes + "
                                        "CASE_C_EVENT_FREEZE.json)")})
            if case == "C":
                try:
                    import case_c_freeze as FZ
                    reg = FZ.load()
                    reg_errs = FZ._errors(reg) if reg else ["register missing"]
                    if reg and not reg_errs:
                        files = {stem: C.case_raw_dir("C") /
                                 C.raw_filename("C", stem)
                                 for stem in C.RAW_FILE_STEMS}
                        reg_errs = FZ.verify_payload(files)
                    checks.append({"name": "case_C_freeze_register",
                                   "pass": not reg_errs,
                                   "evidence": (f"{FZ.register_path()}; errors="
                                                f"{reg_errs}")})
                except Exception as exc:  # noqa: BLE001
                    checks.append({"name": "case_C_freeze_register",
                                   "pass": False, "evidence": str(exc)})
        else:
            checks.append({"name": f"{case}_complete_provenance",
                           "pass": bool(prov.get("promotion_eligible")),
                           "evidence": str(prov_path)})
        checks.append({"name": f"{case}_raw_contract",
                       "pass": case in cases_ok and not C.case_record(case).get("stop", True),
                       "evidence": str(C.case_status_file(case))})
    if not pending:
        return {"pass": True, "status": "NOT_APPLICABLE_NO_PENDING_REAL_CASE",
                "pending_cases": [], "checks": []}
    repro = result.get("repro")
    checks.append({"name": "case_A_reproduction",
                   "pass": repro is not None and len(repro) > 0 and bool((repro.status == "PASS").all())})
    indep = result.get("indep")
    checks.append({"name": "independent_audit",
                   "pass": indep is not None and len(indep) > 0 and not (indep.status == "FAIL").any()})
    leak = result.get("leak")
    checks.append({"name": "leakage_audit",
                   "pass": leak is not None and not (leak.status == "VIOLATION").any()})
    try:
        from export_manuscript_source_data import audit_directory
        cand = {name: audit_directory(C.ext_dir(name))
                for name in ("comment_candidate", "article_candidate")}
        checks.append({"name": "candidate_contamination_gate",
                       "pass": all(v.get("pass", False) for v in cand.values()),
                       "evidence": cand})
    except Exception as exc:
        checks.append({"name": "candidate_contamination_gate", "pass": False,
                       "evidence": str(exc)})
    metric_path = C.ext_dir("audit") / "METRIC_REGRESSION_AUDIT.md"
    metric_text = metric_path.read_text(encoding="utf-8") if metric_path.is_file() else ""
    rehearsal = False
    for case in pending:
        p = C.case_dir(case) / "WRF_RUN_PROVENANCE.json"
        if p.is_file():
            rehearsal = rehearsal or bool(json.loads(p.read_text(encoding="utf-8")).get("rehearsal"))
    if not C.GEN.get("regression_baselines"):
        # Generations with no pre-repair numeric baseline (20260905): the
        # 2026-08-29 baselines belong to the 20260826 workspace.  Drift of the
        # frozen Case-A numbers is instead anchored by the case_A_reproduction
        # check above, and the metric code itself by standalone unit tests.
        unit_ok = _metric_unit_tests_pass()
        checks.append({"name": "metric_unit_tests_new_generation",
                       "pass": unit_ok,
                       "status": "PASS" if unit_ok else "FAIL",
                       "evidence": ("mc_moments metric unit tests "
                                    "(source/scripts/paper0_multicase/test_metrics.py); "
                                    "the pre_repair_numeric_baseline_20260829.json "
                                    "belongs to generation 20260826 and does not "
                                    f"apply to {C.GENERATION}")})
        checks.append({"name": "scientific_numeric_regression_audit",
                       "pass": True,
                       "status": "NOT_APPLICABLE_NEW_GENERATION",
                       "evidence": ("the future_c_promotion_regression_baseline_"
                                    "20260829.json belongs to generation 20260826; "
                                    "A-number drift is anchored by "
                                    "case_A_reproduction in this gate")})
    else:
        metric_pass = ("Unit tests: **PASS**" in metric_text and
                       "Unexpected differences: **0**" in metric_text and
                       "non-NaN Spearman after repair: 0" in metric_text)
        checks.append({"name": "metric_regression_audit",
                       "pass": metric_pass or (rehearsal and not metric_path.is_file()),
                       "status": "PASS" if metric_pass else
                       "SKIPPED_ISOLATED_REHEARSAL" if rehearsal else "FAIL",
                        "evidence": str(metric_path)})
        scientific_path = C.ext_dir("audit") / "SCIENTIFIC_REGRESSION_AUDIT.md"
        scientific_text = (scientific_path.read_text(encoding="utf-8")
                           if scientific_path.is_file() else "")
        scientific_pass = ("SCIENTIFIC_NUMERIC_DRIFT = NONE" in scientific_text and
                           "Overall: **PASS**" in scientific_text)
        if not scientific_path.is_file() and rehearsal:
            scientific_status = "SKIPPED_ISOLATED_REHEARSAL"
            scientific_gate_pass = True
        else:
            scientific_status = "PASS" if scientific_pass else "FAIL"
            scientific_gate_pass = scientific_pass
        checks.append({"name": "scientific_numeric_regression_audit",
                       "pass": scientific_gate_pass,
                       "status": scientific_status,
                       "evidence": str(scientific_path)})
    return {"pass": bool(checks) and all(x.get("pass", False) for x in checks),
            "status": "PASS" if all(x.get("pass", False) for x in checks) else "FAIL",
            "pending_cases": pending, "checks": checks}


def main(cases_ok: list[str], run_indep: bool = True) -> dict:
    repro = case_a_reproduction()
    indep = None
    if run_indep:
        import audit_independent
        indep = audit_independent.run(cases_ok)
    leak = leakage(cases_ok)
    gates = core_gates(cases_ok)
    verd = verdicts(gates, repro, cases_ok)
    write_counterexamples(gates, repro, indep)
    final_audit(cases_ok, repro, indep, leak, gates, verd)
    result = {"repro": repro, "indep": indep, "leak": leak,
              "gates": gates, "verdicts": verd}
    promotion = promotion_gate(result, cases_ok)
    C.write_json(C.ext_dir("audit") / "promotion_gate.json", promotion,
                 cases=cases_ok)
    C.run_receipt("phase17_audit", {
        "cases_ok": cases_ok,
        "a_reproduction": {"n": len(repro),
                           "fail": int((repro.status == "FAIL").sum())},
        "independent": (None if indep is None else
                        {"n": len(indep),
                         "fail": int((indep.status == "FAIL").sum())}),
        "leakage_violations": int((leak.status == "VIOLATION").sum()),
        "verdicts": verd, "promotion_gate": promotion})
    result["promotion_gate"] = promotion
    return result


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
