"""Phase 16 — cross-case core-loss assembly (PLAN §7), master tables M1–M12
(PLAN §24) and the comment/article candidate subsets (PLAN §25–26).

Cross-case cell-identity rule: A and B share the Yakou grid, so cellwise
pattern correlations are meaningful; C is different geography, so C rows are
compared at the distribution level only (never cell-by-cell against A/B).
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
from export_manuscript_source_data import (  # noqa: E402
    assert_manuscript_eligible,
    eligible_subset,
)

MOMENTS = ["delta_dyn", "J", "delta_scalar", "S"]
WINDOWS = {"calibration": C.CAL_SLICE, "evaluation": C.EVAL_SLICE}


def core_loss(cases: list[str]) -> None:
    print("[phase16] cross-case core-loss tables")
    cross = C.ext_dir("cross_case")
    summ_rows, cell_rows, eff_rows = [], [], []
    cellmeans = {}
    for case in cases:
        tab = P1.load_moments(case)
        for wname, (a, b) in WINDOWS.items():
            cm = {m: tab[m][a:b].mean(axis=0) for m in MOMENTS}
            cellmeans[(case, wname)] = cm
            dm = {m: float(cm[m].mean()) for m in MOMENTS}
            summ_rows.append({
                "case": case, "label": C.CASES[case]["label"], "window": wname,
                **{f"{m}_domain_mean": dm[m] for m in MOMENTS},
                "J_over_delta_scalar": (dm["J"] / dm["delta_scalar"]
                                        if abs(dm["delta_scalar"]) > 1e-12 else np.nan),
                "J_over_abs_sum": dm["J"] / (abs(dm["delta_dyn"]) + dm["J"] + 1e-30),
                **{f"{m}_cell_min": float(cm[m].min()) for m in MOMENTS},
                **{f"{m}_cell_max": float(cm[m].max()) for m in MOMENTS},
                **{f"{m}_cell_sd": float(cm[m].std(ddof=0)) for m in MOMENTS},
            })
            for cid in range(36):
                cell_rows.append({"case": case, "window": wname, "cell_id": cid,
                                  **{m: cm[m][cid] for m in MOMENTS}})
    C.write_df(cross / "core_loss_summary.csv", pd.DataFrame(summ_rows).round(9))
    C.write_df(cross / "core_loss_cell_means.csv", pd.DataFrame(cell_rows).round(9))

    if C.GEN.get("bc_synchronized_bootstrap"):
        pair_bc = ("B", "C",
                   "same chronological/synoptic window, different site and "
                   "terrain regime, potentially different local wind "
                   "realization (contemporaneous cross-site contrast; no "
                   "terrain-only attribution)", False)
        pair_ac = ("A", "C",
                   "event + site + terrain regime all changed (combined "
                   "external stress test; total-difference reference only)",
                   False)
    else:
        pair_bc = ("B", "C", "joint event+site shift (independent Case-C window; "
                             "no terrain-only attribution)", False)
        pair_ac = ("A", "C", "event + terrain both changed (total-difference reference only)",
                   False)
    pairs = [("A", "B", "same terrain, event/state changed", True),
             pair_bc, pair_ac]
    for c1, c2, meaning, cells_match in pairs:
        if c1 not in cases or c2 not in cases:
            continue
        for wname in WINDOWS:
            m1 = cellmeans[(c1, wname)]
            m2 = cellmeans[(c2, wname)]
            row = {"pair": f"{c1}_vs_{c2}", "window": wname, "comparison": meaning,
                   "cells_identity_match": cells_match,
                   "source_case": c1, "target_case": c2,
                   **C.status_metadata([c1, c2])}
            for m in MOMENTS:
                d1, d2 = m1[m], m2[m]
                row[f"{m}_domain_mean_{c1}"] = float(d1.mean())
                row[f"{m}_domain_mean_{c2}"] = float(d2.mean())
                row[f"{m}_domain_diff"] = float(d2.mean() - d1.mean())
                row[f"{m}_domain_ratio"] = (float(d2.mean() / d1.mean())
                                            if abs(d1.mean()) > 1e-12 else np.nan)
                pooled = np.sqrt((d1.std(ddof=0) ** 2 + d2.std(ddof=0) ** 2) / 2.0)
                row[f"{m}_cohens_d"] = (float((d2.mean() - d1.mean()) / pooled)
                                        if pooled > 1e-12 else np.nan)
                if cells_match:
                    row[f"{m}_cellwise_pearson"] = M.pearson(d1, d2)
                    row[f"{m}_cellwise_spearman"] = M.spearman(d1, d2)
                else:
                    row[f"{m}_cellwise_pearson"] = np.nan
                    row[f"{m}_cellwise_spearman"] = np.nan
            eff_rows.append(row)
    C.write_df(cross / "core_loss_effect_sizes.csv", pd.DataFrame(eff_rows).round(9))


def master_tables(cases: list[str]) -> None:
    print("[phase16] master tables M1-M12")
    tdir = C.ext_dir("tables")
    cross = C.ext_dir("cross_case")

    def scales_match(values) -> bool:
        return (len(values) == len(expected_scales) and
                np.allclose(sorted(float(x) for x in values),
                            sorted(expected_scales), rtol=0.0, atol=1e-6))

    # ---- M1 case metadata ----
    rows = []
    for case in cases:
        meta = C.CASES[case]
        ts = json.loads((C.case_dir(case) / "terrain_summary.json")
                        .read_text(encoding="utf-8"))
        ws = json.loads((C.case_dir(case) / "wind_state_summary.json")
                        .read_text(encoding="utf-8"))
        rows.append({
            "case": case, "label": meta["label"], "location": meta["location"],
            "lat": meta["lat"], "lon": meta["lon"], "dates": meta["dates"],
            "role": meta["role"], "raw_n": C.RAW_N,
            "retained_n": C.RETAINED_SLICE[1] - C.RETAINED_SLICE[0],
            "calibration_n": C.CAL_SLICE[1] - C.CAL_SLICE[0],
            "evaluation_n": C.EVAL_SLICE[1] - C.EVAL_SLICE[0],
            "mean_elevation_m": ts["domain"]["mean_elevation"],
            "elevation_range_m": ts["domain"]["elevation_range"],
            "terrain_sd_m": ts["domain"]["terrain_sd"],
            "r_sigma_min": ts["descriptor_distributions"]["r_sigma"]["min"],
            "r_sigma_max": ts["descriptor_distributions"]["r_sigma"]["max"],
            "lambda_iso_min": ts["descriptor_distributions"]["lambda_iso"]["min"],
            "lambda_iso_max": ts["descriptor_distributions"]["lambda_iso"]["max"],
            "cal_speed_mean": ws["calibration"]["speed_mean"],
            "eval_speed_mean": ws["evaluation"]["speed_mean"],
            "cal_dominant_sector": ws["calibration"]["dominant_sector_label"],
            "eval_dominant_sector": ws["evaluation"]["dominant_sector_label"],
        })
    C.write_df(tdir / "M1_case_metadata.csv", pd.DataFrame(rows).round(6))

    # ---- M2 representation loss ----
    summ = pd.read_csv(cross / "core_loss_summary.csv")
    keep = ["case", "label", "window"] + \
        [c for c in summ.columns if c.endswith("_domain_mean")] + \
        ["J_over_delta_scalar", "data_status"]
    C.write_df(tdir / "M2_representation_loss.csv", summ[keep])

    # ---- M3 second-order diagnostic ----
    rows = []
    for case in cases:
        grp = pd.read_csv(C.case_dir(case) / "jensen_second_order_groups.csv")
        for wname in WINDOWS:
            g_all = grp[(grp.window == wname) & (grp.wind_group == "all")].iloc[0]
            g_lo = grp[(grp.window == wname) & (grp.wind_group == "below")].iloc[0]
            rows.append({
                "case": case, "window": wname,
                "J_domain_mean": g_all.get("domain_mean_J"),
                "J2nd_domain_mean": g_all.get("domain_mean_J2nd"),
                "rmse": g_all["rmse"], "bias": g_all["bias"], "r": g_all["corr"],
                "cellmean_corr": g_all.get("cellmean_corr"),
                "crosswind_share_of_trC": g_all.get("crosswind_share_of_trC"),
                "low_speed_record_fraction": g_lo["record_fraction"],
                "low_speed_rmse": g_lo["rmse"],
                "low_speed_threshold_ms": C.LOW_WIND_MS,
            })
    C.write_df(tdir / "M3_second_order.csv", pd.DataFrame(rows).round(8))

    # ---- M4 scale dependence ----
    frames = []
    for case in cases:
        s = pd.read_csv(C.case_dir(case) / "scale_summary.csv")
        rp = pd.read_csv(C.case_dir(case) / "scale_rank_persistence.csv")
        expected_scales = {float(C.SCALE_M[k]) for k in C.SCALE_CELLS}
        got = set(s["scale_m"].astype(float))
        assert len(s) == len(WINDOWS) * len(C.SCALE_CELLS), (
            f"case {case}: scale_summary has {len(s)} rows; expected "
            f"{len(WINDOWS) * len(C.SCALE_CELLS)}")
        assert scales_match(got), (
            f"case {case}: scale_summary scales {sorted(got)} != "
            f"{sorted(expected_scales)}")
        # CSV round-tripping makes 333.333333 and 333.33333333 distinct
        # binary floats; join on the frozen physical scale rounded to 1e-6 m.
        s["_scale_key"] = s["scale_m"].astype(float).round(6)
        rp = rp.copy()
        rp["_scale_key"] = rp["scale_m"].astype(float).round(6)
        m = s.merge(rp[["window", "_scale_key", "spearman_J_vs_111",
                        "spearman_S_vs_111"]], on=["window", "_scale_key"])
        m = m.drop(columns=["_scale_key"])
        assert len(m) == len(WINDOWS) * len(C.SCALE_CELLS)
        m.insert(0, "case", case)
        frames.append(m)
    m4 = pd.concat(frames, ignore_index=True)
    expected_rows = len(cases) * len(WINDOWS) * len(C.SCALE_CELLS)
    assert len(m4) == expected_rows, (
        f"M4 row count {len(m4)} != {expected_rows} "
        "(cases x windows x scales)")
    for (case, window), group in m4.groupby(["case", "window"], sort=False):
        actual = set(group["scale_m"].astype(float))
        assert scales_match(actual), (
            f"M4 {case}/{window}: scales {sorted(actual)} != "
            f"{sorted(expected_scales)}")
    C.write_df(tdir / "M4_scale_dependence.csv", m4)

    # ---- M5 geometric recovery ----
    frames = []
    for case in cases:
        met = pd.read_csv(C.case_dir(case) / "interpolation_metrics.csv")
        rec = pd.read_csv(C.case_dir(case) / "interpolation_recovery.csv")
        piv = rec[rec.cell_set == "all36"].pivot(index="model", columns="moment",
                                                 values="recovery_ratio")
        piv.columns = [f"Q_{m}" for m in piv.columns]
        met = met.merge(piv, left_on="model", right_index=True, how="left")
        met.insert(0, "case", case)
        frames.append(met)
    C.write_df(tdir / "M5_geometric_recovery.csv",
               pd.concat(frames, ignore_index=True))

    # ---- M6 terrain relationships ----
    frames = []
    for case in cases:
        tr = pd.read_csv(C.case_dir(case) / "terrain_relationships.csv")
        tr.insert(0, "case", case)
        frames.append(tr)
    C.write_df(tdir / "M6_terrain_relationships.csv",
               pd.concat(frames, ignore_index=True))

    # ---- M7 information-source ranking ----
    rank = pd.read_csv(cross / "information_source_ranking.csv")
    C.write_df(tdir / "M7_information_source_ranking.csv", rank)

    # ---- M8/M9/M10 transfers ----
    t1 = pd.read_csv(cross / "transfer_T1_A_to_B.csv")
    t2 = pd.read_csv(cross / "transfer_T2_history_A_to_B.csv")
    C.write_df(tdir / "M8_transfer_A_to_B.csv",
               pd.concat([t1, t2], ignore_index=True))
    C.write_df(tdir / "M9_transfer_B_to_C.csv",
               pd.read_csv(cross / "transfer_T3_B_to_C.csv"))
    C.write_df(tdir / "M10_transfer_A_to_C.csv",
               pd.read_csv(cross / "transfer_T4_A_to_C.csv"))
    # M11 / M12 are written by phases 15 / 14.


def candidates(cases: list[str]) -> None:
    """Export only claim-eligible rows into manuscript candidate directories.

    The engineering master tables retain C placeholder rows.  This exporter
    is the hard boundary: C-dependent rows are omitted, and malformed or
    unresolved rows raise instead of being copied into a manuscript source
    table.
    """
    print("[phase16] comment/article candidate subsets")
    tdir = C.ext_dir("tables")
    cdir = C.ext_dir("comment_candidate")
    adir = C.ext_dir("article_candidate")
    cdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    for path in list(cdir.glob("*.csv")) + list(adir.glob("M*.csv")):
        # These are generated candidate artefacts, not frozen evidence.  Only
        # the exact stale candidate files are removed before a fresh export.
        # Removal is best-effort: sandboxed trash backends may refuse, and
        # every canonical name is overwritten below anyway; the candidate
        # manifests record the live exported set.
        try:
            path.unlink()
        except OSError as exc:
            print(f"[phase16] note: stale candidate {path.name} not removed "
                  f"({exc}); it will be overwritten")

    def export(df: pd.DataFrame, path: Path, artifact: str) -> bool:
        out = eligible_subset(df, artifact)
        if out.empty:
            return False
        out.to_csv(path, index=False, lineterminator="\n")
        assert_manuscript_eligible(pd.read_csv(path), artifact)
        return True

    status = C.combined_status(*cases)
    dep = C.status_metadata(cases)
    c_meta = C.case_metadata("C")
    c_status = c_meta["case_status"]
    c_ready = C.case_is_science_eligible(c_meta)
    if c_status == C.CASE_STATUS_PLACEHOLDER:
        banner = ("A/B REAL rows exported; C-dependent rows are blocked because "
                  "Case C is PLACEHOLDER (engineering dry-run only)")
    elif c_status == C.CASE_STATUS_REAL_PENDING_AUDIT:
        banner = ("A/B REAL rows exported; C-dependent rows are blocked because "
                  "Case C is REAL_PENDING_AUDIT")
    elif c_ready:
        banner = "all exported rows are REAL, including promoted Case C"
    else:
        banner = f"candidate export is limited by Case C status {c_status}"
    exported = []
    blocked = []

    # C1 core loss
    p = cdir / "C1_core_loss.csv"
    if export(pd.read_csv(tdir / "M2_representation_loss.csv"), p,
              "C1_core_loss"):
        exported.append(p.name)
    else:
        blocked.append("M2_representation_loss.csv")
    # C2 geometric recovery incl. constrained sensitivity
    p = cdir / "C2_geometric_recovery.csv"
    if export(pd.read_csv(tdir / "M5_geometric_recovery.csv"), p,
              "C2_geometric_recovery"):
        exported.append(p.name)
    else:
        blocked.append("M5_geometric_recovery.csv")
    # C3 temporal transfer + joint event+site shift transfer.  The shift rows remain blocked while
    # C is placeholder/pending; after real C promotion they are eligible and
    # are included beside the A->B rows.
    m8 = pd.read_csv(tdir / "M8_transfer_A_to_B.csv")
    m9 = pd.read_csv(tdir / "M9_transfer_B_to_C.csv")
    p = cdir / "C3_transfers.csv"
    c3_source = pd.concat([m8, m9], ignore_index=True) if c_ready else m8
    if export(c3_source, p, "C3_transfers_A_to_B_and_B_to_C"):
        exported.append(p.name)
    else:
        blocked.append("M8_transfer_A_to_B.csv")
    if c_ready:
        assert not eligible_subset(m9, "M9_transfer_B_to_C").empty, (
            "real C was promoted but M9 has no eligible rows")
    else:
        # M9 is deliberately inspected for the gate but is not merged into a
        # source-data candidate while its B->C dependency is unresolved.
        if not eligible_subset(m9, "M9_transfer_B_to_C").empty:
            raise RuntimeError("M9_transfer_B_to_C unexpectedly became eligible without C")
        blocked.append("M9_transfer_B_to_C.csv")
    # C4 information-source capability
    p = cdir / "C4_information_sources.csv"
    if export(pd.read_csv(tdir / "M7_information_source_ranking.csv"), p,
              "C4_information_sources"):
        exported.append(p.name)
    else:
        blocked.append("M7_information_source_ranking.csv")
    bc_blocked_note = ("B->C is blocked until real Dayekou input"
                       if C.GENERATION == "20260826" else
                       "B->C is blocked until real "
                       f"{C.CASES['C']['location']} promotion")
    (cdir / "README.md").write_text(f"""# comment_candidate — future Comment evidence subset

**{banner}**

Pre-registered subset (PLAN §25): C1 representation-loss survival on the
eligible evidence ({'A/B/C' if c_ready else 'A/B; C is an engineering-only blocked slot'}), C2 incomplete geometric refinement
(bilinear/bicubic Q_J/Q_S + parent-vector-constrained sensitivity),
C3 A->B temporal transfer ({'B->C is available for review' if c_ready else bc_blocked_note}), C4
information-source capability (geometry / terrain / history).

Deliberately excluded (PLAN §25): scale law, candidate screening, the J2
figure.  Nothing here modifies the Comment; promotion requires the human
"MULTICASE AUDIT PASSED — PROMOTE" instruction.

data_status: {status}
candidate_rows: {', '.join(exported) if exported else 'none'}
blocked_rows: {', '.join(blocked) if blocked else 'none'}
placeholder_policy: {'C-dependent rows are exported only after C is REAL and promoted.' if not c_ready else 'C-dependent rows are exported because C is REAL and promoted.'}
""", encoding="utf-8")

    # Article superset: only eligible rows from all master tables.  C rows are
    # absent while blocked and appear only after a real-case promotion.
    for f in sorted(tdir.glob("M*.csv")):
        out_path = adir / f.name
        if export(pd.read_csv(f), out_path, f"article_candidate/{f.name}"):
            continue
        blocked.append(f.name)
        if out_path.is_file():
            out_path.unlink()
    (adir / "README.md").write_text(f"""# article_candidate — future Full-Article evidence superset

**{banner}**

Contains the Comment subset plus second-order J2, scale dependence, frozen
scale-aware transfer, terrain-descriptor replication, state stratification,
information-source ranking and full robustness (master tables M1–M12; the
per-case detail lives in ../case_X/).  The manuscript itself is NOT touched
(PLAN §26); promotion requires the human "MULTICASE AUDIT PASSED — PROMOTE"
instruction.

data_status: {status}
placeholder_policy: only rows with science_eligible=true,
contains_placeholder=false, and dependency_status=REAL are present.
""", encoding="utf-8")
    comment_spec = C.CANDIDATE_SPECS["comment_candidate"]
    article_spec = C.CANDIDATE_SPECS["article_candidate"]
    expected_real_cases = [c for c in cases if C.case_is_science_eligible(c)]
    C.write_json(cdir / "candidate_manifest.json", {
        "artifact_kind": "manuscript_source_data_gate",
        "selection_scope": comment_spec["selection_scope"],
        "expected_real_case_ids": expected_real_cases,
        "exported_files": exported,
        "blocked_source_files": sorted(set(blocked)),
        "source_tables": "engineering master tables retain all cases",
        "rule": "candidate exporter fails closed on placeholder/unresolved rows and checks registry metadata",
    }, cases=cases)
    C.write_json(adir / "candidate_manifest.json", {
        "artifact_kind": "manuscript_source_data_gate",
        "selection_scope": article_spec["selection_scope"],
        "expected_real_case_ids": expected_real_cases,
        "exported_files": sorted(p.name for p in adir.glob("M*.csv")),
        "blocked_source_files": sorted(set(blocked)),
        "rule": "candidate exporter fails closed on placeholder/unresolved rows and checks registry metadata",
    }, cases=cases)


def figure_data(cases: list[str]) -> None:
    """Compact CSVs backing the diagnostic figures (figure_data/)."""
    fdir = C.ext_dir("figure_data")
    frames = []
    for case in cases:
        cm = pd.read_csv(C.case_dir(case) / "moments_cell_means_calibration.csv")
        cm.insert(0, "case", case)
        cm.insert(1, "window", "calibration")
        ce = pd.read_csv(C.case_dir(case) / "moments_cell_means_evaluation.csv")
        ce.insert(0, "case", case)
        ce.insert(1, "window", "evaluation")
        frames += [cm, ce]
    C.write_df(fdir / "diag_cell_means_all_cases.csv",
               pd.concat(frames, ignore_index=True))
    frames = []
    for case in cases:
        d = pd.read_csv(C.case_dir(case) / "terrain_descriptors.csv")
        d.insert(0, "case", case)
        frames.append(d)
    C.write_df(fdir / "diag_terrain_descriptors_all_cases.csv",
               pd.concat(frames, ignore_index=True))


def main(cases: list[str]) -> None:
    core_loss(cases)
    master_tables(cases)
    candidates(cases)
    figure_data(cases)
    C.run_receipt("phase16_tables", {"cases": cases})


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
