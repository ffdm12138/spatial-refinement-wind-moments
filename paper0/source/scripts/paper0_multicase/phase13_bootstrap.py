"""Phase 13 — bootstrap design (PLAN §19).

Within-case paired moving-block bootstrap on the 670 evaluation states:
  * master seed 20260826 (per-case spawn keys), 2000 replicates,
    primary block 120 min, sensitivity 60 / 240 min;
  * ALL model pairs consume identical resampled time indices within a
    replicate (time-varying bilinear/bicubic block means are recomputed per
    replicate; terrain / global / same-cell history remain static);
  * headline CIs for J, S, delta_dyn, delta_scalar domain means and the
    Q_J / Q_S recovery ratios ride the same replicates.

B/C synchronized cross-site paired moving-block bootstrap:
  * generation 20260826: RETIRED as of 2026-08-30
    (PLAN_AMENDMENT_CASE_C_20260830.md section 7) -- with Case C as an
    independent event window, identical ordinal indices carry no paired
    meaning; ``paired_bc`` stays retired and reproducible-only.
  * generation 20260905: RE-ENABLED under PLAN_MULTICASE_20260905.md §19.2,
    because Case C (Ebao 2015-12-01) shares the declared chronological/
    synoptic window with Case B.  Preconditions are machine-checked in
    ``phase00_raw_contract.time_alignment_audit`` (same declared window,
    same 4320-state axis, same cadence contract).  Output is renamed to
    ``B_C_synchronized_case_differences.csv`` and every difference must be
    reported as a *paired contemporaneous cross-site difference* -- it
    removes part of the shared time-window variability; it is NOT a causal
    terrain effect and never evidence of identical atmospheric states.

A code-path validation reruns the legacy Case-A design (seed 20260807,
frozen sealed predictions + frozen interpolation evidence) and compares
against the frozen interpolation_bootstrap.csv.
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
import phase12_infosources as P12  # noqa: E402

H = C.PUBLIC_HISTORY_NAME
PAIRS = [
    ("terrain_vs_global", "scalar", "terrain_roughness_equation", "global_calibration_mean"),
    ("terrain_vs_bilinear", "scalar", "terrain_roughness_equation", "bilinear"),
    ("terrain_vs_bicubic", "scalar", "terrain_roughness_equation", "bicubic"),
    ("terrain_vs_history", "scalar", "terrain_roughness_equation", H),
    ("joint_vs_global", "spread", "joint_terrain_equation", "global_calibration_mean"),
    ("joint_vs_bilinear", "spread", "joint_terrain_equation", "bilinear"),
    ("joint_vs_bicubic", "spread", "joint_terrain_equation", "bicubic"),
    ("joint_vs_history", "spread", "joint_terrain_equation", H),
]


def _case_arrays(case: str) -> dict:
    P = P12.build_predictions(case)
    tab = P1.load_moments(case)
    ea, eb = C.EVAL_SLICE
    return {
        "pred": P["pred"],
        "obs": {"scalar": P["obs"]["scalar"], "spread": P["obs"]["spread"]},
        "J": tab["J"][ea:eb], "S": tab["S"][ea:eb],
        "dyn": tab["delta_dyn"][ea:eb], "dsc": tab["delta_scalar"][ea:eb],
    }


def _interp_J_S(case: str) -> dict:
    import phase05_interpolation as P5
    ea, eb = C.EVAL_SLICE
    out = {}
    with np.load(P5.interp_npz_path(case)) as z:
        for m in ("bilinear", "bicubic"):
            out[("J", m)] = z[f"{m}_J"][ea:eb]
            out[("S", m)] = z[f"{m}_spread"][ea:eb]
    return out


def within_case(case: str) -> None:
    print(f"[phase13] case {case}: within-case paired bootstrap")
    arr = _case_arrays(case)
    interp = _interp_J_S(case)
    pred, obs = arr["pred"], arr["obs"]
    n = obs["scalar"].shape[0]
    assert n == C.EVAL_SLICE[1] - C.EVAL_SLICE[0]
    rng = C.rng_for(f"bootstrap_within_{case}")

    pair_rows, head_rows = [], []
    for length in C.BLOCK_LENGTHS:
        d = {lab: np.zeros(C.N_REPLICATES) for lab, *_ in PAIRS}
        ra_ = {lab: np.zeros(C.N_REPLICATES) for lab, *_ in PAIRS}
        rb_ = {lab: np.zeros(C.N_REPLICATES) for lab, *_ in PAIRS}
        heads = {k: np.zeros(C.N_REPLICATES) for k in
                 ["J", "S", "delta_dyn", "delta_scalar",
                  "Q_J_bilinear", "Q_J_bicubic", "Q_S_bilinear", "Q_S_bicubic"]}
        for rep in range(C.N_REPLICATES):
            idx = M.block_resample(rng, n, length)   # identical for everything
            om = {g: obs[g][idx].mean(axis=0) for g in ("scalar", "spread")}
            pm = {}
            for (g, m), p in pred.items():
                pm[(g, m)] = p[idx].mean(axis=0) if p.ndim == 2 else p
            for lab, g, ma, mb in PAIRS:
                ra = M.cell_rmse(om[g], pm[(g, ma)])
                rb = M.cell_rmse(om[g], pm[(g, mb)])
                d[lab][rep] = ra - rb
                ra_[lab][rep] = ra
                rb_[lab][rep] = rb
            jm = arr["J"][idx].mean()
            sm = arr["S"][idx].mean()
            heads["J"][rep] = jm
            heads["S"][rep] = sm
            heads["delta_dyn"][rep] = arr["dyn"][idx].mean()
            heads["delta_scalar"][rep] = arr["dsc"][idx].mean()
            for m in ("bilinear", "bicubic"):
                heads[f"Q_J_{m}"][rep] = interp[("J", m)][idx].mean() / jm
                heads[f"Q_S_{m}"][rep] = interp[("S", m)][idx].mean() / sm
        for lab, g, ma, mb in PAIRS:
            lo, hi = np.percentile(d[lab], [2.5, 97.5])
            pair_rows.append({
                "case": case, "pair": lab,
                "group": "scalar_increment" if g == "scalar" else "spread",
                "model_a": ma, "model_b": mb, "block_minutes": length,
                "replicates": C.N_REPLICATES,
                "dRMSE_est": float(d[lab].mean()),
                "dRMSE_median": float(np.median(d[lab])),
                "ci_low": float(lo), "ci_high": float(hi),
                "p_dRMSE_negative": float((d[lab] < 0).mean()),
                "rmse_a_mean": float(ra_[lab].mean()),
                "rmse_b_mean": float(rb_[lab].mean()),
            })
        for k, v in heads.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            head_rows.append({
                "case": case, "quantity": k, "block_minutes": length,
                "replicates": C.N_REPLICATES,
                "estimate": float(v.mean()), "median": float(np.median(v)),
                "ci_low": float(lo), "ci_high": float(hi),
                "note": ("time-sampling uncertainty conditional on this domain; "
                         "not geographic uncertainty"),
            })
    bdir = C.ext_dir("bootstrap")
    C.write_df(bdir / f"case_{case}_paired_bootstrap.csv",
               pd.DataFrame(pair_rows).round(9), case=case)
    C.write_df(bdir / f"case_{case}_headline_ci.csv",
               pd.DataFrame(head_rows).round(9), case=case)
    C.run_receipt(f"phase13_bootstrap_case_{case}", {
        "case": case, "seed_spawn": list(C.SEED_SPAWN[f"bootstrap_within_{case}"]),
        "replicates": C.N_REPLICATES, "blocks": list(C.BLOCK_LENGTHS)})


def paired_bc(*, _reproduce_superseded_artifact: bool = False) -> None:
    """RETIRED. Kept only as the generator of the already-written, registered
    ``bootstrap/B_C_paired_case_differences.csv`` so that artifact stays
    reproducible; it must not run in the pipeline
    (PLAN_AMENDMENT_CASE_C_20260830.md s7).

    The implementation also lacks the length-consistency assertion that
    ``within_case`` has: ``n`` is taken from B alone, so a C payload with a
    different state count would index out of range rather than mis-pair.
    """
    if not _reproduce_superseded_artifact:
        raise RuntimeError(
            "paired_bc is retired: it presupposes that B and C share one December "
            "time axis, which the 2026-08-30 amendment ends. Use per-case "
            "within_case() CIs; for a C-B contrast, resample each case "
            "independently and label it as unpaired.")
    print("[phase13] SYNCHRONIZED B/C PAIRED BOOTSTRAP IS SUPERSEDED — "
          "explicit artifact reproduction only, not science")
    A = {cs: _case_arrays(cs) for cs in ("B", "C")}
    I = {cs: _interp_J_S(cs) for cs in ("B", "C")}
    terr = {cs: {"scalar": A[cs]["pred"][("scalar", "terrain_roughness_equation")],
                 "spread": A[cs]["pred"][("spread", "joint_terrain_equation")]}
            for cs in ("B", "C")}
    n = A["B"]["obs"]["scalar"].shape[0]
    rng = C.rng_for("bootstrap_paired_BC")

    quantities = (["J", "S"]
                  + [f"Q_J_{m}" for m in ("bilinear", "bicubic")]
                  + [f"Q_S_{m}" for m in ("bilinear", "bicubic")]
                  + [f"interp_rmse_scalar_{m}" for m in ("bilinear", "bicubic")]
                  + [f"interp_rmse_spread_{m}" for m in ("bilinear", "bicubic")]
                  + ["terrain_rmse_scalar", "terrain_rmse_spread"])

    def per_case(cs: str, idx: np.ndarray) -> dict[str, float]:
        a = A[cs]
        out = {"J": a["J"][idx].mean(), "S": a["S"][idx].mean()}
        om_sc = a["obs"]["scalar"][idx].mean(axis=0)
        om_sp = a["obs"]["spread"][idx].mean(axis=0)
        for m in ("bilinear", "bicubic"):
            out[f"Q_J_{m}"] = I[cs][("J", m)][idx].mean() / out["J"]
            out[f"Q_S_{m}"] = I[cs][("S", m)][idx].mean() / out["S"]
            out[f"interp_rmse_scalar_{m}"] = M.cell_rmse(
                om_sc, a["pred"][("scalar", m)][idx].mean(axis=0))
            out[f"interp_rmse_spread_{m}"] = M.cell_rmse(
                om_sp, a["pred"][("spread", m)][idx].mean(axis=0))
        out["terrain_rmse_scalar"] = M.cell_rmse(om_sc, terr[cs]["scalar"])
        out["terrain_rmse_spread"] = M.cell_rmse(om_sp, terr[cs]["spread"])
        return out

    rows = []
    for length in C.BLOCK_LENGTHS:
        diffs = {q: np.zeros(C.N_REPLICATES) for q in quantities}
        vals_b = {q: np.zeros(C.N_REPLICATES) for q in quantities}
        vals_c = {q: np.zeros(C.N_REPLICATES) for q in quantities}
        for rep in range(C.N_REPLICATES):
            idx = M.block_resample(rng, n, length)   # ONE draw for both sites
            vb = per_case("B", idx)
            vc = per_case("C", idx)
            for q in quantities:
                diffs[q][rep] = vc[q] - vb[q]
                vals_b[q][rep] = vb[q]
                vals_c[q][rep] = vc[q]
        for q in quantities:
            lo, hi = np.percentile(diffs[q], [2.5, 97.5])
            rows.append({
                "quantity": q, "difference": "C_minus_B",
                "block_minutes": length, "replicates": C.N_REPLICATES,
                "diff_est": float(diffs[q].mean()),
                "diff_median": float(np.median(diffs[q])),
                "ci_low": float(lo), "ci_high": float(hi),
                "p_diff_negative": float((diffs[q] < 0).mean()),
                "B_mean": float(vals_b[q].mean()),
                "C_mean": float(vals_c[q].mean()),
                "design": ("synchronized cross-site moving-block bootstrap: "
                           "B and C consume identical resampled time blocks "
                           "per replicate (same December time axis)"),
            })
    C.write_df(C.ext_dir("bootstrap") / "B_C_paired_case_differences.csv",
               pd.DataFrame(rows).round(9),
               status_value=C.combined_status("B", "C"))
    C.run_receipt("phase13_paired_BC", {
        "seed_spawn": list(C.SEED_SPAWN["bootstrap_paired_BC"]),
        "replicates": C.N_REPLICATES, "blocks": list(C.BLOCK_LENGTHS)})


def synchronized_bc() -> None:
    """Synchronized cross-site moving-block bootstrap for the 20260905
    contemporaneous pair (PLAN_MULTICASE_20260905.md §19.2).

    Enabled only for the generation whose contract declares B and C on one
    shared chronological window.  Per replicate, B and C consume the exact
    same resampled chronological blocks, so ``C - B`` contrasts are paired
    contemporaneous cross-site differences: they partially remove shared
    time-window variability.  This is NOT causal terrain identification.
    """
    if not C.GEN.get("bc_synchronized_bootstrap"):
        raise RuntimeError(
            "synchronized_bc requires a generation contract that declares the "
            "B/C contemporaneous window (generation "
            f"{C.GENERATION} does not); paired_bc stays retired for it")
    print("[phase13] synchronized cross-site B/C paired bootstrap "
          "(paired contemporaneous cross-site differences)")
    A = {cs: _case_arrays(cs) for cs in ("B", "C")}
    I = {cs: _interp_J_S(cs) for cs in ("B", "C")}
    terr = {cs: {"scalar": A[cs]["pred"][("scalar", "terrain_roughness_equation")],
                 "spread": A[cs]["pred"][("spread", "joint_terrain_equation")]}
            for cs in ("B", "C")}
    n_b = A["B"]["obs"]["scalar"].shape[0]
    n_c = A["C"]["obs"]["scalar"].shape[0]
    if n_b != n_c or n_b != C.EVAL_SLICE[1] - C.EVAL_SLICE[0]:
        raise RuntimeError(
            "synchronized B/C bootstrap requires identical evaluation axis "
            f"lengths (B {n_b}, C {n_c}, expected "
            f"{C.EVAL_SLICE[1] - C.EVAL_SLICE[0]}); Phase-0 time-alignment "
            "precondition violated")
    ta_path = C.ext_dir("raw_contract") / "time_alignment_audit.json"
    ta = json.loads(ta_path.read_text(encoding="utf-8")) if ta_path.is_file() else {}
    if not ta.get("synchronized_bootstrap_preconditions_met", False):
        raise RuntimeError(
            "Phase-0 time_alignment_audit has not confirmed the B/C "
            "synchronized-bootstrap preconditions; refusing to run")
    n = n_b
    rng = C.rng_for("bootstrap_paired_BC")

    quantities = (["J", "S"]
                  + [f"Q_J_{m}" for m in ("bilinear", "bicubic")]
                  + [f"Q_S_{m}" for m in ("bilinear", "bicubic")]
                  + [f"interp_rmse_scalar_{m}" for m in ("bilinear", "bicubic")]
                  + [f"interp_rmse_spread_{m}" for m in ("bilinear", "bicubic")]
                  + ["terrain_rmse_scalar", "terrain_rmse_spread"])

    def per_case(cs: str, idx: np.ndarray) -> dict[str, float]:
        a = A[cs]
        out = {"J": a["J"][idx].mean(), "S": a["S"][idx].mean()}
        om_sc = a["obs"]["scalar"][idx].mean(axis=0)
        om_sp = a["obs"]["spread"][idx].mean(axis=0)
        for m in ("bilinear", "bicubic"):
            out[f"Q_J_{m}"] = I[cs][("J", m)][idx].mean() / out["J"]
            out[f"Q_S_{m}"] = I[cs][("S", m)][idx].mean() / out["S"]
            out[f"interp_rmse_scalar_{m}"] = M.cell_rmse(
                om_sc, a["pred"][("scalar", m)][idx].mean(axis=0))
            out[f"interp_rmse_spread_{m}"] = M.cell_rmse(
                om_sp, a["pred"][("spread", m)][idx].mean(axis=0))
        out["terrain_rmse_scalar"] = M.cell_rmse(om_sc, terr[cs]["scalar"])
        out["terrain_rmse_spread"] = M.cell_rmse(om_sp, terr[cs]["spread"])
        return out

    rows = []
    for length in C.BLOCK_LENGTHS:
        diffs = {q: np.zeros(C.N_REPLICATES) for q in quantities}
        vals_b = {q: np.zeros(C.N_REPLICATES) for q in quantities}
        vals_c = {q: np.zeros(C.N_REPLICATES) for q in quantities}
        for rep in range(C.N_REPLICATES):
            idx = M.block_resample(rng, n, length)   # ONE draw for both sites
            vb = per_case("B", idx)
            vc = per_case("C", idx)
            for q in quantities:
                diffs[q][rep] = vc[q] - vb[q]
                vals_b[q][rep] = vb[q]
                vals_c[q][rep] = vc[q]
        for q in quantities:
            lo, hi = np.percentile(diffs[q], [2.5, 97.5])
            rows.append({
                "quantity": q, "difference": "C_minus_B",
                "block_minutes": length, "replicates": C.N_REPLICATES,
                "diff_est": float(diffs[q].mean()),
                "diff_median": float(np.median(diffs[q])),
                "ci_low": float(lo), "ci_high": float(hi),
                "p_diff_negative": float((diffs[q] < 0).mean()),
                "B_mean": float(vals_b[q].mean()),
                "C_mean": float(vals_c[q].mean()),
                "design": ("synchronized cross-site moving-block bootstrap: B "
                           "and C consume identical resampled chronological "
                           "blocks per replicate (declared contemporaneous "
                           "2015-12-01 window); the contrast is a paired "
                           "contemporaneous cross-site difference, NOT a "
                           "causal terrain effect"),
            })
    C.write_df(C.ext_dir("bootstrap") /
               "B_C_synchronized_case_differences.csv",
               pd.DataFrame(rows).round(9),
               status_value=C.combined_status("B", "C"))
    C.run_receipt("phase13_synchronized_BC", {
        "seed_spawn": list(C.SEED_SPAWN["bootstrap_paired_BC"]),
        "replicates": C.N_REPLICATES, "blocks": list(C.BLOCK_LENGTHS),
        "wording": "paired contemporaneous cross-site difference"})


def legacy_code_path_validation() -> pd.DataFrame:
    """Re-run the frozen Case-A bootstrap design (seed 20260807) from frozen
    inputs and compare against the frozen interpolation_bootstrap.csv."""
    print("[phase13] legacy code-path validation (seed 20260807)")
    tab = P1.load_moments("A")
    ea, eb = C.EVAL_SLICE
    obs = {"scalar_increment": tab["delta_scalar"][ea:eb],
           "spread": tab["S"][ea:eb]}
    n = eb - ea
    mc = pd.read_csv(C.FROZEN_MODEL_CELLS).sort_values("cell_id")
    pred = {}
    for group, models in [("scalar_increment",
                           ["zero_correction", "global_training_mean",
                            "cell_mean_persistence", "terrain_roughness_equation"]),
                          ("spread",
                           ["global_training_mean", "cell_mean_persistence",
                            "roughness_only", "exposure_only", "locked_joint_equation"])]:
        prefix = "mean" if group == "scalar_increment" else "spread"
        for m in models:
            pred[(group, m)] = mc[f"{prefix}_{m}_predicted"].to_numpy(float)
    ct = pd.read_csv(ROOT_INTERP_CT)
    interp = {}
    for mode in ("bilinear", "bicubic"):
        for group, col in [("scalar_increment", "delta_scalar"), ("spread", "spread")]:
            sub = ct[ct.method == mode].pivot(index="state_index",
                                              columns="cell_id", values=col)
            interp[(group, mode)] = sub.values.astype(float)

    LEGACY_PAIRS = [
        ("terrain_vs_global", "scalar_increment", "terrain_roughness_equation", "global_training_mean"),
        ("terrain_vs_bilinear", "scalar_increment", "terrain_roughness_equation", "bilinear"),
        ("terrain_vs_bicubic", "scalar_increment", "terrain_roughness_equation", "bicubic"),
        ("terrain_vs_persistence", "scalar_increment", "terrain_roughness_equation", "cell_mean_persistence"),
        ("joint_vs_global", "spread", "locked_joint_equation", "global_training_mean"),
        ("joint_vs_bilinear", "spread", "locked_joint_equation", "bilinear"),
        ("joint_vs_bicubic", "spread", "locked_joint_equation", "bicubic"),
        ("joint_vs_persistence", "spread", "locked_joint_equation", "cell_mean_persistence"),
        ("joint_vs_exposure_only", "spread", "locked_joint_equation", "exposure_only"),
    ]

    def prediction_for(group, model):
        if model in ("bilinear", "bicubic"):
            return lambda idx: interp[(group, model)][idx].mean(axis=0)
        return lambda idx: pred[(group, model)]

    rng = np.random.default_rng(C.LEGACY_BOOTSTRAP_SEED)
    rows = []
    for length in C.BLOCK_LENGTHS:   # legacy order 60/120/240
        d = {lab: np.zeros(C.N_REPLICATES) for lab, *_ in LEGACY_PAIRS}
        rmses = {(lab, k): np.zeros(C.N_REPLICATES)
                 for lab, *_ in LEGACY_PAIRS for k in (0, 1)}
        for rep in range(C.N_REPLICATES):
            idx = M.block_resample(rng, n, length)
            for lab, group, ma, mb in LEGACY_PAIRS:
                om = obs[group][idx].mean(axis=0)
                ra = M.cell_rmse(om, prediction_for(group, ma)(idx))
                rb = M.cell_rmse(om, prediction_for(group, mb)(idx))
                d[lab][rep] = ra - rb
                rmses[(lab, 0)][rep] = ra
                rmses[(lab, 1)][rep] = rb
        for lab, group, ma, mb in LEGACY_PAIRS:
            lo, hi = np.percentile(d[lab], [2.5, 97.5])
            rows.append(dict(pair=lab, group=group, model_a=ma, model_b=mb,
                             block_minutes=length, replicates=C.N_REPLICATES,
                             dRMSE_est=float(d[lab].mean()),
                             dRMSE_median=float(np.median(d[lab])),
                             ci_low=float(lo), ci_high=float(hi),
                             p_dRMSE_negative=float((d[lab] < 0).mean()),
                             rmse_a_mean=float(rmses[(lab, 0)].mean()),
                             rmse_b_mean=float(rmses[(lab, 1)].mean())))
    mine = pd.DataFrame(rows)
    frozen = pd.read_csv(C.FROZEN_INTERP_BOOTSTRAP)
    comp_rows = []
    for _, fr in frozen.iterrows():
        mr = mine[(mine.pair == fr["pair"])
                  & (mine.block_minutes == fr["block_minutes"])]
        for col in ("dRMSE_est", "dRMSE_median", "ci_low", "ci_high",
                    "p_dRMSE_negative", "rmse_a_mean", "rmse_b_mean"):
            tol = 2e-3 if col == "p_dRMSE_negative" else 1e-5
            diff = abs(float(mr[col].iloc[0]) - float(fr[col]))
            comp_rows.append({
                "pair": fr["pair"], "block_minutes": fr["block_minutes"],
                "column": col, "recomputed": float(mr[col].iloc[0]),
                "frozen": float(fr[col]), "abs_diff": diff, "tolerance": tol,
                "status": "PASS" if diff <= tol else "FAIL"})
    comp = pd.DataFrame(comp_rows)
    C.write_df(C.ext_dir("bootstrap") / "case_A_legacy_bootstrap_validation.csv",
               comp, case="A")
    n_fail = int((comp.status == "FAIL").sum())
    print(f"  legacy bootstrap validation: {len(comp) - n_fail}/{len(comp)} PASS")
    return comp


ROOT_INTERP_CT = C.ROOT / "source" / "evidence" / "npj_refactor" / "interpolation_cell_time.csv"


if __name__ == "__main__":
    args = sys.argv[1:] or ["A", "B", "C", "paired", "legacy"]
    for cs in [a for a in args if a in C.CASE_ORDER]:
        within_case(cs)
    if "paired" in args:
        paired_bc()
    if "legacy" in args:
        legacy_code_path_validation()
