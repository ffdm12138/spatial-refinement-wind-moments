"""Phases 5–6 — geometric refinement replication + parent-vector-constrained
sensitivity (PLAN §10–11), under the frozen interpolation contract
(qa/npj_refactor/interpolation_contract_frozen.json):

  * native 6x6 parent field = d02[:, ::27, ::27] (U and V separately)
  * torch F.interpolate(size=(162,162), mode=bilinear|bicubic,
    align_corners=False); scalar speed computed AFTER interpolation
  * prediction path uses d02 instantaneous U/V ONLY (no d05, no terrain,
    no history, no station); d05 is loaded only after all predictions
  * scored on the frozen final 670 evaluation states; parent centres at
    fine index 13 + 27k; frozen edge-value extension
  * representative state: evaluation state whose d02 domain-mean wind speed
    (1/36 sum over parent-vector magnitudes) is closest to the median,
    earliest tie (Case-A frozen answer: global index 3955 — recomputed here
    as a cross-check)
  * parent-vector-constrained variant: per 27x27 block constant U/V offset
    returning the block mean to the d02 parent vector exactly
    (max component residual < 1e-6 m/s, machine precision expected)

Full-window (0..4320) interpolated moment tables are also persisted for the
spin-up sensitivity; the PRIMARY scores remain evaluation-only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import mc_moments as M  # noqa: E402
import phase01_moments as P1  # noqa: E402

MODES = ("bilinear", "bicubic")
MOM_COLS = ["delta_scalar", "J", "delta_dyn", "spread", "mean_scalar_speed"]


def interp_npz_path(case: str) -> Path:
    return C.case_dir(case) / "interpolation_cell_time_full.npz"


def run_synthetic_tests() -> dict:
    """Frozen synthetic contract tests (must pass BEFORE any evaluation)."""
    import torch
    from torch.nn.functional import interpolate as F_interp
    results = {}
    size = C.GRID
    ones6 = torch.ones((1, 1, 6, 6))
    i6, j6 = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    plane = (0.5 * i6 - 0.3 * j6 + 1.2).astype(np.float32)
    idx = (np.arange(size) + 0.5) * 6.0 / size - 0.5
    ia, ja = np.meshgrid(idx, idx, indexing="ij")
    ana = 0.5 * ia - 0.3 * ja + 1.2
    hull = (idx >= 0) & (idx <= 5)
    centres = C.PARENT_CENTRE_FINE_INDICES
    for mode in MODES:
        with torch.no_grad():
            y1 = F_interp(ones6, size=(size, size), mode=mode,
                          align_corners=False)[0, 0].numpy()
            yp = F_interp(torch.from_numpy(plane[None, None]), size=(size, size),
                          mode=mode, align_corners=False)[0, 0].numpy()
        const_max = float(np.abs(y1 - 1.0).max())
        assert const_max < (1e-6 if mode == "bilinear" else 1e-5), (mode, const_max)
        lin = float(np.abs(yp - ana)[np.ix_(hull, hull)].max())
        assert lin < (1e-4 if mode == "bilinear" else 1e-1), (mode, lin)
        align = float(np.abs(yp[np.ix_(centres, centres)] - plane).max())
        assert align < 1e-5, (mode, align)
        edge = max(float(np.abs(yp[0:13, 0:13] - plane[0, 0]).max()),
                   float(np.abs(yp[149:162, 149:162] - plane[5, 5]).max()))
        assert edge < (1e-4 if mode == "bilinear" else 3e-1), (mode, edge)
        results[mode] = {"constant_max_abs_diff": const_max,
                         "linear_in_hull_max_abs_diff": lin,
                         "parent_centre_alignment_max_abs_diff": align,
                         "edge_extension_max_abs_diff": edge}
    return results


def interpolate_full(case: str) -> dict:
    """Prediction path: d02 ONLY.  Returns per-mode full-window moment tables
    and the constrained variant, plus the representative-state provenance."""
    import torch
    from torch.nn.functional import interpolate as F_interp

    u02, v02 = C.load_case_wind(case, "d02")
    u6 = np.ascontiguousarray(u02[:, ::C.BLOCK, ::C.BLOCK])
    v6 = np.ascontiguousarray(v02[:, ::C.BLOCK, ::C.BLOCK])
    C.free_case_wind(case)
    assert u6.shape == (C.RAW_N, 6, 6)

    # parent vectors / magnitudes from the native field (d02 only)
    u02p = u6.astype(np.float64).reshape(C.RAW_N, 36)
    v02p = v6.astype(np.float64).reshape(C.RAW_N, 36)
    U02_mag = np.hypot(u02p, v02p)

    # representative state (frozen rule, d02 only, evaluation window)
    ea, eb = C.EVAL_SLICE
    dm = U02_mag[ea:eb].mean(axis=1)
    med = float(np.median(dm))
    dev = np.abs(dm - med)
    rel = int(np.flatnonzero(dev == dev.min())[0])     # earliest tie
    rep = {"selection_rule": ("closest to median d02 domain-mean wind speed "
                              "(1/36 sum of parent-vector magnitudes) over the "
                              "670 evaluation states; earliest tie"),
           "selection_uses": "d02 only",
           "domain_mean_d02_speed_median": med,
           "evaluation_relative_index": rel,
           "global_tensor_index": ea + rel,
           "domain_mean_d02_speed_at_selected": float(dm[rel]),
           "absolute_deviation_from_median": float(dev[rel])}

    out = {"rep": rep, "U02_mag": U02_mag, "u02p": u02p, "v02p": v02p,
           "rep_fields": {"d02_u": u6[ea + rel].copy(),
                          "d02_v": v6[ea + rel].copy(),
                          "d02_6x6_speed": np.hypot(
                              u6[ea + rel].astype(np.float64),
                              v6[ea + rel].astype(np.float64))}}

    chunk = 720
    for mode in MODES:
        tabs = {k: np.zeros((C.RAW_N, 36)) for k in
                ["u_bar", "v_bar", "s_bar", "sd",
                 "u_bar_c", "v_bar_c", "s_bar_c", "sd_c"]}
        max_resid = 0.0
        rep_speed = None
        rep_speed_c = None
        with torch.no_grad():
            for a in range(0, C.RAW_N, chunk):
                b = min(a + chunk, C.RAW_N)
                ui = F_interp(torch.from_numpy(u6[a:b][:, None]),
                              size=(C.GRID, C.GRID), mode=mode,
                              align_corners=False)[:, 0].numpy()
                vi = F_interp(torch.from_numpy(v6[a:b][:, None]),
                              size=(C.GRID, C.GRID), mode=mode,
                              align_corners=False)[:, 0].numpy()
                assert np.isfinite(ui).all() and np.isfinite(vi).all()
                ui64 = ui.astype(np.float64)
                vi64 = vi.astype(np.float64)
                sp = np.hypot(ui64, vi64)
                tabs["u_bar"][a:b] = M.parent_block_means(ui64)
                tabs["v_bar"][a:b] = M.parent_block_means(vi64)
                tabs["s_bar"][a:b] = M.parent_block_means(sp)
                s2 = M.parent_block_means(sp * sp)
                tabs["sd"][a:b] = np.sqrt(np.maximum(
                    s2 - tabs["s_bar"][a:b] ** 2, 0.0))
                # ---- parent-vector-constrained variant (Phase 6) ----
                off_u = (u02p[a:b] - tabs["u_bar"][a:b])
                off_v = (v02p[a:b] - tabs["v_bar"][a:b])
                ou = np.repeat(np.repeat(off_u.reshape(-1, 6, 6), C.BLOCK, 1),
                               C.BLOCK, 2)
                ov = np.repeat(np.repeat(off_v.reshape(-1, 6, 6), C.BLOCK, 1),
                               C.BLOCK, 2)
                uc = ui64 + ou
                vc = vi64 + ov
                spc = np.hypot(uc, vc)
                tabs["u_bar_c"][a:b] = M.parent_block_means(uc)
                tabs["v_bar_c"][a:b] = M.parent_block_means(vc)
                tabs["s_bar_c"][a:b] = M.parent_block_means(spc)
                s2c = M.parent_block_means(spc * spc)
                tabs["sd_c"][a:b] = np.sqrt(np.maximum(
                    s2c - tabs["s_bar_c"][a:b] ** 2, 0.0))
                resid = max(float(np.abs(tabs["u_bar_c"][a:b] - u02p[a:b]).max()),
                            float(np.abs(tabs["v_bar_c"][a:b] - v02p[a:b]).max()))
                max_resid = max(max_resid, resid)
                gidx = ea + rel
                if a <= gidx < b:
                    rep_speed = sp[gidx - a].astype(np.float32)
                    rep_speed_c = spc[gidx - a].astype(np.float32)
        assert max_resid < 1e-6, (case, mode, "constrained residual", max_resid)
        out[mode] = {
            "u_bar": tabs["u_bar"], "v_bar": tabs["v_bar"],
            "s_bar": tabs["s_bar"], "sd": tabs["sd"],
            "u_vec": np.hypot(tabs["u_bar"], tabs["v_bar"]),
            "s_bar_c": tabs["s_bar_c"], "sd_c": tabs["sd_c"],
            "u_vec_c": np.hypot(tabs["u_bar_c"], tabs["v_bar_c"]),
            "u_bar_c": tabs["u_bar_c"], "v_bar_c": tabs["v_bar_c"],
            "constrained_max_component_residual": max_resid,
            "rep_speed": rep_speed, "rep_speed_c": rep_speed_c,
        }
    return out


def moment_tables(res: dict, mode: str, constrained: bool) -> dict:
    sfx = "_c" if constrained else ""
    u_vec = res[mode][f"u_vec{sfx}"]
    s_bar = res[mode][f"s_bar{sfx}"]
    return {
        "delta_dyn": u_vec - res["U02_mag"],
        "J": s_bar - u_vec,
        "delta_scalar": s_bar - res["U02_mag"],
        "spread": res[mode][f"sd{sfx}"],
        "mean_scalar_speed": s_bar,
    }


def main(case: str) -> None:
    print(f"[phase05-06] case {case}: interpolation + constrained sensitivity")
    cdir = C.case_dir(case)
    qa: dict = {}
    frozen_sha = hashlib.sha256(C.FROZEN_INTERP_CONTRACT.read_bytes()).hexdigest()
    qa["frozen_contract_sha256"] = frozen_sha
    qa["synthetic_tests"] = run_synthetic_tests()

    # ---------------- prediction path (d02 only) ----------------
    res = interpolate_full(case)
    rep = res["rep"]
    ea, eb = C.EVAL_SLICE
    n_eval = eb - ea

    # persist full-window npz (spin-up sensitivity consumes this)
    np.savez_compressed(
        interp_npz_path(case),
        **{f"{m}_{k}": moment_tables(res, m, False)[k]
           for m in MODES for k in MOM_COLS if k != "mean_scalar_speed"},
        **{f"{m}_mean_scalar_speed": moment_tables(res, m, False)["mean_scalar_speed"]
           for m in MODES},
        **{f"{m}_constrained_{k}": moment_tables(res, m, True)[k]
           for m in MODES for k in MOM_COLS},
        U02_mag=res["U02_mag"])
    C.write_json(interp_npz_path(case).with_suffix(".metadata.json"), {
        "artifact": "interpolation_cell_time_full.npz",
        "shape_contract": "full 4320-state interpolation moment arrays",
        "models": ["bilinear", "bicubic", "bilinear_constrained",
                   "bicubic_constrained"],
    }, case=case)

    # ---------------- evaluation step (d05 loaded only now) -----------------
    qa["prediction_complete_before_reference_load"] = True
    tab = P1.load_moments(case)      # d05 reference moments (from phase01 npz)
    ref = {"delta_scalar": tab["delta_scalar"][ea:eb],
           "J": tab["J"][ea:eb],
           "delta_dyn": tab["delta_dyn"][ea:eb],
           "spread": tab["S"][ea:eb],
           "mean_scalar_speed": tab["s_bar"][ea:eb]}

    # representative-state fields npz (figure-only derived asset)
    u05, v05 = C.load_case_wind(case, "d05")
    g = rep["global_tensor_index"]
    d05u = u05[g].astype(np.float32)
    d05v = v05[g].astype(np.float32)
    C.free_case_wind(case)
    np.savez_compressed(
        cdir / "representative_state_fields.npz",
        d02_u=res["rep_fields"]["d02_u"], d02_v=res["rep_fields"]["d02_v"],
        d02_6x6_speed=res["rep_fields"]["d02_6x6_speed"],
        bilinear_speed=res["bilinear"]["rep_speed"],
        bicubic_speed=res["bicubic"]["rep_speed"],
        bilinear_constrained_speed=res["bilinear"]["rep_speed_c"],
        bicubic_constrained_speed=res["bicubic"]["rep_speed_c"],
        d05_u=d05u, d05_v=d05v,
        d05_speed=np.hypot(d05u.astype(np.float64), d05v.astype(np.float64)),
        state_global=g, state_relative=rep["evaluation_relative_index"],
        data_status=C.data_status(case))
    C.write_json(cdir / "representative_state_fields.metadata.json", {
        "artifact": "representative_state_fields.npz",
        "purpose": "diagnostic representative-state fields only",
    }, case=case)
    C.write_json(cdir / "representative_state_provenance.json", rep, case=case)

    # ---------------- eval-only cell-time CSV + metrics ---------------------
    ct_rows = []
    metric_rows, rec_rows, cm_rows = [], [], []
    variants = ([(m, False, m) for m in MODES]
                + [(m, True, f"{m}_constrained") for m in MODES])
    for mode, constrained, label in variants:
        mom = {k: v[ea:eb] for k, v in moment_tables(res, mode, constrained).items()}
        ident = float(np.abs(mom["delta_scalar"]
                             - (mom["delta_dyn"] + mom["J"])).max())
        assert ident < 1e-8, (label, "identity", ident)
        jmin = float(mom["J"].min())
        assert jmin > -1e-9, (label, "J negative", jmin)
        qa[f"{label}_identity_max_abs"] = ident
        qa[f"{label}_J_min"] = jmin
        # cell-time rows (vectorized assembly)
        dfm = pd.DataFrame({
            "state_index": np.repeat(np.arange(n_eval), 36),
            "cell_id": np.tile(np.arange(36), n_eval),
            "method": label,
            **{k: np.round(mom[k].reshape(-1), 8) for k in MOM_COLS}})
        ct_rows.append(dfm)

        # period-mean metrics + all-record RMSE + vector preservation
        sfx = "_c" if constrained else ""
        du = res[mode][f"u_bar{sfx}"][ea:eb] - res["u02p"][ea:eb]
        dv = res[mode][f"v_bar{sfx}"][ea:eb] - res["v02p"][ea:eb]
        e_vec = float(np.sqrt((du ** 2 + dv ** 2).mean()))
        dyn_dom = float(mom["delta_dyn"].mean())
        for moment, group in [("delta_scalar", "scalar_increment"),
                              ("spread", "spread")]:
            pred_cm = mom[moment].mean(axis=0)
            ref_cm = ref[moment].mean(axis=0)
            met = M.cell_metrics(pred_cm, ref_cm)
            row = {"group": group, "model": label, "n_cells": 36, **met,
                   "pred_mean": float(pred_cm.mean()),
                   "obs_mean": float(ref_cm.mean()),
                   "all_record_rmse": (float(np.sqrt(np.mean(
                       (mom["mean_scalar_speed"] - ref["mean_scalar_speed"]) ** 2)))
                       if moment == "delta_scalar" else np.nan),
                   "E_vec_rmse": e_vec, "delta_dyn_domain_mean": dyn_dom}
            metric_rows.append(row)
            for cid in range(36):
                cm_rows.append({"group": group, "model": label, "cell_id": cid,
                                "pred_mean_cell": round(float(pred_cm[cid]), 8)})
        # recovery ratios (all36 + inner16)
        for moment, _q in [("delta_scalar", "Q_delta"), ("J", "Q_J"),
                           ("spread", "Q_S")]:
            for cells, cname in [(None, "all36"), (C.INNER16_IDS, "inner16")]:
                pm = mom[moment] if cells is None else mom[moment][:, cells]
                rm = ref[moment] if cells is None else ref[moment][:, cells]
                den = float(rm.mean(axis=0).mean())
                assert den > 0, (label, moment, "denominator")
                rec_rows.append({"moment": moment, "model": label,
                                 "cell_set": cname,
                                 "recovery_ratio": round(float(
                                     pm.mean(axis=0).mean()) / den, 6)})

    # reference block appended to cell-time CSV (frozen layout)
    dfr = pd.DataFrame({
        "state_index": np.repeat(np.arange(n_eval), 36),
        "cell_id": np.tile(np.arange(36), n_eval),
        "method": "reference",
        **{k: np.round(ref[k].reshape(-1), 8) for k in MOM_COLS}})
    ct = pd.concat(ct_rows + [dfr], ignore_index=True)
    C.write_df(cdir / "interpolation_cell_time.csv", ct, case=case)
    C.write_df(cdir / "interpolation_cell_means.csv",
               pd.DataFrame(cm_rows), case=case)
    C.write_df(cdir / "interpolation_metrics.csv",
               pd.DataFrame(metric_rows), case=case)
    C.write_df(cdir / "interpolation_recovery.csv",
               pd.DataFrame(rec_rows), case=case)

    # split constrained outputs (frozen A layout keeps them separate)
    mdf = pd.DataFrame(metric_rows)
    rdf = pd.DataFrame(rec_rows)
    C.write_df(cdir / "parent_vector_constrained_metrics.csv",
               mdf[mdf.model.str.endswith("_constrained")], case=case)
    C.write_df(cdir / "parent_vector_constrained_recovery.csv",
               rdf[rdf.model.str.endswith("_constrained")], case=case)

    contract = {
        "contract_id": "npj-interpolation-baselines-v1 (multicase extension)",
        "frozen_contract_sha256": frozen_sha,
        "case": case,
        "evaluation_slice": list(C.EVAL_SLICE),
        "n_states": n_eval, "n_parent_cells": 36,
        "n_cell_time_records": n_eval * 36,
        "prediction_inputs": ["U10_d02.pt", "V10_d02.pt"],
        "prediction_excludes": ["d05", "terrain", "history", "station"],
        "prediction_complete_before_reference_load": True,
        "representative_state": rep,
        "constrained_max_component_residual": {
            m: res[m]["constrained_max_component_residual"] for m in MODES},
        "constrained_tolerance_m_s-1": 1e-6,
        "known_limitation": ("independent block offsets may introduce "
                             "parent-boundary discontinuities"),
        "qa": {k: v for k, v in qa.items() if k != "synthetic_tests"},
        "synthetic_tests": qa["synthetic_tests"],
        "status": "PASS",
    }
    C.write_json(cdir / "interpolation_contract.json", contract, case=case)
    C.run_receipt(f"phase05_interpolation_case_{case}", {
        "case": case, "data_status": C.data_status(case),
        "representative_state_global": rep["global_tensor_index"],
        "constrained_residuals": contract["constrained_max_component_residual"]})


if __name__ == "__main__":
    for cs in (sys.argv[1:] or ["A"]):
        main(cs)
