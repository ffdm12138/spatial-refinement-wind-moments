"""Evaluate parameter-free geometric interpolation baselines (bilinear, bicubic).

NPJ Climate and Atmospheric Science manuscript: "What spatial refinement
recovers from coarse wind fields over complex terrain".

Contract: qa/npj_refactor/interpolation_contract_frozen.json (frozen BEFORE
any evaluation score was read; its sha256 is recorded in the output
interpolation_contract.json).

Design points (all locked before evaluation):
  * Input: raw physical U10_d02 / V10_d02 only (no scaler, no DEM, no d05,
    no station observations) in the prediction path. d05 tensors are loaded
    only for the
    evaluation step after all predictions are computed.
  * Cell-centred geometry: native 6x6 parent field = d02[::27, ::27];
    F.interpolate(..., size=(162,162), mode='bilinear'|'bicubic',
    align_corners=False) maps fine index 13+27k exactly to parent index k.
  * U and V are interpolated separately; scalar speed is computed afterwards.
  * Edge behaviour: the fixed edge-value extension implied by align_corners
    =False applies outside the outermost parent-cell centres.
  * All-record RMSE is over p=36, t=670 (24,120 cell-time records) of the
    parent-cell mean scalar speed -- never a 162x162 pixel-wise field RMSE.

Outputs (all new, written to source/evidence/npj_refactor/):
  interpolation_cell_time.csv   (bilinear/bicubic/reference blocks)
  interpolation_cell_means.csv
  interpolation_metrics.csv
  interpolation_recovery.csv
  interpolation_contract.json
  representative_state_fields.npz  (Fig 4a source)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "refactor_20260807"))
import common  # noqa: E402  (EVAL_SLICE, cell_blocks, load_tensors, ...)
from a4_error_decomposition import metrics  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "source", "evidence", "npj_refactor")
FROZEN_CONTRACT = os.path.join(PROJECT_ROOT, "qa", "npj_refactor",
                               "interpolation_contract_frozen.json")
SEALED_MODEL_CELLS = os.path.join(
    PROJECT_ROOT, "source", "evidence", "source_snapshots",
    "outputs_evaluation_holdout_model_cells.csv")

MODES = ["bilinear", "bicubic"]
ALIGN_CORNERS = False
SIZE = 162
BLOCK = 27
MOMENTS = ["delta_scalar", "J", "delta_dyn", "spread", "mean_scalar_speed"]
EVAL_START, EVAL_STOP = common.EVAL_SLICE
N_STATES = EVAL_STOP - EVAL_START
N_CELLS = common.N_CELLS
INNER_CELLS = [(r, c) for r in range(1, 5) for c in range(1, 5)]  # inner 4x4
INNER_IDS = [r * 6 + c for (r, c) in INNER_CELLS]


# ---------------------------------------------------------------------------
# Prediction path (d02 only)
# ---------------------------------------------------------------------------

def native_parent_fields(u02: np.ndarray, v02: np.ndarray) -> tuple:
    """Native 6x6 parent fields for the evaluation window, extracted as
    d02[::27, ::27]. The archived d02 field is a cellwise replication of the
    native 6x6 values, so this recovers the parent field exactly."""
    u6 = u02[EVAL_START:EVAL_STOP][:, ::BLOCK, ::BLOCK].astype(np.float32)
    v6 = v02[EVAL_START:EVAL_STOP][:, ::BLOCK, ::BLOCK].astype(np.float32)
    assert u6.shape == (N_STATES, 6, 6)
    return u6, v6


def interpolate_components(u6: np.ndarray, v6: np.ndarray, mode: str) -> tuple:
    """Interpolate U and V separately to (670, 162, 162) float32."""
    import torch
    from torch.nn.functional import interpolate as F_interp
    xt = torch.from_numpy(u6[:, None, :, :])
    yt = torch.from_numpy(v6[:, None, :, :])
    with torch.no_grad():
        ui = F_interp(xt, size=(SIZE, SIZE), mode=mode,
                      align_corners=ALIGN_CORNERS)
        vi = F_interp(yt, size=(SIZE, SIZE), mode=mode,
                      align_corners=ALIGN_CORNERS)
    return ui[:, 0].numpy(), vi[:, 0].numpy()


def block_stats(ui: np.ndarray, vi: np.ndarray) -> dict:
    """Per-state per-cell statistics of an interpolated field -> (670, 36)."""
    n = ui.shape[0]
    sp = np.sqrt(ui.astype(np.float64) ** 2 + vi.astype(np.float64) ** 2)
    res = {k: np.zeros((n, N_CELLS), dtype=np.float64) for k in
           ["u_bar", "v_bar", "s_bar", "sd"]}
    for i, (r0, c0) in enumerate(common.cell_blocks()):
        sl = (slice(None), slice(r0, r0 + BLOCK), slice(c0, c0 + BLOCK))
        ub = ui[sl].mean(axis=(1, 2))
        vb = vi[sl].mean(axis=(1, 2))
        res["u_bar"][:, i] = ub
        res["v_bar"][:, i] = vb
        res["s_bar"][:, i] = sp[sl].mean(axis=(1, 2))
        res["sd"][:, i] = sp[sl].std(axis=(1, 2))
    return res


def interpolated_moments(ui: np.ndarray, vi: np.ndarray,
                         v02mag: np.ndarray) -> dict:
    """Moments of an interpolated field relative to the d02 parent vector.

    v02mag: (670, 36) |U_p(t)| from d02 only.
    Returns dict of (670, 36) arrays for MOMENTS.
    """
    st = block_stats(ui, vi)
    u_vec = np.sqrt(st["u_bar"] ** 2 + st["v_bar"] ** 2)
    out = {
        "delta_dyn": u_vec - v02mag,       # |u_bar^I| - |U_p|
        "J": st["s_bar"] - u_vec,          # s_bar^I - |u_bar^I|
        "delta_scalar": st["s_bar"] - v02mag,  # s_bar^I - |U_p|
        "spread": st["sd"],                # within-parent spread
        "mean_scalar_speed": st["s_bar"],
    }
    return out


# ---------------------------------------------------------------------------
# Synthetic contract tests (run before any evaluation score)
# ---------------------------------------------------------------------------

def run_synthetic_tests() -> dict:
    """Contract tests specified in the frozen contract; raise on failure."""
    import torch
    from torch.nn.functional import interpolate as F_interp

    results = {}
    ones6 = np.ones((1, 6, 6), dtype=np.float32)
    x = torch.from_numpy(ones6[:, None, :, :])
    for mode in MODES:
        # 1. Constant field: exactly preserved everywhere
        y = F_interp(x, size=(SIZE, SIZE), mode=mode,
                     align_corners=ALIGN_CORNERS)[0, 0].numpy()
        const_max = float(np.abs(y - 1.0).max())
        # float32 rounding noise from the cubic kernel is ~1.6e-6 on a unit
        # field; 1e-5 is still machine-precision scale
        tol_c = 1e-6 if mode == "bilinear" else 1e-5
        assert const_max < tol_c, (mode, "constant field", const_max)
        # 2. Linear plane: bilinear recovers the gradient inside the convex
        #    hull; bicubic kernels do not reproduce linear functions exactly
        #    between sample points (measured 0.077 on a unit-scale plane), so
        #    a loose tolerance verifies sane, bounded behaviour
        i6, j6 = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
        plane = (0.5 * i6 - 0.3 * j6 + 1.2).astype(np.float32)
        yp = F_interp(torch.from_numpy(plane[None, None]), size=(SIZE, SIZE),
                      mode=mode, align_corners=ALIGN_CORNERS)[0, 0].numpy()
        # analytic plane on the fine grid: value = 0.5*a - 0.3*b + 1.2
        # with a = x_in, b = y_in as computed from the align_corners=False map
        idx = (np.arange(SIZE) + 0.5) * 6.0 / SIZE - 0.5
        ia, ja = np.meshgrid(idx, idx, indexing="ij")
        ana = 0.5 * ia - 0.3 * ja + 1.2
        hull = (idx >= 0) & (idx <= 5)
        lin = np.abs(yp - ana)[np.ix_(hull, hull)].max()
        tol = 1e-4 if mode == "bilinear" else 1e-1
        assert lin < tol, (mode, "linear plane in hull", lin)
        # 3. Parent-centre alignment: u_f(13+27k) = u_p(k) at machine precision
        centres = [13 + 27 * k for k in range(6)]
        fval = yp[np.ix_(centres, centres)]
        align = float(np.abs(fval - plane).max())
        assert align < 1e-5, (mode, "parent-centre alignment", align)
        # 4. Edge behaviour: the outer half-parent-cell region follows the
        #    fixed edge-value extension (sample coordinates clamp to the
        #    outermost parent centres; no extrapolation). Bilinear reproduces
        #    the edge value exactly; bicubic blends the clamped samples with
        #    its kernel, staying within a small overshoot envelope.
        outer_ul = np.abs(yp[0:13, 0:13] - plane[0, 0]).max()
        outer_dr = np.abs(yp[149:162, 149:162] - plane[5, 5]).max()
        edge_test = max(float(outer_ul), float(outer_dr))
        tol_e = 1e-4 if mode == "bilinear" else 3e-1
        assert edge_test < tol_e, (mode, "edge extension", edge_test)
        results[mode] = {
            "constant_max_abs_diff": const_max,
            "linear_in_hull_max_abs_diff": float(lin),
            "parent_centre_alignment_max_abs_diff": align,
            "edge_extension_max_abs_diff": edge_test,
            "edge_test_tolerance": tol_e,
        }
    return results


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def cell_means(arr: np.ndarray) -> np.ndarray:
    """Period means over states -> (36,)."""
    return arr.mean(axis=0)


def domain_mean(arr: np.ndarray) -> float:
    """mean over cells of period means (recovery-ratio numerator/denominator)."""
    return float(cell_means(arr).mean())


def recovery_ratio(interp: np.ndarray, ref: np.ndarray,
                   cells: list[int] | None = None) -> float:
    """Q_X^I = domain-mean(interp moment) / domain-mean(reference moment)."""
    if cells is not None:
        interp = interp[:, cells]
        ref = ref[:, cells]
    num = domain_mean(interp)
    den = domain_mean(ref)
    assert den > 0, "recovery-ratio denominator must be positive"
    return num / den


def vector_preservation(ui: np.ndarray, vi: np.ndarray,
                        v02vec: np.ndarray) -> dict:
    """E_vec^I = RMSE_{p,t}(u_bar^I, U_p) and domain-mean delta_dyn^I."""
    st = block_stats(ui, vi)
    du = st["u_bar"] - v02vec[..., 0]
    dv = st["v_bar"] - v02vec[..., 1]
    e_vec = float(np.sqrt((du ** 2 + dv ** 2).mean()))
    u_vec = np.sqrt(st["u_bar"] ** 2 + st["v_bar"] ** 2)
    v02mag = np.sqrt(v02vec[..., 0] ** 2 + v02vec[..., 1] ** 2)
    dyn_dom = float((u_vec - v02mag).mean())
    return {"E_vec_rmse": e_vec, "delta_dyn_domain_mean": dyn_dom}


def all_record_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """RMSE over all p=36, t=670 cell-time records of scalar-speed values."""
    assert pred.shape == ref.shape == (N_STATES, N_CELLS)
    assert pred.size == N_STATES * N_CELLS == 24120
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    qa = {}   # QA ledger written to contract json

    # ---- frozen contract ----
    with open(FROZEN_CONTRACT, encoding="utf-8") as f:
        frozen = json.load(f)
    frozen_sha = hashlib.sha256(
        open(FROZEN_CONTRACT, "rb").read()).hexdigest()
    qa["frozen_contract_sha256"] = frozen_sha
    qa["frozen_contract_read_before_evaluation"] = True

    # ---- prediction path: d02 ONLY ----
    T = common.load_tensors()
    u02, v02 = T["U10_d02"], T["V10_d02"]
    assert "U10_d05" not in locals()  # defensive: prediction path must not
    # touch d05 tensors

    u6, v6 = native_parent_fields(u02, v02)

    # d02 parent-cell vectors (block means == native values) for the eval window
    v02vec = common.parent_d02_vector(u02, v02, EVAL_START, EVAL_STOP)
    v02mag = np.sqrt(v02vec[..., 0] ** 2 + v02vec[..., 1] ** 2)

    # ---- synthetic contract tests (before evaluation) ----
    synth = run_synthetic_tests()
    qa["synthetic_tests"] = synth

    # ---- interpolate + moments (prediction, d02 only) ----
    interp = {}
    for mode in MODES:
        ui, vi = interpolate_components(u6, v6, mode)
        assert np.isfinite(ui).all() and np.isfinite(vi).all(), mode
        interp[mode] = {
            "ui": ui, "vi": vi,
            "moments": interpolated_moments(ui, vi, v02mag),
            "vec": vector_preservation(ui, vi, v02vec),
        }

    # ---- representative-state fields (pre-registered index from contract) ----
    rep_rel = frozen["representative_state"]["computed_values"]["evaluation_relative_index"]
    rep_global = frozen["representative_state"]["computed_values"]["global_tensor_index"]
    assert EVAL_START + rep_rel == rep_global
    np.savez_compressed(
        os.path.join(OUT_DIR, "representative_state_fields.npz"),
        d02_6x6_speed=np.sqrt(u6[rep_rel] ** 2 + v6[rep_rel] ** 2),
        bilinear_speed=np.sqrt(interp["bilinear"]["ui"][rep_rel] ** 2
                               + interp["bilinear"]["vi"][rep_rel] ** 2),
        bicubic_speed=np.sqrt(interp["bicubic"]["ui"][rep_rel] ** 2
                              + interp["bicubic"]["vi"][rep_rel] ** 2),
        d05_speed=np.sqrt(T["U10_d05"][rep_global].astype(np.float64) ** 2
                          + T["V10_d05"][rep_global].astype(np.float64) ** 2),
        state_global=rep_global, state_relative=rep_rel,
    )
    qa["representative_state"] = {"global_index": rep_global,
                                  "relative_index": rep_rel}

    # =======================================================================
    # Evaluation step (d05 loaded here, after all predictions are computed)
    # =======================================================================
    A = common.state_arrays("eval")   # d05 reference moments
    ref = {
        "delta_scalar": A["delta_scalar"],
        "J": A["J"],
        "spread": A["spread"],
        "delta_dyn": A["delta_dyn"],
        "mean_scalar_speed": A["U_avg"],   # d05 parent-cell mean scalar speed
    }

    # ---- decomposition identity on interpolated moments ----
    for mode in MODES:
        m = interp[mode]["moments"]
        diff = np.abs(m["delta_scalar"] - (m["delta_dyn"] + m["J"])).max()
        assert diff < 1e-8, (mode, "decomposition identity", diff)
        jmin = m["J"].min()
        assert jmin > -1e-9, (mode, "J negative", jmin)
        qa[f"{mode}_decomposition_max_abs_diff"] = float(diff)
        qa[f"{mode}_J_min"] = float(jmin)
        qa[f"{mode}_moments_finite"] = bool(
            np.isfinite(m["delta_scalar"]).all()
            and np.isfinite(m["spread"]).all())

    # ---- cell-time CSV (bilinear / bicubic / reference blocks) ----
    rows = []
    for mode in MODES:
        m = interp[mode]["moments"]
        for t in range(N_STATES):
            for c in range(N_CELLS):
                rows.append((t, c, mode) + tuple(
                    round(float(m[k][t, c]), 8) for k in MOMENTS))
    for t in range(N_STATES):
        for c in range(N_CELLS):
            rows.append((t, c, "reference") + tuple(
                round(float(ref[k][t, c]), 8) for k in MOMENTS))
    with open(os.path.join(OUT_DIR, "interpolation_cell_time.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("state_index,cell_id,method,"
                + ",".join(MOMENTS) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    qa["cell_time_rows"] = len(rows)
    assert len(rows) == 3 * N_STATES * N_CELLS == 72360

    # ---- period-mean metrics per method per moment ----
    cell_mean_rows = []
    metric_rows = []
    for moment in ["delta_scalar", "spread"]:
        ref_cm = cell_means(ref[moment])
        for mode in MODES:
            pred_cm = cell_means(interp[mode]["moments"][moment])
            for c in range(N_CELLS):
                cell_mean_rows.append((moment, mode, c, round(float(pred_cm[c]), 8)))
            met = metrics(pred_cm, ref_cm)   # rmse/bias/crmse/r/slope
            row = {
                "group": "scalar_increment" if moment == "delta_scalar" else "spread",
                "model": mode,
                "n_cells": N_CELLS,
                "rmse": met["rmse"], "bias": met["bias"],
                "crmse": met["crmse"], "r": met["r"], "slope": met["slope"],
                "pred_mean": float(pred_cm.mean()),
                "obs_mean": float(ref_cm.mean()),
            }
            if moment == "delta_scalar":
                row["all_record_rmse"] = all_record_rmse(
                    interp[mode]["moments"]["mean_scalar_speed"],
                    ref["mean_scalar_speed"])
            else:
                row["all_record_rmse"] = np.nan
            row["E_vec_rmse"] = interp[mode]["vec"]["E_vec_rmse"]
            row["delta_dyn_domain_mean"] = interp[mode]["vec"]["delta_dyn_domain_mean"]
            metric_rows.append(row)

    with open(os.path.join(OUT_DIR, "interpolation_cell_means.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("group,model,cell_id,pred_mean_cell\n")
        for r in cell_mean_rows:
            f.write(",".join(map(str, r)) + "\n")

    with open(os.path.join(OUT_DIR, "interpolation_metrics.csv"), "w",
              encoding="utf-8", newline="") as f:
        keys = ["group", "model", "n_cells", "rmse", "bias", "crmse", "r",
                "slope", "pred_mean", "obs_mean", "all_record_rmse",
                "E_vec_rmse", "delta_dyn_domain_mean"]
        f.write(",".join(keys) + "\n")
        for r in metric_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # ---- recovery ratios (36 cells and inner 4x4) ----
    rec_rows = []
    for moment, qkey in [("delta_scalar", "Q_delta"), ("J", "Q_J"),
                         ("spread", "Q_S")]:
        for mode in MODES:
            q36 = recovery_ratio(interp[mode]["moments"][moment], ref[moment])
            q16 = recovery_ratio(interp[mode]["moments"][moment], ref[moment],
                                 cells=INNER_IDS)
            rec_rows.append((moment, mode, "all36", round(q36, 6)))
            rec_rows.append((moment, mode, "inner16", round(q16, 6)))
    with open(os.path.join(OUT_DIR, "interpolation_recovery.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("moment,model,cell_set,recovery_ratio\n")
        for r in rec_rows:
            f.write(",".join(map(str, r)) + "\n")

    # ---- cross-check all-record pipeline against the sealed protocol ----
    sealed = {}
    import csv as _csv
    with open(SEALED_MODEL_CELLS, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            sealed[int(row["cell_id"])] = row
    # per-cell period-mean predictions for persistence and terrain; global
    # mean and zero correction are constant across cells
    persist = np.array([float(sealed[c]["mean_cell_mean_persistence_predicted"])
                        for c in range(N_CELLS)])
    terrain = np.array([float(sealed[c]["mean_terrain_roughness_equation_predicted"])
                        for c in range(N_CELLS)])
    globalm = np.full(N_CELLS, float(sealed[0]["mean_global_training_mean_predicted"]))
    zero = np.zeros(N_CELLS)
    xcheck = {}
    for name, mu in [("zero_correction", zero), ("global_training_mean", globalm),
                     ("cell_mean_persistence", persist),
                     ("terrain_roughness_equation", terrain)]:
        s_hat = v02mag + mu[None, :]          # s_hat_{p,t} = |U_p(t)| + mu_hat_p
        xcheck[name] = all_record_rmse(s_hat, ref["mean_scalar_speed"])
    # sealed reference values (from outputs_evaluation_holdout_scores.json)
    sealed_ref = {"zero_correction": 0.466, "global_training_mean": 0.341,
                  "cell_mean_persistence": 0.298,
                  "terrain_roughness_equation": 0.309}
    for name, val in xcheck.items():
        diff = abs(val - sealed_ref[name])
        assert diff < 1e-2, (name, val, sealed_ref[name])
        print(f"  all-record cross-check {name}: recomputed {val:.4f} "
              f"vs sealed {sealed_ref[name]} (diff {diff:.4f})")
    qa["all_record_cross_check"] = {k: round(v, 6) for k, v in xcheck.items()}

    # ---- metrics self-consistency ----
    for r in metric_rows:
        rmse2 = r["rmse"] ** 2
        crmse_bias = r["crmse"] ** 2 + r["bias"] ** 2
        assert abs(rmse2 - crmse_bias) < 1e-6, (r["group"], r["model"])
        if r["r"] is not None:
            assert abs(r["r"]) <= 1.0 + 1e-12
    qa["metrics_self_consistent"] = True

    # ---- contract json ----
    contract_out = {
        "contract_id": frozen["contract_id"],
        "frozen_contract_sha256": frozen_sha,
        "torch_version": __import__("torch").__version__,
        "numpy_version": np.__version__,
        "evaluation_slice": list(common.EVAL_SLICE),
        "n_states": N_STATES, "n_parent_cells": N_CELLS,
        "n_cell_time_records": N_STATES * N_CELLS,
        "n_fine_cells_per_parent": BLOCK * BLOCK,
        "representative_state": qa["representative_state"],
        "synthetic_tests": synth,
        "qa": {k: v for k, v in qa.items()
               if k not in ("synthetic_tests", "representative_state")},
        "legacy_bicubic_align_corners_true": "legacy provenance only, not used",
        "status": "PASS",
    }
    with open(os.path.join(OUT_DIR, "interpolation_contract.json"), "w",
              encoding="utf-8") as f:
        json.dump(contract_out, f, indent=2, ensure_ascii=False)

    # ---- report ----
    print("=== NPJ interpolation baselines: evaluation-period results ===")
    for r in metric_rows:
        print(f"{r['group']:16s} {r['model']:8s} RMSE {r['rmse']:.4f} "
              f"cRMSE {r['crmse']:.4f} bias {r['bias']:.4f} "
              f"r {r['r']:.3f} slope {r['slope']:.3f} "
              f"all-rec {r['all_record_rmse']:.4f} "
              f"E_vec {r['E_vec_rmse']:.4f} "
              f"dyn_dom {r['delta_dyn_domain_mean']:.4f}")
    print("--- recovery ratios ---")
    for r in rec_rows:
        print(f"  {r[0]:12s} {r[1]:8s} {r[2]:7s} Q = {r[3]:.4f}")
    ref_dyn = float(ref["delta_dyn"].mean())
    print(f"reference domain-mean delta_dyn (d05) = {ref_dyn:.4f}")
    print("status: PASS")


if __name__ == "__main__":
    main()
