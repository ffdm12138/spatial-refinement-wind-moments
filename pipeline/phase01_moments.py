"""Phases 1–3 — unified estimand recomputation + second-order Jensen diagnostic.

Per case, from raw tensors ONLY (identical mathematical definitions for
A/B/C; PLAN §6–8):
  * full-state (0..4320) per-parent moment tables -> moments_cell_time.npz
    (all downstream statistics, bootstraps and the spin-up sensitivity
    consume this table; raw tensors are reduced exactly once here)
  * summary CSVs for calibration / evaluation / retained windows
  * exact-identity and J>=0 checks (delta_scalar = delta_dyn + J, ddof=0)
  * second-order diagnostic J2nd = sigma_perp^2 / (2 |u_bar|) with the frozen
    Case-A speed bins (identical across cases; 1.5 m/s is a descriptive
    reference only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import mc_moments as M  # noqa: E402

MOMENTS = ["delta_dyn", "J", "delta_scalar", "S"]
WINDOWS = {"calibration": C.CAL_SLICE, "evaluation": C.EVAL_SLICE,
           "retained": C.RETAINED_SLICE}


def moments_npz_path(case: str) -> Path:
    return C.case_dir(case) / "moments_cell_time.npz"


def compute_case(case: str) -> dict[str, np.ndarray]:
    u05, v05 = C.load_case_wind(case, "d05")
    fine = M.fine_moments(u05, v05, 0, C.RAW_N)
    # d05 domain-mean context (used by wind-state and stratification phases)
    dm_speed = np.zeros(C.RAW_N)
    dm_u = np.zeros(C.RAW_N)
    dm_v = np.zeros(C.RAW_N)
    for c0 in range(0, C.RAW_N, 512):
        c1 = min(c0 + 512, C.RAW_N)
        uu = u05[c0:c1].astype(np.float64)
        vv = v05[c0:c1].astype(np.float64)
        dm_speed[c0:c1] = np.hypot(uu, vv).mean(axis=(1, 2))
        dm_u[c0:c1] = uu.mean(axis=(1, 2))
        dm_v[c0:c1] = vv.mean(axis=(1, 2))
    C.free_case_wind(case)

    u02, v02 = C.load_case_wind(case, "d02")
    par = M.parent_vectors_d02(u02, v02, 0, C.RAW_N)
    C.free_case_wind(case)

    tab = {
        "s_bar": fine["s_bar"], "u_bar": fine["u_bar"], "v_bar": fine["v_bar"],
        "U_vec": fine["U_vec"], "J": fine["J"], "S": fine["S"],
        "var_par": fine["var_par"], "var_perp": fine["var_perp"],
        "trC": fine["trC"], "J2nd": fine["J2nd"],
        "u02p": par["u02p"], "v02p": par["v02p"], "U02_mag": par["mag"],
        "delta_dyn": fine["U_vec"] - par["mag"],
        "delta_scalar": fine["s_bar"] - par["mag"],
        "d05_domain_speed": dm_speed, "d05_domain_u": dm_u, "d05_domain_v": dm_v,
    }
    # exact-decomposition and J >= 0 checks
    ident = float(np.abs(tab["delta_scalar"] - (tab["delta_dyn"] + tab["J"])).max())
    jmin = float(tab["J"].min())
    assert ident < 1e-9, f"case {case}: decomposition identity violated ({ident})"
    assert jmin > -1e-12, f"case {case}: negative Jensen gap ({jmin})"
    tab["_identity_max_abs"] = np.float64(ident)
    tab["_J_min"] = np.float64(jmin)
    return tab


def load_moments(case: str) -> dict[str, np.ndarray]:
    with np.load(moments_npz_path(case)) as z:
        return {k: z[k] for k in z.files}


def summarize(case: str, tab: dict) -> None:
    cdir = C.case_dir(case)
    qlist = [5, 25, 50, 75, 95]

    # long-format per-cell summaries over the three windows
    rows = []
    for wname, (a, b) in WINDOWS.items():
        for mom in MOMENTS + ["s_bar", "U_vec", "U02_mag", "J2nd"]:
            arr = tab[mom][a:b]
            qs = np.percentile(arr, qlist, axis=0)
            for cid in range(C.N_PARENT):
                rows.append({
                    "window": wname, "moment": mom, "cell_id": cid,
                    "mean": arr[:, cid].mean(), "median": np.median(arr[:, cid]),
                    "sd": arr[:, cid].std(ddof=0),
                    "iqr": qs[3, cid] - qs[1, cid],
                    **{f"p{q:02d}": qs[i, cid] for i, q in enumerate(qlist)},
                })
    df = pd.DataFrame(rows).round(8)
    C.write_df(cdir / "moments_cell_time_summary.csv", df, case=case)

    for wname, out_name in [("calibration", "moments_cell_means_calibration.csv"),
                            ("evaluation", "moments_cell_means_evaluation.csv")]:
        a, b = WINDOWS[wname]
        rec = {"cell_id": np.arange(C.N_PARENT),
               "row": np.arange(C.N_PARENT) // 6,
               "col": np.arange(C.N_PARENT) % 6}
        for mom in MOMENTS + ["s_bar", "U_vec", "U02_mag", "J2nd", "var_perp", "trC"]:
            rec[f"{mom}_mean"] = tab[mom][a:b].mean(axis=0)
        C.write_df(cdir / out_name, pd.DataFrame(rec).round(9), case=case)

    dom = {}
    for wname, (a, b) in WINDOWS.items():
        cm = {m: tab[m][a:b].mean(axis=0) for m in MOMENTS}
        dm = {m: float(cm[m].mean()) for m in MOMENTS}
        dom[wname] = {
            "n_states": b - a,
            "domain_mean": dm,
            "J_over_delta_scalar": (dm["J"] / dm["delta_scalar"]
                                    if abs(dm["delta_scalar"]) > 1e-12 else None),
            "cell_range": {m: [float(cm[m].min()), float(cm[m].max())]
                           for m in MOMENTS},
            "identity_residual_domain": float(
                abs(dm["delta_scalar"] - (dm["delta_dyn"] + dm["J"]))),
        }
    C.write_json(cdir / "moments_domain_summary.json", {
        "windows": dom,
        "identity_max_abs_cell_time": float(tab["_identity_max_abs"]),
        "J_min_cell_time": float(tab["_J_min"]),
        "estimand_contract": {
            "S": "population SD of the 729 fine-cell speeds, ddof=0",
            "identity": "delta_scalar = delta_dyn + J (exact)",
            "n_fine_per_parent": C.N_FINE},
    }, case=case)


def jensen_second_order(case: str, tab: dict) -> None:
    """PLAN §8 outputs with the frozen Case-A bins (identical across cases)."""
    cdir = C.case_dir(case)
    a0, b0 = C.RETAINED_SLICE
    # cell-time CSV over the retained window
    n = b0 - a0
    idx_state = np.repeat(np.arange(a0, b0), C.N_PARENT)
    idx_cell = np.tile(np.arange(C.N_PARENT), n)
    cols = {"state_index": idx_state, "cell_id": idx_cell}
    for k, name in [("J", "J_exact"), ("J2nd", "J2nd"), ("trC", "trC"),
                    ("var_perp", "sigma_perp2"), ("U_vec", "u_bar_mag")]:
        cols[name] = tab[k][a0:b0].reshape(-1)
    df = pd.DataFrame(cols)
    df["perp_share_of_trC"] = np.where(df["trC"] > 1e-15,
                                       df["sigma_perp2"] / df["trC"], np.nan)
    df["J2nd_minus_J"] = df["J2nd"] - df["J_exact"]
    df["window"] = np.where(df["state_index"] < C.CAL_SLICE[1],
                            "calibration", "evaluation")
    C.write_df(cdir / "jensen_second_order_cell_time.csv", df.round(8), case=case)

    # frozen speed bins on |u_bar| (cell-time records), per window
    edges = C.SPEED_BIN_EDGES
    bin_rows, grp_rows = [], []
    for wname, (a, b) in [("calibration", C.CAL_SLICE), ("evaluation", C.EVAL_SLICE)]:
        u = tab["U_vec"][a:b].reshape(-1)
        j = tab["J"][a:b].reshape(-1)
        j2 = tab["J2nd"][a:b].reshape(-1)
        err = j2 - j
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = (u >= lo) & (u < hi)
            if m.sum() == 0:
                continue
            lab = f"{lo}-{hi}" if np.isfinite(hi) else f"{lo}+"
            bin_rows.append({
                "window": wname, "bin_label": lab, "bin_lo": lo,
                "bin_hi": (hi if np.isfinite(hi) else np.inf),
                "n_records": int(m.sum()),
                "J_mean": j[m].mean(), "J2nd_mean": j2[m].mean(),
                "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                "bias": float(err[m].mean()),
                "err_q05": np.percentile(err[m], 5),
                "err_median": np.percentile(err[m], 50),
                "err_q95": np.percentile(err[m], 95),
            })
        for gname, m in [("below", u < C.LOW_WIND_MS), ("above", u >= C.LOW_WIND_MS)]:
            n_m = int(m.sum())
            r = M.pearson(j[m], j2[m]) if n_m > 2 else np.nan
            grp_rows.append({
                "window": wname, "wind_group": gname,
                "threshold_ms": C.LOW_WIND_MS,
                "threshold_role": "descriptive reference only (frozen from Case A)",
                "n_records": n_m,
                "record_fraction": float(m.mean()),
                "rmse": (float(np.sqrt(np.mean(err[m] ** 2))) if n_m else np.nan),
                "bias": (float(err[m].mean()) if n_m else np.nan),
                "corr": float(r),
            })
        # window-level summary appended to groups
        cm_j = tab["J"][a:b].mean(axis=0)
        cm_j2 = tab["J2nd"][a:b].mean(axis=0)
        crosswind = float(np.mean(tab["var_perp"][a:b]) /
                          max(np.mean(tab["trC"][a:b]), 1e-15))
        grp_rows.append({
            "window": wname, "wind_group": "all",
            "threshold_ms": np.nan, "threshold_role": "",
            "n_records": int((b - a) * C.N_PARENT),
            "record_fraction": 1.0,
            "rmse": float(np.sqrt(np.mean((j2 - j) ** 2))),
            "bias": float(np.mean(j2 - j)),
            "corr": M.pearson(j, j2),
            "domain_mean_J": float(cm_j.mean()),
            "domain_mean_J2nd": float(cm_j2.mean()),
            "cellmean_corr": M.pearson(cm_j, cm_j2),
            "crosswind_share_of_trC": crosswind,
        })
    C.write_df(cdir / "jensen_second_order_bins.csv",
               pd.DataFrame(bin_rows).round(8), case=case)
    C.write_df(cdir / "jensen_second_order_groups.csv",
               pd.DataFrame(grp_rows).round(8), case=case)


def main(case: str) -> None:
    print(f"[phase01-03] case {case}: moments + second-order diagnostic")
    tab = compute_case(case)
    np.savez_compressed(moments_npz_path(case),
                        **{k: v for k, v in tab.items()})
    C.write_json(moments_npz_path(case).with_suffix(".metadata.json"), {
        "artifact": "moments_cell_time.npz",
        "shape_contract": "full 4320-state x 36-parent moment arrays",
    }, case=case)
    summarize(case, tab)
    jensen_second_order(case, tab)
    C.run_receipt(f"phase01_moments_case_{case}", {
        "case": case, "data_status": C.data_status(case),
        "identity_max_abs": float(tab["_identity_max_abs"]),
        "J_min": float(tab["_J_min"])})


if __name__ == "__main__":
    for cs in (sys.argv[1:] or ["A"]):
        main(cs)
