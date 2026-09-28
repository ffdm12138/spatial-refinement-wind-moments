"""Phase 4 — scale dependence (PLAN §9).

Vector-average U/V over k x k blocks FIRST (k in {1, 3, 9, 27} = 111.11 /
333.33 / 1000 / 3000 m), then compute scalar quantities within each parent.
At 3000 m J = S = 0 must hold exactly by construction (asserted).

Outputs per case: scale_cell_time.npz (bulk), scale_cell_time.csv (retained
window), scale_cell_means.csv, scale_summary.csv, scale_normalized_by_cell.csv,
scale_rank_persistence.csv.
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

WINDOWS = {"calibration": C.CAL_SLICE, "evaluation": C.EVAL_SLICE}


def scale_npz_path(case: str) -> Path:
    return C.case_dir(case) / "scale_cell_time.npz"


def main(case: str) -> None:
    print(f"[phase04] case {case}: scale dependence")
    cdir = C.case_dir(case)
    tab = P1.load_moments(case)
    u05, v05 = C.load_case_wind(case, "d05")

    bulk = {}
    for k in C.SCALE_CELLS:
        sm = M.scale_moments(u05, v05, 0, C.RAW_N, k)
        bulk[f"J_{k}"] = sm["J_l"]
        bulk[f"S_{k}"] = sm["S_l"]
        bulk[f"s_bar_{k}"] = sm["s_bar_l"]
        # scale-independent parent mean-vector magnitude must agree with the
        # k=1 fine computation (vector aggregation is exactly nested)
        dev = float(np.abs(sm["U_vec"] - tab["U_vec"]).max())
        assert dev < 1e-8, f"case {case} k={k}: nested vector-mean violated ({dev})"
    C.free_case_wind(case)

    # structural endpoint: J = S = 0 exactly at 3000 m
    j27 = float(np.abs(bulk["J_27"]).max())
    s27 = float(np.abs(bulk["S_27"]).max())
    assert j27 < 1e-9 and s27 < 1e-9, \
        f"case {case}: 3000 m endpoint not exact (J {j27}, S {s27})"

    np.savez_compressed(scale_npz_path(case), **bulk,
                        delta_dyn=tab["delta_dyn"], U02_mag=tab["U02_mag"])
    C.write_json(scale_npz_path(case).with_suffix(".metadata.json"), {
        "artifact": "scale_cell_time.npz",
        "shape_contract": "full 4320-state scale moment arrays",
        "scales_m": {str(k): C.SCALE_M[k] for k in C.SCALE_CELLS},
    }, case=case)

    # ---- cell-time CSV over the retained window ----
    a0, b0 = C.RETAINED_SLICE
    n = b0 - a0
    frames = []
    for k in C.SCALE_CELLS:
        frames.append(pd.DataFrame({
            "scale_m": C.SCALE_M[k],
            "state_index": np.repeat(np.arange(a0, b0), C.N_PARENT),
            "cell_id": np.tile(np.arange(C.N_PARENT), n),
            "J": bulk[f"J_{k}"][a0:b0].reshape(-1),
            "S": bulk[f"S_{k}"][a0:b0].reshape(-1),
            "delta_dyn": tab["delta_dyn"][a0:b0].reshape(-1),
            "delta_scalar": (tab["delta_dyn"][a0:b0]
                             + bulk[f"J_{k}"][a0:b0]).reshape(-1),
        }))
    ct = pd.concat(frames, ignore_index=True).round(6)
    C.write_df(cdir / "scale_cell_time.csv", ct, case=case)
    del ct, frames

    # ---- cell means (calibration + evaluation), normalized, ranks ----
    mean_rows, norm_rows, rank_rows, summ_rows = [], [], [], []
    for wname, (a, b) in WINDOWS.items():
        cm = {}
        for k in C.SCALE_CELLS:
            cm[k] = {
                "J": bulk[f"J_{k}"][a:b].mean(axis=0),
                "S": bulk[f"S_{k}"][a:b].mean(axis=0),
                "s_bar": bulk[f"s_bar_{k}"][a:b].mean(axis=0),
            }
        dd = tab["delta_dyn"][a:b].mean(axis=0)
        for k in C.SCALE_CELLS:
            for cid in range(C.N_PARENT):
                mean_rows.append({
                    "window": wname, "scale_m": C.SCALE_M[k], "cell_id": cid,
                    "row": cid // 6, "col": cid % 6,
                    "J": cm[k]["J"][cid], "S": cm[k]["S"][cid],
                    "delta_dyn": dd[cid],
                    "delta_scalar": dd[cid] + cm[k]["J"][cid],
                    "scale_scalar_mean": cm[k]["s_bar"][cid],
                    "J_111": cm[1]["J"][cid], "S_111": cm[1]["S"][cid],
                })
                jn = (cm[k]["J"][cid] / cm[1]["J"][cid]
                      if cm[1]["J"][cid] > 1e-12 else np.nan)
                sn = (cm[k]["S"][cid] / cm[1]["S"][cid]
                      if cm[1]["S"][cid] > 1e-12 else np.nan)
                norm_rows.append({
                    "window": wname, "scale_m": C.SCALE_M[k], "cell_id": cid,
                    "J_norm": jn, "S_norm": sn,
                })
            rank_rows.append({
                "window": wname, "scale_m": C.SCALE_M[k],
                "spearman_J_vs_111": (M.spearman(cm[k]["J"], cm[1]["J"])
                                      if k != 27 else np.nan),
                "spearman_S_vs_111": (M.spearman(cm[k]["S"], cm[1]["S"])
                                      if k != 27 else np.nan),
                "note": ("rank undefined at the 3000 m structural endpoint "
                         "(J = S = 0)" if k == 27 else ""),
            })
            summ_rows.append({
                "window": wname, "scale_m": C.SCALE_M[k],
                "J_domain_mean": float(cm[k]["J"].mean()),
                "S_domain_mean": float(cm[k]["S"].mean()),
                "J_domain_median": float(np.median(cm[k]["J"])),
                "S_domain_median": float(np.median(cm[k]["S"])),
                "J_norm_domain": (float(cm[k]["J"].mean() / cm[1]["J"].mean())
                                  if cm[1]["J"].mean() > 0 else np.nan),
                "S_norm_domain": (float(cm[k]["S"].mean() / cm[1]["S"].mean())
                                  if cm[1]["S"].mean() > 0 else np.nan),
                "delta_dyn_domain_mean": float(dd.mean()),
                "delta_scalar_domain_mean": float(dd.mean() + cm[k]["J"].mean()),
            })
    C.write_df(cdir / "scale_cell_means.csv",
               pd.DataFrame(mean_rows).round(9), case=case)
    C.write_df(cdir / "scale_normalized_by_cell.csv",
               pd.DataFrame(norm_rows).round(8), case=case)
    C.write_df(cdir / "scale_rank_persistence.csv",
               pd.DataFrame(rank_rows).round(6), case=case)
    C.write_df(cdir / "scale_summary.csv",
               pd.DataFrame(summ_rows).round(8), case=case)
    # A case-level scale summary is the only legal source for the master M4
    # table.  Keep the complete four-scale contract explicit at the producer,
    # so a later assembler cannot silently inherit a 1000/3000-only subset.
    scale_summary = pd.DataFrame(summ_rows)
    assert len(scale_summary) == len(WINDOWS) * len(C.SCALE_CELLS)
    expected_scales = {float(C.SCALE_M[k]) for k in C.SCALE_CELLS}
    for window in WINDOWS:
        got = set(scale_summary.loc[scale_summary.window == window, "scale_m"].astype(float))
        assert got == expected_scales, (
            f"case {case} {window}: scale contract mismatch; got {sorted(got)}, "
            f"expected {sorted(expected_scales)}")
    C.run_receipt(f"phase04_scale_case_{case}", {
        "case": case, "data_status": C.data_status(case),
        "endpoint_J_max_abs": j27, "endpoint_S_max_abs": s27,
        "scale_rows": len(scale_summary),
        "scales_m": sorted(expected_scales)})


if __name__ == "__main__":
    for cs in (sys.argv[1:] or ["A"]):
        main(cs)
