"""Phase 15 — frozen scale-aware closure transfer to B/C (PLAN §21).

Applies the Case-A FROZEN scale-aware closures (coefficients from
source/evidence/npj_refactor/scale_closure/scale_closure_lock.json; never
refit on B/C):

    J_hat(l, p) = G(l; alpha_J) * (a_J + b_J * r_sigma_p)
    S_hat(l, p) = a_S * G(l; alpha_S0) + c_S * lambda_iso_p * G(l; alpha_ST)
    G(l; alpha) = [(1 - l/L) / (1 - l0/L)]^alpha,  L = 3000 m, l0 = 111.11 m

at 111.11 / 333.33 / 1000 m against each case's evaluation-period scale cell
means.  Case A rows serve as the legacy reproduction cross-check against the
frozen scale_aware_transfer_metrics.csv.  Every B/C application carries
in_range / extrapolation flags on the consumed descriptors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import mc_moments as M  # noqa: E402
import phase07_terrain as P7  # noqa: E402


def closure_predictions(case: str) -> dict:
    lock = C.frozen_scale_lock()
    aJ, bJ = lock["model_J"]["beta"]
    alphaJ = lock["model_J"]["alpha"]
    aS, cS = lock["model_S"]["beta"]
    aS0 = lock["model_S"]["alpha0"]
    aST = lock["model_S"]["alpha_t"]
    desc = P7.descriptors(case)
    r_sigma = desc["r_sigma"].to_numpy(float)
    lam = desc["lambda_iso"].to_numpy(float)
    out = {}
    for k in (1, 3, 9):
        ell = C.SCALE_M[k]
        out[("J", ell)] = C.g_scale(ell, alphaJ) * (aJ + bJ * r_sigma)
        out[("S", ell)] = (aS * C.g_scale(ell, aS0)
                           + cS * lam * C.g_scale(ell, aST))
    return out


def main(cases: list[str]) -> None:
    print(f"[phase15] frozen scale-aware closure transfer: {cases}")
    a_ranges = C.frozen_a_calibration_ranges()
    all_rows = {}
    for case in cases:
        pred = closure_predictions(case)
        cm = pd.read_csv(C.case_dir(case) / "scale_cell_means.csv")
        cm = cm[cm.window == "evaluation"]
        desc = P7.descriptors(case)
        flags = {
            "J": ((desc["r_sigma"] >= a_ranges["r_sigma"][0])
                  & (desc["r_sigma"] <= a_ranges["r_sigma"][1])),
            "S": ((desc["lambda_iso"] >= a_ranges["lambda_iso"][0])
                  & (desc["lambda_iso"] <= a_ranges["lambda_iso"][1])),
        }
        rows = []
        for moment in ("J", "S"):
            for k in (1, 3, 9):
                ell = C.SCALE_M[k]
                sub = cm[np.isclose(cm.scale_m, ell)].sort_values("cell_id")
                obs = sub[moment].to_numpy(float)
                p = pred[(moment, ell)]
                met = M.cell_metrics(p, obs)
                rows.append({
                    "case": case, "moment": moment, "scale_m": ell, "n": 36,
                    "r": met["r"], "rmse": met["rmse"], "bias": met["bias"],
                    "crmse": met["crmse"], "slope": met["slope"],
                    "spearman": M.spearman(p, obs),
                    "source": "unchanged 2680-fitted closure (Case A frozen)",
                    "segment": "final 670 states",
                    "n_cells_extrapolating": int((~flags[moment]).sum()),
                    "extrapolation_flag": bool((~flags[moment]).any()),
                })
        df = pd.DataFrame(rows).round(9)
        all_rows[case] = df
        if case in ("B", "C"):
            C.write_df(C.ext_dir("cross_case")
                       / f"frozen_scale_closure_transfer_{case}.csv",
                       df, status_value=C.combined_status("A", case))
        else:
            C.write_df(C.ext_dir("cross_case")
                       / "frozen_scale_closure_transfer_A_reproduction.csv",
                       df, case="A")
    combined = pd.concat(all_rows.values(), ignore_index=True)
    C.write_df(C.ext_dir("tables") / "M11_frozen_scale_closure_transfer.csv",
               combined)
    C.run_receipt("phase15_scaleclosure", {"cases": cases})


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
