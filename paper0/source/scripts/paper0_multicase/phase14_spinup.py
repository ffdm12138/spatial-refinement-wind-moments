"""Phase 14 — matched-window / spin-up cutoff robustness (PLAN §5.1, §20).

Pre-registered cutoffs {6h=360, 12h=720, 970min=970, 18h=1080, 24h=1440}
(1-min cadence).  970 is ALWAYS primary; the sensitivity never selects the
window, is never used to pick the best result, and reports direction and
magnitude for every cutoff (summary level only — no terrain refits).

Definitions (declared in ANALYSIS_CONTRACT.json before any science output):
  * headline J / delta_dyn / delta_scalar / S at cutoff c: domain means over
    the retained window [c, 4320) and over the calibration analogue
    [c, 3650);
  * Q_J / Q_S at cutoff c: domain-mean ratios over the retained window
    [c, 4320) (sensitivity-only definition; the PRIMARY Q_J / Q_S remain the
    frozen evaluation-only [3650, 4320) values, repeated in the table for
    reference — they do not depend on the cutoff by construction).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import phase01_moments as P1  # noqa: E402
import phase05_interpolation as P5  # noqa: E402


def main(cases: list[str]) -> None:
    print(f"[phase14] spin-up sensitivity for cases {cases}")
    rows = []
    for case in cases:
        tab = P1.load_moments(case)
        with np.load(P5.interp_npz_path(case)) as z:
            interp = {(m, k): z[f"{m}_{k}"]
                      for m in ("bilinear", "bicubic") for k in ("J", "spread")}
        ea, eb = C.EVAL_SLICE
        ref_eval = {"J": tab["J"][ea:eb].mean(), "S": tab["S"][ea:eb].mean()}
        q_eval = {f"Q_{q}_{m}": interp[(m, k)][ea:eb].mean() / ref_eval[q]
                  for m in ("bilinear", "bicubic")
                  for q, k in [("J", "J"), ("S", "spread")]}
        primary = {}
        for name, cut in C.SPINUP_CUTOFFS.items():
            ret = slice(cut, C.RAW_N)
            cal = slice(cut, C.CAL_SLICE[1])
            row = {
                "case": case, "cutoff": name, "cutoff_states": cut,
                "is_primary": name == C.PRIMARY_CUTOFF,
                "n_retained": C.RAW_N - cut,
                "J_retained": float(tab["J"][ret].mean()),
                "delta_dyn_retained": float(tab["delta_dyn"][ret].mean()),
                "delta_scalar_retained": float(tab["delta_scalar"][ret].mean()),
                "S_retained": float(tab["S"][ret].mean()),
                "J_calibration": float(tab["J"][cal].mean()),
                "delta_dyn_calibration": float(tab["delta_dyn"][cal].mean()),
                "delta_scalar_calibration": float(tab["delta_scalar"][cal].mean()),
                "S_calibration": float(tab["S"][cal].mean()),
            }
            jr = float(tab["J"][ret].mean())
            sr = float(tab["S"][ret].mean())
            for m in ("bilinear", "bicubic"):
                row[f"Q_J_{m}_retained_sensitivity"] = float(
                    interp[(m, "J")][ret].mean() / jr)
                row[f"Q_S_{m}_retained_sensitivity"] = float(
                    interp[(m, "spread")][ret].mean() / sr)
                row[f"Q_J_{m}_eval_primary"] = float(q_eval[f"Q_J_{m}"])
                row[f"Q_S_{m}_eval_primary"] = float(q_eval[f"Q_S_{m}"])
            rows.append(row)
            if name == C.PRIMARY_CUTOFF:
                primary = row
        # mechanical qualitative-change flags vs the primary cutoff
        for row in rows:
            if row["case"] != case:
                continue
            row["J_sign_change_vs_primary"] = (
                np.sign(row["J_retained"]) != np.sign(primary["J_retained"]))
            row["delta_dyn_sign_change_vs_primary"] = (
                np.sign(row["delta_dyn_retained"])
                != np.sign(primary["delta_dyn_retained"]))
            row["J_rel_change_vs_primary"] = (
                (row["J_retained"] - primary["J_retained"])
                / abs(primary["J_retained"]) if primary["J_retained"] else np.nan)
            row["S_rel_change_vs_primary"] = (
                (row["S_retained"] - primary["S_retained"])
                / abs(primary["S_retained"]) if primary["S_retained"] else np.nan)
    df = pd.DataFrame(rows).round(9)
    df["primary_note"] = ("primary window is always 970; sensitivity cutoffs "
                          "report direction/magnitude only and never select "
                          "the window")
    C.write_df(C.ext_dir("audit") / "spinup_sensitivity.csv", df)
    C.write_df(C.ext_dir("tables") / "M12_spinup_sensitivity.csv", df)
    C.run_receipt("phase14_spinup", {"cases": cases,
                                     "cutoffs": C.SPINUP_CUTOFFS})


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
