"""Assemble the information-source comparison table and the NPJ claim ledger.

Combines the sealed evaluation scores (HOLDOUT_BASELINE_SCORES.csv +
error_decomposition.csv) with the new interpolation baselines
(npj_refactor/interpolation_metrics.csv) into one canonical table:

  information_sources.csv  (Moment | Method | Information class |
                           Information used | RMSE | Centred RMSE | Bias |
                           r | Slope | All-record RMSE)

and writes npj_claim_ledger.csv listing the evaluation-period claims the NPJ
manuscript makes, each tied to its evidence file.

Old evidence files are only read, never modified.
"""
from __future__ import annotations

import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "refactor_20260807"))
import common  # noqa: E402

OUT = os.path.join(common.PROJECT_ROOT, "source", "evidence", "npj_refactor")

INFO_CLASS = {
    "zero_correction": "no fine-scale information",
    "global_training_mean": "no fine-scale information",
    "bilinear": "geometric interpolation",
    "bicubic": "geometric interpolation",
    "terrain_roughness_equation": "static terrain",
    "roughness_only": "static terrain",
    "exposure_only": "static terrain",
    "locked_joint_equation": "static terrain",
    "cell_mean_persistence": "fine-grid history",
}

INFO_USED = {
    "zero_correction": "none",
    "global_training_mean": "one calibration-period domain mean",
    "bilinear": "instantaneous d02 U and V, no fitted parameters",
    "bicubic": "instantaneous d02 U and V, no fitted parameters",
    "terrain_roughness_equation": "static subgrid terrain and two fitted coefficients",
    "roughness_only": "static subgrid terrain and two fitted coefficients",
    "exposure_only": "static subgrid terrain and two fitted coefficients",
    "locked_joint_equation": "static subgrid terrain and fitted coefficients",
    "cell_mean_persistence": "36 calibration-period d05 cell moments",
}

ROWS_ORDER_MEAN = ["zero_correction", "global_training_mean", "bilinear",
                   "bicubic", "terrain_roughness_equation", "cell_mean_persistence"]
ROWS_ORDER_SPREAD = ["global_training_mean", "bilinear", "bicubic",
                     "roughness_only", "exposure_only", "locked_joint_equation",
                     "cell_mean_persistence"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    scores = pd.read_csv(os.path.join(
        common.PROJECT_ROOT, "source", "evidence", "HOLDOUT_BASELINE_SCORES.csv"))
    ed = pd.read_csv(os.path.join(
        common.PROJECT_ROOT, "source", "evidence", "refactor_20260807",
        "error_decomposition.csv"))
    im = pd.read_csv(os.path.join(OUT, "interpolation_metrics.csv"))

    def crmse_of(group, model):
        row = ed[(ed.group == group) & (ed.model == model)]
        return None if row.empty else float(row["crmse"].iloc[0])

    rows = []
    for group, models, group_name in [
            ("mean", ROWS_ORDER_MEAN, "scalar_increment"),
            ("spread", ROWS_ORDER_SPREAD, "spread")]:
        for m in models:
            if m in ("bilinear", "bicubic"):
                mm = im[(im.group == group_name) & (im.model == m)]
                rmse, bias, r, slope, ar = (float(mm["rmse"].iloc[0]),
                                            float(mm["bias"].iloc[0]),
                                            float(mm["r"].iloc[0]),
                                            float(mm["slope"].iloc[0]),
                                            float(mm["all_record_rmse"].iloc[0]))
                crmse = float(mm["crmse"].iloc[0])
            else:
                s = scores[(scores.group == group) & (scores.name == m)]

                def num(v):
                    return None if str(v).strip() == "--" else float(v)

                rmse = num(s["cell_rmse"].iloc[0])
                bias = num(s["cell_bias"].iloc[0])
                r = num(s["cell_corr"].iloc[0])
                slope = num(s["cell_slope"].iloc[0])
                ar = num(s["scalar_rmse"].iloc[0])
                crmse = crmse_of(group_name, m)
            # full precision from the sealed sources (round 2: the earlier
            # round(., 4) created 0.001 rounding artefacts, e.g. crmse 0.191460
            # -> 0.1915 -> 0.192, that contradicted the sealed 3-decimal values)
            rows.append({
                "group": group_name, "method": m,
                "info_class": INFO_CLASS[m],
                "information_used": INFO_USED[m],
                "rmse": rmse, "crmse": crmse,
                "bias": bias, "r": r, "slope": slope,
                "all_record_rmse": ar,
            })

    with open(os.path.join(OUT, "information_sources.csv"), "w",
              encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- claim ledger (evaluation-period claims of the NPJ manuscript) ----
    claims = [
        ("C-N01", "Evaluation-period scalar-increment RMSE of the bilinear "
                  "and bicubic interpolation baselines (36 cell means)",
         "source_data/interpolation_metrics.csv"),
        ("C-N02", "Evaluation-period within-parent spread RMSE of the "
                  "interpolation baselines",
         "source_data/interpolation_metrics.csv"),
        ("C-N03", "Domain-mean moment recovery ratios Q_delta, Q_J, Q_S for "
                  "bilinear and bicubic (36 cells and inner 4x4)",
         "source_data/interpolation_recovery.csv"),
        ("C-N04", "Mean vector-preservation error and domain-mean delta_dyn "
                  "of the interpolated fields",
         "source_data/interpolation_metrics.csv"),
        ("C-N05", "Interpolation contract frozen before evaluation; synthetic "
                  "tests, parent-centre alignment, edge extension, "
                  "representative state, frozen-contract sha256",
         "source_data/interpolation_contract.json"),
        ("C-N06", "Time-varying paired block bootstrap of terrain vs "
                  "bilinear/bicubic (scalar) and joint vs bilinear/bicubic "
                  "(spread); interpolation predictions recomputed per "
                  "replicate; code path validated against the sealed five pairs",
         "source_data/interpolation_bootstrap.csv"),
        ("C-N07", "Information-source comparison across no fine-scale "
                  "information, geometric interpolation, static terrain and "
                  "fine-grid history",
         "source_data/information_sources.csv"),
    ]
    with open(os.path.join(OUT, "npj_claim_ledger.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "claim", "evidence_file"])
        w.writerows(claims)

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nnpj_claim_ledger rows:", len(claims))


if __name__ == "__main__":
    main()
