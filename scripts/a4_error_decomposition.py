"""a4: Evaluation-period error decomposition (plan 4.4).

From the 36-cell evaluation time-mean table (model_cells), decompose each
model's error into domain-mean bias and centred spatial pattern error:

    raw RMSE over 36 cells
    domain-mean bias  B = mean(pred) - mean(obs)
    centred RMSE (CRMSE) = sqrt(mean(((pred-mean pred) - (obs-mean obs))^2))
    correlation r
    evaluation slope (OLS pred on obs)
    calibration slope (train-period fitted vs observed, from F05_F06)

Outputs:
  error_decomposition.csv / .json
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import common

OUT = common.EVIDENCE_DIR
os.makedirs(OUT, exist_ok=True)

MODELS_MEAN = ["zero_correction", "global_training_mean", "cell_mean_persistence",
               "terrain_roughness_equation"]
MODELS_SPREAD = ["global_training_mean", "cell_mean_persistence", "roughness_only",
                 "exposure_only", "locked_joint_equation"]


def metrics(p, o):
    p = np.asarray(p, float)
    o = np.asarray(o, float)
    raw = np.sqrt(np.mean((p - o) ** 2))
    bias = float(np.mean(p - o))
    pc, oc = p - p.mean(), o - o.mean()
    crmse = float(np.sqrt(np.mean((pc - oc) ** 2)))
    r = float((pc * oc).sum() / np.sqrt((pc * pc).sum() * (oc * oc).sum() + 1e-30))
    slope = float(np.polyfit(o, p, 1)[0])
    return dict(rmse=float(raw), bias=bias, crmse=crmse, r=r, slope=slope)


def main():
    mc = pd.read_csv(os.path.join(common.PROJECT_ROOT,
        "source/evidence/source_snapshots/outputs_evaluation_holdout_model_cells.csv"))
    f56 = pd.read_csv(os.path.join(common.PROJECT_ROOT,
        "source/figure_data/main/F05_F06_terrain_parameterization.csv"))
    rows = []
    for group, models, obs_col, pred_col in [
        ("scalar_increment", MODELS_MEAN, "mean_observed", None),
        ("spread", MODELS_SPREAD, "spread_observed", None),
    ]:
        obs = mc[obs_col].values
        for m in models:
            col = f"{group}_{m}_predicted" if group == "spread" else f"mean_{m}_predicted"
            pred = mc[col].values
            met = metrics(pred, obs)
            # calibration slope: training fitted vs observed per cell
            if m == "zero_correction":
                cal_slope = 0.0
            elif m in ("terrain_roughness_equation", "roughness_only", "exposure_only",
                       "locked_joint_equation"):
                col = "mean_predicted_train" if group == "scalar_increment" else "spread_predicted_train"
                colo = "mean_observed_train" if group == "scalar_increment" else "spread_observed_train"
                cal_slope = float(np.polyfit(f56[colo], f56[col], 1)[0])
            else:
                cal_slope = float("nan")
            rows.append(dict(group=group, model=m, n_cells=36,
                             rmse=met["rmse"], bias=met["bias"], crmse=met["crmse"],
                             r=met["r"], slope=met["slope"], calibration_slope=cal_slope,
                             pred_mean=float(pred.mean()), obs_mean=float(obs.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/error_decomposition.csv", index=False)
    out = {
        "scalar_common_bias": float(df[df.group == "scalar_increment"].bias.mode().iloc[0]),
        "spread_common_bias": float(df[df.group == "spread"].bias.mode().iloc[0]),
        "rows": df.to_dict("records"),
    }
    with open(f"{OUT}/error_decomposition.json", "w") as f:
        json.dump(out, f, indent=1)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
