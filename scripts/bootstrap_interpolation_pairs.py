"""Time-varying paired moving-block bootstrap including the interpolation
baselines (NPJ manuscript).

Differs from the sealed evaluation bootstrap (a5) only in that the bilinear /
bicubic predictions are time-varying (670 x 36 per state), so their block-mean
prediction is recomputed for every replicate:

    mu_hat^{I,*}_p = (1/N*) sum_{t in T*} y_hat^I_{p,t}

Terrain / global / persistence remain static period-mean predictors. All
methods share the identical resampled time blocks (paired design, seed
20260807, 2000 replicates, 60/120/240 min).

The original five pairs are recomputed with the same code path and validated
against the sealed paired_bootstrap.csv; the old evidence file and script are
NOT modified (hash-checked separately).

Output: source/evidence/npj_refactor/interpolation_bootstrap.csv
         (8 figure pairs x 3 block lengths)
         + printed validation of the original 5 pairs vs the sealed values.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "refactor_20260807"))
import common  # noqa: E402
from a5_paired_bootstrap import block_resample, cell_rmse, RNG_SEED, \
    N_REPLICATES, BLOCK_LENGTHS  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "source", "evidence", "npj_refactor")
SEALED_BOOT = os.path.join(PROJECT_ROOT, "source", "evidence",
                           "refactor_20260807", "paired_bootstrap.csv")

# 8 figure pairs: scalar group (terrain vs 4 information sources) and
# spread group (joint vs 4 information sources)
PAIRS = [
    ("terrain_vs_global", "scalar_increment",
     "terrain_roughness_equation", "global_training_mean"),
    ("terrain_vs_bilinear", "scalar_increment",
     "terrain_roughness_equation", "bilinear"),
    ("terrain_vs_bicubic", "scalar_increment",
     "terrain_roughness_equation", "bicubic"),
    ("terrain_vs_persistence", "scalar_increment",
     "terrain_roughness_equation", "cell_mean_persistence"),
    ("joint_vs_global", "spread", "locked_joint_equation", "global_training_mean"),
    ("joint_vs_bilinear", "spread", "locked_joint_equation", "bilinear"),
    ("joint_vs_bicubic", "spread", "locked_joint_equation", "bicubic"),
    ("joint_vs_persistence", "spread", "locked_joint_equation", "cell_mean_persistence"),
]
# Original sealed pairs, recomputed here ONLY to validate the code path
OLD_PAIRS = [
    ("terrain_vs_global", "scalar_increment",
     "terrain_roughness_equation", "global_training_mean"),
    ("terrain_vs_persistence", "scalar_increment",
     "terrain_roughness_equation", "cell_mean_persistence"),
    ("joint_vs_global", "spread", "locked_joint_equation", "global_training_mean"),
    ("joint_vs_persistence", "spread", "locked_joint_equation", "cell_mean_persistence"),
    ("joint_vs_exposure_only", "spread", "locked_joint_equation", "exposure_only"),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    A = common.state_arrays("eval")
    obs = {"scalar_increment": A["delta_scalar"], "spread": A["spread"]}  # (670, 36)
    n = obs["scalar_increment"].shape[0]
    assert n == 670

    # ---- static predictions from the sealed holdout table ----
    mc = pd.read_csv(os.path.join(
        common.PROJECT_ROOT, "source/evidence/source_snapshots",
        "outputs_evaluation_holdout_model_cells.csv"))
    pred = {}
    for group, models in [("scalar_increment", ["zero_correction",
                                                "global_training_mean",
                                                "cell_mean_persistence",
                                                "terrain_roughness_equation"]),
                          ("spread", ["global_training_mean",
                                      "cell_mean_persistence",
                                      "roughness_only", "exposure_only",
                                      "locked_joint_equation"])]:
        for m in models:
            prefix = "mean" if group == "scalar_increment" else "spread"
            pred[(group, m)] = mc[f"{prefix}_{m}_predicted"].values.astype(float)

    # ---- time-varying interpolation predictions (670 x 36) ----
    ct = pd.read_csv(os.path.join(OUT_DIR, "interpolation_cell_time.csv"))
    interp = {}
    for mode in ("bilinear", "bicubic"):
        for group, col in [("scalar_increment", "delta_scalar"), ("spread", "spread")]:
            sub = ct[ct.method == mode].pivot(index="state_index",
                                              columns="cell_id", values=col)
            interp[(group, mode)] = sub.values.astype(float)  # (670, 36)
            assert interp[(group, mode)].shape == (670, 36)

    def prediction_for(group, model):
        """Block-mean prediction for a model on the resampled block idx."""
        if model in ("bilinear", "bicubic"):
            return lambda idx: interp[(group, model)][idx].mean(axis=0)
        return lambda idx: pred[(group, model)]  # static

    ALL_PAIRS = PAIRS + [("joint_vs_exposure_only", "spread",
                          "locked_joint_equation", "exposure_only")]
    rng = np.random.default_rng(RNG_SEED)
    all_rows, valid = [], {}
    for length in BLOCK_LENGTHS:
        d = {label: np.zeros(N_REPLICATES) for label, *_ in ALL_PAIRS}
        rmses = {(label, k): np.zeros(N_REPLICATES)
                 for label, *_ in ALL_PAIRS for k in (0, 1)}
        for rep in range(N_REPLICATES):
            idx = block_resample(rng, n, length)   # identical blocks for all models
            for label, group, ma, mb in ALL_PAIRS:
                om = obs[group][idx].mean(axis=0)
                pa = prediction_for(group, ma)(idx)
                pb = prediction_for(group, mb)(idx)
                ra, rb = cell_rmse(om, pa), cell_rmse(om, pb)
                d[label][rep] = ra - rb
                rmses[(label, 0)][rep] = ra
                rmses[(label, 1)][rep] = rb
        for label, group, ma, mb in ALL_PAIRS:
            lo, hi = np.percentile(d[label], [2.5, 97.5])
            all_rows.append(dict(pair=label, group=group, model_a=ma, model_b=mb,
                                 block_minutes=length, replicates=N_REPLICATES,
                                 dRMSE_est=float(d[label].mean()),
                                 dRMSE_median=float(np.median(d[label])),
                                 ci_low=float(lo), ci_high=float(hi),
                                 p_dRMSE_negative=float((d[label] < 0).mean()),
                                 rmse_a_mean=float(rmses[(label, 0)].mean()),
                                 rmse_b_mean=float(rmses[(label, 1)].mean())))

    df = pd.DataFrame([r for r in all_rows if r["pair"] in [p[0] for p in PAIRS]])
    df.to_csv(os.path.join(OUT_DIR, "interpolation_bootstrap.csv"), index=False)
    with open(os.path.join(OUT_DIR, "interpolation_bootstrap.json"), "w",
              encoding="utf-8") as f:
        json.dump({"seed": RNG_SEED, "n_replicates": N_REPLICATES,
                   "block_lengths": BLOCK_LENGTHS,
                   "note": "Interpolation predictions recomputed per replicate; "
                           "terrain/global/persistence static; identical blocks "
                           "for all methods (paired).",
                   "rows": df.to_dict("records")}, f, indent=1)

    # ---- validate the code path against the sealed 5 pairs (all_rows holds
    #      the full set incl. joint_vs_exposure_only) ----
    sealed = pd.read_csv(SEALED_BOOT)
    ok = True
    for label, group, ma, mb in OLD_PAIRS:
        mine = pd.DataFrame(
            [r for r in all_rows if r["pair"] == label]).sort_values("block_minutes")
        theirs = sealed[(sealed.pair == label)].sort_values("block_minutes")
        assert len(mine) == len(theirs) == 3, (label, len(mine), len(theirs))
        for col in ("dRMSE_est", "dRMSE_median", "ci_low", "ci_high",
                    "p_dRMSE_negative", "rmse_a_mean", "rmse_b_mean"):
            diff = float(np.abs(mine[col].values - theirs[col].values).max())
            valid[(label, col)] = diff
            if diff > 1e-6:
                ok = False
                print(f"  VALIDATION FAIL {label} {col}: diff {diff:.3e}")
    print(f"bootstrap code-path validation vs sealed 5 pairs: {'PASS' if ok else 'FAIL'}")
    with open(os.path.join(OUT_DIR, "bootstrap_validation.json"), "w",
              encoding="utf-8") as f:
        json.dump({"code_path_matches_sealed": bool(ok),
                   "max_diffs": {f"{k[0]}::{k[1]}": v for k, v in valid.items()}},
                  f, indent=1)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
