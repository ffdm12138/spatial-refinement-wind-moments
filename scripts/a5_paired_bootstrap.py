"""a5: Paired moving-block bootstrap of evaluation-period RMSE (plan 4.5).

Evaluation period: 670 states x 36 parent cells, per-state observed targets
(delta_scalar, spread) recomputed from the raw archived tensors; model
predictions are the locked period-mean values from the sealed holdout table
(model_cells), held static.

Procedure per replicate:
  1. draw overlapping moving blocks (length L) of the 670-state axis until
     >= 670 states are covered; truncate to 670.
  2. recompute per-cell time means of the observed targets on the resampled
     states.
  3. compute the 36-cell RMSE for every model (prediction - resampled mean).
  4. form paired differences of RMSE between specified model pairs.

Block-length sensitivity: 60 / 120 / 240 min. Replicates: 2000, seed fixed.

Outputs:
  paired_bootstrap.csv  - per pair, per block length: dRMSE estimate + 95% CI
  paired_bootstrap.json - summary incl. P(dRMSE < 0) and block-length table
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import common

OUT = common.EVIDENCE_DIR
os.makedirs(OUT, exist_ok=True)
RNG_SEED = 20260807
N_REPLICATES = 2000
BLOCK_LENGTHS = [60, 120, 240]

PAIRS = [  # (label, group, model_a, model_b)
    ("terrain_vs_global", "scalar_increment", "terrain_roughness_equation", "global_training_mean"),
    ("terrain_vs_persistence", "scalar_increment", "terrain_roughness_equation", "cell_mean_persistence"),
    ("joint_vs_global", "spread", "locked_joint_equation", "global_training_mean"),
    ("joint_vs_persistence", "spread", "locked_joint_equation", "cell_mean_persistence"),
    ("joint_vs_exposure_only", "spread", "locked_joint_equation", "exposure_only"),
]


def block_resample(rng, n, length, rng_block=True):
    """Overlapping moving-block resampling of the index axis 0..n-1."""
    n_blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n - length + 1, size=n_blocks) if rng_block else \
        rng.integers(0, max(n - length + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + length) for s in starts])[:n]
    return idx


def cell_rmse(obs_mean, pred):
    """36-cell RMSE between resampled per-cell means and static predictions."""
    return float(np.sqrt(np.mean((obs_mean - pred) ** 2)))


def main():
    A = common.state_arrays("eval")
    obs = {"scalar_increment": A["delta_scalar"], "spread": A["spread"]}  # (670, 36)
    n = obs["scalar_increment"].shape[0]
    assert n == 670

    mc = pd.read_csv(os.path.join(common.PROJECT_ROOT,
        "source/evidence/source_snapshots/outputs_evaluation_holdout_model_cells.csv"))
    pred = {}
    for group, models in [("scalar_increment", ["zero_correction", "global_training_mean",
                                                "cell_mean_persistence", "terrain_roughness_equation"]),
                          ("spread", ["global_training_mean", "cell_mean_persistence",
                                      "roughness_only", "exposure_only", "locked_joint_equation"])]:
        for m in models:
            prefix = "mean" if group == "scalar_increment" else "spread"
            col = f"{prefix}_{m}_predicted"
            pred[(group, m)] = mc[col].values.astype(float)  # (36,)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for length in BLOCK_LENGTHS:
        d = {label: np.zeros(N_REPLICATES) for label, *_ in PAIRS}
        rmses = {(label, k): np.zeros(N_REPLICATES) for label, *_ in PAIRS for k in (0, 1)}
        for rep in range(N_REPLICATES):
            idx = block_resample(rng, n, length)
            for label, group, ma, mb in PAIRS:
                om = obs[group][idx].mean(axis=0)              # resampled per-cell means
                ra = cell_rmse(om, pred[(group, ma)])
                rb = cell_rmse(om, pred[(group, mb)])
                d[label][rep] = ra - rb
                rmses[(label, 0)][rep] = ra
                rmses[(label, 1)][rep] = rb
        for label, group, ma, mb in PAIRS:
            lo, hi = np.percentile(d[label], [2.5, 97.5])
            rows.append(dict(
                pair=label, group=group, model_a=ma, model_b=mb,
                block_minutes=length, replicates=N_REPLICATES,
                dRMSE_est=float(d[label].mean()), dRMSE_median=float(np.median(d[label])),
                ci_low=float(lo), ci_high=float(hi),
                p_dRMSE_negative=float((d[label] < 0).mean()),
                rmse_a_mean=float(rmses[(label, 0)].mean()),
                rmse_b_mean=float(rmses[(label, 1)].mean()),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/paired_bootstrap.csv", index=False)
    with open(f"{OUT}/paired_bootstrap.json", "w") as f:
        json.dump({"seed": RNG_SEED, "n_replicates": N_REPLICATES,
                   "block_lengths": BLOCK_LENGTHS,
                   "note": "All models share identical resampled time blocks (paired).",
                   "rows": df.to_dict("records")}, f, indent=1)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
