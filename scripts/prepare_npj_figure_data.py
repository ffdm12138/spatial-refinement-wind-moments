"""Prepare NPJ figure source data.

Extends F06_chronological_evaluation.csv with the interpolation baseline
columns (period-mean 36-cell predictions and their centred variants), and
writes the pre-registered representative-state scalar-speed fields for
Fig. 4a (already produced by evaluate_interpolation_baselines.py as
representative_state_fields.npz).

Old figure data is only read; the extended F06 file is written in place
(figure_data is a journal-visible artifact being redesigned for NPJ).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FDIR = os.path.join(PROJECT_ROOT, "source", "figure_data", "main")
EVDIR = os.path.join(PROJECT_ROOT, "source", "evidence", "npj_refactor")


def main() -> None:
    f06 = pd.read_csv(os.path.join(FDIR, "F06_chronological_evaluation.csv"))
    cm = pd.read_csv(os.path.join(EVDIR, "interpolation_cell_means.csv"))

    def add_column(group, model, colname):
        sub = cm[(cm.group == group) & (cm.model == model)]
        assert len(sub) == 36, (group, model, len(sub))
        vals = sub.sort_values("cell_id")["pred_mean_cell"].values
        f06[colname] = vals
        f06[f"{colname}_c"] = vals - vals.mean()

    add_column("delta_scalar", "bilinear", "pred_scalar_bilinear")
    add_column("delta_scalar", "bicubic", "pred_scalar_bicubic")
    add_column("spread", "bilinear", "pred_spread_bilinear")
    add_column("spread", "bicubic", "pred_spread_bicubic")

    f06.to_csv(os.path.join(FDIR, "F06_chronological_evaluation.csv"),
               index=False)
    print("F06 columns now:", [c for c in f06.columns])

    # verify the representative-state fields exist
    npz = np.load(os.path.join(EVDIR, "representative_state_fields.npz"))
    for k in ("d02_6x6_speed", "bilinear_speed", "bicubic_speed", "d05_speed",
              "state_global", "state_relative"):
        assert k in npz, k
    print("representative_state_fields.npz OK:",
          {k: npz[k].shape for k in ("d02_6x6_speed", "bilinear_speed",
                                     "bicubic_speed", "d05_speed")})


if __name__ == "__main__":
    main()
