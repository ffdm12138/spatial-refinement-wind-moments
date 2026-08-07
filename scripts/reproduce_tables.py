"""Reconstruct Tables 3 and 4 of the manuscript from the released derived data.

Level-1 reproduction: consumes only `figure_data/` and `source_data/` inside
this package (or the internal workspace layout).

  Table 3 (fitted equations)  — refitted by OLS on the 36 calibration-period
      cell means from figure_data/F05_F06_terrain_parameterization.csv and
      compared with the frozen manuscript coefficients. The comparison uses a
      tolerance of 1e-4 and reports the difference; it is a reproducibility
      check on a fitted result, not a byte-for-byte assertion.
  Table 4 (information sources) — rebuilt from
      source_data/information_sources.csv with the same row layout as the
      manuscript table.

Outputs go to `<package>/reproduced/tables/` by default (or `--output-dir`);
the canonical `tables/` directory is never modified.

Usage:
    python scripts/reproduce_tables.py [--output-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Frozen manuscript coefficients (main text, Table 3): (name, [coefs], fit spec)
FROZEN_EQUATIONS = {
    "scalar-increment roughness fit": {
        "coefs": (0.079296, 5.578546),
        "formula": "mu_Delta = b0 + b1 * r_sigma",
    },
    "spread joint fit (roughness IQR + isotropic exposure)": {
        "coefs": (0.383082, -0.875890, 9.375677),
        "formula": "mu_S = b0 + b1 * r_IQR + b2 * lambda_iso",
    },
}
REFIT_TOL = 1e-4

# Table 4 row order, matching the manuscript table
TABLE4_ORDER_SCALAR = ["zero_correction", "global_training_mean", "bilinear",
                       "bicubic", "terrain_roughness_equation",
                       "cell_mean_persistence"]
TABLE4_ORDER_SPREAD = ["global_training_mean", "bilinear", "bicubic",
                       "roughness_only", "exposure_only",
                       "locked_joint_equation", "cell_mean_persistence"]


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients for design matrix x (column of ones first)."""
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def table3_markdown(f56: pd.DataFrame) -> list[str]:
    d = f56.sort_values("cell_id")
    lines = ["# Table 3 — fitted equations (refitted from the released cell means)\n"]
    ok_all = True
    for name, spec in FROZEN_EQUATIONS.items():
        if name.startswith("scalar"):
            x = np.column_stack([np.ones(36), d.mean_roughness.values])
            y = d.mean_observed_train.values
        else:
            x = np.column_stack([np.ones(36), d.spread_roughness.values,
                                 d.spread_lambda.values])
            y = d.spread_observed_train.values
        coef = _ols(x, y)
        diff = np.abs(coef - np.asarray(spec["coefs"]))
        ok = bool(np.all(diff <= REFIT_TOL))
        ok_all &= ok
        lines.append(f"## {name}")
        lines.append(f"- formula: {spec['formula']}")
        lines.append(f"- refitted coefficients: {', '.join(f'{c:.6f}' for c in coef)}")
        lines.append(f"- manuscript coefficients: "
                     f"{', '.join(f'{c:.6f}' for c in spec['coefs'])}")
        lines.append(f"- max |difference|: {diff.max():.2e} "
                     f"(tolerance {REFIT_TOL:.0e}) -> {'PASS' if ok else 'FAIL'}")
        lines.append("")
    return lines, ok_all


def table4_markdown(src: pd.DataFrame) -> list[str]:
    lines = ["# Table 4 — comparison across information sources "
             "(evaluation period, final 670 states)\n"]
    cols = ["RMSE", "Centred RMSE", "Bias", "r", "Slope", "All-record RMSE"]
    for group, order, header in [
            ("scalar_increment", TABLE4_ORDER_SCALAR,
             "(a) Period-mean scalar increment"),
            ("spread", TABLE4_ORDER_SPREAD, "(b) Period-mean within-parent spread")]:
        lines.append(f"## {header}\n")
        lines.append("| Model | Information used | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (2 + len(cols)))
        d = src[src.group == group].set_index("method").loc[order]
        for mname, r in d.iterrows():
            def fmt(v):
                return "--" if pd.isna(v) else f"{float(v):.3f}"
            lines.append(f"| {mname} | {r.information_used} | "
                         + " | ".join(fmt(r[c]) for c in
                                      ["rmse", "crmse", "bias", "r", "slope",
                                       "all_record_rmse"]) + " |")
        lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reproduced" / "tables",
                    help="output directory (default: reproduced/tables)")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    f56 = pd.read_csv(ROOT / "figure_data" / "F05_F06_terrain_parameterization.csv")
    src = pd.read_csv(ROOT / "source_data" / "information_sources.csv")

    t3, ok3 = table3_markdown(f56)
    t4 = table4_markdown(src)
    text = "\n".join(t3 + t4) + "\n"
    (out_dir / "tables.md").write_text(text, encoding="utf-8")
    (out_dir / "tables.txt").write_text(text, encoding="utf-8")

    print(f"reproduce_tables: written to {out_dir}/ (tables.md/tables.txt)")
    print(f"  Table 3 refit comparison: {'PASS' if ok3 else 'FAIL'}"
          f" (tolerance {REFIT_TOL:.0e})")
    if not ok3:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
