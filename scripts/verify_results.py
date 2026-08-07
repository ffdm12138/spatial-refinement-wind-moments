"""Verify the integrity of this release package (Level 1, fully offline).

Checks performed (exit code 0 only if all pass):

  1. Checksum manifest — every file listed in SHA256SUMS.txt verifies, and
     every released file (except SHA256SUMS.txt itself) is listed.
  2. Interpolation record size — source_data/interpolation_cell_time.csv has
     72,360 data rows = 3 methods x 24,120 (reference, bilinear, bicubic).
  3. RMSE decomposition — RMSE^2 == cRMSE^2 + bias^2 for every
     information_sources.csv row where all three are finite, and for every
     interpolation_metrics.csv row (full precision, tolerance 1e-8).
  4. Locked scientific values — moment recovery ratios Q_delta/Q_J/Q_S
     (interpolation_recovery.csv, tolerance 1e-4), vector-preservation error
     E_vec (interpolation_metrics.csv, tolerance 1e-3).
  5. Exact decomposition — domain-mean delta_dyn + J == delta_scalar from
     figure_data/F03_physical_decomposition.csv (tolerance 1e-6; the
     published CSV stores rounded cell means, so the identity holds to
     ~5e-8, not machine precision).
  6. Finiteness — every numeric field must be finite except an explicit
     NA allow-list (r and slope are not applicable for the constant
     zero/global-mean baselines; all_record_rmse is not applicable for the
     spread group).
  7. File contracts — every evidence_file in npj_claim_ledger.csv and every
     source_csv in npj_figure_contracts.csv exists in this package.
  8. Canonical figures — the six Fig*.pdf reference outputs exist in figures/.

Usage:
    python scripts/verify_results.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
failures: list[str] = []

# Per-file/column rules for legitimate non-finite values:
#   "na"  -> missing values allowed (not applicable / not computed); all
#            present values must be finite
#   "inf" -> positive infinity allowed as a sentinel (open-ended final bin of
#            the wind-speed bins in Fig 3); NaN is not allowed
# Everything else must be finite; any unexpected NaN/Inf is a defect.
COLUMN_RULES = {
    "candidate_screening_summary.csv": {"bootstrap_shelter_low": "na",
                                        "bootstrap_shelter_high": "na"},
    "error_decomposition.csv": {"calibration_slope": "na"},
    "F04_jensen_binned.csv": {"bin_hi": "inf"},
}


def check(cond: bool, msg: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_ok(s: pd.Series) -> bool:
    return bool(np.isfinite(s.astype(float)).all())


def main() -> None:
    print("== verify_results: release package integrity ==")

    # ---- 1. checksum manifest ----
    print("  -- SHA256SUMS coverage ==")
    sums = {}
    for ln in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        h, rel = ln.split("  ", 1)
        sums[rel] = h
    listed = set(sums)
    on_disk = set()
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel == "SHA256SUMS.txt" or "/.git/" in "/" + rel or rel.startswith(".git/"):
            continue
        if any(seg in ("reproduced", "__pycache__", ".pytest_cache")
               for seg in Path(rel).parts):
            continue
        on_disk.add(rel)
    missing_listed = [r for r in listed if not (ROOT / r).exists()]
    unlisted = sorted(on_disk - listed)
    bad = []
    for rel, h in sums.items():
        if (ROOT / rel).exists() and sha256(ROOT / rel) != h:
            bad.append(rel)
    check(not missing_listed and not unlisted and not bad,
          f"SHA256SUMS: {len(sums)} entries, {len(on_disk)} files on disk; "
          f"missing={len(missing_listed)} unlisted={len(unlisted)} "
          f"mismatched={len(bad)}")

    # ---- 2. interpolation record size ----
    print("  -- interpolation_cell_time rows ==")
    ct = pd.read_csv(ROOT / "source_data" / "interpolation_cell_time.csv")
    n_rows = len(ct)
    counts = ct.method.value_counts().to_dict()
    check(n_rows == 72360 and all(counts.get(m) == 24120
                                   for m in ("reference", "bilinear", "bicubic")),
          f"interpolation_cell_time.csv: {n_rows} rows, per-method {counts} "
          f"(expect 72360 = 3 x 24120)")

    # ---- 3. RMSE decomposition ----
    print("  -- RMSE^2 = cRMSE^2 + bias^2 ==")
    src = pd.read_csv(ROOT / "source_data" / "information_sources.csv")
    n_checked = 0
    for _, r in src.iterrows():
        if pd.isna(r.rmse) or pd.isna(r.crmse) or pd.isna(r.bias):
            continue
        lhs, rhs = float(r.rmse) ** 2, float(r.crmse) ** 2 + float(r.bias) ** 2
        n_checked += 1
        # 3-decimal rounding of the published table allows ~1e-3 slack
        check(abs(lhs - rhs) <= 2e-3,
              f"info_sources {r.group}/{r.method}: {lhs:.6f} vs {rhs:.6f}")
    met = pd.read_csv(ROOT / "source_data" / "interpolation_metrics.csv")
    for _, r in met.iterrows():
        lhs = float(r.rmse) ** 2
        rhs = float(r.crmse) ** 2 + float(r.bias) ** 2
        check(abs(lhs - rhs) <= 1e-8,
              f"metrics {r.group}/{r.model}: {lhs:.12f} vs {rhs:.12f}")
    check(n_checked == 13,
          f"all 13 finite rmse/crmse/bias rows of information_sources checked "
          f"(checked {n_checked})")

    # ---- 4. locked scientific values ----
    print("  -- locked moment recovery and E_vec ==")
    rec = pd.read_csv(ROOT / "source_data" / "interpolation_recovery.csv")
    locked_rec = {
        ("delta_scalar", "bilinear", "all36"): -0.3081,
        ("delta_scalar", "bicubic", "all36"): -0.0272,
        ("J", "bilinear", "all36"): 0.0955,
        ("J", "bicubic", "all36"): 0.1552,
        ("spread", "bilinear", "all36"): 0.3945,
        ("spread", "bicubic", "all36"): 0.4873,
    }
    for (moment, model, cell_set), exp in locked_rec.items():
        got = float(rec[(rec.moment == moment) & (rec.model == model)
                        & (rec.cell_set == cell_set)].recovery_ratio.iloc[0])
        check(abs(got - exp) <= 1e-4,
              f"Q_{moment} {model}: {got:.4f} (expect {exp})")
    locked_evec = {"bilinear": 0.4444, "bicubic": 0.2504}
    for model, exp in locked_evec.items():
        got = float(met[(met.group == "scalar_increment")
                        & (met.model == model)].E_vec_rmse.iloc[0])
        check(abs(got - exp) <= 1e-3,
              f"E_vec {model}: {got:.4f} (expect {exp})")

    # ---- 5. exact decomposition ----
    print("  -- exact decomposition ==")
    f03 = pd.read_csv(ROOT / "figure_data" / "F03_physical_decomposition.csv")
    dyn = float(f03.delta_dynamic_mean.mean())
    jg = float(f03.jensen_gap_mean.mean())
    ds = float(f03.delta_scalar_mean.mean())
    check(abs((dyn + jg) - ds) <= 1e-6,
          f"delta_dyn + J = {dyn + jg:.10f} == delta_scalar {ds:.10f} "
          f"(diff {abs((dyn + jg) - ds):.2e})")

    # ---- 6. finiteness with explicit NA allow-list ----
    print("  -- field-level finiteness ==")
    allow_r_slope = {"zero_correction", "global_training_mean"}
    for path in sorted((ROOT / "source_data").glob("*.csv")):
        df = pd.read_csv(path)
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            s = df[col]
            key_col = ("method" if "method" in df.columns
                       else "model" if "model" in df.columns else None)
            rule = COLUMN_RULES.get(path.name, {}).get(col)
            if rule == "na":
                check(finite_ok(s[s.notna()]),
                      f"{path.name}.{col}: non-NA values finite (NA allow-list)")
            elif rule == "inf":
                check(not s.isna().any() and finite_ok(s[np.isfinite(s)])
                      and (np.isinf(s)).sum() > 0,
                      f"{path.name}.{col}: inf sentinel allowed, NaN not allowed")
            elif path.name == "outputs_evaluation_holdout_baseline_scores.csv" \
                    and col in ("estimate", "ci_low", "ci_high"):
                # sealed holdout records: the metric is undefined for some
                # baseline/target combinations, flagged by the "undefined"
                # string column
                unexpected = df[col].isna() & df["undefined"].isna()
                check(not unexpected.any() and finite_ok(s[s.notna()]),
                      f"{path.name}.{col}: NA only on undefined-flagged rows; "
                      f"unexpected {int(unexpected.sum())}")
            elif col in ("r", "slope", "calibration_slope"):
                rows_na = (df[df[col].isna()][key_col].tolist()
                           if key_col else [])
                unexpected = [m for m in rows_na if m not in allow_r_slope]
                check(not unexpected and finite_ok(s[s.notna()]),
                      f"{path.name}.{col}: NA only on {allow_r_slope}; "
                      f"unexpected {unexpected}")
            elif col == "all_record_rmse":
                na_groups = (df[df[col].isna()]["group"].tolist()
                             if "group" in df.columns else [])
                unexpected = [g for g in na_groups if g != "spread"]
                check(not unexpected and finite_ok(s[s.notna()]),
                      f"{path.name}.{col}: NA only on spread; unexpected {unexpected}")
            else:
                check(finite_ok(s), f"{path.name}.{col}: all finite")
    for path in sorted((ROOT / "figure_data").glob("*.csv")):
        df = pd.read_csv(path)
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            rule = COLUMN_RULES.get(path.name, {}).get(col)
            if rule == "na":
                ok = finite_ok(df[col][df[col].notna()])
            elif rule == "inf":
                ok = not df[col].isna().any() \
                     and finite_ok(df[col][np.isfinite(df[col])])
            else:
                ok = finite_ok(df[col])
            if not ok:
                check(False, f"{path.name}.{col}: unexpected non-finite values")

    # ---- 7. file contracts ----
    print("  -- claim ledger and figure contracts ==")
    ledger = pd.read_csv(ROOT / "source_data" / "npj_claim_ledger.csv")
    bad_refs = [e for e in ledger.evidence_file if not (ROOT / e).exists()]
    check(not bad_refs and all(str(e).startswith("source_data/")
                               for e in ledger.evidence_file),
          f"npj_claim_ledger: {len(ledger)} claims, all evidence_file "
          f"release-relative and existing (bad: {bad_refs})")
    contracts = pd.read_csv(ROOT / "source_data" / "npj_figure_contracts.csv")
    bad_src = [s for s in contracts.source_csv
               if any(not (ROOT / p).exists() for p in s.split(";"))]
    check(not bad_src and len(contracts) == 6
          and set(contracts.figure_id) == {
              "Fig1_study_design", "Fig2_coarse_graining_loss",
              "Fig3_jensen_covariance", "Fig4_geometric_refinement",
              "Fig5_terrain_organization", "Fig6_information_sources_state"},
          f"npj_figure_contracts: {len(contracts)} figures, source_csv "
          f"existing (bad: {bad_src})")

    # ---- 8. canonical figures ----
    print("  -- canonical figures ==")
    figdir = ROOT / "figures"
    stems = ["Fig1_study_design", "Fig2_coarse_graining_loss",
             "Fig3_jensen_covariance", "Fig4_geometric_refinement",
             "Fig5_terrain_organization", "Fig6_information_sources_state"]
    missing = [f"{s}.pdf" for s in stems if not (figdir / f"{s}.pdf").exists()]
    check(not missing, f"figures/: six canonical PDFs present (missing {missing})")

    print()
    if failures:
        print(f"verify_results: {len(failures)} FAILURE(S)")
        for f_ in failures:
            print("  -", f_)
        sys.exit(1)
    print("verify_results: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
