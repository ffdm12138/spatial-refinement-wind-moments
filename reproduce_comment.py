#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduce the figures and audited tables of the npj Comment.

"Increasing spatial sampling cannot restore subgrid wind moments"
(npj Climate and Atmospheric Science, Comment).

Three real two-way-coupled WRF cases under one fixed analysis contract:
  A = Yakou, September 2015   (mechanism / reference case)
  B = Yakou, December 2015    (same site, different event)
  C = Ebao,  December 2015    (different site, same event)

Two modes
---------
tables (default, no data download needed)
    Verifies the shipped audited figure-source tables (figure_source/,
    SHA256-checked), runs headline numeric checks and regenerates the three
    Comment figures into figures/.

raw --data /path/to/data_root
    Full re-derivation from the raw serialized WRF tensors (GitHub Release
    tag `raw-data-multicase-v1`): raw-contract validation -> parent/fine
    moments -> bilinear/bicubic interpolation controls (+ parent-vector
    constraint) -> terrain descriptors/relations -> temporal and cross-site
    transfers -> information-source comparison -> manuscript candidate
    tables -> independent 36/36 interpolation QA -> comparison against the
    shipped audited tables -> figures.

The raw data root must contain one sub-directory per case, each holding the
seven serialized tensors, e.g. (names match the Release assets):

  data_root/
    1min_yakou20150924/  HGT_d05_yakou20150924.pt  T2_d02_...pt  T2_d05_...pt
                         U10_d02_...pt U10_d05_...pt V10_d02_...pt V10_d05_...pt
    1min_yakou20151201/  (same seven stems with the _yakou20151201 suffix)
    1min_ebao20151201/   (same seven stems with the _ebao20151201 suffix)

Examples
--------
  python reproduce_comment.py
  python reproduce_comment.py --mode raw --data /path/to/data_root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL_TABLES = 5e-3   # cross-machine tolerance for regeneration comparisons
TOL_QA = 2e-3       # independent 36/36 QA tolerance (canonical table rounding)

CASE_DIRS = {
    "case_A_yakou0924": "A",
    "case_B_yakou1201": "B",
    "case_C_ebao1201": "C",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _configure(data_root: str | None) -> None:
    os.environ.setdefault("NPJ_MULTICASE_GENERATION", "20260905")
    if data_root:
        os.environ["NPJ_MULTICASE_DATA_ROOT"] = str(Path(data_root).resolve())
    sys.path.insert(0, str(ROOT / "pipeline"))


# ---------------------------------------------------------------------------
# tables mode
# ---------------------------------------------------------------------------
def check_figure_source() -> None:
    sums = ROOT / "figure_source" / "SHA256SUMS.txt"
    if not sums.is_file():
        raise SystemExit("FAIL: figure_source/SHA256SUMS.txt missing")
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        path = ROOT / "figure_source" / name.strip()
        actual = sha256_of(path)
        if actual != digest:
            raise SystemExit(
                f"FAIL: sha256 mismatch for {name.strip()}: {actual} != {digest}")
    print("[tables] figure_source sha256 verification: PASS")


def numeric_spot_checks() -> None:
    import pandas as pd

    c1 = pd.read_csv(ROOT / "figure_source" / "C1_core_loss.csv")
    c2 = pd.read_csv(ROOT / "figure_source" / "C2_geometric_recovery.csv")
    c3 = pd.read_csv(ROOT / "figure_source" / "C3_transfers.csv")

    ratios = c1["J_domain_mean"] / c1["delta_scalar_domain_mean"]
    assert ratios.min() >= 1.00 and ratios.max() <= 1.03, "J/delta ratio out of range"
    assert (c1["delta_dyn_domain_mean"].abs() <= 0.007).all(), "delta_dyn too large"

    qj = c2["Q_J"].unique()
    qs = c2["Q_spread"].unique()
    assert 0.095 - 1e-3 <= qj.min() and qj.max() <= 0.279 + 1e-3, "Q_J range"
    assert 0.247 - 1e-3 <= qs.min() and qs.max() <= 0.487 + 1e-3, "Q_S range"

    t1 = c3[(c3.test == "T1") & (c3.segment == "evaluation_670")]
    t3 = c3[(c3.test == "T3") & (c3.segment == "evaluation_670")]
    a2b_delta = float(t1[t1.model == "A_terrain_delta"]["spearman"].iloc[0])
    b2c_delta = float(t3[t3.model == "B_cal_delta_vs_r_sigma"]["spearman"].iloc[0])
    assert abs(a2b_delta - 0.589) < 1e-2 and abs(b2c_delta - (-0.030)) < 1e-2
    print("[tables] headline numeric spot checks: PASS "
          f"(J/delta in [{ratios.min():.3f},{ratios.max():.3f}], "
          f"Q_J max {qj.max():.3f}, Q_S max {qs.max():.3f}, "
          f"A->B delta r_s {a2b_delta:.3f}, B->C delta r_s {b2c_delta:+.3f})")


def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import make_comment_figures as F

    paths = [F.fig1(), F.fig2(), F.fig3()]
    for p in paths:
        print("[figures] wrote", Path(p).relative_to(ROOT))


def run_tables_mode() -> int:
    check_figure_source()
    numeric_spot_checks()
    make_figures()
    print("TABLES MODE: PASS (shipped audited tables verified; figures regenerated)")
    return 0


# ---------------------------------------------------------------------------
# raw mode
# ---------------------------------------------------------------------------
def _compare_csv(fresh: Path, shipped: Path, label: str) -> None:
    import numpy as np
    import pandas as pd

    a = pd.read_csv(fresh)
    b = pd.read_csv(shipped)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise SystemExit(f"FAIL: {label}: schema/row mismatch")
    for col in a.columns:
        if a[col].dtype.kind in "if" and b[col].dtype.kind in "if":
            diff = np.abs(a[col].to_numpy(float) - b[col].to_numpy(float))
            if diff.max() > TOL_TABLES:
                raise SystemExit(
                    f"FAIL: {label}: column {col} max diff {diff.max():.3g} "
                    f"> tolerance {TOL_TABLES}")
        elif not (a[col].astype(str) == b[col].astype(str)).all():
            raise SystemExit(f"FAIL: {label}: column {col} values differ")
    print(f"[raw] {label}: matches shipped audited table (tol {TOL_TABLES})")


def run_raw_mode(data_root: str) -> int:
    import mc_config as C
    import phase00_raw_contract as P0
    import phase01_moments as P1
    import phase05_interpolation as P5
    import phase04_scale as P4
    import phase07_terrain as P7
    import phase09_transfers as P9
    import phase10_windstate as P10
    import phase12_infosources as P12
    import phase16_tables as P16

    cases = ["A", "B", "C"]
    print("[raw] phase00 raw contract ...")
    P0.main(cases)
    for case in cases:
        print(f"[raw] phase01 moments ({case}) ...")
        P1.main(case)
    for case in cases:
        print(f"[raw] phase04 scale decomposition ({case}) ...")
        P4.main(case)
    for case in cases:
        print(f"[raw] phase05 interpolation controls ({case}) ...")
        P5.main(case)
    for case in cases:
        print(f"[raw] phase07 terrain descriptors/relations ({case}) ...")
        P7.main(case)
    P7.regime_comparison(list(cases))
    print("[raw] phase09 transfers (A->B, B->C) ...")
    P9.main(set(cases))
    summaries = {}
    for case in cases:
        print(f"[raw] phase10 wind state ({case}) ...")
        summaries[case] = P10.main(case)
    P10.wind_state_comparison(list(cases), summaries)
    for case in cases:
        print(f"[raw] phase12 information sources ({case}) ...")
        P12.main(case)
    P12.ranking(list(cases))
    print("[raw] phase16 manuscript candidate tables ...")
    P16.main(cases)

    print("[raw] independent 36/36 interpolation QA ...")
    import independent_recovery_qa

    rc = independent_recovery_qa.main()
    if rc != 0:
        raise SystemExit("FAIL: independent interpolation QA did not pass 36/36")
    # The independent QA is the Comment-scope audit gate.  With it passing,
    # promote the freshly re-derived Case C from REAL_PENDING_AUDIT to REAL
    # (the only code path that does so), then re-export the candidate
    # tables so they cover all three cases.
    print("[raw] independent QA passed; promoting Case C to REAL ...")
    C.promote_real_case("C", {"audit": "independent_recovery_qa",
                              "result": "PASS", "scope": "comment raw reproduction"})
    P16.main(cases)

    fresh = C.EXT / "comment_candidate"
    for name in ("C1_core_loss.csv", "C2_geometric_recovery.csv",
                 "C3_transfers.csv", "C4_information_sources.csv"):
        _compare_csv(fresh / name, ROOT / "figure_source" / name, name)

    os.environ["COMMENT_TABLES_DIR"] = str(fresh)
    make_figures()
    receipt = ROOT / "results" / "independent_recovery_recompute.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    print(f"[raw] 36/36 QA verdict: {payload['verdict']} "
          f"({sum(1 for v in payload['results'].values() if v['pass'])}/"
          f"{len(payload['results'])} rows)")
    print("RAW MODE: PASS (full re-derivation matches the shipped audited tables)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("tables", "raw"), default="tables",
                    help="tables (default): verify shipped tables + figures; "
                         "raw: full re-derivation from serialized WRF tensors")
    ap.add_argument("--data", default=None,
                    help="raw data root containing the three case sub-directories "
                         "(Release tag raw-data-multicase-v1)")
    args = ap.parse_args()
    if args.mode == "raw" and not args.data:
        ap.error("--mode raw requires --data /path/to/data_root")
    _configure(args.data)
    if args.mode == "tables":
        return run_tables_mode()
    return run_raw_mode(args.data)


if __name__ == "__main__":
    sys.exit(main())
