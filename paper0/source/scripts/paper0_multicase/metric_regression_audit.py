"""Audit the constant-correlation fix and numeric non-regression.

The compact pre-repair snapshot is captured by
``capture_numeric_baseline.py`` before a full rebuild.  This audit then checks
the known erroneous constant-baseline Spearman fields and compares A/B science
tables while allowing only the documented Spearman correction.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402


META_FIELDS = set(C.OUTPUT_METADATA_FIELDS)


def run_unit_tests() -> tuple[bool, str]:
    import test_metrics

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_metrics)
    stream = __import__("io").StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return result.wasSuccessful(), stream.getvalue().strip()


def _is_constant_baseline(row: dict) -> bool:
    label = " ".join(str(row.get(k, "")) for k in ("model", "method"))
    label = label.lower()
    return "global" in label or "zero_correction" in label


def _equal(old, new, tol: float = 2e-6) -> bool:
    old_nan = pd.isna(old)
    new_nan = pd.isna(new)
    if old_nan and new_nan:
        return True
    if old_nan != new_nan:
        return False
    if isinstance(old, (int, float, np.number)) and isinstance(
            new, (int, float, np.number)):
        return bool(math.isclose(float(old), float(new), rel_tol=1e-8,
                                 abs_tol=tol))
    return str(old) == str(new)


def compare_baseline(baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures = []
    corrections = []
    compared = 0
    for rel, records in baseline.get("files", {}).items():
        current_path = C.EXT / rel
        if not current_path.is_file():
            failures.append(f"missing current table: {rel}")
            continue
        old = pd.DataFrame(records)
        new = pd.read_csv(current_path)
        if old.shape[0] != new.shape[0]:
            failures.append(f"{rel}: row count {len(new)} != {len(old)}")
            continue
        for i in range(len(old)):
            old_row = old.iloc[i].to_dict()
            new_row = new.iloc[i].to_dict()
            for column in old.columns:
                if column not in new.columns or column in META_FIELDS:
                    continue
                compared += 1
                ov, nv = old_row[column], new_row[column]
                if column == "spearman" and _is_constant_baseline(old_row):
                    if pd.notna(ov) and pd.isna(nv):
                        corrections.append({"file": rel, "row": i,
                                             "model": old_row.get("model",
                                                                      old_row.get("method", "")),
                                             "old": float(ov), "fixed": None})
                        continue
                if not _equal(ov, nv):
                    failures.append(
                        f"{rel}[row {i}] {column}: old={ov!r}, new={nv!r}")
    return {"compared_fields": compared, "corrections": corrections,
            "failures": failures}


def scan_current_constant_rows() -> dict:
    bad = []
    checked = []
    for path in sorted(C.EXT.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "spearman" not in df.columns:
            continue
        for i, row in df.iterrows():
            record = row.to_dict()
            if not _is_constant_baseline(record):
                continue
            checked.append(str(path.relative_to(C.EXT)) + f"[{i}]")
            if pd.notna(row["spearman"]):
                bad.append({"file": str(path.relative_to(C.EXT)),
                            "row": int(i), "spearman": float(row["spearman"]),
                            "model": record.get("model", record.get("method", ""))})
    return {"checked": checked, "bad": bad}


def write_report(result: dict, output: Path, baseline_path: Path,
                 tests_ok: bool, test_log: str) -> None:
    all_meta = C.status_metadata(list(C.CASE_ORDER))
    status_lines = [
        f"* Case {c}: `case_status={C.case_metadata(c)['case_status']}`, "
        f"`science_eligible={str(C.case_metadata(c)['science_eligible']).lower()}`, "
        f"`contains_placeholder={str(C.case_metadata(c)['contains_placeholder']).lower()}`"
        for c in C.CASE_ORDER
    ]
    lines = [
        "# METRIC_REGRESSION_AUDIT — Paper0 multicase", "",
        "This audit is an engineering/non-regression record.  It does not "
        "promote any Case C payload or change claim eligibility.", "",
        *status_lines,
        f"* Aggregate dependency status: `{all_meta['dependency_status']}`",
        f"* Baseline: `{baseline_path}`",
        "", "## Constant-vector contract", "",
        "For a constant prediction or constant reference, Pearson r, Spearman "
        "rho and regression slope are NaN.  Spearman uses "
        "`scipy.stats.spearmanr` semantics for ties.", "",
        f"* Unit tests: **{'PASS' if tests_ok else 'FAIL'}**",
        f"* Constant-baseline rows checked: {len(result['constant']['checked'])}",
        f"* Constant-baseline rows with non-NaN Spearman after repair: "
        f"{len(result['constant']['bad'])}",
        "",
        "## Confirmed corrected fields", "",
        "| artifact | row | model/method | old Spearman | repaired Spearman |",
        "|---|---:|---|---:|---:|",
    ]
    if result["comparison"]["corrections"]:
        for item in result["comparison"]["corrections"]:
            lines.append(f"| {item['file']} | {item['row']} | "
                         f"{item['model']} | {item['old']:.9f} | NaN |")
    else:
        lines.append("| none observed in the saved pre-repair snapshot | | | | |")
    lines += [
        "", "## Numeric non-regression", "",
        f"* Compared {result['comparison']['compared_fields']} fields across "
        f"the saved A/B/core-table snapshot.",
        f"* Unexpected differences: **{len(result['comparison']['failures'])}**",
        "* Scope of accepted change: constant-baseline Spearman only.",
        "* Core RMSE, terrain relationships, history comparisons, representation "
        "metrics and A->B conclusions are expected to remain unchanged.",
    ]
    if result["comparison"]["failures"]:
        lines += ["", "Unexpected differences:"]
        lines.extend(f"* {x}" for x in result["comparison"]["failures"])
    if test_log:
        lines += ["", "## Test runner", "", "```text", test_log, "```"]
    lines += ["", f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(baseline_path: str | Path, output: str | Path) -> bool:
    baseline = Path(baseline_path).expanduser().resolve()
    if not baseline.is_file():
        raise FileNotFoundError(f"missing baseline snapshot: {baseline}")
    tests_ok, test_log = run_unit_tests()
    comparison = compare_baseline(baseline)
    constant = scan_current_constant_rows()
    result = {"comparison": comparison, "constant": constant}
    write_report(result, Path(output).expanduser().resolve(), baseline,
                 tests_ok, test_log)
    passed = tests_ok and not comparison["failures"] and not constant["bad"]
    print(json.dumps({"pass": passed, "unit_tests": tests_ok,
                      "corrections": len(comparison["corrections"]),
                      "unexpected_differences": len(comparison["failures"]),
                      "constant_rows_bad": len(constant["bad"])}, indent=2))
    return bool(passed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=str(
        C.QA / "pre_repair_numeric_baseline_20260829.json"))
    ap.add_argument("--output", default=str(
        C.EXT / "audit" / "METRIC_REGRESSION_AUDIT.md"))
    args = ap.parse_args()
    return 0 if audit(args.baseline, args.output) else 2


if __name__ == "__main__":
    raise SystemExit(main())
