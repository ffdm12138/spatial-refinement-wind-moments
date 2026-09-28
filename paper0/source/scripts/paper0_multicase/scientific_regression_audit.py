"""Audit future Case-C promotion for scientific numeric drift.

Only the protected A/B and A->B scope is compared.  C-dependent rows are
explicitly identified and excluded from the numeric comparison because a real
Dayekou archive is expected to supply new values there.  This is a regression
audit, not a claim-strength or author-approval gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402


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


def _dependencies(relative: str, row: dict) -> list[str]:
    if str(row.get("case", "")) in C.CASE_ORDER:
        return [str(row["case"])]
    found: list[str] = []
    for field in ("case_status", "data_status"):
        found.extend(C._cases_from_status_value(str(row.get(field, ""))))
    if found:
        return list(dict.fromkeys(found))
    if relative.startswith("case_A/"):
        return ["A"]
    if relative.startswith("case_B/"):
        return ["B"]
    if relative.startswith("cross_case/transfer_T1_A_to_B"):
        return ["A", "B"]
    if relative.startswith("cross_case/transfer_T2_history_A_to_B"):
        return ["A", "B"]
    return []


def _protected(item: dict, relative: str) -> bool:
    deps = item.get("dependencies", [])
    if "C" in deps:
        return False
    # An unlabelled row is not silently treated as protected.  The capture
    # script labels all current protected files, but fail closed if a future
    # table loses its dependency metadata.
    return bool(deps) and set(deps).issubset({"A", "B"})


def compare(baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    compared = 0
    protected_rows = 0
    skipped_c_rows = 0
    for relative, items in baseline.get("rows", {}).items():
        current_path = C.EXT / relative
        if not current_path.is_file():
            failures.append(f"missing current table: {relative}")
            continue
        current = pd.read_csv(current_path)
        if len(current) != len(items):
            failures.append(f"{relative}: row count {len(current)} != {len(items)}")
            continue
        for index, item in enumerate(items):
            old_row = item.get("record", {})
            new_row = current.iloc[index].to_dict()
            new_deps = _dependencies(relative, new_row)
            if "C" in item.get("dependencies", []) or "C" in new_deps:
                skipped_c_rows += 1
                continue
            if not _protected(item, relative) or not new_deps or "C" in new_deps:
                failures.append(f"{relative}[row {index}]: dependency scope is not protected")
                continue
            protected_rows += 1
            for column, old_value in old_row.items():
                if column in C.OUTPUT_METADATA_FIELDS:
                    continue
                if column not in current.columns:
                    failures.append(f"{relative}[row {index}]: missing column {column}")
                    continue
                compared += 1
                new_value = new_row[column]
                if not _equal(old_value, new_value):
                    failures.append(
                        f"{relative}[row {index}] {column}: old={old_value!r}, new={new_value!r}")
    return {"compared_fields": compared, "protected_rows": protected_rows,
            "skipped_C_rows": skipped_c_rows, "failures": failures}


def audit(baseline_path: str | Path, output: str | Path) -> dict:
    baseline = Path(baseline_path).expanduser().resolve()
    if not baseline.is_file():
        raise FileNotFoundError(f"missing future-C baseline: {baseline}")
    comparison = compare(baseline)
    passed = not comparison["failures"] and comparison["protected_rows"] > 0
    result = {
        "pass": passed,
        "SCIENTIFIC_NUMERIC_DRIFT": "NONE" if passed else "FOUND",
        "baseline": str(baseline),
        "case_C_status": C.case_metadata("C")["case_status"],
        "case_C_science_eligible": C.case_is_science_eligible("C"),
        "comparison": comparison,
    }
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SCIENTIFIC_REGRESSION_AUDIT — future Case-C promotion", "",
             "This audit protects the pre-existing A/B and A→B scientific numeric scope. "
             "C-dependent rows are retained in the baseline but are intentionally "
             "excluded from numeric comparison because real Dayekou values may differ.", "",
             f"* Case C live status: `{result['case_C_status']}`",
             f"* Baseline: `{baseline}`",
             f"* Protected rows compared: {comparison['protected_rows']}",
             f"* Numeric fields compared: {comparison['compared_fields']}",
             f"* C-dependent rows skipped by design: {comparison['skipped_C_rows']}",
             f"* SCIENTIFIC_NUMERIC_DRIFT = {result['SCIENTIFIC_NUMERIC_DRIFT']}",
             f"* Overall: **{'PASS' if passed else 'FAIL'}**", ""]
    if comparison["failures"]:
        lines += ["Unexpected differences:"]
        lines.extend(f"* {failure}" for failure in comparison["failures"])
    else:
        lines.append("No protected A/B or A→B numeric differences were found.")
    lines += ["", f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.write_json(out.with_suffix(".json"), result, cases=list(C.CASE_ORDER))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=str(
        C.QA / "future_c_promotion_regression_baseline_20260829.json"))
    ap.add_argument("--output", default=str(
        C.EXT / "audit" / "SCIENTIFIC_REGRESSION_AUDIT.md"))
    args = ap.parse_args()
    return 0 if audit(args.baseline, args.output)["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
