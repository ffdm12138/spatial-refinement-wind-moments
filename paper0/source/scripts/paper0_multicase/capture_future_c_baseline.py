"""Capture the post-repair scientific baseline used by future Case-C audits.

The baseline protects the already-accepted A/B and A->B numeric scope.  Rows
whose dependency set contains C are retained for provenance, but the future-C
audit deliberately does not compare their numeric values: a real Dayekou
payload is allowed to change those rows while the fixed table schema remains
unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_numeric_baseline as legacy  # noqa: E402
import mc_config as C  # noqa: E402


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


def capture(output: str | Path | None = None) -> dict:
    target = Path(output).expanduser().resolve() if output else (
        C.QA / "future_c_promotion_regression_baseline_20260829.json")
    payload = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "protect post-repair A/B and A->B science while allowing future C values to change",
        "source_files": list(legacy.FILES),
        "metadata_fields_excluded_from_numeric_comparison": list(C.OUTPUT_METADATA_FIELDS),
        "rows": {},
        "missing": [],
    }
    for relative in legacy.FILES:
        path = C.EXT / relative
        if not path.is_file():
            payload["missing"].append(relative)
            continue
        frame = pd.read_csv(path)
        rows = []
        for row in frame.to_dict(orient="records"):
            rows.append({"dependencies": _dependencies(relative, row),
                         "record": row})
        payload["rows"][relative] = rows
    if payload["missing"]:
        raise FileNotFoundError(f"future-C baseline sources missing: {payload['missing']}")
    payload["protected_rows"] = sum(
        1 for rows in payload["rows"].values()
        for item in rows if "C" not in item["dependencies"])
    payload["C_dependent_rows_retained_for_audit_only"] = sum(
        1 for rows in payload["rows"].values()
        for item in rows if "C" in item["dependencies"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                 allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target),
                      "protected_rows": payload["protected_rows"],
                      "C_dependent_rows": payload["C_dependent_rows_retained_for_audit_only"]},
                     ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    capture(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
