"""Run the bidirectional manuscript-candidate boundary audit.

The forward direction rejects placeholder/pending dependencies.  The reverse
direction requires C-dependent rows in every candidate scope whose contract
promises all real cases once the live registry promotes C.  The report is
written beside the other Paper0 extension audits and never edits manuscript
sources.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
from export_manuscript_source_data import audit_directory  # noqa: E402


def _dependencies(row: pd.Series) -> list[str]:
    if "case" in row.index and str(row.get("case", "")) in C.CASE_ORDER:
        return [str(row["case"])]
    for field in ("case_status", "data_status"):
        deps = C._cases_from_status_value(str(row.get(field, "")))
        if deps:
            return deps
    return []


def _audit_one(root: Path) -> dict:
    name = root.name
    spec = C.CANDIDATE_SPECS[name]
    result = audit_directory(root)
    c_rows = 0
    unknown_rows = 0
    eligible_rows = 0
    stale_rows = []
    for path in sorted(root.glob("**/*.csv")) if root.is_dir() else []:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for index, row in frame.iterrows():
            deps = _dependencies(row)
            if not deps:
                unknown_rows += 1
                continue
            if "C" in deps:
                c_rows += 1
            eligible_rows += 1
            expected = C.status_metadata(deps)
            for field in C.OUTPUT_METADATA_FIELDS:
                actual = row.get(field)
                if field in {"science_eligible", "contains_placeholder"}:
                    actual = str(actual).strip().lower() in {"true", "1", "yes"}
                if str(actual) != str(expected[field]) and not (
                        field in {"science_eligible", "contains_placeholder"}
                        and bool(actual) == bool(expected[field])):
                    stale_rows.append({"file": str(path), "row": int(index),
                                       "field": field, "actual": actual,
                                       "expected": expected[field],
                                       "dependencies": deps})
    c_real = C.case_is_science_eligible("C")
    direction_forward = bool(result.get("pass") and unknown_rows == 0 and
                             (c_real or c_rows == 0))
    direction_reverse = (not spec.get("requires_case_C_when_real") or
                         not c_real or c_rows > 0)
    passed = bool(direction_forward and direction_reverse and not stale_rows)
    return {
        "root": str(root), "selection_scope": spec["selection_scope"],
        "live_C_status": C.case_metadata("C")["case_status"],
        "live_C_science_eligible": c_real,
        "candidate_audit": result,
        "rows_with_known_dependencies": eligible_rows,
        "C_dependent_rows": c_rows,
        "unknown_dependency_rows": unknown_rows,
        "stale_registry_rows": stale_rows,
        "forward_blocked_scope_check": bool(direction_forward),
        "reverse_real_case_completeness_check": bool(direction_reverse),
        "pass": passed,
    }


def audit(output: str | Path | None = None,
          extension: str | Path | None = None,
          qa_dir: str | Path | None = None,
          registry_path: str | Path | None = None) -> dict:
    C.set_output_dirs(extension, qa_dir, registry_path)
    results = {_name: _audit_one(C.ext_dir(_name))
               for _name in ("comment_candidate", "article_candidate")}
    passed = all(item["pass"] for item in results.values())
    result = {"pass": passed, "case_C": C.case_metadata("C"),
              "candidates": results}
    out = Path(output).expanduser().resolve() if output else (
        C.ext_dir("audit") / "CANDIDATE_GATE_AUDIT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CANDIDATE_GATE_AUDIT — Paper0 multicase", "",
             "The candidate gate is bidirectional: blocked C dependencies cannot "
             "enter, and a promoted C cannot be silently omitted from an all-real "
             "candidate scope.", "",
             f"* Live Case C: `{result['case_C']['case_status']}`",
             f"* Overall: **{'PASS' if passed else 'FAIL'}**", "",
             "| candidate | scope | C rows | unknown rows | forward gate | reverse gate | status |",
             "|---|---|---:|---:|---|---|---|"]
    for name, item in results.items():
        lines.append(f"| {name} | {item['selection_scope']} | "
                     f"{item['C_dependent_rows']} | {item['unknown_dependency_rows']} | "
                     f"{'PASS' if item['forward_blocked_scope_check'] else 'FAIL'} | "
                     f"{'PASS' if item['reverse_real_case_completeness_check'] else 'FAIL'} | "
                     f"{'PASS' if item['pass'] else 'FAIL'} |")
        if item["stale_registry_rows"]:
            lines.append(f"* `{name}` stale rows: {item['stale_registry_rows']}")
    lines += ["", "The audit covers generated candidate CSVs only; it does not "
              "modify either Paper0 manuscript directory."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.write_json(out.with_suffix(".json"), result, cases=list(C.CASE_ORDER))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=C._json_default))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--extension", default=None)
    ap.add_argument("--qa-dir", default=None)
    ap.add_argument("--registry-path", default=None)
    args = ap.parse_args()
    return 0 if audit(args.output, args.extension, args.qa_dir,
                      args.registry_path)["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
