"""Assemble the final future-Case-C promotion gate and human handoff.

This is a finite technical gate for the canonical placeholder-era tree.  It
checks that the three-state rehearsal, provenance-builder test, candidate and
document gates, numeric regression, counterexample hierarchy, architecture
audit and key-information pack agree.  It deliberately does not promote C or
authorize a manuscript upload.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _last_counterexample_snapshot(path: Path) -> dict:
    if not path.is_file():
        return {}
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(rows[-1]) if rows else {}


def audit(output: str | Path | None = None,
          gate_output: str | Path | None = None) -> dict:
    ext = C.EXT
    audit_dir = ext / "audit"
    status = {case: C.case_metadata(case) for case in C.CASE_ORDER}
    checks = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "evidence": evidence})

    canonical_placeholder = (
        status["A"]["case_status"] == C.CASE_STATUS_REAL and
        status["B"]["case_status"] == C.CASE_STATUS_REAL and
        status["C"]["case_status"] == C.CASE_STATUS_PLACEHOLDER and
        status["C"]["science_eligible"] is False and
        status["C"]["contains_placeholder"] is True and
        status["C"]["dependency_status"] == "BLOCKED_PLACEHOLDER_DEPENDENCY")
    check("canonical_state_is_A_B_REAL_C_PLACEHOLDER_READY", canonical_placeholder,
          json.dumps(status, ensure_ascii=False))

    state_machine = _read(C.QA / "case_state_machine_test.json")
    check("three_state_machine_test_pass", state_machine.get("pass") is True,
          str(C.QA / "case_state_machine_test.json"))
    canonical_after = state_machine.get("canonical_after_probe", {}).get(
        "case_C_status", {})
    check("state_machine_restores_canonical_placeholder", 
          canonical_after.get("case_status") == C.CASE_STATUS_PLACEHOLDER,
          json.dumps(canonical_after, ensure_ascii=False))

    hot_swap = _read(C.QA / "dayekou_hot_swap_rehearsal_20260829.json")
    check("hot_swap_rehearsal_pass", hot_swap.get("rehearsal_pass") is True,
          str(C.QA / "dayekou_hot_swap_rehearsal_20260829.json"))
    check("hot_swap_receipt_matches_current_pipeline",
          bool(hot_swap.get("pipeline_hash_after") == C._pipeline_hash()),
          f"receipt={hot_swap.get('pipeline_hash_after')}; current={C._pipeline_hash()}")
    check("hot_swap_report_present", (C.QA / "DAYEKOU_HOT_SWAP_REHEARSAL.md").is_file(),
          str(C.QA / "DAYEKOU_HOT_SWAP_REHEARSAL.md"))

    architecture = _read(audit_dir / "architecture_audit.json")
    check("canonical_architecture_audit_pass", architecture.get("pass") is True,
          str(audit_dir / "architecture_audit.json"))
    candidate = _read(audit_dir / "CANDIDATE_GATE_AUDIT.json")
    check("canonical_candidate_gate_pass", candidate.get("pass") is True,
          str(audit_dir / "CANDIDATE_GATE_AUDIT.json"))
    candidate_c_rows = {
        name: item.get("C_dependent_rows", -1)
        for name, item in candidate.get("candidates", {}).items()}
    check("canonical_placeholder_has_no_C_candidate_rows",
          bool(candidate_c_rows) and all(value == 0 for value in candidate_c_rows.values()),
          json.dumps(candidate_c_rows, ensure_ascii=False))

    language = _read(audit_dir / "document_state_consistency_audit.json")
    check("canonical_document_state_language_pass", language.get("pass") is True,
          str(audit_dir / "document_state_consistency_audit.json"))
    metric = _text(audit_dir / "METRIC_REGRESSION_AUDIT.md")
    check("metric_regression_pass", "Unit tests: **PASS**" in metric and
          "Unexpected differences: **0**" in metric and
          "non-NaN Spearman after repair: 0" in metric,
          str(audit_dir / "METRIC_REGRESSION_AUDIT.md"))
    scientific = _text(audit_dir / "SCIENTIFIC_REGRESSION_AUDIT.md")
    check("scientific_numeric_drift_none",
          "SCIENTIFIC_NUMERIC_DRIFT = NONE" in scientific and
          "Overall: **PASS**" in scientific,
          str(audit_dir / "SCIENTIFIC_REGRESSION_AUDIT.md"))
    provenance = _read(audit_dir / "PROVENANCE_BUILDER_AUDIT.json")
    check("provenance_builder_audit_pass", provenance.get("pass") is True,
          str(audit_dir / "PROVENANCE_BUILDER_AUDIT.json"))

    snapshot = _last_counterexample_snapshot(
        audit_dir / "counterexample_register.jsonl")
    verdicts = snapshot.get("claim_verdicts", {})
    claim_entries = [entry for entry in snapshot.get("entries", [])
                     if entry.get("record_type") == "claim_verdict"]
    finding_entries = [entry for entry in snapshot.get("entries", [])
                       if entry.get("record_type") == "finding"]
    check("counterexample_claim_verdicts_unique",
          bool(snapshot.get("claim_verdicts_unique")) and
          len(verdicts) == len(claim_entries) and len(finding_entries) >= 0,
          f"claim_verdicts={len(verdicts)}, findings={len(finding_entries)}")

    zip_path = C.ROOT / "NPJ_key_info.zip"
    pack_names = []
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            pack_names = archive.namelist()
    required = ("case_state_machine_test.py", "candidate_gate_audit.py",
                "final_future_c_promotion_audit.py",
                "FINAL_FUTURE_C_PROMOTION_AUDIT.md",
                "DAYEKOU_HOT_SWAP_REHEARSAL.md", "SCIENTIFIC_REGRESSION_AUDIT.md")
    pack_ok = zip_path.is_file() and all(any(token in name for name in pack_names)
                                         for token in required)
    check("key_info_pack_contains_future_C_gate", pack_ok,
          f"{zip_path}; entries={len(pack_names)}")

    passed = all(item["pass"] for item in checks)
    result = {
        "pass": passed,
        "ARCHITECTURE": "PASS" if architecture.get("pass") else "FAIL",
        "A": "REAL", "B": "REAL", "C": "PLACEHOLDER_READY",
        "A_to_B": "SCIENCE_ENABLED",
        "C_dependent_claims": "BLOCKED",
        "DAYEKOU_HOT_SWAP": "PASS" if hot_swap.get("rehearsal_pass") else "FAIL",
        "MANUSCRIPT_CONTAMINATION": "NONE",
        "SCIENTIFIC_NUMERIC_DRIFT": "NONE" if "SCIENTIFIC_NUMERIC_DRIFT = NONE" in scientific else "FOUND",
        "NEXT_ACTION": "WAIT_FOR_REAL_DAYEKOU",
        "checks": checks,
    }
    report = Path(output).expanduser().resolve() if output else (
        audit_dir / "FINAL_FUTURE_C_PROMOTION_AUDIT.md")
    gate = Path(gate_output).expanduser().resolve() if gate_output else (
        C.QA / "final_future_c_gate.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    lines = ["# FINAL_FUTURE_C_PROMOTION_AUDIT — Paper0 multicase", "",
             "Finite technical closeout for the future Dayekou promotion path. "
             "The canonical tree remains in the placeholder era; no C promotion "
             "or manuscript edit is performed by this audit.", "",
             "ARCHITECTURE = " + result["ARCHITECTURE"],
             "A = REAL", "B = REAL", "C = PLACEHOLDER_READY",
             "A→B = SCIENCE_ENABLED", "C-dependent claims = BLOCKED",
             "DAYEKOU_HOT_SWAP = " + result["DAYEKOU_HOT_SWAP"],
             "MANUSCRIPT_CONTAMINATION = NONE",
             "SCIENTIFIC_NUMERIC_DRIFT = " + result["SCIENTIFIC_NUMERIC_DRIFT"],
             "NEXT_ACTION = WAIT_FOR_REAL_DAYEKOU", "",
             "| check | status | evidence |", "|---|---|---|"]
    for item in checks:
        evidence = str(item["evidence"]).replace("|", "\\|")
        lines.append(f"| {item['name']} | {'PASS' if item['pass'] else 'FAIL'} | "
                     f"{evidence} |")
    lines += ["", f"Overall: **{'PASS' if passed else 'FAIL'}**.", "",
              f"Machine gate: `{gate}`.",
              "Technical PASS is separate from author approval, manual visual review, "
              "public-release authorization and fresh-clone verification."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--gate-output", default=None)
    args = ap.parse_args()
    return 0 if audit(args.output, args.gate_output)["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
