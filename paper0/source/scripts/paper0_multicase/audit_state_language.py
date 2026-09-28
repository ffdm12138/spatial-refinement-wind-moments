"""Audit generated multicase documents against the live case-state registry.

This is a deliberately narrow language scanner.  It checks current generated
status documents and candidate manifests, while leaving the preregistration
and historical archive untouched.  The scanner is designed to catch the
failure mode where a real Case C has been promoted but an old placeholder
sentence remains in a generated README or provenance page.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402


DOCUMENTS = (
    "README.md",
    "PROVENANCE.md",
    "CLAIMS_PROVISIONAL.md",
    "MANIFEST.md",
    "raw_contract/data_lineage.md",
    "audit/FINAL_MULTICASE_P0_AUDIT.md",
    "audit/FINAL_MULTICASE_ARCHITECTURE_AUDIT.md",
    "audit/CANDIDATE_GATE_AUDIT.md",
    "audit/SCIENTIFIC_REGRESSION_AUDIT.md",
    "audit/PROVENANCE_BUILDER_AUDIT.md",
    "audit/promotion_gate.json",
    "comment_candidate/README.md",
    "comment_candidate/candidate_manifest.json",
    "article_candidate/README.md",
    "article_candidate/candidate_manifest.json",
)
OPTIONAL_DOCUMENTS = {"audit/PROVENANCE_BUILDER_AUDIT.md"}


def _status() -> str:
    return C.case_metadata("C")["case_status"]


def _violations(text: str, status: str) -> list[str]:
    hits: list[str] = []
    lines = text.splitlines()

    def add(pattern: str, description: str) -> None:
        regex = re.compile(pattern, re.IGNORECASE)
        for number, line in enumerate(lines, 1):
            if regex.search(line):
                hits.append(f"L{number}: {description}: {line.strip()}")

    # These are status claims, not generic policy words in a historical
    # preregistration.  The latter is intentionally outside DOCUMENTS.
    if status == C.CASE_STATUS_REAL:
        add(r"case\s*c[^\n]*(?:\bPLACEHOLDER_READY\b|\bPLACEHOLDER\b|synthetic[^\n]*\bplaceholder\b)",
            "real C retains placeholder wording")
        add(r"C\s*=\s*PLACEHOLDER(?:_READY)?", "real C status is placeholder")
        add(r"waiting\s+for\s+Dayekou", "real C still waits for Dayekou")
        add(r"C-dependent[^\n]*\bblocked\b", "real C dependency remains blocked")
        add(r"C-dependent[^\n]*candidate[^\n]*blocked", "real C candidate remains blocked")
    elif status == C.CASE_STATUS_REAL_PENDING_AUDIT:
        add(r"case\s*c[^\n]*(?:\bPLACEHOLDER_READY\b|\bPLACEHOLDER\b|synthetic[^\n]*\bplaceholder\b)",
            "pending C retains placeholder wording")
        add(r"case\s*c[^\n]*science_eligible\s*=\s*true", "pending C is marked eligible")
        add(r"case\s*c[^\n]*(?:status\s*:\s*REAL\b|case_status\s*=\s*REAL\b)",
            "pending C is marked final REAL")
        add(r"C\s*=\s*REAL_AUDITED", "pending C is marked audited")
    elif status == C.CASE_STATUS_PLACEHOLDER:
        add(r"case\s*c[^\n]*(?:status\s*:\s*REAL\b|case_status\s*=\s*REAL\b|REAL and audit-eligible)",
            "placeholder C is marked REAL")
        add(r"C\s*=\s*REAL(?:_AUDITED)?", "placeholder C status is REAL")
        add(r"C-dependent claims[^\n]*(?:enabled|available)",
            "placeholder C dependency is enabled")
        add(r"(?:A->C|B->C)[^\n]*(?:available|supported|enabled)",
            "placeholder spatial claim is enabled")
    else:
        add(r"case\s*c[^\n]*(?:science_eligible\s*=\s*true|PLACEHOLDER|REAL)",
            "unresolved C state has a final-status sentence")
    return hits


def audit(output: str | Path | None = None,
          extension: str | Path | None = None,
          qa_dir: str | Path | None = None,
          registry_path: str | Path | None = None) -> dict:
    C.set_output_dirs(extension, qa_dir, registry_path)
    ext = C.EXT
    status = _status()
    files = {}
    for relative in DOCUMENTS:
        path = ext / relative
        if not path.is_file():
            if relative in OPTIONAL_DOCUMENTS:
                files[relative] = {"exists": False, "pass": True,
                                   "violations": ["not available in this isolated state probe"]}
                continue
            files[relative] = {"exists": False, "pass": False,
                               "violations": ["missing generated document"]}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = _violations(text, status)
        files[relative] = {"exists": True, "pass": not hits,
                           "violations": hits}
    passed = all(item["pass"] for item in files.values())
    result = {"pass": passed, "case_C_status": status,
              "case_C_metadata": C.case_metadata("C"), "files": files}
    out = Path(output) if output else C.ext_dir("audit") / "DOCUMENT_STATE_CONSISTENCY_AUDIT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# DOCUMENT_STATE_CONSISTENCY_AUDIT — Paper0 multicase", "",
             f"Case C live status: `{status}`.",
             f"Overall: **{'PASS' if passed else 'FAIL'}**.", "",
             "The scanner checks current generated documents only; historical "
             "preregistration wording is not treated as current state.", "",
             "| document | status | violations |", "|---|---|---|"]
    for relative, item in files.items():
        detail = "<br>".join(str(x).replace("|", "\\|")
                             for x in item["violations"]) or "none"
        lines.append(f"| {relative} | {'PASS' if item['pass'] else 'FAIL'} | {detail} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.write_json(C.ext_dir("audit") / "document_state_consistency_audit.json",
                 result, cases=list(C.CASE_ORDER))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--extension", default=None)
    ap.add_argument("--qa-dir", default=None)
    ap.add_argument("--registry-path", default=None)
    args = ap.parse_args()
    result = audit(args.output, args.extension, args.qa_dir, args.registry_path)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
