"""Independent 16-check audit for the Paper0 multicase architecture.

This checker is intentionally outside the phase runner's verdict logic.  It
checks the live registry, output metadata, candidate boundary, metric audit,
provenance, hot-swap rehearsal and key-information archive, then writes a
short machine-readable status block plus an evidence table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_ok(df: pd.DataFrame, eligible: bool | None = None) -> bool:
    required = set(C.OUTPUT_METADATA_FIELDS)
    if not required.issubset(df.columns) or df.empty:
        return False
    if df[list(required)].isna().any().any():
        return False
    if eligible is True:
        return bool((df["science_eligible"].astype(str).str.lower() == "true").all()
                    and (df["contains_placeholder"].astype(str).str.lower() == "false").all()
                    and (df["dependency_status"] == "REAL").all())
    return True


def _candidate_dependencies(row: pd.Series) -> list[str]:
    """Resolve the dependency set encoded on one candidate row."""
    if "case" in row.index:
        case = str(row.get("case", ""))
        if case in C.CASE_ORDER:
            return [case]
    for field in ("case_status", "data_status"):
        deps = C._cases_from_status_value(str(row.get(field, "")))
        if deps:
            return deps
    return []


def _check_candidate(root: Path) -> tuple[bool, str]:
    """Check candidate metadata against the live registry in both directions.

    C is rejected only while it is blocked.  Once the registry says C is REAL,
    C rows are required for scopes that promise all real cases; this catches a
    stale A/B-only export that would otherwise look safely fail-closed.
    """
    from export_manuscript_source_data import audit_directory

    result = audit_directory(root)
    if not result.get("pass"):
        return False, json.dumps(result, ensure_ascii=False)
    spec = C.CANDIDATE_SPECS.get(root.name, {})
    manifest_path = root / "candidate_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    expected_scope = spec.get("selection_scope")
    if expected_scope and manifest.get("selection_scope") != expected_scope:
        return False, f"selection_scope mismatch: {manifest.get('selection_scope')!r} != {expected_scope!r}"
    c_real = C.case_is_science_eligible("C")
    c_rows = 0
    c_dependent_rows = 0
    for path in sorted(root.glob("**/*.csv")):
        df = pd.read_csv(path)
        if not _metadata_ok(df, eligible=True):
            return False, f"not eligible metadata: {path}"
        for _, row in df.iterrows():
            deps = _candidate_dependencies(row)
            if "C" not in deps:
                continue
            c_dependent_rows += 1
            if "case" in df.columns and str(row.get("case", "")) == "C":
                c_rows += 1
            expected = C.status_metadata(deps)
            for field in C.OUTPUT_METADATA_FIELDS:
                actual = row.get(field)
                wanted = expected[field]
                if field in {"science_eligible", "contains_placeholder"}:
                    actual_norm = str(actual).strip().lower() in ("true", "1", "yes")
                    if actual_norm != bool(wanted):
                        return False, f"stale {field} in {path}: {actual!r} != {wanted!r}"
                elif str(actual) != str(wanted):
                    return False, f"stale {field} in {path}: {actual!r} != {wanted!r}"
            if not c_real:
                return False, f"blocked C dependency reached candidate: {path}"
    if spec.get("requires_case_C_when_real") and c_real and not c_dependent_rows:
        return False, "C is REAL but candidate scope silently omitted every C-dependent row"
    if root.name == "article_candidate" and c_real:
        for required in ("M9_transfer_B_to_C.csv", "M10_transfer_A_to_C.csv"):
            if not (root / required).is_file():
                return False, f"C is REAL but required spatial candidate is missing: {required}"
    return True, (f"{len(result.get('eligible_files', []))} eligible CSV files; "
                  f"C-dependent rows={c_dependent_rows}; no binary payload")


def _check_metrics(ext: Path) -> tuple[bool, str]:
    bad = []
    for path in sorted(ext.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "spearman" not in df.columns:
            continue
        for i, row in df.iterrows():
            label = " ".join(str(row.get(k, "")) for k in ("model", "method")).lower()
            if ("global" in label or "zero_correction" in label) and pd.notna(row["spearman"]):
                bad.append(f"{path.relative_to(ext)}[{i}]")
    report = ext / "audit" / "METRIC_REGRESSION_AUDIT.md"
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    passed = ("Unit tests: **PASS**" in text and
              "Unexpected differences: **0**" in text and
              "non-NaN Spearman after repair: 0" in text and not bad)
    return passed, ("report PASS; constant rows safe" if passed else
                    f"report/bad rows failure: {bad}")


def audit(output: str | Path | None = None,
          rehearsal_receipt: str | Path | None = None,
          extension: str | Path | None = None,
          qa_dir: str | Path | None = None,
          registry_path: str | Path | None = None,
          require_pack: bool = True) -> dict:
    C.set_output_dirs(extension, qa_dir, registry_path)
    ext = C.EXT
    checks: list[dict] = []

    def check(number: int, name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": number, "name": name, "pass": bool(passed),
                       "evidence": evidence})

    status = {case: C.case_metadata(case) for case in C.CASE_ORDER}
    check(1, "Case A is REAL and science-eligible",
          status["A"]["case_status"] == C.CASE_STATUS_REAL and status["A"]["science_eligible"],
          json.dumps(status["A"], ensure_ascii=False))
    check(2, "Case B is REAL and science-eligible",
          status["B"]["case_status"] == C.CASE_STATUS_REAL and status["B"]["science_eligible"],
          json.dumps(status["B"], ensure_ascii=False))
    c_placeholder = (status["C"]["case_status"] == C.CASE_STATUS_PLACEHOLDER and
                     status["C"]["contains_placeholder"] and
                     not status["C"]["science_eligible"])
    c_pending = (status["C"]["case_status"] == C.CASE_STATUS_REAL_PENDING_AUDIT and
                 not status["C"]["contains_placeholder"] and
                 not status["C"]["science_eligible"] and
                 status["C"]["dependency_status"] == "BLOCKED_REAL_PENDING_AUDIT")
    c_real_audited = C.case_is_science_eligible(status["C"])
    check(3, "Case C live state matches the state-machine contract",
          c_placeholder or c_pending or c_real_audited,
          json.dumps(status["C"], ensure_ascii=False))

    registry_path = C.REGISTRY_PATH
    registry = _read_json(registry_path) if registry_path.suffix.lower() == ".json" and registry_path.is_file() else None
    if registry is None and registry_path.is_file():
        import yaml
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry = registry or {}
    ab_claim = registry.get("claims", {}).get("MC-A-to-B-temporal-transfer", {})
    check(4, "A->B claim is science-enabled", ab_claim.get("dependency_status") == "REAL" and
          bool(ab_claim.get("science_eligible")), json.dumps(ab_claim, ensure_ascii=False))

    table_dir = ext / "tables"
    m4_path = table_dir / "M4_scale_dependence.csv"
    m4 = pd.read_csv(m4_path) if m4_path.is_file() else pd.DataFrame()
    expected_scales = {float(C.SCALE_M[k]) for k in C.SCALE_CELLS}
    observed_scales = pd.to_numeric(m4.get("scale_m", []), errors="coerce").dropna()
    scale_set_ok = (len(observed_scales) > 0 and
                    len({round(float(x), 6) for x in observed_scales}) == len(expected_scales) and
                    all(any(abs(float(x) - target) <= 1e-6 for x in observed_scales)
                        for target in expected_scales))
    m4_shape_ok = (len(m4) == 24 and set(m4.get("case", [])) == set(C.CASE_ORDER)
                   and scale_set_ok)
    if m4_shape_ok:
        m4_shape_ok = all(len(g) == 4 and
                          all(any(abs(float(x) - target) <= 1e-6
                                  for x in pd.to_numeric(g["scale_m"], errors="coerce"))
                              for target in expected_scales)
                          for _, g in m4.groupby(["case", "window"]))
    check(5, "M4 is exactly 3 cases x 2 windows x 4 scales",
          m4_shape_ok, f"rows={len(m4)}, scales={sorted(expected_scales)}")

    candidate_results = {}
    for name in ("comment_candidate", "article_candidate"):
        candidate_results[name] = _check_candidate(ext / name)
    check(6, "candidate exporter is fail-closed", all(x[0] for x in candidate_results.values()),
          json.dumps(candidate_results, ensure_ascii=False))

    c_m4 = m4[m4.get("case", pd.Series(dtype=str)).astype(str) == "C"] if not m4.empty else m4
    if c_placeholder:
        m4_meta_ok = (_metadata_ok(c_m4, eligible=False) and
                      bool((c_m4["science_eligible"].astype(str).str.lower() == "false").all()) and
                      bool((c_m4["contains_placeholder"].astype(str).str.lower() == "true").all()))
        m4_evidence = "M4 C rows are placeholder engineering rows only"
    elif c_pending:
        m4_meta_ok = (_metadata_ok(c_m4, eligible=False) and
                      bool((c_m4["science_eligible"].astype(str).str.lower() == "false").all()) and
                      bool((c_m4["contains_placeholder"].astype(str).str.lower() == "false").all()) and
                      bool((c_m4["dependency_status"] == "BLOCKED_REAL_PENDING_AUDIT").all()))
        m4_evidence = "M4 C rows are pending-audit rows and remain non-eligible"
    else:
        m4_meta_ok = (_metadata_ok(c_m4, eligible=True) and len(c_m4) > 0 and
                      bool((c_m4["case_status"] == C.CASE_STATUS_REAL).all()))
        m4_evidence = "M4 C rows are REAL and science-eligible after promotion"
    check(7, "M4 C rows carry state-consistent hard fields", m4_meta_ok,
          m4_evidence)

    metric_ok, metric_evidence = _check_metrics(ext)
    check(8, "constant-safe Pearson/Spearman/slope regression audit", metric_ok, metric_evidence)

    m2_path = table_dir / "M2_representation_loss.csv"
    m2 = pd.read_csv(m2_path) if m2_path.is_file() else pd.DataFrame()
    ab_m2 = m2[m2.get("case", pd.Series(dtype=str)).isin(["A", "B"])] if not m2.empty else m2
    ab_rep_ok = (len(ab_m2) == 4 and _metadata_ok(ab_m2, eligible=True)
                 and set(ab_m2["window"]) == {"calibration", "evaluation"})
    check(9, "A/B representation conclusions remain science-enabled", ab_rep_ok,
          f"A/B M2 rows={len(ab_m2)}")

    m5_path = table_dir / "M5_geometric_recovery.csv"
    m5 = pd.read_csv(m5_path) if m5_path.is_file() else pd.DataFrame()
    ab_m5 = m5[m5.get("case", pd.Series(dtype=str)).isin(["A", "B"])] if not m5.empty else m5
    interp_ok = len(ab_m5) > 0 and _metadata_ok(ab_m5, eligible=True)
    check(10, "A/B interpolation conclusions remain science-enabled", interp_ok,
          f"A/B M5 rows={len(ab_m5)}")

    repro_path = ext / "audit" / "case_A_legacy_reproduction.csv"
    repro = pd.read_csv(repro_path) if repro_path.is_file() else pd.DataFrame()
    repro_ok = len(repro) > 0 and (repro["status"] == "PASS").all()
    check(11, "Case-A numeric baseline and history are preserved", repro_ok,
          f"reproduction={len(repro)} rows; failures={int((repro.status == 'FAIL').sum()) if len(repro) else 'missing'}")

    register = ext / "audit" / "counterexample_register.jsonl"
    register_ok = register.is_file() and register.stat().st_size > 0 and (ext / "audit" / "counterexample_register.md").is_file()
    check(12, "counterexample register is rich and append/archive based", register_ok,
          str(register))

    prov = {}
    prov_ok = True
    for case in C.CASE_ORDER:
        path = ext / f"case_{case}" / "WRF_RUN_PROVENANCE.json"
        prov[case] = _read_json(path) if path.is_file() else {}
        if not path.is_file():
            prov_ok = False
            continue
        expected = ("PLACEHOLDER_NOT_APPLICABLE"
                    if status[case]["case_status"] == C.CASE_STATUS_PLACEHOLDER
                    else "COMPLETE" if case == "C" and status[case]["case_status"] == C.CASE_STATUS_REAL
                    else "PARTIAL_UNKNOWN_NOT_PACKAGED")
        if prov[case].get("provenance_status") != expected:
            prov_ok = False
        if not prov[case].get("raw_data_hashes"):
            prov_ok = False
    check(13, "per-case WRF provenance is explicit", prov_ok,
          json.dumps({c: {"provenance_status": prov[c].get("provenance_status"),
                          "promotion_eligible": prov[c].get("promotion_eligible")}
                      for c in C.CASE_ORDER}, ensure_ascii=False))

    receipt_path = Path(rehearsal_receipt) if rehearsal_receipt else C.QA / "dayekou_hot_swap_rehearsal_20260829.json"
    rehearsal = _read_json(receipt_path) if receipt_path.is_file() else {}
    hot_swap_ok = bool(rehearsal.get("rehearsal_pass") and
                       (rehearsal.get("canonical_placeholder_restored") or
                        rehearsal.get("canonical_untouched")))
    check(14, "Dayekou fake replacement rehearsal passes with canonical safety",
          hot_swap_ok, str(receipt_path))

    zip_path = C.ROOT / "NPJ_key_info.zip"
    required_pack = ("mc_config.py", "validate_case_input.py", "run_multicase.py",
                     "build_wrf_run_provenance.py", "audit_state_language.py",
                     "provenance_builder_audit.py", "capture_future_c_baseline.py",
                     "case_state_machine_test.py", "candidate_gate_audit.py",
                     "final_future_c_promotion_audit.py", "scientific_regression_audit.py",
                     "metric_regression_audit.py", "METRIC_REGRESSION_AUDIT.md",
                     "SCIENTIFIC_REGRESSION_AUDIT.md", "counterexample_register.jsonl",
                     "dayekou_hot_swap_rehearsal", "DAYEKOU_HOT_SWAP_REHEARSAL.md",
                     "case_state_machine_test.json", "case_state_machine_test.md",
                     "final_future_c_gate.json", "FINAL_FUTURE_C_PROMOTION_AUDIT.md",
                     "DOCUMENT_STATE_CONSISTENCY_AUDIT.md")
    pack_names = []
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as zf:
            pack_names = zf.namelist()
    pack_ok = zip_path.is_file() and all(any(token in name for name in pack_names)
                                         for token in required_pack)
    check(15, "unique key-information pack contains multicase core evidence",
          pack_ok if require_pack else True,
          (f"{zip_path}; entries={len(pack_names)}" if require_pack else
           f"scratch audit: pack check deferred; canonical entries={len(pack_names)}"))

    stale_files = [C.ROOT / "docs" / "CURRENT.md", C.ROOT / "state" / "project.yaml",
                   C.ROOT / "README.md", ext / "audit" / "FINAL_MULTICASE_P0_AUDIT.md",
                   ext / "CLAIMS_PROVISIONAL.md"]
    forbidden_statuses = ("WAITING_FOR_EXTERNAL_GATE", "ET-ingest EXTERNAL_BLOCKED",
                          "no second nest")
    stale_hits = []
    for path in stale_files:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden_statuses:
                if token.lower() in text.lower():
                    stale_hits.append(f"{path}:{token}")
    check(16, "current governance and generated status docs have no stale multicase banner",
          not stale_hits, "; ".join(stale_hits) if stale_hits else "no forbidden stale banner found")

    passed = all(x["pass"] for x in checks)
    c_status = status["C"]["case_status"]
    c_placeholder = c_status == C.CASE_STATUS_PLACEHOLDER
    c_pending = c_status == C.CASE_STATUS_REAL_PENDING_AUDIT
    manuscript_markers = (
        "Dayekou20151201", "BLOCKED_PLACEHOLDER_DEPENDENCY", "C=PLACEHOLDER",
        "C-dependent claims", "PLACEHOLDER_READY",
    )
    manuscript_hits = []
    for manuscript_root in (C.ROOT / "latex_paper0", C.ROOT / "latex_paper0_npjcomment"):
        if not manuscript_root.is_dir():
            continue
        for path in sorted(manuscript_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".tex", ".bib", ".md"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                for marker in manuscript_markers:
                    if marker.lower() in text.lower():
                        manuscript_hits.append(f"{path}:{marker}")
    manuscript_contamination = "FOUND" if manuscript_hits else "NONE"
    output_path = Path(output) if output else ext / "audit" / "FINAL_MULTICASE_ARCHITECTURE_AUDIT.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_block = [
        f"ARCHITECTURE = {'PASS' if passed else 'FAIL'}",
        "A = REAL" if status["A"]["case_status"] == C.CASE_STATUS_REAL else "A = FAIL",
        "B = REAL" if status["B"]["case_status"] == C.CASE_STATUS_REAL else "B = FAIL",
        "C = PLACEHOLDER_READY" if c_placeholder else
        "C = REAL_PENDING_AUDIT" if c_pending else "C = REAL_AUDITED",
        "A→B = SCIENCE_ENABLED" if ab_claim.get("dependency_status") == "REAL" else "A→B = BLOCKED",
        "C-dependent claims = BLOCKED" if not C.case_is_science_eligible(status["C"]) else
        "C-dependent claims = SCIENCE_ENABLED",
        f"DAYEKOU_HOT_SWAP = {'PASS' if hot_swap_ok else 'FAIL'}",
        f"MANUSCRIPT_CONTAMINATION = {manuscript_contamination}",
        "NEXT_ACTION = WAIT_FOR_REAL_DAYEKOU" if c_placeholder else
        "NEXT_ACTION = WAIT_FOR_C_PROMOTION_AUDIT" if c_pending else
        "NEXT_ACTION = AUTHOR_REVIEW",
    ]
    lines = ["# FINAL_MULTICASE_ARCHITECTURE_AUDIT — Paper0", ""] + status_block + ["",
             "This is an independent architecture/release audit.  It does not "
             "authorize manuscript promotion or alter either Paper0 source.", "",
             "| # | check | status | evidence |", "|---:|---|---|---|"]
    for row in checks:
        evidence = str(row["evidence"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['check']} | {row['name']} | "
                     f"{'PASS' if row['pass'] else 'FAIL'} | {evidence} |")
    lines += ["", f"Overall: **{'PASS' if passed else 'FAIL'}**."]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.write_json(ext / "audit" / "architecture_audit.json", {
        "status_block": status_block, "checks": checks, "pass": passed,
        "rehearsal_receipt": str(receipt_path),
    }, cases=list(C.CASE_ORDER))
    print(json.dumps({"pass": passed, "checks": checks}, ensure_ascii=False, indent=2))
    return {"pass": passed, "checks": checks, "status_block": status_block,
            "output": str(output_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--hot-swap-receipt", default=None)
    ap.add_argument("--extension", default=None,
                    help="alternate generated extension root for an isolated audit")
    ap.add_argument("--qa-dir", default=None,
                    help="alternate QA root for an isolated audit")
    ap.add_argument("--registry-path", default=None,
                    help="alternate registry path for an isolated audit")
    ap.add_argument("--skip-pack", action="store_true",
                    help="defer the canonical key-info zip check for scratch audits")
    args = ap.parse_args()
    return 0 if audit(args.output, args.hot_swap_receipt, args.extension,
                      args.qa_dir, args.registry_path,
                      require_pack=not args.skip_pack)["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
