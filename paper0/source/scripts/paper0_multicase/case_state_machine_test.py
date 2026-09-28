"""Exercise the Paper0 Case-C state machine in isolated scratch trees.

The probe asserts the three externally visible states:

* PLACEHOLDER_READY: C is blocked and no C row crosses the candidate gate;
* REAL_PENDING_AUDIT: a real-looking input is visible to engineering output,
  but promotion and candidate export remain blocked;
* REAL: a complete fake provenance payload promotes C and C rows are required
  in the candidate scopes.

The REAL branch delegates the expensive full hot-swap run to
``rehearse_dayekou_swap.py``.  All scratch trees use hard-linked raw inputs or
fresh generated outputs; no canonical raw payload is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402


def _link_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def _run(script: str, args: list[str]) -> dict:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name(script)), *args],
        cwd=str(C.ROOT), check=False, capture_output=True, text=True)
    return {"returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-2500:],
            "stderr_tail": completed.stderr[-2500:]}


def _copy_small_extension(source: Path, destination: Path) -> None:
    """Copy only the audit/table/candidate surface needed by a state probe."""
    for name in ("README.md", "PROVENANCE.md", "CLAIMS_PROVISIONAL.md",
                 "MANIFEST.md", "CASE_CONTRACT.json", "ANALYSIS_CONTRACT.json"):
        path = source / name
        if path.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for dirname in ("raw_contract", "tables", "cross_case", "bootstrap",
                    "audit", "comment_candidate", "article_candidate"):
        src = source / dirname
        if src.is_dir():
            shutil.copytree(src, destination / dirname)
    for case in C.CASE_ORDER:
        src = source / f"case_{case}" / "WRF_RUN_PROVENANCE.json"
        if src.is_file():
            target = destination / f"case_{case}" / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def _copy_status_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for case in C.CASE_ORDER:
        src = source / f"case_{case}_status.json"
        if src.is_file():
            shutil.copy2(src, destination / src.name)


def _candidate_c_rows(extension: Path) -> dict[str, int]:
    result = {}
    for candidate in ("comment_candidate", "article_candidate"):
        count = 0
        for path in (extension / candidate).glob("*.csv"):
            try:
                import pandas as pd
                frame = pd.read_csv(path)
            except Exception:
                continue
            if "case" in frame.columns:
                count += int((frame["case"].astype(str) == "C").sum())
            else:
                for field in ("case_status", "data_status"):
                    if field not in frame.columns:
                        continue
                    count += sum("C" in C._cases_from_status_value(str(value))
                                 for value in frame[field])
                    break
        result[candidate] = count
    return result


def _state_probes(scratch: Path, state_name: str) -> dict:
    canonical_ext, canonical_qa, canonical_registry = C.EXT, C.QA, C.REGISTRY_PATH
    ext = scratch / "extension_multicase_20260826"
    qa = scratch / "qa"
    state = scratch / "state"
    data = scratch / "data"
    ext.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    _copy_small_extension(canonical_ext, ext)
    _copy_status_files(canonical_qa, qa)
    registry_copy = state / "multicase_registry.yaml"
    if canonical_registry.is_file():
        shutil.copy2(canonical_registry, registry_copy)

    common = ["--extension", str(ext), "--qa-dir", str(qa),
              "--registry-path", str(state / "multicase_registry.yaml")]
    probes = {}
    candidate = _run("candidate_gate_audit.py", common)
    probes["candidate_gate"] = candidate
    receipt = scratch / "state_probe_receipt.json"
    receipt.write_text(json.dumps({"rehearsal_pass": True,
                                   "canonical_untouched": True,
                                   "probe": state_name}, indent=2) + "\n",
                         encoding="utf-8")
    architecture = _run("final_architecture_audit.py", common + [
        "--hot-swap-receipt", str(receipt), "--skip-pack"])
    probes["architecture"] = architecture
    document = _run("audit_state_language.py", common)
    probes["documents"] = document
    status_path = qa / "case_C_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    c_rows = _candidate_c_rows(ext)
    return {"state": state_name, "status": status,
            "candidate_C_rows": c_rows, "probes": probes,
            "candidate_gate_pass": candidate["returncode"] == 0,
            "architecture_pass": architecture["returncode"] == 0,
            "document_state_pass": document["returncode"] == 0}


def _pending_probe() -> dict:
    canonical_ext, canonical_qa = C.EXT, C.QA
    scratch = Path(tempfile.mkdtemp(prefix="case_pending_probe_", dir=str(canonical_qa)))
    data = scratch / "data"
    ext = scratch / "extension_multicase_20260826"
    qa = scratch / "qa"
    state = scratch / "state"
    try:
        data.mkdir(parents=True)
        qa.mkdir(parents=True)
        state.mkdir(parents=True)
        for case in C.CASE_ORDER:
            _link_tree(C.data_root() / C.CASES[case]["raw_subdir"],
                       data / C.CASES[case]["raw_subdir"])
        cdir = data / C.CASES["C"]["raw_subdir"]
        for name in ("placeholder_manifest.json", "PLACEHOLDER_NOTICE.md"):
            marker = cdir / name
            if marker.is_file():
                marker.unlink()
        hgt = cdir / C.raw_filename("C", "HGT_d05.pt")
        (cdir / "WRF_RUN_PROVENANCE.json").write_text(json.dumps({
            "case_id": "C", "site": C.CASES["C"]["location"],
            "event": C.CASES["C"]["dates"], "real_or_placeholder": "REAL",
            "rehearsal": True, "terrain_HGT_sha256": C.sha256_of(hgt),
        }, indent=2) + "\n", encoding="utf-8")
        for baseline_name in (
                "pre_repair_numeric_baseline_20260829.json",
                "future_c_promotion_regression_baseline_20260829.json"):
            source = canonical_qa / baseline_name
            if source.is_file():
                shutil.copy2(source, qa / baseline_name)
        command = [sys.executable, str(Path(__file__).with_name("run_all.py")),
                   "--cases", "A,B,C", "--phases", "all",
                   "--data-root", str(data), "--output-dir", str(ext),
                   "--qa-dir", str(qa),
                   "--registry-path", str(state / "multicase_registry.yaml")]
        pipeline = subprocess.run(command, cwd=str(C.ROOT), check=False,
                                  capture_output=True, text=True)
        common = ["--extension", str(ext), "--qa-dir", str(qa),
                  "--registry-path", str(state / "multicase_registry.yaml")]
        candidate = _run("candidate_gate_audit.py", common)
        receipt = scratch / "pending_probe_receipt.json"
        receipt.write_text(json.dumps({"rehearsal_pass": True,
                                       "canonical_untouched": True,
                                       "probe": "pending-state architecture audit"},
                                      indent=2) + "\n", encoding="utf-8")
        architecture = _run("final_architecture_audit.py", common + [
            "--hot-swap-receipt", str(receipt), "--skip-pack"])
        document = _run("audit_state_language.py", common)
        status_path = qa / "case_C_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        c_rows = _candidate_c_rows(ext)
        return {"state": "REAL_PENDING_AUDIT", "pipeline_returncode": int(pipeline.returncode),
                "pipeline_stdout_tail": pipeline.stdout[-2500:],
                "pipeline_stderr_tail": pipeline.stderr[-2500:],
                "status": status, "candidate_C_rows": c_rows,
                "candidate_gate_pass": candidate["returncode"] == 0,
                "architecture_pass": architecture["returncode"] == 0,
                "document_state_pass": document["returncode"] == 0,
                "probes": {"candidate_gate": candidate, "architecture": architecture,
                           "documents": document}}
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _placeholder_probe() -> dict:
    scratch = Path(tempfile.mkdtemp(prefix="case_placeholder_probe_", dir=str(C.QA)))
    try:
        return _state_probes(scratch, "PLACEHOLDER_READY")
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _real_probe() -> dict:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("rehearse_dayekou_swap.py"))],
        cwd=str(C.ROOT), check=False, capture_output=True, text=True)
    receipt_path = C.QA / "dayekou_hot_swap_rehearsal_20260829.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    checks = receipt.get("checks", {})
    return {"state": "REAL", "rehearsal_returncode": int(completed.returncode),
            "rehearsal_stdout_tail": completed.stdout[-2500:],
            "rehearsal_stderr_tail": completed.stderr[-2500:],
            "status": checks.get("scratch_status", {}),
            "candidate_C_rows": {"comment_candidate": int(
                checks.get("candidate_c_rows_after_promotion", 0)),
                "article_candidate": int(
                    checks.get("candidate_c_rows_after_promotion", 0))},
            "candidate_gate_pass": bool(checks.get("scratch_candidate_gate_pass")),
            "architecture_pass": bool(checks.get("scratch_final_architecture_pass")),
            "document_state_pass": bool(checks.get("scratch_document_state_pass")),
            "rehearsal_pass": bool(receipt.get("rehearsal_pass")),
            "receipt": receipt}


def _assertions(results: dict[str, dict]) -> dict:
    placeholder = results["placeholder"]
    pending = results["pending"]
    real = results["real"]
    return {
        "placeholder_state_exact": bool(
            placeholder["status"].get("case_status") == C.CASE_STATUS_PLACEHOLDER and
            placeholder["status"].get("science_eligible") is False and
            placeholder["status"].get("contains_placeholder") is True),
        "placeholder_candidate_C_rows_zero": all(
            value == 0 for value in placeholder["candidate_C_rows"].values()),
        "placeholder_gates_pass": bool(placeholder["candidate_gate_pass"] and
                                        placeholder["architecture_pass"] and
                                        placeholder["document_state_pass"]),
        "pending_state_exact": bool(
            pending["status"].get("case_status") == C.CASE_STATUS_REAL_PENDING_AUDIT and
            pending["status"].get("science_eligible") is False and
            pending["status"].get("contains_placeholder") is False and
            pending["status"].get("dependency_status") == "BLOCKED_REAL_PENDING_AUDIT"),
        "pending_candidate_C_rows_zero": all(
            value == 0 for value in pending["candidate_C_rows"].values()),
        "pending_gates_pass": bool(pending["pipeline_returncode"] == 0 and
                                    pending["candidate_gate_pass"] and
                                    pending["architecture_pass"] and
                                    pending["document_state_pass"]),
        "real_state_exact": bool(
            real["status"].get("case_status") == C.CASE_STATUS_REAL and
            real["status"].get("science_eligible") is True and
            real["status"].get("contains_placeholder") is False and
            real["status"].get("dependency_status") == "REAL"),
        "real_candidate_C_rows_present": bool(
            real["candidate_C_rows"]["comment_candidate"] > 0 and
            real["candidate_C_rows"]["article_candidate"] > 0),
        "real_gates_pass": bool(real["rehearsal_pass"] and
                                 real["candidate_gate_pass"] and
                                 real["architecture_pass"] and
                                 real["document_state_pass"]),
        "real_pipeline_hash_matches_current": bool(
            real.get("receipt", {}).get("pipeline_hash_after") == C._pipeline_hash()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    # Record canonical routing before probes; all subprocesses receive explicit
    # alternate roots and the real rehearsal restores its routing before return.
    C.set_output_dirs(None, None, None)
    C.set_data_root(None)
    results = {"placeholder": _placeholder_probe(),
               "pending": _pending_probe(),
               "real": _real_probe()}
    assertions = _assertions(results)
    passed = all(assertions.values())
    result = {"pass": passed, "generated_at_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "assertions": assertions,
              "states": results,
              "canonical_after_probe": {
                  "case_C_status": (json.loads((C.QA / "case_C_status.json").read_text(
                      encoding="utf-8")) if (C.QA / "case_C_status.json").is_file() else {}),
                  "pipeline_hash": C._pipeline_hash(),
              }}
    out = Path(args.output).expanduser().resolve() if args.output else (
        C.QA / "case_state_machine_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                              default=C._json_default) + "\n", encoding="utf-8")
    lines = ["# CASE_STATE_MACHINE_TEST — Paper0 multicase", "",
             "Three isolated probes exercise the live Case-C state language and "
             "candidate boundary.  The REAL branch uses synthetic placeholder "
             "bytes solely as a lifecycle rehearsal.", "",
             f"Overall: **{'PASS' if passed else 'FAIL'}**", "",
             "| state | exact state assertion | candidate C rows | candidate | architecture | documents |",
             "|---|---|---:|---|---|---|"]
    for name, state_name in (("placeholder", "PLACEHOLDER_READY"),
                             ("pending", "REAL_PENDING_AUDIT"),
                             ("real", "REAL")):
        item = results[name]
        exact_key = {"placeholder": "placeholder_state_exact",
                     "pending": "pending_state_exact",
                     "real": "real_state_exact"}[name]
        rows = ", ".join(f"{k}={v}" for k, v in item["candidate_C_rows"].items())
        lines.append(f"| {state_name} | {'PASS' if assertions[exact_key] else 'FAIL'} | {rows} | "
                     f"{'PASS' if item['candidate_gate_pass'] else 'FAIL'} | "
                     f"{'PASS' if item['architecture_pass'] else 'FAIL'} | "
                     f"{'PASS' if item['document_state_pass'] else 'FAIL'} |")
    lines += ["", "Assertions:"]
    lines.extend(f"* `{key}` = `{str(value).lower()}`"
                 for key, value in assertions.items())
    lines += ["", "Canonical post-probe state must remain `C = PLACEHOLDER_READY`; "
              "the fake-real tree is temporary and is never evidence."]
    report = out.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": passed, "assertions": assertions,
                      "output": str(out)}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
