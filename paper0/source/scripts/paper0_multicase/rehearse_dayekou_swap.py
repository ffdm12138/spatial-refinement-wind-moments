"""Run an isolated fake-real Dayekou replacement rehearsal.

The live Case-C directory is never used as the rehearsal worktree.  A/B/C
payloads are hard-linked into a temporary data root, the placeholder marker is
removed only in that temporary copy, a complete rehearsal provenance sidecar
is added, and ``run_multicase.py --replace-case C --mode real`` is exercised.
The temporary tree is removed before the canonical receipt is written.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import case_c_freeze as FZ  # noqa: E402


def _link_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def _hash_snapshot(root: Path, case: str) -> dict[str, str]:
    case_root = root / C.CASES[case]["raw_subdir"]
    return {
        name: C.sha256_of(case_root / name)
        for stem in C.RAW_FILE_STEMS
        for name in [C.raw_filename(case, stem)]
    }


def _run_probe(script_name: str, args: list[str]) -> dict:
    """Run one isolated audit probe and retain compact diagnostics."""
    command = [sys.executable,
               str(Path(__file__).with_name(script_name)), *args]
    completed = subprocess.run(command, cwd=str(C.ROOT), check=False,
                               capture_output=True, text=True)
    return {
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> int:
    # Resolve and retain canonical locations before routing mc_config to the
    # isolated rehearsal.  The pointer is read-only and is never rewritten.
    C.set_output_dirs(None, None, None)
    C.set_data_root(None)
    canonical_ext, canonical_qa, canonical_registry = C.EXT, C.QA, C.REGISTRY_PATH
    canonical_data_root = C.data_root()
    live_c = canonical_data_root / C.CASES["C"]["raw_subdir"]
    before_hashes = _hash_snapshot(canonical_data_root, "C")
    before_manifest = C.sha256_of(live_c / "placeholder_manifest.json")
    before_notice = C.sha256_of(live_c / "PLACEHOLDER_NOTICE.md")
    pipeline_hash_before = C._pipeline_hash()

    scratch = Path(tempfile.mkdtemp(prefix="dayekou_hot_swap_rehearsal_",
                                    dir=str(canonical_qa)))
    scratch_data = scratch / "data"
    scratch_ext = scratch / "extension_multicase_20260826"
    scratch_qa = scratch / "qa"
    scratch_state = scratch / "state"
    incoming = scratch / "incoming_dayekou20151201"
    receipt = {
        "rehearsal_kind": "fake-real replacement; no scientific interpretation",
        "canonical_data_root": str(canonical_data_root),
        "canonical_case": "C",
        "canonical_case_status_before": "PLACEHOLDER",
        "scratch_root": str(scratch),
        "replacement_pipeline_exit": None,
        "scratch_c_promoted": False,
        "canonical_placeholder_restored": False,
        "pipeline_hash_before": pipeline_hash_before,
        "pipeline_hash_after": None,
        "checks": {},
        "rehearsal_pass": False,
    }
    rc = 2
    try:
        scratch_data.mkdir(parents=True, exist_ok=True)
        scratch_qa.mkdir(parents=True, exist_ok=True)
        scratch_state.mkdir(parents=True, exist_ok=True)
        for case in C.CASE_ORDER:
            _link_tree(canonical_data_root / C.CASES[case]["raw_subdir"],
                       scratch_data / C.CASES[case]["raw_subdir"])
        # The regression snapshots are small and make the isolated scratch run
        # exercise the same metric/scientific gates as a real future arrival.
        for baseline_name in (
                "pre_repair_numeric_baseline_20260829.json",
                "future_c_promotion_regression_baseline_20260829.json"):
            source = canonical_qa / baseline_name
            if source.is_file():
                shutil.copy2(source, scratch_qa / baseline_name)
        _link_tree(scratch_data / C.CASES["C"]["raw_subdir"], incoming)
        (incoming / "placeholder_manifest.json").unlink()
        (incoming / "PLACEHOLDER_NOTICE.md").unlink()
        fake_prov = {
            "case_id": "C",
            "site": C.CASES["C"]["location"],
            "event": C.CASES["C"]["dates"],
            "real_or_placeholder": "REAL",
            "rehearsal": True,
            "namelist_input_sha256": "rehearsal-namelist-input",
            "namelist_wps_sha256": "rehearsal-namelist-wps",
            "geo_em_hashes": {"geo_em.d05.nc": "rehearsal-geo-em"},
            "terrain_HGT_sha256": C.sha256_of(incoming / "HGT_d05.pt"),
            "landuse_sha256": "rehearsal-landuse",
            "wrf_version": "rehearsal-WRF-version",
            "physics_configuration": "rehearsal-physics-configuration",
            "preprocessing_version": "rehearsal-preprocessing",
            "timestamp_cadence": "1-minute archived serialized order; rehearsal only",
            "provenance_note": "Synthetic placeholder bytes used solely to exercise the real-input contract.",
        }
        (incoming / "WRF_RUN_PROVENANCE.json").write_text(
            json.dumps(fake_prov, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

        # A REAL install is admissible only against a dated event freeze register of
        # the payload bytes (PLAN_AMENDMENT_CASE_C_20260830.md section 3.1), so the
        # rehearsal freezes first and then swaps, exactly as an author would.
        scratch_ext.mkdir(parents=True, exist_ok=True)
        FZ.capture(incoming, "2015-12-01 to 2015-12-03 (rehearsal window)",
                   "dayekou_hot_swap_rehearsal",
                   notes="Synthetic placeholder bytes; lifecycle rehearsal only, "
                         "never science.",
                   target=scratch_ext / FZ.REGISTER_NAME)

        command = [sys.executable,
                   str(Path(__file__).with_name("run_multicase.py")),
                   "--cases", "A,B,C", "--phases", "all",
                   "--data-root", str(scratch_data),
                   "--output-dir", str(scratch_ext),
                   "--qa-dir", str(scratch_qa),
                   "--registry-path", str(scratch_state / "multicase_registry.yaml"),
                   "--replace-case", "C", "--input", str(incoming), "--mode", "real"]
        print("[rehearsal] invoking isolated hot-swap pipeline")
        completed = subprocess.run(command, cwd=str(C.ROOT), check=False)
        rc = int(completed.returncode)
        receipt["replacement_pipeline_exit"] = rc

        scratch_status_path = scratch_qa / "case_C_status.json"
        scratch_status = (json.loads(scratch_status_path.read_text(encoding="utf-8"))
                          if scratch_status_path.is_file() else {})
        m4_path = scratch_ext / "tables" / "M4_scale_dependence.csv"
        m4 = pd.read_csv(m4_path) if m4_path.is_file() else pd.DataFrame()
        candidate_c_rows = 0
        for path in (scratch_ext / "comment_candidate").glob("*.csv"):
            df = pd.read_csv(path)
            if "case" in df.columns:
                candidate_c_rows += int((df["case"].astype(str) == "C").sum())
        receipt["scratch_c_promoted"] = scratch_status.get("case_status") == C.CASE_STATUS_REAL
        receipt["checks"].update({
            "scratch_status": scratch_status,
            "m4_rows": int(len(m4)),
            "m4_expected_rows": 24,
            "m4_schema_pass": bool(len(m4) == 24),
            "candidate_c_rows_after_promotion": candidate_c_rows,
            "candidate_contains_real_c_rows": candidate_c_rows > 0,
            "scratch_registry_exists": (scratch_state / "multicase_registry.yaml").is_file(),
        })

        # Exercise the same state-aware candidate, document and independent
        # architecture gates while the real scratch tree still exists.
        common = ["--extension", str(scratch_ext), "--qa-dir", str(scratch_qa),
                  "--registry-path", str(scratch_state / "multicase_registry.yaml")]
        candidate_probe = _run_probe("candidate_gate_audit.py", common)
        architecture_receipt = scratch / "scratch_hot_swap_receipt.json"
        architecture_receipt.write_text(json.dumps({
            "rehearsal_pass": bool(receipt["replacement_pipeline_exit"] == 0 and
                                   receipt["scratch_c_promoted"] and
                                   receipt["checks"].get("candidate_contains_real_c_rows")),
            "canonical_untouched": True,
            "probe": "real-state architecture audit",
        }, indent=2) + "\n", encoding="utf-8")
        architecture_probe = _run_probe(
            "final_architecture_audit.py",
            common + ["--hot-swap-receipt", str(architecture_receipt),
                      "--skip-pack"])
        document_probe = _run_probe("audit_state_language.py", common)
        receipt["checks"].update({
            "candidate_gate_probe": candidate_probe,
            "scratch_candidate_gate_pass": candidate_probe["returncode"] == 0,
            "scratch_architecture_probe": architecture_probe,
            "scratch_final_architecture_pass": architecture_probe["returncode"] == 0,
            "document_state_probe": document_probe,
            "scratch_document_state_pass": document_probe["returncode"] == 0,
        })
    except Exception as exc:  # noqa: BLE001 - rehearsal receipt must retain the cause
        receipt["checks"]["exception"] = repr(exc)
        print(f"[rehearsal] ERROR: {exc}", file=sys.stderr)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
        # Restore canonical routing before checking the live placeholder and
        # writing the durable receipt.
        C.set_output_dirs(str(canonical_ext), str(canonical_qa),
                          str(canonical_registry))
        C.set_data_root(str(canonical_data_root))
        after_hashes = _hash_snapshot(canonical_data_root, "C")
        after_manifest = C.sha256_of(live_c / "placeholder_manifest.json")
        after_notice = C.sha256_of(live_c / "PLACEHOLDER_NOTICE.md")
        canonical_status_path = canonical_qa / "case_C_status.json"
        canonical_status = (json.loads(canonical_status_path.read_text(encoding="utf-8"))
                            if canonical_status_path.is_file() else {})
        pipeline_hash_after = C._pipeline_hash()
        restored = (before_hashes == after_hashes and
                     before_manifest == after_manifest and
                     before_notice == after_notice and
                     canonical_status.get("case_status") == C.CASE_STATUS_PLACEHOLDER)
        receipt["canonical_placeholder_restored"] = restored
        receipt["pipeline_hash_after"] = pipeline_hash_after
        receipt["checks"].update({
            "scratch_removed": not scratch.exists(),
            "canonical_payload_hashes_unchanged": before_hashes == after_hashes,
            "canonical_placeholder_manifest_unchanged": before_manifest == after_manifest,
            "canonical_placeholder_notice_unchanged": before_notice == after_notice,
            "canonical_status_after": canonical_status,
            "pipeline_hash_unchanged": pipeline_hash_before == pipeline_hash_after,
        })
        receipt["rehearsal_pass"] = bool(
            receipt["replacement_pipeline_exit"] == 0 and
            receipt["scratch_c_promoted"] and
            receipt["checks"].get("m4_schema_pass") and
            receipt["checks"].get("candidate_contains_real_c_rows") and
            receipt["checks"].get("scratch_candidate_gate_pass") and
            receipt["checks"].get("scratch_final_architecture_pass") and
            receipt["checks"].get("scratch_document_state_pass") and
            restored and receipt["checks"].get("scratch_removed") and
            receipt["checks"].get("pipeline_hash_unchanged"))
        report_lines = [
            "# DAYEKOU_HOT_SWAP_REHEARSAL — isolated fake-real replacement", "",
            "This rehearsal uses synthetic placeholder bytes only to exercise the "
            "real-input lifecycle.  It is not scientific evidence and it does not "
            "authorize manuscript changes.", "",
            f"* Replacement pipeline exit: `{receipt['replacement_pipeline_exit']}`",
            f"* Scratch C promoted to REAL: `{receipt['scratch_c_promoted']}`",
            f"* Scratch candidate gate: `{receipt['checks'].get('scratch_candidate_gate_pass')}`",
            f"* Scratch architecture audit: `{receipt['checks'].get('scratch_final_architecture_pass')}`",
            f"* Scratch document-state audit: `{receipt['checks'].get('scratch_document_state_pass')}`",
            f"* Canonical placeholder restored/untouched: `{receipt['canonical_placeholder_restored']}`",
            f"* Overall rehearsal: **{'PASS' if receipt['rehearsal_pass'] else 'FAIL'}**",
            "", "The canonical C payload, placeholder manifest/notice and status "
            "remain byte/state consistent after the scratch tree is removed.",
        ]
        (canonical_qa / "DAYEKOU_HOT_SWAP_REHEARSAL.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8")
        receipt["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        C.write_json(canonical_qa / "dayekou_hot_swap_rehearsal_20260829.json",
                     receipt, cases=["C"])
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["rehearsal_pass"] else (rc or 2)


if __name__ == "__main__":
    raise SystemExit(main())
