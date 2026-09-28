"""Stable one-command entry point for the Paper0 multicase pipeline.

Normal rerun::

    python run_multicase.py

Dayekou hot swap::

    python run_multicase.py --replace-case C \
        --input <data-root>/1min_dayekou20151201 --mode real

The replacement path validates the source before touching the registered raw
directory, installs it transactionally, marks Case C REAL_PENDING_AUDIT, and
delegates to ``run_all.py``.  Only the downstream audit gate may promote the
case to REAL.  A/B replacement is intentionally refused: those archives are
outside this hot-swap contract and are protected by the Paper0 freeze.
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
import validate_case_input as V  # noqa: E402


OPTIONAL_INPUT_SIDECARS = (
    *V.PROVENANCE_SIDECARS,
    "timestamps.json", "time_axis.json", "timestamps.npy", "time_axis.npy",
    "static_fields.json", "static_field_manifest.json", "terrain_metadata.json",
)


def _copy_real_payload(case: str, source: Path, validation: dict) -> dict:
    """Install a validated payload and leave an explicit recovery backup."""
    destination = C.case_raw_dir(case)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_name = f".{destination.name}.incoming_{os.getpid()}"
    stage = Path(tempfile.mkdtemp(prefix=stage_name + "_",
                                  dir=str(destination.parent)))
    backup = None
    try:
        files = V.resolve_payload_files(case, source)
        for stem, src in files.items():
            shutil.copy2(src, stage / C.raw_filename(case, stem))
        for name in OPTIONAL_INPUT_SIDECARS:
            src = source / name
            if src.is_file():
                shutil.copy2(src, stage / name)
        pending = {
            "case": case,
            "status": C.CASE_STATUS_REAL_PENDING_AUDIT,
            "installed_from": str(source),
            "validated_receipt": validation,
            "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
            "note": "engineering marker; only the passing audit may promote REAL",
        }
        (stage / "real_pending_audit.json").write_text(
            json.dumps(pending, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

        if destination.exists():
            backup_root = C.QA / "replacement_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            backup = backup_root / f"{destination.name}_{stamp}_{os.getpid()}"
            shutil.move(str(destination), str(backup))
        try:
            shutil.move(str(stage), str(destination))
        except Exception:
            if backup is not None and not destination.exists():
                shutil.move(str(backup), str(destination))
            raise
        return {"destination": str(destination),
                "backup": None if backup is None else str(backup),
                "validation": validation}
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _run_standard(args: argparse.Namespace, data_root: Path) -> int:
    command = [sys.executable, str(Path(__file__).with_name("run_all.py")),
               "--cases", args.cases, "--phases", args.phases,
               "--data-root", str(data_root)]
    for flag, value in (("--output-dir", args.output_dir),
                        ("--qa-dir", args.qa_dir),
                        ("--registry-path", args.registry_path)):
        if value:
            command.extend([flag, value])
    if args.no_independent_audit:
        command.append("--no-independent-audit")
    print("[run_multicase] standard pipeline:", " ".join(command))
    completed = subprocess.run(command, cwd=str(C.ROOT), check=False)
    return int(completed.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="A,B,C")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--qa-dir", default=None)
    ap.add_argument("--registry-path", default=None)
    ap.add_argument("--no-independent-audit", action="store_true")
    ap.add_argument("--replace-case", choices=("C", "c"), default=None)
    ap.add_argument("--input", default=None,
                    help="validated seven-tensor payload directory")
    ap.add_argument("--mode", choices=("real",), default=None)
    args = ap.parse_args()

    if args.replace_case is None and any((args.input, args.mode)):
        ap.error("--input and --mode require --replace-case C")
    if args.replace_case is not None and (not args.input or args.mode != "real"):
        ap.error("--replace-case C requires --input PATH --mode real")
    if args.replace_case is not None and "C" not in args.cases.upper().split(","):
        ap.error("--replace-case C requires C in --cases")

    C.set_output_dirs(args.output_dir, args.qa_dir, args.registry_path)
    C.set_data_root(args.data_root)
    data_root = C.data_root()

    replacement = None
    if args.replace_case is not None:
        source = Path(args.input).expanduser().resolve()
        print(f"[run_multicase] validating real Case C input: {source}")
        try:
            validation = V.validate_payload("C", source, mode="real")
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"[run_multicase] VALIDATION FAILED: {exc}", file=sys.stderr)
            return 2
        if validation["stop"]:
            print(json.dumps(validation, indent=2, ensure_ascii=False),
                  file=sys.stderr)
            return 2
        replacement = _copy_real_payload("C", source, validation)
        C.run_receipt("replace_case_C", {
            "case": "C", "mode": "real", "replacement": replacement,
            "status": C.CASE_STATUS_REAL_PENDING_AUDIT,
        })
        print("[run_multicase] Case C installed as REAL_PENDING_AUDIT")

    rc = _run_standard(args, data_root)
    if rc != 0:
        if replacement:
            print("[run_multicase] pipeline failed; Case C remains pending; "
                  f"backup retained at {replacement['backup']}", file=sys.stderr)
        return rc
    if replacement:
        status = C.load_case_status("C")
        print("[run_multicase] Case C final status:",
              status.get("case_status"), status.get("dependency_status"))
        if status.get("case_status") != C.CASE_STATUS_REAL:
            print("[run_multicase] audit did not promote Case C; backup retained "
                  f"at {replacement['backup']}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
