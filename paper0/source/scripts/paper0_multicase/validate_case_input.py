"""Validate one multicase payload before it enters the standard pipeline.

The validator is intentionally independent of the downstream science phases.
It checks the stable Case-A-shaped input interface (seven tensors, 4320
states, 162 x 162 fine grid, exact 6 x 6 -> 27 x 27 coarse tiling), finite
values, the declared archived-order chronology contract, and placeholder
branding.  It never changes a case status and never promotes a payload.

Examples
--------
    python validate_case_input.py --case C --data-root <root>
    python validate_case_input.py --case C --input <payload-dir> --mode real
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402

import case_c_freeze as FZ  # noqa: E402


PROVENANCE_SIDECARS = ("WRF_RUN_PROVENANCE.json", "case_input_metadata.json")
REQUIRED_REAL_PROVENANCE = (
    "namelist_input_sha256", "namelist_wps_sha256", "geo_em_hashes",
    "terrain_HGT_sha256", "landuse_sha256", "wrf_version",
    "physics_configuration", "preprocessing_version", "timestamp_cadence",
)
UNKNOWN_PROVENANCE = {"", "UNKNOWN", "UNKNOWN_NOT_PACKAGED", "NOT_PROVIDED"}


def _load_raw_tensor(path: Path) -> np.ndarray:
    """Load without coercing dtype (Phase 0's contract must see the dtype)."""
    import torch

    tensor = torch.load(str(path), map_location="cpu", weights_only=True)
    if not hasattr(tensor, "detach") or not hasattr(tensor, "numpy"):
        raise TypeError(f"{path.name}: expected a torch tensor payload")
    return tensor.detach().cpu().numpy()


def resolve_payload_files(case: str, input_dir: str | Path) -> dict[str, Path]:
    """Resolve logical stems to source files, accepting only stable names.

    Real archives may arrive with the registered per-case suffix (A/B) or the
    logical bare names.  The destination interface is always canonicalized by
    ``run_multicase.py``; accepting both here avoids a future filename-only
    code edit while rejecting silently mixed payloads.
    """
    case = case.upper()
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory missing: {root}")
    expected_disk = {C.raw_filename(case, stem): stem for stem in C.RAW_FILE_STEMS}
    expected_bare = {stem: stem for stem in C.RAW_FILE_STEMS}
    if all((root / name).is_file() for name in expected_disk):
        found = {stem: root / name for name, stem in expected_disk.items()}
    elif all((root / name).is_file() for name in expected_bare):
        found = {stem: root / name for name, stem in expected_bare.items()}
    else:
        missing = sorted(set(expected_disk) - {p.name for p in root.glob("*.pt")})
        raise ValueError(
            f"{root}: payload must contain the seven registered tensor files "
            f"for Case {case}; missing canonical names: {missing}")
    pt_names = {p.name for p in root.glob("*.pt")}
    allowed = set(expected_disk) | set(expected_bare)
    unexpected = sorted(pt_names - allowed)
    if unexpected:
        raise ValueError(f"{root}: unexpected .pt files: {unexpected}")
    return found


def _placeholder_match(case: str, files: dict[str, Path]) -> tuple[bool, bool]:
    """Return (manifest_present, all_hashes_match)."""
    parents = {p.parent for p in files.values()}
    if len(parents) != 1:
        return False, False
    manifest_path = next(iter(parents)) / "placeholder_manifest.json"
    if not manifest_path.is_file():
        return False, False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = payload.get("files", {})
    matches = True
    for stem, path in files.items():
        disk_name = path.name
        logical_names = (disk_name, stem)
        entry = next((listed.get(name) for name in logical_names
                      if name in listed), None)
        if not entry or entry.get("sha256") != C.sha256_of(path):
            matches = False
    return True, matches


def _chronology_check(root: Path, n_states: int) -> dict:
    """Check timestamps when supplied, otherwise record the archive contract.

    The .pt archives intentionally do not carry absolute timestamps.  The
    inherited chronological ordering is therefore a declared provenance
    assumption, not something that can be reconstructed from tensor bytes.
    """
    candidates = [root / "timestamps.json", root / "time_axis.json",
                  root / "timestamps.npy", root / "time_axis.npy"]
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(values, dict):
                values = values.get("timestamps", values.get("time_axis", []))
        else:
            values = np.load(path, allow_pickle=False).tolist()
        arr = np.asarray(values)
        if len(arr) != n_states:
            return {"pass": False, "status": "TIMESTAMP_LENGTH_MISMATCH",
                    "path": path.name, "n": int(len(arr)),
                    "expected": int(n_states)}
        ordered = bool(np.all(arr[1:] >= arr[:-1])) if len(arr) > 1 else True
        return {"pass": ordered,
                "status": "TIMESTAMP_ARRAY_CHECKED" if ordered else "TIMESTAMP_NOT_MONOTONIC",
                "path": path.name, "n": int(len(arr))}
    return {"pass": True, "status": "ARCHIVED_SERIAL_ORDER_CONTRACT",
            "note": "absolute timestamps are not stored in the tensor archive"}


def _complete_provenance_value(value) -> bool:
    """Return False for explicit unknowns and empty nested metadata."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().upper() not in UNKNOWN_PROVENANCE
    if isinstance(value, dict):
        return bool(value) and all(_complete_provenance_value(v)
                                  for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_complete_provenance_value(v) for v in value)
    return True


def _provenance_check(case: str, root: Path, files: dict[str, Path],
                      mode: str) -> dict:
    """Check identity and static-field provenance without inventing WRF facts."""
    sidecar_path = next((root / name for name in PROVENANCE_SIDECARS
                         if (root / name).is_file()), None)
    result = {
        "pass": True,
        "status": "NOT_REQUIRED_FOR_PLACEHOLDER" if mode == "placeholder"
                  else "MISSING_SIDECAR" if sidecar_path is None
                  else "CHECKED",
        "source": None if sidecar_path is None else sidecar_path.name,
        "errors": [],
    }
    if sidecar_path is None:
        if mode == "real":
            result["pass"] = False
            result["errors"].append(
                "real payload requires WRF_RUN_PROVENANCE.json or "
                "case_input_metadata.json")
        return result
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["pass"] = False
        result["errors"].append(f"provenance sidecar unreadable: {exc}")
        return result
    if not isinstance(payload, dict):
        result["pass"] = False
        result["errors"].append("provenance sidecar must contain a JSON object")
        return result
    result["declared_case_id"] = payload.get("case_id")
    result["declared_site"] = payload.get("site")
    result["declared_event"] = payload.get("event")
    if str(payload.get("case_id", "")).upper() != case:
        result["errors"].append(
            f"case_id {payload.get('case_id')!r} != registered {case}")
    expected_site = C.CASES[case]["location"].split("/")[0].strip().lower()
    if expected_site not in str(payload.get("site", "")).lower():
        result["errors"].append(
            f"site {payload.get('site')!r} does not identify {expected_site}")
    if case == "C":
        # A declared date is not verifiable from the payload: the .pt files carry no
        # absolute time, so the bytes are the only attestation of the frozen window
        # (PLAN_AMENDMENT_CASE_C_20260830.md section 3.1).
        result["event_selection_status"] = FZ.selection_status()
        freeze_errors = FZ.verify_payload(files)
        if mode == "real" or FZ.load() is not None:
            result["errors"].extend(freeze_errors)
    else:
        expected_date = str(C.CASES[case].get("dates", "")).split(" to ")[0].strip()
        if expected_date and expected_date not in str(payload.get("event", "")):
            result["errors"].append(
                "event metadata does not identify the registered "
                f"{expected_date} episode")
    hgt_path = files["HGT_d05.pt"]
    hgt_hash = C.sha256_of(hgt_path)
    declared_hgt = payload.get("terrain_HGT_sha256", payload.get("hgt_sha256"))
    if declared_hgt != hgt_hash:
        result["errors"].append(
            f"terrain/HGT hash {declared_hgt!r} != observed {hgt_hash}")
    if mode == "real":
        if str(payload.get("real_or_placeholder", "")).upper() == "PLACEHOLDER":
            result["errors"].append("provenance sidecar labels the payload PLACEHOLDER")
        for key in REQUIRED_REAL_PROVENANCE:
            if not _complete_provenance_value(payload.get(key)):
                result["errors"].append(
                    f"missing or unknown required real provenance field: {key}")
    result["pass"] = not result["errors"]
    result["status"] = "PASS" if result["pass"] else "FAIL"
    return result


def validate_payload(case: str, input_dir: str | Path | None = None,
                     mode: str = "auto") -> dict:
    """Return a complete validation receipt; raise on a hard input failure."""
    case = case.upper()
    if case not in C.CASE_ORDER:
        raise ValueError(f"unknown case id {case}; stable ids are A, B, C")
    root = (Path(input_dir).expanduser().resolve() if input_dir is not None
            else C.case_raw_dir(case))
    files = resolve_payload_files(case, root)
    result = {
        "case_id": case,
        "input_dir": str(root),
        "mode_requested": mode,
        "checks": {},
        "files": {},
        "stop": False,
        "stop_reasons": [],
    }

    def fail(reason: str) -> None:
        result["stop"] = True
        result["stop_reasons"].append(reason)

    arrays: dict[str, np.ndarray] = {}
    for stem, path in files.items():
        try:
            arr = _load_raw_tensor(path)
        except Exception as exc:  # noqa: BLE001 - turn loader failures into a receipt
            fail(f"{stem}: tensor load failed: {exc}")
            continue
        arrays[stem] = arr
        result["files"][stem] = {
            "source_name": path.name,
            "sha256": C.sha256_of(path),
            "bytes": path.stat().st_size,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite": bool(np.isfinite(arr).all()),
        }
        if arr.dtype != np.float32:
            fail(f"{stem}: dtype {arr.dtype} != float32")
        if not result["files"][stem]["finite"]:
            fail(f"{stem}: NaN/Inf present")

    wind_stems = ["U10_d02.pt", "V10_d02.pt", "U10_d05.pt", "V10_d05.pt"]
    for stem in wind_stems + ["T2_d02.pt", "T2_d05.pt"]:
        arr = arrays.get(stem)
        if arr is None:
            continue
        if arr.shape != (C.RAW_N, C.GRID, C.GRID):
            fail(f"{stem}: shape {arr.shape} != ({C.RAW_N},{C.GRID},{C.GRID})")
    hgt = arrays.get("HGT_d05.pt")
    if hgt is not None and hgt.shape != (C.GRID, C.GRID):
        fail(f"HGT_d05.pt: shape {hgt.shape} != ({C.GRID},{C.GRID})")
    result["checks"]["state_count"] = all(
        arrays.get(stem, np.empty((0,))).shape[:1] == (C.RAW_N,)
        for stem in wind_stems + ["T2_d02.pt", "T2_d05.pt"])
    if not result["checks"]["state_count"]:
        fail("state count is not 4320; inherited 2680/670 split is not applied")

    for stem in ("U10_d02.pt", "V10_d02.pt"):
        arr = arrays.get(stem)
        if arr is None or arr.shape != (C.RAW_N, C.GRID, C.GRID):
            continue
        native = arr[:, ::C.BLOCK, ::C.BLOCK]
        tiled = np.repeat(np.repeat(native, C.BLOCK, axis=1), C.BLOCK, axis=2)
        dev = float(np.abs(arr - tiled).max())
        result["files"][stem]["d02_tiling_max_abs_dev"] = dev
        if dev != 0.0:
            fail(f"{stem}: d02 6x6 -> 27x27 tiling deviation {dev}")
    result["checks"]["parent_mapping"] = {
        "pass": True,
        "native_parent_grid": "6x6 row-major",
        "block": C.BLOCK,
        "fine_cells_per_parent": C.N_FINE,
        "parent_centre_fine_indices": C.PARENT_CENTRE_FINE_INDICES,
    }
    result["checks"]["chronology"] = _chronology_check(root, C.RAW_N)
    if not result["checks"]["chronology"]["pass"]:
        fail(result["checks"]["chronology"]["status"])
    manifest_present, manifest_match = _placeholder_match(case, files)
    result["placeholder_manifest_present"] = manifest_present
    result["placeholder_manifest_match"] = manifest_match
    placeholder = bool(manifest_present and manifest_match)
    if mode == "placeholder" and not placeholder:
        fail("mode=placeholder requested, but payload is not a matching placeholder")
    if mode == "real" and placeholder:
        fail("mode=real requested, but payload still matches the placeholder manifest")
    provenance = _provenance_check(case, root, files, mode)
    result["checks"]["provenance"] = provenance
    if mode == "real" and not provenance["pass"]:
        for error in provenance["errors"]:
            fail(error)
    if placeholder:
        status = C.CASE_STATUS_PLACEHOLDER
        data_status = "PLACEHOLDER"
    else:
        status = C.CASE_STATUS_REAL_PENDING_AUDIT if mode == "real" else C.CASE_STATUS_REAL
        data_status = "REAL"
    if result["stop"]:
        status = C.CASE_STATUS_FAILED
    result.update({
        "data_status": data_status,
        "case_status": status,
        "science_eligible": bool(status == C.CASE_STATUS_REAL and not result["stop"]),
        "contains_placeholder": placeholder,
        "dependency_status": ("FAILED" if result["stop"] else
                               "BLOCKED_PLACEHOLDER_DEPENDENCY" if placeholder else
                               "BLOCKED_REAL_PENDING_AUDIT" if status == C.CASE_STATUS_REAL_PENDING_AUDIT else
                               "REAL"),
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=list(C.CASE_ORDER))
    ap.add_argument("--input", default=None,
                    help="payload directory; defaults to the registered raw directory")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--mode", choices=("auto", "real", "placeholder"), default="auto")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    C.set_data_root(args.data_root)
    try:
        receipt = validate_payload(args.case, args.input, args.mode)
    except Exception as exc:  # noqa: BLE001 - CLI should return a useful hard failure
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 1 if receipt["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
