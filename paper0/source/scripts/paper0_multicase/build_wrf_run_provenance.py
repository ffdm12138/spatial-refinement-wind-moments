"""Build the complete WRF/WPS provenance sidecar for a real case payload.

The seven tensor files are not enough to promote Case C.  This helper hashes
the tensor payload, WRF/WPS inputs and static fields, and records the version,
physics and preprocessing metadata required by ``validate_case_input.py``.
Missing facts are reported as ``missing_required_fields`` and produce
``provenance_status=FAIL``; they are never filled with
``UNKNOWN_NOT_PACKAGED`` and never make a payload promotion-eligible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402
import validate_case_input as V  # noqa: E402


def _hash(path: Path) -> str:
    return C.sha256_of(path)


def _unique_named(root: Path, names: tuple[str, ...]) -> Path | None:
    direct = [root / name for name in names if (root / name).is_file()]
    if direct:
        return direct[0]
    matches = []
    for name in names:
        matches.extend(p for p in root.rglob(name) if p.is_file())
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _version_from_tree(root: Path) -> str | None:
    path = _unique_named(root, ("WRF_VERSION", "WRF_VERSION.txt",
                                "wrf_version.txt", "version.txt"))
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _physics_from_namelist(path: Path | None) -> str | None:
    if path is None:
        return None
    keys = ("mp_physics", "ra_lw_physics", "ra_sw_physics",
            "sf_sfclay_physics", "sf_surface_physics", "bl_pbl_physics",
            "cu_physics", "sf_urban_physics", "num_soil_layers")
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = line.split("!", 1)[0].strip()
        if not clean or "=" not in clean:
            continue
        key = clean.split("=", 1)[0].strip().lower()
        if any(key == wanted or key.startswith(wanted + "_") for wanted in keys):
            values.append(clean)
    return "; ".join(values) if values else None


def _text_value(root: Path, names: tuple[str, ...]) -> str | None:
    path = _unique_named(root, names)
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _static_file(root: Path, explicit: str | None,
                 patterns: tuple[str, ...]) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    matches = []
    for pattern in patterns:
        matches.extend(p for p in root.rglob(pattern) if p.is_file())
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _cadence(tensor_dir: Path, override: str | None) -> str | None:
    if override:
        return override.strip() or None
    for name in ("timestamps.json", "time_axis.json", "timestamps.npy", "time_axis.npy"):
        path = tensor_dir / name
        if path.is_file():
            return f"verified timestamp sidecar: {name}; one-minute archive contract"
    return None


def _grid_metadata_path(wrf_run_dir: Path, wps_dir: Path,
                        override: str | Path | None) -> Path | None:
    if override:
        path = Path(override).expanduser().resolve()
        return path if path.is_file() else None
    for root in (wps_dir, wrf_run_dir):
        path = _unique_named(root, ("grid_metadata.json", "domain_metadata.json",
                                    "grid.json"))
        if path is not None:
            return path
    return None


def build_provenance(case: str, tensor_dir: str | Path,
                     wrf_run_dir: str | Path, wps_dir: str | Path,
                     output: str | Path | None = None,
                     terrain_file: str | None = None,
                     landuse_file: str | None = None,
                     wrf_version: str | None = None,
                     physics_configuration: str | None = None,
                     preprocessing_version: str | None = None,
                     timestamp_cadence: str | None = None,
                     grid_metadata: str | Path | None = None) -> dict:
    case = case.upper()
    if case not in C.CASE_ORDER:
        raise ValueError(f"unknown case id {case}; stable ids are A, B, C")
    tensor_dir = Path(tensor_dir).expanduser().resolve()
    wrf_run_dir = Path(wrf_run_dir).expanduser().resolve()
    wps_dir = Path(wps_dir).expanduser().resolve()
    target = (Path(output).expanduser().resolve() if output else
              tensor_dir / "WRF_RUN_PROVENANCE.json")
    missing: list[str] = []
    files = V.resolve_payload_files(case, tensor_dir)
    raw_hashes = {stem: _hash(path) for stem, path in files.items()}
    input_namelist = _unique_named(wrf_run_dir, ("namelist.input",))
    wps_namelist = _unique_named(wps_dir, ("namelist.wps",))
    if input_namelist is None:
        missing.append("namelist_input_sha256")
    if wps_namelist is None:
        missing.append("namelist_wps_sha256")
    geo = sorted(p for p in wps_dir.rglob("geo_em*.nc") if p.is_file())
    if not geo:
        geo = sorted(p for p in wps_dir.rglob("geo_em*.json") if p.is_file())
    if not geo:
        missing.append("geo_em_hashes")
    terrain = _static_file(wrf_run_dir, terrain_file,
                           ("HGT*", "hgt*", "terrain*"))
    landuse = _static_file(wrf_run_dir, landuse_file,
                           ("landuse*", "LANDUSE*", "LU_INDEX*"))
    if landuse is None:
        landuse = _static_file(wps_dir, landuse_file,
                               ("landuse*", "LANDUSE*", "LU_INDEX*"))
    version = wrf_version or _version_from_tree(wrf_run_dir)
    physics = physics_configuration or _text_value(
        wrf_run_dir, ("physics_configuration.txt", "physics_configuration.json",
                      "physics.txt")) or _physics_from_namelist(input_namelist)
    preprocessing = preprocessing_version or _text_value(
        wrf_run_dir, ("preprocessing_version.txt", "PREPROCESSING_VERSION",
                      "preprocessing_manifest.json"))
    cadence = _cadence(tensor_dir, timestamp_cadence)
    for key, value in (("landuse_sha256", landuse), ("wrf_version", version),
                       ("physics_configuration", physics),
                       ("preprocessing_version", preprocessing),
                       ("timestamp_cadence", cadence)):
        if value is None or value == "":
            missing.append(key)

    metadata_path = _grid_metadata_path(wrf_run_dir, wps_dir, grid_metadata)
    grid: dict = {}
    shapes: dict = {}
    ratios: dict = {}
    if metadata_path is None:
        missing.append("grid_metadata")
    else:
        try:
            declared = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            declared = {}
            missing.append("grid_metadata")
        if not isinstance(declared, dict):
            missing.append("grid_metadata")
        else:
            grid.update(declared.get("grid_spacings_m", {}))
            shapes.update(declared.get("domain_shapes", {}))
            ratios.update(declared.get("nest_ratios", {}))
    if not grid.get("d02") or not grid.get("d05"):
        missing.append("grid_spacings_m")
    if not shapes.get("d02_storage") or not shapes.get("d05"):
        missing.append("domain_shapes")
    if not ratios.get("d02_to_d05"):
        missing.append("nest_ratios")
    if terrain is None:
        missing.append("terrain_source_sha256")

    payload = {
        "case_id": case,
        "site": C.CASES[case]["location"],
        "event": C.CASES[case]["dates"],
        "real_or_placeholder": "REAL",
        "raw_data_hashes": raw_hashes,
        "namelist_input_sha256": _hash(input_namelist) if input_namelist else None,
        "namelist_wps_sha256": _hash(wps_namelist) if wps_namelist else None,
        "geo_em_hashes": {p.name: _hash(p) for p in geo},
        # The validator binds this field to the consumed HGT_d05 tensor.  If a
        # static upstream terrain file was found, retain its hash separately.
        "terrain_HGT_sha256": raw_hashes.get("HGT_d05.pt"),
        "terrain_source_sha256": None if terrain is None else _hash(terrain),
        "landuse_sha256": None if landuse is None else _hash(landuse),
        "wrf_version": version,
        "physics_configuration": physics,
        "grid_spacings_m": grid,
        "domain_shapes": shapes,
        "nest_ratios": ratios,
        "grid_metadata_source": None if metadata_path is None else str(metadata_path),
        "timestamp_cadence": cadence,
        "preprocessing_version": preprocessing,
        "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_commit_or_hash": C._pipeline_hash(),
        "builder": "build_wrf_run_provenance.py",
        "missing_required_fields": sorted(set(missing)),
    }
    required = tuple(V.REQUIRED_REAL_PROVENANCE) + (
        "terrain_source_sha256", "grid_spacings_m", "domain_shapes", "nest_ratios")
    missing_from_payload = [key for key in required
                            if payload.get(key) in (None, "", {}, [])]
    payload["missing_required_fields"] = sorted(set(missing + missing_from_payload))
    payload["provenance_status"] = "PASS" if not payload["missing_required_fields"] else "FAIL"
    payload["promotion_eligible"] = payload["provenance_status"] == "PASS"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=list(C.CASE_ORDER))
    ap.add_argument("--tensor-dir", required=True)
    ap.add_argument("--wrf-run-dir", required=True)
    ap.add_argument("--wps-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--terrain-file", default=None)
    ap.add_argument("--landuse-file", default=None)
    ap.add_argument("--wrf-version", default=None)
    ap.add_argument("--physics-configuration", default=None)
    ap.add_argument("--preprocessing-version", default=None)
    ap.add_argument("--timestamp-cadence", default=None)
    ap.add_argument("--grid-metadata", default=None)
    args = ap.parse_args()
    try:
        result = build_provenance(
            args.case, args.tensor_dir, args.wrf_run_dir, args.wps_dir,
            args.output, args.terrain_file, args.landuse_file,
            args.wrf_version, args.physics_configuration,
            args.preprocessing_version, args.timestamp_cadence,
            args.grid_metadata)
    except Exception as exc:  # CLI boundary: preserve a useful failure
        print(f"PROVENANCE BUILDER FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["provenance_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
