"""Exercise the provenance builder on an isolated, hard-linked fixture."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_wrf_run_provenance as B  # noqa: E402
import mc_config as C  # noqa: E402
import validate_case_input as V  # noqa: E402


def main() -> int:
    C.set_output_dirs(None, None, None)
    C.set_data_root(None)
    canonical = C.case_raw_dir("C")
    result = {"fixture": "isolated hard-linked C tensor payload",
              "pass": False, "checks": {}}
    with tempfile.TemporaryDirectory(prefix="provenance_builder_audit_",
                                     dir=str(C.QA)) as temp:
        root = Path(temp)
        tensor = root / "1min_dayekou20151201"
        wrf = root / "wrf_run"
        wps = root / "WPS"
        tensor.mkdir()
        wrf.mkdir()
        wps.mkdir()
        source_files = V.resolve_payload_files("C", canonical)
        for stem, source in source_files.items():
            os.link(source, tensor / source.name)
        (wrf / "namelist.input").write_text("&physics\n mp_physics=6,\n "
                                             "ra_lw_physics=1,\n/\n", encoding="utf-8")
        (wps / "namelist.wps").write_text("&share\n max_dom=2,\n/\n", encoding="utf-8")
        for name in ("geo_em.d02.nc", "geo_em.d05.nc"):
            (wps / name).write_bytes(name.encode("ascii"))
        (wps / "grid_metadata.json").write_text(json.dumps({
            "grid_spacings_m": {"d02": 3000.0, "d05": float(C.DX_M)},
            "domain_shapes": {"d02_storage": [C.RAW_N, C.GRID, C.GRID],
                              "d05": [C.RAW_N, C.GRID, C.GRID]},
            "nest_ratios": {"d02_to_d05": C.BLOCK},
        }, indent=2) + "\n", encoding="utf-8")
        (wrf / "HGT.d05.bin").write_bytes(b"terrain-static-fixture")
        (wrf / "landuse.d05.bin").write_bytes(b"landuse-static-fixture")
        (wrf / "WRF_VERSION").write_text("WRF 4.5.2 fixture", encoding="utf-8")
        (wrf / "physics_configuration.txt").write_text(
            "mp_physics=6; ra_lw_physics=1", encoding="utf-8")
        (wrf / "preprocessing_version.txt").write_text(
            "WPS-preprocess fixture v1", encoding="utf-8")
        (tensor / "timestamps.json").write_text(
            json.dumps({"timestamps": list(range(C.RAW_N))}) + "\n",
            encoding="utf-8")
        out = tensor / "WRF_RUN_PROVENANCE.json"
        payload = B.build_provenance("C", tensor, wrf, wps, out)
        incomplete_tensor = root / "incomplete_payload"
        incomplete_wrf = root / "incomplete_wrf"
        incomplete_wps = root / "incomplete_wps"
        incomplete_tensor.mkdir()
        incomplete_wrf.mkdir()
        incomplete_wps.mkdir()
        for stem, source in source_files.items():
            os.link(source, incomplete_tensor / source.name)
        incomplete = B.build_provenance("C", incomplete_tensor, incomplete_wrf,
                                        incomplete_wps,
                                        incomplete_tensor / "WRF_RUN_PROVENANCE.json")
        result["checks"] = {
            "provenance_status": payload.get("provenance_status"),
            "missing_required_fields": payload.get("missing_required_fields"),
            "promotion_eligible": payload.get("promotion_eligible"),
            "raw_hash_count": len(payload.get("raw_data_hashes", {})),
            "hgt_hash_matches_tensor": payload.get("terrain_HGT_sha256") ==
                                        C.sha256_of(tensor / "HGT_d05.pt"),
            "geo_em_hash_count": len(payload.get("geo_em_hashes", {})),
            "static_landuse_hash_present": bool(payload.get("landuse_sha256")),
            "terrain_source_hash_present": bool(payload.get("terrain_source_sha256")),
            "grid_metadata_present": bool(payload.get("grid_metadata_source")),
            "timestamp_cadence_explicit": "verified timestamp sidecar" in
                                          str(payload.get("timestamp_cadence")),
            "incomplete_status": incomplete.get("provenance_status"),
            "incomplete_missing_field_count": len(
                incomplete.get("missing_required_fields", [])),
            "incomplete_no_unknown_fill": "UNKNOWN_NOT_PACKAGED" not in json.dumps(incomplete),
            "incomplete_not_promotion_eligible": not incomplete.get("promotion_eligible"),
        }
        result["pass"] = bool(
            result["checks"]["provenance_status"] == "PASS" and
            not result["checks"]["missing_required_fields"] and
            result["checks"]["promotion_eligible"] and
            result["checks"]["raw_hash_count"] == 7 and
            result["checks"]["hgt_hash_matches_tensor"] and
            result["checks"]["geo_em_hash_count"] == 2 and
            result["checks"]["static_landuse_hash_present"] and
            result["checks"]["terrain_source_hash_present"] and
            result["checks"]["grid_metadata_present"] and
            result["checks"]["timestamp_cadence_explicit"] and
            result["checks"]["incomplete_status"] == "FAIL" and
            result["checks"]["incomplete_missing_field_count"] > 0 and
            result["checks"]["incomplete_no_unknown_fill"] and
            result["checks"]["incomplete_not_promotion_eligible"])
    out = C.ext_dir("audit") / "PROVENANCE_BUILDER_AUDIT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# PROVENANCE_BUILDER_AUDIT — Paper0 multicase", "",
             "The builder was exercised on an isolated hard-linked tensor fixture; "
             "no canonical raw file or manuscript was changed.", "",
             f"Overall: **{'PASS' if result['pass'] else 'FAIL'}**.", "",
             "| check | result |", "|---|---|"]
    for key, value in result["checks"].items():
        lines.append(f"| {key} | `{json.dumps(value, ensure_ascii=False)}` |")
    lines += ["", "Missing required metadata is a hard `FAIL`; the builder never "
              "converts missing provenance to `UNKNOWN_NOT_PACKAGED`.", "",
              "Standard future command: `build_wrf_run_provenance.py --case C "
              "--tensor-dir <payload> --wrf-run-dir <WRF> --wps-dir <WPS> "
              "--output <payload>/WRF_RUN_PROVENANCE.json`. "]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.write_json(C.ext_dir("audit") / "provenance_builder_audit.json", result,
                 cases=["C"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
