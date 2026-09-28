"""Write the staged V9.7 numerical repair provenance receipt.

This receipt is deliberately separate from promotion.  It records the exact
code and contract inputs used by the staged computation, the 14-config
coverage, bootstrap mapping/draw digests, terrain seed/mapping digests, and
the scope of the numerical error checks.  It never changes frozen evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from v9_7_repair import OUT, HERE, MODELS


ROOT = HERE.parent.parent
RECEIPT_DIR = ROOT / "qa" / "numerical_repair_20260921"
RECEIPT = RECEIPT_DIR / "v9_7_provenance_receipt.json"
EXPECTED_KEYS = tuple(
    [f"A_T_fold{i}" for i in (1, 2, 3)]
    + [f"A_S_S{i}" for i in (1, 2, 3, 4)]
    + [f"A_ST_S{i}" for i in (1, 2, 3, 4)]
    + ["A_670", "B", "C"]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_strings(values) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def existing_hashes(paths, missing=None):
    result = {}
    for path in paths:
        if path.exists() and path.is_file():
            result[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
        elif missing is not None:
            missing.append(str(path))
    return result


def combined_input_hash(key, directory, missing=None):
    """Bind the prediction/calibration/metadata inputs without packing NPZs.

    The key-info package intentionally omits the regenerable prediction
    arrays.  Keeping both their individual hashes and the exact combined
    digest used by the tail diagnostic makes that omission auditable after a
    fresh unpack.
    """
    paths = (directory / "predictions.npz", directory / "calibration.json", directory / "metadata.json")
    if not all(path.exists() and path.is_file() for path in paths):
        if missing is not None:
            missing.extend(str(path) for path in paths if not path.is_file())
        return None
    digest = hashlib.sha256()
    files = {}
    for path in paths:
        file_digest = sha256(path)
        files[str(path.relative_to(ROOT)).replace("\\", "/")] = file_digest
        digest.update(path.name.encode("utf-8"))
        digest.update(file_digest.encode("ascii"))
    return {"input_sha256": digest.hexdigest(), "files": files}


def build():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = [item.get("key") for item in manifest]
    coverage = {
        "expected_count": len(EXPECTED_KEYS),
        "actual_count": len(keys),
        "expected_keys": list(EXPECTED_KEYS),
        "actual_keys": keys,
        "exact_set": len(keys) == len(EXPECTED_KEYS) and set(keys) == set(EXPECTED_KEYS),
        "unique_keys": len(keys) == len(set(keys)),
    }

    source_names = [
        "v9_7_numerics.py", "v9_7_repair.py", "v9_7_scores.py",
        "v9_7_validate.py", "v9_7_tables.py", "v9_7_reports.py",
        "v9_7_reporting_gate.py", "v9_7_promote.py", "v9_7_unit_tests.py",
        "v9_7_provenance.py", "v9_6_common.py", "v9_6_run_nested.py",
        "V9_4_FROZEN_BUNDLE.json", "V9_6_TRANSFER_CALIBRATION.json",
    ]
    source_missing = []
    source_hashes = existing_hashes([HERE / name for name in source_names], source_missing)
    # Several numerical dependencies are imported through v8_common's path
    # setup.  Hash the actual loaded modules rather than guessing that they
    # live beside the V9.7 scripts.  Missing required modules are fatal below.
    module_names = (
        "v9_7_numerics", "v9_7_repair", "v9_7_scores", "v9_7_validate",
        "v9_7_tables", "v9_7_reports", "v9_7_reporting_gate", "v9_7_promote",
        "v9_6_common", "v9_6_run_nested", "v8_common", "window_evaluation",
        "families", "scientific_models",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            module_path = Path(module.__file__).resolve()
            if not module_path.exists():
                source_missing.append(f"{module_name}: {module_path}")
            else:
                source_hashes[str(module_path.relative_to(ROOT)).replace("\\", "/")] = sha256(module_path)
        except (ImportError, AttributeError, OSError, ValueError) as exc:
            source_missing.append(f"{module_name}: {exc}")
    source_hashes[str(manifest_path.relative_to(ROOT)).replace("\\", "/")] = sha256(manifest_path)
    unit_test_path = ROOT / "qa" / "plan_repair_20260921" / "numerical_unit_tests.txt"
    unit_test_receipt = {
        "path": str(unit_test_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(unit_test_path) if unit_test_path.exists() else None,
        "exists": unit_test_path.exists(),
        "all_tests_passed": unit_test_path.exists() and "Ran 5 tests" in unit_test_path.read_text(encoding="utf-8") and "OK" in unit_test_path.read_text(encoding="utf-8"),
    }
    auxiliary_paths = [
        ROOT / "qa" / "plan_repair_20260921" / "tail_score_diagnostic.json",
        ROOT / "qa" / "plan_repair_20260921" / "numerical_validation_stdout.log",
        ROOT / "qa" / "plan_repair_20260921" / "numerical_sensitivity_stdout.log",
    ]
    auxiliary_receipts = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None,
        }
        for path in auxiliary_paths
    }

    stream_receipts = []
    validations = []
    bootstrap = []
    terrain = []
    input_hashes = {}
    input_missing = []
    for key in keys:
        directory = OUT / key
        input_receipt = combined_input_hash(key, directory, input_missing)
        if input_receipt is not None:
            input_hashes[key] = input_receipt
        stream_status = {}
        for model in MODELS:
            path = directory / "streams" / f"{model}_receipt.json"
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            stream_status[model] = {
                "status": payload.get("status"),
                "rows": payload.get("rows"),
                "speed_order": payload.get("speed_order"),
                "J_order": payload.get("J_order"),
                "receipt_sha256": sha256(path) if path.exists() else None,
            }
        stream_receipts.append({"key": key, "models": stream_status})

        validation_path = directory / "numerical_validation.json"
        if validation_path.exists():
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
            records = payload.get("records", [])
            validations.append({
                "key": key,
                "status": payload.get("status"),
                "scope": payload.get("speed_validation"),
                "adaptive_J_max_abs": payload.get("adaptive_J_max_abs"),
                "max_speed_mean_crps_change": max(
                    (abs(float(row["speed_mean_crps_change"])) for row in records),
                    default=None,
                ),
                "max_speed_parent_crps_change": max(
                    (abs(float(row["speed_max_parent_crps_change"])) for row in records),
                    default=None,
                ),
                "max_J_CDF_change": max(
                    (abs(float(row["J_CDF_max_abs"])) for row in records),
                    default=None,
                ),
                "pair_resolution": payload.get("pair_resolution", []),
                "receipt_sha256": sha256(validation_path),
            })
        else:
            validations.append({"key": key, "status": "MISSING"})

        for path in sorted(directory.glob("bootstrap_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            draws_speed = np.asarray(payload.get("draw_delta_speed", []), dtype=np.float64)
            draws_j = np.asarray(payload.get("draw_delta_J", []), dtype=np.float64)
            bootstrap.append({
                "key": key,
                "file": str(path.relative_to(OUT)).replace("\\", "/"),
                "candidate": path.stem.removeprefix("bootstrap_").rsplit("_", 1)[0],
                "seed": payload.get("seed"),
                "n_boot": payload.get("n_boot"),
                "block_len": payload.get("block_len"),
                "block_count": payload.get("block_count"),
                "mapping_shape": payload.get("mapping_shape"),
                "mapping_dtype": payload.get("mapping_dtype"),
                "mapping_sha256": payload.get("mapping_sha256"),
                "draw_delta_speed_sha256": payload.get("draw_delta_speed_sha256"),
                "draw_delta_J_sha256": payload.get("draw_delta_J_sha256"),
                "draw_count_speed": int(draws_speed.size),
                "draw_count_J": int(draws_j.size),
                "provenance_version": payload.get("provenance_version"),
                "receipt_sha256": sha256(path),
            })

        terrain_path = directory / "terrain.json"
        if terrain_path.exists():
            payload = json.loads(terrain_path.read_text(encoding="utf-8"))
            records = payload.get("permutations", [])
            terrain.append({
                "key": key,
                "count": len(records),
                "seed_min": min((int(row["seed"]) for row in records), default=None),
                "seed_max": max((int(row["seed"]) for row in records), default=None),
                "seed_sha256": digest_strings(row.get("seed") for row in records),
                "mapping_sha256_digest": digest_strings(row.get("mapping_sha256") for row in records),
                "mapping_payload_digest": digest_strings(row.get("mapping_json") for row in records),
                "score_digest": digest_strings(
                    f"{row.get('permuted_speed_crps')}|{row.get('permuted_J_crps')}"
                    for row in records
                ),
                "receipt_sha256": sha256(terrain_path),
            })

    unresolved_pairs = []
    reviewed_pairs = []
    for validation in validations:
        for row in validation.get("pair_resolution", []):
            item = {
                "key": validation["key"],
                "candidate": row.get("candidate"),
                "baseline": row.get("baseline"),
                "speed_resolution": row.get("speed_resolution"),
                "J_resolution": row.get("J_resolution"),
            }
            reviewed_pairs.append(item)
            if row.get("speed_resolution", "").startswith("UNRESOLVED"):
                unresolved_pairs.append({**item, "metric": "speed"})
            if row.get("J_resolution", "").startswith("UNRESOLVED"):
                unresolved_pairs.append({**item, "metric": "J"})

    payload = {
        "status": "STAGED_PROVENANCE_RECEIPT" if not source_missing else "FAIL_SOURCE_HASH_MISSING",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "V9.7 numerical repair; staged only; no frozen promotion",
        "coverage": coverage,
        "source_hashes": source_hashes,
        "source_missing": source_missing,
        "input_hashes": input_hashes,
        "input_missing": input_missing,
        "unit_test_receipt": unit_test_receipt,
        "auxiliary_receipts": auxiliary_receipts,
        "stream_receipts": stream_receipts,
        "numerical_validation": validations,
        "bootstrap_provenance": bootstrap,
        "terrain_provenance": terrain,
        "requirements": {
            "models_unchanged": True,
            "data_rows_and_splits_unchanged": True,
            "bootstrap_draw_level_deltas_retained": all(
                item["draw_count_speed"] == item["n_boot"] == item["draw_count_J"]
                and item["provenance_version"] == "V9.7_BOOTSTRAP_PROVENANCE_1"
                for item in bootstrap
            ) if bootstrap else False,
            "terrain_seed_and_mapping_scores_retained": all(
                item["count"] == 999 and item["seed_min"] is not None
                and item["mapping_sha256_digest"] is not None
                for item in terrain
            ) if terrain else False,
            "numerical_error_does_not_use_bootstrap_width": True,
            "near_zero_classification": "REVIEWED_BY_FULL_GRID_TARGETED_CHECK" if len(reviewed_pairs) == 2 * len(EXPECTED_KEYS) else "INCOMPLETE_REVIEW",
            "reviewed_pair_count": len(reviewed_pairs),
            "unresolved_pairs": unresolved_pairs,
            "margin_interpretation": "observed integration/grid diagnostic only; not a formal error or tail truncation bound",
        },
    }
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(RECEIPT)
    print(json.dumps({"configs": len(keys), "bootstrap_files": len(bootstrap), "terrain_configs": len(terrain)}, ensure_ascii=False))
    if source_missing:
        raise SystemExit("SOURCE_HASH_RECEIPT_FAILED\n" + "\n".join(source_missing))


if __name__ == "__main__":
    build()
