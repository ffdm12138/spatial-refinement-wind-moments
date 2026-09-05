"""Shared configuration for the Paper0 multicase pipeline.

The pipeline is GENERATION-AWARE.  The active generation is selected with the
environment variable ``NPJ_MULTICASE_GENERATION`` (default ``20260905``) and
defined in ``mc_generations.py``:

  * ``20260826`` — the historical two-case audit architecture with a
    placeholder Case-C slot (retired; kept only for provenance).
  * ``20260905`` — three-real-case generation (A = yakou0924, B = yakou1201,
    C = ebao1201) under the pre-registered 20260905 plan; the placeholder
    site is fully retired, contemporaneous cross-site B<->C framing,
    master seed 20260905.  This is the default (and only supported)
    generation in this public reproduction package.

Implements the author's pre-registered contract for the active generation
(verbatim, recorded BEFORE any real-Case science of that generation).

Design rules (binding):
  * All writes go ONLY to ``results/<GEN extension>/`` and the run's QA
    subdirectory.  Shipped contracts and audited tables are read-only.
  * No machine paths are hard-coded.  The raw-data root comes from the
    environment variable ``NPJ_MULTICASE_DATA_ROOT`` (CLI-overridable),
    with a documented machine-local pointer file
    ``qa/paper0_multicase/data_root_pointer.json`` as fallback.
  * Analysis window, estimand definitions, interpolation contract, scale
    contract, descriptor definitions and public naming follow the frozen
    Case-A conventions exactly (see FROZEN_* loaders below).
  * Master seed per generation (20260826 / 20260905).  All randomness is
    derived from it through numpy SeedSequence spawn keys (see seed_for()).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from mc_generations import GENERATIONS

# ---------------------------------------------------------------------------
# Active generation (see mc_generations.py docstring)
# ---------------------------------------------------------------------------
GENERATION = (os.environ.get("NPJ_MULTICASE_GENERATION", "20260905").strip()
              or "20260905")
if GENERATION not in GENERATIONS:
    raise RuntimeError(
        f"unknown NPJ_MULTICASE_GENERATION {GENERATION!r}; registered "
        f"generations: {sorted(GENERATIONS)}")
GEN = GENERATIONS[GENERATION]
EXTENSION_NAME = GEN["extension_name"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[1]                       # repository root
EXT = ROOT / "results" / EXTENSION_NAME
QA = ROOT / "results" / "qa"
REGISTRY_PATH = ROOT / "results" / "multicase_registry.yaml"
SCRIPTS_DIR = _THIS.parent

# Read-only frozen references (never written)
FROZEN_ROUTE_LOCK = ROOT / "contracts" / "outputs_selection_route_lock.json"
FROZEN_MODEL_CELLS = ROOT / "contracts" / "outputs_evaluation_holdout_model_cells.csv"
FROZEN_BASELINE_SCORES = ROOT / "contracts" / "outputs_evaluation_holdout_baseline_scores.csv"
FROZEN_TERRAIN_METRICS = ROOT / "contracts" / "outputs_prepared_terrain_metrics.csv"
FROZEN_SEALED_MANIFEST = ROOT / "contracts" / "outputs_prepared_sealed_holdout_manifest.json"
FROZEN_INTERP_CONTRACT = ROOT / "contracts" / "interpolation_contract_frozen.json"
FROZEN_INTERP_METRICS = ROOT / "contracts" / "interpolation_metrics.csv"
FROZEN_INTERP_RECOVERY = ROOT / "contracts" / "interpolation_recovery.csv"
FROZEN_INTERP_BOOTSTRAP = ROOT / "contracts" / "interpolation_bootstrap.csv"
FROZEN_PVC_METRICS = ROOT / "contracts" / "parent_vector_constrained_metrics.csv"
FROZEN_PVC_RECOVERY = ROOT / "contracts" / "parent_vector_constrained_recovery.csv"
FROZEN_INFO_SOURCES = ROOT / "contracts" / "information_sources.csv"
FROZEN_ERROR_DECOMP = ROOT / "contracts" / "error_decomposition.csv"
FROZEN_J2_JSON = ROOT / "contracts" / "jensen_second_order.json"
FROZEN_J2_CSV = ROOT / "contracts" / "jensen_second_order.csv"
FROZEN_BOUNDARY = ROOT / "contracts" / "boundary_sensitivity.csv"
FROZEN_SPATIAL_CV = ROOT / "contracts" / "spatial_cv.csv"
FROZEN_STRATIFIED = ROOT / "contracts" / "state_stratified_terrain.csv"
FROZEN_WINDSTATE = ROOT / "contracts" / "wind_state_summary.json"
FROZEN_SCALE_LOCK = ROOT / "contracts" / "scale_closure_lock.json"
FROZEN_SCALE_DESC = ROOT / "contracts" / "scale_closure" / "terrain_descriptors.csv"
FROZEN_SCALE_CAL_MEANS = ROOT / "contracts" / "scale_closure" / "scale_cell_means_calibration.csv"
FROZEN_SCALE_TRANSFER = ROOT / "contracts" / "scale_closure" / "scale_aware_transfer_metrics.csv"
FROZEN_CAL_EVAL_CONTRACT = ROOT / "contracts" / "calibration_evaluation_contract.json"

# ---------------------------------------------------------------------------
# Grid / window constants (frozen; identical for A, inherited by B/C)
# ---------------------------------------------------------------------------
RAW_N = 4320
EXCLUDED_INITIAL = 970
CAL_SLICE = (970, 3650)          # 2680 states
EVAL_SLICE = (3650, 4320)        # 670 states
RETAINED_SLICE = (970, 4320)     # 3350 states
GRID = 162
BLOCK = 27
N_PARENT = 36
N_FINE = BLOCK * BLOCK           # 729
L_M = 3000.0
DX_M = L_M / BLOCK               # 111.111... m
SCALE_CELLS = (1, 3, 9, 27)
SCALE_M = {1: DX_M, 3: 3 * DX_M, 9: 9 * DX_M, 27: L_M}
PARENT_CENTRE_FINE_INDICES = [13 + 27 * k for k in range(6)]
INNER16_IDS = [r * 6 + c for r in range(1, 5) for c in range(1, 5)]

# Second-order diagnostic conventions (frozen from Case A)
SPEED_BIN_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, np.inf]
LOW_WIND_MS = 1.5                # descriptive reference only, never retuned

# Wind-state conventions (frozen from Case A)
SECTOR_LABELS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SECTOR_CENTRES_DEG = tuple(float(x) for x in range(0, 360, 45))
CALM_CUTOFF_MPS = 1.0e-12
MIN_STATES_PER_SECTOR = 300
OLS_INTERVAL_MULTIPLIER = 1.984
TERCILE_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)

# Bootstrap design (PLAN section 19; per-generation master seed)
MASTER_SEED = GEN["master_seed"]
N_REPLICATES = 2000
BLOCK_LENGTHS = (60, 120, 240)
PRIMARY_BLOCK = 120
LEGACY_BOOTSTRAP_SEED = 20260807   # ONLY for the A code-path validation

# Spin-up sensitivity (PLAN section 5.1 / 20); primary is always 970
SPINUP_CUTOFFS = {"6h": 360, "12h": 720, "970min": 970, "18h": 1080, "24h": 1440}
PRIMARY_CUTOFF = "970min"

# Public naming lock: journal-facing tables use "same-cell historical mean";
# cell_mean_persistence appears only as internal compatibility key.
PUBLIC_HISTORY_NAME = "same_cell_historical_mean"
INTERNAL_HISTORY_KEY = "cell_mean_persistence"

# ---------------------------------------------------------------------------
# Case registry (generation-specific; see mc_generations.py)
# ---------------------------------------------------------------------------
CASES = copy.deepcopy(GEN["cases"])
CASE_DIRS = dict(GEN["case_dirs"])
CASE_ORDER = ("A", "B", "C")
CASE_INDEX = {"A": 0, "B": 1, "C": 2}

# These are deliberately separate from the historical
# ``expected_placeholder`` fields in CASE_CONTRACT.json.  The latter records
# what was expected when the preregistered contract was written; the live
# state below is resolved from the raw payload and the Phase-0 audit.
CASE_STATUS_REAL = "REAL"
CASE_STATUS_PLACEHOLDER = "PLACEHOLDER"
CASE_STATUS_REAL_PENDING_AUDIT = "REAL_PENDING_AUDIT"
CASE_STATUS_FAILED = "FAILED"
CASE_STATUS_NOT_RUN = "NOT_RUN"
OUTPUT_METADATA_FIELDS = (
    "data_status", "case_status", "science_eligible",
    "contains_placeholder", "dependency_status",
)

CLAIM_DEFINITIONS = (
    {
        "claim_id": "MC-A-single-case",
        "label": "Case A single-case representation evidence",
        "dependencies": ["A"],
    },
    {
        "claim_id": "MC-B-single-case",
        "label": "Case B single-case representation evidence",
        "dependencies": ["B"],
    },
    {
        "claim_id": "MC-C-single-case",
        "label": "Case C single-case representation evidence",
        "dependencies": ["C"],
    },
    {
        "claim_id": "MC-A-to-B-temporal-transfer",
        "label": "A to B same-terrain event transfer",
        "dependencies": ["A", "B"],
    },
    {
        "claim_id": "MC-A-plus-B-replication",
        "label": "A plus B same-terrain multi-event replication",
        "dependencies": ["A", "B"],
    },
    {
        "claim_id": "MC-B-to-C-joint-shift-transfer",
        "label": "B to C joint event-and-site shift transfer (secondary "
                 "equation-transfer stress test, not a spatial-only claim)",
        "dependencies": ["B", "C"],
        "secondary": True,
    },
    {
        "claim_id": "MC-A-to-C-joint-shift-transfer",
        "label": "A to C joint event-and-site shift transfer (secondary "
                 "equation-transfer stress test, not a spatial-only claim)",
        "dependencies": ["A", "C"],
        "secondary": True,
    },
    {
        "claim_id": "MC-C-independent-replication",
        "label": "Case C independent site-event replication of the representation-"
                 "level deficit and the geometric-recovery limitation",
        "dependencies": ["C"],
    },
    {
        "claim_id": "MC-A-plus-B-plus-C-replication",
        "label": "HEADLINE: multi-event / multi-site replication that a coarse "
                 "parent vector under-determines J and S and that geometric "
                 "refinement does not restore them",
        "dependencies": ["A", "B", "C"],
        "headline": True,
    },
    {
        "claim_id": "MC-M4-scale-master-schema",
        "label": "M4 case by window by scale engineering schema",
        "dependencies": ["A", "B", "C"],
        "engineering_only": True,
    },
)

# Generation-scoped claim labels (IDs and dependencies never change; the
# 20260830 amendment banned ID churn).  For generation 20260826 this is a
# no-op, so the audited labels are reproduced exactly.
CLAIM_DEFINITIONS = tuple(
    {**d, "label": GEN["claim_label_overrides"].get(d["claim_id"], d["label"])}
    for d in CLAIM_DEFINITIONS
)

# Candidate scopes are part of the machine contract rather than prose in a
# README.  ``all_real_cases`` means that the exporter must not silently omit a
# newly promoted real case from a per-case artifact; pair/aggregate artifacts
# carry their own dependency status and are checked against the same registry.
CANDIDATE_SPECS = {
    "comment_candidate": {
        "selection_scope": "comment_core_evidence",
        "requires_case_C_when_real": True,
    },
    "article_candidate": {
        "selection_scope": "all_real_cases",
        "requires_case_C_when_real": True,
    },
}

# Logical file stems (identical to the legacy bare names, no case suffix).
# The on-disk name is resolved per case by raw_filename() using each case's
# filename_suffix (2026-08-29: Case A raw dir renamed 1min -> 1min_yakou20150924
# and its .pt files gained the _yakou20150924 suffix; Case B files carry
# _yakou20151201; Case C placeholder files remain bare until real data arrives).
RAW_FILE_STEMS = ["HGT_d05.pt", "T2_d02.pt", "T2_d05.pt",
                  "U10_d02.pt", "U10_d05.pt", "V10_d02.pt", "V10_d05.pt"]
SCIENCE_STEMS = ["U10_d02.pt", "V10_d02.pt", "U10_d05.pt", "V10_d05.pt", "HGT_d05.pt"]
MANIFEST_ONLY_STEMS = ["T2_d02.pt", "T2_d05.pt"]  # never enter P0 wind science


def raw_filename(case: str, stem: str) -> str:
    """Actual on-disk name for a logical stem, e.g.
    raw_filename("A", "U10_d02.pt") == "U10_d02_yakou20150924.pt".
    stem must end with '.pt'; returns stem unchanged when no suffix is set."""
    suffix = CASES[case].get("filename_suffix", "")
    return stem.removesuffix(".pt") + suffix + ".pt" if suffix else stem


def raw_stem(case: str, disk_name: str) -> str:
    """Inverse of raw_filename: logical stem from an actual disk name.
    The case suffix is inserted BEFORE '.pt', e.g.
    raw_stem("A", "U10_d02_yakou20150924.pt") == "U10_d02.pt"."""
    suffix = CASES[case].get("filename_suffix", "")
    if not suffix:
        return disk_name
    return disk_name.removesuffix(".pt").removesuffix(suffix) + ".pt"

# ---------------------------------------------------------------------------
# Data root resolution
# ---------------------------------------------------------------------------
DATA_ROOT_ENV = "NPJ_MULTICASE_DATA_ROOT"
_data_root_override: Path | None = None


def set_output_dirs(output_dir: str | None = None,
                    qa_dir: str | None = None,
                    registry_path: str | None = None) -> None:
    """Route generated outputs for a normal run or an isolated rehearsal.

    The default remains the canonical workspace tree.  Rehearsals use this
    hook so their raw payloads, reports, registry and receipts cannot touch
    the live evidence tree.
    """
    global EXT, QA, REGISTRY_PATH
    if output_dir:
        EXT = Path(output_dir).resolve()
    if qa_dir:
        QA = Path(qa_dir).resolve()
    if registry_path:
        REGISTRY_PATH = Path(registry_path).resolve()
    elif output_dir:
        REGISTRY_PATH = EXT.parent / "state" / "multicase_registry.yaml"


def set_data_root(path: str | None) -> None:
    global _data_root_override
    _data_root_override = Path(path).resolve() if path else None


def data_root() -> Path:
    """Resolve the multicase raw-data root.

    Priority: CLI override > NPJ_MULTICASE_DATA_ROOT env var >
    machine-local pointer file qa/paper0_multicase/data_root_pointer.json.
    """
    if _data_root_override is not None:
        return _data_root_override
    env = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    pointer_file = QA / "data_root_pointer.json"
    if pointer_file.is_file():
        payload = json.loads(pointer_file.read_text(encoding="utf-8"))
        return Path(payload["data_root"]).expanduser().resolve()
    raise RuntimeError(
        f"Raw-data root not configured: set {DATA_ROOT_ENV} or create the "
        f"machine-local pointer file {pointer_file} "
        '(JSON: {"data_root": "<path containing '
        + ", ".join(CASES[c]["raw_subdir"] for c in CASE_ORDER) + '>"}).')


def case_raw_dir(case: str) -> Path:
    return data_root() / CASES[case]["raw_subdir"]


def case_dir(case: str) -> Path:
    d = EXT / CASE_DIRS[case]
    d.mkdir(parents=True, exist_ok=True)
    return d


def ext_dir(name: str) -> Path:
    d = EXT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Seed policy (all derived from MASTER_SEED via spawn keys; documented in
# ANALYSIS_CONTRACT.json)
# ---------------------------------------------------------------------------
SEED_SPAWN = {
    "placeholder_synoptic": (100,),
    "placeholder_case_B": (101,),
    "placeholder_case_C": (102,),
    "placeholder_terrain_C": (103,),
    "bootstrap_within_A": (0, 1),
    "bootstrap_within_B": (1, 1),
    "bootstrap_within_C": (2, 1),
    "bootstrap_paired_BC": (99, 1),
    "relationship_boot_A": (0, 2),
    "relationship_boot_B": (1, 2),
    "relationship_boot_C": (2, 2),
}


def rng_for(purpose: str) -> np.random.Generator:
    key = SEED_SPAWN[purpose]
    return np.random.default_rng(np.random.SeedSequence(MASTER_SEED, spawn_key=key))


# ---------------------------------------------------------------------------
# Frozen reference loaders (read-only)
# ---------------------------------------------------------------------------
def frozen_route_lock() -> dict:
    return json.loads(FROZEN_ROUTE_LOCK.read_text(encoding="utf-8"))


def frozen_a_equations() -> dict:
    """Frozen Case-A terrain equations (calibration 2680, evaluation-frozen)."""
    lock = frozen_route_lock()
    models = lock["baselines"] if "baselines" in lock else {}
    # canonical location: lock["models"] carries mean + spread; per-baseline
    # coefficients live under lock["baseline_models"] or flat "models"
    out = {}
    flat = lock.get("models", {})
    if "mean" in flat:
        out["delta_vs_r_sigma"] = {
            "coefficients": flat["mean"]["coefficients"],
            "features": flat["mean"]["feature_order"],
        }
    if "spread" in flat:
        out["spread_joint"] = {
            "coefficients": flat["spread"]["coefficients"],
            "features": flat["spread"]["feature_order"],
        }
    # roughness_only / exposure_only from the per-baseline blocks
    def _walk(d):
        found = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "baseline_role" in v and "coefficients" in v:
                    found[v["baseline_role"]] = v
                else:
                    found.update(_walk(v))
        return found
    roles = _walk(lock)
    for role, key in [("roughness_only", "spread_roughness_only"),
                      ("exposure_only", "spread_exposure_only"),
                      ("terrain_roughness_equation", "delta_vs_r_sigma_check"),
                      ("locked_joint_equation", "spread_joint_check")]:
        if role in roles:
            out[key] = {"coefficients": roles[role]["coefficients"],
                        "features": roles[role]["feature_order"]}
    del models
    return out


def frozen_scale_lock() -> dict:
    return json.loads(FROZEN_SCALE_LOCK.read_text(encoding="utf-8"))


def frozen_a_calibration_ranges() -> dict:
    """Frozen A calibration descriptor ranges for in_range / extrapolation flags."""
    import pandas as pd
    lock = frozen_scale_lock()
    rng = dict(lock["terrain_descriptors"]["ranges"])
    tm = pd.read_csv(FROZEN_TERRAIN_METRICS)
    rng["r_iqr"] = [float(tm["terrain_iqr_ratio"].min()),
                    float(tm["terrain_iqr_ratio"].max())]
    return {"r_sigma": [float(x) for x in rng["r_sigma"]],
            "lambda_iso": [float(x) for x in rng["lambda_iso"]],
            "r_iqr": rng["r_iqr"]}


def frozen_a_history() -> dict:
    """Frozen A calibration same-cell historical means (sealed holdout table)."""
    import pandas as pd
    mc = pd.read_csv(FROZEN_MODEL_CELLS).sort_values("cell_id")
    return {
        "scalar": mc["mean_cell_mean_persistence_predicted"].to_numpy(float),
        "spread": mc["spread_cell_mean_persistence_predicted"].to_numpy(float),
        "global_scalar": float(mc["mean_global_training_mean_predicted"].iloc[0]),
        "global_spread": float(mc["spread_global_training_mean_predicted"].iloc[0]),
        "terrain_scalar": mc["mean_terrain_roughness_equation_predicted"].to_numpy(float),
        "terrain_spread_joint": mc["spread_locked_joint_equation_predicted"].to_numpy(float),
    }


def g_scale(ell: float, alpha: float, L: float = L_M, l0: float = DX_M) -> float:
    """Frozen P1 scale law G(l; alpha) = [(1 - l/L)/(1 - l0/L)]^alpha."""
    num = 1.0 - ell / L
    den = 1.0 - l0 / L
    if ell >= L - 1e-12:
        return 0.0
    if abs(ell - l0) < 1e-9:
        return 1.0
    return float(max(num / den, 0.0) ** alpha)


# ---------------------------------------------------------------------------
# data_status detection (placeholder branding contract)
# ---------------------------------------------------------------------------
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def case_status_file(case: str) -> Path:
    QA.mkdir(parents=True, exist_ok=True)
    if GEN.get("qa_status_subdir"):
        out = QA / GEN["qa_status_subdir"]
        out.mkdir(parents=True, exist_ok=True)
        return out / f"case_{case}_status.json"
    return QA / f"case_{case}_status.json"


def detect_data_status(case: str, raw_hashes: dict[str, str]) -> dict:
    """PLACEHOLDER iff a placeholder_manifest.json exists in the raw dir and
    every raw-file hash matches it; hash mismatch or no manifest => REAL."""
    rdir = case_raw_dir(case)
    manifest_path = rdir / "placeholder_manifest.json"
    if not manifest_path.is_file():
        prior = {}
        prior_path = case_status_file(case)
        if prior_path.is_file():
            try:
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = {}
        pending_marker = rdir / "real_pending_audit.json"
        newly_swapped = pending_marker.is_file()
        prior_case_status = prior.get("case_status", "")
        pending = (newly_swapped or prior_case_status in
                   (CASE_STATUS_PLACEHOLDER, CASE_STATUS_REAL_PENDING_AUDIT)
                   or prior.get("data_status") == "PLACEHOLDER"
                   or (case == "C" and prior_case_status != CASE_STATUS_REAL))
        return {
            "data_status": "REAL",
            "case_status": (CASE_STATUS_REAL_PENDING_AUDIT if pending
                             else CASE_STATUS_REAL),
            "placeholder_manifest_present": False,
            "detail": ("no placeholder manifest in raw directory; real payload "
                        "requires the Phase-17 audit before promotion"
                        if pending else "no placeholder manifest in raw directory"),
            "pending_marker_present": newly_swapped,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = manifest.get("files", {})
    mismatched = [n for n, h in raw_hashes.items()
                  if n in listed and listed[n].get("sha256") != h]
    missing = [n for n in raw_hashes if n not in listed]
    if not mismatched and not missing:
        return {"data_status": "PLACEHOLDER", "case_status": CASE_STATUS_PLACEHOLDER,
                "placeholder_manifest_present": True,
                "detail": "all raw hashes match the placeholder manifest",
                "placeholder_seed": manifest.get("seed")}
    return {"data_status": "REAL", "case_status": CASE_STATUS_REAL_PENDING_AUDIT,
            "placeholder_manifest_present": True,
            "detail": ("raw hashes differ from the stale placeholder manifest "
                       f"(mismatched={mismatched}, unlisted={missing}); real data "
                       "assumed swapped in — delete the stale manifest/notice"),
            "stale_manifest": True}


def load_case_status(case: str) -> dict:
    p = case_status_file(case)
    if not p.is_file():
        raise RuntimeError(
            f"case {case}: phase00 raw contract has not run (missing {p})")
    return json.loads(p.read_text(encoding="utf-8"))


def data_status(case: str) -> str:
    return load_case_status(case)["data_status"]


def case_record(case: str) -> dict:
    """Return the live Phase-0 status record for one case."""
    try:
        rec = load_case_status(case)
    except RuntimeError:
        return {
            "case": case, "data_status": "UNKNOWN",
            "case_status": CASE_STATUS_NOT_RUN, "stop": True,
            "stop_reasons": ["Phase-0 status is missing"],
        }
    rec.setdefault("case_status", rec.get("data_status", CASE_STATUS_NOT_RUN))
    rec.setdefault("stop", False)
    return rec


def metadata_from_values(data: str, status: str,
                         stop: bool = False) -> dict:
    """Build the four-field output contract without rereading a status file."""
    if stop:
        status = CASE_STATUS_FAILED
    placeholder = data == "PLACEHOLDER" or status == CASE_STATUS_PLACEHOLDER
    eligible = status == CASE_STATUS_REAL and not stop
    if stop:
        dep = "FAILED"
    elif placeholder:
        dep = "BLOCKED_PLACEHOLDER_DEPENDENCY"
    elif status == CASE_STATUS_REAL_PENDING_AUDIT:
        dep = "BLOCKED_REAL_PENDING_AUDIT"
    elif status == CASE_STATUS_REAL:
        dep = "REAL"
    else:
        dep = "NOT_RUN"
    return {
        "case_status": status,
        "data_status": data,
        "science_eligible": bool(eligible),
        "contains_placeholder": bool(placeholder),
        "dependency_status": dep,
    }


def case_metadata(case: str) -> dict:
    rec = case_record(case)
    return metadata_from_values(rec.get("data_status", "UNKNOWN"),
                                rec.get("case_status", CASE_STATUS_NOT_RUN),
                                bool(rec.get("stop", False)))


def case_is_science_eligible(case_or_record: str | dict) -> bool:
    """Single registry-backed predicate for scientific eligibility.

    A real tensor payload is not enough: the live status must be ``REAL``, the
    hard output flags must agree, and the dependency must resolve to ``REAL``.
    This helper is intentionally shared by candidate, audit, and document
    code so that ``REAL_PENDING_AUDIT`` cannot leak through as a near-real case.
    """
    meta = (case_metadata(case_or_record) if isinstance(case_or_record, str)
            else case_or_record)
    return bool(meta.get("case_status") == CASE_STATUS_REAL and
                meta.get("science_eligible") is True and
                meta.get("contains_placeholder") is False and
                meta.get("dependency_status") == "REAL")


def case_state(case: str) -> str:
    """Return the live case status used by state-aware docs and audits."""
    return str(case_metadata(case).get("case_status", CASE_STATUS_NOT_RUN))


def case_state_label(case: str) -> str:
    """Human-readable current-state sentence for generated documentation."""
    meta = case_metadata(case)
    status = meta["case_status"]
    if status == CASE_STATUS_PLACEHOLDER:
        return ("PLACEHOLDER_READY; current payload is engineering-only and "
                "is not science eligible")
    if status == CASE_STATUS_REAL_PENDING_AUDIT:
        return ("REAL_PENDING_AUDIT; real payload has been ingested but has "
                "not passed promotion gates")
    if status == CASE_STATUS_REAL:
        return "REAL; science eligible after provenance and science-audit promotion"
    if status == CASE_STATUS_FAILED:
        return "FAILED; raw contract or downstream audit stopped this case"
    return f"{status}; science eligibility is unavailable"


def _cases_from_status_value(value: str | None) -> list[str]:
    if not value:
        return []
    found = []
    for token in str(value).split(";"):
        m = re.match(r"^([A-Z])=", token.strip())
        if m and m.group(1) in CASE_ORDER:
            found.append(m.group(1))
    if not found and str(value).strip() in CASE_ORDER:
        found.append(str(value).strip())
    return list(dict.fromkeys(found))


def status_metadata(cases: list[str] | tuple[str, ...]) -> dict:
    """Aggregate output metadata for a dependency set."""
    cases = list(dict.fromkeys(cases))
    if not cases:
        return {
            "case_status": "NOT_RUN", "data_status": "UNKNOWN",
            "science_eligible": False, "contains_placeholder": False,
            "dependency_status": "NOT_RUN",
        }
    rows = {c: case_metadata(c) for c in cases}
    statuses = [rows[c]["case_status"] for c in cases]
    data = [rows[c]["data_status"] for c in cases]
    if any(s == CASE_STATUS_FAILED for s in statuses):
        dep = "FAILED"
    elif any(s == CASE_STATUS_PLACEHOLDER for s in statuses):
        dep = "BLOCKED_PLACEHOLDER_DEPENDENCY"
    elif any(s == CASE_STATUS_REAL_PENDING_AUDIT for s in statuses):
        dep = "BLOCKED_REAL_PENDING_AUDIT"
    elif all(s == CASE_STATUS_REAL for s in statuses):
        dep = "REAL"
    else:
        dep = "NOT_RUN"
    return {
        "case_status": (statuses[0] if len(statuses) == 1 else
                        ";".join(f"{c}={rows[c]['case_status']}" for c in cases)),
        "data_status": (data[0] if len(data) == 1 else
                        ";".join(f"{c}={rows[c]['data_status']}" for c in cases)),
        "science_eligible": bool(all(rows[c]["science_eligible"] for c in cases)),
        "contains_placeholder": bool(any(rows[c]["contains_placeholder"] for c in cases)),
        "dependency_status": dep,
    }


def claim_metadata(claim_id: str) -> dict:
    definition = next((x for x in CLAIM_DEFINITIONS if x["claim_id"] == claim_id), None)
    if definition is None:
        raise KeyError(f"unknown multicase claim: {claim_id}")
    meta = status_metadata(definition["dependencies"])
    return {**definition, **meta, "claim_id": claim_id}


def write_registry(cases: list[str] | tuple[str, ...] = CASE_ORDER) -> Path:
    """Write the sole live case/claim dependency registry as YAML."""
    import yaml
    cases = list(dict.fromkeys(cases))
    live = {case: case_metadata(case) for case in CASE_ORDER}
    if all(live[c]["case_status"] == CASE_STATUS_REAL for c in CASE_ORDER):
        stage = "MULTICASE_ALL_REAL_AUDITED"
    elif (live["A"]["case_status"] == CASE_STATUS_REAL and
          live["B"]["case_status"] == CASE_STATUS_REAL and
          live["C"]["case_status"] == CASE_STATUS_PLACEHOLDER):
        stage = "MULTICASE_AB_REAL_C_SLOT_READY"
    elif live["C"]["case_status"] == CASE_STATUS_REAL_PENDING_AUDIT:
        stage = "MULTICASE_C_REAL_PENDING_AUDIT"
    else:
        stage = "MULTICASE_STATUS_REQUIRES_REVIEW"
    registry = {
        "registry_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_stage": stage,
        "case_order": list(CASE_ORDER),
        "cases": {},
        "claims": {},
        "output_contract": {
            "required_fields": ["case_status", "science_eligible",
                                 "contains_placeholder", "dependency_status"],
            "legacy_compatibility_field": "data_status",
            "placeholder_policy": "engineering dry-run allowed; manuscript export denied",
        },
    }
    for case in CASE_ORDER:
        meta = case_metadata(case)
        registry["cases"][case] = {
            **CASES[case],
            **meta,
            "raw_directory": str(case_raw_dir(case)),
        }
    for definition in CLAIM_DEFINITIONS:
        registry["claims"][definition["claim_id"]] = claim_metadata(
            definition["claim_id"])
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(yaml.safe_dump(registry, sort_keys=False,
                                            allow_unicode=True), encoding="utf-8")
    return REGISTRY_PATH


def clear_pending_marker(case: str) -> None:
    marker = case_raw_dir(case) / "real_pending_audit.json"
    if marker.is_file():
        marker.unlink()


def promote_real_case(case: str, audit_receipt: dict | None = None) -> Path:
    """Promote a newly swapped real case only after the audit gate passes.

    This is the only code path that changes ``REAL_PENDING_AUDIT`` to ``REAL``.
    A caller must have already checked the complete Phase-17 audit; the
    function still refuses an incomplete provenance contract as a second
    fail-closed guard.
    """
    case = case.upper()
    rec = case_record(case)
    if rec.get("data_status") != "REAL":
        raise RuntimeError(f"case {case}: cannot promote non-REAL input")
    if rec.get("case_status") != CASE_STATUS_REAL_PENDING_AUDIT:
        raise RuntimeError(
            f"case {case}: promotion requires {CASE_STATUS_REAL_PENDING_AUDIT}, "
            f"got {rec.get('case_status')}")
    prov_path = case_dir(case) / "WRF_RUN_PROVENANCE.json"
    if not prov_path.is_file():
        raise RuntimeError(f"case {case}: missing {prov_path}")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    eligible = bool(prov.get("promotion_eligible"))
    if (not eligible and GEN.get("promote_with_partial_provenance")
            and prov.get("provenance_status") == "PARTIAL_UNKNOWN_NOT_PACKAGED"):
        # 20260905 generation: documented Case-B precedent (2026-08-29
        # architecture repair).  Upstream WRF artefacts are
        # UNKNOWN_NOT_PACKAGED and identity rests on the observed tensor
        # hashes plus the byte-frozen schema-v2 event register, which the
        # Phase-17 promotion gate has already verified as its own check.
        eligible = True
    if not eligible:
        raise RuntimeError(
            f"case {case}: WRF provenance is incomplete; promotion refused")
    payload = {
        "case": case,
        "data_status": "REAL",
        "case_status": CASE_STATUS_REAL,
        "science_eligible": True,
        "contains_placeholder": False,
        "dependency_status": "REAL",
        "stop": False,
        "stop_reasons": [],
        "raw_hashes": rec.get("raw_hashes", rec.get("hashes", {})),
        "detail": "real payload promoted after a passing standard audit",
        "promoted_after_audit": audit_receipt or {},
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    case_status_file(case).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    clear_pending_marker(case)
    write_case_provenance(case, payload)
    write_registry(CASE_ORDER)
    return case_status_file(case)


def _pipeline_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def write_case_provenance(case: str, raw_record: dict | None = None) -> Path:
    """Write the unified WRF provenance contract for one case.

    Missing upstream WRF artefacts are explicit ``UNKNOWN_NOT_PACKAGED``
    values.  They are never guessed from the tensor payload and keep a newly
    swapped real Case C in ``REAL_PENDING_AUDIT`` until the author supplies a
    complete provenance record.
    """
    rec = raw_record or case_record(case)
    raw_hashes = rec.get("hashes", rec.get("raw_hashes", {}))
    if not raw_hashes:
        raw_hashes = rec.get("raw_hashes", {})
    unknown = "UNKNOWN_NOT_PACKAGED"
    hgt_name = raw_filename(case, "HGT_d05.pt")
    payload = {
        "case_id": case,
        "site": CASES[case]["location"],
        "event": CASES[case]["dates"],
        "real_or_placeholder": rec.get("data_status", "UNKNOWN"),
        "raw_data_hashes": raw_hashes,
        "namelist_input_sha256": unknown,
        "namelist_wps_sha256": unknown,
        "geo_em_hashes": unknown,
        "terrain_HGT_sha256": raw_hashes.get(hgt_name, unknown),
        "landuse_sha256": unknown,
        "wrf_version": unknown,
        "physics_configuration": unknown,
        "grid_spacings_m": {"d02": 3000.0, "d05": float(DX_M)},
        "domain_shapes": {"d02_storage": [RAW_N, GRID, GRID],
                           "d05": [RAW_N, GRID, GRID]},
        "nest_ratios": {"d02_to_d05": BLOCK},
        "timestamp_cadence": "1-minute archived serialized order; absolute timestamps not packaged",
        "preprocessing_version": unknown,
        "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_commit_or_hash": _pipeline_hash(),
        "provenance_source": "tensor hash + registered case metadata; upstream WRF files are not packaged",
    }
    # A future real archive may ship a sidecar.  Merge its declared provenance
    # after preserving the registered case identity and freshly observed raw
    # hashes; no sidecar can relabel A/B/C or alter the tensor hash record.
    rdir = case_raw_dir(case)
    for sidecar_name in ("WRF_RUN_PROVENANCE.json", "case_input_metadata.json"):
        sidecar = rdir / sidecar_name
        if not sidecar.is_file():
            continue
        try:
            extra = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            extra = {}
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key not in {"case_id", "site", "event", "raw_data_hashes",
                               "generation_timestamp_utc", "pipeline_commit_or_hash"}:
                    payload[key] = value
        payload["provenance_source"] = f"{sidecar_name} + observed tensor hashes"
        break
    required = ["namelist_input_sha256", "namelist_wps_sha256", "geo_em_hashes",
                "landuse_sha256", "wrf_version", "physics_configuration",
                "preprocessing_version"]
    complete = (payload["real_or_placeholder"] == "REAL" and
                all(payload.get(k) not in (None, "", unknown, {}) for k in required))
    payload["provenance_status"] = ("COMPLETE" if complete else
                                    "PLACEHOLDER_NOT_APPLICABLE" if
                                    payload["real_or_placeholder"] == "PLACEHOLDER" else
                                    "PARTIAL_UNKNOWN_NOT_PACKAGED")
    payload["promotion_eligible"] = bool(complete)
    out = case_dir(case) / "WRF_RUN_PROVENANCE.json"
    write_json(out, payload, case=case)
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_pt(path: Path) -> np.ndarray:
    """Load a torch .pt tensor as numpy float32 (weights_only for safety)."""
    import torch
    t = torch.load(str(path), map_location="cpu", weights_only=True)
    a = np.asarray(t.numpy(), dtype=np.float32)
    del t
    return a


_tensor_cache: dict[tuple, np.ndarray] = {}


def load_case_wind(case: str, grid: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (U, V) float32 arrays for a case and grid ('d02'|'d05'), cached."""
    key = (case, grid)
    ku, kv = key + ("u",), key + ("v",)
    if ku not in _tensor_cache:
        rdir = case_raw_dir(case)
        _tensor_cache[ku] = load_pt(rdir / raw_filename(case, f"U10_{grid}.pt"))
        _tensor_cache[kv] = load_pt(rdir / raw_filename(case, f"V10_{grid}.pt"))
    return _tensor_cache[ku], _tensor_cache[kv]


def free_case_wind(case: str | None = None) -> None:
    if case is None:
        _tensor_cache.clear()
        return
    for k in [k for k in _tensor_cache if k[0] == case]:
        del _tensor_cache[k]


def load_case_hgt(case: str) -> np.ndarray:
    return load_pt(case_raw_dir(case) / raw_filename(case, "HGT_d05.pt")).astype(np.float64)


def write_json(path: Path, payload: dict, case: str | None = None,
               cases: list[str] | tuple[str, ...] | None = None,
               claim_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "extension": EXTENSION_NAME}
    dep_cases = list(cases or ([case] if case is not None else []))
    if dep_cases:
        env.update(status_metadata(dep_cases))
        if case is not None:
            env["case"] = case
        else:
            env["cases"] = dep_cases
        if env["contains_placeholder"]:
            env["branding"] = "PLACEHOLDER DRY-RUN — NOT SCIENCE"
    if claim_id is not None:
        env["claim_id"] = claim_id
    payload = {"_meta": env, **payload}
    payload["_meta"] = env
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               default=_json_default) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_df(path: Path, df, case: str | None = None,
             status_value: str | None = None,
             cases: list[str] | tuple[str, ...] | None = None) -> None:
    """Write a DataFrame with mandatory dependency metadata and LF endings.

    ``case`` gives row-level metadata for a single case.  A ``case`` column
    gives row-level metadata for mixed case tables.  ``status_value``/``cases``
    are for aggregate tables whose dependency set is uniform.  A caller may
    also provide all metadata columns explicitly for tables such as A-vs-B,
    B-vs-C and A-vs-C where each row has a different dependency pair.
    """
    df = df.copy()
    if case is not None:
        meta = case_metadata(case)
        for key, value in meta.items():
            df[key] = value
    elif "case" in df.columns:
        if len(df):
            cases_in_rows = [str(c) for c in df["case"]]
            if not all(c in CASE_ORDER for c in cases_in_rows):
                raise ValueError(
                    f"{path.name}: invalid case identifiers in case column")
            per_row = [case_metadata(c) for c in cases_in_rows]
            for key in OUTPUT_METADATA_FIELDS:
                df[key] = [m[key] for m in per_row]
        else:
            meta = status_metadata(list(cases or []))
            for key in OUTPUT_METADATA_FIELDS:
                df[key] = meta[key]
    elif status_value is not None:
        dep_cases = _cases_from_status_value(status_value)
        meta = status_metadata(dep_cases)
        meta["data_status"] = status_value
        for key in OUTPUT_METADATA_FIELDS:
            value = meta[key]
            df[key] = value
    elif cases is not None:
        meta = status_metadata(list(cases))
        for key in OUTPUT_METADATA_FIELDS:
            df[key] = meta[key]
    elif "data_status" in df.columns:
        # Backward-compatible inference for aggregate producers that already
        # carry the legacy status string.  This keeps the new four-field gate
        # mandatory without forcing every historical helper to know whether a
        # mixed row depends on A/B, B/C or A/C.
        inferred = []
        for value in df["data_status"]:
            text = str(value)
            dep_cases = _cases_from_status_value(text)
            if dep_cases:
                inferred.append(status_metadata(dep_cases))
            elif text in {"REAL", "PLACEHOLDER", "UNKNOWN"}:
                inferred.append(metadata_from_values(
                    text, (CASE_STATUS_PLACEHOLDER if text == "PLACEHOLDER"
                           else CASE_STATUS_REAL if text == "REAL"
                           else CASE_STATUS_NOT_RUN)))
            else:
                raise ValueError(
                    f"{path.name}: cannot infer dependency cases from "
                    f"data_status={text!r}")
        if len(inferred):
            for key in OUTPUT_METADATA_FIELDS:
                df[key] = [row[key] for row in inferred]
        else:
            meta = status_metadata(list(cases or []))
            for key in OUTPUT_METADATA_FIELDS:
                df[key] = meta[key]
    elif all(key in df.columns for key in OUTPUT_METADATA_FIELDS):
        # Explicit row-level metadata is used for mixed dependency tables.
        # Do not silently repair it: validate below and fail on nulls.
        pass
    else:
        raise ValueError(
            f"{path.name}: cannot infer output dependency metadata; provide "
            "case, a case column, cases/status_value, or all metadata fields")
    missing = [key for key in OUTPUT_METADATA_FIELDS if key not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing output metadata {missing}")
    for key in OUTPUT_METADATA_FIELDS:
        if df[key].isna().any():
            raise ValueError(f"{path.name}: null output metadata in {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def combined_status(*cases: str) -> str:
    return ";".join(f"{c}={data_status(c)}" for c in cases)


def combined_case_status(*cases: str) -> str:
    return ";".join(f"{c}={case_record(c).get('case_status', 'NOT_RUN')}"
                     for c in cases)


def run_receipt(step: str, payload: dict) -> Path:
    """Append a receipt json under qa/paper0_multicase/run_receipts/."""
    rdir = QA / "run_receipts"
    rdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out = rdir / f"{stamp}_{step}.json"
    base = {
        "step": step,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_seed": MASTER_SEED,
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
    }
    merged = {**base, **payload}
    found_cases = []
    for key in ("cases_ok", "cases_requested"):
        value = merged.get(key, [])
        if isinstance(value, (list, tuple)):
            found_cases.extend(str(c) for c in value if str(c) in CASE_ORDER)
    if isinstance(merged.get("case"), str) and merged["case"] in CASE_ORDER:
        found_cases.append(merged["case"])
    if isinstance(merged.get("cases"), dict):
        found_cases.extend(str(c) for c in merged["cases"] if str(c) in CASE_ORDER)
    found_cases = list(dict.fromkeys(found_cases))
    if found_cases:
        meta = status_metadata(found_cases)
        merged["status_metadata"] = meta
        for key in OUTPUT_METADATA_FIELDS:
            merged[key] = meta[key]
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False,
                              default=_json_default) + "\n", encoding="utf-8")
    return out
