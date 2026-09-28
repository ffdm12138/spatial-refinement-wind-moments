"""run_all — one-command orchestrator for the Paper0 multicase extension.

    python source/scripts/paper0_multicase/run_all.py
        [--cases A,B,C] [--phases contracts,00,01,...,17,figs,docs | all]
        [--data-root PATH] [--output-dir PATH] [--qa-dir PATH]
        [--registry-path PATH] [--no-independent-audit]

Order of operations (PLAN-faithful):
  contracts  CASE_CONTRACT.json + ANALYSIS_CONTRACT.json are written BEFORE
             any science output (pre-registration lock)
  00         raw contract + data_status detection (STOP per case on mismatch)
  01,04,05,07,10,12   per-case science phases
  09,13,14,15,16      cross-case transfers / bootstraps / sensitivity / tables
  17         audits + final audit document
  figs       diagnostic figures
  docs       README / MANIFEST / PROVENANCE / CLAIMS_PROVISIONAL

Idempotent: rerunning overwrites all derived outputs deterministically.
After swapping real data into the placeholder directories this same command
re-detects data_status via the hash mismatch and regenerates everything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402

ALL_PHASES = ["contracts", "00", "01", "04", "05", "07", "10", "12",
              "09", "13", "14", "15", "16", "17", "figs", "docs"]


# ---------------------------------------------------------------------------
# Pre-registration contracts (written before science)
# ---------------------------------------------------------------------------
def write_contracts() -> None:
    C.EXT.mkdir(parents=True, exist_ok=True)
    case_contract = {
        "extension": C.EXTENSION_NAME,
        "plan": C.GEN["plan_reference"],
        "design": C.GEN["design"],
        "cases": C.CASES,
        "comparisons": C.GEN["comparisons"],
        "raw_file_set": C.RAW_FILE_STEMS,
        "science_files": C.SCIENCE_STEMS,
        "manifest_only_files": C.MANIFEST_ONLY_STEMS,
        "analysis_window": {
            "raw_states": C.RAW_N,
            "excluded_initial": [0, C.EXCLUDED_INITIAL],
            "calibration": list(C.CAL_SLICE),
            "evaluation": list(C.EVAL_SLICE),
            "B_C_wording": ("fixed inherited analysis-window duration, "
                            "specified before examining the new-case science "
                            "results; never justified via AWS availability, "
                            "never retuned"),
            "state_count_rule": ("a case whose state count differs from 4320 "
                                 "is STOPPED with a discrepancy report; the "
                                 "split is never redefined automatically"),
        },
        "data_status_rule": ("raw files hashed against placeholder_manifest."
                             "json: all-match -> PLACEHOLDER; mismatch or no "
                             "manifest -> REAL; every output carries per-case "
                             "data_status"),
    }
    C.write_json(C.EXT / "CASE_CONTRACT.json", case_contract,
                 cases=list(C.CASE_ORDER))

    analysis_contract = {
        "master_seed": C.MASTER_SEED,
        "seed_spawn_keys": {k: list(v) for k, v in C.SEED_SPAWN.items()},
        "estimands": {
            "U_p": "parent vector magnitude (native d02)",
            "u_bar_p": "magnitude of the equal-weight mean fine vector (729 cells)",
            "s_bar_p": "mean fine-cell scalar speed",
            "J": "s_bar_p - |u_bar_p| >= 0",
            "delta_dyn": "|u_bar_p| - |U_p|",
            "delta_scalar": "s_bar_p - |U_p| = delta_dyn + J (exact)",
            "S": "population SD of the 729 fine speeds, ddof=0 (never sample SD)",
        },
        "grid": {"d05": [C.GRID, C.GRID], "parents": "6x6 row-major",
                 "block": C.BLOCK, "fine_per_parent": C.N_FINE,
                 "parent_centre_fine_indices": C.PARENT_CENTRE_FINE_INDICES},
        "scales_m": {str(k): C.SCALE_M[k] for k in C.SCALE_CELLS},
        "scale_rule": ("vector-average U/V over k x k blocks first, then "
                       "scalar quantities; J = S = 0 exact at 3000 m"),
        "interpolation": {
            "contract": "qa/npj_refactor/interpolation_contract_frozen.json",
            "frozen_contract_sha256": C.sha256_of(C.FROZEN_INTERP_CONTRACT),
            "modes": ["bilinear", "bicubic"], "align_corners": False,
            "output_size": [C.GRID, C.GRID],
            "component_order": "U/V separately, speed afterwards",
            "prediction_inputs": "d02 instantaneous U/V only",
            "scored_on": "final 670 evaluation states",
            "representative_state_rule": ("closest to median d02 domain-mean "
                                          "wind speed (1/36 sum of parent "
                                          "magnitudes), d02 only, earliest tie"),
            "constrained_variant": ("per-block constant U/V offset; block mean "
                                    "returns d02 parent vector exactly; max "
                                    "component residual < 1e-6 m/s"),
        },
        "second_order": {"formula": "J2nd = sigma_perp^2 / (2 |u_bar|)",
                         "speed_bin_edges": [float(x) for x in C.SPEED_BIN_EDGES[:-1]] + ["inf"],
                         "bin_variable": "|u_bar_p| per cell-time record",
                         "low_wind_reference_ms": C.LOW_WIND_MS,
                         "role": "descriptive reference only, never a tuned threshold"},
        "terrain": {
            "descriptors": ["r_sigma", "r_iqr", "lambda_iso"],
            "lambda_iso": ("8 directions, non-wrapping, physical direction, "
                           "cardinal 1 / diagonal 1/sqrt(2), dx/(L*L), "
                           "average of 8"),
            "screening": "FORBIDDEN (frozen identities only)",
            "relations": ["delta_scalar ~ r_sigma", "J ~ r_sigma",
                          "S ~ lambda_iso", "S ~ r_iqr + lambda_iso",
                          "S ~ r_iqr (companion)"],
            "fit_window": "calibration only",
            "in_range_rule": ("every cross-case terrain application carries "
                              "per-descriptor in_range/extrapolation flags "
                              "vs the source case's calibration range"),
        },
        "transfers": {
            "T1": "A frozen equations -> B (no refit)",
            "T2": "A calibration same-cell history / terrain / global -> B",
            "T3": (
                "frozen FORMS fit on B calibration only -> C unchanged; "
                "contemporaneous cross-site spatial transfer (same "
                "chronological/synoptic window, new geography and terrain "
                "regime, potentially different local wind realization); "
                "NOT a pure terrain transfer; secondary, never a headline "
                "gate"
                if C.GEN.get("bc_synchronized_bootstrap") else
                "frozen FORMS fit on B calibration only -> C unchanged; "
                "external equation-transfer stress test under a JOINT "
                "event+site shift (not spatial-only; never a headline gate)"),
            "T4": (
                "A frozen equations -> C (kept regardless of outcome); "
                "combined event+site external transfer -- event, site, "
                "terrain regime and local wind realization all change; "
                "strictest external stress test, secondary"
                if C.GEN.get("bc_synchronized_bootstrap") else
                "A frozen equations -> C (kept regardless of outcome); "
                "external equation-transfer stress test, secondary"),
            "history_boundary": ("A->B same-cell history legal (same grid); "
                                 "NO cross-site same-cell history for C; C "
                                 "gets only within-site calibration->evaluation "
                                 "persistence"),
            "frozen_sources": {
                "route_lock_sha256": C.sha256_of(C.FROZEN_ROUTE_LOCK),
                "model_cells_sha256": C.sha256_of(C.FROZEN_MODEL_CELLS),
                "scale_lock_sha256": C.sha256_of(C.FROZEN_SCALE_LOCK)},
        },
        "information_sources": {
            "scalar": ["zero_correction", "global_calibration_mean", "bilinear",
                       "bicubic", "terrain_roughness_equation",
                       C.PUBLIC_HISTORY_NAME],
            "spread": ["global_calibration_mean", "bilinear", "bicubic",
                       "roughness_only", "exposure_only",
                       "joint_terrain_equation", C.PUBLIC_HISTORY_NAME],
            "metrics": ["rmse", "crmse", "bias", "r", "slope",
                        "all_record_rmse (scalar only)"],
            "naming_lock": ("public name 'same-cell historical mean'; "
                            "cell_mean_persistence internal key only"),
        },
        "bootstrap": {
            "replicates": C.N_REPLICATES,
            "blocks_min": list(C.BLOCK_LENGTHS), "primary_block": C.PRIMARY_BLOCK,
            "pair_rule": ("all model pairs consume identical resampled time "
                          "indices within a replicate"),
            "BC_synchronized_ENABLED" if C.GEN.get("bc_synchronized_bootstrap")
            else "BC_paired_SUPERSEDED": (
                "ENABLED for generation 20260905 under "
                "PLAN_MULTICASE_20260905.md section 19.2: Case C (Ebao) "
                "declares the same 2015-12-01 chronological/synoptic window as "
                "Case B, machine-checked by the Phase-0 time_alignment_audit "
                "(same declared window, same 4320-state axis, same cadence "
                "contract). Each replicate draws ONE set of resampled "
                "chronological blocks consumed by BOTH sites; output is "
                "bootstrap/B_C_synchronized_case_differences.csv. Every "
                "contrast must be reported as a PAIRED CONTEMPORANEOUS "
                "CROSS-SITE DIFFERENCE: it removes part of the shared "
                "time-window variability, is NOT a causal terrain effect, and "
                "is never evidence of identical atmospheric states."
                if C.GEN.get("bc_synchronized_bootstrap") else
                "retired 2026-08-30 by PLAN_AMENDMENT_CASE_C_20260830.md section 7: "
                "its premise was that B and C share one December time axis, which "
                "fails once Case C is an independent event window -- identical "
                "resampled indices at the same ordinal position across two "
                "unrelated events carry no paired meaning. The already-written "
                "bootstrap/B_C_paired_case_differences.csv is kept on disk and "
                "registered, never deleted, and stays science_eligible=False. "
                "Replacement inference: per-case moving-block bootstrap with "
                "case-level estimates and intervals; a C-B contrast, if needed, "
                "uses INDEPENDENT block resampling per case and must be labelled "
                "as such."),
            "ci_meaning": ("time-sampling uncertainty conditional on the "
                           "domain; not geographic uncertainty"),
            "legacy_validation_seed": C.LEGACY_BOOTSTRAP_SEED,
        },
        "spinup": {
            "cutoffs_states": C.SPINUP_CUTOFFS, "primary": C.PRIMARY_CUTOFF,
            "rule": ("sensitivity only; never selects the window; headline "
                     "J/dyn/scalar/S over [c,4320) and [c,3650); Q_J/Q_S at "
                     "cutoff c over [c,4320) as a sensitivity-only definition "
                     "(primary Q remains evaluation-only)"),
        },
        "core_gates_mechanical": {
            "gate1": "J and S domain means > 1e-3 m/s in the new case",
            "gate1_note": ("large delta_dyn is NOT a failure; reading becomes "
                           "'Jensen dominance is event/site specific'"),
            "gate2": "max interpolation Q_J < 0.5 and Q_S < 0.9 (else headline counterexample)",
            "gate3": "frozen-identity slopes > 0 and correlation > 0.3",
            "gate4": "A->B Spearman >= 0.7 for delta and spread predictions",
            "gate5": ("best of T3/T4 evaluation Spearman >= 0.7 (external "
                      "site transfer: T3 contemporaneous cross-site, T4 "
                      "combined event+site stress test; never a spatial-only "
                      "or terrain-only transfer)" if C.GEN.get("bc_synchronized_bootstrap")
                      else ("best of T3/T4 evaluation Spearman >= 0.7 (JOINT event+site "
                            "shift stress test; not a spatial-only transfer)")),
            "gate_tiering": ("PRIMARY, and the only anchors of headline claim "
                             "MC-A-plus-B-plus-C-replication: gate1 (representation "
                             "deficit exists in each case) + gate2 (geometric "
                             "recovery remains incomplete). SECONDARY, describing "
                             "conditionality and never vetoing the headline: "
                             "gate3, gate4, gate5. Numeric thresholds and the "
                             "mechanical verdict-ladder strings are UNCHANGED by "
                             "the 2026-08-30 amendment, so the amendment itself "
                             "strengthens no claim."),
            "verdict_rule": ("CORE_CLAIM_CHALLENGED if headline counterexample; "
                             "MIXED if gate1 fails or gate2 partially fails; "
                             "STRONGLY_STRENGTHENED if gates 1-5 all pass; "
                             "else STRENGTHENED_WITH_LIMITS; Article verdict "
                             "additionally requires the Case-A reproduction "
                             "to be clean"),
        },
        "prohibitions": ["no outcome-based window retuning",
                         "no case-specific bins", "no descriptor screening",
                         "no metric swaps", "no cell-subset changes",
                         "counterexamples are registered, never deleted"],
    }
    C.write_json(C.EXT / "ANALYSIS_CONTRACT.json", analysis_contract,
                 cases=list(C.CASE_ORDER))
    print("[contracts] CASE_CONTRACT.json + ANALYSIS_CONTRACT.json written")


# ---------------------------------------------------------------------------
# Documentation (README with swap-in, MANIFEST, PROVENANCE, CLAIMS)
# ---------------------------------------------------------------------------
def _case_state_banner(case_meta: dict[str, dict]) -> str:
    """Build a current-state banner without assuming anything about Case C."""
    c_status = case_meta["C"]["case_status"]
    if c_status == C.CASE_STATUS_PLACEHOLDER:
        return ("**A/B REAL evidence is enabled; Case C remains a PLACEHOLDER "
                "engineering slot.** C-dependent claims and candidate rows are blocked.")
    if c_status == C.CASE_STATUS_REAL_PENDING_AUDIT:
        return ("**A/B REAL evidence is enabled; Case C is "
                "REAL_PENDING_AUDIT.** C-dependent claims and candidate rows "
                "remain blocked until promotion gates pass.")
    if c_status == C.CASE_STATUS_REAL:
        return ("**All cases REAL and audit-eligible.** C-dependent claims and "
                "candidate rows are enabled subject to their individual verdicts.")
    return f"**Case C status is {c_status}.** C-dependent claims remain blocked pending review."


def _case_transition_note(case_meta: dict[str, dict]) -> str:
    c_status = case_meta["C"]["case_status"]
    if c_status == C.CASE_STATUS_PLACEHOLDER:
        return ("Until promotion, engineering tables may retain C rows, but every "
                "such row is marked `science_eligible=false` and candidate export "
                "rejects it.")
    if c_status == C.CASE_STATUS_REAL_PENDING_AUDIT:
        return ("The real C payload is installed for audit only; engineering tables "
                "may retain pending rows, but candidate export remains blocked until "
                "the promotion gate passes.")
    if c_status == C.CASE_STATUS_REAL:
        return ("The C payload passed validation, complete WRF provenance and the "
                "promotion gate; C-dependent candidate rows now carry the same "
                "REAL hard fields as their live registry dependencies.")
    return "C-dependent outputs remain blocked until the live case state is resolved."


def write_docs(cases_ok: list[str]) -> None:
    all_cases = list(C.CASE_ORDER)
    case_meta = {c: C.case_metadata(c) for c in all_cases}
    project_meta = C.status_metadata(all_cases)
    status_line = ", ".join(
        f"{c}={case_meta[c]['case_status']}"
        for c in all_cases)
    status_detail = "\n".join(
        f"* Case {c}: `case_status={case_meta[c]['case_status']}`, "
        f"`data_status={case_meta[c]['data_status']}`, "
        f"`science_eligible={str(case_meta[c]['science_eligible']).lower()}`, "
        f"`contains_placeholder={str(case_meta[c]['contains_placeholder']).lower()}`, "
        f"`dependency_status={case_meta[c]['dependency_status']}`"
        for c in all_cases)
    c_status = case_meta["C"]["case_status"]
    placeholder = c_status == C.CASE_STATUS_PLACEHOLDER
    pending = c_status == C.CASE_STATUS_REAL_PENDING_AUDIT
    c_real = C.case_is_science_eligible(case_meta["C"])
    banner = _case_state_banner(case_meta)
    transition_note = _case_transition_note(case_meta)

    c_provenance_path = C.case_dir("C") / "WRF_RUN_PROVENANCE.json"
    c_provenance = {}
    if c_provenance_path.is_file():
        try:
            c_provenance = json.loads(c_provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            c_provenance = {}
    if c_real:
        c_provenance_note = (
            "* Case C REAL provenance: `provenance_status="
            f"{c_provenance.get('provenance_status', 'UNKNOWN')}`, "
            f"`promotion_eligible={str(c_provenance.get('promotion_eligible', False)).lower()}`; "
            "the seven observed tensor hashes are recorded in "
            "`case_C/WRF_RUN_PROVENANCE.json`.  The promotion receipt is "
            "`qa/paper0_multicase/case_C_status.json` and the audit receipt is "
            "`audit/promotion_gate.json`."
        )
    elif pending:
        c_provenance_note = (
            "* Case C real input is installed but pending promotion; its observed "
            "tensor hashes and provenance state are recorded in "
            "`case_C/WRF_RUN_PROVENANCE.json`.  Candidate export remains blocked."
        )
    elif placeholder:
        c_provenance_note = (
            "* Case C's current payload is the deterministic synthetic placeholder "
            "generated by `source/scripts/paper0_multicase/gen_placeholders.py`; its "
            "values are engineering dry-run values only and are not evidence."
        )
    else:
        c_provenance_note = "* Case C provenance is unavailable; all C-dependent outputs remain blocked."

    if C.GENERATION == "20260905":
        case_bullets = f"""* Case A = SEP_YAKOU (`1min_yakou20150924`, real legacy data; frozen results
  stay authoritative — reproduced read-only in
  `audit/case_A_legacy_reproduction.csv`).  Note: on 2026-08-29 the raw dir
  was renamed from `1min` (byte-identical content; on-disk .pt files carry
  the `_yakou20150924` suffix).
* Case B = DEC_YAKOU (`1min_yakou20151201`): **REAL**, same site and terrain,
  different atmospheric episode — the clean temporal replication leg.
* Case C = DEC_EBAO (`1min_ebao20151201`): **{C.case_state_label('C')}**,
  contemporaneous cross-site / terrain-regime replication.  Site = Ebao with
  author-supplied WRF namelist reference coordinates (`ref_lat=37.9492`,
  `ref_lon=100.9151`); the analysed-window mapping is not packaged on this
  machine and is never inherited from the retired Dayekou plan.  Identity is
  byte-attested by `CASE_C_EVENT_FREEZE.json` (schema v2), because the `.pt`
  files carry no absolute time.  B and C share the declared 2015-12-01 ~
  2015-12-03 window: B<->C contrasts are **paired contemporaneous cross-site
  differences** — never causal terrain effects, never claims of identical
  atmospheric states.  See `PLAN_MULTICASE_20260905.md`.
"""
        plan_line = (f"Pre-registered author contract: {C.GEN['plan_reference']} "
                     "(verbatim, do not edit).")
        rerun_env_hint = ("<data root containing 1min_yakou20150924, "
                          "1min_yakou20151201, 1min_ebao20151201>")
        swap_section = ""
    else:
        case_bullets = """* Case A = SEP_YAKOU (`1min_yakou20150924`, real legacy data; frozen results
  stay authoritative — reproduced read-only in
  `audit/case_A_legacy_reproduction.csv`).  Note: on 2026-08-29 the raw dir
  was renamed from `1min` (byte-identical content; on-disk .pt files carry
  the `_yakou20150924` suffix).
* Case B = DEC_YAKOU (`1min_yakou20151201`): **REAL**, same terrain, new event.
* Case C = DAYEKOU_REPLICATION (`1min_dayekou20151201` holds the placeholder
  dry-run only): **""" + C.case_state_label('C') + """**, independent site-event replication
  under a joint site + terrain + event shift.  Stable identity is
  `C` / `dayekou` / `independent_site_event`; the event window is
  `UNRESOLVED_PRE_SCIENCE` until `CASE_C_EVENT_FREEZE.json` freezes it — by
  raw-byte SHA-256, since the `.pt` files carry no absolute time — before any
  science phase runs.  See `PLAN_AMENDMENT_CASE_C_20260830.md`.
"""
        plan_line = ("Pre-registered author contract: `PLAN_PREREGISTERED.md` "
                     "(verbatim, do not edit).")
        rerun_env_hint = ("<data root containing 1min_yakou20150924, "
                          "1min_yakou20151201, 1min_dayekou20151201>")
        swap_section = """## One-command Dayekou hot swap

```powershell
& $env:NPJ_PYTHON source/scripts/paper0_multicase/run_multicase.py `
  --replace-case C `
  --input "<path>\\\\1min_dayekou20151201" `
  --mode real
```

The entry point validates the seven-file stable schema, exact Case-C identity,
chronology/static-field provenance and hashes before installing the payload.
It writes `REAL_PENDING_AUDIT`, runs the standard pipeline, and changes the
status to `REAL` only after the raw contract, complete WRF provenance,
independent audit, leakage audit and candidate contamination gates pass.  A
failed gate leaves the case pending and never labels it REAL.

""" if C.GEN.get("docs_hot_swap_section") else ""

    readme = f"""# {C.EXTENSION_NAME} — Paper0 multicase evidence build

{banner}

{plan_line}  Machine-readable locks: `CASE_CONTRACT.json`,
`ANALYSIS_CONTRACT.json` (written before any science output).  Current live
case state:
{status_detail}

Project dependency status: `{project_meta['case_status']}`, 
`science_eligible={str(project_meta['science_eligible']).lower()}`, 
`contains_placeholder={str(project_meta['contains_placeholder']).lower()}`, 
`dependency_status={project_meta['dependency_status']}`.

{case_bullets}
Code: `source/scripts/paper0_multicase/` (run receipts in
`qa/paper0_multicase/run_receipts/`).  Outputs: `raw_contract/`,
`{C.CASE_DIRS['A']}/`, `{C.CASE_DIRS['B']}/`, `{C.CASE_DIRS['C']}/`,
`cross_case/`, `bootstrap/`, `tables/` (M1–M12), `figure_data/`,
`diagnostic_figures/` (watermarked, never manuscript art), `audit/` (final
audit, independent reproduction, leakage, counterexamples, spin-up),
`comment_candidate/`, `article_candidate/`.

## One-command rerun

```powershell
$env:NPJ_MULTICASE_DATA_ROOT = "{rerun_env_hint}"
& $env:NPJ_PYTHON source/scripts/paper0_multicase/run_multicase.py
```

(Without the env var the machine-local pointer file
`qa/paper0_multicase/data_root_pointer.json` is used.)

{swap_section}No case ID, table schema, figure interface, scale column or comparison
definition changes.  {transition_note} Neither the rerun nor the hot-swap entry point edits
`latex_paper0/`, `latex_paper0_npjcomment/`, `data_paper0/frozen/` or
`source/evidence/`.
"""
    (C.EXT / "README.md").write_text(readme, encoding="utf-8")

    prereg_line = (
        "* Pre-registration: `PLAN_MULTICASE_20260905.md` recorded verbatim "
        "2026-09-05 BEFORE any Case-C (Ebao) science; `CASE_CONTRACT.json` "
        "and `ANALYSIS_CONTRACT.json` are its machine-readable locks and were "
        "written before any science output of this build."
        if C.GENERATION == "20260905" else
        "* Pre-registration: `PLAN_PREREGISTERED.md` recorded verbatim 2026-08-27\n"
        "  BEFORE Case B/C real data arrival; `CASE_CONTRACT.json` and\n"
        "  `ANALYSIS_CONTRACT.json` are its machine-readable locks and were written\n"
        "  before any science output of this build.")

    prov = f"""# PROVENANCE — {C.EXTENSION_NAME}

{banner}

{prereg_line}
* Raw inputs: per-case directories under the multicase data root
  (`{C.DATA_ROOT_ENV}`); SHA256 of every consumed file in
  `raw_contract/raw_hashes.json`; Case A hashes additionally match the sealed
  holdout manifest.
{chr(10).join(f"  * Case {c}: {case_meta[c]['data_status']} / {case_meta[c]['case_status']}; WRF provenance: `case_{c}/WRF_RUN_PROVENANCE.json`." for c in all_cases)}
{c_provenance_note}
* Frozen Case-A references consumed READ-ONLY: interpolation contract
  (`qa/npj_refactor/interpolation_contract_frozen.json`), route lock
  (terrain-equation coefficients), sealed holdout model-cell table
  (same-cell history / global means), scale-closure lock (J_hat/S_hat
  coefficients), plus the evidence CSVs used by
  `audit/case_A_legacy_reproduction.csv`.  SHA256 of the frozen coefficient
  sources are recorded in `ANALYSIS_CONTRACT.json`.
* All randomness (placeholders, bootstraps, relationship CIs) derives from
  master seed {C.MASTER_SEED} through the spawn keys listed in
  `ANALYSIS_CONTRACT.json`; the legacy Case-A bootstrap code-path validation
  alone reuses the frozen 20260807 seed.
* Station data: none anywhere in this extension (inputs are the seven raw
  tensors per case only).  T2 tensors are manifest-only and never enter the
  wind science.
* Every tabular output carries `data_status`, `case_status`,
  `science_eligible`, `contains_placeholder` and `dependency_status`.  JSON,
  binary and figure outputs carry the same fields in `_meta`/sidecars.
  Resolution is claim-level, not a global placeholder gate.
"""
    (C.EXT / "PROVENANCE.md").write_text(prov, encoding="utf-8")

    claim_rows = []
    for definition in C.CLAIM_DEFINITIONS:
        cm = C.claim_metadata(definition["claim_id"])
        if definition.get("engineering_only"):
            status = "ENGINEERING_ONLY"
        elif cm["dependency_status"] == "REAL" and cm["science_eligible"]:
            status = "REAL / SCIENCE_ENABLED"
        else:
            status = cm["dependency_status"]
        claim_rows.append(
            f"| {definition['claim_id']} | "
            f"{', '.join(definition['dependencies'])} | {definition['label']} "
            f"| {status} |")

    claims = f"""# CLAIMS_PROVISIONAL — {C.EXTENSION_NAME}

{banner}

Claim state is dependency-scoped.  A/B claims are not blocked by the C lifecycle.
{('C-dependent claims remain blocked and C numbers are never promoted while Case C is a placeholder.' if placeholder else
  'C-dependent claims remain blocked pending promotion while Case C is REAL_PENDING_AUDIT.' if pending else
  'C-dependent claims are enabled subject to their individual empirical verdicts now that Case C is REAL.' if c_real else
  'C-dependent claims remain blocked until Case C has a resolved live state.')}

Machine status: `case_status={project_meta['case_status']}`, 
`science_eligible={str(project_meta['science_eligible']).lower()}`, 
`contains_placeholder={str(project_meta['contains_placeholder']).lower()}`, 
`dependency_status={project_meta['dependency_status']}`.

| claim_id | dependencies | claim | live status |
|---|---|---|---|
{chr(10).join(claim_rows)}

Verdict lines live in `audit/FINAL_MULTICASE_P0_AUDIT.md`.  Promotion
requires the human instruction `MULTICASE AUDIT PASSED — PROMOTE`.
"""
    (C.EXT / "CLAIMS_PROVISIONAL.md").write_text(claims, encoding="utf-8")

    # MANIFEST: enumerate outputs with sizes + hashes (skip the PLAN itself)
    lines = [f"# MANIFEST — {C.EXTENSION_NAME}", "",
             banner, "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; "
             f"case_status: {project_meta['case_status']}.", "",
             "| file | bytes | sha256 (first 16) |", "|---|---|---|"]
    for p in sorted(C.EXT.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.md":
            continue
        rel = p.relative_to(C.EXT).as_posix()
        lines.append(f"| {rel} | {p.stat().st_size} | {C.sha256_of(p)[:16]} |")
    (C.EXT / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[docs] README / PROVENANCE / CLAIMS_PROVISIONAL / MANIFEST written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="A,B,C")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None,
                    help="alternate extension output root (isolated rehearsal)")
    ap.add_argument("--qa-dir", default=None,
                    help="alternate QA/receipt root (isolated rehearsal)")
    ap.add_argument("--registry-path", default=None,
                    help="alternate live registry path (isolated rehearsal)")
    ap.add_argument("--no-independent-audit", action="store_true")
    args = ap.parse_args()
    C.set_output_dirs(args.output_dir, args.qa_dir, args.registry_path)
    C.set_data_root(args.data_root)
    cases = [c.strip().upper() for c in args.cases.split(",") if c.strip()]
    phases = (ALL_PHASES if args.phases == "all"
              else [p.strip() for p in args.phases.split(",")])
    t0 = time.time()
    timings: dict[str, float] = {}

    def clock(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)

    if "contracts" in phases:
        write_contracts()
        clock("contracts")

    cases_ok = list(cases)
    if "00" in phases:
        import phase00_raw_contract as P0
        records = P0.main(cases)
        cases_ok = [c for c in cases if not records[c]["stop"]]
        clock("00")
    else:
        cases_ok = [c for c in cases
                    if not C.load_case_status(c).get("stop", False)]
    print(f"[run_all] cases passing the raw contract: {cases_ok}")

    per_case = {"01": "phase01_moments", "04": "phase04_scale",
                "05": "phase05_interpolation", "07": "phase07_terrain",
                "10": "phase10_windstate", "12": "phase12_infosources"}
    summaries = {}
    for ph, mod_name in per_case.items():
        if ph not in phases:
            continue
        mod = __import__(mod_name)
        for case in cases_ok:
            out = mod.main(case)
            if ph == "10":
                summaries[case] = out
        if ph == "10" and len(cases_ok) > 1:
            import phase10_windstate as P10
            P10.wind_state_comparison(cases_ok, summaries)
        if ph == "07" and cases_ok:
            import phase07_terrain as P7
            P7.regime_comparison(cases_ok)
        if ph == "12" and len(cases_ok) > 1:
            import phase12_infosources as P12
            P12.ranking(cases_ok)
        clock(ph)

    if "09" in phases:
        import phase09_transfers as P9
        P9.main(set(cases_ok))
        clock("09")
    if "13" in phases:
        import phase13_bootstrap as P13
        for case in cases_ok:
            P13.within_case(case)
        # 20260905: the synchronized B/C bootstrap is re-enabled under
        # PLAN_MULTICASE_20260905.md section 19.2 (declared contemporaneous
        # window, Phase-0 time_alignment_audit preconditions).  For generation
        # 20260826 paired_bc() stays retired per the 2026-08-30 amendment s7;
        # its existing output remains on disk, registered and
        # science_eligible=False.
        if (C.GEN.get("bc_synchronized_bootstrap")
                and {"B", "C"}.issubset(set(cases_ok))):
            P13.synchronized_bc()
        if "A" in cases_ok:
            P13.legacy_code_path_validation()
        clock("13")
    if "14" in phases:
        import phase14_spinup as P14
        P14.main(cases_ok)
        clock("14")
    if "15" in phases:
        import phase15_scaleclosure as P15
        P15.main(cases_ok)
        clock("15")
    metric_audit = None
    scientific_audit = None
    audit_result = None
    if "16" in phases:
        import phase16_tables as P16
        P16.main(cases_ok)
        clock("16")
    if "17" in phases:
        baseline_path = C.QA / "pre_repair_numeric_baseline_20260829.json"
        metric_report = C.ext_dir("audit") / "METRIC_REGRESSION_AUDIT.md"
        if C.GEN.get("regression_baselines") and baseline_path.is_file():
            import metric_regression_audit as MRA
            try:
                metric_audit = bool(MRA.audit(baseline_path, metric_report))
            except Exception as exc:  # keep the failure explicit in the receipt
                metric_audit = False
                metric_report.parent.mkdir(parents=True, exist_ok=True)
                metric_report.write_text(
                    "# METRIC_REGRESSION_AUDIT — FAILED\n\n"
                    f"The audit runner raised: {exc}\n", encoding="utf-8")
            if not metric_audit:
                print("[run_all] metric regression audit FAILED", file=sys.stderr)
        else:
            print(f"[run_all] metric baseline not present; regression audit is "
                  f"not available at {baseline_path}")
        scientific_baseline = C.QA / "future_c_promotion_regression_baseline_20260829.json"
        scientific_report = C.ext_dir("audit") / "SCIENTIFIC_REGRESSION_AUDIT.md"
        if C.GEN.get("regression_baselines") and scientific_baseline.is_file():
            import scientific_regression_audit as SRA
            try:
                scientific_audit = bool(SRA.audit(scientific_baseline,
                                                  scientific_report)["pass"])
            except Exception as exc:  # keep a failed scientific audit explicit
                scientific_audit = False
                scientific_report.parent.mkdir(parents=True, exist_ok=True)
                scientific_report.write_text(
                    "# SCIENTIFIC_REGRESSION_AUDIT — FAILED\n\n"
                    f"The audit runner raised: {exc}\n"
                    "\nSCIENTIFIC_NUMERIC_DRIFT = FOUND\n", encoding="utf-8")
            if not scientific_audit:
                print("[run_all] scientific regression audit FAILED", file=sys.stderr)
        else:
            print(f"[run_all] future-C numeric baseline not present; scientific "
                  f"regression audit is not available at {scientific_baseline}")
        import phase17_audit as P17
        audit_result = P17.main(cases_ok,
                                run_indep=not args.no_independent_audit)
        pending = [c for c in cases_ok
                   if C.case_record(c).get("case_status") ==
                   C.CASE_STATUS_REAL_PENDING_AUDIT]
        promotion = audit_result.get("promotion_gate", {})
        if pending and promotion.get("pass"):
            print(f"[run_all] promotion gate PASS; promoting {pending}")
            for case in pending:
                C.promote_real_case(case, promotion)
            # Promotion changes only dependency metadata.  Re-emit all master
            # tables/candidates and rerun Phase 17 so no pending label survives
            # in a real C science artifact.
            if "16" in phases:
                P16.main(cases_ok)
            if C.GEN.get("regression_baselines") and baseline_path.is_file():
                import metric_regression_audit as MRA
                metric_audit = bool(MRA.audit(baseline_path, metric_report))
            if (C.GEN.get("regression_baselines")
                    and scientific_baseline.is_file()):
                import scientific_regression_audit as SRA
                scientific_audit = bool(SRA.audit(scientific_baseline,
                                                  scientific_report)["pass"])
            audit_result = P17.main(cases_ok,
                                    run_indep=not args.no_independent_audit)
        elif pending:
            print("[run_all] promotion gate did not pass; pending case remains "
                  "REAL_PENDING_AUDIT")
        clock("17")
    if "figs" in phases:
        import diag_figures as DF
        DF.main(cases_ok)
        clock("figs")
    if "docs" in phases:
        write_docs(cases_ok)
        clock("docs")

    total = round(time.time() - t0, 1)
    C.run_receipt("run_all", {
        "cases_requested": cases, "cases_ok": cases_ok, "phases": phases,
        "data_status": {c: C.data_status(c) for c in cases_ok},
        "case_status": {c: C.case_record(c).get("case_status") for c in cases_ok},
        "science_eligible": {c: C.case_metadata(c)["science_eligible"] for c in cases_ok},
        "contains_placeholder": {c: C.case_metadata(c)["contains_placeholder"] for c in cases_ok},
        "dependency_status": {c: C.case_metadata(c)["dependency_status"] for c in cases_ok},
        "metric_regression_audit": metric_audit,
        "scientific_regression_audit": scientific_audit,
        "promotion_gate": (None if audit_result is None else
                            audit_result.get("promotion_gate")),
        "layout_note": "2026-08-29: Case A raw dir renamed 1min->1min_yakou20150924 "
                       "(byte-identical); per-case filename_suffix in CASE_CONTRACT.json",
        "raw_hashes_file": f"data_paper0/{C.EXTENSION_NAME}/raw_contract/raw_hashes.json",
        "timings_s": timings, "total_s": total,
    })
    print(f"[run_all] DONE in {total:.0f}s — timings {timings}")
    return 4 if metric_audit is False or scientific_audit is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
