"""Generation registry for the Paper0 multicase pipeline.

The pipeline is generation-aware: each generation is a self-contained
author contract (workspace tree, case identities, seed, wording locks).
Selecting a generation is done through the environment variable
``NPJ_MULTICASE_GENERATION`` (see ``mc_config.py``).

``20260826``
    The historical two-case audit architecture with a placeholder Case-C slot
    (PLAN_PREREGISTERED.md 2026-08-27 + PLAN_AMENDMENT_CASE_C_20260830.md).
    Left exactly as it was on 2026-08-30; rerunning this generation must
    reproduce the 20260826 workspace byte-for-byte semantics.

``20260905``
    Three-real-case generation authorised by the author on 2026-09-05:
    A = yakou0924, B = yakou1201, C = ebao1201.  The placeholder site is fully retired
    (no directory, no filename, no site string).  Scientific framing is
    tightened per PLAN_MULTICASE_20260905.md: A->B is the clean temporal
    replication, B<->C is a *contemporaneous cross-site / terrain-regime
    replication* (never a terrain-only causal contrast), and A->C is a
    combined external stress test.  Master seed 20260905.  The synchronized
    B/C moving-block bootstrap is re-enabled because B and C now share the
    declared 2015-12-01 window (output renamed to
    ``B_C_synchronized_case_differences.csv``), and Case-C promotion follows
    the documented Case-B precedent: upstream WRF artefacts are
    ``UNKNOWN_NOT_PACKAGED`` and identity rests on tensor hashes plus the
    byte-frozen event register (schema v2, site Ebao).
"""
from __future__ import annotations

GENERATIONS: dict[str, dict] = {
    # ------------------------------------------------------------------
    "20260826": {
        "extension_name": "extension_multicase_20260826",
        "plan_reference": (
            "PLAN_PREREGISTERED.md (verbatim author contract, 2026-08-27) "
            "as partially superseded by PLAN_AMENDMENT_CASE_C_20260830.md"),
        "design": (
            "hierarchical replication across increasing event-site shift "
            "(NOT a 2x2 factorial; NOT a controlled terrain-only design)"),
        "master_seed": 20260826,
        "case_dirs": {"A": "case_A", "B": "case_B", "C": "case_C"},
        "qa_status_subdir": "",
        "cases": {
            "A": {
                "case_id": "A",
                "label": "SEP_YAKOU",
                "raw_subdir": "1min_yakou20150924",
                "filename_suffix": "_yakou20150924",
                "location": "Yakou",
                "lat": 38.0142, "lon": 100.2421,
                "dates": "2015-09-24 to 2015-09-27",
                "role": "legacy reference / original discovery case",
                "expected_placeholder": False,
            },
            "B": {
                "case_id": "B",
                "label": "DEC_YAKOU",
                "raw_subdir": "1min_yakou20151201",
                "filename_suffix": "_yakou20151201",
                "location": "Yakou",
                "lat": 38.0142, "lon": 100.2421,
                "dates": "2015-12-01 to 2015-12-03",
                "role": "same terrain, different atmospheric event/state",
                "expected_placeholder": True,
            },
            "C": {
                "case_id": "C",
                "label": "DAYEKOU_REPLICATION",
                # Placeholder dry-run location only.  The REAL payload directory
                # is named by CASE_C_EVENT_FREEZE.json once the window is frozen
                # (amendment section 2).
                "raw_subdir": "1min_placeholder20151201",
                "filename_suffix": "",
                "location": "retired placeholder site (not used)",
                "lat": 38.556, "lon": 100.286,
                "dates": "UNRESOLVED - frozen by CASE_C_EVENT_FREEZE.json",
                "event_date": None,
                "event_selection_status": "UNRESOLVED_PRE_SCIENCE",
                "role": ("independent site-event replication: new site, new terrain "
                         "setting and an independent atmospheric event (joint domain "
                         "shift; NOT a terrain-only causal comparison)"),
                "expected_placeholder": True,
            },
        },
        "claim_label_overrides": {},
        "comparisons": {
            "A_vs_B": ("same site and terrain, independent atmospheric event "
                       "(same-site temporal replication)"),
            "B_vs_C": ("new site, new terrain setting and an independent event: "
                       "joint site-event shift. Role is generated from the frozen "
                       "Case-C window -- same_period_cross_site when the dates "
                       "overlap, otherwise independent_cross_site_event_"
                       "replication. terrain_only_causal = false in every branch."),
            "A_vs_C": ("joint site-event shift; total-difference reference only; "
                       "terrain_only_causal = false"),
        },
        "bc_synchronized_bootstrap": False,
        "time_alignment_audit": False,
        "promote_with_partial_provenance": False,
        "regression_baselines": True,
        "case_c_freeze_schema": "v1",
        "thunder": "twelve",
        "docs_hot_swap_section": True,
    },
    # ------------------------------------------------------------------
    "20260905": {
        "extension_name": "extension_multicase_20260905",
        "plan_reference": (
            "PLAN_MULTICASE_20260905.md (verbatim author contract, 2026-09-05; "
            "supersedes PLAN_PREREGISTERED.md and "
            "PLAN_AMENDMENT_CASE_C_20260830.md inside this workspace)"),
        "design": (
            "controlled replication triangle with a contemporaneous cross-site "
            "leg: A->B same-site temporal replication, B<->C contemporaneous "
            "cross-site / terrain-regime replication, A->C combined external "
            "stress test (NOT a 2x2 factorial; terrain_only_causal = false in "
            "every comparison)"),
        "master_seed": 20260905,
        "case_dirs": {"A": "case_A_yakou0924", "B": "case_B_yakou1201",
                      "C": "case_C_ebao1201"},
        "qa_status_subdir": "gen_20260905",
        "cases": {
            "A": {
                "case_id": "A",
                "label": "SEP_YAKOU",
                "raw_subdir": "1min_yakou20150924",
                "filename_suffix": "_yakou20150924",
                "location": "Yakou",
                "lat": 38.0142, "lon": 100.2421,
                "dates": "2015-09-24 to 2015-09-27",
                "role": "legacy reference / original discovery case",
                "expected_placeholder": False,
            },
            "B": {
                "case_id": "B",
                "label": "DEC_YAKOU",
                "raw_subdir": "1min_yakou20151201",
                "filename_suffix": "_yakou20151201",
                "location": "Yakou",
                "lat": 38.0142, "lon": 100.2421,
                "dates": "2015-12-01 to 2015-12-03",
                "role": "same site / same terrain, different atmospheric episode "
                        "(temporal replication leg)",
                "expected_placeholder": False,
            },
            "C": {
                "case_id": "C",
                "label": "DEC_EBAO",
                "raw_subdir": "1min_ebao20151201",
                "filename_suffix": "_ebao20151201",
                "location": "Ebao",
                "lat": 37.9492, "lon": 100.9151,
                "coordinates_provenance": (
                    "author-supplied WRF namelist reference coordinates "
                    "(ref_lat=37.9492, ref_lon=100.9151), 2026-09-05 session; "
                    "the mapping from these reference coordinates to the analysed "
                    "6x6 d02 footprint / 162x162 d05 window is not packaged on "
                    "this machine and is recorded as such, never inherited from "
                    "the retired placeholder plan"),
                "dates": "2015-12-01 to 2015-12-03",
                "role": ("contemporaneous cross-site / terrain-regime replication: "
                         "same chronological/synoptic time window as Case B while "
                         "differing in site, terrain regime and potentially local "
                         "wind realization (NOT a terrain-only causal comparison)"),
                "expected_placeholder": False,
            },
        },
        "claim_label_overrides": {
            "MC-B-to-C-joint-shift-transfer": (
                "B to C contemporaneous cross-site / terrain-regime transfer "
                "(secondary equation-transfer stress test; reports paired "
                "contemporaneous cross-site differences, never a causal "
                "terrain effect)"),
            "MC-A-to-C-joint-shift-transfer": (
                "A to C combined event+site external transfer (secondary "
                "equation-transfer stress test; event, site, terrain regime and "
                "local wind realization all change)"),
            "MC-C-independent-replication": (
                "Case C (Ebao 2015-12-01) independent external-site replication "
                "of the representation-level deficit and the geometric-recovery "
                "limitation"),
        },
        "comparisons": {
            "A_vs_B": ("same site and terrain, different atmospheric episode "
                       "(clean temporal replication; terrain and parent-cell "
                       "identity are fixed, so differences are attributable to "
                       "the event/wind-state change)"),
            "B_vs_C": ("contemporaneous cross-site / terrain-regime replication: "
                       "B and C sample the same chronological/synoptic time "
                       "window while differing in site, terrain regime and "
                       "potentially local wind realization. Forbidden wording: "
                       "'B->C isolates the terrain effect', 'atmospheric state "
                       "is identical', or any terrain-only causal attribution. "
                       "terrain_only_causal = false."),
            "A_vs_C": ("combined external stress test: event, site, terrain "
                       "regime and local wind realization all change; no "
                       "single-factor causal attribution"),
        },
        "bc_synchronized_bootstrap": True,
        "time_alignment_audit": True,
        "promote_with_partial_provenance": True,
        "regression_baselines": False,
        "case_c_freeze_schema": "v2",
        "thunder": "sixteen",
        "docs_hot_swap_section": False,
    },
}
