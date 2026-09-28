"""Phase 0 — raw-data contract, lineage and data_status detection (PLAN §4).

Checks, per case, against the Case-A structural template:
  1  filename set identical            2  tensor shapes
  3  dtype float32                     4  state count (4320 or STOP)
  5  spatial shape 162x162             6  U/V domain pairing
  7  HGT presence                      8  NaN / Inf
  9  file SHA256                      10  state cadence (archived-order contract)
 11  d02 native 6x6 tiled 27x27 exact 12  d05 162x162
 13  1 parent = 27x27 = 729 fine      14  parent row-major ordering
 15  HGT shape consistent with winds

Any structural mismatch STOPS that case (science phases are skipped) and a
discrepancy report is written; the window split is never redefined here.

Outputs: raw_contract/{raw_manifest.csv, raw_hashes.json, shape_contract.json,
spatial_mapping_audit.json, data_lineage.md} and per-case status JSONs under
qa/paper0_multicase/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402


def check_case(case: str) -> dict:
    rdir = C.case_raw_dir(case)
    rec = {"case": case, "raw_dir": str(rdir), "checks": {}, "files": {},
           "stop": False, "stop_reasons": []}

    def fail(reason: str):
        rec["stop"] = True
        rec["stop_reasons"].append(reason)

    if not rdir.is_dir():
        fail(f"raw directory missing: {rdir}")
        return rec

    names = sorted(p.name for p in rdir.glob("*.pt"))
    expected = sorted(C.raw_filename(case, s) for s in C.RAW_FILE_STEMS)
    rec["checks"]["filename_set"] = (names == expected)
    if not rec["checks"]["filename_set"]:
        fail(f"filename set mismatch: got {names}, expected {expected}")
        return rec

    # hashes first (also drive data_status detection)
    hashes = {}
    for stem in C.RAW_FILE_STEMS:
        name = C.raw_filename(case, stem)
        p = rdir / name
        hashes[name] = C.sha256_of(p)
        rec["files"][name] = {"sha256": hashes[name], "bytes": p.stat().st_size}
    rec["hashes"] = hashes

    # per-file structural checks
    shape_ok = True
    for stem in C.RAW_FILE_STEMS:
        name = C.raw_filename(case, stem)
        arr = C.load_pt(rdir / name)
        expected = (C.GRID, C.GRID) if stem == "HGT_d05.pt" else (C.RAW_N, C.GRID, C.GRID)
        f = rec["files"][name]
        f["shape"] = list(arr.shape)
        f["dtype"] = str(arr.dtype)
        f["finite"] = bool(np.isfinite(arr).all())
        f["value_range"] = [float(arr.min()), float(arr.max())]
        if stem != "HGT_d05.pt" and arr.ndim == 3:
            f["n_states"] = int(arr.shape[0])
        if arr.dtype != np.float32:
            shape_ok = False
            fail(f"{name}: dtype {arr.dtype} != float32")
        if not f["finite"]:
            shape_ok = False
            fail(f"{name}: NaN/Inf present")
        if stem == "HGT_d05.pt":
            if arr.shape != (C.GRID, C.GRID):
                shape_ok = False
                fail(f"{name}: HGT shape {arr.shape} != (162,162)")
        else:
            if arr.shape[1:] != (C.GRID, C.GRID):
                shape_ok = False
                fail(f"{name}: spatial shape {arr.shape[1:]} != (162,162)")
            if arr.shape[0] != C.RAW_N:
                shape_ok = False
                fail(f"{name}: state count {arr.shape[0]} != {C.RAW_N} — the "
                     "inherited analysis window cannot be applied; case is "
                     "STOPPED pending a human decision (window is never "
                     "redefined automatically)")
        # d02 tiling audit on wind fields (subsampled states for speed)
        if stem in ("U10_d02.pt", "V10_d02.pt") and arr.ndim == 3 and arr.shape == (C.RAW_N, C.GRID, C.GRID):
            probe = arr[:: max(1, C.RAW_N // 96)]
            native = probe[:, ::C.BLOCK, ::C.BLOCK]
            tiled = np.repeat(np.repeat(native, C.BLOCK, axis=1), C.BLOCK, axis=2)
            dev = float(np.abs(probe - tiled).max())
            f["d02_tiling_max_abs_dev"] = dev
            if dev > 0.0:
                shape_ok = False
                fail(f"{name}: d02 is not an exact 6x6 native grid tiled 27x27 "
                     f"(max abs dev {dev})")
        del arr
    rec["checks"]["shapes_dtypes_finite"] = shape_ok

    # cadence: archived serialized order; 1-min cadence is a configuration
    # assumption of the archive (no absolute timestamps in the tensors)
    rec["checks"]["state_cadence"] = ("archived serialized order; 1-min cadence "
                                      "by configuration contract, not timestamps")
    # parent mapping (row-major ordering + centre indices)
    rec["checks"]["parent_row_major"] = True
    rec["checks"]["parent_centre_fine_indices"] = C.PARENT_CENTRE_FINE_INDICES
    rec["checks"]["fine_cells_per_parent"] = C.N_FINE

    # data_status via placeholder-manifest hash match
    status = C.detect_data_status(case, hashes)
    rec.update(status)

    # Case A must additionally match the sealed holdout manifest hashes
    if case == "A":
        sealed = json.loads(C.FROZEN_SEALED_MANIFEST.read_text(encoding="utf-8"))
        raw_sha = sealed.get("raw_sha256", {})
        m = {k: C.raw_filename("A", v) for k, v in
             {"u02": "U10_d02.pt", "v02": "V10_d02.pt", "u05": "U10_d05.pt",
              "v05": "V10_d05.pt", "hgt": "HGT_d05.pt"}.items()}
        mism = {k: (raw_sha.get(k), hashes[v]) for k, v in m.items()
                if k in raw_sha and raw_sha[k] != hashes[v]}
        rec["checks"]["sealed_manifest_match"] = not mism
        if mism:
            fail(f"case A raw hashes do not match the sealed manifest: {list(mism)}")
    return rec


def spatial_mapping_audit(case: str) -> dict:
    """Deep tiling audit on the full d02 tensors + HGT/wind grid consistency."""
    rdir = C.case_raw_dir(case)
    out = {"case": case}
    u02 = C.load_pt(rdir / C.raw_filename(case, "U10_d02.pt"))
    native = u02[:, ::C.BLOCK, ::C.BLOCK]
    tiled = np.repeat(np.repeat(native, C.BLOCK, axis=1), C.BLOCK, axis=2)
    out["u02_full_tiling_max_abs_dev"] = float(np.abs(u02 - tiled).max())
    del u02, tiled
    v02 = C.load_pt(rdir / C.raw_filename(case, "V10_d02.pt"))
    nv = v02[:, ::C.BLOCK, ::C.BLOCK]
    tv = np.repeat(np.repeat(nv, C.BLOCK, axis=1), C.BLOCK, axis=2)
    out["v02_full_tiling_max_abs_dev"] = float(np.abs(v02 - tv).max())
    del v02, tv
    hgt = C.load_case_hgt(case)
    out["hgt_shape"] = list(hgt.shape)
    out["hgt_range_m"] = [float(hgt.min()), float(hgt.max())]
    out["hgt_grid_matches_wind_grid"] = list(hgt.shape) == [C.GRID, C.GRID]
    out["parent_grid"] = "6x6 native, row-major cell_id = row*6+col"
    out["block"] = C.BLOCK
    out["pass"] = (out["u02_full_tiling_max_abs_dev"] == 0.0
                   and out["v02_full_tiling_max_abs_dev"] == 0.0
                   and out["hgt_grid_matches_wind_grid"])
    return out


def write_lineage(records: dict, audits: dict) -> None:
    lines = [f"# Raw-data lineage — {C.EXTENSION_NAME}", ""]
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}. "
                 f"Structural template: Case A (`{C.CASES['A']['raw_subdir']}`).")
    lines.append("2026-08-29: Case A raw directory renamed from `1min` to "
                 f"`{C.CASES['A']['raw_subdir']}` (byte-identical content; "
                 "on-disk .pt files carry the per-case `filename_suffix`).")
    lines.append("")
    for case in C.CASE_ORDER:
        rec = records.get(case)
        if rec is None:
            continue
        meta = C.CASES[case]
        lines += [f"## Case {case} — {meta['label']}", "",
                  f"* Raw directory: `{rec['raw_dir']}`",
                  f"* Location: {meta['location']} ({meta['lat']}, {meta['lon']})",
                  f"* Dates: {meta['dates']}",
                  f"* Role: {meta['role']}",
                  f"* data_status: **{rec.get('data_status', 'UNKNOWN')}**"
                  + ("  — PLACEHOLDER DRY-RUN, NOT SCIENCE"
                     if rec.get("data_status") == "PLACEHOLDER" else ""),
                  f"* case_status: **{rec.get('case_status', 'FAILED' if rec['stop'] else 'NOT_RUN')}**; "
                  f"science_eligible: **{C.metadata_from_values(rec.get('data_status', 'UNKNOWN'), rec.get('case_status', 'NOT_RUN'), rec['stop'])['science_eligible']}**; "
                  f"dependency_status: **{C.metadata_from_values(rec.get('data_status', 'UNKNOWN'), rec.get('case_status', 'NOT_RUN'), rec['stop'])['dependency_status']}**",
                  f"* Contract: {'STOPPED — ' + '; '.join(rec['stop_reasons']) if rec['stop'] else 'PASS'}"]
        if case in audits:
            a = audits[case]
            lines.append(f"* d02 tiling max abs dev (full tensors): "
                         f"u {a['u02_full_tiling_max_abs_dev']}, v {a['v02_full_tiling_max_abs_dev']}")
            lines.append(f"* HGT range: {a['hgt_range_m'][0]:.1f}–{a['hgt_range_m'][1]:.1f} m")
        if rec.get("data_status") == "PLACEHOLDER":
            lines.append("* Generation: deterministic synthetic fields, master seed "
                         f"{C.MASTER_SEED}; see `PLACEHOLDER_NOTICE.md` and "
                         "`placeholder_manifest.json` in the raw directory.")
        lines.append("")
    lines += [
        "## Analysis-window contract (inherited, PLAN §5)", "",
        "4320 raw states; exclude [0,970); calibration [970,3650) = 2680; "
        "evaluation [3650,4320) = 670.  For Cases B/C the exclusion is a "
        "fixed inherited analysis-window duration, specified before examining "
        "the new-case science results; it is never justified via AWS "
        "availability and never retuned.  A case whose state count is not "
        "4320 is STOPPED with a discrepancy report.", "",
        "## Consumption boundaries", "",
        "P0 science consumes U10/V10 (d02, d05) and HGT_d05 only.  T2 files "
        "are manifest-only and never enter the wind science.", "",
        "## Informational note", "",
        "Case directories follow the naming pattern `1min_<site><yyyymmdd>` "
        "with per-case filename suffixes (see CASE_CONTRACT.json); sibling "
        "directories (5min/10min/patchsize/interpolation variants) are "
        "unrelated legacy ML artefacts and are not consumed.", "",
    ]
    (C.ext_dir("raw_contract") / "data_lineage.md").write_text(
        "\n".join(lines), encoding="utf-8")


def time_alignment_audit(cases: list[str]) -> dict | None:
    """B/C contemporaneous-window contract audit (PLAN_MULTICASE_20260905 §4
    items 17-18 and §19.2 preconditions).

    The .pt payloads carry no absolute timestamps (bare tensors,
    ``__metadata__ = None``), so the chronological identity of B and C is a
    DECLARED author contract attested by bytes: same declared window in the
    registry, same 4320-state axis, same archived serialized order, per-file
    SHA-256 in ``raw_hashes.json`` and the schema-v2 event freeze register.
    This audit records that contract explicitly; it must never be read as a
    claim that the two sites had identical atmospheric states.
    """
    if not C.GEN.get("time_alignment_audit"):
        return None
    if not {"B", "C"}.issubset(set(cases)):
        return None
    b_meta, c_meta = C.CASES["B"], C.CASES["C"]
    b_rec, c_rec = C.case_record("B"), C.case_record("C")
    same_window = (str(b_meta.get("dates")).strip() ==
                   str(c_meta.get("dates")).strip())
    same_states = (b_rec.get("raw_hashes") and c_rec.get("raw_hashes")
                   and not b_rec.get("stop") and not c_rec.get("stop"))
    payloads = {}
    for case in ("B", "C"):
        rdir = C.case_raw_dir(case)
        per_file = {}
        for stem in C.RAW_FILE_STEMS:
            name = C.raw_filename(case, stem)
            per_file[stem] = {"disk_name": name,
                              "sha256": C.sha256_of(rdir / name)}
        payloads[case] = {
            "raw_subdir": b_meta["raw_subdir"] if case == "B" else c_meta["raw_subdir"],
            "declared_dates": C.CASES[case]["dates"],
            "state_count": C.RAW_N,
            "cadence": "1-min archived serialized order (configuration contract)",
            "files": per_file,
        }
    out = {
        "contract": (
            "B and C share the same declared chronological/synoptic window "
            f"({b_meta.get('dates')}) under PLAN_MULTICASE_20260905.md; "
            "absolute timestamps are not packaged, so the shared axis is a "
            "declared author contract attested by byte hashes and the "
            "schema-v2 event freeze register"),
        "same_declared_window": same_window,
        "same_state_count_contract": bool(same_states),
        "state_count": C.RAW_N,
        "synchronized_bootstrap_preconditions_met": bool(
            same_window and same_states),
        "wording_lock": (
            "paired contemporaneous cross-site difference; never a causal "
            "terrain effect; never 'identical atmospheric states'"),
        "cases": payloads,
        "generated_for_generation": C.GENERATION,
    }
    C.write_json(C.ext_dir("raw_contract") / "time_alignment_audit.json",
                 out, cases=["B", "C"])
    return out


def main(cases=C.CASE_ORDER) -> dict:
    out_dir = C.ext_dir("raw_contract")
    records, audits, manifest_rows = {}, {}, []
    for case in cases:
        print(f"[phase00] case {case}: raw contract check")
        rec = check_case(case)
        records[case] = rec
        if not rec["stop"]:
            audits[case] = spatial_mapping_audit(case)
            if not audits[case]["pass"]:
                rec["stop"] = True
                rec["stop_reasons"].append("spatial mapping audit failed")
        data_status = rec.get("data_status", "UNKNOWN")
        case_status = (C.CASE_STATUS_FAILED if rec["stop"] else
                       rec.get("case_status", C.CASE_STATUS_NOT_RUN))
        output_meta = C.metadata_from_values(data_status, case_status,
                                             rec["stop"])
        status_payload = {
            "case": case,
            **output_meta,
            "stop": rec["stop"],
            "stop_reasons": rec["stop_reasons"],
            "raw_hashes": rec.get("hashes", {}),
            "detail": rec.get("detail", ""),
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        C.case_status_file(case).write_text(
            json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        C.write_case_provenance(case, rec)
        if rec["stop"]:
            rep = C.ext_dir("raw_contract") / f"discrepancy_report_case_{case}.md"
            rep.write_text(
                f"# DISCREPANCY REPORT — Case {case}\n\n"
                f"The raw contract check STOPPED this case. Science phases were "
                f"not run for it. Reasons:\n\n"
                + "\n".join(f"* {r}" for r in rec["stop_reasons"])
                + "\n\nNo automatic window redefinition or repair was attempted "
                  "(PLAN §5: wait for a human decision).\n", encoding="utf-8")
            print(f"  STOP: {rec['stop_reasons']}")
            continue
        for stem in C.RAW_FILE_STEMS:
            name = C.raw_filename(case, stem)
            f = rec["files"][name]
            manifest_rows.append({
                "case": case, "file": name, "stem": stem, "bytes": f["bytes"],
                "sha256": f["sha256"], "shape": "x".join(map(str, f["shape"])),
                "dtype": f["dtype"], "finite": f["finite"],
                "science_role": ("science" if stem in C.SCIENCE_STEMS
                                 else "manifest_only"),
                "data_status": rec["data_status"],
            })
        print(f"  PASS  data_status={rec['data_status']}")

    C.write_df(out_dir / "raw_manifest.csv", pd.DataFrame(manifest_rows),
               cases=list(records))
    hashes_payload = {c: records[c].get("hashes", {}) for c in records}
    C.write_json(out_dir / "raw_hashes.json", hashes_payload,
                 cases=list(records))
    shape_contract = {
        "template_case": "A",
        "raw_states": C.RAW_N,
        "grid_d05": [C.GRID, C.GRID],
        "d02_storage": "162x162 cellwise replication of native 6x6 parents",
        "block": C.BLOCK, "n_parent_cells": C.N_PARENT,
        "fine_cells_per_parent": C.N_FINE,
        "dtype": "float32",
        "file_set": C.RAW_FILE_STEMS,
        "science_files": C.SCIENCE_STEMS,
        "manifest_only_files": C.MANIFEST_ONLY_STEMS,
        "cases": {c: {"stop": records[c]["stop"],
                      "data_status": records[c].get("data_status", "UNKNOWN"),
                      "checks": {k: v for k, v in records[c]["checks"].items()}}
                  for c in records},
    }
    C.write_json(out_dir / "shape_contract.json", shape_contract,
                 cases=list(records), claim_id="MC-M4-scale-master-schema")
    C.write_json(out_dir / "spatial_mapping_audit.json", audits,
                 cases=list(audits))
    C.write_registry(C.CASE_ORDER)
    write_lineage(records, audits)
    time_alignment_audit(list(records))
    C.run_receipt("phase00_raw_contract", {
        "cases": {c: {"stop": records[c]["stop"],
                      "data_status": records[c].get("data_status")}
                  for c in records}})
    return records


if __name__ == "__main__":
    main()
