#!/usr/bin/env python3
"""Refresh input hashes in both Paper0 figure source maps.

Run this only after the numerical-repair owner has promoted the final evidence
tables. It updates provenance hashes and never runs a plotter or changes any
evidence file. The two maps are kept byte-identical so the root verifier and
the manuscript workspace cannot silently select different source rows.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAPS = (ROOT / "FIGURE_SOURCE_MAP.csv", ROOT / "latex_paper0" / "FIGURE_SOURCE_MAP.csv", ROOT / "paper0_frozen_evidence" / "FIGURE_SOURCE_MAP.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(value: str) -> Path:
    candidates = [ROOT / "paper0_frozen_evidence" / value, ROOT / value]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(value)


def refresh(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        if not row["script"].startswith("latex_paper0/"):
            row["script"] = "latex_paper0/" + row["script"]
        if not (ROOT / row["script"]).is_file():
            raise FileNotFoundError(row["script"])
        if not row["input"].startswith("paper0_frozen_evidence/"):
            row["input"] = "paper0_frozen_evidence/" + row["input"]
        if "case_C" in row["panel"]:
            row["figure"] = "SI2"
        elif "terrain" in row["panel"] or "parent_ID" in row["panel"]:
            row["figure"] = "SI1"
        if not row["output"].startswith("latex_paper0/"):
            row["output"] = "latex_paper0/" + row["output"]
        if not (ROOT / row["output"]).is_file():
            raise FileNotFoundError(row["output"])
        composite = {"Fig1":"fig1_representation.pdf", "Fig2":"fig2_development.pdf",
                     "Fig3":"fig3_transfer.pdf", "Fig4":"fig4_ablations.pdf",
                     "SI1":"si_terrain_sensitivity.pdf", "SI2":"si_case_C_regimes.pdf", "SI3":"si_Qa_geometry.pdf"}
        row["manuscript_output"] = "latex_paper0/figures/" + composite[row["figure"]]
        row["script_hash"] = sha256(ROOT / row["script"])
        row["output_hash"] = sha256(ROOT / row["manuscript_output"])
        row["status"] = "CURRENT_V9_7_WITH_NUMERICAL_QUALIFICATIONS"
        row["numerical_scope"] = "7 development pair-metric signs unresolved; see current Results/Table 1; no A670/B/C flag" if row["figure"] in ("Fig2", "Fig3") else "See current caption; no formal numerical bound claimed"
        source = resolve_input(row["input"])
        row["hash"] = sha256(source)
        auxiliary = row.get("auxiliary_input", "").strip()
        if auxiliary:
            if not auxiliary.startswith("paper0_frozen_evidence/"):
                auxiliary = "paper0_frozen_evidence/" + auxiliary
                row["auxiliary_input"] = auxiliary
            row["auxiliary_hash"] = sha256(resolve_input(auxiliary))
        else:
            row["auxiliary_hash"] = ""
    return rows


def main() -> None:
    source_map = MAPS[0]
    with source_map.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "input" not in fieldnames or "hash" not in fieldnames:
            raise ValueError(f"Invalid figure source map: {source_map}")
        rows = refresh(list(reader))
    with source_map.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(dict.fromkeys((reader.fieldnames or []) + ["manuscript_output", "script_hash", "output_hash", "numerical_scope"]))
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    rendered = buffer.getvalue()
    for target in MAPS:
        target.write_text(rendered, encoding="utf-8", newline="")
    print(f"Refreshed {len(rows)} figure source rows in {len(MAPS)} maps")


if __name__ == "__main__":
    main()
