"""Capture compact pre-repair numeric baselines for the multicase regression audit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import mc_config as C


FILES = (
    "case_A/interpolation_metrics.csv",
    "case_A/interpolation_recovery.csv",
    "case_A/terrain_relationships.csv",
    "case_A/information_sources.csv",
    "case_A/scale_summary.csv",
    "case_B/interpolation_metrics.csv",
    "case_B/interpolation_recovery.csv",
    "case_B/terrain_relationships.csv",
    "case_B/information_sources.csv",
    "case_B/scale_summary.csv",
    "cross_case/core_loss_summary.csv",
    "cross_case/transfer_T1_A_to_B.csv",
    "cross_case/transfer_T2_history_A_to_B.csv",
    "tables/M2_representation_loss.csv",
    "tables/M5_geometric_recovery.csv",
    "tables/M6_terrain_relationships.csv",
    "tables/M8_transfer_A_to_B.csv",
)


def main() -> int:
    out = C.QA / "pre_repair_numeric_baseline_20260829.json"
    payload = {"captured_at_utc": __import__("time").strftime(
                   "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
               "files": {}}
    missing = []
    for rel in FILES:
        path = C.EXT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        payload["files"][rel] = pd.read_csv(path).to_dict(orient="records")
    if missing:
        raise FileNotFoundError(f"baseline source files missing: {missing}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=True) + "\n", encoding="utf-8")
    print(f"captured {len(payload['files'])} files -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
