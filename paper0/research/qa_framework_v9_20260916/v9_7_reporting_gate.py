"""Source-derived reporting checks for the V9.7 numerical repair.

This module is a reader only.  ``v9_7_reports.py`` owns report/table writes;
these helpers read the promoted CSVs and check that their point estimates and
interval endpoints reach the manuscript.  No section heading, old count, or
historical sentence is an oracle.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def ci_relation(sl: float, sh: float, jl: float, jh: float) -> str:
    """Classify two metric intervals using their actual endpoints."""
    if not all(_finite(value) for value in (sl, sh, jl, jh)) or sl > sh or jl > jh:
        raise ValueError("invalid confidence interval")
    speed_positive, speed_negative = sl > 0, sh < 0
    gap_positive, gap_negative = jl > 0, jh < 0
    speed_zero, gap_zero = sl <= 0 <= sh, jl <= 0 <= jh
    if speed_positive and gap_positive:
        return "QA_BOTH_CI_SUPPORTED"
    if speed_negative and gap_negative:
        return "JS_BOTH_CI_SUPPORTED"
    if (speed_positive and gap_zero) or (gap_positive and speed_zero):
        return "ONE_CI_POSITIVE_OTHER_SPANS_ZERO"
    if (speed_negative and gap_zero) or (gap_negative and speed_zero):
        return "JS_ONE_CI_NEGATIVE_OTHER_SPANS_ZERO"
    if (speed_positive and gap_negative) or (speed_negative and gap_positive):
        return "TRADE_OFF"
    return "INDISTINGUISHABLE"


def point_relation(speed: float, gap: float) -> str:
    """Classify point estimates, preserving opposite-metric rows."""
    if not _finite(speed) or not _finite(gap):
        raise ValueError("point estimate is not finite")
    if speed > 0 and gap > 0:
        return "QA_STRICT_POINT_DOMINANCE"
    if speed < 0 and gap < 0:
        return "JS_STRICT_POINT_DOMINANCE"
    if (speed > 0 and gap < 0) or (speed < 0 and gap > 0):
        return "TRADE_OFF"
    return "INDISTINGUISHABLE"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number_variants(value: float) -> set[str]:
    variants: set[str] = set()
    for places in (3, 4, 5, 6, 7, 8):
        fixed = f"{value:.{places}f}"
        variants.add(fixed)
        trimmed = fixed.rstrip("0").rstrip(".") or "0"
        variants.add(trimmed)
        if not trimmed.startswith("-") and trimmed != "0":
            variants.add("+" + trimmed)
    return variants


def _contains_number(text: str, value: float) -> bool:
    for token in _number_variants(float(value)):
        if re.search(rf"(?<![0-9.+-]){re.escape(token)}(?![0-9.])", text):
            return True
    return False


def _point_value(row: Mapping[str, str], metric: str) -> float:
    for field in (f"delta_{metric}_point", f"delta_{metric}_mean", f"delta_{metric}"):
        if field in row and str(row[field]).strip():
            return float(row[field])
    raise KeyError(f"no point field for {metric}")


def _interval_values(row: Mapping[str, str], metric: str) -> tuple[float, float]:
    return float(row[f"delta_{metric}_ci_low"]), float(row[f"delta_{metric}_ci_high"])


def _selected(rows: Iterable[Mapping[str, str]], **criteria: str) -> list[Mapping[str, str]]:
    return [row for row in rows if all(str(row.get(key, "")) == value for key, value in criteria.items())]


def classify_rows(evidence: str | Path) -> dict[str, Any]:
    """Return source-derived point/CI counts and adverse-row evidence.

    The result is diagnostic and has no fixed expected win count.  Callers can
    require the presence of both signs, interval overlap, and an opposite
    metric while retaining the actual classification counts in the receipt.
    """
    evidence = Path(evidence)
    point_rows = _read_rows(evidence / "QA_VS_JS_FINAL_A.csv")
    bootstrap_rows = _read_rows(evidence / "QA_VS_JS_FINAL_BOOTSTRAP.csv")
    classified_rows = _read_rows(evidence / "QA_VS_JS_FINAL_CLASSIFIED.csv")
    point_by_key = {(row.get("level", ""), row.get("fold", "")): row for row in point_rows}
    boot_by_key = {
        (row.get("level", ""), row.get("fold", ""), row.get("candidate", "")): row
        for row in bootstrap_rows
    }
    class_by_key = {(row.get("level", ""), row.get("fold", "")): row for row in classified_rows}
    errors: list[str] = []
    try:
        import importlib.util
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("resolution_metadata", root/"scripts/numerical_resolution_metadata.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.verify(root)
        for row in classified_rows:
            if any(row.get(k)!=v for k,v in module.expected(root,row).items()):
                errors.append("classification numerical status missing or inconsistent")
    except (OSError,ValueError,KeyError) as exc:
        errors.append(f"numerical resolution metadata: {exc}")
    point_labels: list[str] = []
    ci_labels: list[str] = []
    positive_cells = negative_cells = spanning_cells = opposite_rows = 0
    for key, source in point_by_key.items():
        classified = class_by_key.get(key)
        if classified is None:
            errors.append(f"missing classified row {key}")
            continue
        for candidate in ("QA_dyn", "QA_const"):
            speed = float(source[f"{candidate}_delta_s"])
            gap = float(source[f"{candidate}_delta_J"])
            relation = point_relation(speed, gap)
            if classified.get(f"{candidate}_point_relation") != relation:
                errors.append(f"{key}/{candidate}: point relation mismatch")
            point_labels.append(relation)
            positive_cells += int(speed > 0) + int(gap > 0)
            negative_cells += int(speed < 0) + int(gap < 0)
            opposite_rows += int(speed * gap < 0)
            boot = boot_by_key.get((*key, candidate))
            if boot is None:
                errors.append(f"missing bootstrap row {key}/{candidate}")
                continue
            sl, sh = _interval_values(boot, "speed")
            jl, jh = _interval_values(boot, "J")
            relation_ci = ci_relation(sl, sh, jl, jh)
            if classified.get(f"{candidate}_ci_relation") != relation_ci:
                errors.append(f"{key}/{candidate}: CI relation mismatch")
            ci_labels.append(relation_ci)
            spanning_cells += int(sl <= 0 <= sh) + int(jl <= 0 <= jh)
    js_rows = _read_rows(evidence / "JS_UNCERTAINTY_ABLATION.csv")
    for row in js_rows:
        # Only the J marginal is required to be identical.  Correlation can
        # affect the joint speed score, so equality of speed is not an
        # acceptance condition and must not be smuggled in as one.
        metric = "J"
        ind, joint = float(row[f"crps_{metric}_JS_ind"]), float(row[f"crps_{metric}_JS_joint"])
        if not (_finite(ind) and _finite(joint)) or abs(ind - joint) > 1e-12:
            errors.append(f"{row.get('level')}/{row.get('fold')}: JS_ind/joint {metric} marginal differs")
    return {
        "point_relation": {label: point_labels.count(label) for label in sorted(set(point_labels))},
        "ci_relation": {label: ci_labels.count(label) for label in sorted(set(ci_labels))},
        "positive_point_cells": positive_cells,
        "negative_point_cells": negative_cells,
        "intervals_spanning_zero": spanning_cells,
        "opposite_metric_rows": opposite_rows,
        "js_marginal_rows": len(js_rows),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def manuscript_checks(evidence: str | Path, results: str, table: str) -> list[str]:
    """Check promoted point/CI values in generated table and results text.

    This intentionally does not parse section titles.  A manuscript rewrite
    can reorder headings while every source number remains traceable.
    """
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("numeric_binding", root / "scripts/manuscript_numeric_binding.py")
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    return binding.manuscript_checks(evidence, results, table)



def re_table_line(line: str) -> bool:
    """Compatibility helper retained for the legacy verifier."""
    return " & " in line and bool(re.search(r"(?:fold1|fold2|fold3|S[1-4])", line))
