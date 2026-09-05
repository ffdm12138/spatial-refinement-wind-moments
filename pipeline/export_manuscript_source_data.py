"""Fail-closed exporter for manuscript candidate source data.

This module is deliberately small and reusable by Phase 16 and by future
table/figure/abstract/response-letter exporters.  A source-data artifact is
written only when every row is scientifically eligible, contains no
placeholder, and has a fully resolved REAL dependency set.  C-dependent rows
therefore remain available to the engineering dry-run but cannot cross into a
manuscript candidate directory while Case C is a placeholder or pending audit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import mc_config as C

REQUIRED_FIELDS = ("case_status", "science_eligible", "contains_placeholder",
                   "dependency_status")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _same_metadata(actual, expected, field: str) -> bool:
    if field in {"science_eligible", "contains_placeholder"}:
        return ((str(actual).strip().lower() in ("true", "1", "yes"))
                == bool(expected))
    return str(actual) == str(expected)


def assert_registry_metadata(df: pd.DataFrame, artifact: str) -> None:
    """Check row metadata against the live case/dependency registry.

    ``case`` rows are checked against one case.  Aggregate rows are checked
    from their canonical ``case_status``/``data_status`` dependency string.
    This makes a candidate both fail closed on placeholders and fail closed on
    stale metadata after a real Case-C promotion.
    """
    for index, row in df.iterrows():
        dependencies = []
        if "case" in df.columns:
            raw_case = str(row.get("case", ""))
            if raw_case in C.CASE_ORDER:
                dependencies = [raw_case]
        if not dependencies:
            for field in ("case_status", "data_status"):
                dependencies = C._cases_from_status_value(str(row.get(field, "")))
                if dependencies:
                    break
        if not dependencies:
            continue
        expected = (C.case_metadata(dependencies[0]) if len(dependencies) == 1
                    else C.status_metadata(dependencies))
        for field in REQUIRED_FIELDS:
            if not _same_metadata(row.get(field), expected[field], field):
                raise RuntimeError(
                    f"{artifact}: row {index} has stale {field}={row.get(field)!r}; "
                    f"registry expects {expected[field]!r} for {dependencies}")


def assert_manuscript_eligible(df: pd.DataFrame, artifact: str) -> None:
    """Reject a candidate artifact if any required gate is absent or false."""
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        raise RuntimeError(f"{artifact}: missing fail-closed fields {missing}")
    if len(df) == 0:
        raise RuntimeError(f"{artifact}: empty manuscript candidate")
    eligible = _bool_series(df["science_eligible"])
    placeholder = _bool_series(df["contains_placeholder"])
    dep = df["dependency_status"].astype(str)
    bad = (~eligible) | placeholder | (dep != "REAL")
    if bad.any():
        sample = df.loc[bad, [f for f in REQUIRED_FIELDS if f in df.columns]].head(3)
        raise RuntimeError(
            f"{artifact}: placeholder or unresolved dependency reached manuscript "
            f"export; offending rows:\n{sample.to_string(index=False)}")
    if df["case_status"].astype(str).str.contains(
            "PLACEHOLDER|PENDING|FAILED|NOT_RUN", case=False, regex=True).any():
        raise RuntimeError(f"{artifact}: unresolved case_status reached manuscript export")
    assert_registry_metadata(df, artifact)


def eligible_subset(df: pd.DataFrame, artifact: str) -> pd.DataFrame:
    """Return only eligible rows, while refusing malformed metadata."""
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        raise RuntimeError(f"{artifact}: missing fail-closed fields {missing}")
    mask = (_bool_series(df["science_eligible"]) &
            ~_bool_series(df["contains_placeholder"]) &
            (df["dependency_status"].astype(str) == "REAL"))
    out = df.loc[mask].copy()
    if len(out):
        assert_manuscript_eligible(out, artifact)
    return out


def write_eligible_csv(df: pd.DataFrame, path: str | Path,
                       artifact: str) -> bool:
    """Write a non-empty eligible subset; return False when all rows are blocked."""
    out = eligible_subset(df, artifact)
    if out.empty:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")
    assert_manuscript_eligible(pd.read_csv(path), artifact)
    return True


def audit_directory(root: str | Path) -> dict:
    """Audit an existing candidate directory without writing anything."""
    root = Path(root)
    checked, blocked, malformed = [], [], []
    for path in sorted(root.glob("**/*.csv")) if root.is_dir() else []:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # malformed CSV is a hard candidate failure
            malformed.append({"file": path.as_posix(), "error": str(exc)})
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
        if missing:
            malformed.append({"file": path.as_posix(), "missing": missing})
            continue
        try:
            subset = eligible_subset(df, str(path))
            assert_registry_metadata(df, str(path))
        except (RuntimeError, ValueError) as exc:
            malformed.append({"file": path.as_posix(), "error": str(exc)})
            continue
        if len(subset) == len(df):
            checked.append(path.as_posix())
        else:
            blocked.append(path.as_posix())
    forbidden = [p.as_posix() for pattern in ("**/*.pt", "**/*.npz")
                 for p in sorted(root.glob(pattern))] if root.is_dir() else []
    return {"root": str(root), "eligible_files": checked,
            "blocked_files": blocked, "malformed_files": malformed,
            "forbidden_binary_files": forbidden,
            "pass": not blocked and not malformed and not forbidden}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    args = ap.parse_args()
    print(json.dumps(audit_directory(args.root), indent=2, ensure_ascii=False))
