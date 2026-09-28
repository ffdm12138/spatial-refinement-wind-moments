"""Shared helpers for Paper0 figure generation.

The plotting scripts deliberately read only the evidence bundle.  This module
keeps path resolution and provenance helpers in one place; it does not compute
or alter any scientific result.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
LATEX_DIR = SCRIPT_DIR.parent
ROOT_DIR = LATEX_DIR.parent
FIGURE_DIR = LATEX_DIR / "figures"
EVIDENCE_DIR = ROOT_DIR / "paper0_frozen_evidence"


def resolve_paths() -> tuple[Path, Path]:
    """Return the manuscript figure directory and frozen evidence directory."""
    if not EVIDENCE_DIR.is_dir():
        raise RuntimeError(f"Missing evidence directory: {EVIDENCE_DIR}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR, EVIDENCE_DIR


def sha256(path: Path) -> str:
    """Return the byte hash used by FIGURE_SOURCE_MAP.csv."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, str]:
    """Return a stable source record for a figure provenance sidecar/map."""
    try:
        relative = path.relative_to(EVIDENCE_DIR)
    except ValueError:
        relative = path.relative_to(ROOT_DIR)
    return {"input": str(relative).replace("\\", "/"), "hash": sha256(path)}


def require_columns(frame, columns: Iterable[str], source: Path) -> None:
    """Fail closed when an evidence table does not match its declared schema."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source.name}: missing required columns: {', '.join(missing)}")


def finite_rows(frame, columns: Iterable[str], source: Path) -> None:
    """Fail closed on malformed numeric values used by a figure."""
    import numpy as np

    cols = list(columns)
    require_columns(frame, cols, source)
    if frame.empty:
        raise ValueError(f"{source.name}: empty evidence table")
    values = frame[cols].apply(lambda col: col.astype(float))
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{source.name}: non-finite plotting value")

