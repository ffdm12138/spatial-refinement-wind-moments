"""Shared publication style for the manuscript figure suite.

Style conventions are ported from the manuscript's companion figure style's
``scripts/benchmark_pipeline/style.py`` so that all figures render as
one visual system: Times New Roman at ~8.4 pt, in-axes white-backed panel
labels, clean spines, and a uniform colorbar helper.  The project colour
palette and the four-format output (PDF/PNG/600-dpi TIFF + EPS via the
caller's save wrapper; SVG dropped in the 2026-08-06 cleanup) are preserved.

2026-08-06 final-story round: two rc presets. Main figures are drawn on a
canvas of exactly 170 mm (the main-text width, margin=20 mm on A4) so that
``\\includegraphics[width=\\textwidth]`` renders 1:1 and every font >= 8 pt.
Supplement figures keep their 174-mm canvas (legacy supplement textwidth,
margin=18 mm) but
ticks/legends were raised from 7.7/7.6 to 8.0 pt after measuring the final
insertion size (measured 7.7 pt < 8 pt at 1:1). All alpha transparency was
removed from shared styling so that EPS output carries no PostScript
transparency and PDF/TIFF/EPS stay visibly identical.

The path constants below resolve relative to this module's parent directory
(the package root). In a packaged copy of the project the evidence tables are
consolidated under ``source_data``, so the evidence paths fall back to that
folder whenever it is present; otherwise they point at the individual
evidence subfolders of the working copy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# Use Times New Roman when installed; fall back to the metrically similar,
# redistributable Liberation Serif on clean Linux builds. The selected font is
# embedded in vector outputs; no font file is included in the package.
_AVAILABLE_FONTS = {f.name for f in font_manager.fontManager.ttflist}
PUBLICATION_FONT = "Times New Roman" if "Times New Roman" in _AVAILABLE_FONTS else "Liberation Serif"

# Publication palette (stable across the figure suite).
BLUE = "#2c5d87"
TEAL = "#2a9d8f"
ORANGE = "#e07a32"
RED = "#b23a48"
INK = "#23313f"
GREY = "#6b7280"
LIGHT = "#e7edf2"

# Main-text textwidth (A4, margin=20 mm) and legacy supplement textwidth
# (margin=18 mm).
MAIN_TEXTWIDTH_MM = 170.0
MAIN_WIDTH_IN = MAIN_TEXTWIDTH_MM / 25.4  # 6.6929 in -> 1:1 at \textwidth
ESM_TEXTWIDTH_MM = 174.0
ESM_WIDTH_IN = ESM_TEXTWIDTH_MM / 25.4  # 6.8504 in

# Repository paths (see module docstring for the packaged-copy fallback).
ROOT = Path(__file__).resolve().parents[1]
IS_RELEASE = (ROOT / "source_data").is_dir()
DATA = ROOT / "figure_data" / "main" if (ROOT / "figure_data" / "main").is_dir() else ROOT / "figure_data"
EVID = ROOT / "evidence" / "refactor_20260807" if not IS_RELEASE else ROOT / "source_data"
SNAP = ROOT / "evidence" / "source_snapshots" if not IS_RELEASE else ROOT / "source_data"
NPJ = ROOT / "evidence" / "npj_refactor" if not IS_RELEASE else ROOT / "source_data"


def _base_rc() -> dict:
    return {
        "font.family": [PUBLICATION_FONT],
        "font.serif": [PUBLICATION_FONT],
        "font.sans-serif": [PUBLICATION_FONT],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,  # embed TrueType outlines, never Type 3
        "ps.fonttype": 42,
        "font.size": 8.4,
        "axes.labelsize": 8.4,
        "axes.linewidth": 0.65,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
        "lines.linewidth": 1.15,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


def set_main_rc() -> None:
    """rc for Fig. 3-6: 170-mm canvas, minimum final font >= 8 pt at 1:1."""

    rc = _base_rc()
    rc.update(
        {
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )
    mpl.rcParams.update(rc)


def set_supplement_rc() -> None:
    """rc for FigS1-S7: 174-mm canvas; tick/legend raised 7.7/7.6 -> 8.0 pt
    (measured 7.7 pt at the final 1:1 insertion size, below the 8-pt floor)."""

    rc = _base_rc()
    rc.update(
        {
            "axes.titlesize": 8.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )
    mpl.rcParams.update(rc)


def panel_label(ax: plt.Axes, label: str, *, x: float = 0.015, y: float = 0.985, ha: str = "left", va: str = "top") -> None:
    """In-axes bold panel label with an opaque white backing box (no alpha:
    EPS must not carry PostScript transparency)."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9.0,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.2},
        zorder=20,
    )


def clean_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color="#D9D9D9", linewidth=0.5, zorder=0)


def add_shared_colorbar(fig, mappable, axes, label: str, *, fraction: float = 0.030, pad: float = 0.018, **kwargs):
    cbar = fig.colorbar(mappable, ax=axes, fraction=fraction, pad=pad, **kwargs)
    cbar.set_label(label)
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(width=0.5, length=2.2)
    return cbar


def save_figure(
    fig: plt.Figure,
    stem: str,
    subdir: str = "main",
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0.025,
) -> None:
    """Save PDF, 320-dpi preview PNG and 600-dpi LZW TIFF for one figure (SVG dropped in the 2026-08-06 cleanup)."""

    out_dir = Path(__file__).resolve().parents[1] / "figures" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(
        out_dir / f"{stem}.tiff",
        dpi=600,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def read_evid(name: str) -> pd.DataFrame:
    return pd.read_csv(EVID / name)


def save(fig: plt.Figure, stem: str, subdir: str = "main", *, bbox_inches: str = "tight") -> None:
    save_figure(fig, stem, subdir, bbox_inches=bbox_inches)
    # submission requires EPS for the main figures; matplotlib emits vector EPS directly
    out_dir = ROOT / "figures" / subdir
    fig.savefig(out_dir / f"{stem}.eps", bbox_inches=bbox_inches, pad_inches=0.025)


def tint(hex_color: str, frac: float = 0.8) -> str:
    """Opaque light tint of a palette colour (no alpha: EPS-safe)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac))


def spatial_map(ax, values6x6, title, cmap, vmin, vmax, label, annot="extrema",
                cbar_fraction=0.046, cbar_pad=0.02):
    """6x6 parent-cell heat map; annot='extrema' marks only the largest and
    smallest cells, annot=None removes all value text."""
    m = np.asarray(values6x6, float).reshape(6, 6)
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.tick_params(length=0)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title, pad=3.5)
    for r in range(6):
        for c in range(6):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="white", lw=0.8, zorder=3))
    if annot == "extrema":
        flat = m.ravel()
        for idx in (int(np.argmin(flat)), int(np.argmax(flat))):
            r, c = divmod(idx, 6)
            v = m[r, c]
            # 8-pt "-0.04" is wider than one cell; keep edge-column values
            # inside the map and give every value a contrasting outline
            dx = 0.30 if c == 0 else (-0.30 if c == 5 else 0.0)
            tcol = "white" if abs(v) > 0.5 * max(abs(vmin), abs(vmax), 1e-9) else INK
            ax.text(c + dx, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.0, color=tcol, zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.2,
                                                foreground=INK if tcol == "white" else "white")])
    add_shared_colorbar(ax.figure, im, ax, label,
                        fraction=cbar_fraction, pad=cbar_pad)
    return im
