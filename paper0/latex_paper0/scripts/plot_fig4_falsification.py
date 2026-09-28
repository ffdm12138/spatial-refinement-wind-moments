#!/usr/bin/env python3
"""Generate supplementary terrain rank-sensitivity figures.

Terrain derangement is kept out of the Article main figures. These plots show
the specified within-test perturbation and parent-ID comparisons as sensitivity
evidence; labels avoid causal or p-value language.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import finite_rows, require_columns, resolve_paths


def _enforce_composite_font_floor(fig, minimum: float = 8.5) -> None:
    """Raise composite-only text to a readable article-width minimum."""
    fig.canvas.draw()
    for artist in fig.findobj():
        if hasattr(artist, "get_fontsize") and hasattr(artist, "set_fontsize"):
            if artist.get_fontsize() < minimum:
                artist.set_fontsize(minimum)


FOLD_ORDER = [
    ("A_T", "fold1"),
    ("A_T", "fold2"),
    ("A_T", "fold3"),
    ("A_S", "S1"),
    ("A_S", "S2"),
    ("A_S", "S3"),
    ("A_S", "S4"),
    ("A_ST", "S1"),
    ("A_ST", "S2"),
    ("A_ST", "S3"),
    ("A_ST", "S4"),
]


def load_terrain_data(ev_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    perm_path = ev_dir / "TERRAIN_WITHIN_TEST_PERM_999.csv"
    attribution_path = ev_dir / "TERRAIN_ATTRIBUTION_FINAL.csv"
    perm = pd.read_csv(perm_path)
    attribution = pd.read_csv(attribution_path)
    require_columns(
        perm,
        [
            "level",
            "fold",
            "permutation_index",
            "seed",
            "mapping_sha256",
            "speed_degradation_vs_intact",
            "J_degradation_vs_intact",
        ],
        perm_path,
    )
    require_columns(
        attribution,
        [
            "level",
            "fold",
            "Intact_Terrain",
            "Parent_ID_Memorization",
            "Parent_ID_Plus_Clim",
            "No_Terrain_Dynamic_Only",
        ],
        attribution_path,
    )
    finite_rows(perm, ["speed_degradation_vs_intact", "J_degradation_vs_intact"], perm_path)
    finite_rows(
        attribution,
        ["Intact_Terrain", "Parent_ID_Memorization", "Parent_ID_Plus_Clim", "No_Terrain_Dynamic_Only"],
        attribution_path,
    )
    return perm, attribution


def _ordered_labels(frame: pd.DataFrame) -> list[str]:
    return [f"{level.replace('A_', '')}:{fold}" for level, fold in FOLD_ORDER if ((frame["level"] == level) & (frame["fold"] == fold)).any()]


def draw_permutation(ax, perm: pd.DataFrame, column: str, title: str, color: str, show_note: bool = True) -> None:
    groups = []
    labels = []
    for level, fold in FOLD_ORDER:
        values = perm[(perm["level"] == level) & (perm["fold"] == fold)][column].to_numpy()
        if len(values) == 0:
            continue
        groups.append(values)
        labels.append(f"{level.replace('A_', '')}:{fold}")
    if not groups:
        raise ValueError(f"No permutation draws found for {column}")
    ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True, boxprops={"facecolor": color, "alpha": 0.45}, medianprops={"color": "black"}, whiskerprops={"color": color}, capprops={"color": color})
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Perturbed − intact\nCRPS [m/s]", fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    if show_note:
        ax.text(0.02, 0.03, "specified within-test perturbation; not a causal or p-value test", transform=ax.transAxes, fontsize=7, color="dimgray")
    ax.grid(True, axis="y", alpha=0.25)


def draw_parent_id(ax, attribution: pd.DataFrame, legend_loc: str = "best", legend_bbox=None, legend_ncol: int = 1, show_note: bool = True) -> None:
    frame = attribution.set_index(["level", "fold"]).reindex(FOLD_ORDER).dropna().reset_index()
    x = np.arange(len(frame))
    intact = frame["Intact_Terrain"].to_numpy()
    for offset, column, color, marker, label in [
        (-0.18, "Parent_ID_Memorization", "steelblue", "o", "parent-ID memorization"),
        (0.0, "Parent_ID_Plus_Clim", "purple", "s", "parent-ID + fixed-moment control"),
        (0.18, "No_Terrain_Dynamic_Only", "gray", "^", "dynamic-only inputs"),
    ]:
        delta = frame[column].to_numpy() - intact
        ax.scatter(x + offset, delta, color=color, marker=marker, s=25, label=label, zorder=3)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{level.replace('A_', '')}:{fold}" for level, fold in zip(frame["level"], frame["fold"])], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Baseline − intact\nCRPS [m/s]", fontsize=8.5)
    ax.set_title("Parent-ID and terrain-input sensitivity", fontsize=10, fontweight="bold")
    if show_note:
        ax.text(0.02, 0.03, "+ means intact terrain has lower CRPS; descriptive sensitivity only", transform=ax.transAxes, fontsize=7, color="dimgray")
    legend_kwargs = {"loc": legend_loc, "fontsize": 7, "ncol": legend_ncol}
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    ax.legend(**legend_kwargs)
    ax.grid(True, alpha=0.25)


def _save_single(draw, args, path: Path, figsize=(7.0, 4.5)) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    draw(ax, *args) if isinstance(args, tuple) else draw(ax, args)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    fig_dir, ev_dir = resolve_paths()
    perm, attribution = load_terrain_data(ev_dir)

    _save_single(draw_permutation, (perm, "speed_degradation_vs_intact", "SI: speed terrain permutation sensitivity", "skyblue"), fig_dir / "si_terrain_permutation_speed.pdf")
    _save_single(draw_permutation, (perm, "J_degradation_vs_intact", r"SI: $J$ terrain permutation sensitivity", "salmon"), fig_dir / "si_terrain_permutation_J.pdf")
    _save_single(draw_parent_id, attribution, fig_dir / "si_terrain_parent_id_comparison.pdf")

    # Legacy aliases retained for the old reproducibility receipt.
    _save_single(draw_permutation, (perm[perm["level"].eq("A_T") & perm["fold"].eq("fold1")], "speed_degradation_vs_intact", "(a) Within-test permutation: speed", "skyblue"), fig_dir / "fig4a_perm_null_speed.pdf")
    _save_single(draw_permutation, (perm[perm["level"].eq("A_T") & perm["fold"].eq("fold1")], "J_degradation_vs_intact", r"(b) Within-test permutation: $J$", "salmon"), fig_dir / "fig4b_perm_null_J.pdf")
    _save_single(draw_parent_id, attribution, fig_dir / "fig4c_parent_id_comparison.pdf")

    # Supplementary terrain composite at article width: vertical stacking keeps
    # the 11 fold labels and sensitivity legend legible.
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.9), dpi=300)
    draw_permutation(axes[0], perm, "speed_degradation_vs_intact", "(a) Speed perturbation", "skyblue", show_note=False)
    draw_permutation(axes[1], perm, "J_degradation_vs_intact", r"(b) $J$ perturbation", "salmon", show_note=False)
    draw_parent_id(axes[2], attribution, legend_loc="upper center", legend_bbox=(0.5, -0.42), legend_ncol=3, show_note=False)
    axes[2].set_title("(c) Parent-ID and terrain-input sensitivity", fontsize=10, fontweight="bold")
    _enforce_composite_font_floor(fig)
    fig.suptitle("Supplementary terrain sensitivity", fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.97), h_pad=1.8)
    fig.text(
        0.5,
        0.025,
        "Specified within-test perturbation; not a causal or p-value test.\n"
        "+ means intact terrain has lower CRPS; descriptive sensitivity only.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="dimgray",
    )
    fig.savefig(fig_dir / "si_terrain_sensitivity.pdf")
    plt.close(fig)
    print("Saved supplementary terrain sensitivity panels")


if __name__ == "__main__":
    main()
