#!/usr/bin/env python3
"""Generate the Paper0 partition and covariance ablation figure.

The panels are deliberately descriptive. They compare tested pipelines under
the same fold protocol; they do not identify a physical mechanism, establish
independence of Q and a, or certify complete spatial-joint calibration.
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


def _fold_labels(frame: pd.DataFrame) -> list[str]:
    return [f"{level.replace('A_', '')}:{fold}" for level, fold in zip(frame["level"], frame["fold"])]


def load_ablation_data(ev_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dyn_path = ev_dir / "DYN_VS_CONST_NESTED.csv"
    m2_path = ev_dir / "M2_VS_M3_NESTED.csv"
    scatter_path = ev_dir / "EMPIRICAL_QA_SCATTER_SAMPLE_5000.csv"
    dyn = pd.read_csv(dyn_path)
    m2 = pd.read_csv(m2_path)
    scatter = pd.read_csv(scatter_path)
    require_columns(
        dyn,
        [
            "level",
            "fold",
            "delta_speed_point_const_minus_dyn",
            "delta_speed_ci_low",
            "delta_speed_ci_high",
            "delta_J_point_const_minus_dyn",
            "delta_J_ci_low",
            "delta_J_ci_high",
        ],
        dyn_path,
    )
    require_columns(
        m2,
        [
            "level",
            "fold",
            "delta_speed_M3_minus_M2",
            "delta_speed_ci_low",
            "delta_speed_ci_high",
            "delta_J_M3_minus_M2",
            "delta_J_ci_low",
            "delta_J_ci_high",
        ],
        m2_path,
    )
    require_columns(scatter, ["Q", "a", "U"], scatter_path)
    finite_rows(
        dyn,
        [
            "delta_speed_point_const_minus_dyn",
            "delta_speed_ci_low",
            "delta_speed_ci_high",
            "delta_J_point_const_minus_dyn",
            "delta_J_ci_low",
            "delta_J_ci_high",
        ],
        dyn_path,
    )
    finite_rows(
        m2,
        [
            "delta_speed_M3_minus_M2",
            "delta_speed_ci_low",
            "delta_speed_ci_high",
            "delta_J_M3_minus_M2",
            "delta_J_ci_low",
            "delta_J_ci_high",
        ],
        m2_path,
    )
    finite_rows(scatter, ["Q", "a", "U"], scatter_path)
    return dyn, m2, scatter


def _plot_delta_panel(ax, frame: pd.DataFrame, speed_point: str, speed_low: str, speed_high: str, j_point: str, j_low: str, j_high: str, title: str, subtitle: str, speed_label: str, j_label: str, show_note: bool = True) -> None:
    x = np.arange(len(frame))
    labels = _fold_labels(frame)
    offsets = (-0.14, 0.14)
    ax.errorbar(
        x + offsets[0],
        frame[speed_point],
        yerr=np.vstack([frame[speed_point] - frame[speed_low], frame[speed_high] - frame[speed_point]]),
        fmt="o",
        color="steelblue",
        ms=4,
        capsize=2,
        label=speed_label,
    )
    ax.errorbar(
        x + offsets[1],
        frame[j_point],
        yerr=np.vstack([frame[j_point] - frame[j_low], frame[j_high] - frame[j_point]]),
        fmt="s",
        color="firebrick",
        ms=4,
        capsize=2,
        label=j_label,
    )
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CRPS difference [m/s]", fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if show_note:
        ax.text(0.02, 0.03, subtitle, transform=ax.transAxes, fontsize=7, color="dimgray")
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(True, alpha=0.25)


def draw_partition_ablation(ax, dyn: pd.DataFrame, show_note: bool = True) -> None:
    _plot_delta_panel(
        ax,
        dyn,
        "delta_speed_point_const_minus_dyn",
        "delta_speed_ci_low",
        "delta_speed_ci_high",
        "delta_J_point_const_minus_dyn",
        "delta_J_ci_low",
        "delta_J_ci_high",
        r"(a) Partition ablation: $a(X)$ vs fixed $a^*$",
        "+ means the dynamic partition has lower CRPS; CI bars retain mixed outcomes",
        r"speed $\Delta$ (const $-$ dynamic)",
        r"$J$ $\Delta$ (const $-$ dynamic)",
        show_note,
    )


def draw_covariance_ablation(ax, m2: pd.DataFrame, show_note: bool = True) -> None:
    _plot_delta_panel(
        ax,
        m2,
        "delta_speed_M3_minus_M2",
        "delta_speed_ci_low",
        "delta_speed_ci_high",
        "delta_J_M3_minus_M2",
        "delta_J_ci_low",
        "delta_J_ci_high",
        r"(b) Covariance ablation: independent M2 vs joint M3",
        "+ means the independent M2 pipeline has lower CRPS",
        r"speed $\Delta$ (M3 $-$ M2)",
        r"$J$ $\Delta$ (M3 $-$ M2)",
        show_note,
    )


def draw_Qa_geometry(ax, scatter: pd.DataFrame, show_note: bool = True) -> None:
    points = ax.scatter(scatter["Q"], scatter["a"], c=scatter["U"], cmap="viridis", s=6, alpha=0.45, rasterized=False)
    ax.set_xscale("log")
    ax.set_xlabel(r"Total target variance $Q$ [$\mathrm{m}^2\,\mathrm{s}^{-2}$]", fontsize=8.5)
    ax.set_ylabel(r"Partition ratio $a$", fontsize=8.5)
    ax.set_title(r"$(Q,a)$ target geometry (descriptive)", fontsize=10, fontweight="bold")
    # The SI caption states this boundary without covering scatter points.
    colorbar = plt.colorbar(points, ax=ax)
    colorbar.set_label(r"coarse speed $U$ [m/s]", fontsize=8)
    ax.grid(True, alpha=0.25)


def _save_single(draw, data, path: Path, figsize=(6.4, 4.5)) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    draw(ax, *data) if isinstance(data, tuple) else draw(ax, data)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    fig_dir, ev_dir = resolve_paths()
    dyn, m2, scatter = load_ablation_data(ev_dir)

    _save_single(draw_partition_ablation, dyn, fig_dir / "fig4a_partition_ablation.pdf")
    _save_single(draw_covariance_ablation, m2, fig_dir / "fig4b_covariance_ablation.pdf")
    _save_single(draw_Qa_geometry, scatter, fig_dir / "si_Qa_geometry.pdf")

    # Legacy aliases retained for the old reproducibility receipt.
    _save_single(draw_Qa_geometry, scatter, fig_dir / "fig3a_Q_vs_a_scatter.pdf")
    _save_single(draw_partition_ablation, dyn, fig_dir / "fig3b_const_vs_dyn_a.pdf")
    _save_single(draw_covariance_ablation, m2, fig_dir / "fig3c_M2_vs_M3_coupling.pdf")

    # Article-width composite: two rows preserve the fold labels and both CI
    # series without shrinking the 13-inch strip to unreadable text.
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.0), dpi=300)
    draw_partition_ablation(axes[0], dyn, show_note=False)
    draw_covariance_ablation(axes[1], m2, show_note=False)
    _enforce_composite_font_floor(fig)
    fig.suptitle("Partition and covariance ablations", fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.97), h_pad=1.8)
    fig.text(
        0.5,
        0.035,
        "+ means the named first pipeline has lower CRPS; partition CI bars retain mixed outcomes.\n"
        "+ independent M2 is compared with joint M3 under the same fold protocol.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="dimgray",
    )
    fig.savefig(fig_dir / "fig4_ablations.pdf")
    plt.close(fig)
    print("Saved Figure 4 ablation panels and SI Q-a geometry")


if __name__ == "__main__":
    main()


