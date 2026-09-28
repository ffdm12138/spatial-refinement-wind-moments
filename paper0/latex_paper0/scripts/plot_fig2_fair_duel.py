#!/usr/bin/env python3
"""Generate the development-fold representation comparison.

All 11 development folds are plotted, including negative and trade-off
examples. The score is the frozen window-pooled, parent-cell marginal CRPS;
the figure does not imply a time-resolved or spatial-joint evaluation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figure_utils import finite_rows, require_columns, resolve_paths


def _enforce_composite_font_floor(fig, minimum: float = 8.5) -> None:
    """Raise composite-only text to a readable article-width minimum."""
    fig.canvas.draw()
    for artist in fig.findobj():
        if hasattr(artist, "get_fontsize") and hasattr(artist, "set_fontsize"):
            if artist.get_fontsize() < minimum:
                artist.set_fontsize(minimum)


def _add_composite_footer(fig, ax, text: str, pad: float = 0.045) -> None:
    """Place a panel qualification below the axes, outside the data area."""
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + 0.01,
        bbox.y0 - pad,
        text,
        ha="left",
        va="top",
        fontsize=8.5,
        color="dimgray",
    )


FOLDS = [
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


def load_plot_rows(ev_dir: Path) -> pd.DataFrame:
    bootstrap_path = ev_dir / "QA_VS_JS_FINAL_BOOTSTRAP.csv"
    classified_path = ev_dir / "QA_VS_JS_FINAL_CLASSIFIED.csv"
    bootstrap = pd.read_csv(bootstrap_path)
    classified = pd.read_csv(classified_path)
    require_columns(
        bootstrap,
        [
            "level",
            "fold",
            "candidate",
            "baseline",
            "delta_speed_ci_low",
            "delta_speed_ci_high",
            "delta_J_ci_low",
            "delta_J_ci_high",
        ],
        bootstrap_path,
    )
    require_columns(
        classified,
        [
            "level",
            "fold",
            "QA_dyn_speed_point_diff",
            "QA_dyn_J_point_diff",
            "QA_dyn_ci_relation",
        ],
        classified_path,
    )

    bootstrap = bootstrap[
        (bootstrap["candidate"] == "QA_dyn")
        & (bootstrap["baseline"] == "Prob_JS_joint")
    ].copy()
    rows = []
    for level, fold in FOLDS:
        cls = classified[(classified["level"] == level) & (classified["fold"] == fold)]
        boot = bootstrap[(bootstrap["level"] == level) & (bootstrap["fold"] == fold)]
        if len(cls) != 1 or len(boot) != 1:
            raise ValueError(f"Missing unique final row for {level}/{fold}")
        c_row = cls.iloc[0]
        b_row = boot.iloc[0]
        rows.append(
            {
                "level": level,
                "fold": fold,
                "label": f"{level.replace('A_', '')}:{fold}",
                "speed_point": float(c_row["QA_dyn_speed_point_diff"]),
                "speed_ci_low": float(b_row["delta_speed_ci_low"]),
                "speed_ci_high": float(b_row["delta_speed_ci_high"]),
                "J_point": float(c_row["QA_dyn_J_point_diff"]),
                "J_ci_low": float(b_row["delta_J_ci_low"]),
                "J_ci_high": float(b_row["delta_J_ci_high"]),
                "classification": str(c_row["QA_dyn_ci_relation"]),
            }
        )
    frame = pd.DataFrame(rows)
    finite_rows(
        frame,
        [
            "speed_point",
            "speed_ci_low",
            "speed_ci_high",
            "J_point",
            "J_ci_low",
            "J_ci_high",
        ],
        classified_path,
    )
    return frame


def draw_speed(ax, frame: pd.DataFrame, show_note: bool = True) -> None:
    y = np.arange(len(frame))
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.hlines(y, frame["speed_ci_low"], frame["speed_ci_high"], color="royalblue", lw=1.5)
    ax.scatter(frame["speed_point"], y, color="navy", s=22, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_{\rm speed}=\mathrm{CRPS}_{JS}-\mathrm{CRPS}_{QA}$ [m/s]", fontsize=8.5)
    ax.set_title("(a) Speed: development folds", fontsize=10, fontweight="bold")
    if show_note:
        ax.text(
            0.02,
            0.02,
            "window-pooled parent-cell marginal; + favors QA",
            transform=ax.transAxes,
            fontsize=7,
            color="dimgray",
        )
    ax.grid(True, alpha=0.25)


def draw_gap(ax, frame: pd.DataFrame, show_note: bool = True) -> None:
    y = np.arange(len(frame))
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.hlines(y, frame["J_ci_low"], frame["J_ci_high"], color="indianred", lw=1.5)
    ax.scatter(frame["J_point"], y, color="darkred", marker="s", s=22, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta_J=\mathrm{CRPS}_{JS}-\mathrm{CRPS}_{QA}$ [m/s]", fontsize=8.5)
    ax.set_title("(b) Aggregation gap $J$: development folds", fontsize=10, fontweight="bold")
    if show_note:
        ax.text(
            0.02,
            0.02,
            "all folds retained, including reversals and intervals spanning zero",
            transform=ax.transAxes,
            fontsize=7,
            color="dimgray",
        )
    ax.grid(True, alpha=0.25)


CLASS_COLORS = {
    "QA_BOTH_CI_SUPPORTED": "forestgreen",
    "ONE_CI_POSITIVE_OTHER_SPANS_ZERO": "darkorange",
    "INDISTINGUISHABLE": "slategray",
    "JS_BOTH_CI_SUPPORTED": "firebrick",
    "JS_ONE_CI_NEGATIVE_OTHER_SPANS_ZERO": "purple",
    "TRADE_OFF": "dodgerblue",
}
CLASS_LABELS = {
    "QA_BOTH_CI_SUPPORTED": "QA: both CIs positive",
    "ONE_CI_POSITIVE_OTHER_SPANS_ZERO": "one positive, one spans zero",
    "INDISTINGUISHABLE": "both CIs span zero",
    "JS_BOTH_CI_SUPPORTED": "JS: both CIs negative",
    "JS_ONE_CI_NEGATIVE_OTHER_SPANS_ZERO": "one negative, one spans zero",
    "TRADE_OFF": "opposite point/CI directions",
}

CLASS_ANNOTATION_OFFSETS = {
    "T:fold1": (12, 14),
    "T:fold2": (-25, 40),
    "T:fold3": (14, 24),
    "S:S1": (70, -2),
    "S:S2": (10, -10),
    "S:S3": (6, 36),
    "S:S4": (22, 16),
    "ST:S1": (60, -18),
    "ST:S2": (75, 0),
    "ST:S3": (18, 14),
    "ST:S4": (14, 15),
}


def draw_classification(
    ax,
    frame: pd.DataFrame,
    legend_loc: str = "best",
    legend_bbox=None,
    legend_ncol: int = 1,
    show_note: bool = True,
    annotation_offsets=None,
) -> None:
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axvline(0, color="gray", ls="--", lw=1)
    for _, row in frame.iterrows():
        color = CLASS_COLORS.get(row["classification"], "black")
        ax.hlines(row["J_point"], row["speed_ci_low"], row["speed_ci_high"], color=color, lw=1.0)
        ax.vlines(row["speed_point"], row["J_ci_low"], row["J_ci_high"], color=color, lw=1.0)
        ax.plot(row["speed_point"], row["J_point"], "o", color=color, ms=4)
        offset = (3, 3) if annotation_offsets is None else annotation_offsets.get(row["label"], (5, 5))
        dx, dy = offset
        annotation_kwargs = {
            "xytext": offset,
            "textcoords": "offset points",
            "fontsize": 6.8,
            "ha": "left" if dx >= 0 else "right",
        }
        if annotation_offsets is not None:
            annotation_kwargs["arrowprops"] = {
                "arrowstyle": "-",
                "color": color,
                "lw": 0.6,
                "alpha": 0.8,
                "shrinkA": 0,
                "shrinkB": 2,
            }
        annotation_kwargs["annotation_clip"] = False
        ax.annotate(row["label"], (row["speed_point"], row["J_point"]), **annotation_kwargs)
    ax.set_xlabel(r"$\Delta_{\rm speed}$ [m/s]", fontsize=8.5)
    ax.set_ylabel(r"$\Delta_J$ [m/s]", fontsize=8.5)
    ax.set_title("(c) Paired point and CI classifications", fontsize=10, fontweight="bold")
    if annotation_offsets is not None:
        ax.set_ylim(-0.022, max(frame["J_ci_high"]) + 0.015)
    if show_note:
        ax.text(
            0.02,
            0.02,
            "point dominance and dual-CI support are separate fields",
            transform=ax.transAxes,
            fontsize=7,
            color="dimgray",
        )
    present = [key for key in CLASS_LABELS if key in set(frame["classification"])]
    handles = [Line2D([0], [0], marker="o", color=CLASS_COLORS[key], lw=0, label=CLASS_LABELS[key]) for key in present]
    if handles:
        legend_kwargs = {"loc": legend_loc, "fontsize": 6.8, "ncol": legend_ncol}
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        ax.legend(handles=handles, **legend_kwargs)
    ax.grid(True, alpha=0.25)


def _save_single(draw, frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    draw(ax, frame)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    fig_dir, ev_dir = resolve_paths()
    frame = load_plot_rows(ev_dir)

    # Active panel names used by the revised manuscript.
    _save_single(draw_speed, frame, fig_dir / "fig2a_development_speed.pdf")
    _save_single(draw_gap, frame, fig_dir / "fig2b_development_J.pdf")
    _save_single(draw_classification, frame, fig_dir / "fig2c_development_classification.pdf")

    # Legacy aliases retained for the old reproducibility receipt.
    _save_single(draw_speed, frame, fig_dir / "fig2a_speed_crps_folds.pdf")
    _save_single(draw_gap, frame, fig_dir / "fig2b_J_crps_folds.pdf")
    _save_single(draw_classification, frame, fig_dir / "fig2c_pareto_classification.pdf")

    # Article-width composite: one panel per row keeps the 11 fold labels and
    # CI annotations readable at approximately 6.8 in final width.
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.9), dpi=300)
    draw_speed(axes[0], frame, show_note=False)
    draw_gap(axes[1], frame, show_note=False)
    draw_classification(
        axes[2],
        frame,
        legend_loc="upper center",
        legend_bbox=(0.5, -0.34),
        legend_ncol=2,
        show_note=False,
        annotation_offsets=CLASS_ANNOTATION_OFFSETS,
    )
    _enforce_composite_font_floor(fig)
    fig.suptitle("Development comparison: window-pooled parent-cell marginals", fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.97), h_pad=1.4)
    # The caption carries the scope; repeated footers collide with x labels.
    fig.savefig(fig_dir / "fig2_development.pdf")
    plt.close(fig)
    print("Saved Figure 2 development panels and combined figure")


if __name__ == "__main__":
    main()
