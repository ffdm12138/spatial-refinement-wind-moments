#!/usr/bin/env python3
"""Generate the Paper0 representation figure.

The figure keeps the reference-state identity and deployment approximation in
separate panels. Panel (b) is explicitly a fixed-V coordinate statement;
panel (c) shows the U-based deployment proxy and is labelled schematic. The
script writes the new combined figure and legacy one-panel files so old
reproduction receipts remain readable during the transition.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figure_utils import resolve_paths


def _enforce_composite_font_floor(fig, minimum: float = 8.5) -> None:
    """Keep text legible when the composite is placed at article width.

    The legacy one-panel files retain their original typography.  Only the
    active composite receives this floor after all legends and tick labels
    have been created.
    """
    fig.canvas.draw()
    for artist in fig.findobj():
        if hasattr(artist, "get_fontsize") and hasattr(artist, "set_fontsize"):
            if artist.get_fontsize() < minimum:
                artist.set_fontsize(minimum)


def _add_composite_footer(fig, ax, text: str, pad: float = 0.07) -> None:
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


def draw_panel_a(ax, show_note=True, legend_loc="upper left", legend_bbox=None):
    """Illustrate paired variance coordinates, without a three-state narrative."""
    allocation=0.35
    ax.barh([0], [1-allocation], color="navy", height=.42, label=r"Scalar variance $S^2=(1-a)Q$")
    ax.barh([0], [allocation], left=1-allocation, color="crimson", height=.42, label=r"Gap-related $aQ=2VJ+J^2$")
    ax.text(.325,0,r"$(1-a)Q$",color="white",ha="center",va="center",fontsize=12)
    ax.text(.825,0,r"$aQ$",color="white",ha="center",va="center",fontsize=12)
    ax.set_xlim(0,1);ax.set_ylim(-.6,.9);ax.set_yticks([])
    ax.set_xlabel(r"Fraction of total horizontal vector variance $Q$")
    ax.set_title("(a) Total variance and its allocation (schematic)",fontsize=10,fontweight="bold")
    ax.legend(loc="upper left",fontsize=9,frameon=False)
    ax.spines[['left','top','right']].set_visible(False)


def draw_panel_b(ax, show_note: bool = True):
    """Draw the coordinate relation at an explicitly fixed reference speed."""
    V = 5.0
    j_values = np.linspace(0.01, 3.0, 200)
    s_values = np.linspace(0.01, 4.0, 200)
    jj, ss = np.meshgrid(j_values, s_values)
    qq = ss**2 + 2 * V * jj + jj**2
    aa = (2 * V * jj + jj**2) / qq

    cp_q = ax.contour(
        jj,
        ss,
        np.log10(qq),
        levels=8,
        colors="navy",
        linestyles="-",
        alpha=0.7,
    )
    ax.clabel(cp_q, inline=True, fmt=r"$\log_{10}Q=%.1f$", fontsize=7)
    cp_a = ax.contour(
        jj,
        ss,
        aa,
        levels=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        colors="crimson",
        linestyles="--",
        alpha=0.8,
    )
    ax.clabel(cp_a, inline=True, fmt=r"$a=%.1f$", fontsize=7)
    ax.set_xlabel(r"Aggregation gap $J$ [m/s]", fontsize=9)
    ax.set_ylabel(r"Scalar spread $S=\sigma_s$ [m/s]", fontsize=9)
    ax.set_title(
        r"(b) $(J,S)\leftrightarrow(Q,a)$ at fixed $V=5$ m/s",
        fontsize=10,
        fontweight="bold",
    )
    if show_note:
        ax.text(
            0.03,
            0.03,
            "reference-state coordinate identity",
            transform=ax.transAxes,
            fontsize=7.5,
            color="dimgray",
        )
    ax.legend(
        [
            Line2D([0], [0], color="navy", lw=2),
            Line2D([0], [0], color="crimson", lw=2, ls="--"),
        ],
        [r"$Q$ isocontours", r"$a$ contours"],
        loc="upper left",
        fontsize=7.5,
    )
    ax.grid(True, alpha=0.25)


def draw_panel_c(ax):
    """Draw the deployment curve and make the U/V substitution explicit."""
    u_values = np.linspace(0.5, 15.0, 100)
    q_fixed = 4.0
    for a_value, color, linestyle in [
        (0.2, "teal", "-."),
        (0.5, "darkorange", "-"),
        (0.8, "purple", "--"),
    ]:
        qa_mu = np.sqrt(u_values**2 + a_value * q_fixed)
        ax.plot(
            u_values,
            qa_mu - u_values,
            color=color,
            ls=linestyle,
            lw=1.8,
            label=rf"$a={a_value},Q={q_fixed}$",
        )
    ax.axhline(0.5, color="gray", ls=":", lw=1.5, label=r"schematic JS $\hat J$")
    ax.set_xlabel(r"Deployment coarse speed $U$ [m/s]", fontsize=9)
    ax.set_ylabel(r"Proxy gap $\hat J_U=\hat\mu-U$ [m/s]", fontsize=9)
    ax.set_title(
        r"(c) Deployment mapping uses $U$ as a proxy",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.25)


def _save_single(draw, path: Path, figsize=(5.7, 4.4)) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    draw(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    fig_dir, _ = resolve_paths()

    # Active panel names used by the revised manuscript.
    _save_single(draw_panel_a, fig_dir / "fig1a_representation.pdf")
    _save_single(draw_panel_b, fig_dir / "fig1b_fixed_V_coordinates.pdf")
    _save_single(draw_panel_c, fig_dir / "fig1c_deployment_U_proxy.pdf")

    # Legacy aliases retained for the old reproducibility receipt.
    _save_single(draw_panel_a, fig_dir / "fig1a_three_states.pdf")
    _save_single(draw_panel_b, fig_dir / "fig1b_coordinate_isomorphism.pdf")
    _save_single(draw_panel_c, fig_dir / "fig1c_deployment_recovery.pdf")

    # Article-width composite: stack the panels so labels are not reduced to
    # 2.5--4 pt when the old 16-inch strip is placed in a 16-cm text column.
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.8), dpi=300)
    draw_panel_a(axes[0], show_note=False, legend_loc="upper left", legend_bbox=(1.0, 1.0))
    draw_panel_b(axes[1], show_note=False)
    draw_panel_c(axes[2])
    _enforce_composite_font_floor(fig)
    fig.suptitle(
        "Representation and deployment boundary",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.97), h_pad=1.5)
    # These qualifications are already stated in the manuscript caption.
    # In-panel footers would collide with the following stacked panel title.
    fig.savefig(fig_dir / "fig1_representation.pdf")
    plt.close(fig)
    print("Saved Figure 1 representation panels and combined figure")


if __name__ == "__main__":
    main()
