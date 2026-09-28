#!/usr/bin/env python3
"""Generate the retrospective and cross-case transfer figure.

Panel (a) is the 670-state Case A evaluation and is labelled retrospective
because the window was read during earlier selection. Panels (b,c) are frozen
no-refit transfers to Cases B and C. The Case C U-regime diagnosis remains a
supplementary scalar diagnostic and is never used as a mechanism claim.
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


def _load_transfer(path: Path, case: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        [
            "case",
            "candidate",
            "baseline",
            "delta_speed_point",
            "delta_speed_ci_low",
            "delta_speed_ci_high",
            "delta_J_point",
            "delta_J_ci_low",
            "delta_J_ci_high",
        ],
        path,
    )
    frame = frame[(frame["case"] == case) & (frame["baseline"] == "Prob_JS_joint")].copy()
    frame = frame[frame["candidate"] != "Climatology"].copy()
    if frame.empty:
        raise ValueError(f"{path.name}: no canonical transfer rows for case {case}")
    finite_rows(
        frame,
        [
            "delta_speed_point",
            "delta_speed_ci_low",
            "delta_speed_ci_high",
            "delta_J_point",
            "delta_J_ci_low",
            "delta_J_ci_high",
        ],
        path,
    )
    return frame


def load_data(ev_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    heldout_path = ev_dir / "A_HELD_OUT_670_BOOTSTRAP.csv"
    bc_path = ev_dir / "BC_TRANSFER_FINAL_BOOTSTRAP.csv"
    diag_path = ev_dir / "CASE_C_FINAL_DIAGNOSIS.csv"
    heldout = _load_transfer(heldout_path, "A")
    case_b = _load_transfer(bc_path, "B")
    case_c = _load_transfer(bc_path, "C")
    diagnosis = pd.read_csv(diag_path)
    require_columns(diagnosis, ["case", "regime", "model", "sample_fraction", "mae_mu_sample_level"], diag_path)
    diagnosis = diagnosis[diagnosis["case"] == "C"].copy()
    finite_rows(diagnosis, ["sample_fraction", "mae_mu_sample_level"], diag_path)
    return heldout, case_b, case_c, diagnosis


def draw_transfer(ax, frame: pd.DataFrame, title: str, subtitle: str, show_note: bool = True) -> None:
    labels = frame["candidate"].astype(str).tolist()
    y = np.arange(len(frame))
    speed_y = y - 0.14
    j_y = y + 0.14
    ax.errorbar(
        frame["delta_speed_point"],
        speed_y,
        xerr=np.vstack([frame["delta_speed_point"] - frame["delta_speed_ci_low"], frame["delta_speed_ci_high"] - frame["delta_speed_point"]]),
        fmt="o",
        color="navy",
        ms=4,
        capsize=2,
        label="speed",
    )
    ax.errorbar(
        frame["delta_J_point"],
        j_y,
        xerr=np.vstack([frame["delta_J_point"] - frame["delta_J_ci_low"], frame["delta_J_ci_high"] - frame["delta_J_point"]]),
        fmt="s",
        color="darkred",
        ms=4,
        capsize=2,
        label=r"$J$",
    )
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta=\mathrm{CRPS}_{JS}-\mathrm{CRPS}_{candidate}$ [m/s]", fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if show_note:
        ax.text(0.02, 0.03, subtitle, transform=ax.transAxes, fontsize=7, color="dimgray")
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(True, alpha=0.25)


def draw_case_c_regimes(ax, diagnosis: pd.DataFrame) -> None:
    regimes = [
        "low_speed_U_lt_6",
        "moderate_speed_6_le_U_lt_10",
        "high_speed_U_ge_10",
    ]
    labels = ["low\n$U<6$", "moderate\n$6\u2264U<10$", "high\n$U\u226510$"]
    models = ["QA_dyn", "Prob_JS_joint"]
    x = np.arange(len(regimes))
    width = 0.35
    colors = ["teal", "crimson"]
    for offset, model, color in zip((-width / 2, width / 2), models, colors):
        vals = []
        for regime in regimes:
            row = diagnosis[(diagnosis["regime"] == regime) & (diagnosis["model"] == model)]
            if len(row) != 1:
                raise ValueError(f"Missing unique Case C regime row: {regime}/{model}")
            vals.append(float(row.iloc[0]["mae_mu_sample_level"]))
        ax.bar(x + offset, vals, width=width, color=color, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Scalar speed MAE [m/s]", fontsize=8.5)
    ax.set_title("SI: Case C error by coarse-speed regime", fontsize=10, fontweight="bold")
    # The SI caption states this boundary without covering the bars.
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.25)


def _save_single(draw, args, path: Path, figsize=(6.3, 4.5)) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    draw(ax, *args) if isinstance(args, tuple) else draw(ax, args)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    fig_dir, ev_dir = resolve_paths()
    heldout, case_b, case_c, diagnosis = load_data(ev_dir)

    _save_single(draw_transfer, (heldout, "(a) Case A: 670-state retrospective evaluation", "not independent confirmation; no refit in this window"), fig_dir / "fig3a_A670_retrospective.pdf")
    _save_single(draw_transfer, (case_b, "(b) Case B: frozen retrospective transfer", "full transfer window; canonical JS comparator"), fig_dir / "fig3b_case_B_transfer.pdf")
    _save_single(draw_transfer, (case_c, "(c) Case C: frozen retrospective transfer", "full transfer window; canonical JS comparator"), fig_dir / "fig3c_case_C_transfer.pdf")
    _save_single(draw_case_c_regimes, diagnosis, fig_dir / "si_case_C_regimes.pdf")

    # Legacy aliases retained for the old reproducibility receipt.
    _save_single(draw_transfer, (case_b, "(a) Case B frozen transfer", "legacy panel alias"), fig_dir / "fig5a_case_B_transfer.pdf")
    _save_single(draw_transfer, (case_c, "(b) Case C frozen transfer", "legacy panel alias"), fig_dir / "fig5b_case_C_transfer.pdf")
    _save_single(draw_case_c_regimes, diagnosis, fig_dir / "fig5c_case_C_regimes.pdf")

    # Article-width composite: stack the retrospective and two transfer panels
    # so candidate labels and uncertainty intervals remain legible.
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.9), dpi=300)
    draw_transfer(axes[0], heldout, "(a) A: 670-state retrospective", "no refit; post-development evaluation", show_note=False)
    draw_transfer(axes[1], case_b, "(b) B: retrospective seasonal transfer", "no refit; full transfer window", show_note=False)
    draw_transfer(axes[2], case_c, "(c) C: retrospective site/season transfer", "no refit; full transfer window", show_note=False)
    _enforce_composite_font_floor(fig)
    fig.suptitle("Retrospective evaluation and transfer", fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.97), h_pad=1.4)
    # Panel titles and caption carry the scope without crowding x labels.
    fig.savefig(fig_dir / "fig3_transfer.pdf")
    plt.close(fig)
    print("Saved Figure 3 transfer panels and SI Case C regime diagnostic")


if __name__ == "__main__":
    main()
