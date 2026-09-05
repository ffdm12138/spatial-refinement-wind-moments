"""Generate the three multicase Comment figures from audited source tables.

Every plotted number is read at run time from the frozen, audited candidate
tables of the 20260905 multicase generation:

  comment_candidate/C1_core_loss.csv            -> Fig. 1
  comment_candidate/C2_geometric_recovery.csv   -> Fig. 2  (the 36/36-verified table)
  comment_candidate/C3_transfers.csv            -> Fig. 3 (a,b)
  comment_candidate/C4_information_sources.csv  -> Fig. 3 (c)

No manually typed plotted values.  Run with the 20260905 generation active:

  NPJ_MULTICASE_GENERATION=20260905 python make_comment_figures.py

Outputs (vector PDF) go to the repository figures/ directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mc_config

CASE_COLORS = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
CASE_LABELS = {"A": "A\nYakou Sep", "B": "B\nYakou Dec", "C": "C\nEbao Dec"}
CASE_TAGS = {"A": "Yakou Sep", "B": "Yakou Dec", "C": "Ebao Dec"}
CAL_COLOR = "#9ecae1"
EVAL_COLOR = "#08519c"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def out_dir() -> Path:
    d = mc_config.ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def candidate_dir() -> Path:
    """Audited figure-source tables.  Default: the shipped `figure_source/`
    directory; set COMMENT_TABLES_DIR to point at a freshly regenerated
    `results/<extension>/comment_candidate` directory instead."""
    import os
    override = os.environ.get("COMMENT_TABLES_DIR", "").strip()
    if override:
        return Path(override)
    return mc_config.ROOT / "figure_source"


# ---------------------------------------------------------------------------
# Figure 1 — replication of the representation loss across A/B/C
# ---------------------------------------------------------------------------
def fig1() -> Path:
    df = pd.read_csv(candidate_dir() / "C1_core_loss.csv")
    cases = ["A", "B", "C"]
    quantities = [
        ("J_domain_mean", "Jensen gap $J$ (m\\,s$^{-1}$)", "$J$"),
        ("delta_scalar_domain_mean", "Scalar increment $\\Delta_{\\rm scalar}$ (m\\,s$^{-1}$)",
         "$\\Delta_{\\rm scalar}$"),
        ("delta_dyn_domain_mean", "Mean-vector term $\\Delta_{\\rm dyn}$ (m\\,s$^{-1}$)",
         "$\\Delta_{\\rm dyn}$"),
        ("S_domain_mean", "Within-parent spread $S$ (m\\,s$^{-1}$)", "$S$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.5))
    x = np.arange(3)
    w = 0.36
    for ax, (col, ylabel, tag) in zip(axes, quantities):
        cal = [df[(df.case == c) & (df.window == "calibration")][col].iloc[0]
               for c in cases]
        ev = [df[(df.case == c) & (df.window == "evaluation")][col].iloc[0]
              for c in cases]
        ax.bar(x - w / 2, cal, w, color=CAL_COLOR, edgecolor="black",
               linewidth=0.5, label="calibration (2,680 states)")
        ax.bar(x + w / 2, ev, w, color=EVAL_COLOR, edgecolor="black",
               linewidth=0.5, label="evaluation (670 states)")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_TAGS[c] for c in cases])
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + quantities.index((col, ylabel, tag)))}) {tag}")
        lo = min(min(cal), min(ev), 0.0)
        hi = max(max(cal), max(ev), 0.0)
        pad = 0.12 * (hi - lo) if hi > lo else 0.05 * max(abs(hi), 0.01)
        ax.set_ylim(lo - pad, hi + pad)
    # J/delta_scalar ratio annotations on panel (b)
    ax_b = axes[1]
    for i, c in enumerate(cases):
        for dx, win in ((-w / 2, "calibration"), (w / 2, "evaluation")):
            j = df[(df.case == c) & (df.window == win)]["J_domain_mean"].iloc[0]
            d = df[(df.case == c) & (df.window == win)][
                "delta_scalar_domain_mean"].iloc[0]
            ax_b.annotate(f"{j / d:.3f}", (i + dx, d),
                          textcoords="offset points", xytext=(0, 2),
                          ha="center", fontsize=6.2, rotation=90)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CAL_COLOR, edgecolor="black",
                      linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor=EVAL_COLOR, edgecolor="black",
                      linewidth=0.5),
    ]
    fig.legend(handles, ["calibration (2,680 states)", "evaluation (670 states)"],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.93))
    out = out_dir() / "Fig1_representation_loss_multicase.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — geometry-only recovery across all four controls (36/36 verified)
# ---------------------------------------------------------------------------
def fig2() -> Path:
    df = pd.read_csv(candidate_dir() / "C2_geometric_recovery.csv")
    models = ["bilinear", "bicubic", "bilinear_constrained", "bicubic_constrained"]
    model_labels = ["bilinear", "bicubic", "bilinear\n+ constraint",
                    "bicubic\n+ constraint"]
    model_colors = ["#bdbdbd", "#636363", "#c6dbef", "#2171b5"]
    cases = ["A", "B", "C"]
    moments = [
        ("Q_delta_scalar", "Recovery of $\\Delta_{\\rm scalar}$, $Q_\\Delta$"),
        ("Q_J", "Recovery of the Jensen gap, $Q_J$"),
        ("Q_spread", "Recovery of within-parent spread, $Q_S$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))
    for ax, (col, title) in zip(axes, moments):
        pos, vals, colors, tickpos, ticklab = [], [], [], [], []
        cursor = 0.0
        for c in cases:
            start = cursor
            for m in models:
                row = df[(df.case == c) & (df.model == m)]
                vals.append(float(row[col].iloc[0]))
                colors.append(model_colors[models.index(m)])
                pos.append(cursor)
                cursor += 1.0
            tickpos.append((start + cursor - 1) / 2)
            ticklab.append(CASE_TAGS[c])
            cursor += 0.7
        vals = np.asarray(vals)
        ax.bar(pos, vals, 0.82, color=colors, edgecolor="black", linewidth=0.4)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.axhline(1.0, color="#cb181d", linewidth=0.8, linestyle="--")
        ax.text(pos[-1] + 0.55, 1.0, "complete recovery", color="#cb181d",
                fontsize=6.5, va="center")
        ax.set_xticks(tickpos)
        ax.set_xticklabels(ticklab)
        ax.set_title(title)
        ax.set_ylim(min(vals.min() * 1.25, -0.05), max(1.12, vals.max() * 1.12))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black",
                             linewidth=0.4) for c in model_colors]
    fig.legend(handles, ["bilinear", "bicubic", "bilinear + constraint",
                         "bicubic + constraint"],
               loc="upper center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.92))
    out = out_dir() / "Fig2_geometric_recovery_multicase.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — persistence versus portability, with the C-local reference
# ---------------------------------------------------------------------------
def fig3() -> Path:
    tr = pd.read_csv(candidate_dir() / "C3_transfers.csv")
    infos = pd.read_csv(candidate_dir() / "C4_information_sources.csv")

    def tval(test: str, model: str, col: str, segment: str = "evaluation_670"):
        row = tr[(tr.test == test) & (tr.model == model) & (tr.segment == segment)]
        return float(row[col].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))

    # (a) pattern persistence (Spearman, evaluation windows)
    ax = axes[0]
    labels = ["A$\\to$B\n$\\Delta_{\\rm scalar}$", "A$\\to$B\nspread $S$",
              "B$\\to$C\n$\\Delta_{\\rm scalar}$", "B$\\to$C\nspread $S$"]
    vals = [
        tval("T1", "A_terrain_delta", "spearman"),
        tval("T1", "A_terrain_spread_joint", "spearman"),
        tval("T3", "B_cal_delta_vs_r_sigma", "spearman"),
        tval("T3", "B_cal_S_vs_lambda_iso", "spearman"),
    ]
    colors = ["#2ca02c", "#2ca02c", "#d62728", "#d62728"]
    x = np.arange(4)
    ax.bar(x, vals, 0.7, color=colors, edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3 if v >= 0 else -9), ha="center", fontsize=6.5)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("Spearman $r_s$ (pattern)")
    ax.set_title("(a) Same-site persistence vs cross-site transfer")
    ax.set_ylim(min(min(vals) * 1.35, -0.1), 1.38)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#2ca02c",
                             edgecolor="black", linewidth=0.4),
               plt.Rectangle((0, 0), 1, 1, facecolor="#d62728",
                             edgecolor="black", linewidth=0.4)]
    ax.legend(handles, ["same site, new event (A$\\to$B)",
                        "new site, same event (B$\\to$C)"],
              loc="upper right", frameon=False, fontsize=6.2)

    # (b) cross-site amplitude failure for spread (B-fitted relation at C)
    ax = axes[1]
    ref = tval("T3", "B_cal_S_vs_lambda_iso", "obs_mean")
    pred = tval("T3", "B_cal_S_vs_lambda_iso", "pred_mean")
    bias = tval("T3", "B_cal_S_vs_lambda_iso", "bias")
    ax.bar([0, 1], [ref, pred], 0.55, color=["#6baed6", "#fdae6b"],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ebao reference\n(mean spread $S$)",
                        "Yakou-fitted terrain\nrelation at Ebao"], fontsize=7)
    ax.set_ylabel("Domain-mean spread $S$ (m\\,s$^{-1}$)")
    ax.set_title("(b) Amplitude does not transfer")
    ax.set_ylim(0, max(ref, pred) * 1.45)
    ax.annotate(f"bias ${bias:+.2f}$ m\\,s$^{{-1}}$\n"
                f"ratio ${pred / ref:.2f}$",
                (0.5, max(ref, pred) * 1.36), ha="center", fontsize=7,
                va="top")

    # (c) C-local reference: Ebao's own calibration information works
    ax = axes[2]
    c_rows = infos[(infos.case == "C") & (infos.group == "scalar_increment")]
    s_rows = infos[(infos.case == "C") & (infos.group == "spread")]

    def rmse(group: pd.DataFrame, method: str) -> float:
        row = group[group.method == method]
        return float(row["rmse"].iloc[0])

    scalar_vals = [
        rmse(c_rows, "zero_correction"),
        rmse(c_rows, "global_calibration_mean"),
        rmse(c_rows, "terrain_roughness_equation"),
        rmse(c_rows, "same_cell_historical_mean"),
    ]
    spread_vals = [
        rmse(s_rows, "global_calibration_mean"),
        rmse(s_rows, "joint_terrain_equation"),
        rmse(s_rows, "same_cell_historical_mean"),
    ]
    lab_s = ["none\n(zero)", "domain\nmean", "static\nterrain", "cell\nhistory"]
    lab_p = ["domain\nmean", "static\nterrain", "cell\nhistory"]
    xs1 = np.arange(4)
    xs2 = np.arange(3) + 5.0
    ax.bar(xs1, scalar_vals, 0.7, color="#c6dbef", edgecolor="black",
           linewidth=0.4)
    ax.bar(xs2, spread_vals, 0.7, color="#fdae6b", edgecolor="black",
           linewidth=0.4)
    for xi, v in list(zip(xs1, scalar_vals)) + list(zip(xs2, spread_vals)):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=6.3)
    ax.set_xticks(list(xs1) + list(xs2))
    ax.set_xticklabels(lab_s + lab_p, fontsize=6.4)
    ax.set_ylabel("Evaluation RMSE (m\\,s$^{-1}$)")
    ax.set_title("(c) Ebao-local reference (own calibration)")
    ax.set_ylim(0, max(max(scalar_vals), max(spread_vals)) * 1.25)
    ax.axvline(3.85, color="grey", linewidth=0.6, linestyle=":")
    ax.text(1.5, ax.get_ylim()[1] * 0.97, "$\\Delta_{\\rm scalar}$",
            ha="center", va="top", fontsize=7.5)
    ax.text(6.0, ax.get_ylim()[1] * 0.97, "spread $S$", ha="center",
            va="top", fontsize=7.5)

    fig.tight_layout(w_pad=1.6)
    out = out_dir() / "Fig3_persistence_portability.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    if mc_config.GENERATION != "20260905":
        raise RuntimeError(
            f"Comment figures require generation 20260905, got {mc_config.GENERATION}")
    paths = [fig1(), fig2(), fig3()]
    for p in paths:
        print("WROTE", p)


if __name__ == "__main__":
    main()
