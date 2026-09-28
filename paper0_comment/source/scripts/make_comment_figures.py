#!/usr/bin/env python
"""Render the three live Comment figures from unchanged pre-existing evidence.
Presentation-only successor (2026-09-10 editorial pass): never writes to its inputs.
Call through finalize_submission.py --comment --rebuild-figures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "research/three_state_diagnostic_20260906"
OUT = HERE / "outputs"
FIGD = ROOT / "latex_paper0_npjcomment/manuscript/figures_v2"
PREVIEW = Path(os.environ.get("NPJ_COMMENT_SCRATCH_DIR", str(ROOT / "tmp/pdfs/comment_final_20260908")))
QA = Path(os.environ.get("NPJ_COMMENT_QA_DIR", str(ROOT / "qa/comment_finalization_20260908")))
EXT = ROOT / "data_paper0/extension_multicase_20260905"
CASES = {"A": "case_A_yakou0924", "B": "case_B_yakou1201", "C": "case_C_ebao1201"}
CASE_LBL = {"A": "A  Yakou Sep", "B": "B  Yakou Dec", "C": "C  Ebao Dec"}
# Okabe-Ito
C3 = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}
CCASE = {"A": "#005b96", "B": "#D97706", "C": "#059669"}
CGREY = "#666666"
WIN = {"cal": "analysis", "eval": "held-out"}  # display labels only; data column unchanged

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "pdf.fonttype": 42,
})


def load_moments(case):
    z = np.load(EXT / CASES[case] / "moments_cell_time.npz")
    return {k: z[k].astype(np.float64) for k in z.files if not k.startswith("_")}


# Transverse-diagnostic speed mask (Methods).  Must match
# comment_pipeline.contract.SPEED_MASK and diag_run.SPEED_MASK, otherwise the
# displayed subsample and the printed legend rho describe different samples.
SPEED_MASK = 1.0  # m/s


def rng():
    return np.random.default_rng(20260906)


# ------------------------------------------------------------------ FIG 1
def fig1():
    summ = pd.read_csv(OUT / "three_state_summary.csv")
    fig = plt.figure(figsize=(7.2, 5.15), dpi=100)
    # Reserve physical space for the mathtext xlabel and log-axis ylabel.
    # Keep the save guard active across Agg/PDF font metrics and preview DPIs.
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.30,
                          left=0.10, right=0.975, top=0.93, bottom=0.115)

    # (a) schematic: four layers, content group geometrically centred
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_title("(a) Three-state accounting", loc="left", fontweight="bold")
    bw = 0.26
    boxes = [("native coarse", r"$S_c=|\mathbf{U}_c|$", 0.02),
             ("fine vector\naggregate", r"$S_v=|\overline{\mathbf{u}_f}|$", 0.37),
             ("fine scalar\nmean", r"$S_s=\overline{|\mathbf{u}_f|}$", 0.72)]
    for name, formula, x in boxes:
        ax.add_patch(Rectangle((x, 0.60), bw, 0.24, fc="#EDF2F7", ec="#333333",
                               lw=0.8, transform=ax.transAxes))
        ax.text(x + bw / 2, 0.766, name, transform=ax.transAxes, ha="center",
                va="center", fontsize=6.0)
        ax.text(x + bw / 2, 0.658, formula, transform=ax.transAxes, ha="center",
                va="center", fontsize=9.5)
    for x0, x1, lab, col in ((0.28, 0.37, r"$\Delta_{\rm mag}=S_v-S_c$", "#D55E00"),
                             (0.63, 0.72, r"$J=S_s-S_v\geq0$", "#0072B2")):
        ax.add_patch(FancyArrowPatch((x0, 0.72), (x1, 0.72), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=10, color=col, lw=1.4))
        ax.text((x0 + x1) / 2, 0.47, lab, transform=ax.transAxes, ha="center",
                fontsize=7.8, color=col)
    ax.text(0.5, 0.31, r"$\Delta_{\rm total}=\Delta_{\rm mag}+J$  (exact identity)",
            transform=ax.transAxes, ha="center", fontsize=8.5)
    ax.text(0.5, 0.12, "Intermediate state makes coarse\u2013aggregate\nconsistency testable",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.8,
            color=CGREY, style="italic")

    # (b) decomposition dumbbells
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("(b) Where the discrepancy comes from", loc="left", fontweight="bold")
    labels, Jm, Dm, DMm = [], [], [], []
    for _, r in summ.iterrows():
        labels.append(f"{r['case']}\n{WIN[r['window']]}")
        Jm.append(r["J_mean"]); Dm.append(r["dt_mean"]); DMm.append(r["dmag_mean"])
    x = np.arange(len(labels))
    ax.axhline(0, color="k", lw=0.6, ls=":")
    for xi in range(len(labels)):
        ax.plot([xi, xi], [Dm[xi], Jm[xi]], color="#999999", lw=1.5, zorder=2)
    ax.plot(x, Dm, "o", ms=6.5, color="#666666", zorder=3, label=r"$\Delta_{\rm total}$")
    ax.plot(x, Jm, "o", ms=5.5, color="#0072B2", zorder=4, label=r"$J$")
    ax.plot(x, DMm, "d", ms=4.5, color="#D55E00", zorder=3, label=r"$\Delta_{\rm mag}$")
    for xi, (d, j) in enumerate(zip(Dm, Jm)):
        ax.text(xi, max(d, j) + 0.016, f"{j/d:.2f}", ha="center", fontsize=6.5,
                color="#0072B2", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel(r"domain mean (m s$^{-1}$)")
    ax.set_ylim(-0.04, 0.52)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.2,
              handletextpad=0.35)

    # (c) vector consistency vs gap
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("(c) Resolved vector consistency vs the gap", loc="left", fontweight="bold")
    ev = summ[summ.window == "eval"]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, ev.J_mean, w, color="#0072B2", label=r"$J$")
    ax.bar(x + w / 2, ev.E_vec_mean, w, color="#D55E00", label=r"$E_{\rm vec}$")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1)
    for xi, r in ev.reset_index().iterrows():
        pct = 100 * r.E_vec_mean / r.J_mean
        ax.text(xi + w / 2, r.E_vec_mean * 1.35, f"{pct:.1f}%", ha="center",
                fontsize=6.5, color="#D55E00")
        ax.text(xi - w / 2, r.J_mean * 1.35, f"{r.J_mean:.2f}", ha="center",
                fontsize=6.5, color="#0072B2")
    ax.set_xticks(x, [CASE_LBL[c] for c in ev["case"]])
    ax.set_ylabel(r"domain mean (m s$^{-1}$, log)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2,
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.15,
              handletextpad=0.4, labelspacing=0.3)
    ax.text(0.03, 0.92, r"median $\Delta\theta<0.02^\circ$ (masked)",
            transform=ax.transAxes, fontsize=7, style="italic")

    # (d) Delta_total vs J scatter (held-out cell-times)
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title(r"(d) $\Delta_{\rm total}$ vs $J$ (held-out states)", loc="left",
                 fontweight="bold")
    displayed = []
    for case in CASES:
        m = load_moments(case)
        sl = slice(3650, 4320)
        dt = m["delta_scalar"][sl].ravel(); J = m["J"][sl].ravel()
        idx = rng().choice(dt.size, 1500, replace=False)
        ax.scatter(J[idx], dt[idx], s=2.5, alpha=0.35, color=C3[case],
                   linewidths=0, label=CASE_LBL[case])
        displayed.append((J[idx], dt[idx]))
    if displayed:
        xx = np.concatenate([x for x,y in displayed])
        yy = np.concatenate([y for x,y in displayed])
        px = max(float(np.ptp(xx)), 0.1) * 0.04
        py = max(float(np.ptp(yy)), 0.1) * 0.04
        xlim = (min(0., float(xx.min())) - px, float(xx.max()) + px)
        ylim = (min(0., float(yy.min())) - py, float(yy.max()) + py)
        ax.plot([0, min(xlim[1], ylim[1])], [0, min(xlim[1], ylim[1])],
                '--', color='#64748B', lw=0.8, label='$y=x$')
        ax.axhline(0, color='#64748B', lw=0.6, zorder=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(r"$J$ (m s$^{-1}$)"); ax.set_ylabel(r"$\Delta_{\rm total}$ (m s$^{-1}$)")
    ax.legend(frameon=False, loc="upper left", markerscale=4, handlelength=1.0)
    fig.canvas.draw()
    save_checked(fig, "Fig1_three_state_multicase")
    plt.close(fig)


# ------------------------------------------------------------------ FIG 2
def fig2():
    mech = pd.read_csv(OUT / "mechanism_correlations.csv")
    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 1.0], wspace=0.34,
                          left=0.055, right=0.985, top=0.86, bottom=0.155)

    # (a) exact counterexamples
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")

    # left half: four opposing unit vectors from a common origin
    ax.text(0.245, 0.90, "Opposing", ha="center", fontsize=6.4,
            fontweight="bold", transform=ax.transAxes, color="#005b96")
    ax.text(0.245, 0.815, "equal speeds", ha="center", fontsize=5.6,
            color="#555555", transform=ax.transAxes)
    origin = (0.245, 0.50)
    # isotropic length: equal on-canvas length in every direction
    fig.canvas.draw()
    bb = ax.get_window_extent()
    L_ax = 0.135                      # target length as a fraction of axes width
    Lx = L_ax
    Ly = L_ax * (bb.width / bb.height)   # compensate the unequal axes aspect
    for ang in (90, 0, 180, 270):
        rad = np.radians(ang)
        ax.add_patch(FancyArrowPatch(origin, (origin[0] + Lx * np.cos(rad),
                                              origin[1] + Ly * np.sin(rad)),
                                     transform=ax.transAxes, arrowstyle="-|>",
                                     mutation_scale=7.0, color="#005b96", lw=1.4))
    ax.plot([origin[0]], [origin[1]], "o", ms=3.0, color="#005b96",
            transform=ax.transAxes, zorder=4)
    ax.text(0.245, 0.175, r"$J=1>0$" + "\n" + r"$\sigma_s=0$", ha="center",
            va="center", fontsize=7.0, color="#C2410C", transform=ax.transAxes,
            linespacing=1.30)

    # right half: four collinear arrows of differing length (same direction)
    ax.text(0.755, 0.90, "Collinear", ha="center", fontsize=6.4,
            fontweight="bold", transform=ax.transAxes, color="#6497b1")
    ax.text(0.755, 0.815, "differing speeds", ha="center", fontsize=5.6,
            color="#555555", transform=ax.transAxes)
    for y_i, L_i in zip([0.665, 0.585, 0.505, 0.425], [0.055, 0.095, 0.135, 0.175]):
        ax.add_patch(FancyArrowPatch((0.640, y_i), (0.640 + L_i, y_i),
                                     transform=ax.transAxes, arrowstyle="-|>",
                                     mutation_scale=6.5, color="#6497b1", lw=1.4))
    ax.text(0.755, 0.175, r"$J=0$" + "\n" + r"$\sigma_s>0$", ha="center",
            va="center", fontsize=7.0, color="#C2410C", transform=ax.transAxes,
            linespacing=1.30)
    ax.text(0.5, 0.045, r"$\sigma_s$ is neither necessary nor sufficient for nonzero $J$",
            ha="center", va="center", fontsize=5.5, style="italic",
            color="#334155", transform=ax.transAxes)

    # (b) J vs transverse predictor (case by marker shape and case colour)
    ax = fig.add_subplot(gs[0, 1])
    lim = (6e-5, 4.0)
    ax.plot(lim, lim, "--", color="#64748B", lw=0.8, zorder=2)
    ax.text(1.15, 0.55, "$y=x$", fontsize=5.8, color="#64748B", rotation=45,
            ha="center", va="center")
    mk = {"A": "o", "B": "s", "C": "^"}
    for case in CASES:
        m = load_moments(case); an = slice(970, None)
        keep = m["U_vec"][an].ravel() >= SPEED_MASK
        J = m["J"][an].ravel()[keep]
        pred = (m["var_perp"][an] / (2 * m["U_vec"][an])).ravel()[keep]
        idx = rng().choice(J.size, 650, replace=False)
        rho = mech.loc[mech.case == case, "sp_J_perp"].iloc[0]
        ax.scatter(pred[idx], J[idx], s=6.0, alpha=0.26, color=C3[case], marker=mk[case],
                   linewidths=0, label=f"{CASE_LBL[case].split()[0]} ({rho:.2f})", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_box_aspect(1)
    ax.grid(True, which="major", ls=":", lw=0.5, color="#CBD5E1", alpha=0.7, zorder=1)
    ax.set_xlabel(r"$\sigma_\perp^2/(2\,S_v)\ (\mathrm{m\ s^{-1}})$", fontsize=7.2)
    ax.set_ylabel(r"$J\ (\mathrm{m\ s^{-1}})$", fontsize=7.2)
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1e0]))
    ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"],
                       fontsize=6.6)
    ax.xaxis.set_minor_formatter(NullFormatter())
    leg = ax.legend(frameon=False, loc="lower right", fontsize=5.8, markerscale=2.4,
                    handletextpad=0.25, borderaxespad=0.25, labelspacing=0.24)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    # (c) rank agreement (line + marker; colour encodes the statistic)
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(3)
    series = [(r"Transverse $\sigma_\perp^2/(2S_v)$", "sp_J_perp", "#0072B2", "o"),
              (r"Parallel $\sigma_\parallel^2/(2S_v)$", "sp_J_par", "#56B4E9", "s"),
              (r"Scalar spread $\sigma_s$", "sp_J_S", "#999999", "^")]
    for name, col, color, marker in series:
        y = mech[col].values
        ax.plot(x, y, marker=marker, color=color, lw=1.2, ms=4.2, label=name, zorder=3)
        for xi, v in zip(x, y):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=5.4, color=color, zorder=4)
    ax.set_xticks(x, [CASE_LBL[c].split()[0] for c in mech["case"]])
    ax.set_box_aspect(1)
    ax.set_ylabel(r"Spearman $\rho_s$ with $J$", fontsize=7.2)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(1.0, color="#64748B", lw=0.6, ls="--", zorder=2, alpha=0.8)
    ax.grid(axis="y", color="#E2E8F0", ls="-", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", fontsize=5.5, handlelength=1.5,
              handletextpad=0.35, labelspacing=0.20, borderaxespad=0.15)

    # one shared baseline for the three column titles
    TITLE_Y = 0.965
    for _ax, _lab in zip(fig.axes,
                         ["(a) Exact counterexamples",
                          "(b) Transverse relation",
                          "(c) Rank agreement"]):
        _x = _ax.get_position().x0
        fig.text(_x, TITLE_Y, _lab, ha="left", va="top",
                 fontsize=8.5, fontweight="bold")
    save_checked(fig, "Fig2_vector_organization_multicase")
    plt.close(fig)


# ------------------------------------------------------------------ FIG 3
def fig3():
    mat = pd.read_csv(OUT / "two_level_matrix.csv")
    vec_rmse = {case: pd.read_csv(EXT / CASES[case] / "interpolation_metrics.csv")
                .query("group == 'scalar_increment'").set_index("model")
                for case in CASES}
    mat["drift"] = [vec_rmse[r["case"]].loc[r["method"], "E_vec_rmse"] for _, r in mat.iterrows()]

    m_mark = {"bilinear": "o", "bicubic": "s"}
    methods = ["bilinear", "bicubic", "bilinear_constrained", "bicubic_constrained"]
    m_labels = ["Ordinary\nbilinear", "Ordinary\nbicubic",
                "Constrained\nbilinear", "Constrained\nbicubic"]
    case_offsets = {"A": -0.15, "B": 0.0, "C": 0.15}

    fig = plt.figure(figsize=(7.2, 3.05))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.085, right=0.975, top=0.88, bottom=0.245)
    fig.legend(handles=[Line2D([], [], marker="o", color=CCASE[c], ls="", ms=5.5,
                               label=CASE_LBL[c]) for c in ("A", "B", "C")],
               frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.00),
               fontsize=6.8, columnspacing=1.6)

    # (a) coarse-vector conservation error (log axis)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("(a) Coarse-vector conservation", loc="left", fontweight="bold", fontsize=8.2)
    for i, m in enumerate(methods):
        for _, r in mat[mat["method"] == m].iterrows():
            base = "bicubic" if "bicubic" in m else "bilinear"
            ax.scatter(i + case_offsets[r["case"]], r["drift"], color=CCASE[r["case"]],
                       marker=m_mark[base], s=32, zorder=3, alpha=0.9, linewidths=0.6)
    ax.axvline(1.5, color="#CBD5E1", lw=0.8, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 8)
    ax.text(0.75, 2e-3, "conservation fails", ha="center", fontsize=5.8,
            color="#DC2626", fontweight="bold")
    ax.text(2.25, 2e-3, "machine precision", ha="center", fontsize=5.8,
            color="#005b96", fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(m_labels, fontsize=6.4)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel(r"Coarse-vector RMSE (m s$^{-1}$)", fontsize=7.5)
    ax.grid(axis="y", ls=":", lw=0.5, color="#E2E8F0", zorder=0)

    # (b) subgrid-statistic recovery ratios
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("(b) Subgrid-statistic recovery", loc="left", fontweight="bold", fontsize=8.2)
    targets = [("Q_J", "o", -0.10, r"Aggregation gap $J$"),
               ("Q_S", "^", 0.10, r"Scalar spread $\sigma_s$")]
    for i, m in enumerate(methods):
        for _, r in mat[mat["method"] == m].iterrows():
            for col, marker, dx, _lab in targets:
                ax.scatter(i + case_offsets[r["case"]] + dx, r[col], color=CCASE[r["case"]],
                           marker=marker, s=26, zorder=3, alpha=0.9, linewidths=0.5)
    ax.axhline(1.0, color="#64748B", lw=0.7, ls="--", zorder=1)
    ax.text(3.4, 1.035, "equal reference mean", fontsize=6.2, color="#64748B", ha="right")
    ax.axhline(0.5, color="#CBD5E1", lw=0.6, ls=":", zorder=1)
    ax.text(1.5, 0.535, r"all twelve outputs recover $<0.5$ of both reference means",
            ha="center", fontsize=6.0, color=CGREY, style="italic")
    ax.set_xticks(range(4)); ax.set_xticklabels(m_labels, fontsize=6.4)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, 1.2)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"Recovery ratio $Q_M$")
    ax.grid(axis="y", ls=":", lw=0.5, color="#E2E8F0", zorder=0)
    ax.legend(handles=[Line2D([], [], marker="o", color="#334155", ls="", ms=5,
                              label=r"Aggregation gap $J$"),
                       Line2D([], [], marker="^", color="#334155", ls="", ms=5,
                              label=r"Scalar spread $\sigma_s$")],
              frameon=False, loc="upper left", fontsize=6.2, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)
    save_checked(fig, "Fig3_two_level_recovery_multicase")
    plt.close(fig)


# ------------------------------------------------------------------ provenance
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checked(fig, stem):
    FIGD.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Visible non-tick text bounds, supplemented by page-level visual review.
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label]
    failures = []
    for t in texts:
        if t.get_visible() and t.get_text().strip():
            b = t.get_window_extent(renderer)
            if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height:
                failures.append(t.get_text())
    if failures:
        raise ValueError(f"{stem}: text outside canvas: {failures}")
    fig.savefig(FIGD / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(PREVIEW / f"{stem}.png", dpi=150)


def provenance():
    inputs = [OUT / n for n in ("three_state_summary.csv", "vector_consistency_summary.csv",
              "mechanism_correlations.csv", "two_level_matrix.csv", "provenance_recompute.csv")]
    inputs += [EXT / CASES[c] / "moments_cell_time.npz" for c in CASES]
    inputs += [EXT / CASES[c] / "interpolation_metrics.csv" for c in CASES]
    doc = {
        "scope": "presentation only; no new predictions, experiments or fits",
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256(Path(__file__)),
        "inputs": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
        "outputs": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(FIGD.glob("*.pdf"))},
        "conventions": {
            "windows": "analysis [970,3650); held-out eval [3650,4320); inherited exclusion [0,970)",
            "fig3_a_x": "existing E_vec_rmse for all 12 outputs on a log axis (1e-16 to 8); NOT heterogeneous L1_metric",
            "fig3_a_y": "same values as fig3_a_x (conservation error); no fabricated observations",
            "fig3_b_y": "Q_J and Q_S; same domain/window mean denominator as frozen tables",
            "fig3_layout": "two quantitative panels; conceptual flowchart removed as redundant with Fig.1 and text",
            "color_semantics": "case colour is stable within each figure; Fig.2(b) additionally uses case markers",
            "text_bounds": "visible non-tick text stays on figure canvas",
            "input_tables_written": False}}
    (QA / "figure_provenance.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def render_all():
    fig1()
    fig2()
    fig3()
    provenance()
    print("Rendered 3 presentation-only figures from unchanged evidence.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=(1, 2, 3),
                        help="Render only the affected figure; leave other PDFs unchanged.")
    args = parser.parse_args()
    if args.figure is None:
        render_all()
    else:
        {1: fig1, 2: fig2, 3: fig3}[args.figure]()
        provenance()
