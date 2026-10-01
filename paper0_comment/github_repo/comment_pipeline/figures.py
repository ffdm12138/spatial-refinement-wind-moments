"""Render Figs. 1-3 of the Comment directly from the released archives.

The rendering implements the released figures' *own* quantities:

    Fig.1  three-state accounting; domain/window means; held-out E_vec vs J;
           held-out cell-time scatter
    Fig.2  exact counterexamples; J vs sigma_perp^2/(2 S_v) with a 650-point
           fixed-seed display subsample; rank agreement J vs J_2nd vs sigma_s
    Fig.3  coarse-vector conservation RMSE; recovery ratios Q_J, Q_sigma_s

Style is presentation-only and does not enter any reported number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullFormatter

from . import contract as C
from .states import ang_diff_deg, correlations

C3 = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}
CCASE = {"A": "#005b96", "B": "#D97706", "C": "#059669"}
CASE_LBL = {"A": "A  Yakou Sep", "B": "B  Yakou Dec", "C": "C  Ebao Dec"}
M_MARK = {"bilinear": "o", "bicubic": "s"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "pdf.fonttype": 42,
})


def rng():
    return np.random.default_rng(C.PLOT_SEED)


def _save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=180)


def render_case(case: str, m: dict, iv: dict, out) -> None:
    """Per-case diagnostic figure, mirroring Fig.1's (c)/(d) content."""
    sl = slice(*C.HELDOUT)
    J = m["J"][sl].ravel()
    e = np.hypot(m["u_bar"][sl] - m["u02p"][sl],
                 m["v_bar"][sl] - m["v02p"][sl]).ravel()
    idx = rng().choice(J.size, min(1500, J.size), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].scatter(J[idx], e[idx], s=4, alpha=0.25, color=C3[case], linewidths=0)
    lim = (0, max(J[idx].max(), e[idx].max()) * 1.05)
    axes[0].plot(lim, lim, "--", color="#64748B", lw=0.8)
    # The x axis is the aggregation gap J, not Delta_mag: the title must say so.
    axes[0].set_xlabel(r"$J$ (m s$^{-1}$)")
    axes[0].set_ylabel(r"$E_{\rm vec}$ (m s$^{-1}$)")
    axes[0].set_title(f"({case}) Held-out $J$ vs $E_{{\\rm vec}}$",
                      loc="left", fontweight="bold")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)

    # Direction statistics exclude cell-times where either mean-vector speed is
    # below the Methods threshold, matching the reported median d_theta.
    dth = ang_diff_deg(m["u_bar"][sl], m["v_bar"][sl],
                       m["u02p"][sl], m["v02p"][sl])
    mask = np.minimum(m["U_vec"][sl], m["U02_mag"][sl]) >= C.SPEED_MASK
    kept = dth[mask]
    axes[1].hist(kept, bins=80, color=C3[case], alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"coarse--aggregate direction difference (deg)")
    axes[1].set_ylabel("cell--times (speed-masked)")
    axes[1].set_title(f"({case}) Direction consistency", loc="left",
                      fontweight="bold")
    fig.tight_layout()
    _save(fig, out, f"Fig_case_{case}")
    plt.close(fig)


def render_panels(summary_rows: list[dict], out, moments: dict | None = None,
                  case_mean_scatter: bool = True) -> None:
    """Fig.1, all four panels, mirroring the released figure.

    (a) three-state accounting schematic
    (b) where the discrepancy comes from (analysis-window decomposition)
    (c) resolved vector consistency vs the gap
    (d) Delta_total vs J over held-out cell-times

    Panel (d) plots real points from the held-out window ``[3650, 4320)``; its
    coverage is asserted by the caller's check suite.
    """
    from matplotlib.patches import Rectangle

    df = pd.DataFrame(summary_rows)
    fig = plt.figure(figsize=(7.2, 5.15))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.30,
                          left=0.085, right=0.975, top=0.93, bottom=0.08)

    # (a) schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_title("a) Three-state accounting", loc="left", fontweight="bold")
    bw = 0.26
    for name, formula, x in (
            ("native coarse", r"$S_c=|\mathbf{U}_c|$", 0.02),
            ("fine vector\naggregate", r"$S_v=|\overline{\mathbf{u}_f}|$", 0.37),
            ("fine scalar\nmean", r"$S_s=\overline{|\mathbf{u}_f|}$", 0.72)):
        ax.add_patch(Rectangle((x, 0.60), bw, 0.24, fc="#EDF2F7", ec="#333333",
                               lw=0.8, transform=ax.transAxes))
        ax.text(x + bw / 2, 0.766, name, transform=ax.transAxes, ha="center",
                va="center", fontsize=6.0)
        ax.text(x + bw / 2, 0.658, formula, transform=ax.transAxes, ha="center",
                va="center", fontsize=9.5)
    for x0, x1, lab, col in (
            (0.28, 0.37, r"$\Delta_{\rm mag}=S_v-S_c$", "#D55E00"),
            (0.63, 0.72, r"$J=S_s-S_v\geq0$", "#0072B2")):
        ax.add_patch(FancyArrowPatch((x0, 0.72), (x1, 0.72),
                                     transform=ax.transAxes, arrowstyle="-|>",
                                     mutation_scale=10, color=col, lw=1.4))
        ax.text((x0 + x1) / 2, 0.47, lab, transform=ax.transAxes, ha="center",
                fontsize=7.8, color=col)
    ax.text(0.5, 0.31, r"$\Delta_{\rm total}=\Delta_{\rm mag}+J$  (exact identity)",
            transform=ax.transAxes, ha="center", fontsize=8.5)
    ax.text(0.5, 0.12, "Intermediate state makes coarse\u2013aggregate\n"
                       "consistency testable",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.8,
            color="#64748B", style="italic")

    # (b) decomposition dumbbells
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("b) Where the discrepancy comes from", loc="left",
                 fontweight="bold")
    # Six groups: each case contributes its analysis-window and held-out means,
    # so the panel shows the same decomposition in both windows rather than
    # silently reporting one window per case.
    base = list(df["case"])
    dfa = df.assign(window="analysis")
    dfe = df.assign(window="eval")
    long = pd.concat([dfa, dfe], ignore_index=True)
    x = np.arange(len(long))
    ax.axhline(0, color="k", lw=0.6, ls=":")
    Dm = long["dt_mean_eval"].where(long["window"] == "eval",
                                    long["dt_mean"]).values
    Jm = long["J_mean_eval"].where(long["window"] == "eval",
                                   long["J_mean"]).values
    DMm = long["dmag_mean_eval"].where(long["window"] == "eval",
                                       long["dmag_mean"]).values
    for xi in range(len(long)):
        ax.plot([xi, xi], [Dm[xi], Jm[xi]], color="#999999", lw=1.5, zorder=2)
    ax.plot(x, Dm, "o", ms=5.5, color="#666666", zorder=3,
            label=r"$\Delta_{\rm total}$")
    ax.plot(x, Jm, "o", ms=5.0, color="#0072B2", zorder=4, label=r"$J$")
    ax.plot(x, DMm, "d", ms=4.0, color="#D55E00", zorder=3,
            label=r"$\Delta_{\rm mag}$")
    for xi, (d, j) in enumerate(zip(Dm, Jm)):
        if abs(d) > 1e-9:
            ax.text(xi, max(d, j) + 0.016, f"{j/d:.2f}", ha="center",
                    fontsize=6.0, color="#0072B2", fontweight="bold")
    ax.set_xticks(x, [f"{b}\n{'analysis' if w == 'analysis' else 'held-out'}"
                      for b, w in zip(base * 2,
                                      ["analysis"] * len(base) + ["eval"] * len(base))])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel(r"domain mean (m s$^{-1}$)")
    ax.set_ylim(-0.04, 0.52)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2,
              handletextpad=0.35)

    # (c) vector consistency vs gap
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("c) Resolved vector consistency vs the gap", loc="left",
                 fontweight="bold")
    xx = np.arange(len(df))
    w = 0.38
    # Panel (c) is the held-out evaluation comparison, so every quantity must
    # come from the held-out window: J_eval, E_vec_eval and the ratio of the two.
    # Using the analysis-window J_mean as the denominator here would report a
    # cross-window ratio (A 3.0% instead of 2.6%, C 2.1% instead of 1.4%).
    ax.bar(xx - w / 2, df["J_mean_eval"], w, color="#0072B2", label=r"$J$")
    ax.bar(xx + w / 2, df["E_vec_mean"], w, color="#D55E00",
           label=r"$E_{\rm vec}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    for xi in range(len(df)):
        pct = df["E_vec_over_J_eval_percent"].iloc[xi]
        ax.text(xi + w / 2, df["E_vec_mean"].iloc[xi] * 1.35, f"{pct:.1f}%",
                ha="center", fontsize=6.5, color="#D55E00")
        ax.text(xi - w / 2, df["J_mean_eval"].iloc[xi] * 1.35,
                f"{df['J_mean_eval'].iloc[xi]:.2f}", ha="center", fontsize=6.5,
                color="#0072B2")
    ax.set_xticks(xx, [CASE_LBL[c] for c in df["case"]])
    ax.set_ylabel(r"domain mean (m s$^{-1}$, log)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2,
              handletextpad=0.4, labelspacing=0.3)
    ax.text(0.03, 0.92, r"median $\Delta\theta<0.02^\circ$ (masked)",
            transform=ax.transAxes, fontsize=7, style="italic")

    # (d) Delta_total vs J over held-out cell-times
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title(r"d) $\Delta_{\rm total}$ vs $J$ (held-out states)",
                 loc="left", fontweight="bold")
    displayed = []
    n_points = 0
    if moments:
        for case, m in moments.items():
            sl = slice(*C.HELDOUT)
            dt = m["delta_scalar"][sl].ravel()
            J = m["J"][sl].ravel()
            idx = rng().choice(dt.size, min(1500, dt.size), replace=False)
            ax.scatter(J[idx], dt[idx], s=2.5, alpha=0.35, color=C3[case],
                       linewidths=0, label=CASE_LBL[case])
            n_points += idx.size
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
    ax.set_xlabel(r"$J$ (m s$^{-1}$)")
    ax.set_ylabel(r"$\Delta_{\rm total}$ (m s$^{-1}$)")
    ax.legend(frameon=False, loc="upper left", markerscale=4, handlelength=1.0)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fig1_panel_d_points.txt").write_text(str(n_points), encoding="utf-8")
    _save(fig, out, "Fig1_three_state_multicase")
    plt.close(fig)


def render_mechanism(mechs: list[dict], out, moments: dict | None = None,
                     n_subsample: int = 650) -> None:
    """Fig.2, all three panels, mirroring the released figure.

    (a) exact counterexamples
    (b) J vs the transverse predictor, with a ``n_subsample``-point fixed-seed
        display subsample per case (this is the panel that previously carried
        an empty scatter).
    (c) rank agreement of J with the transverse, parallel and scalar-spread
        predictors
    """
    mech = pd.DataFrame(mechs)
    fig = plt.figure(figsize=(7.2, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 1.0], wspace=0.34,
                          left=0.055, right=0.985, top=0.86, bottom=0.155)

    # (a) exact counterexamples
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0.245, 0.90, "Opposing", ha="center", fontsize=7.0, fontweight="bold",
            transform=ax.transAxes, color="#005b96")
    ax.text(0.245, 0.815, "equal speeds", ha="center", fontsize=7.0,
            color="#555555", transform=ax.transAxes)
    fig.canvas.draw()
    bb = ax.get_window_extent()
    L_ax = 0.135
    Lx, Ly = L_ax, L_ax * (bb.width / bb.height)
    origin = (0.245, 0.50)
    for ang in (90, 0, 180, 270):
        rad = np.radians(ang)
        ax.add_patch(FancyArrowPatch(
            origin, (origin[0] + Lx * np.cos(rad), origin[1] + Ly * np.sin(rad)),
            transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7.0,
            color="#005b96", lw=1.4))
    ax.plot([origin[0]], [origin[1]], "o", ms=3.0, color="#005b96",
            transform=ax.transAxes, zorder=4)
    ax.text(0.245, 0.175, r"$J=1>0$" + "\n" + r"$\sigma_s=0$", ha="center",
            va="center", fontsize=7.0, color="#C2410C", transform=ax.transAxes,
            linespacing=1.30)
    ax.text(0.755, 0.90, "Collinear", ha="center", fontsize=7.0,
            fontweight="bold", transform=ax.transAxes, color="#6497b1")
    ax.text(0.755, 0.815, "differing speeds", ha="center", fontsize=7.0,
            color="#555555", transform=ax.transAxes)
    for y_i, L_i in zip([0.665, 0.585, 0.505, 0.425],
                        [0.055, 0.095, 0.135, 0.175]):
        ax.add_patch(FancyArrowPatch((0.640, y_i), (0.640 + L_i, y_i),
                                     transform=ax.transAxes, arrowstyle="-|>",
                                     mutation_scale=6.5, color="#6497b1", lw=1.4))
    ax.text(0.755, 0.175, r"$J=0$" + "\n" + r"$\sigma_s>0$", ha="center",
            va="center", fontsize=7.0, color="#C2410C", transform=ax.transAxes,
            linespacing=1.30)
    ax.text(0.5, 0.045, r"$\sigma_s$ is neither necessary nor sufficient" + "\n" +
                        r"for nonzero $J$", ha="center", va="center", fontsize=7.0,
            style="italic", color="#334155", transform=ax.transAxes)

    # (b) J vs transverse predictor, with the real display subsample
    ax = fig.add_subplot(gs[0, 1])
    lim = (6e-5, 4.0)
    ax.plot(lim, lim, "--", color="#64748B", lw=0.8, zorder=2)
    ax.text(1.15, 0.55, "$y=x$", fontsize=7.0, color="#64748B", rotation=45,
            ha="center", va="center")
    mk_case = {"A": "o", "B": "s", "C": "^"}
    n_plotted = 0
    if moments:
        for case in C.CASES:
            m = moments[case]
            an = slice(C.ANALYSIS[0], None)
            J = m["J"][an]
            sv = m["U_vec"][an]
            pred = m["var_perp"][an] / (2 * sv)
            # The panel displays the same masked sample the reported
            # correlation is computed on: the low-Sv tail is where the
            # approximation's denominator vanishes.
            keep = sv >= C.SPEED_MASK
            Jk, pk = J[keep].ravel(), pred[keep].ravel()
            idx = rng().choice(Jk.size, min(n_subsample, Jk.size), replace=False)
            rho = mech.loc[mech.case == case, "sp_J_perp"].iloc[0]
            ax.scatter(pk[idx], Jk[idx], s=6.0, alpha=0.26, color=C3[case],
                       marker=mk_case[case], linewidths=0,
                       label=f"{CASE_LBL[case].split()[0]} ({rho:.2f})", zorder=3)
            n_plotted += idx.size
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_box_aspect(1)
    ax.grid(True, which="major", ls=":", lw=0.5, color="#CBD5E1", alpha=0.7,
            zorder=1)
    ax.set_xlabel(r"$\sigma_\perp^2/(2\,S_v)\ (\mathrm{m\ s^{-1}})$",
                  fontsize=8.5)
    ax.set_ylabel(r"$J\ (\mathrm{m\ s^{-1}})$", fontsize=8.5)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1e0]))
    ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$",
                        r"$10^{0}$"], fontsize=7.0)
    ax.xaxis.set_minor_formatter(NullFormatter())
    leg = ax.legend(frameon=False, loc="lower right", fontsize=7.0, markerscale=2.4,
                    handletextpad=0.25, borderaxespad=0.25, labelspacing=0.24)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    # (c) rank agreement
    ax = fig.add_subplot(gs[0, 2])
    xx = np.arange(3)
    for name, col, color, marker in (
            (r"Transverse $\sigma_\perp^2/(2S_v)$", "sp_J_perp", "#0072B2", "o"),
            (r"Parallel $\sigma_\parallel^2/(2S_v)$", "sp_J_par", "#56B4E9", "s"),
            (r"Scalar spread $\sigma_s$", "sp_J_S", "#999999", "^")):
        y = mech[col].values
        ax.plot(xx, y, marker=marker, color=color, lw=1.2, ms=4.2, label=name,
                zorder=3)
        for xi, v in zip(xx, y):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.0, color=color,
                        zorder=4)
    ax.set_xticks(xx, [CASE_LBL[c].split()[0] for c in mech["case"]])
    ax.set_box_aspect(1)
    ax.set_ylabel(r"Spearman $\rho_s$ with $J$", fontsize=8.5)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(1.0, color="#64748B", lw=0.6, ls="--", zorder=2, alpha=0.8)
    ax.grid(axis="y", color="#E2E8F0", ls="-", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", fontsize=7.0, handlelength=1.5,
              handletextpad=0.35, labelspacing=0.20, borderaxespad=0.15)

    for _ax, _lab in zip(fig.axes,
                         ["a) Exact counterexamples", "b) Transverse relation",
                          "c) Rank agreement"]):
        fig.text(_ax.get_position().x0, 0.965, _lab, ha="left", va="top",
                 fontsize=8.5, fontweight="bold")

    out.mkdir(parents=True, exist_ok=True)
    (out / "fig2_panel_b_points.txt").write_text(str(n_plotted), encoding="utf-8")
    _save(fig, out, "Fig2_vector_organization_multicase")
    plt.close(fig)


def render_recovery(rows: list[dict], out) -> None:
    """Fig.3 panels (a) and (b)."""
    df = pd.DataFrame(rows)
    methods = ["bilinear", "bicubic", "bilinear_constrained", "bicubic_constrained"]
    labels = ["Ordinary\nbilinear", "Ordinary\nbicubic",
              "Constrained\nbilinear", "Constrained\nbicubic"]
    off = {"A": -0.15, "B": 0.0, "C": 0.15}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.legend(handles=[Line2D([], [], marker="o", color=CCASE[c], ls="", ms=5.5,
                               label=CASE_LBL[c]) for c in C.CASES],
               frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.8)

    for i, mth in enumerate(methods):
        sub = df[df.method == mth]
        base = "bicubic" if "bicubic" in mth else "bilinear"
        for _, r in sub.iterrows():
            v = r["L1_metric_rmse"]
            if v is None or not np.isfinite(v):
                # Never fabricate a position: mark the method as unavailable.
                axes[0].scatter(i + off[r["case"]], 1e-16, color=CCASE[r["case"]],
                                marker="x", s=30, alpha=0.9, linewidths=1.0, zorder=4)
                continue
            axes[0].scatter(i + off[r["case"]], v,
                            color=CCASE[r["case"]], marker=M_MARK[base], s=32, alpha=0.9)
    axes[0].axvline(1.5, color="#CBD5E1", lw=0.8, ls="--")
    axes[0].set_yscale("log"); axes[0].set_ylim(1e-16, 8)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labels, fontsize=6.4)
    axes[0].set_xlim(-0.5, 3.5)
    axes[0].set_ylabel(r"Coarse-vector RMSE (m s$^{-1}$)")
    axes[0].set_title("a) Coarse-vector conservation", loc="left", fontweight="bold")
    axes[0].grid(axis="y", ls=":", lw=0.5, color="#E2E8F0")

    for i, mth in enumerate(methods):
        sub = df[df.method == mth]
        for _, r in sub.iterrows():
            for col, marker, dx in (("Q_J", "o", -0.10), ("Q_S", "^", 0.10)):
                axes[1].scatter(i + off[r["case"]] + dx, r[col], color=CCASE[r["case"]],
                                marker=marker, s=26, alpha=0.9)
    axes[1].axhline(1.0, color="#64748B", lw=0.7, ls="--")
    axes[1].axhline(0.5, color="#CBD5E1", lw=0.6, ls=":")
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(labels, fontsize=6.4)
    axes[1].set_xlim(-0.5, 3.5); axes[1].set_ylim(0, 1.2)
    axes[1].set_ylabel(r"Recovery ratio $Q_M$")
    axes[1].set_title("b) Subgrid-statistic recovery", loc="left", fontweight="bold")
    axes[1].grid(axis="y", ls=":", lw=0.5, color="#E2E8F0")
    axes[1].legend(handles=[Line2D([], [], marker="o", color="#334155", ls="", ms=5,
                                   label=r"Aggregation gap $J$"),
                            Line2D([], [], marker="^", color="#334155", ls="", ms=5,
                                   label=r"Scalar spread $\sigma_s$")],
                   frameon=False, loc="upper left", fontsize=6.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "Fig3_two_level_recovery_multicase")
    plt.close(fig)
