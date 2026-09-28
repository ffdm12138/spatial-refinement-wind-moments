"""Diagnostic figures (PLAN §30) — QA only, never manuscript art.

Every figure carries the watermark "DIAGNOSTIC ONLY — NOT MANUSCRIPT ART"
plus a red PLACEHOLDER banner wherever Case B/C data is shown while those
cases are placeholders.  Output: extension diagnostic_figures/ as PNG.
source/figures/, make_figures_npj.py and manuscript assets are never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402

FDIR = C.ext_dir("diagnostic_figures")
COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#9467bd"}


def _brand(fig, cases) -> None:
    fig.text(0.5, 0.5, "DIAGNOSTIC ONLY — NOT MANUSCRIPT ART",
             fontsize=22, color="0.75", alpha=0.35, ha="center", va="center",
             rotation=28, zorder=1000)
    blocked = [c for c in cases if not C.case_metadata(c)["science_eligible"]]
    if blocked:
        fig.text(0.5, 0.965,
                 f"NON-SCIENCE / BLOCKED DEPENDENCY (case {', '.join(blocked)})",
                 fontsize=10, color="crimson", ha="center", va="top",
                 fontweight="bold")


def _save(fig, name: str, cases) -> None:
    _brand(fig, cases)
    fig.savefig(FDIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    C.write_json(FDIR / f"{name}.metadata.json", {
        "artifact": name,
        "artifact_kind": "diagnostic_only_not_manuscript_art",
        "watermark": "DIAGNOSTIC ONLY — NOT MANUSCRIPT ART",
    }, cases=cases)
    print(f"  wrote diagnostic_figures/{name}")


def fig01_terrain(cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.2))
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        hgt = C.load_case_hgt(case)
        im = ax.imshow(hgt, origin="lower", cmap="terrain")
        ax.set_title(f"Case {case} ({C.CASES[case]['label']}) HGT\n"
                     f"range {hgt.min():.0f}-{hgt.max():.0f} m")
        for k in range(1, 6):
            ax.axhline(k * 27 - 0.5, color="k", lw=0.4, alpha=0.5)
            ax.axvline(k * 27 - 0.5, color="k", lw=0.4, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "diag01_terrain_overview.png", cases)


def fig02_core_moments(cases):
    m2 = pd.read_csv(C.ext_dir("tables") / "M2_representation_loss.csv")
    moms = ["delta_dyn_domain_mean", "J_domain_mean", "delta_scalar_domain_mean",
            "S_domain_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, window in zip(axes, ("calibration", "evaluation")):
        sub = m2[m2.window == window]
        x = np.arange(len(moms))
        w = 0.8 / len(cases)
        for i, case in enumerate(cases):
            row = sub[sub.case == case].iloc[0]
            ax.bar(x + i * w, [row[m] for m in moms], w,
                   label=f"{case}", color=COLORS[case])
        ax.set_xticks(x + w, ["dyn", "J", "scalar", "S"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{window} domain means")
        ax.set_ylabel("m/s")
        ax.legend()
    _save(fig, "diag02_core_moments.png", cases)


def fig03_recovery(cases):
    m5 = pd.read_csv(C.ext_dir("tables") / "M5_geometric_recovery.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    models = ["bilinear", "bicubic", "bilinear_constrained", "bicubic_constrained"]
    for ax, q in zip(axes, ("Q_J", "Q_spread")):
        x = np.arange(len(models))
        w = 0.8 / len(cases)
        for i, case in enumerate(cases):
            sub = m5[(m5.case == case) & (m5.group == "scalar_increment")]
            vals = [float(sub[sub.model == m][q].iloc[0]) for m in models]
            ax.bar(x + i * w, vals, w, label=case, color=COLORS[case])
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x + w, models, rotation=20, fontsize=8)
        ax.set_title(f"{q} (evaluation 670)")
        ax.legend()
    _save(fig, "diag03_interpolation_recovery.png", cases)


def fig04_windstate_ab(cases):
    use = [c for c in ("A", "B") if c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for case in use:
        ts = pd.read_csv(C.case_dir(case) / "wind_state_timeseries.csv")
        axes[0].hist(ts.speed, bins=40, alpha=0.5, label=case,
                     color=COLORS[case], density=True)
        ws = json.loads((C.case_dir(case) / "wind_state_summary.json")
                        .read_text(encoding="utf-8"))
        freq = ws["calibration"]["sector_frequencies"]
        axes[1].plot(range(8), freq, "o-", label=f"{case} cal",
                     color=COLORS[case])
    axes[0].set_xlabel("d05 domain-mean speed (m/s)")
    axes[0].set_title("A vs B speed distributions (retained window)")
    axes[0].legend()
    axes[1].set_xticks(range(8), C.SECTOR_LABELS)
    axes[1].set_title("direction-sector frequencies")
    axes[1].legend()
    _save(fig, "diag04_windstate_A_vs_B.png", use)


def fig05_descriptors_bc(cases):
    use = [c for c in ("B", "C") if c in cases]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, d in zip(axes, ("r_sigma", "r_iqr", "lambda_iso")):
        for case in use:
            v = pd.read_csv(C.case_dir(case) / "terrain_descriptors.csv")[d]
            ax.hist(v, bins=12, alpha=0.55, label=case, color=COLORS[case])
        rng = C.frozen_a_calibration_ranges()[d if d != "r_iqr" else "r_iqr"]
        ax.axvline(rng[0], color="k", ls=":", lw=1)
        ax.axvline(rng[1], color="k", ls=":", lw=1,
                   label="A calibration range")
        ax.set_title(d)
        ax.legend(fontsize=7)
    _save(fig, "diag05_terrain_descriptors_B_vs_C.png", use)


def _transfer_scatter(ax, cell_csv: Path, model: str, color: str, label: str):
    df = pd.read_csv(cell_csv)
    sub = df[df.model == model]
    ax.scatter(sub.observed_eval_mean, sub.prediction, s=22, color=color,
               label=label, alpha=0.8)
    out = sub[~sub.in_range.astype(bool)]
    if len(out):
        ax.scatter(out.observed_eval_mean, out.prediction, s=60,
                   facecolor="none", edgecolor="red", lw=1.2,
                   label="extrapolating cells")
    lo = min(sub.observed_eval_mean.min(), sub.prediction.min())
    hi = max(sub.observed_eval_mean.max(), sub.prediction.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
    ax.set_xlabel("observed evaluation cell mean")
    ax.set_ylabel("transferred prediction")
    ax.legend(fontsize=7)


def fig06_transfer_ab(cases):
    if "B" not in cases:
        return
    p = C.ext_dir("cross_case") / "transfer_A_to_B_cell_means.csv"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    _transfer_scatter(axes[0], p, "A_terrain_delta", COLORS["A"],
                      "A frozen delta eq -> B")
    axes[0].set_title("T1: A terrain -> B (delta_scalar)")
    _transfer_scatter(axes[1], p, "A_same_cell_historical_mean_scalar",
                      "#2ca02c", "A history -> B")
    axes[1].set_title("T2: A same-cell history -> B (delta_scalar)")
    _save(fig, "diag06_temporal_transfer_A_to_B.png", ["A", "B"])


def fig07_transfer_bc(cases):
    if not {"B", "C"}.issubset(set(cases)):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    _transfer_scatter(axes[0], C.ext_dir("cross_case") / "transfer_B_to_C_cell_means.csv",
                      "B_cal_delta_vs_r_sigma", COLORS["B"], "B-cal eq -> C")
    axes[0].set_title("T3: B calibration -> C (delta_scalar)")
    _transfer_scatter(axes[1], C.ext_dir("cross_case") / "transfer_A_to_C_cell_means.csv",
                      "A_terrain_delta", COLORS["A"], "A frozen eq -> C")
    axes[1].set_title("T4: A frozen -> C (delta_scalar)")
    _save(fig, "diag07_spatial_transfer_to_C.png", ["A", "B", "C"])


def fig08_scale(cases):
    m4 = pd.read_csv(C.ext_dir("tables") / "M4_scale_dependence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, q in zip(axes, ("J_norm_domain", "S_norm_domain")):
        for case in cases:
            sub = m4[(m4.case == case) & (m4.window == "evaluation")].sort_values("scale_m")
            ax.plot(sub.scale_m, sub[q], "o-", label=case, color=COLORS[case])
        ax.set_xscale("log")
        ax.set_xlabel("target scale (m)")
        ax.set_title(f"{q} (evaluation)")
        ax.legend()
    _save(fig, "diag08_scale_dependence.png", cases)


def fig09_infosources(cases):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, grp in zip(axes, ("scalar_increment", "spread")):
        rows = []
        for case in cases:
            df = pd.read_csv(C.case_dir(case) / "information_sources.csv")
            sub = df[df.group == grp][["method", "rmse"]]
            sub["case"] = case
            rows.append(sub)
        allr = pd.concat(rows)
        methods = allr[allr.case == cases[0]].method.tolist()
        x = np.arange(len(methods))
        w = 0.8 / len(cases)
        for i, case in enumerate(cases):
            vals = [float(allr[(allr.case == case) & (allr.method == m)].rmse.iloc[0])
                    for m in methods]
            ax.bar(x + i * w, vals, w, label=case, color=COLORS[case])
        ax.set_xticks(x + w, methods, rotation=25, fontsize=7, ha="right")
        ax.set_title(f"{grp}: evaluation cell-mean RMSE")
        ax.set_ylabel("m/s")
        ax.legend()
    _save(fig, "diag09_information_sources.png", cases)


def fig10_spinup(cases):
    sp = pd.read_csv(C.ext_dir("audit") / "spinup_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["6h", "12h", "970min", "18h", "24h"]
    for ax, q in zip(axes, ("J_retained", "S_retained")):
        for case in cases:
            sub = sp[sp.case == case].set_index("cutoff").loc[order]
            ax.plot(range(len(order)), sub[q], "o-", label=case,
                    color=COLORS[case])
        ax.axvline(order.index("970min"), color="k", ls=":", lw=1)
        ax.set_xticks(range(len(order)), order)
        ax.set_title(f"{q} vs spin-up cutoff (primary = 970min)")
        ax.legend()
    _save(fig, "diag10_spinup_sensitivity.png", cases)


def main(cases: list[str]) -> None:
    print("[diag] rendering diagnostic figures")
    fig01_terrain(cases)
    fig02_core_moments(cases)
    fig03_recovery(cases)
    fig04_windstate_ab(cases)
    fig05_descriptors_bc(cases)
    fig06_transfer_ab(cases)
    fig07_transfer_bc(cases)
    fig08_scale(cases)
    fig09_infosources(cases)
    fig10_spinup(cases)
    C.run_receipt("diag_figures", {"cases": cases})


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
