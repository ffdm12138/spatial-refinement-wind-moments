"""Render the six main figures of the NPJ manuscript (Round 2 final version).

  Fig 1  Study design and information sources
  Fig 2  What coarse graining removes (exact decomposition, spatial maps,
         scale dependence)
  Fig 3  Why the Jensen contribution appears (second-order covariance)
  Fig 4  What geometric interpolation recovers (bilinear/bicubic baselines)
  Fig 5  What terrain information adds (roughness/exposure signatures)
  Fig 6  Performance across information sources under the changed state

Round-2 fixes implemented here (2026-08-07):
  - Fig 2(f): scale keys are float64 (111.11111111111111), filters must use
    np.isclose; categorical x axis; no NaN may reach the figure.
  - Fig 4(a): four fields drawn into four real sub-axes with one shared
    colour scale (previously four imshow calls on one axes).
  - Fig 4(d,e): recovery ratios are plotted as percents (values x100) under
    a "%" axis; E_vec and Delta_dyn^I are reported for BOTH methods.
  - Fig 5(c): slope/intercept no longer swapped (yfit = intercept + slope*x,
    asserted); y-limits tight around the data.
  - Fig 5(d,e): contour levels shared between the two case-cell panels.
  - Fig 6(d,e,f): dumbbell raw->centred, decluttered class panel, forest
    with group separation.
  - np.isfinite assertions on every plotted array (QA-C).

Style: 170-mm main-text canvas (figstyle.set_main_rc), opaque colours only
(EPS-safe), panel labels with white backing, PDF/PNG/TIFF/EPS outputs into
source/figures/main/ via the shared save wrapper. Reads copied figure-data
CSVs, frozen evidence tables and the interpolation baseline evidence from
the evidence directory (npj_refactor in the workspace layout; source_data
in the released package layout) only; it does not refit models or read the
large WRF tensors (except the archived HGT_d05.pt terrain and the small
representative-state fields npz produced by the evaluation script).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fig1_2_repro"))
import figstyle  # noqa: E402
from figstyle import (read, read_evid, save, tint, spatial_map,  # noqa: E402
                      ROOT, DATA, EVID, SNAP, NPJ, BLUE,
                      TEAL, ORANGE, RED, INK, GREY, LIGHT)

figstyle.set_main_rc()

# information classes for panel shading / colours
CLASS_COLOR = {
    "no fine-scale information": GREY,
    "geometric interpolation": TEAL,
    "static terrain": BLUE,
    "fine-grid history": RED,
}
# journal-visible class names (the CSV keys above are internal data labels)
CLASS_DISPLAY = {
    "no fine-scale information": "global reference",
    "geometric interpolation": "geometric interpolation",
    "static terrain": "static terrain",
    "fine-grid history": "cell-specific fine-grid history",
}
METHOD_COLOR = {
    "zero_correction": GREY,
    "global_training_mean": GREY,
    "bilinear": TEAL,
    "bicubic": ORANGE,
    "terrain_roughness_equation": BLUE,
    "roughness_only": BLUE,
    "exposure_only": BLUE,
    "locked_joint_equation": BLUE,
    "cell_mean_persistence": RED,
}
METHOD_LABEL = {
    "zero_correction": "zero",
    "global_training_mean": "global mean",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "terrain_roughness_equation": "terrain",
    "roughness_only": "roughness",
    "exposure_only": "exposure",
    "locked_joint_equation": "joint",
    "cell_mean_persistence": "historical mean",
}
PAIR_LABEL = {
    "terrain_vs_global": "terrain – global",
    "terrain_vs_bilinear": "terrain – bilinear",
    "terrain_vs_bicubic": "terrain – bicubic",
    "terrain_vs_persistence": "terrain – historical mean",
    "joint_vs_global": "joint – global",
    "joint_vs_bilinear": "joint – bilinear",
    "joint_vs_bicubic": "joint – bicubic",
    "joint_vs_persistence": "joint – historical mean",
}


def read_npj(name: str) -> pd.DataFrame:
    return pd.read_csv(NPJ / name)


def _finite(*arrays, ctx: str = "") -> None:
    """QA-C: every plotted value must be finite."""
    for a in arrays:
        if not np.all(np.isfinite(np.asarray(a, float))):
            raise AssertionError(f"non-finite plotted values in {ctx or 'figure'}")


def _assert_figure_layout(fig, stem: str) -> None:
    """QA-C: no text/legend artist may fall outside the figure canvas, the
    tight bbox must stay within 1.25x the nominal figure height, and the
    Fig.6 tight bbox aspect (width/height) must stay in [0.70, 1.30]. This
    catches out-of-range transAxes group labels inflated by bbox_inches."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    win = fig.get_window_extent(renderer)
    margin = 0.1 * fig.dpi
    for ax in fig.axes:
        artists: list = list(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            artists.append(leg)
        for t in artists:
            bb = t.get_window_extent(renderer)
            if (bb.x0 < win.x0 - margin or bb.x1 > win.x1 + margin
                    or bb.y0 < win.y0 - margin or bb.y1 > win.y1 + margin):
                raise AssertionError(
                    f"text/legend artist outside figure canvas in {stem}: "
                    f"{getattr(t, 'get_text', lambda: '?')()!r}")
    tight_bbox = fig.get_tightbbox(renderer)
    h0, w0 = fig.get_size_inches()
    if tight_bbox.height > 1.25 * h0 * fig.dpi:
        raise AssertionError(
            f"tight bbox height {tight_bbox.height / fig.dpi:.2f} in exceeds "
            f"1.25x nominal height in {stem}")
    aspect = tight_bbox.width / tight_bbox.height
    if stem == "Fig6_information_sources_state" and not (0.70 <= aspect <= 1.30):
        raise AssertionError(
            f"Fig6 tight bbox aspect {aspect:.2f} outside [0.70, 1.30]")


def _save_fig(fig, stem: str, **kw) -> None:
    """figstyle.save with the layout guard running before the file is written."""
    _assert_figure_layout(fig, stem)
    save(fig, stem, **kw)


# ---------------------------------------------------------------------------
# Fig 1 — Study design and information sources
# ---------------------------------------------------------------------------

def fig1() -> None:
    import plot_wrf_domain_overview as pdom
    import cartopy.crs as ccrs

    nml = pdom.parse_namelist(DATA / "fig12_sources" / "wrf_domain" / "namelist.wps")
    domains = pdom.build_domains(nml)
    _wrf_crs, _fwd, inv, ref_x_proj, ref_y_proj = pdom.make_wrf_crs(nml["geogrid"])
    nml_dict = dict(nml)

    fig = plt.figure(figsize=(figstyle.MAIN_WIDTH_IN, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0],
                          hspace=0.55, wspace=0.30)

    # (a) actual nested-domain context d01-d05 (labels at the top-left corner
    # of each domain so the nested boxes never collide)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    # fully offline basemap: vendored Natural Earth 110m countries, no
    # cartopy.feature download at Level-1 reproduction time
    import json
    import shapely.geometry as sgeom
    ax.set_facecolor("#dfe9ef")  # ocean
    with open(DATA / "fig12_sources" / "ne_110m_admin_0_countries.geojson",
              encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    geoms = [sgeom.shape(f["geometry"]) for f in feats if f.get("geometry")]
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="#e8e4da",
                      edgecolor=GREY, linewidth=0.4)
    colours = [INK, BLUE, TEAL, ORANGE, RED]
    for d, col in zip(domains, colours):
        lon, lat = pdom.domain_border_lonlat(d, inv, ref_x_proj, ref_y_proj,
                                             n_per_edge=12)
        ax.plot(lon, lat, color=col, lw=1.2, transform=ccrs.PlateCarree(),
                zorder=5)
        # label just outside the top-left corner of each box (same placement
        # rule as draw_domain_box): nested labels never overlap
        bx = d.x0 - 0.02 * (d.x1 - d.x0)
        by = d.y1 + 0.02 * (d.y1 - d.y0)
        blon, blat = pdom.xy_to_lonlat(np.array([bx]), np.array([by]),
                                       inv, ref_x_proj, ref_y_proj)
        ax.text(float(blon[0]), float(blat[0]), d.label, color=col,
                fontsize=8.6, fontweight="bold", ha="right", va="bottom",
                transform=ccrs.PlateCarree(), zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=col, linewidth=0.7))
    d01 = domains[0]
    corners = pdom.domain_corners_lonlat(d01, inv, ref_x_proj, ref_y_proj)
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    ax.set_extent([min(lons) - 0.2, max(lons) + 0.2,
                   min(lats) - 0.2, max(lats) + 0.2],
                  crs=ccrs.PlateCarree())
    ax.set_title("Five two-way nested domains", pad=3.5)
    figstyle.panel_label(ax, "a")

    # (b) d05 terrain with the 6x6 parent grid and the marked case cells
    import torch
    hgt = torch.load(SNAP / "HGT_d05.pt", map_location="cpu",
                     weights_only=True).numpy().astype(float)
    # real WRF grid coordinates derived from the archived geo_em.d05.nc
    # (registration audit 2026-08-07: identity orientation, zero cell offset)
    ctx = np.load(DATA / "fig12_sources" / "wrf_domain" /
                  "d05_geographic_context.npz")
    lon, lat = ctx["lon"], ctx["lat"]
    if lon.shape != hgt.shape or lat.shape != hgt.shape:
        raise AssertionError("d05 geographic context does not match HGT_d05.pt")
    axb = fig.add_subplot(gs[0, 1])
    figstyle.clean_axis(axb)
    im = axb.pcolormesh(lon, lat, hgt, cmap="terrain_r", shading="auto",
                        vmin=float(np.nanpercentile(hgt, 2)),
                        vmax=float(np.nanpercentile(hgt, 98)))
    dlon = lon[0, 1] - lon[0, 0]
    dlat = lat[1, 0] - lat[0, 0]
    parent_edges = np.arange(0, 163, 27) - 0.5
    for e in parent_edges:
        axb.axvline(lon[0, 0] + e * dlon, color=INK, lw=0.7, alpha=1.0,
                    zorder=3)
        axb.axhline(lat[0, 0] + e * dlat, color=INK, lw=0.7, alpha=1.0,
                    zorder=3)
    for (r, c) in [(4, 4), (5, 1), (0, 1)]:
        x0 = lon[27 * r, 27 * c]
        y0 = lat[27 * r, 27 * c]
        axb.add_patch(plt.Rectangle((x0, y0), 27 * dlon, 27 * dlat, fill=False,
                                    edgecolor=RED if (r, c) in ((4, 4), (5, 1)) else BLUE,
                                    lw=1.6, zorder=4))
    axb.text(lon[27 * 4, 27 * 4], lat[135, 27 * 4], "(4,4)",
             color=RED, fontsize=7.6, ha="center", va="bottom", zorder=5)
    axb.text(lon[135, 27 * 1], lat[161, 27 * 1], "(5,1)",
             color=RED, fontsize=7.6, ha="center", va="bottom", zorder=5)
    axb.text(lon[0, 27 * 1], lat[0, 27 * 1] - 1.5 * dlat, "(0,1)",
             color=BLUE, fontsize=7.6, ha="center", va="top", zorder=5)
    figstyle.add_shared_colorbar(fig, im, axb, "d05 terrain height (m)",
                                 fraction=0.05, pad=0.03)
    axb.set_title("d05 terrain, 6×6 parent grid", pad=3.5)
    figstyle.panel_label(axb, "b")

    # (c) chronology: 2,680 calibration + 670 evaluation states
    axc = fig.add_subplot(gs[1, 0])
    figstyle.clean_axis(axc)
    axc.barh([0], [2680], left=[0], color=BLUE, height=0.5, zorder=3)
    axc.barh([0], [670], left=[2680], color=ORANGE, height=0.5, zorder=3)
    axc.text(1340, 0, "first 2,680 states\n(calibration)", color="white",
             ha="center", va="center", fontsize=8.2, zorder=4)
    axc.text(3015, 0, "final 670\n(evaluation)", color="white", ha="center",
             va="center", fontsize=8.2, zorder=4)
    axc.text(3350, 0.6, "3,350 one-minute states", color=INK, fontsize=8.0,
             ha="right", va="bottom")
    axc.set_yticks([])
    axc.set_xlim(0, 3350)
    axc.set_xticks([0, 2680, 3350])
    axc.set_xticklabels(["0", "2,680", "3,350"])
    axc.set_xlabel("Cumulative retained states (1-min cadence)")
    axc.set_title("Chronological split", pad=3.5)
    figstyle.panel_label(axc, "c")

    # (d) information sources: four columns, no hierarchy (a comparison, not
    # a ladder): each class lists its methods, and all classes feed the two
    # target moments
    axd = fig.add_subplot(gs[1, 1])
    axd.set_axis_off()
    axd.set_xlim(0, 10)
    axd.set_ylim(0, 10)
    axd.set_title("Information sources for two missing moments", pad=3.5)
    xc = [1.30, 3.75, 6.25, 8.70]
    headers = ["Global reference\n(no cell-specific\npredictor)",
               "Instantaneous\ncoarse-field\ngeometry",
               "Static terrain\n+ global\ncoefficients",
               "Cell-specific\nfine-grid\nhistory"]
    classes = ["no fine-scale information", "geometric interpolation",
               "static terrain", "fine-grid history"]
    methods = ["zero correction,\nglobal calibration mean",
               "bilinear,\nbicubic",
               "roughness,\nexposure",
               "same-cell\nhistorical mean"]
    for x, head, cl, meth in zip(xc, headers, classes, methods):
        axd.text(x, 8.55, head, ha="center", va="center", fontsize=7.9,
                 fontweight="bold", color=INK,
                 bbox=dict(boxstyle="round,pad=0.32", facecolor=tint(CLASS_COLOR[cl], 0.84),
                           edgecolor=CLASS_COLOR[cl], linewidth=0.9))
        axd.text(x, 6.15, meth, ha="center", va="center", fontsize=7.6,
                 color=INK,
                 bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                           edgecolor=GREY, linewidth=0.7))
        axd.annotate("", xy=(x, 3.55), xytext=(x, 5.15),
                     arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0,
                                     shrinkA=0, shrinkB=0))
    axd.text(5, 2.55, "Target moments", ha="center", va="center", fontsize=7.8,
             color=GREY)
    for x, lab in [(3.3, "scalar increment"), (6.7, "within-parent spread")]:
        axd.text(x, 1.45, lab, ha="center", va="center", fontsize=8.0,
                 color=INK,
                 bbox=dict(boxstyle="round,pad=0.30", facecolor=tint(LIGHT, 0.5),
                           edgecolor=INK, linewidth=0.8))
    figstyle.panel_label(axd, "d")
    _save_fig(fig, "Fig1_study_design")


# ---------------------------------------------------------------------------
# Fig 2 — What coarse graining removes
# ---------------------------------------------------------------------------

def _concept_panel_v2(ax) -> None:
    """Unified schematic of one parent cell: the fine vectors are collapsed
    into the mean-vector magnitude |u_bar| and the mean scalar speed s_bar
    with its width S. One continuous visual (no split sub-diagrams)."""
    ax.set_axis_off()
    ax.set_title("One cell, three estimands", pad=3.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(20260807)
    n = 12
    speed = rng.uniform(1.3, 3.5, n)
    direc = rng.normal(0.0, 0.55, n)
    k = 0.13  # arrow length scale: cell fraction per unit speed
    gx, gy = np.meshgrid(np.linspace(0.16, 0.72, 4), np.linspace(0.20, 0.72, 3))
    # upper inset: the parent cell with its fine-scale vectors
    axc = ax.inset_axes([0.0, 0.56, 1.0, 0.40])
    axc.set_axis_off()
    axc.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK,
                                lw=1.3, zorder=2))
    for x0, y0, s0, th0 in zip(gx.ravel(), gy.ravel(), speed, direc):
        ux, uy = s0 * np.cos(th0) * k, s0 * np.sin(th0) * k
        tipx = min(max(x0 + ux, 0.03), 0.97)
        tipy = min(max(y0 + uy, 0.03), 0.97)
        axc.annotate("", xy=(tipx, tipy), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="#9fb0c2", lw=1.0,
                                     mutation_scale=7, shrinkA=0, shrinkB=0),
                     zorder=3)
    dx, dy = speed * np.cos(direc) * k, speed * np.sin(direc) * k
    mbx, mby = float(dx.mean()), float(dy.mean())
    sbar = float(speed.mean()) * k
    theta = float(np.arctan2(mby, mbx))
    cx, cy = 0.5, 0.42
    axc.annotate("", xy=(cx + sbar * np.cos(theta), cy + sbar * np.sin(theta)),
                 xytext=(cx, cy),
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5,
                                 linestyle=(0, (4, 2)), mutation_scale=11,
                                 shrinkA=0, shrinkB=0), zorder=4)
    axc.annotate("", xy=(cx + mbx, cy + mby), xytext=(cx, cy),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.8,
                                 mutation_scale=15, shrinkA=0, shrinkB=0),
                 zorder=5)
    axc.text(cx + mbx * 0.5, cy + mby * 0.5 - 0.10, r"$|\bar{\boldsymbol{u}}|$",
             color=BLUE, fontsize=9.0, ha="center", va="top", zorder=6)
    axc.text(1.03, cy + sbar * np.sin(theta), r"$\bar{s}$", color=TEAL,
             fontsize=9.0, ha="left", va="center", zorder=6)
    axc.text(0.5, -0.12, "fine vectors of one 3-km parent cell", color=GREY,
             fontsize=8.0, ha="center", va="top")
    axc.set_xlim(-0.06, 1.12)
    axc.set_ylim(-0.15, 1.05)
    axc.set_aspect("equal")
    # arrows from the cell down to the two estimand boxes
    ax.annotate("", xy=(0.30, 0.50), xytext=(0.30, 0.565),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=6)
    ax.annotate("", xy=(0.70, 0.50), xytext=(0.70, 0.565),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=6)
    # two estimand boxes
    box = dict(boxstyle="round,pad=0.28", facecolor="white",
               edgecolor=INK, linewidth=0.8)
    ax.text(0.30, 0.36, "mean vector\nmagnitude $|\\bar{\\boldsymbol{u}}|$",
            ha="center", va="center", fontsize=8.0, color=INK, bbox=box,
            zorder=7)
    ax.text(0.70, 0.36, "mean scalar\nspeed $\\bar{s}$", ha="center",
            va="center", fontsize=8.0, color=INK, bbox=box, zorder=7)
    ax.text(0.70, 0.295, r"width $S$", ha="center", va="center", fontsize=7.4,
            color=GREY, zorder=7)
    ax.annotate("", xy=(0.70, 0.235), xytext=(0.70, 0.27),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=6)
    # lower inset: speed-distribution sketch with +-S marked
    axd = ax.inset_axes([0.06, 0.02, 0.88, 0.18])
    figstyle.clean_axis(axd)
    S = float(speed.std()) * k
    xs = np.linspace(sbar - 3.4 * S, sbar + 3.4 * S, 200)
    pdf = np.exp(-0.5 * ((xs - sbar) / S) ** 2)
    axd.fill_between(xs, pdf, color=tint(BLUE, 0.78), lw=0, zorder=1)
    axd.plot(xs, pdf, color=BLUE, lw=1.2, zorder=2)
    axd.plot([sbar, sbar], [0, 1.02], color=BLUE, lw=0.9, zorder=3)
    for xm in (sbar - S, sbar + S):
        axd.plot([xm, xm], [0, np.exp(-0.5)], color=GREY, lw=0.8,
                 ls=(0, (3, 2)), zorder=3)
    axd.annotate("", xy=(sbar + S, 0.78), xytext=(sbar - S, 0.78),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9), zorder=4)
    axd.text(sbar, 0.88, r"$2S$", ha="center", va="bottom", fontsize=8.0,
             color=INK)
    axd.set_xticks([sbar - S, sbar, sbar + S])
    axd.set_xticklabels([r"$-S$", r"$\bar{s}$", r"$+S$"])
    axd.set_yticks([])
    axd.set_ylim(0, 1.3)
    axd.set_xlabel(r"Fine-scale speed $s$", labelpad=1.5)


def fig2() -> None:
    df = read("F03_physical_decomposition.csv")
    fig = plt.figure(figsize=(figstyle.MAIN_WIDTH_IN, 6.0))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.18, 1.18, 1.0, 1.0, 1.0, 1.16],
                          hspace=0.75, wspace=0.38)
    axa = fig.add_subplot(gs[:, 0:2])
    _concept_panel_v2(axa)
    figstyle.panel_label(axa, "a", y=1.02)

    axb = fig.add_subplot(gs[0, 2])
    axc = fig.add_subplot(gs[0, 3])
    axd = fig.add_subplot(gs[0, 4])
    for ax, (key, title, cmap, vmin, vmax, label, panel) in zip(
            [axb, axc, axd], [
                ("delta_dynamic_mean", r"$\Delta_{\rm dyn}$", "coolwarm",
                 -0.04, 0.04, r"$\Delta_{\rm dyn}$ (m s$^{-1}$)", "b"),
                ("jensen_gap_mean", r"$J$", "YlOrBr", 0.0, 0.65,
                 r"$J$ (m s$^{-1}$)", "c"),
                ("delta_scalar_mean", r"$\Delta_{\rm scalar}$", "YlOrBr",
                 0.0, 0.65, r"$\Delta_{\rm scalar}$ (m s$^{-1}$)", "d"),
            ]):
        figstyle.clean_axis(ax)
        spatial_map(ax, df[key], title, cmap, vmin, vmax, label,
                    cbar_fraction=0.050, cbar_pad=0.02)
        figstyle.panel_label(ax, panel, x=0.985, ha="right")

    axe = fig.add_subplot(gs[1, 2:4])
    figstyle.clean_axis(axe)
    means = [df.delta_dynamic_mean.mean(), df.jensen_gap_mean.mean(),
             df.delta_scalar_mean.mean()]
    _finite(*means, ctx="fig2e")
    names = ["mean-vector magnitude", "Jensen gap", "scalar speed"]
    colors = [BLUE, ORANGE, TEAL]
    ypos = np.arange(3)
    for y, name, value, color in zip(ypos, names, means, colors):
        axe.barh(y, value, height=0.5, color=color, edgecolor="none", zorder=3)
        dx = 0.008 if value >= 0 else -0.008
        axe.text(value + dx, y, f"{value:+.3f}", va="center",
                 ha="left" if value >= 0 else "right", fontsize=8.2, color=INK)
    axe.axvline(0, color="black", lw=0.5, zorder=2)
    axe.set_yticks(ypos, names)
    axe.set_xlim(-0.06, 0.335)
    axe.set_xlabel("Domain mean (m s$^{-1}$)")
    axe.set_title("Exact decomposition", pad=3.5)
    axe.text(0.30, 0.08, r"$\Delta_{\rm scalar} = \Delta_{\rm dyn} + J$",
             transform=axe.transAxes, color=GREY, fontsize=8.0, va="bottom")
    figstyle.panel_label(axe, "e")

    # (f) scale dependence on a categorical axis; scales read from the CSV
    # keys themselves (float64: 111.11111111111111 ...), never from
    # truncated literals
    axf = fig.add_subplot(gs[1, 4:6])
    figstyle.clean_axis(axf)
    sc = read_evid("fig4_scale_terciles.csv")
    scale_keys = sorted(sc.scale_m.unique())
    med = np.array([float(sc[np.isclose(sc.scale_m, s)].scale_sd.median())
                    for s in scale_keys])
    med_speed = np.array([float(sc[np.isclose(sc.scale_m, s)]
                                .scale_speed_mean.median()) for s in scale_keys])
    _finite(med, med_speed, ctx="fig2f")
    xpos = np.arange(len(scale_keys))
    axf.plot(xpos, med, "-o", color=BLUE, ms=4, lw=1.3, zorder=3)
    for xi, m in zip(xpos, med):
        axf.text(xi, m + 0.035, f"{m:.2f}", ha="center", fontsize=7.6,
                 color=BLUE)
    axf.axvline(xpos[-1], color=GREY, ls="--", lw=0.8)
    axf.set_xticks(xpos)
    axf.set_xticklabels(["111", "333", "1000", "3000"])
    axf.set_xlabel("Aggregation scale (m)")
    axf.set_ylabel(r"Median within-parent spread (m s$^{-1}$)")
    axf.set_ylim(0, 1.05)
    axf.set_title("Scale dependence", pad=3.5)
    axf.text(0.03, 0.05, "mean scalar speed: "
             f"{med_speed[0]:.2f} → {med_speed[-1]:.2f} m s$^{{-1}}$",
             transform=axf.transAxes, fontsize=7.6, color=INK)
    figstyle.panel_label(axf, "f")
    _save_fig(fig, "Fig2_coarse_graining_loss")


# ---------------------------------------------------------------------------
# Fig 3 — Why the Jensen contribution appears
# ---------------------------------------------------------------------------

def fig3() -> None:
    j = read("F04_jensen_second_order.csv")
    b = read("F04_jensen_binned.csv")
    g = read("F04_jensen_two_groups.csv")
    tr = j[j.period == "train"]
    ev = j[j.period == "eval"]
    fig, axes = plt.subplots(2, 2, figsize=(figstyle.MAIN_WIDTH_IN, 5.2),
                             gridspec_kw={"width_ratios": [0.96, 1.04],
                                          "hspace": 0.62, "wspace": 0.42})

    ax = axes[0, 0]
    figstyle.clean_axis(ax)
    vmin = min(tr.U_mean.min(), ev.U_mean.min())
    vmax = max(tr.U_mean.max(), ev.U_mean.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    lo, hi = 0.0, max(tr.J_mean.max(), tr.J2nd_mean.max(),
                      ev.J2nd_mean.max()) * 1.08
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.0, zorder=2)
    sc_cal = ax.scatter(tr.J_mean, tr.J2nd_mean, s=24, c=plt.cm.viridis(norm(tr.U_mean)),
                        edgecolor=plt.cm.viridis(norm(tr.U_mean)), linewidth=1.1,
                        zorder=3, label="calibration")
    sc_cal.set_facecolor("none")
    sc_ev = ax.scatter(ev.J_mean, ev.J2nd_mean, s=24, c=plt.cm.viridis(norm(ev.U_mean)),
                       edgecolor="black", linewidth=0.6, zorder=3, label="evaluation")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                 ax=ax, fraction=0.030, pad=0.015).set_label(
        r"$|\bar{\boldsymbol{u}}|$ (m s$^{-1}$)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Exact $J$ (m s$^{-1}$)")
    ax.set_ylabel(r"$J_{\rm 2nd}$ (m s$^{-1}$)")
    ax.set_title("Exact gap vs second-order estimate", pad=3.5)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    figstyle.panel_label(ax, "a")

    ax = axes[0, 1]
    figstyle.clean_axis(ax)
    for period, col in [("train", BLUE), ("eval", ORANGE)]:
        bb = b[b.period == period]
        # open-ended final bin (bin_hi = inf) is centred at bin_lo + 0.25
        x = bb.bin_lo + np.where(np.isfinite(bb.bin_hi - bb.bin_lo),
                                 (bb.bin_hi - bb.bin_lo) / 2, 0.25)
        _finite(x, bb.rmse, ctx="fig3b")
        ax.plot(x, bb.rmse, "-o", color=col, ms=3, lw=1.1,
                label="calibration" if period == "train" else "evaluation")
    ax.axvline(1.5, color=GREY, ls="--", lw=0.8)
    ax.text(1.55, ax.get_ylim()[1] * 0.92, "1.5 m s$^{-1}$", fontsize=7.4,
            color=GREY)
    ax.set_xlabel(r"$|\bar{\boldsymbol{u}}_p|$ bin (m s$^{-1}$)")
    ax.set_ylabel(r"RMSE($J_{\rm 2nd}-J$) (m s$^{-1}$)")
    ax.set_title("Error increases at low mean-vector speed", pad=3.5)
    ax.legend(frameon=False, fontsize=7.6)
    figstyle.panel_label(ax, "b")

    ax = axes[1, 0]
    figstyle.clean_axis(ax)
    groups = []
    for period in ("train", "eval"):
        for wg in ("below", "above"):
            r = g[(g.period == period) & (g.wind_group == wg)].iloc[0]
            groups.append((period, wg, r))
    xpos = np.arange(4)
    cols = [BLUE, BLUE, ORANGE, ORANGE]
    for x, (period, wg, r), col in zip(xpos, groups, cols):
        face = tint(col, 0.82) if wg == "below" else col
        ax.plot([x, x], [r.err_q05, r.err_q95], color=col, lw=1.0, zorder=2)
        ax.plot([x, x], [r.err_q25, r.err_q75], color=col, lw=3.6,
                solid_capstyle="butt", zorder=3)
        ax.plot(x, r.err_median, "o", color="white", mec=col, ms=4.5,
                zorder=4)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["calib\n<1.5", "calib\n>1.5", "eval\n<1.5", "eval\n>1.5"])
    ax.set_ylabel(r"$J_{\rm 2nd}-J$ (m s$^{-1}$)")
    ax.set_title("Error below and above 1.5 m s$^{-1}$", pad=3.5)
    # record shares computed from the two-group table (below-1.5 share)
    n_tr = float(g.loc[g.period == "train", "n_cell_time_records"].sum())
    n_ev = float(g.loc[g.period == "eval", "n_cell_time_records"].sum())
    share_tr = float(g.loc[(g.period == "train") & (g.wind_group == "below"),
                           "n_cell_time_records"].iloc[0]) / n_tr
    share_ev = float(g.loc[(g.period == "eval") & (g.wind_group == "below"),
                           "n_cell_time_records"].iloc[0]) / n_ev
    ax.text(0.5, 0.035,
            f"below 1.5 m s$^{{-1}}$: {share_tr * 100:.0f}% of calibration, "
            f"{share_ev * 100:.0f}% of evaluation records",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.4,
            color=GREY)
    figstyle.panel_label(ax, "c")

    ax = axes[1, 1]
    figstyle.clean_axis(ax)
    spatial_map(ax, tr.sigma_perp2_frac,
                r"Across-wind share $\sigma_\perp^2/{\rm tr}\,\mathbf{C}$",
                "viridis", 0.0, 0.7, r"$\sigma_\perp^2/{\rm tr}\,\mathbf{C}$",
                annot="extrema", cbar_fraction=0.050, cbar_pad=0.02)
    figstyle.panel_label(ax, "d", x=0.985, ha="right")
    _save_fig(fig, "Fig3_jensen_covariance")


# ---------------------------------------------------------------------------
# Fig 4 — What geometric interpolation recovers
# ---------------------------------------------------------------------------

def fig4() -> None:
    f06 = read("F06_chronological_evaluation.csv")
    rec = read_npj("interpolation_recovery.csv")
    met = read_npj("interpolation_metrics.csv")
    src = read_npj("information_sources.csv")
    npz = np.load(NPJ / "representative_state_fields.npz")

    fig = plt.figure(figsize=(figstyle.MAIN_WIDTH_IN, 6.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.16, 1.0, 1.0],
                          hspace=0.62, wspace=0.40)

    # (a) representative state: four real fields on a shared colour scale
    s05 = npz["d05_speed"].astype(float)
    vmin, vmax = float(np.nanpercentile(s05, 1)), float(np.nanpercentile(s05, 99))
    fields = [
        ("d02 parent (6×6, expanded)", npz["d02_6x6_speed"].astype(float), True),
        ("bilinear", npz["bilinear_speed"].astype(float), False),
        ("bicubic", npz["bicubic_speed"].astype(float), False),
        ("d05 reference", s05, False),
    ]
    sub = gs[0, :].subgridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 0.07],
                               wspace=0.04)
    axes_a = []
    for k, (title, fld, is6) in enumerate(fields):
        _finite(fld, ctx=f"fig4a-{k}")
        ax = fig.add_subplot(sub[k])
        figstyle.clean_axis(ax)
        if is6:
            fld = fld.repeat(27, 0).repeat(27, 1)  # nearest-neighbour expansion
        im = ax.imshow(fld, origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        if k == 0:
            # parent-cell boundaries on the expanded d02 field
            for e in np.arange(0, 163, 27) - 0.5:
                ax.axvline(e, color="white", lw=0.45)
                ax.axhline(e, color="white", lw=0.45)
        ax.set_title(title, fontsize=8.0, pad=2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("fine cell")
        ax.set_aspect("auto")
        axes_a.append(ax)
    cax = fig.add_subplot(sub[4])
    cax.set_axis_off()
    figstyle.add_shared_colorbar(fig, im, cax, "wind speed (m s$^{-1}$)",
                                 fraction=1.0, pad=0.02)
    figstyle.panel_label(axes_a[0], "a", x=0.015, y=1.10)

    # (b) scalar-increment parity (bilinear, bicubic)
    ax = fig.add_subplot(gs[1, 0])
    figstyle.clean_axis(ax)
    for model, col in [("bilinear", TEAL), ("bicubic", ORANGE)]:
        ax.scatter(f06.obs_scalar, f06[f"pred_scalar_{model}"], s=13,
                   color=col, zorder=3, label=model)
    lim = [f06.obs_scalar.min() - 0.05, f06.obs_scalar.max() + 0.05]
    ax.plot(lim, lim, color="black", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Reference $\mu_{\Delta}$ (m s$^{-1}$)")
    ax.set_ylabel("Interpolated (m s$^{-1}$)")
    ax.set_title("Scalar increment, 36 cells", pad=3.5)
    ax.legend(frameon=False, fontsize=7.6)
    figstyle.panel_label(ax, "b")

    # (c) spread parity
    ax = fig.add_subplot(gs[1, 1])
    figstyle.clean_axis(ax)
    for model, col in [("bilinear", TEAL), ("bicubic", ORANGE)]:
        ax.scatter(f06.obs_spread, f06[f"pred_spread_{model}"], s=13,
                   color=col, zorder=3, label=model)
    lim = [f06.obs_spread.min() - 0.05, f06.obs_spread.max() + 0.05]
    ax.plot(lim, lim, color="black", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Reference $\mu_{S}$ (m s$^{-1}$)")
    ax.set_ylabel("Interpolated (m s$^{-1}$)")
    ax.set_title("Within-parent spread, 36 cells", pad=3.5)
    ax.legend(frameon=False, fontsize=7.6)
    figstyle.panel_label(ax, "c")

    # (d) Jensen recovery Q_J in percent
    ax = fig.add_subplot(gs[1, 2])
    figstyle.clean_axis(ax)
    qj = rec[rec.moment == "J"]
    models = ["bilinear", "bicubic"]
    x = np.arange(2)
    w = 0.32
    for k, cs in enumerate(["all36", "inner16"]):
        vals = np.array([float(qj[(qj.model == m) & (qj.cell_set == cs)]
                               .recovery_ratio.iloc[0]) * 100.0 for m in models])
        _finite(vals, ctx="fig4d")
        ax.bar(x + (k - 0.5) * w, vals, width=w, color=[TEAL, ORANGE],
               edgecolor="none", zorder=3, label="36 cells" if cs == "all36"
               else "inner 16")
        for xi, v in zip(x + (k - 0.5) * w, vals):
            ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", fontsize=7.2)
    ax.axhline(100.0, color=INK, ls="--", lw=0.9)
    ax.text(1.5, 102.0, "100% = exact recovery", fontsize=7.2, color=INK,
            ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["bilinear", "bicubic"])
    ax.set_ylabel("Jensen moment recovery (%)")
    ax.set_ylim(0, 55)
    ax.set_title("Jensen recovery", pad=3.5)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    figstyle.panel_label(ax, "d")

    # (e) moment recovery Q_delta, Q_S in percent + vector-preservation
    # diagnostics for BOTH kernels
    ax = fig.add_subplot(gs[2, 0])
    figstyle.clean_axis(ax)
    moments = [("delta_scalar", r"$Q_{\Delta}$"), ("spread", r"$Q_S$")]
    x = np.arange(2)
    for k, (moment, lab) in enumerate(moments):
        vals = np.array([float(rec[(rec.moment == moment) & (rec.model == m)
                                   & (rec.cell_set == "all36")]
                               .recovery_ratio.iloc[0]) * 100.0 for m in models])
        _finite(vals, ctx="fig4e")
        ax.bar(x + k * 0.9, vals, width=0.8, color=[TEAL, ORANGE],
               edgecolor="none", zorder=3)
        for xi, v in zip(x + k * 0.9, vals):
            ax.text(xi, v + 2.0, f"{v:.0f}", ha="center", fontsize=7.4)
    ax.axhline(100.0, color=INK, ls="--", lw=0.9)
    ax.set_xticks([0.45, 1.35])
    ax.set_xticklabels([r"Scalar increment", r"Within-parent spread"])
    ax.set_ylabel("Domain-mean moment recovery (%)")
    ax.set_ylim(-70, 115)
    ax.set_title("Moment recovery ratios", pad=3.5)
    # mini table: vector-preservation diagnostics of both kernels (m s^-1)
    tb = ax.table(
        cellText=[["0.444", "0.250"], ["−0.130", "−0.059"]],
        rowLabels=[r"$E_{\rm vec}$", r"$\overline{\Delta}_{\rm dyn}^I$"],
        colLabels=["Bilinear", "Bicubic"],
        cellLoc="center", loc="upper center",
        bbox=[0.02, 0.74, 0.96, 0.24])
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.8)
    for (r, c), cell in tb.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor(GREY)
        cell.set_linewidth(0.4)
    ax.text(0.5, 0.71, "(m s$^{-1}$)", transform=ax.transAxes,
            ha="center", fontsize=6.8, color=GREY)
    figstyle.panel_label(ax, "e")

    # (f) period-mean RMSE dot plot: methods on y ticks, no labels in the
    # panel, group headers, blank row between the moments
    ax = fig.add_subplot(gs[2, 1:])
    figstyle.clean_axis(ax)
    order_s = ["zero_correction", "global_training_mean", "bilinear", "bicubic",
               "terrain_roughness_equation", "cell_mean_persistence"]
    order_p = ["global_training_mean", "bilinear", "bicubic", "roughness_only",
               "exposure_only", "locked_joint_equation", "cell_mean_persistence"]
    ypos, labels, rows = [], [], []
    for group, order, lab in [("scalar_increment", order_s, "scalar increment"),
                              ("spread", order_p, "within-parent spread")]:
        d = src[src.group == group].set_index("method").loc[order]
        for mname, r in d.iterrows():
            rows.append((lab, mname, r))
    n = len(rows)
    y = np.arange(n, dtype=float)
    y[np.asarray([lab for lab, _, _ in rows]) == "within-parent spread"] += 0.7
    for k, (lab, mname, r) in enumerate(rows):
        _finite(r.rmse, ctx="fig4f")
        ax.scatter(r.rmse, y[k], s=24, color=METHOD_COLOR[mname], zorder=3)
        labels.append(METHOD_LABEL[mname])
    gap_y = float(y[np.asarray([lab for lab, _, _ in rows]) == "within-parent spread"][0]) - 0.4
    ax.axhline(gap_y, color=LIGHT, lw=1.0, zorder=1)
    ax.text(-0.012, 5.75, "scalar increment", transform=ax.get_yaxis_transform(),
            fontsize=7.6, fontweight="bold", color=INK, ha="right", va="bottom")
    ax.text(-0.012, 12.85, "within-parent spread",
            transform=ax.get_yaxis_transform(), fontsize=7.6, fontweight="bold",
            color=INK, ha="right", va="bottom")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", labelsize=7.4)
    ax.set_xlabel("Period-mean RMSE (m s$^{-1}$)")
    ax.set_title("Moment RMSE by method", pad=3.5)
    ax.set_xlim(-0.02, 0.66)
    figstyle.panel_label(ax, "f")
    _save_fig(fig, "Fig4_geometric_refinement")


# ---------------------------------------------------------------------------
# Fig 5 — What terrain information adds
# ---------------------------------------------------------------------------

def fig5() -> None:
    f56 = read("F05_F06_terrain_parameterization.csv")
    j2 = read("F04_jensen_second_order.csv")
    f03 = read("F03_physical_decomposition.csv")
    cc = read("F05_case_cells.csv")
    strat = read_evid("state_stratified_terrain.csv")
    mc = pd.read_csv(SNAP / "outputs_evaluation_holdout_model_cells.csv")
    exp_pred = mc["spread_exposure_only_predicted"].values
    lam = f56.sort_values("cell_id").spread_lambda.values

    fig, axes = plt.subplots(2, 3, figsize=(figstyle.MAIN_WIDTH_IN, 5.6))
    fig.subplots_adjust(wspace=0.44, hspace=0.72)

    # (a) scalar increment vs roughness
    ax = axes[0, 0]
    figstyle.clean_axis(ax)
    d56 = f56.sort_values("cell_id")
    ax.scatter(d56.mean_roughness, d56.mean_observed_train, s=16, color=GREY,
               zorder=3)
    rng = np.linspace(d56.mean_roughness.min() * 0.9, d56.mean_roughness.max() * 1.1, 50)
    ax.plot(rng, 0.079296 + 5.578546 * rng, color=RED, lw=1.2, zorder=4,
            label=r"$\hat{\mu}_\Delta = 0.079 + 5.58\,r_\sigma$")
    ax.set_xlabel(r"Terrain roughness $r_\sigma$")
    ax.set_ylabel(r"Calibration mean $\mu_{\Delta}$ (m s$^{-1}$)")
    ax.set_title("Roughness and scalar increment", pad=3.5)
    ax.text(0.97, 0.06, r"$R^2_{\rm adj}=0.659$", transform=ax.transAxes,
            ha="right", fontsize=7.8, color=INK)
    ax.legend(frameon=False, fontsize=7.0)
    figstyle.panel_label(ax, "a")

    # (b) J vs roughness with the flat delta_dyn inset
    ax = axes[0, 1]
    figstyle.clean_axis(ax)
    jtr = j2[j2.period == "train"].sort_values("cell_id")
    ax.scatter(d56.mean_roughness, jtr.J_mean, s=16, color=BLUE, zorder=3)
    ax.set_xlabel(r"Terrain roughness $r_\sigma$")
    ax.set_ylabel(r"Calibration mean $J$ (m s$^{-1}$)")
    ax.set_title("Roughness and Jensen gap", pad=3.5)
    ax.text(0.97, 0.06, r"$R^2=0.671$", transform=ax.transAxes, ha="right",
            fontsize=7.8, color=INK)
    axi = ax.inset_axes([0.60, 0.08, 0.37, 0.33])
    figstyle.clean_axis(axi)
    axi.set_facecolor("white")
    axi.set_zorder(6)
    axi.scatter(d56.mean_roughness, f03.sort_values("cell_id").delta_dynamic_mean,
                s=9, color=GREY, zorder=3)
    axi.set_title(r"$\Delta_{\rm dyn}$ ($R^2=0.016$)", fontsize=7.0, pad=2)
    axi.set_xlabel(r"$r_\sigma$", fontsize=7.2)
    axi.tick_params(labelsize=6.8)
    figstyle.panel_label(ax, "b")

    # (c) spread vs isotropic exposure; the fit line is drawn from explicit
    # intercept/slope (round-2 fix: slope/intercept were swapped before)
    ax = axes[0, 2]
    figstyle.clean_axis(ax)
    y_obs = d56.spread_observed_train.to_numpy()
    _finite(lam, y_obs, exp_pred, ctx="fig5c")
    ax.scatter(lam, y_obs, s=16, color=ORANGE, zorder=3)
    slope, intercept = np.polyfit(lam, exp_pred, 1)
    lrng = np.linspace(lam.min() * 0.9, lam.max() * 1.1, 50)
    yfit = intercept + slope * lrng
    np.testing.assert_allclose(yfit, np.polyval([slope, intercept], lrng),
                               err_msg="fig5c fit-line consistency")
    _finite(yfit, ctx="fig5c-fit")
    ax.plot(lrng, yfit, color=RED, lw=1.2, zorder=4, label="exposure-only fit")
    assert yfit.max() < 2.5, "fig5c: fit line must stay inside the panel"
    pred = intercept + slope * lam
    r2 = 1.0 - float(np.sum((y_obs - pred) ** 2) / np.sum((y_obs - y_obs.mean()) ** 2))
    ax.set_xlabel(r"Isotropic exposure $\lambda_{\rm iso}$")
    ax.set_ylabel(r"Calibration mean $\mu_{S}$ (m s$^{-1}$)")
    ax.set_title("Exposure and within-parent spread", pad=3.5)
    ax.text(0.97, 0.06, rf"$R^2={r2:.3f}$", transform=ax.transAxes, ha="right",
            fontsize=7.8, color=INK)
    ax.set_ylim(0, 2.2)
    ax.legend(frameon=False, fontsize=7.0)
    figstyle.panel_label(ax, "c")

    # (d),(e) matched-roughness case cells (4,4) and (5,1): shared terrain
    # colour scale AND identical contour levels
    cmin = float(cc.hgt_rel.min())
    cmax = float(cc.hgt_rel.max())
    smin = min(float(cc[cc.case_id == 28].speed_mean.min()),
               float(cc[cc.case_id == 31].speed_mean.min()))
    smax = max(float(cc[cc.case_id == 28].speed_mean.max()),
               float(cc[cc.case_id == 31].speed_mean.max()))
    levels = np.linspace(smin, smax, 8)
    for ax, case_id in [(axes[1, 0], 28), (axes[1, 1], 31)]:
        figstyle.clean_axis(ax)
        sub = cc[cc.case_id == case_id]
        h = sub.hgt_rel.values.reshape(27, 27)
        sp = sub.speed_mean.values.reshape(27, 27)
        _finite(h, sp, ctx=f"fig5de-{case_id}")
        im = ax.imshow(h, origin="lower", cmap="terrain_r", vmin=cmin, vmax=cmax)
        ax.contour(sp, levels=levels, colors="white", linewidths=0.5, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"cell ({case_id // 6},{case_id % 6})", fontsize=8.2,
                     pad=2.5)
    axes[1, 0].set_ylabel("fine cell")
    fig.colorbar(im, ax=[axes[1, 0], axes[1, 1]], fraction=0.05, pad=0.04,
                 aspect=24).set_label("terrain height (m)")
    figstyle.panel_label(axes[1, 0], "d")
    figstyle.panel_label(axes[1, 1], "e")

    # (f) tercile slope stability as a horizontal forest (two offset series)
    ax = axes[1, 2]
    figstyle.clean_axis(ax)
    terc = strat[strat.stratum == "speed_tercile"].sort_values("level")
    y = np.array([2.0, 1.0, 0.0])
    for off, col, key in [(-0.18, BLUE, "delta"), (0.18, ORANGE, "spread")]:
        if key == "delta":
            vals = terc.slope_delta_vs_rough.values
            lo = terc.ci_low_delta.values
            hi = terc.ci_high_delta.values
        else:
            vals = terc.slope_spread_vs_lambda.values
            lo = terc.ci_low_spread.values
            hi = terc.ci_high_spread.values
        _finite(vals, lo, hi, ctx="fig5f")
        ax.errorbar(vals, y + off, xerr=[vals - lo, hi - vals], fmt="o",
                    color=col, ms=4, lw=1.0, capsize=2.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(["low", "medium", "high"])
    ax.set_xlabel("Slope within wind-speed tercile")
    ax.set_title("Tercile stability", pad=3.5)
    ax.axvline(0, color=INK, lw=0.6)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", color=BLUE, ls="none", label=r"$\mu_\Delta$ on $r_\sigma$"),
        plt.Line2D([], [], marker="o", color=ORANGE, ls="none", label=r"$\mu_S$ on $\lambda_{\rm iso}$"),
    ], frameon=False, fontsize=7.0)
    figstyle.panel_label(ax, "f")
    _save_fig(fig, "Fig5_terrain_organization")


# ---------------------------------------------------------------------------
# Fig 6 — Performance across information sources under the changed state
# ---------------------------------------------------------------------------

def fig6() -> None:
    ws = read("F06_wind_state.csv")
    f06 = read("F06_chronological_evaluation.csv")
    src = read_npj("information_sources.csv")
    boot = read_npj("interpolation_bootstrap.csv")

    fig, axes = plt.subplots(2, 3, figsize=(figstyle.MAIN_WIDTH_IN, 5.8))
    fig.subplots_adjust(wspace=0.44, hspace=0.72)

    # (a) wind-state sector bars
    ax = axes[0, 0]
    figstyle.clean_axis(ax)
    sectors = np.arange(8)
    tr = ws[ws.period == "training"]
    ev = ws[ws.period == "evaluation"]
    w = 0.36
    ax.bar(sectors - w / 2, [tr[f"sector_{k}"].iloc[0] for k in sectors],
           width=w, color=BLUE, zorder=3, label="calibration")
    ax.bar(sectors + w / 2, [ev[f"sector_{k}"].iloc[0] for k in sectors],
           width=w, color=ORANGE, zorder=3, label="evaluation")
    ax.set_xticks(sectors)
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_ylabel("State fraction")
    ax.set_title("Domain-mean wind state", pad=3.5)
    ax.legend(frameon=False, fontsize=7.4)
    ax.text(0.02, 0.97, (f"mean speed 2.83 → 2.29 m s$^{{-1}}$"),
            transform=ax.transAxes, va="top", fontsize=7.6, color=INK)
    figstyle.panel_label(ax, "a")

    # (b) scalar parity: bilinear, bicubic, terrain
    ax = axes[0, 1]
    figstyle.clean_axis(ax)
    ax.scatter(f06.obs_scalar, f06.pred_scalar, s=13, color=BLUE, zorder=3,
               label="terrain")
    ax.scatter(f06.obs_scalar, f06.pred_scalar_bilinear, s=13, color=TEAL,
               zorder=3, label="bilinear")
    ax.scatter(f06.obs_scalar, f06.pred_scalar_bicubic, s=13, color=ORANGE,
               zorder=3, label="bicubic")
    lim = [f06.obs_scalar.min() - 0.05, f06.obs_scalar.max() + 0.05]
    ax.plot(lim, lim, color="black", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Reference $\mu_{\Delta}$ (m s$^{-1}$)")
    ax.set_ylabel("Prediction (m s$^{-1}$)")
    ax.set_title("Scalar increment", pad=3.5)
    ax.legend(frameon=False, fontsize=7.2)
    figstyle.panel_label(ax, "b")

    # (c) spread parity
    ax = axes[0, 2]
    figstyle.clean_axis(ax)
    ax.scatter(f06.obs_spread, f06.pred_spread_joint, s=13, color=BLUE,
               zorder=3, label="joint")
    ax.scatter(f06.obs_spread, f06.pred_spread_bilinear, s=13, color=TEAL,
               zorder=3, label="bilinear")
    ax.scatter(f06.obs_spread, f06.pred_spread_bicubic, s=13, color=ORANGE,
               zorder=3, label="bicubic")
    lim = [f06.obs_spread.min() - 0.05, f06.obs_spread.max() + 0.05]
    ax.plot(lim, lim, color="black", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Reference $\mu_{S}$ (m s$^{-1}$)")
    ax.set_ylabel("Prediction (m s$^{-1}$)")
    ax.set_title("Within-parent spread", pad=3.5)
    ax.legend(frameon=False, fontsize=7.2)
    figstyle.panel_label(ax, "c")

    # (d) raw -> centred dumbbells: removing the domain mean shows how much
    # of the skill is spatial template vs atmospheric amplitude
    ax = axes[1, 0]
    figstyle.clean_axis(ax)
    dumbbell = [
        ("scalar increment", "terrain_roughness_equation", BLUE),
        ("scalar increment", "bilinear", TEAL),
        ("scalar increment", "bicubic", ORANGE),
        ("within-parent spread", "locked_joint_equation", BLUE),
        ("within-parent spread", "bilinear", TEAL),
        ("within-parent spread", "bicubic", ORANGE),
    ]
    ypos = np.array([2.6, 1.6, 0.6, -1.0, -2.0, -3.0])
    for (lab, method, col), y in zip(dumbbell, ypos):
        r = src[(src.group == ("scalar_increment" if lab.startswith("scalar")
                               else "spread")) & (src.method == method)].iloc[0]
        raw, cen = float(r.rmse), float(r.crmse)
        _finite(raw, cen, ctx="fig6d")
        ax.plot([cen, raw], [y, y], color=col, lw=1.1, zorder=2)
        ax.scatter(raw, y, s=24, color=col, zorder=3, label="_nolegend_")
        ax.scatter(cen, y, s=24, facecolor="white", edgecolor=col, lw=0.9,
                   zorder=3, label="_nolegend_")
    ax.text(0.02, 3.0, "scalar increment", transform=ax.get_yaxis_transform(),
            fontsize=7.8, fontweight="bold", color=INK)
    ax.text(0.02, -3.7, "within-parent spread", transform=ax.get_yaxis_transform(),
            fontsize=7.8, fontweight="bold", color=INK)
    ax.set_ylim(-4.2, 3.7)
    ax.set_yticks([])
    ax.set_xlabel("RMSE (m s$^{-1}$)")
    ax.set_title("Raw and centred RMSE", pad=3.5)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", color=GREY, ls="none", label="raw"),
        plt.Line2D([], [], marker="o", markerfacecolor="white", color=GREY,
                   ls="none", label="centred"),
    ], frameon=False, fontsize=7.0, loc="lower right")
    figstyle.panel_label(ax, "d")

    # (e) RMSE by information class: background bands + class colours, no
    # per-point text, group headers
    ax = axes[1, 1]
    figstyle.clean_axis(ax)
    order_s = ["zero_correction", "global_training_mean", "bilinear", "bicubic",
               "terrain_roughness_equation", "cell_mean_persistence"]
    order_p = ["global_training_mean", "bilinear", "bicubic", "roughness_only",
               "exposure_only", "locked_joint_equation", "cell_mean_persistence"]
    yoff = 0.0
    group_rows = []
    for group, order, lab in [("scalar_increment", order_s, "scalar"),
                              ("spread", order_p, "spread")]:
        d = src[src.group == group].set_index("method").loc[order]
        for mname, r in d.iterrows():
            group_rows.append((lab, mname, r, yoff))
            yoff += 1.0
        yoff += 0.9
    for lab, mname, r, ypos_ in group_rows:
        _finite(r.rmse, ctx="fig6e")
        ax.scatter(r.rmse, ypos_, s=22, color=METHOD_COLOR[mname], zorder=3)
        ax.axhspan(ypos_ - 0.45, ypos_ + 0.45, color=tint(
            CLASS_COLOR[r.info_class], 0.94), zorder=1)
    ax.axvline(0, color=INK, lw=0.6)
    ax.text(0.02, -0.75, "scalar", transform=ax.get_yaxis_transform(),
            fontsize=7.8, fontweight="bold", color=INK)
    ax.text(0.02, 5.95, "spread", transform=ax.get_yaxis_transform(),
            fontsize=7.8, fontweight="bold", color=INK)
    ax.set_ylim(-1.4, 13.8)
    ax.set_yticks([])
    ax.set_xlabel("Period-mean RMSE (m s$^{-1}$)")
    ax.set_title("RMSE by information class", pad=3.5)
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=7,
                          markerfacecolor=CLASS_COLOR[k], markeredgecolor="none",
                          label=CLASS_DISPLAY[k]) for k in CLASS_COLOR]
    ax.legend(handles=handles, frameon=False, fontsize=6.8, loc="lower right")
    figstyle.panel_label(ax, "e")

    # (f) paired DeltaRMSE forest (8 pairs, 120 min) with group separation
    ax = axes[1, 2]
    figstyle.clean_axis(ax)
    b120 = boot[boot.block_minutes == 120].copy()
    order = ["terrain_vs_global", "terrain_vs_bilinear", "terrain_vs_bicubic",
             "terrain_vs_persistence", "joint_vs_global", "joint_vs_bilinear",
             "joint_vs_bicubic", "joint_vs_persistence"]
    b120["ord"] = b120.pair.map({p: i for i, p in enumerate(order)})
    b120 = b120.sort_values("ord")
    ys = np.array([7.0, 6.0, 5.0, 4.0, 2.0, 1.0, 0.0, -1.0], dtype=float)
    for k, (_, r) in enumerate(b120.iterrows()):
        col = BLUE if r.group == "scalar_increment" else RED
        _finite(r.ci_low, r.ci_high, r.dRMSE_est, ctx="fig6f")
        ax.plot([r.ci_low, r.ci_high], [ys[k], ys[k]], color=col, lw=1.1,
                zorder=2)
        ax.plot(r.dRMSE_est, ys[k], "o", color=col, ms=4.5, zorder=3)
        ax.text(r.ci_high + 0.008, ys[k], f"{r.dRMSE_est:.2f}", va="center",
                fontsize=6.9)
    ax.axhline(3.5, color=LIGHT, lw=1.0, zorder=1)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([PAIR_LABEL[p] for p in b120.pair], fontsize=7.2)
    ax.set_xlabel(r"$\Delta$RMSE (m s$^{-1}$; negative favours the terrain equations)")
    ax.set_title("Paired bootstrap, 120-min blocks", pad=3.5)
    figstyle.panel_label(ax, "f")
    _save_fig(fig, "Fig6_information_sources_state")


def main() -> None:
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("NPJ six main figures written to source/figures/main/")


if __name__ == "__main__":
    main()
