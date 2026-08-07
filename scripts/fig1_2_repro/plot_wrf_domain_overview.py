#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw a WRF nested-domain overview figure from ./namelist.wps.

Figure layout
-------------
Left half      : d01-d05 nested domains on satellite imagery.
Right-top      : zoom-in view of d03-d05 on satellite imagery, connected from d03.
Right-bottom   : d05 terrain elevation with contours and a reference-point marker,
                 connected from d05 in the right-top panel.

Default input files, assuming this script is placed in the WPS/WRF case root:
    ./namelist.wps
    ./geo_em.d05.nc

Example:
    python plot_wrf_domain_overview.py
    python plot_wrf_domain_overview.py --output wrf_domain_overview.pdf
    python plot_wrf_domain_overview.py --offline

Dependencies:
    matplotlib, numpy, pyproj, cartopy, xarray or netCDF4

Notes:
    1) Satellite basemap uses Esri World Imagery tiles and therefore needs internet access.
       If the network is unavailable, the script automatically redraws with Cartopy's
       offline Natural Earth background. You can also force offline mode with --offline.
    2) The d05 elevation panel reads HGT_M from geo_em.d05.nc. Please run geogrid.exe
       first if this file does not exist.
"""

from __future__ import print_function

import argparse
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    mpl.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.ticker as mticker
    from matplotlib.patches import ConnectionPatch, Rectangle
except Exception as exc:  # pragma: no cover
    raise SystemExit("缺少 matplotlib。请先安装：conda install matplotlib 或 pip install matplotlib\n%s" % exc)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
except Exception as exc:  # pragma: no cover
    raise SystemExit("缺少 cartopy。建议安装：conda install -c conda-forge cartopy\n%s" % exc)

try:
    from pyproj import CRS, Transformer
except Exception as exc:  # pragma: no cover
    raise SystemExit("缺少 pyproj。建议安装：conda install -c conda-forge pyproj\n%s" % exc)


# These coordinates are the authoritative station location used by the source
# preprocessing and manuscript.  The visual reference archive contained a
# different hard-coded marker, so keep the plotting source tied to the actual
# processed station/grid convention instead of inheriting that stale value.
STATION_LAT = 38.0142
STATION_LON = 100.2421
PROJECTION_REF_LAT = 38.0142
PROJECTION_REF_LON = 100.3421
EARTH_RADIUS_M = 6370000.0  # WRF/WPS spherical earth default for map projections


@dataclass
class Domain:
    """WRF domain geometry in the d01 projected coordinate system."""

    dom_id: int
    parent_id: int
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int
    e_we: int
    e_sn: int
    dx: float
    dy: float
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def label(self) -> str:
        return "d%02d" % self.dom_id

    @property
    def center_xy(self) -> Tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))


class EsriWorldImagery(cimgt.GoogleTiles):
    """Esri World Imagery tile source in the Web-Mercator/XYZ tile scheme."""

    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/%d/%d/%d" % (z, y, x)
        )


def strip_fortran_comment(line: str) -> str:
    """Remove comments introduced by ! outside single/double quotes."""
    in_single = False
    in_double = False
    out = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "!" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out)


def split_fortran_values(value: str) -> List[str]:
    """Split a Fortran namelist value list by commas outside quotes."""
    value = value.strip().rstrip(",")
    parts: List[str] = []
    token: List[str] = []
    in_single = False
    in_double = False
    for ch in value:
        if ch == "'" and not in_double:
            in_single = not in_single
            token.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            token.append(ch)
        elif ch == "," and not in_single and not in_double:
            item = "".join(token).strip()
            if item:
                parts.append(item)
            token = []
        else:
            token.append(ch)
    item = "".join(token).strip()
    if item:
        parts.append(item)
    return parts


def parse_scalar(token: str) -> Any:
    token = token.strip().rstrip(",")
    if len(token) >= 2 and ((token[0] == "'" and token[-1] == "'") or (token[0] == '"' and token[-1] == '"')):
        return token[1:-1]

    low = token.lower()
    if low in [".true.", "true", "t"]:
        return True
    if low in [".false.", "false", "f"]:
        return False

    # Fortran exponent D/d -> E/e
    token_e = re.sub(r"([0-9.])[dD]([+-]?[0-9]+)", r"\1e\2", token)
    try:
        if re.search(r"[.eE]", token_e):
            return float(token_e)
        return int(token_e)
    except ValueError:
        return token


def parse_namelist(path: Path) -> Dict[str, Dict[str, Any]]:
    """Very small Fortran namelist parser sufficient for WPS namelist.wps."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    cleaned_lines = [strip_fortran_comment(line) for line in text.splitlines()]
    text = "\n".join(cleaned_lines)

    sections: Dict[str, Dict[str, Any]] = {}
    for sec_match in re.finditer(r"&([A-Za-z0-9_]+)(.*?)(?:^\s*/\s*$)", text, flags=re.S | re.M):
        sec_name = sec_match.group(1).lower()
        body = sec_match.group(2)
        sec_data: Dict[str, Any] = {}
        # Match key = value blocks until the next key = or section end.
        pattern = re.compile(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)"
            r"(?=^\s*[A-Za-z_][A-Za-z0-9_]*\s*=|\Z)",
            flags=re.S | re.M,
        )
        for m in pattern.finditer(body):
            key = m.group(1).lower()
            raw_value = " ".join(m.group(2).split())
            values = [parse_scalar(v) for v in split_fortran_values(raw_value)]
            if len(values) == 1:
                sec_data[key] = values[0]
            else:
                sec_data[key] = values
        sections[sec_name] = sec_data
    return sections


def as_list(value: Any, n: int, name: str) -> List[Any]:
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    if len(values) < n:
        if len(values) == 1:
            values = values * n
        else:
            raise ValueError("namelist 中 %s 的数量不足：需要 %d 个，实际 %d 个" % (name, n, len(values)))
    return values[:n]


def get_required(section: Dict[str, Any], key: str) -> Any:
    key = key.lower()
    if key not in section:
        raise KeyError("namelist.wps 的 &geogrid 中缺少必需字段：%s" % key)
    return section[key]


def make_wrf_crs(geogrid: Dict[str, Any]) -> Tuple[CRS, Transformer, Transformer, float, float]:
    map_proj = str(get_required(geogrid, "map_proj")).strip("'\"").lower()
    ref_lat = float(get_required(geogrid, "ref_lat"))
    ref_lon = float(get_required(geogrid, "ref_lon"))
    stand_lon = float(get_required(geogrid, "stand_lon"))

    if map_proj == "lambert":
        truelat1 = float(get_required(geogrid, "truelat1"))
        truelat2 = float(geogrid.get("truelat2", truelat1))
        proj4 = (
            "+proj=lcc +lat_1={lat1:.12g} +lat_2={lat2:.12g} "
            "+lat_0={lat0:.12g} +lon_0={lon0:.12g} "
            "+a={radius:.12g} +b={radius:.12g} +units=m +no_defs"
        ).format(lat1=truelat1, lat2=truelat2, lat0=ref_lat, lon0=stand_lon, radius=EARTH_RADIUS_M)
    elif map_proj == "mercator":
        truelat1 = float(get_required(geogrid, "truelat1"))
        proj4 = (
            "+proj=merc +lat_ts={lat_ts:.12g} +lon_0={lon0:.12g} "
            "+a={radius:.12g} +b={radius:.12g} +units=m +no_defs"
        ).format(lat_ts=truelat1, lon0=stand_lon, radius=EARTH_RADIUS_M)
    elif map_proj in ["polar", "polar_stereographic"]:
        truelat1 = float(get_required(geogrid, "truelat1"))
        lat_0 = 90.0 if truelat1 >= 0.0 else -90.0
        proj4 = (
            "+proj=stere +lat_0={lat0:.12g} +lat_ts={lat_ts:.12g} +lon_0={lon0:.12g} "
            "+a={radius:.12g} +b={radius:.12g} +units=m +no_defs"
        ).format(lat0=lat_0, lat_ts=truelat1, lon0=stand_lon, radius=EARTH_RADIUS_M)
    elif map_proj in ["lat-lon", "latlon", "regular_ll"]:
        proj4 = "+proj=longlat +a={radius:.12g} +b={radius:.12g} +no_defs".format(radius=EARTH_RADIUS_M)
    else:
        raise ValueError("当前脚本暂不支持 map_proj=%s；已支持 lambert/mercator/polar/lat-lon" % map_proj)

    wrf_crs = CRS.from_proj4(proj4)
    ll_crs = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(ll_crs, wrf_crs, always_xy=True)
    inv = Transformer.from_crs(wrf_crs, ll_crs, always_xy=True)
    ref_x, ref_y = fwd.transform(ref_lon, ref_lat)
    return wrf_crs, fwd, inv, float(ref_x), float(ref_y)


def build_domains(nml: Dict[str, Dict[str, Any]]) -> List[Domain]:
    if "share" not in nml or "geogrid" not in nml:
        raise ValueError("namelist.wps 中需要包含 &share 和 &geogrid 两个段落。")

    share = nml["share"]
    geogrid = nml["geogrid"]
    max_dom = int(share.get("max_dom", geogrid.get("max_dom", 1)))

    parent_id = [int(v) for v in as_list(get_required(geogrid, "parent_id"), max_dom, "parent_id")]
    ratio = [int(v) for v in as_list(get_required(geogrid, "parent_grid_ratio"), max_dom, "parent_grid_ratio")]
    i_start = [int(v) for v in as_list(get_required(geogrid, "i_parent_start"), max_dom, "i_parent_start")]
    j_start = [int(v) for v in as_list(get_required(geogrid, "j_parent_start"), max_dom, "j_parent_start")]
    e_we = [int(v) for v in as_list(get_required(geogrid, "e_we"), max_dom, "e_we")]
    e_sn = [int(v) for v in as_list(get_required(geogrid, "e_sn"), max_dom, "e_sn")]

    dx0 = float(get_required(geogrid, "dx"))
    dy0 = float(get_required(geogrid, "dy"))
    ref_x = float(geogrid.get("ref_x", 0.5 * (e_we[0] + 1.0)))
    ref_y = float(geogrid.get("ref_y", 0.5 * (e_sn[0] + 1.0)))

    domains: List[Domain] = []
    for idx in range(max_dom):
        dom_id = idx + 1
        if idx == 0:
            dx = dx0
            dy = dy0
            x0 = (1.0 - ref_x) * dx
            x1 = (float(e_we[idx]) - ref_x) * dx
            y0 = (1.0 - ref_y) * dy
            y1 = (float(e_sn[idx]) - ref_y) * dy
        else:
            pidx = parent_id[idx] - 1
            if pidx < 0 or pidx >= len(domains):
                raise ValueError("domain d%02d 的 parent_id=%d 不合法。" % (dom_id, parent_id[idx]))
            parent = domains[pidx]
            dx = parent.dx / float(ratio[idx])
            dy = parent.dy / float(ratio[idx])
            x0 = parent.x0 + float(i_start[idx] - 1) * parent.dx
            y0 = parent.y0 + float(j_start[idx] - 1) * parent.dy
            x1 = x0 + float(e_we[idx] - 1) * dx
            y1 = y0 + float(e_sn[idx] - 1) * dy

        domains.append(
            Domain(
                dom_id=dom_id,
                parent_id=parent_id[idx],
                parent_grid_ratio=ratio[idx],
                i_parent_start=i_start[idx],
                j_parent_start=j_start[idx],
                e_we=e_we[idx],
                e_sn=e_sn[idx],
                dx=dx,
                dy=dy,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
            )
        )
    return domains


def xy_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    inv_transformer: Transformer,
    ref_x_proj: float,
    ref_y_proj: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lon, lat = inv_transformer.transform(np.asarray(x) + ref_x_proj, np.asarray(y) + ref_y_proj)
    return np.asarray(lon), np.asarray(lat)


def domain_border_lonlat(
    domain: Domain,
    inv_transformer: Transformer,
    ref_x_proj: float,
    ref_y_proj: float,
    n_per_edge: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    xs_bottom = np.linspace(domain.x0, domain.x1, n_per_edge)
    ys_bottom = np.full_like(xs_bottom, domain.y0)
    ys_right = np.linspace(domain.y0, domain.y1, n_per_edge)
    xs_right = np.full_like(ys_right, domain.x1)
    xs_top = np.linspace(domain.x1, domain.x0, n_per_edge)
    ys_top = np.full_like(xs_top, domain.y1)
    ys_left = np.linspace(domain.y1, domain.y0, n_per_edge)
    xs_left = np.full_like(ys_left, domain.x0)

    xs = np.concatenate([xs_bottom, xs_right, xs_top, xs_left, xs_bottom[:1]])
    ys = np.concatenate([ys_bottom, ys_right, ys_top, ys_left, ys_bottom[:1]])
    return xy_to_lonlat(xs, ys, inv_transformer, ref_x_proj, ref_y_proj)


def domain_corners_lonlat(
    domain: Domain,
    inv_transformer: Transformer,
    ref_x_proj: float,
    ref_y_proj: float,
) -> List[Tuple[float, float]]:
    xs = np.array([domain.x0, domain.x1, domain.x1, domain.x0], dtype=float)
    ys = np.array([domain.y0, domain.y0, domain.y1, domain.y1], dtype=float)
    lon, lat = xy_to_lonlat(xs, ys, inv_transformer, ref_x_proj, ref_y_proj)
    return [(float(lon[i]), float(lat[i])) for i in range(4)]


def domain_center_lonlat(
    domain: Domain,
    inv_transformer: Transformer,
    ref_x_proj: float,
    ref_y_proj: float,
) -> Tuple[float, float]:
    cx, cy = domain.center_xy
    lon, lat = xy_to_lonlat(np.array([cx]), np.array([cy]), inv_transformer, ref_x_proj, ref_y_proj)
    return float(lon[0]), float(lat[0])


def extent_for_domains(
    domains: Sequence[Domain],
    inv_transformer: Transformer,
    ref_x_proj: float,
    ref_y_proj: float,
    pad_frac: float = 0.06,
) -> List[float]:
    all_lons: List[np.ndarray] = []
    all_lats: List[np.ndarray] = []
    for dom in domains:
        lon, lat = domain_border_lonlat(dom, inv_transformer, ref_x_proj, ref_y_proj, n_per_edge=80)
        all_lons.append(lon)
        all_lats.append(lat)

    lons = np.concatenate(all_lons)
    lats = np.concatenate(all_lats)
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))

    lon_pad = max((lon_max - lon_min) * pad_frac, 0.02)
    lat_pad = max((lat_max - lat_min) * pad_frac, 0.02)
    return [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad]


def satellite_zoom_for_extent(extent: Sequence[float]) -> int:
    lon_span = abs(extent[1] - extent[0])
    lat_span = abs(extent[3] - extent[2])
    span = max(lon_span, lat_span)
    if span > 18.0:
        return 4
    if span > 9.0:
        return 5
    if span > 4.5:
        return 6
    if span > 2.0:
        return 7
    if span > 1.0:
        return 8
    if span > 0.45:
        return 9
    return 10


def add_geo_grid(ax, extent: Sequence[float], small: bool = False, rotate_lat: bool = False) -> None:
    """Add lon/lat grid labels with 2 decimal places."""
    fontsize = 11 if small else 12
    x_ticks = np.linspace(extent[0], extent[1], 5)
    y_ticks = np.linspace(extent[2], extent[3], 5)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        linestyle="--",
        alpha=0.75,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(x_ticks)
    gl.ylocator = mticker.FixedLocator(y_ticks)
    gl.xformatter = LongitudeFormatter(number_format=".2f", degree_symbol="°")
    gl.yformatter = LatitudeFormatter(number_format=".2f", degree_symbol="°")
    gl.xlabel_style = {"size": fontsize, "fontname": "Times New Roman"}
    ystyle = {"size": fontsize, "fontname": "Times New Roman"}
    if rotate_lat:
        ystyle["rotation"] = 90
    gl.ylabel_style = ystyle


def add_background(ax, extent: Sequence[float], tiler: Optional[EsriWorldImagery], offline: bool) -> None:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if offline or tiler is None:
        ax.stock_img()
        ax.add_feature(cfeature.LAND, linewidth=0.2, alpha=0.25)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, alpha=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.5)
    else:
        zoom = satellite_zoom_for_extent(extent)
        ax.add_image(tiler, zoom, interpolation="bilinear")
        ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="white", alpha=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.45, edgecolor="white", alpha=0.8)


def draw_domain_box(
    ax,
    domain: Domain,
    inv_transformer: Transformer,
    ref_x_proj: float,
    ref_y_proj: float,
    color: str,
    linewidth: float = 1.6,
    label: bool = True,
    label_size: int = 14,
) -> None:
    lon, lat = domain_border_lonlat(domain, inv_transformer, ref_x_proj, ref_y_proj)
    line = ax.plot(
        lon,
        lat,
        transform=ccrs.PlateCarree(),
        color=color,
        linewidth=linewidth,
        zorder=8,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 1.4, foreground="black", alpha=0.65), pe.Normal()])

    if label:
        # Place label just outside the top-left corner of the domain box.
        bx = domain.x0 - 0.02 * (domain.x1 - domain.x0)
        by = domain.y1 + 0.02 * (domain.y1 - domain.y0)
        blon, blat = xy_to_lonlat(
            np.array([bx]), np.array([by]),
            inv_transformer, ref_x_proj, ref_y_proj,
        )
        clon, clat = float(blon[0]), float(blat[0])
        txt = ax.text(
            clon,
            clat,
            domain.label,
            transform=ccrs.PlateCarree(),
            fontsize=label_size,
            fontweight="bold",
            ha="right",
            va="bottom",
            color="white",
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="black", edgecolor=color, alpha=0.80, linewidth=0.8),
        )
        txt.set_path_effects([pe.Stroke(linewidth=1.0, foreground="black"), pe.Normal()])


def lonlat_to_ax_data(ax, lon: float, lat: float) -> Tuple[float, float]:
    x, y = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
    return float(x), float(y)


def connect_lonlat_to_axes_point(
    fig,
    ax_from,
    ax_to,
    lon: float,
    lat: float,
    to_axes_xy: Tuple[float, float],
    color: str = "0.42",
    linewidth: float = 0.95,
) -> None:
    """Connect a lon/lat point in one map panel to a fractional point on another panel.

    This is used for clean inset-style guide lines.  The source point is tied to
    an actual WRF-domain corner, while the target is tied to the border of the
    enlarged panel, so only two neat guide lines are drawn instead of connecting
    all corresponding geographic corners.
    """
    xy_from = lonlat_to_ax_data(ax_from, lon, lat)
    con = ConnectionPatch(
        xyA=xy_from,
        xyB=to_axes_xy,
        coordsA=ax_from.transData,
        coordsB=ax_to.transAxes,
        axesA=ax_from,
        axesB=ax_to,
        arrowstyle="-",
        color=color,
        linewidth=linewidth,
        linestyle="-",
        alpha=0.92,
        clip_on=False,
        zorder=30,
    )
    con.set_path_effects([pe.Stroke(linewidth=linewidth + 1.25, foreground="white", alpha=0.78), pe.Normal()])
    fig.add_artist(con)


def connect_zoom_guides(
    fig,
    ax_left,
    ax_zoom,
    ax_terrain,
    d03: Domain,
    d05: Domain,
    inv_transformer: Transformer,
    ref_x_proj: float,
    ref_y_proj: float,
) -> None:
    """Draw exactly two guide lines for each zoom relationship.

    1) Left panel -> right-top panel:
       d03 right-bottom/right-top corners connect to the left-bottom/left-top
       corners of the d03-d05 zoom panel.
    2) Right-top panel -> right-bottom panel:
       d05 bottom-left/bottom-right corners connect to the top-left/top-right
       corners of the terrain panel.
    """
    d03_corners = domain_corners_lonlat(d03, inv_transformer, ref_x_proj, ref_y_proj)
    d05_corners = domain_corners_lonlat(d05, inv_transformer, ref_x_proj, ref_y_proj)

    # domain_corners_lonlat order: bottom-left, bottom-right, top-right, top-left.
    d03_bottom_right = d03_corners[1]
    d03_top_right = d03_corners[2]
    connect_lonlat_to_axes_point(fig, ax_left, ax_zoom, d03_bottom_right[0], d03_bottom_right[1], (0.0, 0.0))
    connect_lonlat_to_axes_point(fig, ax_left, ax_zoom, d03_top_right[0], d03_top_right[1], (0.0, 1.0))

    d05_bottom_left = d05_corners[0]
    d05_bottom_right = d05_corners[1]
    connect_lonlat_to_axes_point(fig, ax_zoom, ax_terrain, d05_bottom_left[0], d05_bottom_left[1], (0.0, 1.0))
    connect_lonlat_to_axes_point(fig, ax_zoom, ax_terrain, d05_bottom_right[0], d05_bottom_right[1], (1.0, 1.0))


def read_geo_em_terrain(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            "找不到 %s。d05 海拔图需要 geogrid 输出文件 geo_em.d05.nc；"
            "请先运行 geogrid.exe，或用 --geo 指向正确文件。" % path
        )

    def squeeze_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError("变量维度不是二维或含 Time 的二维变量，实际维度：%s" % (arr.shape,))
        return arr.astype(float)

    # The Windows netCDF libraries used in this environment cannot open a
    # path containing the Chinese workspace directory name, even though the
    # regular filesystem API can.  Use an ASCII temporary copy solely for the
    # read and remove it immediately after the variables are loaded.
    fd, temporary_name = tempfile.mkstemp(prefix="geo_em_", suffix=".nc")
    os.close(fd)
    temporary_path = Path(temporary_name)
    shutil.copyfile(path, temporary_path)

    try:
        try:
            import xarray as xr

            with xr.open_dataset(temporary_path) as ds:
                hgt_name = "HGT_M" if "HGT_M" in ds.variables else "HGT"
                lat_name = "XLAT_M" if "XLAT_M" in ds.variables else ("XLAT" if "XLAT" in ds.variables else None)
                lon_name = "XLONG_M" if "XLONG_M" in ds.variables else ("XLONG" if "XLONG" in ds.variables else None)
                if hgt_name not in ds.variables or lat_name is None or lon_name is None:
                    raise KeyError("geo_em 文件中需要 HGT_M/HGT、XLAT_M/XLAT、XLONG_M/XLONG。")
                hgt = squeeze_2d(ds[hgt_name].values)
                lat = squeeze_2d(ds[lat_name].values)
                lon = squeeze_2d(ds[lon_name].values)
                return lon, lat, hgt
        except ImportError:
            from netCDF4 import Dataset

            with Dataset(str(temporary_path)) as ds:
                hgt_name = "HGT_M" if "HGT_M" in ds.variables else "HGT"
                lat_name = "XLAT_M" if "XLAT_M" in ds.variables else ("XLAT" if "XLAT" in ds.variables else None)
                lon_name = "XLONG_M" if "XLONG_M" in ds.variables else ("XLONG" if "XLONG" in ds.variables else None)
                if hgt_name not in ds.variables or lat_name is None or lon_name is None:
                    raise KeyError("geo_em 文件中需要 HGT_M/HGT、XLAT_M/XLAT、XLONG_M/XLONG。")
                hgt = squeeze_2d(ds.variables[hgt_name][:])
                lat = squeeze_2d(ds.variables[lat_name][:])
                lon = squeeze_2d(ds.variables[lon_name][:])
                return lon, lat, hgt
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def nominal_d05_coordinates(shape, anchor_ij=(87, 100)):
    """DEPRECATED (2026-08-07): anchor-based nominal d05 coordinates.

    The registration audit (internal QA record, 2026-08-07) showed the
    original anchor was wrong: the real WRF grid places the reference point at
    geo_em cell (81, 4) (38.0147 N, 100.2425 E); anchoring it at grid index
    (87, 100) offsets the d05 terrain by roughly +6 rows / +96 columns
    (~12 km).  make_figures_npj.py now reads the real coordinates from
    d05_geographic_context.npz.  This helper is kept only for the legacy
    fig1_2_repro scripts, which are not part of the build or the release.
    """

    ny, nx = shape
    station_lat = 38.0142
    station_lon = 100.2421
    spacing_m = 111.11
    metres_per_degree = 111319.4907
    lat_1d = station_lat + (np.arange(ny) - anchor_ij[0]) * spacing_m / metres_per_degree
    lon_1d = station_lon + (np.arange(nx) - anchor_ij[1]) * spacing_m / (
        metres_per_degree * np.cos(np.deg2rad(station_lat))
    )
    return np.meshgrid(lon_1d, lat_1d)


def add_terrain_panel(ax, geo_file: Path, extent: Sequence[float]):
    _, _, hgt = read_geo_em_terrain(geo_file)
    lon, lat = nominal_d05_coordinates(hgt.shape)
    ax.set_extent(
        [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        crs=ccrs.PlateCarree(),
    )

    finite = np.isfinite(hgt)
    if not finite.any():
        raise ValueError("%s 中的 HGT_M 全部为缺测值。" % geo_file)
    hmin = float(np.nanmin(hgt))
    hmax = float(np.nanmax(hgt))

    mesh = ax.pcolormesh(
        lon,
        lat,
        hgt,
        cmap="terrain",
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    if hmax > hmin:
        step = nice_contour_step(hmax - hmin)
        first = math.ceil(hmin / step) * step
        last = math.floor(hmax / step) * step
        levels = np.arange(first, last + 0.5 * step, step)
        if len(levels) >= 2:
            cs = ax.contour(
                lon,
                lat,
                hgt,
                levels=levels,
                colors="black",
                linewidths=0.35,
                alpha=0.70,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            ax.clabel(cs, inline=True, fmt="%.0f", fontsize=8)

    # The station star marks the processed d05 grid cell (87, 100) on the
    # geo_em grid itself; the numeric index is documented in the text and
    # must not be printed on the figure.
    station_i, station_j = 87, 100
    star_lon = float(lon[station_i, station_j])
    star_lat = float(lat[station_i, station_j])
    star = ax.scatter(
        [star_lon],
        [star_lat],
        marker="*",
        s=210,
        facecolor="red",
        edgecolor="black",
        linewidth=1.2,
        transform=ccrs.PlateCarree(),
        zorder=35,
    )
    star.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white", alpha=0.85), pe.Normal()])
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.35)
    add_geo_grid(ax, extent, small=True, rotate_lat=True)
    # Selected 3-km parent cells: the flattest (0,1) and roughest (4,4) of the
    # 36 canonical parents, annotated with the standard deviation of the d05
    # HGT within each cell (nominal coordinates, consistent with the manuscript).
    for r0, c0, color in [(0, 27, "#2c5d87"), (108, 108, "#e07a32")]:
        block = hgt[r0 : r0 + 27, c0 : c0 + 27]
        std_m = float(np.std(block))
        ax.add_patch(
            Rectangle(
                (float(lon[r0, c0]), float(lat[r0, c0])),
                float(lon[r0, c0 + 27] - lon[r0, c0]),
                float(lat[r0 + 27, c0] - lat[r0, c0]),
                transform=ccrs.PlateCarree(),
                fill=False,
                ec=color,
                lw=1.25,
                zorder=30,
            )
        )
        ax.text(
            float(lon[r0, c0]) + 0.002,
            float(lat[r0, c0]) + 0.001,
            f"σ = {std_m:.3g} m",
            transform=ccrs.PlateCarree(),
            fontsize=7.0,
            color=color,
            zorder=31,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.7),
        )
    ax.text(
        0.50, 0.98, "(c)  d05 Terrain height",
        transform=ax.transAxes,
        fontsize=15, fontweight="bold", ha="center", va="top",
        color="black", zorder=40,
        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78),
    )
    return mesh


def nice_contour_step(data_range: float) -> float:
    """Choose a readable contour interval for terrain height."""
    if data_range <= 0:
        return 100.0
    rough = data_range / 9.0
    exponent = math.floor(math.log10(rough))
    fraction = rough / (10.0 ** exponent)
    if fraction <= 1.5:
        nice = 1.0
    elif fraction <= 3.0:
        nice = 2.0
    elif fraction <= 7.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10.0 ** exponent)


def build_figure(
    namelist_path: Path,
    geo_file: Path,
    output: Path,
    offline: bool = False,
    dpi: int = 300,
    cache_tiles: bool = False,
):
    nml = parse_namelist(namelist_path)
    geogrid = nml["geogrid"]
    _, _, inv_transformer, ref_x_proj, ref_y_proj = make_wrf_crs(geogrid)
    domains = build_domains(nml)

    if len(domains) < 5:
        raise ValueError("当前 namelist 只有 %d 层 domain，但该图要求至少 d01-d05 五层嵌套。" % len(domains))

    d03 = domains[2]
    d05 = domains[4]
    extent_all = extent_for_domains(domains[:5], inv_transformer, ref_x_proj, ref_y_proj, pad_frac=0.055)
    extent_d03_d05 = extent_for_domains(domains[2:5], inv_transformer, ref_x_proj, ref_y_proj, pad_frac=0.080)
    extent_d05 = extent_for_domains([d05], inv_transformer, ref_x_proj, ref_y_proj, pad_frac=0.0)

    tiler = None if offline else EsriWorldImagery(cache=cache_tiles)
    sat_projection = ccrs.PlateCarree() if offline else tiler.crs

    # Compute d05 geographic aspect ratio to eliminate internal whitespace.
    d05_lon_span = extent_d05[1] - extent_d05[0]
    d05_lat_span = extent_d05[3] - extent_d05[2]
    d05_aspect = d05_lon_span / max(d05_lat_span, 1e-6)

    # Compute d03-d05 aspect ratio for the zoom panel in the Web-Mercator
    # projection actually used for the satellite imagery.  A panel sized from
    # the raw degree aspect is wider than the tile coverage, which leaves a
    # blank band at the panel top once the projection enforces its aspect.
    zoom_lon_span = extent_d03_d05[1] - extent_d03_d05[0]
    zoom_lat_span = extent_d03_d05[3] - extent_d03_d05[2]
    _merc = ccrs.Mercator()
    _ll = ccrs.PlateCarree()
    _mx0, _my0 = _merc.transform_point(extent_d03_d05[0], extent_d03_d05[2], _ll)
    _mx1, _my1 = _merc.transform_point(extent_d03_d05[1], extent_d03_d05[3], _ll)
    zoom_merc_aspect = (_mx1 - _mx0) / max(_my1 - _my0, 1e-6)

    # SCI-paper figure: moderate width, taller for vertical breathing room.
    fig_w, fig_h = 13.5, 8.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Right-side panels: terrain and zoom share the same left edge x.
    right_x = 0.615
    terrain_h = 0.420  # figure fraction
    terrain_w = terrain_h * d05_aspect * fig_h / fig_w
    terrain_y = 0.075

    # Square-ish zoom panel matching the Web-Mercator aspect of its extent,
    # sized to fit the vertical gap above the terrain panel.
    zoom_h = 0.420
    zoom_w = zoom_h * zoom_merc_aspect * fig_h / fig_w
    zoom_y = terrain_y + terrain_h + 0.045

    # Left panel fills remaining space.
    left_pos = [0.035, 0.085, right_x - 0.070, 0.845]
    terrain_pos = [right_x, terrain_y, terrain_w, terrain_h]
    zoom_pos = [right_x, zoom_y, zoom_w, zoom_h]

    # Colorbar snug against the terrain panel.
    cbar_gap = 0.005
    cbar_w = 0.012
    cbar_h = terrain_h * 0.85
    cbar_pos = [right_x + terrain_w + cbar_gap,
                terrain_y + (terrain_h - cbar_h) / 2,
                cbar_w, cbar_h]

    ax_left = fig.add_axes(left_pos, projection=sat_projection)
    ax_zoom = fig.add_axes(zoom_pos, projection=sat_projection)
    ax_terrain = fig.add_axes(terrain_pos, projection=ccrs.PlateCarree())
    cax = fig.add_axes(cbar_pos)

    # When Cartopy enforces geographic aspect ratio, a map may shrink inside its
    # assigned rectangle.  These anchors push the actual map frames toward each
    # other, reducing the apparent blank space between the left and right panels.
    ax_left.set_anchor("E")
    ax_zoom.set_anchor("W")
    ax_terrain.set_anchor("W")

    # SCI-paper style: visible spines with subtle linewidth for all panels.
    for ax in (ax_left, ax_zoom, ax_terrain):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("0.3")

    # Domain colors are chosen for visibility over satellite imagery.
    colors = ["#ff3b30", "#ffcc00", "#34c759", "#00c7ff", "#bf5af2"]

    # Left: d01-d05 (only label d01, d02, d03 on boundaries).
    add_background(ax_left, extent_all, tiler, offline)
    for dom, color in zip(domains[:5], colors):
        draw_domain_box(
            ax_left,
            dom,
            inv_transformer,
            ref_x_proj,
            ref_y_proj,
            color=color,
            linewidth=1.55 if dom.dom_id < 5 else 1.85,
            label=dom.dom_id <= 3,
            label_size=14,
        )
    add_geo_grid(ax_left, extent_all, small=False)
    ax_left.set_title("(a)  WRF nested domains: d01–d05", fontsize=15, fontweight="bold", pad=6, loc="center")

    # Right-top: d03-d05 zoom.
    add_background(ax_zoom, extent_d03_d05, tiler, offline)
    for dom, color in zip(domains[2:5], colors[2:5]):
        draw_domain_box(
            ax_zoom,
            dom,
            inv_transformer,
            ref_x_proj,
            ref_y_proj,
            color=color,
            linewidth=1.75 if dom.dom_id < 5 else 2.05,
            label=True,
            label_size=14,
        )
    add_geo_grid(ax_zoom, extent_d03_d05, small=True, rotate_lat=True)
    ax_zoom.set_title("(b)  Zoom: d03–d05", fontsize=15, fontweight="bold", pad=5, loc="center")

    # Right-bottom: d05 terrain.
    terrain_mesh = add_terrain_panel(ax_terrain, geo_file, extent_d05)
    cb = fig.colorbar(terrain_mesh, cax=cax, orientation="vertical")
    cb.set_label("Elevation (m)", fontsize=12, fontname="Times New Roman")
    cb.ax.tick_params(labelsize=11)

    # Inset-style connection lines: two clean guide lines per zoom.
    connect_zoom_guides(
        fig,
        ax_left,
        ax_zoom,
        ax_terrain,
        d03,
        d05,
        inv_transformer,
        ref_x_proj,
        ref_y_proj,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    # Treat output as a stem so the same source figure produces the three
    # submission formats used by the rest of the figure pipeline.
    output_stem = output.with_suffix("")
    # Matplotlib 3.11 can return a broken tight bounding box for Cartopy
    # GeoAxes (it may omit most of the left map and crop the saved figure).
    # Keep the complete figure canvas and add a stable margin for the
    # geographic tick labels instead of relying on GeoAxes.get_tightbbox().
    save_bbox = fig.bbox_inches.padded(0.25)
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white", bbox_inches=save_bbox, pad_inches=0)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches=save_bbox, pad_inches=0)
    fig.savefig(
        output_stem.with_suffix(".tif"),
        dpi=500,
        facecolor="white",
        bbox_inches=save_bbox,
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    return domains, output_stem


def print_domain_summary(domains: Sequence[Domain]) -> None:
    print("\nParsed WRF domains:")
    print("%-5s %-7s %-7s %-8s %-8s %-10s %-10s" % ("dom", "parent", "ratio", "e_we", "e_sn", "dx(m)", "dy(m)"))
    for d in domains:
        print(
            "%-5s %-7d %-7d %-8d %-8d %-10.3f %-10.3f"
            % (d.label, d.parent_id, d.parent_grid_ratio, d.e_we, d.e_sn, d.dx, d.dy)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read ./namelist.wps and draw a three-panel WRF nested-domain figure."
    )
    parser.add_argument("--namelist", default="./namelist.wps", help="Path to namelist.wps. Default: ./namelist.wps")
    parser.add_argument("--geo", default="./geo_em.d05.nc", help="Path to geo_em.d05.nc. Default: ./geo_em.d05.nc")
    parser.add_argument("--output", default="./wrf_domain_overview", help="Output figure stem. Default: ./wrf_domain_overview")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI. Default: 300")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download satellite tiles; use Cartopy's offline Natural Earth background instead.",
    )
    parser.add_argument(
        "--cache-tiles",
        action="store_true",
        help="Cache downloaded satellite tiles locally through Cartopy.",
    )
    parser.add_argument(
        "--print-domains",
        action="store_true",
        help="Print parsed domain summary after plotting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    namelist_path = Path(args.namelist).expanduser().resolve()
    geo_file = Path(args.geo).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not namelist_path.exists():
        print("找不到 namelist 文件：%s" % namelist_path, file=sys.stderr)
        return 2

    try:
        domains, output_path = build_figure(
            namelist_path=namelist_path,
            geo_file=geo_file,
            output=output,
            offline=args.offline,
            dpi=args.dpi,
            cache_tiles=args.cache_tiles,
        )
    except Exception as exc:
        # Satellite tile errors often appear only during fig.savefig(). Redraw once offline.
        if not args.offline:
            print("\n卫星底图绘制失败，准备自动切换到离线 Natural Earth 背景。", file=sys.stderr)
            print("原始错误：%s" % exc, file=sys.stderr)
            try:
                domains, output_path = build_figure(
                    namelist_path=namelist_path,
                    geo_file=geo_file,
                    output=output,
                    offline=True,
                    dpi=args.dpi,
                    cache_tiles=False,
                )
            except Exception as exc2:
                print("\n离线重绘也失败：%s" % exc2, file=sys.stderr)
                return 1
        else:
            print("绘图失败：%s" % exc, file=sys.stderr)
            return 1

    if args.print_domains:
        print_domain_summary(domains)
    print("\n完成：%s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
