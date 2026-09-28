"""From the released native component tensors to the diagnostic archives.

This module closes the gap between the raw ``U10``/``V10`` tensors distributed in
the data Release and the derived ``*_cell_time.npz`` archives the rest of the
pipeline consumes.  It is the public mirror of the internal producers
(``mc_moments.fine_moments`` / ``parent_vectors_d02`` and
``phase05_interpolation.interpolate_full``); the two agree to floating-point
round-off, and ``reproduce_comment.py --from-raw`` asserts that agreement
before using a regenerated archive.

Layout of the released tensors (one file per case, per nest, per component)::

    U10_d02_<tag>.pt   V10_d02_<tag>.pt    native coarse, (4320, 162, 162)
    U10_d05_<tag>.pt   V10_d05_<tag>.pt    fine,          (4320, 162, 162) at 0.111 km

Coordinate convention (identical on both nests; zero relative grid rotation):
axis 0 is the projection row (south -> north) and axis 1 the projection column
(west -> east), so the 27x27 d05 cells covering one d02 cell are exactly one
axis-aligned block and the aggregation is a plain reshape-mean.

Frozen definitions
------------------
``|U_c|``       native coarse vector magnitude
``S_v``         magnitude of the equal-weight mean fine-cell vector
``S_s``         equal-weight mean of the fine-cell scalar speeds
``J``           ``S_s - S_v >= 0``
``Delta_mag``   ``S_v - |U_c|``
``Delta_total`` ``S_s - |U_c| = Delta_mag + J``  (exact identity)
``sigma_s``     population SD (``ddof=0``) of the 729 fine-cell speeds
``var_par``     fine-vector variance projected on the mean-vector direction
``var_perp``    transverse remainder; ``var_par + var_perp = tr(C)``
``J2nd``        ``var_perp / (2 S_v)``, the leading small-fluctuation term

All means use equal weights over the 729 fine cells; no area weighting is
applied anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import contract as C

# ---------------------------------------------------------------------------
# block aggregation
# ---------------------------------------------------------------------------


def _to64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def parent_block_means(field: np.ndarray, block: int = C.BLOCK) -> np.ndarray:
    """``(n, 162, 162) -> (n, 36)`` equal-weight 27x27 block means, row-major."""
    n = field.shape[0]
    g = field.shape[1] // block
    r = field.reshape(n, g, block, g, block).mean(axis=(2, 4))
    return r.reshape(n, g * g)


# ---------------------------------------------------------------------------
# raw tensor access
# ---------------------------------------------------------------------------


def load_component(raw_dir, case: str, nest: str, comp: str) -> np.ndarray:
    """Read one released component tensor, e.g. ``load_component(d, 'B', 'd05', 'U10')``.

    The Release stores plain ``.pt`` files (torch serialisation); ``weights_only``
    keeps the load to tensors and basic types.
    """
    import torch

    # Release convention: <CASE>/<COMP>_<NEST>.pt.  The tag-suffixed spelling
    # used by the upstream WRF working tree is accepted as a fallback so the
    # same code runs against either layout.
    tag = C.CASE_META[case]["tag"]
    for name in (f"{comp}_{nest}.pt", f"{comp}_{nest}_{tag}.pt"):
        path = Path(raw_dir) / name
        if path.is_file():
            break
    else:
        raise FileNotFoundError(
            f"missing released tensor {comp}_{nest}.pt under {raw_dir}; the data "
            f"Release must be unpacked next to the code (see RELEASE/README.md)")
    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    arr = np.asarray(loaded.numpy() if hasattr(loaded, "numpy") else loaded)
    if arr.ndim != 3 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{comp}_{nest} is not a numeric 3-D tensor: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{comp}_{nest} contains non-finite values")
    return arr


def native_parents(raw_dir, case: str) -> dict:
    """Native d02 parent vectors, ``(4320, 36)``: ``u02p``, ``v02p``, ``U_c``.

    The archived d02 field is a cellwise 27x27 replication of the native 6x6
    values, so the block mean recovers the parent vector exactly; the function
    asserts that the stored shape is the expected ``(RAW_N, 162, 162)`` before
    reducing, so a truncated or rescaled release fails loudly instead of
    silently producing a wrong coarse state.
    """
    u = load_component(raw_dir, case, "d02", "U10")
    v = load_component(raw_dir, case, "d02", "V10")
    expected = (C.RAW_N, C.GRID, C.GRID)
    if u.shape != expected or v.shape != expected:
        raise ValueError(f"d02 shapes {u.shape}, {v.shape} != {expected}")
    # The released d02 tensors are a 27x27 replication of each native parent
    # vector.  Sampling one corner is only valid under that contract; validate
    # it in chunks so a malformed tensor cannot silently choose a corner value.
    for name, field in (("U10_d02", u), ("V10_d02", v)):
        for a in range(0, C.RAW_N, 128):
            b = min(a + 128, C.RAW_N)
            blocks = field[a:b].reshape(-1, C.COARSE_EDGE_NATIVE, C.BLOCK,
                                        C.COARSE_EDGE_NATIVE, C.BLOCK)
            residual = np.abs(blocks - blocks[:, :, :1, :, :1]).max()
            if float(residual) > 1e-6:
                raise ValueError(f"{name} is not 27x27 replicated; "
                                 f"max block residual {float(residual):.3e}")
    u6 = np.ascontiguousarray(u[:, ::C.BLOCK, ::C.BLOCK]).astype(np.float64)
    v6 = np.ascontiguousarray(v[:, ::C.BLOCK, ::C.BLOCK]).astype(np.float64)
    u02p = u6.reshape(C.RAW_N, C.N_COARSE)
    v02p = v6.reshape(C.RAW_N, C.N_COARSE)
    return {"u02p": u02p, "v02p": v02p, "U_c": np.hypot(u02p, v02p)}


# ---------------------------------------------------------------------------
# moments (mirror of mc_moments.fine_moments)
# ---------------------------------------------------------------------------


def fine_moments(raw_dir, case: str, chunk: int = 512) -> dict:
    """Fine-grid moments per state per parent cell, ``(4320, 36)`` float64.

    Keys: ``s_bar``, ``u_bar``, ``v_bar``, ``U_vec``, ``J``, ``S``, ``var_par``,
    ``var_perp``, ``trC``, ``J2nd``.
    """
    u = load_component(raw_dir, case, "d05", "U10")
    v = load_component(raw_dir, case, "d05", "V10")
    if u.shape != (C.RAW_N, C.GRID, C.GRID):
        raise ValueError(f"d05 shape {u.shape} != {(C.RAW_N, C.GRID, C.GRID)}")

    keys = ["s_bar", "u_bar", "v_bar", "U_vec", "J", "S",
            "var_par", "var_perp", "trC", "J2nd"]
    out = {k: np.zeros((C.RAW_N, C.N_COARSE)) for k in keys}
    for a in range(0, C.RAW_N, chunk):
        b = min(a + chunk, C.RAW_N)
        uu, vv = _to64(u[a:b]), _to64(v[a:b])
        sp = np.hypot(uu, vv)
        ub, vb, sb = (parent_block_means(x) for x in (uu, vv, sp))
        s2 = parent_block_means(sp * sp)
        suu = parent_block_means(uu * uu) - ub * ub
        svv = parent_block_means(vv * vv) - vb * vb
        suv = parent_block_means(uu * vv) - ub * vb
        uvec = np.hypot(ub, vb)
        sd = np.sqrt(np.maximum(s2 - sb * sb, 0.0))       # population SD, ddof=0
        trc = suu + svv
        with np.errstate(divide="ignore", invalid="ignore"):
            vpar = (ub * ub * suu + 2.0 * ub * vb * suv + vb * vb * svv) / (uvec * uvec)
        vpar = np.where(uvec > 1e-9, vpar, trc)
        with np.errstate(divide="ignore", invalid="ignore"):
            j2 = (trc - vpar) / (2.0 * uvec)
        j2 = np.where(uvec > 1e-9, j2, 0.0)
        sl = slice(a, b)
        out["s_bar"][sl], out["u_bar"][sl], out["v_bar"][sl] = sb, ub, vb
        out["U_vec"][sl], out["S"][sl] = uvec, sd
        out["var_par"][sl], out["var_perp"][sl] = vpar, trc - vpar
        out["trC"][sl], out["J2nd"][sl] = trc, j2
    out["J"] = out["s_bar"] - out["U_vec"]
    return out


def moments_archive(raw_dir, case: str) -> dict:
    """The full ``moments_cell_time.npz`` payload.

    Key set and dtypes match the released archive: the native parents
    (``u02p``, ``v02p``, ``U02_mag``), the fine moments, and the two derived
    differences.  ``delta_dyn = U_vec - U02_mag`` is a resolved-magnitude
    mismatch; ``delta_scalar = J + delta_dyn`` is the exact identity audited in
    the manuscript.
    """
    m = fine_moments(raw_dir, case)
    par = native_parents(raw_dir, case)
    m["u02p"], m["v02p"] = par["u02p"], par["v02p"]
    m["U02_mag"] = par["U_c"]
    m["delta_dyn"] = m["U_vec"] - m["U02_mag"]
    m["delta_scalar"] = m["J"] + m["delta_dyn"]
    # scalar QA receipts, stored as 0-d float64 exactly as in the frozen archive
    m["_J_min"] = np.float64(m["J"].min())
    m["_identity_max_abs"] = np.float64(
        np.abs(m["delta_scalar"] - (m["delta_dyn"] + m["J"])).max())
    # d05 domain-mean context, accumulated in float64 (the tensors are float32;
    # summing them directly in float32 loses ~1e-6 m/s, which is why the frozen
    # archive is reproduced only when the cast happens first)
    d05_u = _to64(load_component(raw_dir, case, "d05", "U10"))
    d05_v = _to64(load_component(raw_dir, case, "d05", "V10"))
    m["d05_domain_u"] = d05_u.mean(axis=(1, 2))
    m["d05_domain_v"] = d05_v.mean(axis=(1, 2))
    m["d05_domain_speed"] = np.hypot(d05_u, d05_v).mean(axis=(1, 2))
    return m


# ---------------------------------------------------------------------------
# interpolation (mirror of phase05_interpolation.interpolate_full)
# ---------------------------------------------------------------------------


def _interp(u6: np.ndarray, v6: np.ndarray, mode: str) -> tuple:
    from torch.nn.functional import interpolate as F_interp
    import torch

    with torch.no_grad():
        ui = F_interp(torch.from_numpy(u6[:, None]), size=(C.GRID, C.GRID),
                      mode=mode, align_corners=False)[:, 0].numpy()
        vi = F_interp(torch.from_numpy(v6[:, None]), size=(C.GRID, C.GRID),
                      mode=mode, align_corners=False)[:, 0].numpy()
    return ui.astype(np.float64), vi.astype(np.float64)


def interpolate_case(raw_dir, case: str, chunk: int = 720) -> dict:
    """Bilinear/bicubic refinement, plain and coarse-vector-constrained.

    Returns, per mode, the ``(4320, 36)`` fields ``u_bar``, ``v_bar``, ``s_bar``,
    ``sd`` plus their ``_c`` constrained counterparts and the largest per-component
    residual of the constraint (which the caller asserts is below ``1e-6 m/s``).
    """
    u = load_component(raw_dir, case, "d02", "U10")
    v = load_component(raw_dir, case, "d02", "V10")
    par = native_parents(raw_dir, case)
    u02p, v02p = par["u02p"], par["v02p"]
    u6 = np.ascontiguousarray(u[:, ::C.BLOCK, ::C.BLOCK])
    v6 = np.ascontiguousarray(v[:, ::C.BLOCK, ::C.BLOCK])

    out = {"u02p": u02p, "v02p": v02p, "U_c": par["U_c"]}
    for mode in C.MODES:
        tabs = {k: np.zeros((C.RAW_N, C.N_COARSE)) for k in
                ("u_bar", "v_bar", "s_bar", "sd",
                 "u_bar_c", "v_bar_c", "s_bar_c", "sd_c")}
        max_resid = 0.0
        for a in range(0, C.RAW_N, chunk):
            b = min(a + chunk, C.RAW_N)
            ui64, vi64 = _interp(u6[a:b], v6[a:b], mode)
            sp = np.hypot(ui64, vi64)
            tabs["u_bar"][a:b] = parent_block_means(ui64)
            tabs["v_bar"][a:b] = parent_block_means(vi64)
            tabs["s_bar"][a:b] = parent_block_means(sp)
            s2 = parent_block_means(sp * sp)
            tabs["sd"][a:b] = np.sqrt(np.maximum(
                s2 - tabs["s_bar"][a:b] ** 2, 0.0))
            # uniform per-block component offset restoring the native block mean
            off_u = u02p[a:b] - tabs["u_bar"][a:b]
            off_v = v02p[a:b] - tabs["v_bar"][a:b]
            ou = np.repeat(np.repeat(off_u.reshape(-1, C.COARSE_EDGE_NATIVE,
                                                    C.COARSE_EDGE_NATIVE),
                                     C.BLOCK, 1), C.BLOCK, 2)
            ov = np.repeat(np.repeat(off_v.reshape(-1, C.COARSE_EDGE_NATIVE,
                                                    C.COARSE_EDGE_NATIVE),
                                     C.BLOCK, 1), C.BLOCK, 2)
            spc = np.hypot(ui64 + ou, vi64 + ov)
            tabs["u_bar_c"][a:b] = parent_block_means(ui64 + ou)
            tabs["v_bar_c"][a:b] = parent_block_means(vi64 + ov)
            tabs["s_bar_c"][a:b] = parent_block_means(spc)
            s2c = parent_block_means(spc * spc)
            tabs["sd_c"][a:b] = np.sqrt(np.maximum(
                s2c - tabs["s_bar_c"][a:b] ** 2, 0.0))
            max_resid = max(max_resid,
                            float(np.abs(tabs["u_bar_c"][a:b] - u02p[a:b]).max()),
                            float(np.abs(tabs["v_bar_c"][a:b] - v02p[a:b]).max()))
        out[mode] = {
            "u_bar": tabs["u_bar"], "v_bar": tabs["v_bar"],
            "s_bar": tabs["s_bar"], "sd": tabs["sd"],
            "u_vec": np.hypot(tabs["u_bar"], tabs["v_bar"]),
            "u_bar_c": tabs["u_bar_c"], "v_bar_c": tabs["v_bar_c"],
            "s_bar_c": tabs["s_bar_c"], "sd_c": tabs["sd_c"],
            "u_vec_c": np.hypot(tabs["u_bar_c"], tabs["v_bar_c"]),
            "constrained_max_component_residual": max_resid,
        }
        # Keep the held-out component metrics beside the regenerated archive.
        # The public NPZ comparison below checks the arrays; these values bind
        # the raw source chain to the two conservation tables that are consumed
        # by the Level-1 gate.
        a_eval, b_eval = C.HELDOUT
        metrics = {}
        for label, ubar, vbar in (
                (mode, tabs["u_bar"], tabs["v_bar"]),
                (f"{mode}_constrained", tabs["u_bar_c"], tabs["v_bar_c"])):
            du = ubar[a_eval:b_eval] - u02p[a_eval:b_eval]
            dv = vbar[a_eval:b_eval] - v02p[a_eval:b_eval]
            metrics[label] = {
                "E_vec_rmse": float(np.sqrt((du ** 2 + dv ** 2).mean())),
                "L1_max_comp_resid": float(
                    max(np.abs(du).max(), np.abs(dv).max())),
            }
        out[mode]["heldout_conservation"] = metrics
    return out


def interpolation_archive(raw_dir, case: str) -> dict:
    """The full ``interpolation_cell_time_full.npz`` payload.

    Per mode (``bilinear``, ``bicubic``) and per variant (plain and ``_constrained``)
    the archive stores the interpolated mean scalar speed, its spread, the
    aggregation gap ``J`` and the two differences, so downstream evaluation can
    recompute Q_J and Q_sigma_s without re-interpolating.
    """
    res = interpolate_case(raw_dir, case)
    flat = {"U02_mag": res["U_c"]}
    for mode in C.MODES:
        d = res[mode]
        J = d["s_bar"] - d["u_vec"]
        dsc = d["s_bar"] - res["U_c"]
        Jc = d["s_bar_c"] - d["u_vec_c"]
        dscc = d["s_bar_c"] - res["U_c"]
        flat[f"{mode}_mean_scalar_speed"] = d["s_bar"]
        flat[f"{mode}_spread"] = d["sd"]
        flat[f"{mode}_J"] = J
        flat[f"{mode}_delta_scalar"] = dsc
        flat[f"{mode}_delta_dyn"] = dsc - J
        flat[f"{mode}_constrained_mean_scalar_speed"] = d["s_bar_c"]
        flat[f"{mode}_constrained_spread"] = d["sd_c"]
        flat[f"{mode}_constrained_J"] = Jc
        flat[f"{mode}_constrained_delta_scalar"] = dscc
        flat[f"{mode}_constrained_delta_dyn"] = dscc - Jc
    # Private receipt: not part of the distributed NPZ key set, but consumed
    # by ``--from-raw`` to bind regenerated held-out metrics to the Release
    # conservation tables.  The caller deliberately excludes underscore keys
    # from the element-wise archive comparison.
    flat["_heldout_conservation"] = {
        mode: res[mode]["heldout_conservation"][mode]
        for mode in C.MODES
    }
    flat["_heldout_conservation"].update({
        f"{mode}_constrained": res[mode]["heldout_conservation"][
            f"{mode}_constrained"]
        for mode in C.MODES
    })
    return flat
