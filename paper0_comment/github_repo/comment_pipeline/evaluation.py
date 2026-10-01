"""Level-1 coarse-state conservation vs Level-2 subgrid-statistic recovery.

Conservation error (Fig.3a), as defined in the Comment caption:

    RMSE = sqrt( < || blockmean(pred) - U_c ||^2 > )

i.e. the root-mean-square of the *vector* difference between the block mean of
the predicted fine field and the native coarse vector.  This is the quantity
reported as ``E_vec_rmse`` in the frozen interpolation metrics tables.

Recovery ratio, defined for a target statistic M in {J, sigma_s}:

    Q_M = <M_hat> / <M>      domain and evaluation-window means over the
                             36 coarse cells and the 670 held-out states

Q_M = 1 is equal-mean recovery.  "Less than half" means Q_M < 0.5.

NO CONVERSION ERROR IS EVER FABRICATED HERE.  If neither the predicted
component fields nor the frozen conservation table is available for a method,
the row is reported with ``L1_metric_rmse = NaN`` and ``L1_available = False``.
It is never filled with a placeholder constant.
"""
from __future__ import annotations

import numpy as np

from . import contract as C
from .interpolation import (conservation_residual,
                            predicted_states, recovery_ratios, reference)

WINDOW = C.HELDOUT

# The four controls, in the order the Comment presents them.
METHODS = ("bilinear", "bicubic", "bilinear_constrained", "bicubic_constrained")


def conservation_from_table(table: dict, method: str) -> float | None:
    """Read the frozen vector-RMSE conservation error for one method.

    ``table`` is the parsed ``interpolation_metrics`` mapping produced by
    :func:`load_conservation_table`.  Returns ``None`` when absent.
    """
    if not table:
        return None
    entry = table.get(method)
    if entry is None:
        return None
    val = entry.get("E_vec_rmse")
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    # RMSE is a non-negative norm.  Treat a negative table value as malformed
    # rather than allowing it to satisfy the ordinary-control upper-bound test.
    if not np.isfinite(val) or val < 0.0:
        return None
    return val


def load_conservation_table(data_root, case: str) -> dict:
    """Assemble the conservation values for one case, by window.

    Returns ``{method: {...}}`` with these keys per method:

    ``E_vec_rmse``
        Vector-magnitude RMSE over the held-out window, from
        ``interpolation_metrics.csv``.  This is the quantity Fig. 3(a) plots.
    ``L1_max_comp_resid``
        Maximum per-component residual over the **held-out** window
        ``[3650, 4320)``, from ``conservation_eval_window.csv``.  This is the
        quantity the Level-1 control and the Comment's ``<1.1e-14`` bound are
        stated against.
    ``full_archive_max_comp_resid``
        The same residual accumulated over the **entire** archive
        ``[0, 4320)``, from ``interpolation_contract.json``.  It is a QA guard,
        not the reported value: the producer accumulates it over
        ``RAW_N = 4320`` states, so it can exceed the held-out bound.

    Any unavailable field degrades to ``None``; nothing is fabricated.
    """
    import csv
    import json

    d = data_root / C.CASE_META[case]["dir"]
    out: dict = {}

    def _slot(model: str) -> dict:
        return out.setdefault(str(model), {
            "E_vec_rmse": None,
            "L1_max_comp_resid": None,
            "full_archive_max_comp_resid": None,
        })

    # 1. Held-out window: both the vector RMSE and the component maximum.
    path = d / "interpolation_metrics.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("group") != "scalar_increment":
                    continue
                model = (row.get("model") or "").strip()
                if not model:
                    continue
                try:
                    _slot(model)["E_vec_rmse"] = float(row["E_vec_rmse"])
                except (KeyError, TypeError, ValueError):
                    pass

    # 2. Held-out component maximum, from the frozen evaluation-window table.
    epath = d / "conservation_eval_window.csv"
    if epath.is_file():
        with epath.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                model = (row.get("method") or "").strip()
                if not model:
                    continue
                raw = row.get("l1_max_comp_resid")
                if raw in (None, ""):
                    continue
                try:
                    _slot(model)["L1_max_comp_resid"] = float(raw)
                except (TypeError, ValueError):
                    pass

    # 3. Full-archive maximum, kept only as a labelled QA field.
    cpath = d / "interpolation_contract.json"
    if cpath.is_file():
        try:
            with cpath.open(encoding="utf-8") as fh:
                contract = json.load(fh)
            resid = contract.get("constrained_max_component_residual") or {}
        except (OSError, ValueError):
            resid = {}
        for model, mx in resid.items():
            try:
                mx = float(mx)
            except (TypeError, ValueError):
                continue
            name = str(model)
            # The contract keys by base mode; the plotted row is the
            # constrained variant of that mode.
            target = f"{name}_constrained" if f"{name}_constrained" in out else name
            _slot(target)["full_archive_max_comp_resid"] = mx

    return out


def recovery_matrix(m: dict, iv: dict, case: str = "",
                    conservation: dict | None = None) -> list[dict]:
    """One row per (case, method): the twelve outputs of the Comment.

    Conservation error is obtained in this priority order, and never invented:

      1. recomputed from predicted components, when the Release ships them;
      2. read from the frozen held-out-window table, when present;
      3. ``NaN`` with ``L1_available = False``.

    The reported ``L1_max_comp_resid`` is always the **held-out window**
    ``[3650, 4320)`` value, matching the Comment.  The full-archive maximum is
    carried alongside in ``L1_full_archive_max_comp`` purely as a QA guard.

    Recovery ratios are always computed from the archived moments.
    """
    sl = slice(*WINDOW)
    ref = reference(m, WINDOW)
    native_u, native_v = m["u02p"][sl], m["v02p"][sl]
    native_mag = m["U02_mag"][sl]

    rows = []
    for method in METHODS:
        constrained = method.endswith("_constrained")
        has_components = (f"{method}_u" in iv and f"{method}_v" in iv)
        entry = (conservation or {}).get(method, {})
        full_arch = entry.get("full_archive_max_comp_resid")

        if has_components:
            u, v = iv[f"{method}_u"][sl], iv[f"{method}_v"][sl]
            # Audit the supplied output exactly; correction belongs to generation.
            pred = predicted_states(u, v, native_mag)
            max_comp, rmse = conservation_residual(u, v, native_u, native_v)
            source = "recomputed"
        else:
            pred = {"J": iv[f"{method}_J"][sl],
                    "spread": iv[f"{method}_spread"][sl],
                    "delta_scalar": iv[f"{method}_delta_scalar"][sl]}
            rmse = conservation_from_table(conservation or {}, method)
            max_comp = entry.get("L1_max_comp_resid")
            if max_comp is None:
                max_comp = float("nan")
            source = "frozen_table" if rmse is not None else "unavailable"

        q = recovery_ratios(pred, ref)

        # Normalise to a single float before any numeric test: a missing table
        # must degrade to NaN and FAIL, never raise and never fabricate.
        rmse_val = float("nan") if rmse is None else float(rmse)
        try:
            max_comp_val = float(max_comp)
        except (TypeError, ValueError, OverflowError):
            max_comp_val = float("nan")

        # A maximum absolute residual is a finite, non-negative quantity by
        # definition.  Reject malformed frozen tables before applying the
        # upper tolerance; otherwise an injected negative value could satisfy
        # ``max_comp <= tolerance`` and be reported as exact conservation.
        max_comp_valid = bool(np.isfinite(max_comp_val) and max_comp_val >= 0.0)

        # The declared criterion is the maximum absolute component residual.
        # RMSE remains a plotted diagnostic, never a substitute certificate.
        rmse_valid = bool(np.isfinite(rmse_val) and rmse_val >= 0.0)
        l1_available = bool(rmse_valid and max_comp_valid)
        l1_pass = bool(l1_available and max_comp_val <= C.L1_TOLERANCE)
        l1_status = ("PASS" if l1_pass else "FAIL") if l1_available else "UNAVAILABLE"

        rows.append({
            "case": case or m.get("case", ""), "method": method,
            "constrained": constrained,
            "L1_available": l1_available,
            "L1_criterion": "max_abs_component_residual",
            "L1_status": l1_status,
            "L1_source": source,
            "L1_max_comp_resid": max_comp_val,
            "L1_full_archive_max_comp": (float(full_arch)
                                         if full_arch is not None
                                         else float("nan")),
            "L1_metric_rmse": rmse_val,
            "L1_pass": l1_pass,
            "Q_J": q["J"], "Q_S": q["spread"], "Q_delta": q["delta_scalar"],
            "L2_J_recovered": bool(q["J"] >= C.L2_HALF),
            "L2_S_recovered": bool(q["spread"] >= C.L2_HALF),
        })
    return rows


def check_row(row: dict, checks: list, check) -> None:
    """Report the twelve outputs against the manuscript's two claims.

    A conservation check is only emitted when the conservation error is
    actually available; an unavailable value is reported as a FAIL, never
    silently skipped or satisfied by a placeholder.
    """
    tag = f"{row['case']}: {row['method']}"

    # Ordinary controls report the archived RMSE as a separate diagnostic.
    # Its availability does not certify the maximum-residual constraint.
    rmse_valid = np.isfinite(row["L1_metric_rmse"]) and row["L1_metric_rmse"] >= 0
    if not rmse_valid or (row["constrained"] and not row["L1_available"]):
        check(f"{tag} conservation error available", None, 0.0, np.inf, "m/s")
    elif row["constrained"]:
        check(f"{tag} conserves coarse vector", row["L1_max_comp_resid"],
              0.0, C.L1_TOLERANCE, "m/s")
    else:
        check(f"{tag} fails conservation", row["L1_metric_rmse"],
              C.L1_TOLERANCE, np.inf, "m/s")

    # The Comment's claim is "recovers less than half", i.e. 0 < Q < 0.5.
    # A negative or non-finite recovery moment is impossible and must fail, so
    # the lower bound is 0.0 rather than an arbitrary -10.0.
    check(f"{tag} Q_J in (0, 0.5)", row["Q_J"], 0.0, C.L2_HALF, "", strict_hi=True)
    check(f"{tag} Q_sigma_s in (0, 0.5)", row["Q_S"], 0.0, C.L2_HALF, "", strict_hi=True)
