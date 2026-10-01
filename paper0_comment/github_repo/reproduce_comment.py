#!/usr/bin/env python
"""Reproduce every statistic and figure of the Comment on wind downscaling.

One entry point.  From a clean checkout plus the frozen data Release:

    python reproduce_comment.py --data /path/to/release_data \
        --figures --out outputs

``--data`` must contain one directory per case (``A``, ``B``, ``C``) holding
the two frozen archives ``moments_cell_time.npz`` and
``interpolation_cell_time_full.npz``.  See ``RELEASE/README.md``.

The script is deliberately free of any absolute path, of any dependency on the
private upstream tensors, and of any network access.  It reports PASS/FAIL per
reproduced quantity against the values quoted in the manuscript.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from comment_pipeline import contract as C          # noqa: E402
from comment_pipeline import (states, interpolation, evaluation, figures,  # noqa: E402
                             source_chain)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="root of the frozen data Release (contains A/, B/, C/)")
    ap.add_argument("--out", type=Path, default=HERE / "outputs",
                    help="output directory for tables and figures")
    ap.add_argument("--figures", action="store_true", help="render Figs. 1-3")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the machine-readable reproduction report here")
    ap.add_argument("--from-raw", action="store_true",
                    help="also regenerate both archives from the released native .pt "
                         "tensors via comment_pipeline.source_chain and assert they "
                         "match the distributed .npz archives")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(name: str, got, lo, hi, unit: str = "",
              strict_hi: bool = False) -> None:
        if got is None:
            ok = False
        elif strict_hi:
            ok = (lo < got < hi)
        else:
            ok = (lo <= got <= hi)
        checks.append({"name": name, "value": got, "expected": [lo, hi],
                       "strict_hi": bool(strict_hi), "unit": unit,
                       "pass": bool(ok)})
        lhs = "(" if strict_hi else "["
        rhs = ")" if strict_hi else "]"
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {got!r}  "
              f"in {lhs}{lo}, {hi}{rhs}")

    print("Case and window contract")
    print(f"  archive states      {C.ARCHIVE_STATES}")
    print(f"  excluded prefix     {C.EXCLUDED}  (not the upstream spin-up)")
    print(f"  analysis window     {C.ANALYSIS}")
    print(f"  held-out window     {C.HELDOUT}")
    print(f"  coarse cells        {C.N_COARSE} contiguous {C.COARSE_EDGE}x{C.COARSE_EDGE}")
    print(f"  fine cells per cell {C.BLOCK}x{C.BLOCK} = {C.BLOCK*C.BLOCK}")
    print(f"  speed mask          >= {C.SPEED_MASK} m/s, population SD (ddof={C.DDof})")

    if args.from_raw:
        print("\nSource chain: raw tensors -> diagnostic archives")
        for case in C.CASES:
            d = args.data / C.CASE_META[case]["dir"]
            if not d.is_dir():
                d = args.data / case
            regen_m = source_chain.moments_archive(d, case)
            ref_m = {k: np.asarray(v) for k, v in states.load(args.data, case).items()}

            # Compare the public payload only.  ``moments_archive`` also carries
            # private scalar receipts (``_J_min`` and ``_identity_max_abs``)
            # which are intentionally excluded by ``states.load``.
            regen_m_public = {k: v for k, v in regen_m.items()
                              if not str(k).startswith("_")}
            moment_keys_ok = set(regen_m_public) == set(ref_m)
            moment_arrays_ok = moment_keys_ok
            worst = None
            for k in sorted(ref_m):
                try:
                    a = np.asarray(ref_m[k], dtype=np.float64)
                    b = np.asarray(regen_m_public[k], dtype=np.float64)
                except (KeyError, TypeError, ValueError):
                    moment_arrays_ok = False
                    continue
                if a.shape != b.shape or not np.isfinite(a).all() \
                        or not np.isfinite(b).all():
                    moment_arrays_ok = False
                    continue
                err = float(np.abs(a - b).max())
                worst = err if worst is None else max(worst, err)
            check(f"{case}: moments raw comparison key/shape/finite contract",
                  moment_keys_ok and moment_arrays_ok, True, True)
            check(f"{case}: moments archive reproduced from raw",
                  worst if moment_arrays_ok and worst is not None else None,
                  0.0, 1e-9)
            regen_i = source_chain.interpolation_archive(d, case)
            ref_i = interpolation.load(args.data, case)
            regen_i_public = {k: v for k, v in regen_i.items()
                              if not str(k).startswith("_")}
            interp_keys_ok = set(regen_i_public) == set(ref_i)
            interp_arrays_ok = interp_keys_ok
            worst_i = None
            for k in sorted(ref_i):
                try:
                    a = np.asarray(ref_i[k], dtype=np.float64)
                    b = np.asarray(regen_i_public[k], dtype=np.float64)
                except (KeyError, TypeError, ValueError):
                    interp_arrays_ok = False
                    continue
                if a.shape != b.shape or not np.isfinite(a).all() \
                        or not np.isfinite(b).all():
                    interp_arrays_ok = False
                    continue
                err = float(np.abs(a - b).max())
                worst_i = err if worst_i is None else max(worst_i, err)
            check(f"{case}: interpolation raw comparison key/shape/finite contract",
                  interp_keys_ok and interp_arrays_ok, True, True)
            check(f"{case}: interpolation archive reproduced from raw",
                  worst_i if interp_arrays_ok and worst_i is not None else None,
                  0.0, 1e-9)

            # The two published conservation tables are consumed by the
            # Level-1 gate.  Recompute both held-out metrics from the raw
            # interpolated fields and require exact method coverage plus
            # finite agreement, so a table cannot be swapped independently of
            # the raw source chain.
            raw_cons = regen_i.get("_heldout_conservation", {})
            ref_cons = evaluation.load_conservation_table(args.data, case)

            def _metric_close(name: str) -> bool:
                methods = evaluation.METHODS
                # The held-out table records the component maximum only for
                # constrained controls; ordinary controls are judged by their
                # vector RMSE and intentionally have no max-component field.
                if name == "L1_max_comp_resid":
                    methods = tuple(m for m in methods if m.endswith("_constrained"))
                for method in methods:
                    got = raw_cons.get(method, {}).get(name)
                    expected = ref_cons.get(method, {}).get(name)
                    try:
                        if (got is None or expected is None
                                or not np.isfinite(got)
                                or not np.isfinite(expected)
                                or abs(float(got) - float(expected)) > 1e-9):
                            return False
                    except (TypeError, ValueError):
                        return False
                return True

            check(f"{case}: held-out conservation RMSE table matches raw",
                  _metric_close("E_vec_rmse"), True, True)
            check(f"{case}: held-out max-component table matches raw",
                  _metric_close("L1_max_comp_resid"), True, True)

    summary_rows = []
    matrix_rows: list[dict] = []
    mech_rows: list[dict] = []
    for case in C.CASES:
        print(f"\nCase {case}  ({C.CASE_META[case]['site']}, "
              f"{C.CASE_META[case]['event']})")
        m = states.load(args.data, case)
        iv = interpolation.load(args.data, case)
        cons = evaluation.load_conservation_table(args.data, case)

        # Structural integrity: shapes, finiteness, non-negativity and the
        # component->magnitude/derived identities.  These gate everything else,
        # so a corrupt archive cannot reach the reporting stage at all.
        problems = states.validate_archive(m, case)
        check(f"{case}: archive structurally valid and self-consistent",
              len(problems) == 0, True, True)
        for pr in problems:
            print(f"    !! {pr}")

        # The interpolation archive carries the predicted fields; validating it
        # is what stops a collapsed (4320, 36) -> (4320, 1) prediction table or a
        # rescaled moment set from being treated as a valid cell-time archive.
        iv_problems = interpolation.validate_archive(iv, case)
        check(f"{case}: interpolation archive structurally valid",
              len(iv_problems) == 0, True, True)
        for pr in iv_problems:
            print(f"    !! {pr}")

        a = states.accounting(m)
        summary_rows.append({"case": case, **a})
        anch = C.ANCHORS_ACCOUNTING[case]
        tol = C.ANCHORS_ACCOUNTING_TOL

        check(f"{case}: domain-mean d_total (m/s)", round(a["dt_mean"], 3),
              C.ANCHORS_DT[case][0], C.ANCHORS_DT[case][1], "m/s")
        check(f"{case}: J/d_total", round(a["J_over_dt"], 2), 1.00, 1.03)
        check(f"{case}: <E_vec> (m/s)", round(a["E_vec_mean"], 5),
              anch[2] - tol["E_vec"], anch[2] + tol["E_vec"], "m/s")
        check(f"{case}: E_vec/J (%)", round(100 * a["E_vec_over_J"], 1),
              anch[3] - tol["E_vec_over_J_percent"],
              anch[3] + tol["E_vec_over_J_percent"], "%")
        check(f"{case}: median d_theta (deg)",
              round(a["dtheta_median_masked"], 4),
              anch[4] - tol["dtheta_median"], anch[4] + tol["dtheta_median"],
              "deg")

        rows = evaluation.recovery_matrix(m, iv, case, conservation=cons)
        if not cons:
            print("  WARNING: interpolation_metrics.csv absent; conservation "
                  "error reported as unavailable rather than fabricated")
        for r in rows:
            evaluation.check_row(r, checks, check)
            # Conservation error must match the frozen table exactly when the
            # table is present.  This is the check that previously did not exist.
            if cons:
                if r["constrained"]:
                    check(f"{case}: {r['method']} L1 within bound",
                          r["L1_metric_rmse"], 0.0,
                          C.ANCHORS_CONSERVATION["constrained_hi"], "m/s")
                else:
                    check(f"{case}: {r['method']} L1 in ordinary band",
                          r["L1_metric_rmse"],
                          C.ANCHORS_CONSERVATION["ordinary_lo"],
                          C.ANCHORS_CONSERVATION["ordinary_hi"], "m/s")
        matrix_rows.extend(rows)

        rho = states.correlations(m)
        mech_rows.append({"case": case, **rho})
        # The rank correlations are quoted to two decimals in the Comment; the
        # anchor is that quoted value, not an arbitrary [-1, 1] envelope.
        san = C.ANCHORS_SPEARMAN[case]
        stol = C.ANCHORS_SPEARMAN_TOL
        check(f"{case}: Spearman J ~ transverse", round(rho["sp_J_perp"], 2),
              san[0] - stol, san[0] + stol)
        check(f"{case}: Spearman J ~ sigma_s", round(rho["sp_J_S"], 2),
              san[1] - stol, san[1] + stol)

        if args.figures:
            figures.render_case(case, m, iv, out)
            print(f"  figures row {case} rendered")

    (out / "reproduced_three_state_summary.csv").write_text(
        states.summary_csv(summary_rows), encoding="utf-8")
    (out / "reproduced_two_level_matrix.csv").write_text(
        states.summary_csv(matrix_rows), encoding="utf-8")
    (out / "reproduced_mechanism_correlations.csv").write_text(
        states.summary_csv(mech_rows), encoding="utf-8")
    if args.figures:
        # Fig.1(d) and Fig.2(b) need the raw held-out / analysis cell-times, so
        # reload the moments once rather than relying on the summary tables.
        fig_moments = {c: states.load(args.data, c) for c in C.CASES}
        figures.render_panels(summary_rows, out, fig_moments)
        figures.render_mechanism(mech_rows, out, fig_moments)
        figures.render_recovery(matrix_rows, out)

        # Coverage: the released Fig.2(b) draws a 650-point subsample for each
        # of three cases, and Fig.1(d) draws 1,500 held-out cell-times for each.
        # An empty or truncated panel must fail, not pass silently.
        n_b = int((out / "fig2_panel_b_points.txt").read_text())
        n_d = int((out / "fig1_panel_d_points.txt").read_text())
        check("Fig.2(b) scatter coverage (650 x 3 cases)", n_b,
              C.PLOT_SUBSAMPLE * len(C.CASES), C.PLOT_SUBSAMPLE * len(C.CASES),
              "points")
        check("Fig.1(d) scatter coverage (1500 x 3 cases)", n_d,
              1500 * len(C.CASES), 1500 * len(C.CASES), "points")

        # Window consistency: panel (c) reports a held-out ratio, so both the
        # numerator and the denominator must come from the held-out window, and
        # panel (b) must carry both windows per case.  These are the checks that
        # would have caught the earlier analysis-J / held-out-E_vec mix-up.
        for row in summary_rows:
            c = row["case"]
            e_vec = row["E_vec_mean"]
            j_expect = row["J_mean_eval"]
            ratio_expect = 100 * e_vec / j_expect
            check(f"{c}: panel c ratio is same-window",
                  abs(ratio_expect - row["E_vec_over_J_eval_percent"]), 0.0, 1e-9)
            check(f"{c}: panel c uses held-out J",
                  abs(row["J_mean_eval"] - row["J_mean"]) > 1e-6, True, True)
        both = sum(1 for r in summary_rows
                   if r.get("J_mean") is not None
                   and r.get("J_mean_eval") is not None)
        check("panel (b) carries both windows per case", both,
              len(C.CASES), len(C.CASES), "cases with both windows")
        for stem in ("Fig1_three_state_multicase", "Fig2_vector_organization_multicase",
                     "Fig3_two_level_recovery_multicase"):
            p = out / f"{stem}.pdf"
            check(f"{stem}.pdf written", p.is_file() and p.stat().st_size > 5000,
                  True, True)
        print("\nFigures written to", out)

    report = {
        "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
        "contract": C.as_dict(),
        "summary": summary_rows,
        "correlations": mech_rows,
        "checks": checks,
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nChecks: {sum(c['pass'] for c in checks)}/{len(checks)}  "
          f"-> {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
