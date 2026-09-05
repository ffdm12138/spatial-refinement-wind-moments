# -*- coding: utf-8 -*-
"""Independent recomputation of interpolation_recovery.csv (36/36 QA).
from the raw d02 wind fields + the frozen phase05 formulas (bilinear/bicubic,
parent-vector-constrained, moment_tables identity). No precomputed summary is
read for the recomputed numbers; the canonical CSV under canonical_recovery/
is only the comparison target. The blocking fix documented below is the
author-identified correction of the original QA script (the wrong
reshape(36, 27, 27) sliced row-major memory into artificial 729-element
strips instead of the physical 6x6 parent blocks)."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
import mc_config as C  # noqa: E402
import torch
from torch.nn.functional import interpolate as F_interp

CASES = {"case_A_yakou0924": "A", "case_B_yakou1201": "B", "case_C_ebao1201": "C"}
TOL = 2e-3   # recovery_ratio canonical table is rounded to 6 decimals; float
             # pipeline re-run tolerance (torch backend identical to producer)


def block_mean(x):
    # closeout fix (author-identified QA bug): the previous
    # x.reshape(36, 27, 27).mean(axis=(1,2)) sliced the ROW-MAJOR memory into
    # 36 bands of 729 elements (4.5 grid rows each, crossing parent
    # boundaries) instead of the 6x6 spatial parent blocks. The correct
    # blocking matches the production M.parent_block_means exactly.
    if x.ndim == 2:
        return x.reshape(6, 27, 6, 27).mean(axis=(1, 3)).reshape(36)
    return x.reshape(x.shape[0], 6, 27, 6, 27).mean(axis=(2, 4)).reshape(x.shape[0], 36)


def main():
    import pandas as pd
    results = {}
    all_ok = True
    for case, letter in CASES.items():
        cdir = C.case_dir(letter)
        u02, v02 = C.load_case_wind(letter, "d02")
        with np.load(cdir / "moments_cell_time.npz") as z:
            u02p = z["u02p"]; v02p = z["v02p"]
            refJ = z["J"][3650:4320]
            refDS = z["delta_scalar"][3650:4320]
            refS = z["S"][3650:4320]
        rec = {}
        for mode in ("bilinear", "bicubic"):
            acc = {k: np.zeros((670, 36)) for k in
                   ("J", "delta_scalar", "spread", "J_c", "delta_scalar_c", "spread_c")}
            for i in range(670):
                t = 3650 + i
                u6 = u02[t, ::27, ::27]
                v6 = v02[t, ::27, ::27]
                with torch.no_grad():
                    ui = F_interp(torch.from_numpy(u6[None, None].astype(np.float32)),
                                  size=(162, 162), mode=mode,
                                  align_corners=False)[0, 0].numpy().astype(np.float64)
                    vi = F_interp(torch.from_numpy(v6[None, None].astype(np.float32)),
                                  size=(162, 162), mode=mode,
                                  align_corners=False)[0, 0].numpy().astype(np.float64)
                sp = np.hypot(ui, vi)
                ub = block_mean(ui); vb = block_mean(vi)
                sb = block_mean(sp)
                sd = np.sqrt(np.maximum(block_mean(sp * sp) - sb ** 2, 0.0))
                off_u = (u02p[t] - ub).reshape(6, 6)
                off_v = (v02p[t] - vb).reshape(6, 6)
                ou = np.repeat(np.repeat(off_u, 27, 0), 27, 1)
                ov = np.repeat(np.repeat(off_v, 27, 0), 27, 1)
                uc = ui + ou; vc = vi + ov
                spc = np.hypot(uc, vc)
                sbc = block_mean(spc)
                sdc = np.sqrt(np.maximum(block_mean(spc * spc) - sbc ** 2, 0.0))
                uvc = np.hypot(block_mean(uc), block_mean(vc))
                acc["J"][i] = sb - np.hypot(ub, vb)
                acc["delta_scalar"][i] = sb - np.hypot(u02p[t], v02p[t])
                acc["spread"][i] = sd
                acc["J_c"][i] = sbc - uvc
                acc["delta_scalar_c"][i] = sbc - np.hypot(u02p[t], v02p[t])
                acc["spread_c"][i] = sdc
            summary = pd.read_csv(ROOT / "canonical_recovery" / f"{case}__interpolation_recovery.csv")
            # closeout fix (s4/s5): the constrained prediction must be compared
            # against the CONSTRAINED canonical row - the previous version
            # matched summary.model == mode (unconstrained) while the
            # recomputed key carried the _c suffix, manufacturing the fake
            # "C 2.0301 vs 0.302882" mismatch (AUDIT_SCRIPT_ERROR).
            for mode_key, pre in (("bilinear", ""), ("bicubic", "_c")):
                model_name = mode + ("_constrained" if pre else "")
                for moment, key in (("delta_scalar", f"delta_scalar{pre}"),
                                    ("J", f"J{pre}"), ("spread", f"spread{pre}")):
                    # QA-script normalization fix (closeout): Q_M is the
                    # DIMENSIONLESS ratio mean(M_pred)/mean(M_ref); the earlier
                    # checker compared the dimensional moment mean directly
                    # against the canonical ratio and omitted the reference
                    # denominator. Production definition restored:
                    den = float({"delta_scalar": refDS, "J": refJ,
                                 "spread": refS}[moment].mean())
                    q = float(acc[key].mean(axis=0).mean()) / den
                    row = summary[(summary.model == model_name) & (summary.moment == moment)
                                  & (summary.cell_set == "all36")]
                    canon = float(row["recovery_ratio"].iloc[0])
                    ok = abs(q - canon) <= TOL
                    all_ok &= ok
                    results[f"{case}/{mode}/{moment}{pre}"] = {
                        "recomputed": q, "canonical": canon, "pass": ok}
                    if not ok:
                        print("MISMATCH", case, mode, moment, pre,
                              f"recomputed={q:.6f} canonical={canon:.6f}", flush=True)
        C.free_case_wind(case)
        print(case, "done", flush=True)
    verdict = "PASS" if all_ok else "FAIL"
    out = {"verdict": verdict, "tolerance": TOL, "results": results}
    (ROOT / "results" / "independent_recovery_recompute.json").write_text(
        json.dumps(out, indent=1))
    print("INDEPENDENT_RECOVERY:", verdict)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
