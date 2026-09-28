"""Independent re-implementations for the audit (PLAN §27).

Every function here recomputes a core quantity WITHOUT importing numeric
helpers from mc_moments (the primary implementation).  Different algorithms /
code paths are used on purpose:

  * moments: explicit per-cell slicing with direct deviation sums
    (primary: reshape + moment identity E[x^2] - E[x]^2)
  * population S: direct numpy .std(ddof=0) on each block
  * J2nd: legacy deviation-covariance formula (du/dv fields)
  * 333/1000 m aggregation: explicit python loops over sub-blocks
    (primary: nested reshape means)
  * bilinear interpolation: closed-form align_corners=False coordinate map
    with clamped (edge-extension) sampling in pure numpy (primary: torch)
  * lambda_iso: the frozen per-profile walking algorithm transcribed from
    the legacy production script (primary: vectorized pair sums)
  * recovery ratios: recomputed from the WRITTEN cell-time CSV outputs
  * bootstrap: independently written resampling + einsum-based means with
    the same seed (identical index streams by construction)

Output: audit/independent_reproduction.csv
        (quantity / case / primary_value / independent_value / abs_diff /
         tolerance / status)
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402  (paths/constants only; no numerics)

ROWS: list[dict] = []


def add(quantity: str, case: str, primary, independent, tol: float) -> None:
    diff = abs(float(primary) - float(independent))
    ROWS.append({"quantity": quantity, "case": case,
                 "primary_value": float(primary),
                 "independent_value": float(independent),
                 "abs_diff": diff, "tolerance": tol,
                 "status": "PASS" if diff <= tol else "FAIL"})


# ---------------------------------------------------------------------------
# 1-3: moments, S, J2nd on a deterministic state subsample
# ---------------------------------------------------------------------------
def indep_moments(case: str, states: np.ndarray) -> dict[str, float]:
    u05, v05 = C.load_case_wind(case, "d05")
    u02, v02 = C.load_case_wind(case, "d02")
    acc = {k: 0.0 for k in ["delta_dyn", "J", "delta_scalar", "S", "J2nd",
                            "identity_max"]}
    count = 0
    for t in states:
        uu = u05[t].astype(np.float64)
        vv = v05[t].astype(np.float64)
        sp = np.sqrt(uu * uu + vv * vv)
        u2 = u02[t].astype(np.float64)
        v2 = v02[t].astype(np.float64)
        for r in range(6):
            for c in range(6):
                sl = (slice(r * 27, r * 27 + 27), slice(c * 27, c * 27 + 27))
                ub = uu[sl].mean()
                vb = vv[sl].mean()
                sb = sp[sl].mean()
                uvec = sqrt(ub * ub + vb * vb)
                Up = sqrt(u2[sl].mean() ** 2 + v2[sl].mean() ** 2)
                J = sb - uvec
                dyn = uvec - Up
                dsc = sb - Up
                acc["identity_max"] = max(acc["identity_max"],
                                          abs(dsc - (dyn + J)))
                acc["J"] += J
                acc["delta_dyn"] += dyn
                acc["delta_scalar"] += dsc
                acc["S"] += sp[sl].std(ddof=0)
                du = uu[sl] - ub
                dv = vv[sl] - vb
                suu = (du * du).mean()
                suv = (du * dv).mean()
                svv = (dv * dv).mean()
                trc = suu + svv
                if uvec > 1e-9:
                    vpar = (ub * ub * suu + 2 * ub * vb * suv + vb * vb * svv) / uvec ** 2
                    acc["J2nd"] += (trc - vpar) / (2 * uvec)
                count += 1
    C.free_case_wind(case)
    return {k: (v / count if k != "identity_max" else v) for k, v in acc.items()}


def primary_moments_subsample(case: str, states: np.ndarray) -> dict[str, float]:
    with np.load(C.case_dir(case) / "moments_cell_time.npz") as z:
        return {"delta_dyn": float(z["delta_dyn"][states].mean()),
                "J": float(z["J"][states].mean()),
                "delta_scalar": float(z["delta_scalar"][states].mean()),
                "S": float(z["S"][states].mean()),
                "J2nd": float(z["J2nd"][states].mean())}


# ---------------------------------------------------------------------------
# 4: scale aggregation by explicit loops
# ---------------------------------------------------------------------------
def indep_scale(case: str, states: np.ndarray, k: int) -> dict[str, float]:
    u05, v05 = C.load_case_wind(case, "d05")
    sub = 27 // k
    accJ = accS = 0.0
    count = 0
    for t in states:
        uu = u05[t].astype(np.float64)
        vv = v05[t].astype(np.float64)
        for r in range(6):
            for c in range(6):
                speeds = []
                usum = vsum = 0.0
                for br in range(sub):
                    for bc in range(sub):
                        r0 = r * 27 + br * k
                        c0 = c * 27 + bc * k
                        um = uu[r0:r0 + k, c0:c0 + k].mean()
                        vm = vv[r0:r0 + k, c0:c0 + k].mean()
                        speeds.append(sqrt(um * um + vm * vm))
                        usum += um
                        vsum += vm
                sp = np.asarray(speeds)
                uvec = sqrt((usum / sub ** 2) ** 2 + (vsum / sub ** 2) ** 2)
                accJ += sp.mean() - uvec
                accS += sp.std(ddof=0)
                count += 1
    C.free_case_wind(case)
    return {"J_l": accJ / count, "S_l": accS / count}


def primary_scale_subsample(case: str, states: np.ndarray, k: int) -> dict[str, float]:
    with np.load(C.case_dir(case) / "scale_cell_time.npz") as z:
        return {"J_l": float(z[f"J_{k}"][states].mean()),
                "S_l": float(z[f"S_{k}"][states].mean())}


# ---------------------------------------------------------------------------
# 5: manual bilinear interpolation (align_corners=False, edge clamp)
# ---------------------------------------------------------------------------
def manual_bilinear(native: np.ndarray, size: int = 162) -> np.ndarray:
    """(6,6) -> (size,size) reproducing torch F.interpolate bilinear
    align_corners=False, computed in float64."""
    src = native.astype(np.float64)
    n_in = src.shape[0]
    x = (np.arange(size) + 0.5) * n_in / size - 0.5
    x0 = np.floor(x).astype(int)
    frac = x - x0
    i0 = np.clip(x0, 0, n_in - 1)
    i1 = np.clip(x0 + 1, 0, n_in - 1)
    w1 = np.clip(frac, 0.0, 1.0)
    w0 = 1.0 - w1
    a = src[np.ix_(i0, i0)] * w0[:, None] * w0[None, :]
    b = src[np.ix_(i0, i1)] * w0[:, None] * w1[None, :]
    c = src[np.ix_(i1, i0)] * w1[:, None] * w0[None, :]
    d = src[np.ix_(i1, i1)] * w1[:, None] * w1[None, :]
    return a + b + c + d


def indep_bilinear_check(case: str, states_rel: np.ndarray) -> dict[str, float]:
    """Max abs field deviation manual-vs-torch on evaluation states, plus an
    independently reduced Q_J(bilinear) on that subsample."""
    import torch
    from torch.nn.functional import interpolate as F_interp
    u02, v02 = C.load_case_wind(case, "d02")
    ea, eb = C.EVAL_SLICE
    max_dev = 0.0
    qj_num = qj_den = 0.0
    with np.load(C.case_dir(case) / "moments_cell_time.npz") as z:
        refJ = z["J"][ea:eb]
    for tr in states_rel:
        t = ea + tr
        u6 = u02[t, ::27, ::27]
        v6 = v02[t, ::27, ::27]
        with torch.no_grad():
            ut = F_interp(torch.from_numpy(u6[None, None]), size=(162, 162),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
            vt = F_interp(torch.from_numpy(v6[None, None]), size=(162, 162),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        um = manual_bilinear(u6)
        vm = manual_bilinear(v6)
        max_dev = max(max_dev, float(np.abs(um - ut).max()),
                      float(np.abs(vm - vt).max()))
        spm = np.sqrt(um * um + vm * vm)
        for r in range(6):
            for c in range(6):
                sl = (slice(r * 27, r * 27 + 27), slice(c * 27, c * 27 + 27))
                sb = spm[sl].mean()
                ub = um[sl].mean()
                vb = vm[sl].mean()
                qj_num += sb - sqrt(ub * ub + vb * vb)
                qj_den += refJ[tr, r * 6 + c]
    C.free_case_wind(case)
    return {"field_max_dev": max_dev, "Q_J_sub": qj_num / qj_den}


def primary_qj_subsample(case: str, states_rel: np.ndarray) -> float:
    ea, eb = C.EVAL_SLICE
    with np.load(C.case_dir(case) / "interpolation_cell_time_full.npz") as z:
        num = z["bilinear_J"][ea:eb][states_rel].mean()
    with np.load(C.case_dir(case) / "moments_cell_time.npz") as z:
        den = z["J"][ea:eb][states_rel].mean()
    return float(num / den)


# ---------------------------------------------------------------------------
# 6: lambda_iso by the frozen per-profile walk
# ---------------------------------------------------------------------------
_DIRS8 = ((1, 0, 1.0), (1, 1, 1 / sqrt(2)), (0, 1, 1.0), (-1, 1, 1 / sqrt(2)),
          (-1, 0, 1.0), (-1, -1, 1 / sqrt(2)), (0, -1, 1.0), (1, -1, 1 / sqrt(2)))


def walk_lambda_iso(z: np.ndarray, dx: float, L: float) -> float:
    n = z.shape[0]
    acc = 0.0
    scale = dx / (L * L)
    for di, dj, w in _DIRS8:
        total = 0.0
        for i in range(n):
            for j in range(n):
                if 0 <= i - di < n and 0 <= j - dj < n:
                    continue
                ii, jj = i, j
                prev = z[ii, jj]
                ii += di
                jj += dj
                while 0 <= ii < n and 0 <= jj < n:
                    cur = z[ii, jj]
                    inc = cur - prev
                    if inc > 0.0:
                        total += inc
                    prev = cur
                    ii += di
                    jj += dj
        acc += scale * w * total
    return acc / 8.0


def indep_lambda(case: str) -> np.ndarray:
    hgt = C.load_case_hgt(case)
    vals = np.zeros(36)
    for cid in range(36):
        r, c = divmod(cid, 6)
        z = hgt[r * 27:(r + 1) * 27, c * 27:(c + 1) * 27]
        vals[cid] = walk_lambda_iso(z, C.DX_M, C.L_M)
    return vals


# ---------------------------------------------------------------------------
# 7: recovery ratios recomputed from the written CSVs
# ---------------------------------------------------------------------------
def indep_recovery_from_csv(case: str) -> pd.DataFrame:
    ct = pd.read_csv(C.case_dir(case) / "interpolation_cell_time.csv")
    ref = ct[ct.method == "reference"]
    out = []
    for model in ("bilinear", "bicubic"):
        sub = ct[ct.method == model]
        for moment in ("delta_scalar", "J", "spread"):
            q = sub[moment].mean() / ref[moment].mean()
            out.append({"model": model, "moment": moment,
                        "recovery_ratio_csv": float(q)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 8: bootstrap with independently written resampling (same seed)
# ---------------------------------------------------------------------------
def indep_bootstrap(case: str) -> dict[str, float]:
    import phase12_infosources as P12   # predictions only (data, not numerics)
    P = P12.build_predictions(case)
    obs = P["obs"]["scalar"]
    pa = P["pred"][("scalar", "terrain_roughness_equation")]
    pb = P["pred"][("scalar", "bicubic")]
    n = obs.shape[0]
    rng = np.random.default_rng(
        np.random.SeedSequence(C.MASTER_SEED,
                               spawn_key=tuple(C.SEED_SPAWN[f"bootstrap_within_{case}"])))
    est = np.zeros(C.N_REPLICATES)
    # replicate ONLY the primary-block segment of the stream: primary loops
    # blocks 60 -> 120 -> 240 with one integers() call per replicate; we
    # consume the 60-block replicates first to align the stream, then draw
    # the 120-block replicates.
    for _ in range(C.N_REPLICATES):
        n_blocks = int(np.ceil(n / 60))
        rng.integers(0, n - 60 + 1, size=n_blocks)
    length = C.PRIMARY_BLOCK
    for rep in range(C.N_REPLICATES):
        n_blocks = int(np.ceil(n / length))
        starts = rng.integers(0, n - length + 1, size=n_blocks)
        idx = np.hstack([np.arange(s, s + length) for s in starts])[:n]
        om = np.einsum("tc->c", obs[idx]) / n
        ra = sqrt(float(np.einsum("c,c->", om - pa, om - pa)) / 36.0)
        pbm = np.einsum("tc->c", pb[idx]) / n
        rb = sqrt(float(np.einsum("c,c->", om - pbm, om - pbm)) / 36.0)
        est[rep] = ra - rb
    return {"dRMSE_est": float(est.mean()),
            "ci_low": float(np.percentile(est, 2.5)),
            "ci_high": float(np.percentile(est, 97.5))}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(cases: list[str]) -> pd.DataFrame:
    ROWS.clear()
    rng = np.random.default_rng(C.MASTER_SEED)   # deterministic subsamples
    del rng
    for case in cases:
        print(f"[audit-independent] case {case}")
        # deterministic subsample: every 97th retained state (35 states)
        states = np.arange(C.RETAINED_SLICE[0], C.RAW_N, 97)
        prim = primary_moments_subsample(case, states)
        ind = indep_moments(case, states)
        for q in ("delta_dyn", "J", "delta_scalar", "S", "J2nd"):
            add(f"{q}_subsample_mean", case, prim[q], ind[q], 1e-9)
        add("decomposition_identity_max_abs", case, 0.0, ind["identity_max"], 1e-9)

        st_sc = states[::6]
        for k in (3, 9):
            prim_s = primary_scale_subsample(case, st_sc, k)
            ind_s = indep_scale(case, st_sc, k)
            add(f"J_{C.SCALE_M[k]:.0f}m_subsample_mean", case,
                prim_s["J_l"], ind_s["J_l"], 1e-9)
            add(f"S_{C.SCALE_M[k]:.0f}m_subsample_mean", case,
                prim_s["S_l"], ind_s["S_l"], 1e-9)

        rel = np.arange(0, C.EVAL_SLICE[1] - C.EVAL_SLICE[0], 37)
        bl = indep_bilinear_check(case, rel)
        add("bilinear_field_manual_vs_torch_max_abs", case, 0.0,
            bl["field_max_dev"], 1e-5)
        add("Q_J_bilinear_subsample", case, primary_qj_subsample(case, rel),
            bl["Q_J_sub"], 1e-4)

        lam_ind = indep_lambda(case)
        lam_prim = pd.read_csv(C.case_dir(case) / "terrain_descriptors.csv")[
            "lambda_iso"].to_numpy(float)
        add("lambda_iso_max_abs_cell_diff", case, 0.0,
            float(np.abs(lam_ind - lam_prim).max()), 1e-10)

        rec_csv = indep_recovery_from_csv(case)
        rec_prim = pd.read_csv(C.case_dir(case) / "interpolation_recovery.csv")
        for _, row in rec_csv.iterrows():
            pv = rec_prim[(rec_prim.model == row["model"])
                          & (rec_prim.moment == row["moment"])
                          & (rec_prim.cell_set == "all36")]["recovery_ratio"].iloc[0]
            add(f"Q_{row['moment']}_{row['model']}_all36", case, float(pv),
                row["recovery_ratio_csv"], 1e-5)

        bt_ind = indep_bootstrap(case)
        bt_prim = pd.read_csv(C.ext_dir("bootstrap")
                              / f"case_{case}_paired_bootstrap.csv")
        pr = bt_prim[(bt_prim.pair == "terrain_vs_bicubic")
                     & (bt_prim.block_minutes == C.PRIMARY_BLOCK)]
        # identical index streams; einsum-vs-mean float64 ordering sits ~4e-10
        add("bootstrap_dRMSE_terrain_vs_bicubic_b120", case,
            float(pr["dRMSE_est"].iloc[0]), bt_ind["dRMSE_est"], 5e-9)
        add("bootstrap_ci_low_terrain_vs_bicubic_b120", case,
            float(pr["ci_low"].iloc[0]), bt_ind["ci_low"], 5e-9)
        add("bootstrap_ci_high_terrain_vs_bicubic_b120", case,
            float(pr["ci_high"].iloc[0]), bt_ind["ci_high"], 5e-9)

    df = pd.DataFrame(ROWS)
    C.write_df(C.ext_dir("audit") / "independent_reproduction.csv",
               df.round(12))
    n_fail = int((df.status == "FAIL").sum())
    print(f"[audit-independent] {len(df) - n_fail}/{len(df)} PASS")
    return df


if __name__ == "__main__":
    run(sys.argv[1:] or ["A"])
