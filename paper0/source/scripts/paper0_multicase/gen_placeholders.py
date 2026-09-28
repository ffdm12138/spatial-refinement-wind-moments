"""Generate BRANDED PLACEHOLDER raw datasets for Case B and Case C.

THESE ARE NOT SCIENCE DATA.  They exist only so the multicase post-processing
suite can be built, dry-run end-to-end and audited before the real December
WRF archives arrive.  Every placeholder directory carries
PLACEHOLDER_NOTICE.md and placeholder_manifest.json; the pipeline detects
placeholder status by hashing the raw files against that manifest, and every
downstream output is stamped data_status=PLACEHOLDER until the author swaps
in real data (hash mismatch flips the status to REAL).

Generation method (documented honestly; fully synthetic, deterministic):
  * Master seed 20260826; numpy SeedSequence spawn keys
    (100,) shared December synoptic driver, (101,) Case-B local fields,
    (102,) Case-C local fields, (103,) Case-C synthetic terrain.
  * Shared synoptic driver (B and C, same 2015-12-01..03 episode): AR(1)
    domain-scale wind vector (1-min cadence, ~6 h correlation time), mean
    north-westerly winter flow of a few m/s with a weak diurnal modulation.
  * Terrain: Case B reuses Case A's HGT_d05.pt byte-identically (same Yakou
    nest).  Case C synthesizes a clearly different, lower-relief
    foothill-transition field (Dayekou-like): south-high to north-low ramp,
    a few smooth ridge/valley bumps, and small-scale roughness that decays
    toward the piedmont, so descriptor distributions differ from Yakou and
    partially leave Case A's calibration range (extrapolation flags fire).
  * Fine winds are terrain-coupled: u,v = synoptic * (1 + a*hp_norm)
    + b*slope_norm*eta + c*zeta, where hp_norm is the high-pass terrain
    anomaly, slope_norm the normalized slope magnitude, and eta/zeta AR(1)
    noise fields with 3-cell spatial correlation.  Rugged cells therefore
    carry larger within-parent spread and Jensen gap, giving non-degenerate
    terrain-relationship fits.
  * d02 = native 6x6 block means of the fine vector field plus small AR(1)
    parent noise (nest-consistency delta_dyn small but nonzero), exactly
    tiled 27x27 to the archived 162x162 layout.
  * T2 (manifest-only in science) = winter Kelvin field with elevation lapse,
    diurnal cycle and mild noise; d02 analogously tiled.

Usage:
  python gen_placeholders.py [--data-root PATH] [--force]

Refuses to overwrite an existing placeholder directory unless --force; NEVER
writes into a directory that fails the placeholder-manifest hash match
(i.e. never overwrites real data).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_config as C  # noqa: E402

T = C.RAW_N
G = C.GRID
MINUTES_PER_DAY = 1440.0


def _seedseq(key: tuple) -> np.random.SeedSequence:
    return np.random.SeedSequence(C.MASTER_SEED, spawn_key=key)


def ar1_series(rng, n, rho, sigma, mean=0.0):
    eps = rng.standard_normal(n) * sigma * np.sqrt(1.0 - rho * rho)
    out = np.empty(n)
    out[0] = mean + rng.standard_normal() * sigma
    for t in range(1, n):
        out[t] = mean + rho * (out[t - 1] - mean) + eps[t]
    return out


def ar1_field(rng, n, shape, rho, sigma):
    """AR(1)-in-time noise field on a coarse spatial grid (float32)."""
    out = np.empty((n,) + shape, dtype=np.float32)
    out[0] = rng.standard_normal(shape).astype(np.float32) * sigma
    scale = sigma * np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + scale * rng.standard_normal(shape).astype(np.float32)
    return out


def upsample_bilinear(field: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsample (n, h, w) -> (n, size, size) via torch, float32."""
    import torch
    from torch.nn.functional import interpolate as F_interp
    out = np.empty((field.shape[0], size, size), dtype=np.float32)
    step = 512
    with torch.no_grad():
        for i in range(0, field.shape[0], step):
            j = min(i + step, field.shape[0])
            x = torch.from_numpy(field[i:j][:, None])
            y = F_interp(x, size=(size, size), mode="bilinear", align_corners=False)
            out[i:j] = y[:, 0].numpy()
    return out


def boxsmooth(z: np.ndarray, k: int) -> np.ndarray:
    """Separable box smooth with edge padding (odd k)."""
    pad = k // 2
    zp = np.pad(z, pad, mode="edge")
    c0 = np.vstack([np.zeros((1, zp.shape[1])), np.cumsum(zp, axis=0)])
    zs = (c0[k:] - c0[:-k]) / k                     # (n, m + 2*pad)
    c1 = np.hstack([np.zeros((zs.shape[0], 1)), np.cumsum(zs, axis=1)])
    return (c1[:, k:] - c1[:, :-k]) / k             # (n, m)


def normalized(a: np.ndarray, clip: float = 3.0) -> np.ndarray:
    s = a.std()
    if s < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - a.mean()) / s, -clip, clip)


def synth_dayekou_terrain() -> np.ndarray:
    """Synthetic Dayekou-like foothill-transition HGT (162x162 float32).

    Rows: index increases northward (matching the archived orientation), so
    the southern mountain flank sits at low row indices and the piedmont at
    high row indices.  Lower relief than Yakou by design.
    """
    rng = np.random.default_rng(_seedseq(C.SEED_SPAWN["placeholder_terrain_C"]))
    r = np.arange(G, dtype=np.float64)[:, None]
    c = np.arange(G, dtype=np.float64)[None, :]
    frac_n = r / (G - 1.0)                       # 0 = south (high), 1 = north (low)
    # foothill ramp ~3050 m (south) flattening into a ~2080 m piedmont basin;
    # the (1-frac_n)^2.4 profile makes the northern tile rows genuinely flat
    # so several Case-C cells fall BELOW Case A's calibration descriptor
    # minima and the extrapolation flags exercise meaningfully.
    base = 2080.0 + 970.0 * (1.0 - frac_n) ** 2.4
    z = base + 22.0 * (1.0 - frac_n) * np.sin(2.0 * np.pi * c / 180.0)
    # a few smooth ridge / valley structures, damped toward the piedmont
    n_bumps = 14
    rows = rng.uniform(0, G, n_bumps)
    cols = rng.uniform(0, G, n_bumps)
    amps = (rng.uniform(-90.0, 130.0, n_bumps)
            * (1.0 - 0.9 * (rows / G) ** 1.3))
    sig = rng.uniform(9.0, 26.0, n_bumps)
    for i in range(n_bumps):
        z += amps[i] * np.exp(-(((r - rows[i]) ** 2 + (c - cols[i]) ** 2)
                                / (2.0 * sig[i] ** 2)))
    # small-scale roughness decaying toward the piedmont
    coarse = rng.standard_normal((54, 54)).astype(np.float32)
    rough = upsample_bilinear(coarse[None], G)[0].astype(np.float64)
    rough = rough / max(rough.std(), 1e-9)
    amp = 1.2 + 46.0 * (1.0 - frac_n) ** 2.2
    z += amp * rough
    return np.ascontiguousarray(z, dtype=np.float32)


def terrain_modulation(hgt64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hp_norm, slope_norm) float32 static modulation patterns from terrain."""
    hp = hgt64 - boxsmooth(hgt64, C.BLOCK)
    hp_n = normalized(hp).astype(np.float32)
    gy, gx = np.gradient(hgt64, C.DX_M, C.DX_M)
    sl = np.hypot(gx, gy)
    sl_n = (sl / max(sl.mean() + 2.0 * sl.std(), 1e-9)).clip(0.0, 1.5).astype(np.float32)
    return hp_n, sl_n


def synoptic_driver() -> dict[str, np.ndarray]:
    """Shared December synoptic episode for B and C (same time axis)."""
    rng = np.random.default_rng(_seedseq(C.SEED_SPAWN["placeholder_synoptic"]))
    rho = float(np.exp(-1.0 / 360.0))            # ~6 h correlation at 1-min cadence
    # mean north-westerly winter flow: from 315 deg -> flow-to vector (+u, -v)
    u_syn = ar1_series(rng, T, rho, 0.9, mean=1.9)
    v_syn = ar1_series(rng, T, rho, 0.9, mean=-1.7)
    tmin = np.arange(T, dtype=np.float64)
    diurnal = 1.0 + 0.15 * np.sin(2.0 * np.pi * (tmin / MINUTES_PER_DAY) - 1.1)
    u_syn *= diurnal
    v_syn *= diurnal
    return {"u": u_syn.astype(np.float32), "v": v_syn.astype(np.float32),
            "tmin": tmin}


def build_case(case: str, hgt32: np.ndarray, syn: dict,
               out_dir: Path) -> dict[str, dict]:
    """Generate and save the 7-file placeholder set for one case."""
    import torch
    rng = np.random.default_rng(_seedseq(C.SEED_SPAWN[f"placeholder_case_{case}"]))
    hgt64 = hgt32.astype(np.float64)
    hp_n, sl_n = terrain_modulation(hgt64)

    rho_local = float(np.exp(-1.0 / 45.0))       # ~45-min local correlation
    a_hp, b_sl, c_bg = 0.15, 1.15, 0.55          # terrain-coupling amplitudes

    def local_noise(tag_offset: int) -> np.ndarray:
        rng_local = np.random.default_rng(
            _seedseq(C.SEED_SPAWN[f"placeholder_case_{case}"] + (tag_offset,)))
        coarse = ar1_field(rng_local, T, (54, 54), rho_local, 1.0)
        return upsample_bilinear(coarse, G)

    eta_u = local_noise(1)
    eta_v = local_noise(2)
    zeta_u = local_noise(3)
    zeta_v = local_noise(4)

    u_fine = np.empty((T, G, G), dtype=np.float32)
    v_fine = np.empty((T, G, G), dtype=np.float32)
    mod = (1.0 + a_hp * hp_n)[None]
    for i in range(0, T, 720):
        j = min(i + 720, T)
        u_fine[i:j] = (syn["u"][i:j, None, None] * mod
                       + b_sl * sl_n[None] * eta_u[i:j] + c_bg * zeta_u[i:j])
        v_fine[i:j] = (syn["v"][i:j, None, None] * mod
                       + b_sl * sl_n[None] * eta_v[i:j] + c_bg * zeta_v[i:j])
    del eta_u, eta_v, zeta_u, zeta_v

    # d02: native 6x6 block means + small AR(1) parent noise, tiled 27x27
    u6 = u_fine.reshape(T, 6, C.BLOCK, 6, C.BLOCK).mean(axis=(2, 4))
    v6 = v_fine.reshape(T, 6, C.BLOCK, 6, C.BLOCK).mean(axis=(2, 4))
    pn_u = ar1_field(rng, T, (6, 6), rho_local, 0.035)
    pn_v = ar1_field(rng, T, (6, 6), rho_local, 0.035)
    u6 = (u6 + pn_u).astype(np.float32)
    v6 = (v6 + pn_v).astype(np.float32)
    u02 = np.repeat(np.repeat(u6, C.BLOCK, axis=1), C.BLOCK, axis=2)
    v02 = np.repeat(np.repeat(v6, C.BLOCK, axis=1), C.BLOCK, axis=2)

    # T2 (manifest-only): winter Kelvin, elevation lapse, diurnal cycle
    lapse = -0.0065 * (hgt64 - hgt64.mean())
    diur = 5.5 * np.sin(2.0 * np.pi * (syn["tmin"] / MINUTES_PER_DAY) - 2.0)
    t2_noise = upsample_bilinear(ar1_field(rng, T, (27, 27), rho_local, 0.7), G)
    t2_fine = (266.0 + lapse[None] + diur[:, None, None].astype(np.float64)
               + t2_noise).astype(np.float32)
    t26 = t2_fine.reshape(T, 6, C.BLOCK, 6, C.BLOCK).mean(axis=(2, 4))
    t202 = np.repeat(np.repeat(t26, C.BLOCK, axis=1), C.BLOCK, axis=2)
    del t2_noise

    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {"U10_d05.pt": u_fine, "V10_d05.pt": v_fine,
               "U10_d02.pt": u02, "V10_d02.pt": v02,
               "T2_d05.pt": t2_fine, "T2_d02.pt": t202,
               "HGT_d05.pt": hgt32}
    files = {}
    for stem, arr in tensors.items():
        name = C.raw_filename(case, stem)
        assert np.isfinite(arr).all(), f"{case} {name}: non-finite values"
        assert arr.dtype == np.float32, f"{case} {name}: dtype {arr.dtype}"
        path = out_dir / name
        if stem == "HGT_d05.pt" and case == "B":
            # byte-identical copy of Case A's terrain (same Yakou nest)
            src = C.case_raw_dir("A") / C.raw_filename("A", "HGT_d05.pt")
            path.write_bytes(src.read_bytes())
        else:
            torch.save(torch.from_numpy(arr), str(path))
        files[name] = {"sha256": C.sha256_of(path), "bytes": path.stat().st_size,
                       "shape": list(arr.shape), "dtype": "float32"}
        print(f"  [{case}] wrote {name}  ({files[name]['bytes']/1e6:.1f} MB)")
    return files


def write_branding(case: str, out_dir: Path, files: dict) -> None:
    gen_path = Path(__file__).resolve()
    rel = gen_path.relative_to(C.ROOT).as_posix()
    manifest = {
        "placeholder": True,
        "case": case,
        "label": C.CASES[case]["label"],
        "seed": C.MASTER_SEED,
        "seed_spawn_keys": {
            "synoptic": list(C.SEED_SPAWN["placeholder_synoptic"]),
            "case": list(C.SEED_SPAWN[f"placeholder_case_{case}"]),
            **({"terrain_C": list(C.SEED_SPAWN["placeholder_terrain_C"])}
               if case == "C" else {}),
        },
        "generator_script": rel,
        "generator_sha256": C.sha256_of(gen_path),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "detection_rule": ("pipeline hashes raw files against this manifest: "
                           "all-match -> data_status=PLACEHOLDER; any mismatch "
                           "-> data_status=REAL (real data swapped in)"),
    }
    (out_dir / "placeholder_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    notice = f"""# PLACEHOLDER DATASET — NOT SCIENCE DATA

Directory: `{out_dir.name}`  (Case {case} = {C.CASES[case]['label']})

**Every tensor in this directory is a deterministic SYNTHETIC placeholder**
generated by `{rel}` (sha256 {manifest['generator_sha256'][:16]}...,
master seed {C.MASTER_SEED}).  It exists only so that the pre-registered
multicase post-processing suite (data_paper0/extension_multicase_20260826/
PLAN_PREREGISTERED.md) could be built and dry-run before the real
{C.CASES[case]['dates']} WRF archive arrives.

* Structure mirrors the real Case-A raw contract: 7 files, d05 = 162x162,
  d02 stored as 162x162 but a native 6x6 parent grid exactly tiled 27x27,
  4320 serialized states, float32, HGT consistent with the wind grid.
* Case B terrain is a byte-identical copy of Case A's HGT_d05.pt (same
  Yakou nest).  Case C terrain is a synthetic lower-relief
  foothill-transition field (Dayekou-like), NOT a real DEM.
* Winds are seeded synthetic fields (shared December synoptic driver +
  terrain-coupled local structure); T2 is a structural stand-in only.
* Any number computed from this directory is branded
  `data_status = PLACEHOLDER` and must never be quoted as evidence.

## Swap-in
Replace the seven `.pt` files with the real archive, using the per-case
on-disk filenames from the CASE registry (`filename_suffix` in
CASE_CONTRACT.json), then delete `placeholder_manifest.json` and this
notice, and rerun `python source/scripts/paper0_multicase/run_all.py`.
The pipeline detects the hash mismatch and flips every output to
`data_status = REAL` (see the extension README for the full numbered
procedure).
"""
    (out_dir / "PLACEHOLDER_NOTICE.md").write_text(notice, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if placeholder dirs already exist")
    args = ap.parse_args()
    C.set_data_root(args.data_root)

    t0 = time.time()
    syn = synoptic_driver()
    receipts = {}
    for case in ("B", "C"):
        out_dir = C.case_raw_dir(case)
        if out_dir.exists() and any(out_dir.iterdir()):
            manifest = out_dir / "placeholder_manifest.json"
            if not manifest.is_file():
                print(f"[STOP] {out_dir} exists WITHOUT a placeholder manifest — "
                      "refusing to touch a possibly-real dataset.")
                return 2
            if not args.force:
                print(f"[skip] {out_dir} already generated (use --force to redo)")
                continue
        print(f"[case {case}] generating placeholder raw data -> {out_dir}")
        if case == "B":
            hgt32 = C.load_pt(C.case_raw_dir("A") / C.raw_filename("A", "HGT_d05.pt"))
        else:
            hgt32 = synth_dayekou_terrain()
        files = build_case(case, hgt32, syn, out_dir)
        write_branding(case, out_dir, files)
        receipts[case] = {n: f["sha256"] for n, f in files.items()}

    C.run_receipt("gen_placeholders", {
        "cases": list(receipts),
        "hashes": receipts,
        "elapsed_s": round(time.time() - t0, 1),
    })
    print(f"done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
