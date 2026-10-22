# Paper0 Comment — full-period post-processing (current revision)

This directory contains the **current 3,350-state Comment post-processing code** for
*Mean wind and subgrid variability in downscaling over complex terrain*.
The existing [`../github_repo/`](../github_repo/) files correspond to an **earlier 670-time
evaluation** and are retained for historical reproduction. Do not use their plotted numbers
as the current manuscript results.

This code covers three real archived WRF–LES dynamical-downscaling cases:
A (Yakou, September), B (Yakou, December), and C (Ebao, December).
The native d02 wind components are the coarse input; d05 components give the model-based
fine-grid reference. All current statistics use archive indices `[970,4320)`,
**3,350 times × 36 parent cells × 729 fine points**, for each case.
A is part of the fit of the fixed example parameterization. B/C are retrospective
transfer-development cases, *not* previously unseen weather or observational validation.

## Data: already released separately

**No original PT tensors are stored here or duplicated in Git.** The existing
[raw-data-multicase-v1](https://github.com/ffdm12138/spatial-refinement-wind-moments/releases/tag/raw-data-multicase-v1)
Release contains the source tensors and its checksum manifest.

Download the **15 required assets**: d02/d05 U10/V10 for A/B/C, and d05 HGT for A/B/C.
GitHub CLI example (from a terminal with repository access):

```sh
gh release download raw-data-multicase-v1 \
  --repo ffdm12138/spatial-refinement-wind-moments \
  --dir ../../../raw3350 \
  --pattern "U10_*.pt" --pattern "V10_*.pt" --pattern "HGT_d05_*.pt"
```

Put the files in a flat `../../../raw3350/` directory outside the cloned Git repository, named as in the Release, or use the
original `1min_<case-tag>/` or `A/`, `B/`, `C/` subdirectory organization.
**The source raw archives are unchanged by the code.** The archives are large:
allow sufficient storage (approximately 5.5 GB of PT inputs), CPU time and space
for generated results. No PyTorch installation or unsafe pickle execution is required
to read these standard little-endian tensor-storage entries.

## Reproduce the Comment

Use Python 3.10+ with requirements.txt. Run the following commands from this
directory (change paths to suit your machine):

```sh
python -m pip install -r requirements.txt
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 prepare
```

Then run the following steps for each case A, B, and C, substituting the case name:

```sh
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 reference --case A
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 predict --case A
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 score --case A --start 0 --end 500
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 score --case A --start 500 --end 1000
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 score --case A --start 1000 --end 1500
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 score --case A --start 1500 --end 2000
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 score --case A --start 2000 --end 2500
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 score --case A --start 2500 --end 3000
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 score --case A --start 3000 --end 3350
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 summary --case A
```

The `summary` step reproduces **20 fixed vector-pair permutations** directly from the
original PT data if a compatible full-period cache is not supplied.
This can be computationally expensive but does not require fitting.
If you have the previously computed complete per-state permutation archives
(`spatial_targets_cell_time_A.npz`, etc.), `--spatial-cache <directory>`
can be supplied to `summary` as a shortcut. The original 670-time *summary*
is **not** an interchangeable cache.

When all three cases have been completed:

```sh
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 finish
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 verify
python reproduce.py --raw-root ../../../raw3350 --workdir ../../../run3350 figures
```

Generated outputs include `MAIN_NUMBERS.csv`, `RESULTS.json`, the complete figure
source summaries, Fig1–Fig4 and the supplementary temporal FigS1.
All output files stay in the chosen `--workdir`, never in the source checkout.

## Statistical definitions and numerical controls

- Three states: `Sc = |U_d02|`, `Sv = |mean(u_d05)|`,
  `Ss = mean(|u_d05|)`; `Ss-Sc = (Sv-Sc) + (Ss-Sv)`.
- `sigma_s` is the **instantaneous spatial** population SD of the 729 fine
  wind speeds, not temporal turbulent kinetic energy.
- Horizontal component variance, within-block spatial averaging at
  3×3, 9×9 and 27×27, and mean cubed speed / its central third moment
  are calculated from the full d05 reference. The third-moment truncation
  error is not a forecast or turbine-power error.
- Mean-corrected bilinear and bicubic interpolation are re-evaluated using
  the native d02 vector means. The fixed tolerance is 10⁻⁶ m/s.
- Model `MU_JS_SD_QA` is a **fixed previously A-calibrated example**, not a
  newly fitted or claimed-new algorithm of this Comment. Its stored model,
  preprocessing, four static terrain descriptors, 3×3 native-neighbour operator,
  Weibull shape inversion and score functions are included.
- Distributional `D` and `W1` integrate over the complete positive support,
  including the Weibull tail. Spatial permutations retain paired U/V components
  and the complete within-cell empirical distribution but change adjacency.
- The common coarse-speed group is **`4 < Sc < 6 m/s`**, a descriptive
  overlap band for comparing similar coarse wind strengths across sites;
  it is *not* a flow-regime threshold. None of the full-3350 observations
  equals either boundary (see `EXPECTED_RESULTS.json`), so the previous
  `4 <= Sc < 6` implementation and this strict interval give **identical
  sample counts and results** (A 19,969; B 33,213; C 11,454).

## Reproducibility scope

The portable source, small static feature tables and model coefficients are
versioned here. The older diagnostic Release remains unchanged.
Exact data-file identity can be checked against the `raw-data-multicase-v1`
release checksum manifest.

A full local verification of the latest code reconstructed all three cases'
3350-time moments and frozen predictions **bit-for-bit**, and every full-period
500-time distribution-score shard. An additional 16-time × 36-cell × 20-seed
permutation replay exactly matched the archived results for all cases.
Numerical reproduction establishes consistency with these WRF–LES archives,
not observational accuracy, causal terrain attribution or independent weather skill.

Historical extraction, exact time/crop alignment and wind-frame records are not
fully independently authenticated. Parallel component variance is not the
along-flow spatial derivative. A/B share a site, B/C share an event, and B/C
have already been seen during prior model development.

## Availability and licensing

The repository is **published for the 2026-10-23 Comment revision**. The original
`raw-data-multicase-v1` release and its checksums have not been re-uploaded,
deleted or modified. The new [final submission release](https://github.com/ffdm12138/spatial-refinement-wind-moments/releases/tag/comment-full3350-20261023)
contains final figure layouts, supplementary diagnostics and six full-period
reference/prediction arrays, with SHA256 checksums. See
[`submission_diagnostics/README.md`](submission_diagnostics/README.md).
The software is MIT licensed; original WRF data retain their existing terms.
