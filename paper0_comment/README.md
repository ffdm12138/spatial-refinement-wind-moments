# Paper0 Comment reproduction

**Current 3,350-time Comment revision:** use [full3350/README.md](full3350/README.md) and [full3350/reproduce.py](full3350/reproduce.py). The historical steps below refer to the earlier 670-time evaluation and must not be used to label full-period statistics. The existing data releases are retained without re-upload.

Three states for auditing reference consistency and subgrid recovery in wind downscaling

This repository contains the Paper0 Comment code, fixed case/window definitions,
figure-source tables and validation summaries; the unchanged diagnostic archives accompany the private Comment Release. The repository
and its existing PT data Release are private during manuscript revision. Access
can be arranged through the corresponding author, Guang Li (lg@lzu.edu.cn).

## Reproduce the Comment

```sh
cd github_repo
python -m pip install -r requirements.txt
gh release download comment-revision-20261001 --repo ffdm12138/spatial-refinement-wind-moments --dir downloaded_diagnostics --pattern "*.npz"
python install_diagnostics.py --input downloaded_diagnostics
python reproduce_comment.py --data RELEASE --figures --out outputs --json outputs/verification.json
```

The six diagnostic archives are installed from the private Comment Release,
with sizes and SHA256 checked before copying. The conservation tables are included in Git. It generates the three Comment figures and verifies their statistics.

## Replay from the deposited PT products

The existing data Release is
[raw-data-multicase-v1](https://github.com/ffdm12138/spatial-refinement-wind-moments/releases/tag/raw-data-multicase-v1).
Its 21 PT files and accompanying checksum manifest are retained. The Comment
replay uses only the twelve U10/V10 files on d02/d05 for the three cases.
`paper0_comment/definitions/PT_RELEASE_MANIFEST.json` records filenames,
SHA256 digests, asset URLs and the Release identity.

Place the four wind files for each case under `paper0_comment/github_repo/RELEASE/A`,
`B` or `C`. Original suffixes are accepted: A uses `yakou20150924`, B uses
`yakou20151201`, and C uses `ebao20151201`. Then run:

```sh
python reproduce_comment.py --data RELEASE --from-raw --figures --out outputs --json outputs/pt_verification.json
```

## Fixed definitions and evidence scope

Each case has 4,320 archived one-minute states. The retained 3,350 states consist
of 2,680 analysis states `[970,3650)` and 670 evaluation states `[3650,4320)`.
These counts refer to time states; the state definition is not multiplied by the
36 coarse cells. Each coarse footprint contains a 27 by 27 fine-cell block,
aggregated by arithmetic mean; speed spread uses population variance (ddof=0).

The code distinguishes native coarse speed, averaged fine-vector speed, mean
fine speed, reference-vector residual, the aggregation gap and speed spread.
It compares ordinary and constrained bilinear/bicubic interpolation on the
fixed evaluation window. Conservation uses the maximum component residual with
the frozen tolerance, and is audited on the supplied output unchanged. Recovery
ratios describe window-mean amplitudes for the two named statistics. The controls
are not independent experiments and do not establish a universal loss bound.

The code license is MIT. Data permissions follow the existing deposit and are
not expanded by this revision. Generated outputs go to the selected output
directory. The manuscript is under editorial revision; no acceptance or public
availability is claimed.

## Mean preservation and subgrid wind statistics

The current figure labels describe wind-speed measures, within-cell wind-speed
standard deviation, mean-wind residuals and ratios of mean subgrid statistics.
This terminology update changes no numerical calculation or plotted sample.
See [residual scopes](definitions/CONSTRAINT_RESIDUAL_SCOPES.md) and the
[six-row summary](definitions/CURRENT_CONSTRAINT_RESIDUALS.csv) for the exact
evaluation-period versus full-archive maximum-component residuals.
The manuscript's <1.1e-14 m/s bound applies to the 670 evaluation output times.
