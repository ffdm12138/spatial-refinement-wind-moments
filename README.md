# spatial-refinement-wind-moments

Reproducibility package for the Comment

> **"Increasing spatial sampling cannot restore subgrid wind moments"**
> Yuqi Feng, Ning Huang, Guang Li (*npj Climate and Atmospheric Science*, Comment).

A parent-cell wind vector does not uniquely determine the subgrid scalar
statistics that coarse graining removes. The Comment quantifies this
representation loss in **three real two-way-coupled WRF simulations** over two
terrain settings of the eastern Qilian Mountains:

| Case | Site | Event | Role in the argument |
|---|---|---|---|
| **A** | Yakou | 24 Sep 2015 (archived window) | mechanism / reference |
| **B** | Yakou | 1 Dec 2015 (archived window) | same site, different event |
| **C** | Ebao | 1 Dec 2015 (archived window) | different site, same event |

Headline findings, identical across all three cases:

- The scalar increment between the fine-grid mean speed and the parent-vector
  magnitude decomposes **exactly** as `Delta_scalar = Delta_dyn + J`, and the
  Jensen gap `J` carries the increment (`J / Delta_scalar` = 1.00–1.03 in all
  six case–window combinations; `|Delta_dyn|` <= 0.007 m/s).
- Bilinear and bicubic interpolation of the parent vectors onto the 111 m grid
  recover at most **0.279 of the Jensen gap** and **0.487 of the within-parent
  spread**; the period-mean scalar increment is not recovered
  (parent-vector-constrained `Q_Delta` <= 0.281).
- Terrain-related spatial organization **persists across events** at the first
  site but its empirical mapping does **not transfer** to the second site,
  whose own local information does recover structure — loss is reproducible;
  recovery is information-limited and context-dependent.

## Repository layout

```
reproduce_comment.py     single entry point (two modes, see below)
pipeline/                the analysis pipeline (frozen contracts, plain Python)
contracts/               frozen analysis contracts and case-A reference tables
figure_source/           audited figure-source tables (C1–C4) + SHA256SUMS
canonical_recovery/      audited interpolation-recovery tables (36/36 QA target)
tests/                   (see pipeline/test_metrics.py) deterministic unit tests
docs/                    data access, reproducibility, method contract
```

## Data

The raw serialized WRF tensors (7 fields x 3 cases, ~2.7 GB total) are
distributed **only** as GitHub Release assets, tag
[`raw-data-multicase-v1`](../../releases/tag/raw-data-multicase-v1), with a
SHA256 manifest. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

You do **not** need the raw data to regenerate the figures: the audited
figure-source tables ship with this repository and are verified by checksum.

## Quick start

```bash
pip install -r requirements.txt

# 1. verify the audited tables and regenerate the three Comment figures
python reproduce_comment.py

# 2. optional: full re-derivation from the raw Release data (~GBs, needs torch)
python reproduce_comment.py --mode raw --data /path/to/data_root
```

`--mode tables` verifies the shipped tables and re-renders
`figures/Fig1_representation_loss_multicase.pdf`,
`figures/Fig2_geometric_recovery_multicase.pdf` and
`figures/Fig3_persistence_portability.pdf`.

`--mode raw` reruns the complete chain — raw-contract validation, parent/fine
moments, the four parameter-free interpolation controls, terrain descriptors,
temporal and cross-site transfers, information-source comparison — then checks
the result against the shipped audited tables and runs the **independent
36/36 interpolation QA** (independent recomputation of all 36 recovery ratios
from the raw d02 fields; see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).

## What d02 / d05 / 27x27 / 6x6 mean

- **d02** is the ~3 km coarse parent domain; its cell values are the *parent
  vectors* (the coarse representation).
- **d05** is the ~111.11 m two-way-nested *fine-grid numerical reference*
  (not observational truth).
- Each analyzed parent cell maps to a contiguous **27 x 27 = 729** block of
  d05 cells; the analyzed region is a contiguous **6 x 6 = 36** parent-cell
  block (162 x 162 fine cells).
- Each run archives 4,320 one-minute states; the first 970 are excluded by the
  inherited analysis convention; the remaining 3,350 split chronologically into
  2,680 calibration and 670 evaluation states.

The full estimand/window/interpolation contract is in
[docs/METHOD_CONTRACT.md](docs/METHOD_CONTRACT.md).

## How to cite

See [CITATION.cff](CITATION.cff). The upstream WRF data archive is
10.11888/atmos.tpdc.303495; the Yakou station archive (auxiliary context only)
is 10.3972/hiwater.315.2016.db.

## License

See [LICENSE](LICENSE).
