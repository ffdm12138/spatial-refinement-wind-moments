# Reproducibility

## Two reproduction modes

### 1. `--mode tables` (default; no data download)

```bash
python reproduce_comment.py
```

Does three things:

1. verifies every file in `figure_source/` against `SHA256SUMS.txt`;
2. runs headline numeric checks on the audited tables (Jensen-gap dominance
   1.00–1.03 in all six case–windows, geometric-recovery ranges
   Q_J = 0.095–0.279 and Q_S = 0.247–0.487, the A→B / B→C Spearman
   contrasts);
3. regenerates the three Comment figures into `figures/` directly from the
   audited tables (no manually typed plotted values anywhere in the figure
   code).

### 2. `--mode raw --data <data_root>` (full re-derivation)

Reruns the whole chain from the raw Release tensors:

```
phase00  raw-contract validation (shapes, hashes, case status)
phase01  parent/fine moments  (Delta_dyn, J, Delta_scalar, S)
phase05  interpolation controls: bilinear, bicubic, and both with the
         parent-vector constraint (adds no prediction-time information)
phase07  terrain descriptors and calibration-window relations
phase09  transfers: A-frozen relations -> B (same site, new event);
         B-fitted relations -> C (new site, same event)
phase12  information-source comparison (none / domain mean / geometry /
         static terrain / cell history), per case
phase16  manuscript candidate tables (C1–C4, fail-closed export)
QA       independent 36/36 interpolation recomputation from raw d02 fields
compare  fresh C1–C4 vs shipped audited tables
figures  Fig.1–Fig.3 regenerated from the fresh tables
```

## Tolerances

| Comparison | Tolerance | Notes |
|---|---|---|
| Independent 36/36 QA vs `canonical_recovery/` | 2e-3 | canonical tables are rounded to 6 decimals; recovery ratios are means over 670 states x 36 cells |
| Fresh C1–C4 vs shipped `figure_source/` | 5e-3 | cross-machine float tolerance (numpy/torch versions) |

## The independent interpolation QA

`pipeline/independent_recovery_qa.py` recomputes all 36 recovery ratios
(3 cases x 4 interpolation models x 3 moments) **from the raw d02 fields**
without reading any precomputed summary; the canonical CSV under
`canonical_recovery/` is only the comparison target. It writes
`results/independent_recovery_recompute.json` with a per-row PASS/FAIL and an
overall verdict; `--mode raw` fails hard unless all 36 rows pass.

Note on spatial blocking: parent-block means must be formed as
`x.reshape(6, 27, 6, 27).mean(axis=(1, 3))` — i.e. the physical 6 x 6 blocks
of 27 x 27 fine cells. A naive `x.reshape(36, 27, 27)` slices the row-major
memory into 36 artificial 729-element strips that cross parent boundaries and
invalidates the comparison; the QA script documents this inline.

## Deterministic unit tests

`pipeline/test_metrics.py` contains the deterministic metric unit tests
(decomposition identity, population-spread convention, parent mapping). Run
from the repository root:

```bash
python pipeline/test_metrics.py
```

## Environment

Python >= 3.10 with `numpy`, `pandas`, `scipy`, `torch`, `matplotlib`,
`PyYAML` (see `requirements.txt`). No GPU required; every stage runs on CPU.
