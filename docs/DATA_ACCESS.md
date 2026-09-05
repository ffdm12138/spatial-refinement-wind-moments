# Data access

## Raw serialized WRF tensors (GitHub Release)

All raw model fields used by the Comment are distributed as assets of the
GitHub Release **`raw-data-multicase-v1`** of this repository
(https://github.com/ffdm12138/spatial-refinement-wind-moments/releases/tag/raw-data-multicase-v1).

Each case ships seven float32 tensors with 4,320 one-minute archived states on
a 162 x 162 grid:

| Stem | Content |
|---|---|
| `U10_d02_*.pt` | 10 m u-wind, coarse parent domain d02 (~3 km) |
| `V10_d02_*.pt` | 10 m v-wind, coarse parent domain d02 |
| `U10_d05_*.pt` | 10 m u-wind, fine nested domain d05 (~111.11 m) |
| `V10_d05_*.pt` | 10 m v-wind, fine nested domain d05 |
| `T2_d02_*.pt` | 2 m temperature, d02 (manifest/provenance only; unused by the Comment estimators) |
| `T2_d05_*.pt` | 2 m temperature, d05 (manifest/provenance only; unused by the Comment estimators) |
| `HGT_d05_*.pt` | static terrain height, d05 |

Case suffixes: `_yakou20150924` (case A), `_yakou20151201` (case B),
`_ebao20151201` (case C). The Release also contains
`raw_sha256_manifest_20260905.json` with the SHA256 checksum of every tensor.

### Expected local layout for `--mode raw`

Download the assets and arrange one directory per case:

```
data_root/
  1min_yakou20150924/   HGT_d05_yakou20150924.pt  T2_d02_yakou20150924.pt ...
  1min_yakou20151201/   HGT_d05_yakou20151201.pt  ... (same seven stems)
  1min_ebao20151201/    HGT_d05_ebao20151201.pt   ... (same seven stems)
```

Then:

```bash
python reproduce_comment.py --mode raw --data /path/to/data_root
```

(Alternatively set the environment variable `NPJ_MULTICASE_DATA_ROOT`.)

## Upstream archives

- Nested WRF fields (case A site archive): DOI
  [10.11888/atmos.tpdc.303495](https://doi.org/10.11888/atmos.tpdc.303495)
- Yakou station observations (auxiliary context only; **not used** by any
  estimator): DOI [10.3972/hiwater.315.2016.db](https://doi.org/10.3972/hiwater.315.2016.db)

## Derived data shipped here

- `figure_source/` — the audited figure-source tables (C1 core loss,
  C2 geometric recovery, C3 transfers, C4 information sources) with
  `SHA256SUMS.txt`. These are the plotted values of Comment Figures 1–3.
- `canonical_recovery/` — the audited interpolation-recovery tables per case;
  the comparison target of the independent 36/36 QA.
- `contracts/` — the frozen analysis contracts (interpolation contract,
  case-A frozen terrain-equation lock and reference tables) that the pipeline
  reads at run time.
