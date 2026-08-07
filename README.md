# What spatial refinement recovers from coarse wind fields over complex terrain — reproducibility package

Code and derived source data for the article *What spatial refinement recovers from coarse wind fields over complex terrain* (Yuqi Feng, Guang Li, Ning Huang), submitted to *npj Climate and Atmospheric Science*.

The manuscript asks which wind statistics are removed when a heterogeneous wind field is collapsed to one parent-cell vector, and which information source — instantaneous coarse-grid geometry, static terrain, or fine-grid history — is required to recover each one. The central result: **spatial refinement and statistical recovery are not the same problem**; geometric interpolation of the coarse field restores spatial detail but not the nonlinear parent-cell moments that coarse representation removed.

## Repository

This package is distributed from https://github.com/ffdm12138/spatial-refinement-wind-moments .

## Contents

- `scripts/` — Level-1 reproduction entry points (`reproduce_figures.py`, `reproduce_tables.py`, `verify_results.py`), the six-figure renderer `make_figures_npj.py`, the shared style/data-access module `figstyle.py`, and the Level-2 analysis scripts (interpolation baselines, paired bootstrap, information-source assembly, figure-data preparation).
- `figure_data/` — figure-source tables and the WRF domain configuration used by the study-design figure.
- `source_data/` — derived evidence tables (interpolation evaluation, recovery ratios, bootstrap, error decomposition, terrain fitting, claim ledger, figure contracts) plus sealed derived summaries of the archived WRF fields.
- `figures/` — canonical reference figures (frozen at release; regenerated copies are written to `reproduced/figures/`, never into `figures/`).
- `tables/` — canonical reference tables (regenerated copies go to `reproduced/tables/`).
- `reproduced/` — regeneration outputs (created by the reproduction scripts; not part of the checksum manifest).
- `SHA256SUMS.txt` — SHA-256 of every released file except itself.
- `evidence_manifest.csv` — provenance of the 41 `source_data/` files (internal source paths, checksums, sizes).
- The WPS/WRF namelists (`source_data/wrf_namelist.wps`, `source_data/wrf_namelist.input`, `figure_data/fig12_sources/wrf_domain/namelist.wps`) are archival configuration copies with machine-specific absolute paths sanitized; all scientific parameters are unchanged.

## Quick start (Level 1 — derived data to figures and tables)

Requires Python 3.11. `environment.yml` (conda) and `requirements.txt` (pip) pin the exact versions used to produce the results; the interpolation contract records `torch==2.9.1+cpu` and `numpy==2.4.1`. The PyTorch CPU build is not on the default PyPI index; `requirements.txt` points at the PyTorch CPU index.

```bash
conda env create -f environment.yml
conda activate npj-wind-moments
python scripts/verify_results.py        # integrity + key-number verification
python scripts/reproduce_tables.py      # Table 3/4 reconstruction (-> reproduced/tables/)
python scripts/reproduce_figures.py     # six figures, four formats (-> reproduced/figures/)
```

`verify_results.py` exits 0 only if all checks pass (checksums, 72,360-row interpolation record, RMSE decompositions, locked recovery ratios, file contracts). The reproduction scripts never modify the canonical `figures/` and `tables/` directories; outputs go to `reproduced/`.

## Reproducibility levels

- **Level 1 (runs from this package alone):** derived data → verification, tables, figures. The six figures and the main tables are regenerated from `figure_data/` and `source_data/` only; no raw tensors, observational station data, or network access are required. The 110-m Natural Earth country geometry used by Fig. 1(a) is vendored at `figure_data/fig12_sources/ne_110m_admin_0_countries.geojson`.
- **Level 2 (raw archive → full derivation):** the analysis chain that produces the derived tables from the raw one-minute nested WRF fields requires the archived dataset (see `DATA_ACCESS.md`) and a checkout-style layout; set `NPJ_RAW_DIR` to the downloaded data. The sealed scripts `a4_error_decomposition.py` and `a5_paired_bootstrap.py` are provided as frozen provenance and are not re-runnable inside this package layout by design.

## Data

The raw one-minute nested WRF fields are archived at the National Tibetan Plateau / Third Pole Environment Data Center (Feng and Li, 2026; DOI 10.11888/atmos.tpdc.303495) and are not redistributed here because of size and third-party-use conditions; Level-1 regeneration does not require them. The archived record is shared with a related machine-learning study; **the present analysis uses only the WRF numerical fields and the derived statistics distributed here — no trained-model outputs from the related study are used.** No observational station data are included in, or used by, this study.

The interpolation cell–time record `source_data/interpolation_cell_time.csv` holds **72,360 rows = 24,120 × 3** (reference, bilinear, bicubic; 670 evaluation states × 36 parent cells per method).

## Checksums

`SHA256SUMS.txt` records the SHA-256 of every released file except itself (it excludes nothing else; `.git/` and regenerable `reproduced/` outputs are not part of the distributed tree). Verify with `sha256sum -c SHA256SUMS.txt`.

## Licensing

- `LICENSE` — MIT license, covering the code in this repository.
- `DATA_LICENSE.md` — licensing of the derived data tables (CC BY 4.0), the Natural Earth auxiliary data (public domain), and the archived WRF dataset (Third Pole Environment Data Center terms; not redistributed here).
