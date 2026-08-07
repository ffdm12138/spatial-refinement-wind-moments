# Data access — raw WRF archive and Level-2 reproduction

## Archived dataset

The raw one-minute nested WRF fields (U10, V10, T2 tensors at 4320 × 162 × 162 per domain) are archived at the National Tibetan Plateau / Third Pole Environment Data Center:

- **Feng, Yuqi and Li, Guang (2026).** Transformer-based high-resolution downscaling model for near-surface wind and temperature fields over complex terrain [Dataset and software]. DOI **10.11888/atmos.tpdc.303495**

The archive is shared with a related machine-learning study currently under consideration elsewhere. The present article (npj Climate and Atmospheric Science) uses **only the WRF numerical fields and the derived statistics distributed in this repository**; no trained-model outputs from the related study are used.

## Level-2 reproduction (raw fields → derived tables)

The full derivation chain (`evaluate_interpolation_baselines.py`, `bootstrap_interpolation_pairs.py`, `prepare_information_sources.py`, `prepare_npj_figure_data.py`, and the sealed `a4_error_decomposition.py` / `a5_paired_bootstrap.py` provenance scripts) reads the raw one-minute tensors and the archived WRF configuration. To run it:

1. Download the archived dataset (DOI above) and note the directory containing the `1min/` tensors.
2. Set the environment variable `NPJ_RAW_DIR` to that directory (the scripts' default paths are overridden by this variable).
3. Run the scripts in dependency order; their outputs reproduce `source_data/*.csv` tables.

The two sealed scripts (`a4_error_decomposition.py`, `a5_paired_bootstrap.py`) are frozen at their release hashes for provenance; they are not re-runnable inside this package's flat layout by design, and the frozen derived tables in `source_data/` remain the authoritative record.

## Redistribution terms

- Raw WRF fields: not redistributed here; use is subject to the Third Pole Environment Data Center terms of the archived record.
- Derived tables in this repository: CC BY 4.0 (see `DATA_LICENSE.md`).
- Natural Earth auxiliary data (110-m coastlines/borders used by the study-design figure): public domain, distributed with the cartopy package.
