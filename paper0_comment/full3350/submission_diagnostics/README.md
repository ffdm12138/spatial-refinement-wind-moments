# Exact post-processing outputs for the 2026-10-23 Comment revision

This directory belongs to the WRF-LES-based **Paper0 Comment**, *Mean wind and subgrid variability in downscaling over complex terrain*.
The scientific source is `../` (original 3350-state reconstruction, predictor and source tables). This directory adds the **final submission figure layouts and SI6 descriptive diagnostics**, without modifying raw data, fitting a model or rerunning the WRF simulation.

- Raw U10/V10/HGT source tensors and checksums: [raw-data-multicase-v1](https://github.com/ffdm12138/spatial-refinement-wind-moments/releases/tag/raw-data-multicase-v1).
- Final submitted figures, verified numerical summaries and SI6 diagnostics: [comment-full3350-20261023](https://github.com/ffdm12138/spatial-refinement-wind-moments/releases/tag/comment-full3350-20261023). This Release supplements, rather than replaces, the unchanged raw-data and historical comment-revision-20261001 Releases.
- Analysis: 3350 states at indices [970,4320) for each of A/B/C; 36 coarse cells each; 27-by-27 d05 blocks.
- `FULL3350_RESULTS.json`, `FULL3350_MAIN_NUMBERS.csv`: frozen full-period summary and primary table.
- `plot_submission_figures.py`: renders final Figures 1-4 using `FULL3350_RESULTS.json` and Matplotlib/NumPy.
- `FIGURE_SOURCE_MAP.json`: submitted figures' audited SHA256 hashes; FigS1 is included as a release asset.
- `summarize_revision_diagnostics.py`: computes SI6 anchor and cellwise/conditional-band statistics from the A/B/C reference and prediction NPZ output.
- `REVISION_DIAGNOSTICS.json`, `CELLWISE_SPREAD_DIAGNOSTICS.csv`, `SPEED_BAND_DIAGNOSTICS.csv`: exact SI6 values and source digests.
- `FINAL_SUBMISSION_ASSET_MANIFEST.json` in the Release lists all uploaded small artifacts, and separately identifies the six validated full-period reference/prediction NPZ replay inputs. Those six arrays can be recreated with the full3350 pipeline from the original WRF PT Release; they are not separately uploaded here.

Commands from this directory (with NumPy, Matplotlib):
```sh
python plot_submission_figures.py --results FULL3350_RESULTS.json --out ./new_figures
python summarize_revision_diagnostics.py --input /path/to/six_downloaded_npz --output ./new_checks
```

The second command requires an output path that does not already exist, with the six regenerated A/B/C reference/prediction NPZ files present in the named input directory. It should reproduce the provided SI6 JSON and CSV outputs. The paper's raw-to-results replay instructions are in `../README.md`. Source archive and data file hashes, not the time of upload, establish identity.

**Evidence boundaries:** This analysis uses WRF-LES references from two events and two sites. B/C were considered during earlier method development; they are not blind tests. Within-cell speed distributions, moment truncations and spatial permutations do not establish observational truth, independent weather generalization or causal terrain/feedback effects.
