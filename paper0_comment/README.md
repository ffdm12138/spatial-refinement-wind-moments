# Paper0 Comment: post-processing code

The current Comment source is under `github_repo/`:

- `reproduce_comment.py`: audit derived A/B/C statistics and optionally render
  figures or compare regenerated archives against the supplied archives.
- `comment_pipeline/`: aggregation, interpolation, conservation and
  three-state recovery calculations, archive validation and figure rendering.
- `diagnose_transverse.py`: transverse diagnostic.
- `requirements.txt`: runtime dependencies.

`source/scripts/make_comment_figures.py` is the manuscript figure renderer
used with the local accepted derived tables. It is included as source only.

Run `python paper0_comment/github_repo/reproduce_comment.py --help` from the
repository root for the audit interface. The audit mode expects A/B/C derived
archives and conservation tables supplied separately. Its `--from-raw` option
also reads the previously released `.pt` tensors and compares recalculated
archives with those supplied reference archives. The current code-only branch
does not include the derived archives or figure output. The existing
`raw-data-multicase-v1` Release retains the raw tensors.

