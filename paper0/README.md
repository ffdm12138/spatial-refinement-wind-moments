# Paper0 Article: post-processing code

This directory mirrors the current local Paper0 Article implementation while
preserving its project-relative paths:

- `source/scripts/paper0_multicase/`: A/B/C tensor validation and moment,
  interpolation, scale, terrain, transfer, bootstrap, table and diagnostic
  post-processing.
- `research/qa_framework_v9_20260916/`: current V9.7 Q--a/JS calibration,
  CDF integration, scoring, ordered-input cache binding and paired block
  bootstrap, plus its V9.6 fit/calibration implementation.
- `research/qa_framework_v8_20260916/` and
  `research/statistical_downscaling_v6_repair_20260915/`: source dependencies
  and small configuration files used by the Article implementation.
- `qa/empirical_revision_20260921/`: train-only E0/E1 empirical baselines and
  their scoring/cache code.
- `scripts/paper0_rescore_frozen.py`: score existing prediction or CDF streams
  without refitting.
- `latex_paper0/scripts/`: manuscript figure and table generation from the
  accepted evidence tables.

The raw A/B/C tensors remain in the existing `raw-data-multicase-v1` Release.
The fitted objects, ordered prediction/reference CDF streams, derived evidence
tables and figures are outside this code-only snapshot; the corresponding
author can provide permitted materials for review. Original bootstrap replay
requires CUDA PyTorch. This directory does not certify a standalone rerun from
raw tensors through all Article results, or a historical WRF simulation rerun.

