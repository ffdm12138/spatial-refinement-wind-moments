# Coarse-grid mean residuals: evaluation and archive scopes

The manuscript and Figure 3 use the 670 output times at indices [3650,4320),
with 36 coarse cells per time. CURRENT_CONSTRAINT_RESIDUALS.csv records the
six mean-preserving method/case maxima for that evaluation period alongside
the separate maxima over all 4,320 archived times at [0,4320).

The largest evaluation-period component residual is 1.0658141036401504e-14
m/s. Thus the stated evaluation bound of <1.1e-14 m/s is supported. The largest
full-archive residual is 1.7763568394002505e-14 m/s (C, mean-preserving bicubic).
Both reductions satisfy the same original 1e-6 m/s constraint tolerance.

The original interpolation_contract.json files are retained byte for byte.
Although their evaluation_slice/n_states fields describe scoring, their
constrained_max_component_residual fields summarize the entire archive.
source_chain.interpolate_case accumulates that maximum over range(0,RAW_N),
and separately reduces HELDOUT for conservation_eval_window.csv.
evaluation.load_conservation_table explicitly reads these two scopes.
The distinction explains the apparent conflicting C values; it is not a new
calibration, a widened tolerance, or an amended numerical result.

The CSV binds each value to its original table/contract hash. Paths are relative
to paper0_comment, and science_code_commit identifies the immutable numerical
baseline. The current repository commit also includes presentation-only figure
label changes; interpolation, statistics, samples, seeds and axes are unchanged.
