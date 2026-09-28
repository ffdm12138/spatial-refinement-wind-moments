# Authorized empirical comparison contract (2026-09-21)

This addition is authorized by the two user attachments; existing fitted models,
data, split, calibration, quadrature and scoring contracts remain unchanged.
The audit already disclosed descriptive J baseline results. This is a fixed
analysis specification before this implementation, not a blind preregistration.

E0 pools training target CDFs with equal time-parent weights. E1 averages within
each training parent. Speed uses the actual 729 fine-speed samples (the existing
float32 empirical reference CDF), not a moment-matched Weibull. J uses empirical
point masses. Both use the complete original speed/J grids and trapezoidal CRPS.
S is a predicted moment, not the speed-CDF scoring variable; it is not relabelled
as speed. Fixed-moment Climatology remains a distinct existing comparator.

Final E0/E1 training is A [970,3650), with 36 parents. Evaluation: A [3650,4320),
B/C [970,4320). E0 applies everywhere. E1 applies to A and B (same Yakou parent
layout); C has no authorized physical parent mapping. Each CV fold uses exactly
the existing outer model-fit rows, reconstructed by level_segments and verified
against prediction row indices. A_T supports E1; A_S/A_ST hold out all samples
of test parents, so E1 is NOT_APPLICABLE, without fallback or target borrowing.

The frozen statistical contract is the P0 V9.7 exact-Nt circular moving-block
bootstrap, 1000 draws, L=120; sensitivities L=60/240. All parents, targets and
models share each time draw. Use the existing QA-vs-JS seed for each configuration,
also shared by E0/E1. Re-average prediction and reference CDFs in every draw;
do not bootstrap fixed row CRPS as a substitute for the nonlinear window score.
Positive delta = empirical CRPS minus QA CRPS. Record all results including losses.
Intervals are descriptive paired intervals, with no multiple-comparison claim.

Persist baseline CDFs, exact training/test rows, per-parent/time contributions
to the full-window score, bootstrap maps/draws, and score provenance. A row score
is CRPS of the full-window predictive CDF against that row's reference; its mean
equals the published window score but it is not a conditional forecast score.
No model refit, target selection, new WRF run or new archive is authorized here.
