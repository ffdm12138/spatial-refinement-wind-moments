# Table 3 — fitted equations (refitted from the released cell means)

## scalar-increment roughness fit
- formula: mu_Delta = b0 + b1 * r_sigma
- refitted coefficients: 0.079296, 5.578546
- manuscript coefficients: 0.079296, 5.578546
- max |difference|: 1.08e-07 (tolerance 1e-04) -> PASS

## spread joint fit (roughness IQR + isotropic exposure)
- formula: mu_S = b0 + b1 * r_IQR + b2 * lambda_iso
- refitted coefficients: 0.383082, -0.875890, 9.375677
- manuscript coefficients: 0.383082, -0.875890, 9.375677
- max |difference|: 1.96e-07 (tolerance 1e-04) -> PASS

# Table 4 — comparison across information sources (evaluation period, final 670 states)

## (a) Period-mean scalar increment

| Model | Information used | RMSE | Centred RMSE | Bias | r | Slope | All-record RMSE |
|---|---|---|---|---|---|---|---|
| zero_correction | none | 0.375 | 0.191 | -0.322 | -- | -- | 0.466 |
| global_training_mean | one calibration-period domain mean | 0.198 | 0.191 | -0.051 | -- | -- | 0.341 |
| bilinear | instantaneous d02 U and V, no fitted parameters | 0.539 | 0.336 | -0.422 | -0.284 | -0.338 | 0.596 |
| bicubic | instantaneous d02 U and V, no fitted parameters | 0.413 | 0.248 | -0.331 | -0.172 | -0.115 | 0.480 |
| terrain_roughness_equation | static subgrid terrain and two fitted coefficients | 0.136 | 0.126 | -0.051 | 0.762 | 0.500 | 0.309 |
| cell_mean_persistence | 36 calibration-period d05 cell moments | 0.109 | 0.096 | -0.051 | 0.868 | 0.690 | 0.298 |

## (b) Period-mean within-parent spread

| Model | Information used | RMSE | Centred RMSE | Bias | r | Slope | All-record RMSE |
|---|---|---|---|---|---|---|---|
| global_training_mean | one calibration-period domain mean | 0.469 | 0.442 | 0.156 | -- | -- | -- |
| bilinear | instantaneous d02 U and V, no fitted parameters | 0.576 | 0.308 | -0.487 | 0.843 | 0.336 | -- |
| bicubic | instantaneous d02 U and V, no fitted parameters | 0.494 | 0.272 | -0.412 | 0.854 | 0.449 | -- |
| roughness_only | static subgrid terrain and two fitted coefficients | 0.315 | 0.273 | 0.156 | 0.786 | 0.620 | -- |
| exposure_only | static subgrid terrain and two fitted coefficients | 0.247 | 0.191 | 0.156 | 0.902 | 0.830 | -- |
| locked_joint_equation | static subgrid terrain and fitted coefficients | 0.247 | 0.191 | 0.156 | 0.902 | 0.830 | -- |
| cell_mean_persistence | 36 calibration-period d05 cell moments | 0.244 | 0.188 | 0.156 | 0.908 | 0.880 | -- |

