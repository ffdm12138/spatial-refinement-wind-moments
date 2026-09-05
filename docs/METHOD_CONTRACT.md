# Method contract (fixed before the multi-case analysis)

This file records the analysis contract shared by all three cases. It is
frozen: the pipeline reads the same constants (`pipeline/mc_config.py`) and
the frozen interpolation contract (`contracts/interpolation_contract_frozen.json`).

## Simulations

- WRF v4.5, five **two-way nested** domains at 9, 3, 1, 0.333 and 0.111 km.
- **d02** (~3 km) supplies the parent-cell vectors; **d05** (~111.11 m =
  3000 m / 27) is the fine-grid **numerical reference**, not observational
  truth.
- Each run covers 3 days and 6 hours. The first 6 simulation hours are model
  spin-up and are **not** part of the 4,320 archived one-minute states, which
  begin at 06 UTC. All model times are UTC.

## Analysis window (inherited, identical for A, B, C)

- Archived states: 4,320 (one-minute cadence).
- First 970 archived states: excluded by the inherited analysis convention
  (a spin-up / data-availability transition distinct from the 6-h WRF
  spin-up).
- Retained: 3,350 states, split **chronologically and in order** into
  - calibration: the first 2,680 retained states,
  - evaluation: the final 670 retained states.
- Station observations never enter estimation, interpolation, fitting or
  scoring.

## Spatial mapping

- Analyzed region: a contiguous **6 x 6 = 36** block of parent cells,
  row-major.
- Each parent cell contains a contiguous **27 x 27 = 729** block of fine
  cells; the full region is 162 x 162 fine cells.
- The 36 parent cells are contiguous and spatially related; they are **not**
  independent spatial replicates.

## Estimands

- Parent vector `U_p = (U_p, V_p)`: d02 cell values.
- Fine mean scalar speed `s_p`: mean of `sqrt(u^2 + v^2)` over the 729 fine
  cells.
- Scalar increment `Delta_scalar = s_p - |U_p|`, with the **exact**
  decomposition `Delta_scalar = Delta_dyn + J` where
  `Delta_dyn = |mean(u_p)| - |U_p|` and `J = mean(|u_p|) - |mean(u_p)| >= 0`
  is the Jensen gap.
- Within-parent spread `S`: **population** standard deviation of the 729
  fine-cell speeds (`ddof = 0`).

## Geometric-recovery controls (evaluation states only)

- `bilinear`, `bicubic`: map native 6 x 6 parent `U`/`V` components onto the
  162 x 162 fine grid **before** scalar speed is computed
  (`torch.nn.functional.interpolate`, `align_corners=False`; the frozen
  contract is hash-locked at run time). No fitted coefficient, terrain field,
  fine-grid history or station value enters.
- `bilinear_constrained`, `bicubic_constrained`: the same fields after a
  per-parent-block constant offset that restores each parent `U`/`V` component
  to 1e-6 m/s; it adds **no prediction-time information**.
- Recovery ratio: `Q_M = mean(M_pred) / mean(M_ref)` over the 670 evaluation
  states and 36 parent cells, for `M` in {`Delta_scalar`, `J`, `S`}.

## Terrain / history / information sources

- Terrain descriptors are computed from the static d05 terrain field only;
  the exposure metric is a study-specific eight-sector DEM descriptor
  (motivated by, but not identical to, frontal-area drag-partition
  frameworks).
- Terrain relations are descriptive calibration-window fits of frozen forms;
  transfer tests apply previously fitted relations **without refitting**.
- `same-cell historical mean` = the cell's own calibration-period fine-grid
  statistic, stored and applied to the evaluation window (within-site only;
  no cross-site history is used anywhere).
- Bootstrap/uncertainty logic, where used, is parent-cell/block-aware; the
  honest replication unit of the study is the case-event-site structure.

## Cases

| Case | Site | Archived window | Simulations compared |
|---|---|---|---|
| A | Yakou | Sep 2015 event | reference / mechanism |
| B | Yakou | Dec 2015 event | same site, different event |
| C | Ebao | Dec 2015 event | different site, same event (contemporaneous with B) |

B and C are contemporaneous December simulations and are compared as paired
contemporaneous cross-site differences; no terrain-only causal contrast is or
can be implied by this design.
