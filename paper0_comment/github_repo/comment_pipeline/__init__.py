"""Comment post-processing pipeline: three-state wind downscaling diagnostic.

Modules
-------
contract      frozen case, window, spatial and statistical definitions
states        S_c, S_v, S_s, Delta_mag, Delta_total, J, E_vec
interpolation bilinear / bicubic controls with and without coarse-vector conservation
evaluation    Level-1 conservation vs Level-2 subgrid-statistic recovery
figures       Fig.1-3 rendering from the released archives

Entry point: ``python reproduce_comment.py --data <release>``.
"""
