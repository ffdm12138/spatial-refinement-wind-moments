"""Generate Table 1 for Paper0 manuscript from frozen evidence tables."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ev_dir = ROOT / "paper0_frozen_evidence"
if not ev_dir.exists():
    ev_dir = Path("paper0_frozen_evidence")

df_boot = pd.read_csv(ev_dir / "QA_VS_JS_FINAL_BOOTSTRAP.csv")
b_joint = df_boot[(df_boot["baseline"] == "Prob_JS_joint") & (df_boot["candidate"] == "QA_dyn")].copy()
df_class = pd.read_csv(ev_dir / "QA_VS_JS_FINAL_CLASSIFIED.csv")
ledger = pd.read_csv(ev_dir / "PAPER0_CLAIM_LEDGER.csv").fillna("")
unresolved = {
    (str(row.case_fold), str(row.target))
    for row in ledger.itertuples()
    if row.candidate == "QA_dyn" and row.comparator == "Prob_JS_joint"
    and str(row.numerical_resolution).startswith("UNRESOLVED")
}

folds_order = [
    ("A_T", "fold1", "Temporal (A\\_T)", "fold1"),
    ("A_T", "fold2", "", "fold2"),
    ("A_T", "fold3", "", "fold3"),
    ("A_S", "S1", "Spatial (A\\_S)", "S1"),
    ("A_S", "S2", "", "S2"),
    ("A_S", "S3", "", "S3"),
    ("A_S", "S4", "", "S4"),
    ("A_ST", "S1", "Spatiotemporal (A\\_ST)", "S1"),
    ("A_ST", "S2", "", "S2"),
    ("A_ST", "S3", "", "S3"),
    ("A_ST", "S4", "", "S4"),
]

lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\footnotesize")
lines.append("\\setlength{\\tabcolsep}{4.5pt}")
lines.append("\\caption{Nested development-fold comparison of Prob-QA (\\texttt{QA\\_dyn}) and Canonical Direct JS (\\texttt{Prob\\_JS\\_joint}) across 11 outer folds on the Case A calibration archive ($t \\in [970, 3650)$). The estimand is the CRPS difference for window-aggregated, parent-cell marginal speed and aggregation-gap distributions, $\\Delta = \\operatorname{CRPS}_{\\rm JS} - \\operatorname{CRPS}_{\\rm QA}$ (positive values indicate lower QA CRPS on the specified complete scoring grid). Point estimates use each fold's held-out window; 95\\% intervals are conditional on the fitted and calibrated pipeline and use 1,000 paired 120-minute circular moving-block bootstrap resamples. The 11 folds form a dependent development summary, not 11 independent experiments. These are per-comparison conditional intervals, without simultaneous-confidence or multiplicity adjustment. Their counts are neither independent replications nor a test of overall significance. Point and interval signs are descriptive; numerical-resolution qualifications are reported separately in the text and are not supplied by bootstrap width.}")
lines.append("\\label{tab:nested_results}")
lines.append("\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}lcccc}")
lines.append("\\toprule")
lines.append("Partition & Fold & $\\Delta_{\\rm speed}$ [m~s$^{-1}$] & Speed 95\\% CI & $\\Delta_J$ [m~s$^{-1}$] & $J$ 95\\% CI \\\\")
lines.append("\\midrule")

last_part = ""
for level, fold, part_label, fold_label in folds_order:
    if part_label and last_part:
        lines.append("\\midrule")
    last_part = level
    c_row = df_class[(df_class["level"] == level) & (df_class["fold"] == fold)].iloc[0]
    b_row = b_joint[(b_joint["level"] == level) & (b_joint["fold"] == fold)].iloc[0]
    sp_diff = f"{c_row['QA_dyn_speed_point_diff']:+.4f}"
    sp_ci = f"[{b_row['delta_speed_ci_low']:+.4f}, {b_row['delta_speed_ci_high']:+.4f}]"
    j_diff = f"{c_row['QA_dyn_J_point_diff']:+.4f}"
    j_ci = f"[{b_row['delta_J_ci_low']:+.4f}, {b_row['delta_J_ci_high']:+.4f}]"
    # Mark the recorded unresolved dynamic-QA entries, independently of CI sign.
    if (f"{level}/{fold}", "speed") in unresolved:
        sp_diff += r"$^{\dagger}$"
    if (f"{level}/{fold}", "J") in unresolved:
        j_diff += r"$^{\dagger}$"
    lines.append(f"{part_label:<25} & {fold_label:<5} & {sp_diff} & {sp_ci} & {j_diff} & {j_ci} \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular*}")
lines.append(r"\par\smallskip\raggedright $\dagger$ Numerical direction unresolved at the observed integration/grid diagnostic margin, regardless of the conditional bootstrap interval. Constant-QA versus JS additionally has unresolved $J$ comparisons in A\_S/S1, A\_S/S2 and A\_ST/S1 (seven pair--metric flags in total). A670 and B/C are not flagged. Diagnostics are not formal numerical error bounds.")
lines.append("\\end{table}")

table_tex = "\n".join(lines) + "\n"

out_path = Path(__file__).resolve().parent.parent / "generated" / "table1.tex"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(table_tex, encoding="utf-8")
print(f"Saved generated Table 1 to: {out_path}")
