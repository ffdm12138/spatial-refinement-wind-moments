"""Regenerate the six manuscript figures from the released derived data.

Level-1 reproduction: consumes only `figure_data/` and `source_data/` inside
this package (or the internal workspace layout); no raw tensors, no
observational station data, no network access beyond the data bundled with
cartopy and matplotlib.

Outputs go to `<package>/reproduced/figures/` by default (or `--output-dir`);
the canonical `figures/` directory is never modified.

Usage:
    python scripts/reproduce_figures.py [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "fig1_2_repro"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reproduced" / "figures",
                    help="output directory (default: reproduced/figures)")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import make_figures_npj

    def redirected_save(fig, stem, subdir="main", *, bbox_inches="tight"):
        """Write the four formats to the requested output directory instead of
        the canonical figures/ tree (make_figures_npj.save is rebound)."""
        del subdir  # flat output layout
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches=bbox_inches, pad_inches=0.025)
        fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches=bbox_inches,
                    pad_inches=0.025)
        fig.savefig(out_dir / f"{stem}.tiff", dpi=600, bbox_inches=bbox_inches,
                    pad_inches=0.025, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(out_dir / f"{stem}.eps", bbox_inches=bbox_inches, pad_inches=0.025)
        plt.close(fig)

    make_figures_npj.save = redirected_save

    for fig_fn in (make_figures_npj.fig1, make_figures_npj.fig2,
                   make_figures_npj.fig3, make_figures_npj.fig4,
                   make_figures_npj.fig5, make_figures_npj.fig6):
        fig_fn()
    n = len(list(out_dir.glob("Fig*.pdf")))
    print(f"reproduce_figures: {n} figures written to {out_dir}/ (pdf/png/tiff/eps)")


if __name__ == "__main__":
    main()
