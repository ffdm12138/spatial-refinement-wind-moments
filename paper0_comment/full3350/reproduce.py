"""Reproduce the Comment full-3350 figures and statistics from the existing data release.

No model fitting or scientific input modification. Runs one bounded step per invocation.
Example:
  python reproduce.py --raw-root path/to/raw --workdir path/to/run prepare
  python reproduce.py --raw-root path/to/raw --workdir path/to/run reference --case A
  python reproduce.py --raw-root path/to/raw --workdir path/to/run predict --case A
  python reproduce.py --raw-root path/to/raw --workdir path/to/run score --case A --start 0 --end 500
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,os,sys,runpy
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw-root',type=Path,required=True,help='Directory of released .pt assets, flat or one subdir per case')
parser.add_argument('--workdir',type=Path,required=True,help='Dedicated new output directory; never inside the repository')
parser.add_argument('step',choices=['prepare','reference','predict','score','permutations','summary','finish','figures','verify'])
parser.add_argument('--case',choices=['A','B','C'])
parser.add_argument('--start',type=int)
parser.add_argument('--end',type=int)
parser.add_argument('--spatial-cache',type=Path,help='Optional prior full-3350 per-state permutation NPZ; not required')
a=parser.parse_args()
out=a.workdir.resolve();raw=a.raw_root.resolve()
if out==HERE.parents[1] or out.is_relative_to(HERE.parents[1]):
    parser.error('workdir must be outside the Git repository to avoid committing generated outputs')
if not raw.is_dir():parser.error('Raw data root does not exist: '+str(raw))
os.environ['COMMENT_WORKDIR']=str(out)
os.environ['COMMENT_RAW_ROOT']=str(raw)
if a.spatial_cache:os.environ['COMMENT_SPATIAL_CACHE']=str(a.spatial_cache.resolve())
import analysis as m
if a.step=='prepare':
    m.init()
else:
    if not (out/'qa/QUALIFICATION.json').is_file():
        parser.error('Run prepare first. Refusing to create output in an arbitrary location.')
    if a.step in ('reference','predict','score','permutations','summary') and not a.case:
        parser.error('--case A|B|C is required for this step')
    if a.step=='reference':m.reference(a.case)
    elif a.step=='predict':m.inference(a.case)
    elif a.step=='score':
        if a.start is None or a.end is None:parser.error('Score requires --start and --end')
        m.score(a.case,a.start,a.end)
    elif a.step=='permutations':m.permutations(a.case)
    elif a.step=='summary':m.summarize(a.case)
    elif a.step=='finish':m.finish()
    elif a.step=='figures':runpy.run_path(str(HERE/'make_figures.py'),run_name='__main__')
    elif a.step=='verify':
        from verify import audit
        audit(out)
