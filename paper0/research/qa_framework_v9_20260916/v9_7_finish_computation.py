"""Finish dependent numerical phases after the staged full-grid streams exist."""
from pathlib import Path
import json,subprocess,sys,time

HERE=Path(__file__).resolve().parent
OUT=HERE/'numerical_repair_20260921'
manifest=json.loads((OUT/'manifest.json').read_text())
models=('QA_dyn','QA_m3','QA_const','Prob_JS_ind','Prob_JS_joint')
while True:
    missing=[f"{m['key']}/{model}" for m in manifest for model in models if not (OUT/m['key']/'streams'/f'{model}_receipt.json').exists()]
    if not missing:break
    print('Awaiting full-grid generation:',len(missing),'streams; first:',missing[0],flush=True)
    time.sleep(30)
for name,args in [('v9_7_scores.py',[]),('v9_7_validate.py',[]),('v9_7_repair.py',['sensitivity']),('v9_7_tables.py',[])]:
    print('Running dependent numerical phase:',name,*args,flush=True)
    subprocess.run([sys.executable,str(HERE/name),*args],check=True,cwd=HERE.parent.parent)
print('All dependent numerical computations complete; promotion is separate.',flush=True)
