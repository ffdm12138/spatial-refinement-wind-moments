"""Portable, fixed-output audit of the published full3350 diagnostic summaries.

This only checks the outputs and fixed reference values; it is not a new independent
weather validation, field experiment or proof of numerical code correctness.
"""
from pathlib import Path
import json,sys,math
HERE=Path(__file__).resolve().parent

def audit(workdir:Path):
    expected=json.loads((HERE/'EXPECTED_RESULTS.json').read_text('utf8'))
    out=workdir.resolve()
    rows=[]
    for case,ref in expected['cases'].items():
        path=out/(case+'_SUMMARY.json')
        if not path.is_file():raise FileNotFoundError('Summary missing '+str(path))
        current=json.loads(path.read_text('utf8'))
        diag=current['groups']['U4to6']
        assert diag['n']==ref['n'],(case,'sample_count',diag['n'],ref['n'])
        import numpy as np
        a=np.load(out/(case+'_reference.npz'),allow_pickle=False)['S_c']
        assert int((a==4).sum())==0 and int((a==6).sum())==0
        assert int(((a>4)&(a<6)).sum())==ref['n']
        metrics={
            'CDF_D':current['transfer']['CDF_D']['mean'],
            'SD_ratio':current['transfer']['SD_amplitude_ratio'],
            'G3':current['moments']['G3'],'L3':current['moments']['L3'],
            'variance_scale27':current['scales']['27']['within_fraction_mean'],
            'permutation_D2x_amplification':current['spatial']['D2_x']['mean_amplification']}
        for key,got in metrics.items():
            target=ref[key]
            assert math.isclose(got,target,rel_tol=2e-7,abs_tol=2e-7),(case,key,got,target)
        rows.append({'case':case,'strict_open_interval_count':diag['n'],'metrics':metrics})
    receipt={'status':'PASS','cases':rows,'window':expected['window'],
             'code_and_statistics_are_not_new_weather_validation':True}
    (out/'PUBLIC_REPRODUCTION_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print('PUBLIC_CHECK_PASS',[(v['case'],v['strict_open_interval_count'])for v in rows])
    return receipt

if __name__=='__main__':audit(Path(sys.argv[1]))
