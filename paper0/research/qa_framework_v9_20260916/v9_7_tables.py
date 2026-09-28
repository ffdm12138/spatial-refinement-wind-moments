"""Build repaired derived tables in staging; never silently promote evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v9_7_repair import OUT,HERE,MODELS,write_json

TABLES=OUT/'tables'


def build():
    TABLES.mkdir(exist_ok=True)
    tables={name:[] for name in ['QA_VS_JS_FINAL_A','QA_VS_JS_FINAL_BOOTSTRAP','JS_UNCERTAINTY_ABLATION','DYN_VS_CONST_NESTED','M2_VS_M3_NESTED','BC_TRANSFER_FINAL','BC_TRANSFER_FINAL_BOOTSTRAP','A_HELD_OUT_670_EVALUATION','A_HELD_OUT_670_BOOTSTRAP','BOOTSTRAP_BLOCK_LENGTH_SENSITIVITY','FALSIFICATION_V9_6','PERMUTATION_SANITY_AUDIT','TERRAIN_ATTRIBUTION_FINAL','TERRAIN_WITHIN_TEST_PERM_999','PERMUTATION_MARGINAL_AUDIT']}
    for meta in json.loads((OUT/'manifest.json').read_text()):
        d=OUT/meta['key'];ev=json.loads((d/'evaluation.json').read_text());s=ev['scores'];boots=ev['bootstrap']
        def boot(c,b):return next(r for r in boots if r['candidate']==c and r['baseline']==b)
        if meta['key'].startswith(('A_T_','A_S_','A_ST_')):
            common=dict(level=meta['level'],fold=meta['fold'],outer_train_rows=meta['train_rows'],outer_test_rows=meta['row_count'])
            row=dict(**common,canonical_js_model='Prob_JS_joint')
            for alias,model in [('QA_dyn','QA_dyn'),('QA_const','QA_const'),('canonical_JS','Prob_JS_joint')]:
                row[f'crps_s_{alias}']=s[model]['speed_crps'];row[f'crps_J_{alias}']=s[model]['J_crps']
            for c in ['QA_dyn','QA_const']:
                bb=boot(c,'Prob_JS_joint')
                for kk,k in [('s','speed'),('J','J')]:
                    delta=s['Prob_JS_joint'][f'{k}_crps']-s[c][f'{k}_crps'];row[f'{c}_delta_{kk}']=delta;row[f'{c}_point_winner_{kk}']='QA' if delta>0 else 'JS' if delta<0 else 'TIE'
                tables['QA_VS_JS_FINAL_BOOTSTRAP'].append(dict(**common,**bb))
            tables['QA_VS_JS_FINAL_A'].append(row)
            a=dict(**common);bb=boot('Prob_JS_ind','Prob_JS_joint')
            for metric in ['speed','J']:
                for tag,model in [('ind','Prob_JS_ind'),('joint','Prob_JS_joint')]:
                    a[f'crps_{metric}_JS_{tag}']=s[model][f'{metric}_crps'];a[f'W1_{metric}_JS_{tag}']=s[model][f'{metric}_w1']
                delta=s['Prob_JS_joint'][f'{metric}_crps']-s['Prob_JS_ind'][f'{metric}_crps']
                a[f'delta_{metric}_point_joint_minus_ind']=delta
                a[f'point_winner_{metric}']='JS_joint' if delta<0 else 'JS_ind' if delta>0 else 'TIE'
                for tail in ['low','high']:a[f'delta_{metric}_ci_{tail}']=bb[f'delta_{metric}_ci_{tail}']
                a[f'p_ind_win_{metric}']=bb[f'p_candidate_{metric}']
            a['ci_winner_J']='TIE' if a['delta_J_ci_low']==a['delta_J_ci_high']==0 else 'RECHECK'
            tables['JS_UNCERTAINTY_ABLATION'].append(a)
            a=dict(**common,a_const=json.loads((d/'calibration.json').read_text())['a_const']);bb=boot('QA_dyn','QA_const')
            for metric in ['speed','J']:
                for tag,model in [('dyn','QA_dyn'),('const','QA_const')]:a[f'crps_{metric}_{tag}']=s[model][f'{metric}_crps']
                a[f'delta_{metric}_point_const_minus_dyn']=s['QA_const'][f'{metric}_crps']-s['QA_dyn'][f'{metric}_crps']
                for tail in ['low','high']:a[f'delta_{metric}_ci_{tail}']=bb[f'delta_{metric}_ci_{tail}']
                a[f'p_dyn_win_{metric}']=bb[f'p_candidate_{metric}']
            tables['DYN_VS_CONST_NESTED'].append(a)
            a=dict(**common);bb=boot('QA_dyn','QA_m3')
            for metric in ['speed','J']:
                for tag,model in [('M2','QA_dyn'),('M3','QA_m3')]:a[f'crps_{metric}_{tag}']=s[model][f'{metric}_crps']
                delta=s['QA_m3'][f'{metric}_crps']-s['QA_dyn'][f'{metric}_crps'];a[f'delta_{metric}_M3_minus_M2']=delta
                for tail in ['low','high']:a[f'delta_{metric}_ci_{tail}']=bb[f'delta_{metric}_ci_{tail}']
                a[f'M2_point_winner_{metric}']='M2' if delta>0 else 'M3' if delta<0 else 'TIE'
            tables['M2_VS_M3_NESTED'].append(a)
            tt=json.loads((d/'terrain.json').read_text())
            for key,name in [('falsification','FALSIFICATION_V9_6'),('sanity','PERMUTATION_SANITY_AUDIT'),('hierarchy','TERRAIN_ATTRIBUTION_FINAL')]:tables[name].append({**common,**tt[key]})
            tables['TERRAIN_WITHIN_TEST_PERM_999'].extend(tt['permutations']);tables['PERMUTATION_MARGINAL_AUDIT'].extend(tt['marginal'])
        else:
            a670=meta['key']=='A_670'
            common=dict(case=meta['case'],calibration_mode='each_row_once_frozen' if a670 else 'each_row_once_primary')
            if a670:common['window']='held_out_670'
            name='A_HELD_OUT_670' if a670 else 'BC_TRANSFER_FINAL'
            for model,score in s.items():
                row=dict(**common,model=model,crps_speed=score['speed_crps'],crps_J=score['J_crps'],W1_speed=score['speed_w1'],W1_J=score['J_w1'],row_count=meta['row_count'],time_start=3650 if a670 else 970,time_end=4320)
                tables[name+'_EVALUATION' if a670 else name].append(row)
            for bb in boots:tables[name+'_BOOTSTRAP'].append(dict(**common,**bb))
        if meta['key']!='A_670':
            for length in (60,240):
                bb=json.loads((d/f'bootstrap_sensitivity_{length}.json').read_text())
                tables['BOOTSTRAP_BLOCK_LENGTH_SENSITIVITY'].append(dict(**common,candidate='QA_dyn',baseline='Prob_JS_joint',**bb))
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(TABLES/f'{name}.csv',index=False)
    sensitivity=[]
    for key in ('B','C'):
        d=OUT/key;primary=json.loads((d/'evaluation.json').read_text())['scores']
        for model in ('QA_dyn','QA_const','Prob_JS_joint'):
            other=json.loads((d/f'sensitivity_{model}.json').read_text())['scores'];r=dict(case=key,model=model)
            for metric in ('speed','J'):
                r[f'crps_{metric}_primary']=primary[model][f'{metric}_crps'];r[f'crps_{metric}_sensitivity']=other[f'{metric}_crps'];r[f'delta_{metric}_sensitivity_minus_primary']=other[f'{metric}_crps']-primary[model][f'{metric}_crps']
            sensitivity.append(r)
    pd.DataFrame(sensitivity).to_csv(TABLES/'CALIBRATION_SENSITIVITY.csv',index=False)
    from v9_6_common import load_case,qa_deploy_v96
    labels,_,_=load_case('C');y=labels.iloc[970*36:4320*36].reset_index(drop=True)
    masks={'low_speed_U_lt_6':y.U<6,'moderate_speed_6_le_U_lt_10':(y.U>=6)&(y.U<10),'high_speed_U_ge_10':y.U>=10,'low_J_lt_median':y.J<y.J.median(),'high_J_ge_median':y.J>=y.J.median()}
    z=np.load(OUT/'C/predictions.npz');mu,j,spread=qa_deploy_v96(np.maximum(z['rawQ'],1e-4),np.clip(z['rawa'],1e-4,1-1e-4),z['U'])
    exps={m:dict(np.load(OUT/'C/streams'/f'{m}_expectations.npz')) for m in MODELS}
    exps['Det_QA']=dict(mu=mu,J=j,S=spread)
    train,_,_=load_case('A');train=train.iloc[970*36:3650*36]
    exps['Climatology']={k:np.full(len(y),train[c].mean()) for k,c in [('mu','U'),('J','J'),('S','S')]}
    diag=[]
    for regime,mask in masks.items():
        for model,ex in exps.items():
            r=dict(case='C',regime=regime,model=model,sample_count=int(mask.sum()),sample_fraction=float(mask.mean()),mae_mu_sample_level=float(np.mean(abs(ex['mu'][mask]-y.mu[mask]))))
            for k in ('mu','J','S'):r[f'bias_{k}']=float(np.mean(ex[k][mask]-y[k][mask]))
            diag.append(r)
    pd.DataFrame(diag).to_csv(TABLES/'CASE_C_FINAL_DIAGNOSIS.csv',index=False)
    ev=json.loads((OUT/'A_670/evaluation.json').read_text());sc=ev['scores'];bb=next(b for b in ev['bootstrap'] if b['candidate']=='QA_dyn' and b['baseline']=='Prob_JS_joint')
    receipt=json.loads((HERE/'A_HELD_OUT_670_RECEIPT.json').read_text())
    receipt.update(evaluation_title='Case A 670-state post-development retrospective evaluation',numerical_repair='2026-09-21: analytic/conditional J CDF and converged speed quadrature; original model definitions, train/evaluation rows and calibrations retained',independent_confirmatory=False)
    cc=receipt['canonical_comparison']
    for metric in ('speed','J'):
        cc[f'qa_{metric}_crps']=sc['QA_dyn'][f'{metric}_crps'];cc[f'js_{metric}_crps']=sc['Prob_JS_joint'][f'{metric}_crps'];cc[f'delta_{metric}_point']=bb[f'delta_{metric}_point'];cc[f'{metric}_ci_95']=[bb[f'delta_{metric}_ci_low'],bb[f'delta_{metric}_ci_high']]
    write_json(TABLES/'A_HELD_OUT_670_RECEIPT.json',receipt)
    import v9_6_1_build_all as b
    b.V96=TABLES;b.step2_build_classified_duel();b.step3_build_m2_m3_interpretation()
    js=pd.DataFrame(tables['JS_UNCERTAINTY_ABLATION']);delta=js.delta_speed_point_joint_minus_ind
    payload=dict(canonical_js_model='Prob_JS_joint',selection_rationale='Original canonical model retained; no J marginal degradation because both formulations have the exact same marginal. Numerical repair does not reselect models.',speed_wins=dict(point_joint_wins=int((delta<0).sum()),point_ind_wins=int((delta>0).sum()),ci_joint_wins=int((js.delta_speed_ci_high<0).sum()),ci_ind_wins=int((js.delta_speed_ci_low>0).sum()),ci_ties=int(((js.delta_speed_ci_low<=0)&(js.delta_speed_ci_high>=0)).sum())),J_marginal=dict(point_ties=int((js.delta_J_point_joint_minus_ind==0).sum()),ci_ties=int(((js.delta_J_ci_low==0)&(js.delta_J_ci_high==0)).sum())),mean_point_deltas_joint_minus_ind=dict(speed=float(delta.mean()),J=float(js.delta_J_point_joint_minus_ind.mean())),status='NUMERICAL_REPAIR_20260921')
    write_json(TABLES/'JS_CANONICAL_FORM.json',payload)
    print('staged',len(tables),'tables',flush=True)


if __name__=='__main__':build()
