"""Independent formula checks plus full-time higher-order convergence checks."""
import json
import hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,expit
from v9_7_repair import OUT,MODELS,write_json
from v9_7_numerics import speed_cdf,gap_cdf,gap_cdf_gpu
from v9_7_scores import truth_stream,get_stream
from window_evaluation import speed_grid,j_grid


def validate():
    for meta in json.loads((OUT/'manifest.json').read_text()):
        d=OUT/meta['key'];receipt=d/'numerical_validation.json'
        if receipt.exists():
            try:
                if json.loads(receipt.read_text()).get('validation_version') == 'V9.7_VALIDATION_2':
                    continue
            except (OSError, ValueError, TypeError):
                pass
        if not all((d/'streams'/f'{m}_receipt.json').exists() for m in MODELS):continue
        z=np.load(d/'predictions.npz');p={k:z[k] for k in ['logQ','logita','logJ','logS']};u=z['U'];cal=json.loads((d/'calibration.json').read_text());n=len(u)
        np_=len(meta['parents']);nt=meta['time_count']
        # The sparse validation grid is retained for the expensive higher
        # order all-row check.  Full-grid versus sparse-grid score differences
        # are recorded separately below so integration convergence is not
        # confused with scoring-grid discretization.
        speed_grid_full=speed_grid();j_grid_full=j_grid()
        speed_grid_sparse=speed_grid_full[::8];j_grid_sparse=j_grid_full[::8]
        truth_speed=truth_stream(meta,'speed');truth_J=truth_stream(meta,'J')
        truth_speed_full=truth_speed.mean(axis=0,dtype=float)
        truth_speed_sparse=truth_speed[:,:,::8].mean(axis=0,dtype=float)
        truth_J_full=truth_J.mean(axis=0,dtype=float)
        truth_J_sparse=truth_J[:,:,::8].mean(axis=0,dtype=float)
        grid_tolerance={'speed_crps_report_threshold':2e-6,'J_crps_report_threshold':2e-6}
        records=[]
        model_scores={}
        for model in MODELS:
            higher=(160,32) if model.startswith('Prob_JS') else 48
            avg=np.zeros((np_,len(speed_grid_sparse)))
            for start in range(0,nt,8):
                a=start*np_;b=min(nt,start+8)*np_
                f,_=speed_cdf({k:v[a:b] for k,v in p.items()},u[a:b],cal,model,speed_grid_sparse,order=higher)
                avg+=f.reshape(-1,np_,len(speed_grid_sparse)).sum(axis=0)/nt
            current_speed_stream=get_stream(meta,model,'speed')
            current_J_stream=get_stream(meta,model,'J')
            current=current_speed_stream[:,:,::8].mean(axis=0,dtype=float)
            delta=np.trapezoid((avg-truth_speed_sparse)**2-(current-truth_speed_sparse)**2,speed_grid_sparse,axis=1)
            dw=np.trapezoid(abs(avg-truth_speed_sparse)-abs(current-truth_speed_sparse),speed_grid_sparse,axis=1)
            ids=np.unique(np.r_[np.linspace(0,n-1,48,dtype=int),np.argsort(u)[[0,-1]]]);pp={k:v[ids] for k,v in p.items()};uu=u[ids]
            jhi=gap_cdf(pp,uu,cal,model,j_grid_full,order=96)
            jbase=gap_cdf_gpu(pp,uu,cal,model,j_grid_full,order=48)
            # Production full-grid and sparse-grid score diagnostics.
            current_speed_full=current_speed_stream.mean(axis=0,dtype=float)
            current_J_full=current_J_stream.mean(axis=0,dtype=float)
            current_speed_sparse=current_speed_full[:,::8]
            current_J_sparse=current_J_full[:,::8]
            speed_full_crps=float(np.trapezoid((current_speed_full-truth_speed_full)**2+truth_speed_full*(1-truth_speed_full),speed_grid_full,axis=1).mean())
            speed_sparse_crps=float(np.trapezoid((current_speed_sparse-truth_speed_sparse)**2+truth_speed_sparse*(1-truth_speed_sparse),speed_grid_sparse,axis=1).mean())
            J_full_crps=float(np.trapezoid((current_J_full-truth_J_full)**2+truth_J_full*(1-truth_J_full),j_grid_full,axis=1).mean())
            J_sparse_crps=float(np.trapezoid((current_J_sparse-truth_J_sparse)**2+truth_J_sparse*(1-truth_J_sparse),j_grid_sparse,axis=1).mean())
            speed_full_w1=float(np.trapezoid(abs(current_speed_full-truth_speed_full),speed_grid_full,axis=1).mean())
            speed_sparse_w1=float(np.trapezoid(abs(current_speed_sparse-truth_speed_sparse),speed_grid_sparse,axis=1).mean())
            J_full_w1=float(np.trapezoid(abs(current_J_full-truth_J_full),j_grid_full,axis=1).mean())
            J_sparse_w1=float(np.trapezoid(abs(current_J_sparse-truth_J_sparse),j_grid_sparse,axis=1).mean())
            speed_tail=float(max(np.max(1-current_speed_full[:,-1]),np.max(1-truth_speed_full[:,-1])))
            J_tail=float(max(np.max(1-current_J_full[:,-1]),np.max(1-truth_J_full[:,-1])))
            row=dict(model=model,higher_speed_order=higher,all_original_rows=n,speed_grid_stride=8,
                     speed_mean_crps_change=float(delta.mean()),speed_max_parent_crps_change=float(abs(delta).max()),speed_mean_W1_change=float(dw.mean()),J_CDF_max_abs=float(abs(jhi-jbase).max()),
                     speed_crps_full_grid=speed_full_crps,speed_crps_sparse_grid=speed_sparse_crps,speed_crps_full_minus_sparse=speed_full_crps-speed_sparse_crps,
                     J_crps_full_grid=J_full_crps,J_crps_sparse_grid=J_sparse_crps,J_crps_full_minus_sparse=J_full_crps-J_sparse_crps,
                     speed_W1_full_grid=speed_full_w1,speed_W1_sparse_grid=speed_sparse_w1,speed_W1_full_minus_sparse=speed_full_w1-speed_sparse_w1,
                     J_W1_full_grid=J_full_w1,J_W1_sparse_grid=J_sparse_w1,J_W1_full_minus_sparse=J_full_w1-J_sparse_w1,
                     speed_tail_mass_at_upper=speed_tail,J_tail_mass_at_upper=J_tail,
                     speed_grid_upper=float(speed_grid_full[-1]),J_grid_upper=float(j_grid_full[-1]),
                     grid_comparison='production full grid vs stride-8 scoring grid; separate from latent integration',
                     J_validation_scope='48 representative original rows plus U extrema; not all rows')
            row['passed']=abs(row['speed_mean_crps_change'])<2e-6 and row['speed_max_parent_crps_change']<2e-5 and row['J_CDF_max_abs']<3e-7 and speed_tail<1e-6 and J_tail<1e-6
            records.append(row);print(meta['key'],row,flush=True)
            model_scores[model]=row
        # Pairwise resolution diagnostics.  A sign is marked resolved only
        # when it exceeds the observed integration and full-vs-sparse grid
        # discrepancies of both models.  These margins are diagnostics, not
        # formal error bounds; J retains the representative-row scope below.
        pairs=(('QA_dyn','Prob_JS_joint'),('QA_const','Prob_JS_joint'))
        pair_resolution=[]
        for candidate,baseline in pairs:
            cand=model_scores[candidate];base=model_scores[baseline]
            speed_delta_full=cand['speed_crps_full_grid']*0 + (base['speed_crps_full_grid']-cand['speed_crps_full_grid'])
            speed_grid_bound=abs(base['speed_crps_full_minus_sparse'])+abs(cand['speed_crps_full_minus_sparse'])
            speed_integration_bound=abs(base['speed_mean_crps_change'])+abs(cand['speed_mean_crps_change'])
            speed_margin=speed_grid_bound+speed_integration_bound
            j_delta_full=base['J_crps_full_grid']-cand['J_crps_full_grid']
            j_grid_bound=abs(base['J_crps_full_minus_sparse'])+abs(cand['J_crps_full_minus_sparse'])
            j_eps=max(base['J_CDF_max_abs'],cand['J_CDF_max_abs'])
            j_cdf_margin=(j_grid_full[-1]-j_grid_full[0])*(2*j_eps+j_eps*j_eps)
            j_margin=j_grid_bound+j_cdf_margin
            pair_resolution.append(dict(candidate=candidate,baseline=baseline,
                speed_delta_full_grid=float(speed_delta_full),speed_observed_discrepancy_margin=float(speed_margin),
                speed_resolution='RESOLVED_BY_OBSERVED_MARGIN' if abs(speed_delta_full)>speed_margin else 'UNRESOLVED_NUMERICAL',
                J_delta_full_grid=float(j_delta_full),J_observed_discrepancy_margin=float(j_margin),
                J_resolution='RESOLVED_BY_OBSERVED_MARGIN' if abs(j_delta_full)>j_margin else 'UNRESOLVED_NUMERICAL_SCOPE',
                margin_interpretation='diagnostic comparison of observed integration/grid changes; not a formal numerical error bound',
                J_margin_scope='CDF discrepancy measured on 48 representative original rows plus U extrema; it is not a full-flow bound'))
        # Direct adaptive integration validates the conditional-Gaussian formula,
        # independent of the Hermite implementation (including correlated M3).
        adaptive=[]
        for idx in np.linspace(0,n-1,5,dtype=int):
            for model in ('QA_dyn','QA_m3'):
                q=p['logQ'][idx];eta=p['logita'][idx];sq=cal['std_logQ'];sa=cal['std_logita'];cov=cal['covariance_QA'][0][1] if model=='QA_m3' else 0
                sj=np.sqrt(sq*sq-cov*cov/(sa*sa));jj=float(np.exp(p['logJ'][idx]));uu=u[idx]
                def integrand(z):
                    loga=-np.logaddexp(0,-(eta+sa*z));mq=q+cov/sa*z
                    return ndtr((np.log(jj*(2*uu+jj))-loga-mq)/sj)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
                exact,err=quad(integrand,-10,10,epsabs=1e-11,epsrel=1e-11)
                calculated=gap_cdf({k:v[idx:idx+1] for k,v in p.items()},u[idx:idx+1],cal,model,np.array([0.,jj]),48)[0,1]
                adaptive.append(abs(exact-calculated))
        payload=dict(validation_version='V9.7_VALIDATION_2',status='PASS' if all(r['passed'] for r in records) and max(adaptive)<1e-8 else 'FAIL',speed_validation='all original rows on every eighth original grid coordinate; compare higher Gaussian orders; production uses complete original grids',score_grid_validation='production full grid and stride-8 sparse grid are compared explicitly; tail mass is reported separately; neither is a formal truncation bound',diagnostic_thresholds=grid_tolerance,records=records,pair_resolution=pair_resolution,adaptive_J_max_abs=float(max(adaptive)))
        write_json(receipt,payload)
        if payload['status']=='PASS':
            validation_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()
            for model in MODELS:
                stream_receipt=d/'streams'/f'{model}_receipt.json'
                stream_payload=json.loads(stream_receipt.read_text())
                generation_status=stream_payload.get('status')
                stream_payload.update(
                    generation_status=generation_status,
                    status='GENERATED_AND_VALIDATED_PASS',
                    validation_status='PASS',
                    validation_version='V9.7_VALIDATION_2',
                    validation_receipt_sha256=validation_sha256,
                )
                write_json(stream_receipt,stream_payload)
        if payload['status']!='PASS':raise AssertionError((meta['key'],payload))


if __name__=='__main__':validate()
