"""Source-derived numerical repair reporting; run only after validated promotion."""
from pathlib import Path
import hashlib
import json,re
import pandas as pd
from v9_7_repair import OUT,HERE,write_json
from v9_7_tables import TABLES

ROOT=HERE.parent.parent
FROZEN=ROOT/'paper0_frozen_evidence'
PROMOTION_RECEIPT=ROOT/'qa'/'numerical_repair_20260921'/'promotion_receipt.json'

PROMOTED_TABLE_NAMES=(
    'QA_VS_JS_FINAL_A', 'QA_VS_JS_FINAL_BOOTSTRAP', 'QA_VS_JS_FINAL_CLASSIFIED',
    'JS_UNCERTAINTY_ABLATION', 'DYN_VS_CONST_NESTED', 'M2_VS_M3_NESTED',
    'BC_TRANSFER_FINAL', 'BC_TRANSFER_FINAL_BOOTSTRAP',
    'A_HELD_OUT_670_EVALUATION', 'A_HELD_OUT_670_BOOTSTRAP',
    'BOOTSTRAP_BLOCK_LENGTH_SENSITIVITY', 'FALSIFICATION_V9_6',
    'PERMUTATION_SANITY_AUDIT', 'TERRAIN_ATTRIBUTION_FINAL',
    'TERRAIN_WITHIN_TEST_PERM_999', 'PERMUTATION_MARGINAL_AUDIT',
    'CALIBRATION_SENSITIVITY', 'CASE_C_FINAL_DIAGNOSIS',
)
PROMOTED_JSON_NAMES=(
    'A_HELD_OUT_670_RECEIPT.json', 'JS_CANONICAL_FORM.json',
    'M2_M3_FINAL_INTERPRETATION.json',
)
PROMOTED_NAMES=frozenset(
    f'{name}.csv' for name in PROMOTED_TABLE_NAMES
) | frozenset(PROMOTED_JSON_NAMES)

# The renderer owns reporting projections only.  Mathematical, contribution,
# routing, and novelty contracts are coordinator-owned and must survive a
# post-numerical render unchanged.
PROTECTED_CONTRACTS = frozenset({
    (ROOT/'PAPER0_MATH_CONTRACT.md').resolve(),
    (FROZEN/'PAPER0_MATH_CONTRACT.md').resolve(),
    (ROOT/'PAPER0_TRACK_DEMARCATION.md').resolve(),
    (FROZEN/'PAPER0_TRACK_DEMARCATION.md').resolve(),
    (ROOT/'PAPER0_CONTRIBUTION_CONTRACT.md').resolve(),
    (FROZEN/'PAPER0_CONTRIBUTION_CONTRACT.md').resolve(),
    (ROOT/'docs'/'NOVELTY_FIREWALL.md').resolve(),
})


def _render_legacy():
    """Compatibility alias for the active renderer."""
    return render()


def verify_promotion_receipt():
    """Require the exact promoted 18-table plus 3-JSON evidence set.

    Manuscript reads are intentionally anchored to ``paper0_frozen_evidence``.
    The staging directory may contain intermediate tables, but it cannot serve
    as a second numerical source for reporting.
    """
    if not PROMOTION_RECEIPT.exists():
        raise FileNotFoundError(f'missing promotion receipt: {PROMOTION_RECEIPT}')
    try:
        receipt=json.loads(PROMOTION_RECEIPT.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f'malformed promotion receipt: {PROMOTION_RECEIPT}') from exc
    if receipt.get('status')!='PROMOTED':
        raise RuntimeError(f"promotion receipt status is not PROMOTED: {receipt.get('status')}")
    files=receipt.get('files')
    if not isinstance(files,dict) or set(files)!=set(PROMOTED_NAMES):
        actual=sorted(files) if isinstance(files,dict) else type(files).__name__
        raise RuntimeError(f'promotion receipt must cover exactly 21 evidence files; got {actual}')
    for name in sorted(PROMOTED_NAMES):
        path=FROZEN/name
        if not path.exists():
            raise FileNotFoundError(f'promoted evidence file missing: {path}')
        recorded=files[name].get('sha256') if isinstance(files[name],dict) else None
        if not isinstance(recorded,str) or len(recorded)!=64:
            raise RuntimeError(f'malformed promoted SHA-256 for {name}')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=recorded:
            raise RuntimeError(f'promoted evidence hash mismatch for {name}')
    return receipt


def update_contracts(summary,c):
    """Update numerical reporting projections without touching coordinator contracts.

    The function is deliberately limited to the claim ledger, factual/statistical
    reporting projections, terrain boundary, storyline, and numerical verdict
    consumers.  In particular, ``PAPER0_MATH_CONTRACT.md`` and the contribution
    and track-demarcation contracts are outside this writer's allowlist.
    """
    def save(path,text):
        target=path.resolve()
        if target in PROTECTED_CONTRACTS:
            raise RuntimeError(f'report renderer attempted to overwrite protected contract: {path}')
        path.write_text(text.rstrip()+'\n',encoding='utf-8')
    read=lambda name:pd.read_csv(FROZEN/f'{name}.csv')
    claims=pd.read_csv(FROZEN/'PAPER0_CLAIM_LEDGER.csv').fillna('')
    cl=read('QA_VS_JS_FINAL_CLASSIFIED');bb=read('BC_TRANSFER_FINAL_BOOTSTRAP')
    statements={
      'CLM-01':f"On the specified complete production scoring grid, QA has a positive point-sign classification for both metrics in {c['point'].get('QA_STRICT_POINT_DOMINANCE',0)}/11 folds. Point classifications: {c['point']}. CI classifications: {c['ci']}. These are descriptive full-grid classifications over related holdouts, not independent replications or certified wins; receipt-qualified near-zero and flipped folds retain their point values and intervals without a directional claim.",
      'CLM-04':f"On the specified complete production scoring grid, independent M2 has a lower-score sign in {c['M2_speed']}/11 folds for speed and {c['M2_J']}/11 folds for J than covariance-coupled M3. These are descriptive full-grid classifications, not certified wins, and do not establish physical independence.",
      'CLM-05':f"On the specified complete production scoring grid, QA_dyn has a lower-score sign in {c['dyn_speed']}/11 folds for speed (QA_const in {11-c['dyn_speed']}/11) and {c['dyn_J']}/11 folds for J (QA_const in {11-c['dyn_J']}/11). These are descriptive full-grid classifications; receipt-qualified near-zero or flipped folds are not directional claims, and dynamic partitioning has no uniform advantage.",
      'CLM-06':f"The specified QA_const terrain-derangement sensitivity yields rank fraction <= 0.05 in {c['terrain_speed']}/11 folds for speed and {c['terrain_J']}/11 for J. These are descriptive perturbation classifications, not formal exchangeable-null p-values, certified wins, or causal evidence."
    }
    for prefix,case in [('CLM-02','B'),('CLM-03','C')]:
        r=bb[(bb.case==case)&(bb.candidate=='QA_dyn')&(bb.baseline=='Prob_JS_joint')].iloc[0]
        statements[prefix]=f"Case {case}: 3,350 states and 120,600 parent-cell rows; frozen retrospective transfer without local refitting. On the specified complete production scoring grid, the speed difference is {r.delta_speed_point:+.5f}, 95% CI [{r.delta_speed_ci_low:+.6f}, {r.delta_speed_ci_high:+.6f}], and the J difference is {r.delta_J_point:+.5f}, 95% CI [{r.delta_J_ci_low:+.6f}, {r.delta_J_ci_high:+.6f}]. These signed values are descriptive classifications; receipt-qualified near-zero or flipped comparisons retain the full point estimate and interval without a directional claim."
    for i,row in claims.iterrows():
        key=str(row.get('claim_id',''))[:6]
        if key not in statements:
            # NUM-* and other ledger metadata rows are preserved verbatim.
            continue
        claims.loc[i,'claim_text_template']=statements[key]
        claims.loc[i,'point_estimate']=statements[key]
        claims.loc[i,'ci_95']='Source-derived 120-minute paired circular bootstrap, 1000 resamples; terrain ranks use the specified perturbation distribution.' if key!='CLM-06' else '999 original within-test derangements per fold; rank fraction (b+1)/1000.'
        claims.loc[i,'caveats_and_limits']='Original model/data protocol retained. Development folds are not independent trials; A670 was consulted during prior selection. Wins/lower-score counts are descriptive signs on the complete production scoring grid. Pair-resolution receipts downgrade near-zero or flipped comparisons while retaining point estimates and intervals; no causal terrain or universal-performance claim.'
        if key=='CLM-06':claims.loc[i,'metric']='Specified perturbation rank fraction'
    for dest in (TABLES,HERE,FROZEN):claims.to_csv(dest/'PAPER0_CLAIM_LEDGER.csv',index=False)
    ledger='# Paper0 claim ledger — corrected numerical baseline\n\nPublication track: Paper0 non-Comment. Current values are generated from the repaired frozen CSVs. The pre-repair ledger is retained in `qa/numerical_repair_20260921/pre_repair_snapshot.zip`.\n\n'
    ledger+='\n\n'.join(f"## {key}\n\n{value}" for key,value in statements.items())
    ledger+='\n\nFull-A constant partition: a_const = 0.48019673331060203. Nested folds use their original outer-train OOF calibrations. Positive bootstrap fractions are descriptive, not p-values. All wins/lower-score counts are sign classifications on the specified complete production scoring grid; they are not certified numerical wins. Pair-resolution receipts downgrade near-zero or flipped folds while retaining their complete point estimates and confidence intervals. The fixed-V bijection does not identify V with deployment proxy U.\n'
    save(ROOT/'PAPER0_CLAIM_LEDGER.md',ledger)
    fact=(ROOT/'PAPER0_FACT_CONTRACT.md').read_text(encoding='utf-8')
    # Re-rendering must replace the prior numerical projection rather than
    # append another canonical-model and Section 5 block.  Older contracts
    # used a different heading, so accept both historical markers.
    marker=re.search(r'\n- \*\*(?:Canonical models retained|Canonical Baselines and Controls)\*\*:',fact)
    if marker:
        fact=fact[:marker.start()]
    fact=fact.rstrip()+'\n'
    fact=fact.replace('**Scientific Result Freeze**: V9.6.1','**Scientific Result Freeze**: original V9.6.1 model/data contract; numerical integration corrected 2026-09-21')
    fact=fact.replace('The final 670-state segment was reserved and was not consumed by the nested representation duel reported here.','These rows are excluded from model fitting, but were consulted in earlier A-validation selection. Their evaluation is post-development retrospective, not independently confirmatory.')
    fact+='''- **Canonical models retained**: QA_dyn uses independent M2; Prob_JS_joint retains the original direct JS joint Gaussian, deployed as mu = U + J; QA_const uses the executed full-A a_const = 0.48019673331060203 (fold-specific constants remain unchanged).
- **Learner and calibration retained**: ridge semiparametric GAM, alpha 0.01, original predictors and splits; 3-fold inner OOF for outer folds, 5-fold each-row-once full-A calibration.
- **Integration repair**: exact JS/constant-QA J marginals, conditional Gaussian integration for dynamic-QA/M3 J, and higher-order speed integration with convergence checks. No new model selection.

## 5. Corrected numerical evidence

'''+summary.split('\n',1)[1]
    for dest in (ROOT,FROZEN):save(dest/'PAPER0_FACT_CONTRACT.md',fact)
    story='''# Paper0 storyline — corrected numerical evidence

The question remains whether the exactly bijective, reference-state-augmented moment coordinates at fixed $V$ yield different empirical predictive performance under a matched learner and an explicit $U$-for-$V$ deployment approximation. The original model families, input features, data and calibration are retained.

The corrected comparisons below support only conditional empirical statements. Gaussian residual assumptions and regularization are not invariant under nonlinear changes of coordinates, so the experiment evaluates the entire statistical/deployment pipeline. Improved conditioning and forward-map damping remain interpretive hypotheses, not identified mechanisms.

Moment/distribution recovery does not establish reconstruction of fine-grid vector fields, joint extremes or spatial dependence. Static-terrain derangements provide specified sensitivity evidence, not causal identification. Comment retains its separate three-state diagnostic scope.

'''+summary
    save(ROOT/'PAPER0_STORYLINE_FINAL.md',story)
    for dest in (FROZEN,HERE):save(dest/'PAPER0_STORYLINE_SKELETON.md',story)
    stat=(ROOT/'PAPER0_STATISTICS_CONTRACT.md').read_text(encoding='utf-8')
    stat=re.sub(r'For example:\n.*?\n---',lambda m:'Current numerical examples:\n\n'+summary.split('## Retrospective evaluation and transfer',1)[1].split('Terrain rank',1)[0]+'\n---',stat,flags=re.S)
    stat=re.sub(r'> Point summary:.*',lambda m:'> Current classifications: '+str(c['ci'])+'. These counts are descriptive.',stat)
    stat=stat.replace('across all 36 parent cells simultaneously','across all held-out parents simultaneously (36 in temporal folds, 9 in quadrant holdouts)').replace('480 to 670 time steps','446 or 448 time steps')
    # Keep the contract header tied to the repaired numerical state on every
    # render.  Replace the whole line so this remains idempotent across the
    # historical V9.6/R2 and already-rendered V9.7 variants.
    stat=re.sub(r'^\*\*Scientific Result Freeze\*\*:.*$',
                '**Scientific Result Freeze**: original V9.6.1 model/data contract; V9.7 numerical repair under current acceptance',
                stat, flags=re.M)
    stat=re.sub(r'^\*\*Prewrite Handoff Phase\*\*:.*$',
                '**Prewrite Handoff Phase**: V9.7 numerical-repair validation; current acceptance is recorded in `qa/numerical_repair_20260921/` receipts',
                stat, flags=re.M)
    # Avoid importing the historical inferential wording even when rendering
    # from an unrendered contract snapshot.
    stat=re.sub(r'The interval [^\n"]*difference is detected\.',
                'The interval is compatible with zero under this conditional bootstrap; no directional claim is made.',
                stat)
    save(ROOT/'PAPER0_STATISTICS_CONTRACT.md',stat)
    terrain=(ROOT/'PAPER0_TERRAIN_CLAIM_BOUNDARY.md').read_text(encoding='utf-8')
    terrain=re.sub(r'## 2\. Cross-Case Transfer:.*',lambda m:'## 2. Retrospective transfer boundary\n\nCase B is Yakou December 2015 (seasonal transfer); Case C is Ebao December 2015 (compound site and season transfer). Keep the original no-refit protocol. Regime MAE and signed bias are diagnostics, not regime-specific distribution CRPS. Numerical outcomes and intervals follow:\n\n'+summary,terrain,flags=re.S)
    terrain=re.sub(r'^\*\*Scientific Result Freeze\*\*:.*$',
                   '**Scientific Result Freeze**: original V9.6.1 model/data contract; V9.7 numerical repair under current acceptance',
                   terrain, flags=re.M)
    terrain=re.sub(r'^\*\*Prewrite Handoff Phase\*\*:.*$',
                   '**Prewrite Handoff Phase**: V9.7 numerical-repair validation; current acceptance is recorded in `qa/numerical_repair_20260921/` receipts',
                   terrain, flags=re.M)
    terrain=re.sub(r'2\. "Breaking the correspondence.*?"',
                   '2. "Breaking the correspondence between terrain descriptors and test locations while preserving the terrain multiset yields the source-derived speed and J rank summaries for the specified QA_const perturbation path; these are predictive sensitivity classifications, not a point-score degradation count or a formal exchangeable-null result."',
                   terrain, count=1)
    save(ROOT/'PAPER0_TERRAIN_CLAIM_BOUNDARY.md',terrain)
    save(ROOT/'latex_paper0/CLAIM_MAP.md','# Paper0 current claim map\n\nSections 1–3 retain the fixed-V mathematical framing and original experiment protocol. Section 4 and Figures 2–4 consume the corrected frozen tables; Supplementary Figures S1–S3 provide the declared sensitivity diagnostics. Figure 1 is the fixed-V representation schematic in Section 2. Section 5 retains interpretation limits.\n\n'+ledger+'\n'+summary)
    verdict='''# Paper0 numerical repair — current verdict

Status: NUMERICAL_RECOMPUTATION_COMPLETE / VALIDATION_AND_SEAL_IN_PROGRESS / AUTHOR_REVIEW_PENDING.

The 2026-09-21 user authorization covers both confirmed defects: low-order latent indicator integration and incorrect stored permutation seeds. The old numerical evidence is superseded by corrected computations. Original raw labels, partitions, predictors, model definitions, hyperparameters and calibrations are unchanged. Missing outer-model caches were replayed and verified against legacy predictions. No new scientific story or model selection was introduced.

Current computational proof is in `qa/numerical_repair_20260921/`; final hash-seal and package receipts determine acceptance. Earlier V9.6.1 and R2.5.1 PASS records describe the pre-repair state and are not certification of these changed numbers.

Comment raw-to-archive reproduction passed 93/93 checks in the preceding audit; its scientific evidence is unchanged in this repair, while any manuscript edits remain editorial only. The local state does not establish public availability. Author review, release publication and upload remain AUTHOR_ONLY.

'''+summary
    for path in [ROOT/'PAPER0_R2_5_1_FINAL_VERDICT.md',ROOT/'PAPER0_PREWRITE_FINAL_VERDICT.md',HERE/'V9_6_1_FINAL_VERDICT.md']:
        save(path,verdict)
    save(HERE/'V9_6_1_STATUS.md',verdict)
    save(ROOT/'R2.5.1_FINAL_AUDIT.md',verdict+'\nThe prior audit report is preserved in the pre-repair snapshot. Current acceptance requires corrected-data checks, figure regeneration, a clean manuscript compile, full manifest verification, and working-tree plus fresh-extracted-package gates.\n')
    for path in [ROOT/'FINAL_CONFIRMATORY_AUDIT_NOTE.md',ROOT/'FINAL_CONFIRMATORY_FREEZE.md']:
        original=path.read_text(encoding='utf-8')
        # Historical chronology is retained, but stale numeric claims are superseded explicitly.
        save(path,'# A670 protocol and numerical repair\n\nThe original protocol/chronology is preserved in `qa/numerical_repair_20260921/pre_repair_snapshot.zip`. The 670-state window was excluded from fitting but consulted during earlier A-validation selection; it is not an independent or preregistered confirmation. The executed full-A a_const is 0.48019673331060203. Original model definitions, train rows [970,3650), evaluation rows [3650,4320), seed 7670 and exact-Nt circular bootstrap (120-minute blocks, 1000 resamples) remain unchanged.\n\n'+summary)
    save(ROOT/'R2.5.1_CHANGELOG.md','# Numerical repair change record — 2026-09-21\n\n- Replaced low-order step-CDF quadrature with exact/conditional Gaussian J integration and converged speed integration.\n- Recomputed all dependent A, A670, B/C, ablation, bootstrap, calibration-sensitivity and terrain tables.\n- Corrected all 10,989 stored permutation seeds while retaining the original mappings.\n- Updated numerical consumers and source-derived verification; eliminated obsolete literal numeric checks and figure counts.\n- Retained raw data, original model definitions/calibrations, Comment science and author-only publication boundaries.\n- Prior state saved in the pre-repair snapshot; final acceptance is recorded separately.\n')


def render():
    """Render the repaired tables into the manuscript in evidence order.

    The numerical tables are the sole source for point estimates and interval
    classifications.  This function is intentionally run only after the
    numerical repair and validation stages have promoted their final tables.
    """
    verify_promotion_receipt()
    read=lambda name:pd.read_csv(FROZEN/f'{name}.csv')
    a=read('QA_VS_JS_FINAL_A');cl=read('QA_VS_JS_FINAL_CLASSIFIED');bb=read('QA_VS_JS_FINAL_BOOTSTRAP')
    dyn=read('DYN_VS_CONST_NESTED');m23=read('M2_VS_M3_NESTED');fals=read('FALSIFICATION_V9_6')
    bc=read('BC_TRANSFER_FINAL');bcboot=read('BC_TRANSFER_FINAL_BOOTSTRAP')
    ev=read('A_HELD_OUT_670_EVALUATION');evb=read('A_HELD_OUT_670_BOOTSTRAP');diag=read('CASE_C_FINAL_DIAGNOSIS')
    counts=dict(
        point=cl.QA_dyn_point_relation.value_counts().to_dict(),
        ci=cl.QA_dyn_ci_relation.value_counts().to_dict(),
        dyn_speed=int((dyn.delta_speed_point_const_minus_dyn>0).sum()),
        dyn_J=int((dyn.delta_J_point_const_minus_dyn>0).sum()),
        M2_speed=int((m23.delta_speed_M3_minus_M2>0).sum()),
        M2_J=int((m23.delta_J_M3_minus_M2>0).sum()),
        terrain_speed=int((fals.f1_speed_empirical_p<=.05).sum()),
        terrain_J=int((fals.f1_J_empirical_p<=.05).sum()),
        parent=int(fals.intact_beats_parent_id_speed.sum()),
        parent_clim=int(fals.intact_beats_parent_id_clim_speed.sum()),
        no_terrain=int(fals.intact_beats_no_terrain_speed.sum()),
    )
    write_json(TABLES/'NUMERICAL_RESULT_COUNTS.json',counts)

    def selected(df,**filters):
        for key,value in filters.items(): df=df[df[key]==value]
        if len(df)!=1: raise ValueError(filters)
        return df.iloc[0]

    def uncertainty(row,metric):
        return f"{row[f'delta_{metric}_point']:+.5f} [{row[f'delta_{metric}_ci_low']:+.6f}, {row[f'delta_{metric}_ci_high']:+.6f}]"

    def interval_relation(low,high):
        if low <= 0 <= high:
            return 'spans zero'
        return 'is entirely positive' if low > 0 else 'is entirely negative'

    def exception_report(level,fold):
        row=selected(bb,level=level,fold=fold,candidate='QA_dyn',baseline='Prob_JS_joint')
        speed_low,speed_high=float(row.delta_speed_ci_low),float(row.delta_speed_ci_high)
        j_low,j_high=float(row.delta_J_ci_low),float(row.delta_J_ci_high)
        speed_point=float(row.delta_speed_point);j_point=float(row.delta_J_point)
        speed_winner='QA' if speed_point>0 else 'JS' if speed_point<0 else 'tie'
        j_winner='QA' if j_point>0 else 'JS' if j_point<0 else 'tie'
        direction='opposite point directions' if speed_winner != j_winner and speed_winner != 'tie' and j_winner != 'tie' else 'the same point direction'
        display_level=level.replace('_',r'\_')
        return (f"{display_level}/{fold} retains $\\Delta_{{\\rm speed}}={speed_point:+.8f}$ "
                f"[{speed_low:+.8f}, {speed_high:+.8f}] and "
                f"$\\Delta_J={j_point:+.8f}$ [{j_low:+.8f}, {j_high:+.8f}] "
                f"(JS$-$QA; descriptive complete-production-grid signs with {direction}; "
                f"the speed interval {interval_relation(speed_low,speed_high)} and the $J$ interval "
                f"{interval_relation(j_low,j_high)}; receipt-qualified near-zero or flipped signs are not "
                f"directional claims).")

    validation_keys=[meta['key'] for meta in json.loads((OUT/'manifest.json').read_text())]
    resolution={}
    unresolved=[]
    for key in validation_keys:
        receipt_path=OUT/key/'numerical_validation.json'
        if not receipt_path.exists():
            raise FileNotFoundError(f'missing V9.7 validation receipt for render: {receipt_path}')
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('validation_version')!='V9.7_VALIDATION_2':
            raise RuntimeError(f'V9.7 validation v2 required before render: {receipt_path}')
        if receipt.get('status')!='PASS':
            raise RuntimeError(f'V9.7 validation receipt is not PASS: {receipt_path}')
        for pair in receipt.get('pair_resolution',[]):
            pair_key=(key,pair.get('candidate'),pair.get('baseline'))
            resolution[pair_key]=pair
            for metric in ('speed','J'):
                state=pair.get(f'{metric}_resolution')
                if state and state.startswith('UNRESOLVED'):
                    unresolved.append(f"{key} {pair.get('candidate')} vs {pair.get('baseline')} {metric}: {state}")
    if unresolved:
        resolution_note=('Pair-resolution diagnostics flag the following signed differences as unresolved under the '
                         'observed integration/grid discrepancy margin: '+', '.join(unresolved)+'. These margins are '
                         'diagnostic comparisons, not formal numerical error bounds, and bootstrap width is not used as '
                         'an integration tolerance; corresponding claims remain numerically unresolved.')
    else:
        resolution_note=('Pair-resolution diagnostics did not flag an unresolved QA-versus-JS sign for the tested '
                         'pairs under the observed integration/grid discrepancy margin. This does not certify a '
                         'scientific winner: all wins/lower-score counts remain descriptive complete-production-grid '
                         'sign classifications. The margins are diagnostic comparisons, not formal numerical error '
                         'bounds, and bootstrap width is not used as an integration tolerance.')

    if unresolved:
        resolution_note_tex=('Numerical-resolution diagnostics leave one or more signed QA-versus-JS differences '
                             'unresolved at the observed integration/grid margin; those comparisons are retained '
                             'without directional claims. The margins are diagnostics, not formal numerical error '
                             'bounds, and bootstrap width is not used as an integration tolerance.')
    else:
        resolution_note_tex=('Receipt-based numerical-resolution diagnostics do not flag an unresolved sign for the '
                             'tested pairs under the observed integration/grid margin. This does not certify a '
                             'scientific winner: wins and lower-score counts remain descriptive classifications on '
                             'the complete production scoring grid. The margins are diagnostics, not formal numerical '
                             'error bounds, and bootstrap width is not an integration tolerance.')

    def exception_with_resolution(level,fold):
        detail=exception_report(level,fold)
        pair=resolution.get((f'{level}_{fold}','QA_dyn','Prob_JS_joint'))
        if pair:
            def status_text(metric):
                status=pair.get(f'{metric}_resolution','')
                if status.startswith('RESOLVED'):
                    return ('the receipt does not flag the signed difference at the observed diagnostic margin; '
                            'the complete-grid sign remains descriptive rather than a certified winner')
                if status.startswith('UNRESOLVED'):
                    return ('the signed difference remains unresolved at the observed diagnostic margin; '
                            'no directional claim is made')
                return 'no signed-difference status was recorded'
            detail+=f" Numerical-resolution check: for speed, {status_text('speed')}; for $J$, {status_text('J')}."
        return detail

    exceptions=' '.join(exception_with_resolution(level,fold) for level,fold in (('A_S','S1'),('A_ST','S1')))

    point_text=', '.join(f"{tag.replace('_',' ').lower()}: {n}" for tag,n in counts['point'].items())
    ci_text=', '.join(f"{tag.replace('_',' ').lower()}: {n}" for tag,n in counts['ci'].items())
    summary=['# Paper0 corrected numerical results — 2026-09-21','',
      'Original labels, rows, predictors, GAM definitions, hyperparameters, model choices and calibrations are retained. Missing outer fit caches were replayed with the original algorithm and matched legacy predictions before changing integration. This repair supersedes the prior numerical baseline; it is not a new model selection.','',
      '## Development folds','']
    for group in ('point','ci'):
        for tag,n in counts[group].items(): summary.append(f'- {tag}: {n}/11 folds; descriptive sign classification on the specified complete production scoring grid, not a certified win.')
    row=selected(bb,level='A_T',fold='fold1',candidate='QA_dyn')
    summary += ['',f"A_T/fold1: speed {uncertainty(row,'speed')}; J {uncertainty(row,'J')}. Positive differences are the QA-favoring sign convention on the complete production scoring grid; intervals are paired circular block-bootstrap 95% intervals. The sign is descriptive and receipt-qualified near-zero or flipped folds retain their point values and intervals without a directional claim.",f'Point classifications: {point_text}. Interval classifications: {ci_text}.',exceptions,resolution_note,'',
      '## Retrospective evaluation and transfer','',
      '| Window | Speed CRPS QA / JS | Speed difference [95% CI] | J CRPS QA / JS | J difference [95% CI] |','|---|---|---|---|---|']
    q=selected(ev,model='QA_dyn');j=selected(ev,model='Prob_JS_joint');r=selected(evb,candidate='QA_dyn',baseline='Prob_JS_joint')
    summary.append(f"| A670 | {q.crps_speed:.6f} / {j.crps_speed:.6f} | {uncertainty(r,'speed')} | {q.crps_J:.6f} / {j.crps_J:.6f} | {uncertainty(r,'J')} |")
    for key,table,boot in [('B',bc[bc.case=='B'],bcboot[bcboot.case=='B']),('C',bc[bc.case=='C'],bcboot[bcboot.case=='C'])]:
        q=selected(table,model='QA_dyn');j=selected(table,model='Prob_JS_joint');r=selected(boot,candidate='QA_dyn',baseline='Prob_JS_joint')
        summary.append(f"| {key} | {q.crps_speed:.6f} / {j.crps_speed:.6f} | {uncertainty(r,'speed')} | {q.crps_J:.6f} / {j.crps_J:.6f} | {uncertainty(r,'J')} |")
    summary += ['', 'A670 is a post-development retrospective held-out evaluation: these rows were excluded from fitting but consulted in earlier A-validation selection. B/C each comprise 3,350 states and 120,600 parent-cell rows; both are December 2015 winter cases sharing the same broader event context rather than independent confirmatory replications.','',
      '## Partition and covariance ablations','',
      f"On the complete production scoring grid, the lower-score sign occurs for dynamic partitioning in {counts['dyn_speed']}/11 folds for speed and {counts['dyn_J']}/11 for J; the independent M2 sign occurs in {counts['M2_speed']}/11 folds for speed and {counts['M2_J']}/11 for J against M3. These are descriptive classifications rather than certified wins and do not identify physical independence.",'',
      '## Terrain sensitivity','',
      f"Specified terrain rank fraction <= 0.05: speed {counts['terrain_speed']}/11 folds; J {counts['terrain_J']}/11 folds. The intact-terrain lower-score sign occurs against parent-ID in {counts['parent']}/11, parent-ID plus fixed-moment control in {counts['parent_clim']}/11 and dynamic-only in {counts['no_terrain']}/11; these are descriptive perturbation classifications.",
      'Terrain perturbations use the original QA_const path, stride-8 grids and all 10,989 original mappings. Rank fractions are specified perturbation-sensitivity summaries, not exchangeable-null p-values or causal evidence.','',
      'Numerical method: exact JS and constant-QA J marginals; 48-point conditional Gaussian integration for dynamic QA/M3 J. Speed uses the original moment-matched Weibull family with 32x32 QA, 32-point constant QA and 96x24 JS integration. Speed higher-order checks cover every original row on a stride-8 grid; J checks use 48 deterministic rows plus coarse-speed extrema, with adaptive five-row QA checks. Supplementary Fig. S1 contains terrain sensitivity, S2 the secondary Case C regime diagnostics, and S3 descriptive $(Q,a)$ geometry.','',
      'The fixed-V coordinate bijection remains exact; deployment substitutes coarse U for unavailable V. Bootstrap positive fractions are descriptive. Fold counts are not independent-trial replications. Comment scientific evidence is unchanged; any Comment manuscript edits are editorial only. Author review, public release and upload remain AUTHOR_ONLY.','']
    text='\n'.join(summary)
    (TABLES/'NUMERICAL_REPAIR_REPORT.md').write_text(text,encoding='utf-8')

    old=(ROOT/'latex_paper0/sections/04_results.tex').read_text(encoding='utf-8')
    figures={}
    for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',old,flags=re.S):
        match=re.search(r'\\label\{([^}]+)\}',block)
        if match: figures[match.group(1)]=block
    getfig=lambda key:figures.get(key,'')
    sec=[r'\section{Results}',r'\label{sec:results}',r'\subsection{Matched-Pipeline Comparison Across Development Folds}',r'\label{subsec:development_comparison}',
      r'The 11 structured development folds compare the factorized representation (\texttt{QA\_dyn}) and direct representation (\texttt{Prob\_JS\_joint}) under the same predictors, training partitions, calibration rules, and window-aggregated parentwise estimand. The repaired predictive-CDF integration is defined in Section~\ref{sec:methods}; the table and classifications retain point, interval, reverse, and zero-compatible outcomes.']
    for level,label in [('A_T','temporal'),('A_S','spatial'),('A_ST','spatiotemporal')]:
        t=a[a.level==level]
        sec.append(f"Across {label} folds, speed CRPS differences range from ${t.QA_dyn_delta_s.min():+.5f}$ to ${t.QA_dyn_delta_s.max():+.5f}$ and gap differences from ${t.QA_dyn_delta_J.min():+.5f}$ to ${t.QA_dyn_delta_J.max():+.5f}$ m~s$^{{-1}}$. Positive values are the QA-favoring sign convention on the complete production scoring grid; they are descriptive signs rather than certified wins.")
    sec.append(f"The point-relation counts are {point_text}; the paired-interval classifications are {ci_text}. These are descriptive complete-production-grid classifications over related development folds, not independent-trial replications or certified wins. Receipt-qualified near-zero or flipped folds retain their full point values and confidence intervals without a directional claim. {exceptions} {resolution_note_tex}")
    sec += [getfig('fig:fig2_development'),r'\input{generated/table1.tex}',r'\subsection{Retrospective Evaluation on the Held-Out Case A Window}',r'\label{subsec:retrospective_evaluation}',r'The final Case A window $[3650,4320)$ contains 670 states and 24,120 parent-cell rows. It was excluded from model fitting but read during earlier A-validation selection, so its scores are post-development retrospective evidence rather than an independent confirmatory test. All values below use the complete production scoring grid; positive differences are descriptive signs, and receipt-qualified near-zero or flipped comparisons retain their point values and intervals without a directional claim.',r'\begin{itemize}']
    for cand in ('QA_dyn','QA_const'):
        q=selected(ev,model=cand);j=selected(ev,model='Prob_JS_joint');r=selected(evb,candidate=cand,baseline='Prob_JS_joint')
        for metric,label in [('speed','speed'),('J','$J$')]:
            lo,hi=r[f'delta_{metric}_ci_low'],r[f'delta_{metric}_ci_high']
            meaning='spans zero' if lo<=0<=hi else 'is entirely positive' if lo>0 else 'is entirely negative'
            cname=cand.replace('_',r'\_')
            sec.append(f"\\item \\texttt{{{cname}}}, {label}: CRPS {q[f'crps_{metric}']:.5f} versus JS {j[f'crps_{metric}']:.5f}; point difference ${r[f'delta_{metric}_point']:+.5f}$ m~s$^{{-1}}$; 95\\% CI $[{lo:+.6f}, {hi:+.6f}]$, which {meaning} on the complete production scoring grid. This is a descriptive sign classification; receipt-qualified near-zero or flipped comparisons receive no directional claim, and the positive-bootstrap fraction is not a $p$-value.")
    control=selected(ev,model='Climatology')
    sec += [r'\end{itemize}',f"The fixed-moment parametric control has speed CRPS {control.crps_speed:.5f} and $J$ CRPS {control.crps_J:.5f}. It uses training means to define a constant Weibull and a point mass for $J$; it is not a complete empirical training marginal.",r'\subsection{Frozen No-Refit Transfer to Cases B and C}',r'\label{subsec:transfer}',r'Case A models trained on $[970,3650)$ transfer without local refitting to Cases B and C, each with 3,350 states and 120,600 parent-cell rows over $[970,4320)$. Case B is seasonal transfer at Yakou; Case C combines a site and season change, so the two cases are related transfer conditions rather than independent weather-event replications. The transfer values below use the complete production scoring grid; they are descriptive sign classifications, and receipt-qualified near-zero or flipped comparisons retain their point values and intervals without a directional claim.']
    for case,title in [('B','Case B: Yakou, December 2015'),('C','Case C: Ebao, December 2015')]:
        q=selected(bc,case=case,model='QA_dyn');j=selected(bc,case=case,model='Prob_JS_joint');r=selected(bcboot,case=case,candidate='QA_dyn',baseline='Prob_JS_joint')
        sec += [r'\subsubsection{'+title+'}',r'\begin{itemize}']
        for metric,label in [('speed','Speed'),('J','$J$')]:
            sec.append(f"\\item {label} CRPS: {q[f'crps_{metric}']:.5f} for QA and {j[f'crps_{metric}']:.5f} for JS; point difference ${r[f'delta_{metric}_point']:+.5f}$ m~s$^{{-1}}$, with 95\\% CI $[{r[f'delta_{metric}_ci_low']:+.6f}, {r[f'delta_{metric}_ci_high']:+.6f}]$.")
        sec.append(r'\end{itemize}')
    sec += [r'Case C speed-regime MAE and signed bias are retained as secondary scalar diagnostics in Supplementary Fig.~S2; they do not define the primary distributional estimand.',getfig('fig:fig3_transfer'),r'\subsection{Partition and Covariance Ablations}',r'\label{subsec:ablations}',f"On the complete production scoring grid, the lower-score sign for dynamic partitioning occurs in {counts['dyn_speed']} of 11 folds for speed and {counts['dyn_J']} of 11 folds for $J$; the independent M2 sign occurs in {counts['M2_speed']} of 11 folds for speed and {counts['M2_J']} of 11 folds for $J$ against M3. These are descriptive classifications rather than certified wins, and receipt-qualified near-zero or flipped comparisons receive no directional claim. These ablations describe statistical choices and do not identify physical independence or a unique mechanism. The descriptive $(Q,a)$ geometry is retained in Supplementary Fig.~S3 and does not establish independence.",getfig('fig:fig4_ablations'),r'\subsection{Terrain Perturbation Sensitivity}',r'\label{subsec:terrain_sensitivity}',r'The original \texttt{QA\_const} $Q$-path with fixed $a^*$ is evaluated on a stride-8 scoring subgrid. Each fold uses 999 within-test-parent terrain derangements, preserving the test terrain multiset and coarse wind forcing. These perturbations break terrain--forcing dependencies; the resulting ranks describe sensitivity under this specified perturbation scheme and are not a formally exchangeable conditional-permutation test.',f"Intact-terrain rank fraction $\\le0.05$: speed {counts['terrain_speed']}/11 folds and $J$ {counts['terrain_J']}/11 folds. In separate point comparisons, the intact-terrain lower-score sign occurs relative to parent-ID in {counts['parent']}/11 folds, parent-ID plus fixed-moment control in {counts['parent_clim']}/11 folds, and dynamic-only predictors in {counts['no_terrain']}/11 folds. These are descriptive perturbation classifications and sensitivity evidence, not certified wins or causal evidence. Composite terrain panels are retained in Supplementary Fig.~S1."]
    (ROOT/'latex_paper0/sections/04_results.tex').write_text('\n\n'.join(x for x in sec if x)+'\n',encoding='utf-8')

    main=ROOT/'latex_paper0/main.tex'
    s=main.read_text(encoding='utf-8')
    r=selected(evb,candidate='QA_dyn',baseline='Prob_JS_joint')
    abstract=(r'''We study probabilistic recovery of window-aggregated parentwise wind distributions after spatial coarse-graining over complex terrain. Under matched predictors, generalized additive learners, splits, and calibration, we compare direct $(J,S)$ targets with variance-partition $(Q,a)$ targets. The map $(V,J,S)\leftrightarrow(V,Q,a)$ is an exact reference-state-augmented bijection at fixed $V$; deployment uses native coarse magnitude $U$ as a proxy for unavailable $V$. Across Case A development folds, a 670-state window, and frozen no-refit transfer to Cases B and C, we report CRPS differences and paired intervals. In the retrospective window, \texttt{QA\_dyn} versus direct JS gives '''
      +f"descriptive complete-grid speed CRPS difference of ${r.delta_speed_point:+.5f}$ m~s$^{{-1}}$ (95\\% CI $[{r.delta_speed_ci_low:+.5f},{r.delta_speed_ci_high:+.5f}]$) and a $J$ difference of ${r.delta_J_point:+.5f}$ m~s$^{{-1}}$ ($[{r.delta_J_ci_low:+.5f},{r.delta_J_ci_high:+.5f}]$); receipt checks qualify near-zero or flipped signs. "
      +r'''The window was read during earlier A selection and is not independent confirmation. The estimand is window-marginal: scores do not test minute-by-minute ordering, time-resolved dynamics, or full-field joint recovery. Ablations and terrain derangements provide conditional sensitivity evidence, not causal identification.''')
    abstract_plain=re.sub(r'\\[A-Za-z]+(?:\{[^{}]*\})?', ' ', abstract)
    abstract_plain=re.sub(r'\$[^$]*\$', ' ', abstract_plain)
    abstract_words=re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",abstract_plain)
    if len(abstract_words)>150:
        raise ValueError(f'generated abstract exceeds 150 words: {len(abstract_words)}')
    if any(token in abstract.lower() for token in ('will be inserted','placeholder','read from the final')):
        raise ValueError('generated abstract contains a process placeholder')
    s=re.sub(r'(?<=\\begin\{abstract\}).*?(?=\\end\{abstract\})',lambda m:'\n'+abstract+'\n',s,flags=re.S)
    main.write_text(s,encoding='utf-8')
    update_contracts(text,counts)
    return text,counts


if __name__=='__main__':render()
