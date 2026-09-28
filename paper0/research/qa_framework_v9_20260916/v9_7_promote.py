"""Promote validated staged repair tables into paper0_frozen_evidence.

Fail-closed: every staged artifact must exist and pass consistency checks
against the pre-repair frozen evidence before any byte is replaced.
The pre-repair byte snapshot lives in qa/numerical_repair_20260921/.
"""
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

from v9_7_repair import OUT, HERE, MODELS, write_json
from v9_7_tables import TABLES

ROOT = HERE.parent.parent
FROZEN = ROOT / 'paper0_frozen_evidence'
RECEIPTS = ROOT / 'qa' / 'numerical_repair_20260921'
PROVENANCE_RECEIPT = RECEIPTS / 'v9_7_provenance_receipt.json'

TABLE_NAMES = [
    'QA_VS_JS_FINAL_A', 'QA_VS_JS_FINAL_BOOTSTRAP', 'QA_VS_JS_FINAL_CLASSIFIED',
    'JS_UNCERTAINTY_ABLATION', 'DYN_VS_CONST_NESTED', 'M2_VS_M3_NESTED',
    'BC_TRANSFER_FINAL', 'BC_TRANSFER_FINAL_BOOTSTRAP',
    'A_HELD_OUT_670_EVALUATION', 'A_HELD_OUT_670_BOOTSTRAP',
    'BOOTSTRAP_BLOCK_LENGTH_SENSITIVITY', 'FALSIFICATION_V9_6',
    'PERMUTATION_SANITY_AUDIT', 'TERRAIN_ATTRIBUTION_FINAL',
    'TERRAIN_WITHIN_TEST_PERM_999', 'PERMUTATION_MARGINAL_AUDIT',
    'CALIBRATION_SENSITIVITY', 'CASE_C_FINAL_DIAGNOSIS',
]
JSON_NAMES = ['A_HELD_OUT_670_RECEIPT.json', 'JS_CANONICAL_FORM.json', 'M2_M3_FINAL_INTERPRETATION.json']
A_FOLDS = [('A_T', f'fold{i}') for i in (1, 2, 3)] + [('A_S', f'S{i}') for i in (1, 2, 3, 4)] + [('A_ST', f'S{i}') for i in (1, 2, 3, 4)]
EXPECTED_CONFIG_KEYS = tuple(
    [f'A_T_fold{i}' for i in (1, 2, 3)]
    + [f'A_S_S{i}' for i in (1, 2, 3, 4)]
    + [f'A_ST_S{i}' for i in (1, 2, 3, 4)]
    + ['A_670', 'B', 'C']
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    problems = []
    manifest = json.loads((OUT / 'manifest.json').read_text())
    manifest_keys = [meta.get('key') for meta in manifest]
    if len(manifest) != len(EXPECTED_CONFIG_KEYS):
        problems.append(f'manifest entries {len(manifest)} != {len(EXPECTED_CONFIG_KEYS)}')
    if len(set(manifest_keys)) != len(manifest_keys):
        problems.append('manifest contains duplicate configuration keys')
    if set(manifest_keys) != set(EXPECTED_CONFIG_KEYS):
        problems.append(
            'manifest configuration set mismatch: '
            f'expected={list(EXPECTED_CONFIG_KEYS)}, actual={manifest_keys}'
        )
    if not PROVENANCE_RECEIPT.exists():
        problems.append('missing V9.7 provenance receipt')
    else:
        try:
            provenance = json.loads(PROVENANCE_RECEIPT.read_text())
            if provenance.get('status') != 'STAGED_PROVENANCE_RECEIPT':
                problems.append(f"provenance receipt status={provenance.get('status')}")
            coverage = provenance.get('coverage', {})
            if not coverage.get('exact_set') or not coverage.get('unique_keys'):
                problems.append('provenance coverage is not an exact unique 14-config set')
            if provenance.get('source_missing'):
                problems.append(f"provenance source_missing={provenance['source_missing']}")
            req = provenance.get('requirements', {})
            if not req.get('bootstrap_draw_level_deltas_retained'):
                problems.append('bootstrap draw-level provenance is incomplete')
            if not req.get('terrain_seed_and_mapping_scores_retained'):
                problems.append('terrain seed/mapping/score provenance is incomplete')
            if req.get('near_zero_classification') != 'REVIEWED_BY_FULL_GRID_TARGETED_CHECK':
                problems.append(f"near-zero review status={req.get('near_zero_classification')}")
            unit_tests = provenance.get('unit_test_receipt', {})
            if not unit_tests.get('exists') or not unit_tests.get('all_tests_passed'):
                problems.append('V9.7 unit-test receipt is missing or not all PASS')
            for relative, recorded in provenance.get('source_hashes', {}).items():
                path = ROOT / relative
                if not path.exists():
                    problems.append(f'provenance source missing at preflight: {relative}')
                elif sha256(path) != recorded:
                    problems.append(f'provenance source hash changed: {relative}')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f'provenance receipt malformed: {exc}')
    for meta in manifest:
        key = meta['key']
        d = OUT / key
        if not (d / 'evaluation.json').exists():
            problems.append(f'{key}: missing evaluation.json')
        for model in MODELS:
            stream_receipt = d / 'streams' / f'{model}_receipt.json'
            if not stream_receipt.exists():
                problems.append(f'{key}: missing stream receipt {model}')
            else:
                try:
                    stream_payload = json.loads(stream_receipt.read_text())
                    if stream_payload.get('status') != 'GENERATED_AND_VALIDATED_PASS':
                        problems.append(f'{key}/{model}: stream receipt not validation-bound PASS')
                    if not stream_payload.get('validation_receipt_sha256'):
                        problems.append(f'{key}/{model}: missing validation receipt hash')
                    elif (d / 'numerical_validation.json').exists() and stream_payload['validation_receipt_sha256'] != sha256(d / 'numerical_validation.json'):
                        problems.append(f'{key}/{model}: validation receipt hash mismatch')
                except (OSError, ValueError, TypeError) as exc:
                    problems.append(f'{key}/{model}: malformed stream receipt: {exc}')
        validation = d / 'numerical_validation.json'
        if not validation.exists():
            problems.append(f'{key}: missing numerical_validation.json')
        else:
            try:
                validation_payload = json.loads(validation.read_text())
                if validation_payload.get('validation_version') != 'V9.7_VALIDATION_2':
                    problems.append(f'{key}: validation version is not V9.7_VALIDATION_2')
                if validation_payload.get('status') != 'PASS':
                    problems.append(f'{key}: numerical validation not PASS')
                records = validation_payload.get('records', [])
                if {row.get('model') for row in records} != set(MODELS) or len(records) != len(MODELS):
                    problems.append(f'{key}: validation records do not cover exactly MODELS')
                required = ('speed_mean_crps_change','speed_max_parent_crps_change',
                            'J_CDF_max_abs','speed_crps_full_minus_sparse',
                            'J_crps_full_minus_sparse','speed_tail_mass_at_upper',
                            'J_tail_mass_at_upper')
                for row in records:
                    if not row.get('passed', False):
                        problems.append(f"{key}/{row.get('model')}: validation record not passed")
                    for field in required:
                        if field not in row or not np.isfinite(float(row[field])):
                            problems.append(f"{key}/{row.get('model')}: invalid validation field {field}")
                pair_rows = validation_payload.get('pair_resolution', [])
                expected_pairs = {('QA_dyn', 'Prob_JS_joint'), ('QA_const', 'Prob_JS_joint')}
                actual_pairs = {(row.get('candidate'), row.get('baseline')) for row in pair_rows}
                if actual_pairs != expected_pairs or len(pair_rows) != len(expected_pairs):
                    problems.append(f'{key}: pair resolution does not cover the two required QA/JS pairs')
                for row in pair_rows:
                    if 'not a formal' not in str(row.get('margin_interpretation', '')):
                        problems.append(f"{key}/{row.get('candidate')}: discrepancy margin semantics missing")
                    # The authorized plan permits an explicit unresolved
                    # classification instead of expanding numerical work.
                    # Higher latent order on the same scoring grid cannot
                    # resolve scoring-grid discretization by itself.
                    for metric, unresolved in (('speed', 'UNRESOLVED_NUMERICAL'), ('J', 'UNRESOLVED_NUMERICAL_SCOPE')):
                        state = row.get(f'{metric}_resolution')
                        if state not in ('RESOLVED_BY_OBSERVED_MARGIN', unresolved):
                            problems.append(f"{key}/{row.get('candidate')}: unsupported {metric} resolution {state}")
                        margin = row.get(f'{metric}_observed_discrepancy_margin')
                        if margin is None or not np.isfinite(float(margin)) or float(margin) < 0:
                            problems.append(f"{key}/{row.get('candidate')}: invalid {metric} diagnostic margin")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                problems.append(f'{key}: malformed numerical validation: {exc}')
        if key.startswith(('A_T_', 'A_S_', 'A_ST_')) and not (d / 'terrain.json').exists():
            problems.append(f'{key}: missing terrain.json')
        if key in ('B', 'C'):
            for model in ('QA_dyn', 'QA_const', 'Prob_JS_joint'):
                if not (d / f'sensitivity_{model}.json').exists():
                    problems.append(f'{key}: missing sensitivity_{model}.json')
    for name in TABLE_NAMES:
        if not (TABLES / f'{name}.csv').exists():
            problems.append(f'missing staged table {name}')
    for name in JSON_NAMES:
        if not (TABLES / name).exists():
            problems.append(f'missing staged json {name}')
    if problems:
        return problems

    # Provenance: repaired permutation table must keep every original mapping
    # and shift each recorded seed by exactly the fold offset.
    frozen_perm = pd.read_csv(FROZEN / 'TERRAIN_WITHIN_TEST_PERM_999.csv')
    staged_perm = pd.read_csv(TABLES / 'TERRAIN_WITHIN_TEST_PERM_999.csv')
    # Do not use a chained inequality here: ``a != b != c`` only tests
    # adjacent comparisons and can accept an invalid staged row count when
    # the frozen count is also stale.  Both counts are independently part of
    # the frozen provenance contract.
    if len(staged_perm) != len(frozen_perm) or len(frozen_perm) != 10989:
        problems.append(
            'permutation row count '
            f'staged={len(staged_perm)} frozen={len(frozen_perm)} expected=10989'
        )
    key_cols = ['level', 'fold', 'permutation_index']
    if frozen_perm.duplicated(key_cols).any():
        problems.append('frozen permutation keys are not unique')
    if staged_perm.duplicated(key_cols).any():
        problems.append('staged permutation keys are not unique')
    try:
        merged = frozen_perm.merge(
            staged_perm, on=key_cols, suffixes=('_frozen', '_staged'),
            validate='one_to_one'
        )
    except (pd.errors.MergeError, KeyError) as exc:
        problems.append(f'permutation key merge invalid: {exc}')
        merged = None
    if merged is not None and len(merged) != len(frozen_perm):
        problems.append('permutation merge lost rows')
    elif merged is not None:
        if not (merged['mapping_sha256_frozen'] == merged['mapping_sha256_staged']).all():
            problems.append('permutation mappings changed')
        if not (merged['mapping_json_frozen'] == merged['mapping_json_staged']).all():
            problems.append('permutation mapping payloads changed')
        for level, fold in A_FOLDS:
            part = merged[(merged.level == level) & (merged.fold == fold)]
            offset = np.unique(part['seed_staged'] - part['seed_frozen'])
            expected = sum(ord(c) for c in f'{level}/{fold}')
            if len(offset) != 1 or offset[0] != expected:
                problems.append(f'{level}/{fold}: seed offset {offset} != {expected}')

    # Schema and cardinality: staged tables must cover every frozen column and row.
    for name in TABLE_NAMES:
        baseline_path = FROZEN / f'{name}.csv'
        if not baseline_path.exists() and name == 'PERMUTATION_SANITY_AUDIT':
            # This existing audit was previously kept beside its generator,
            # rather than copied into the evidence bundle. The repair plan
            # promotes that audit alongside the corrected permutation table.
            baseline_path = HERE / f'{name}.csv'
        frozen = pd.read_csv(baseline_path)
        staged = pd.read_csv(TABLES / f'{name}.csv')
        missing = set(frozen.columns) - set(staged.columns)
        if name == 'QA_VS_JS_FINAL_A':
            # Retire the old collapsed CI labels: QA_DOMINATES formerly
            # included one positive interval with the other spanning zero.
            # The separate point/CI fields in CLASSIFIED are the replacement.
            missing -= {'QA_dyn_pareto_status', 'QA_const_pareto_status'}
        if missing:
            problems.append(f'{name}: staged missing columns {sorted(missing)}')
        if len(staged) != len(frozen):
            problems.append(f'{name}: row count staged={len(staged)} frozen={len(frozen)}')
    return problems


def promote():
    problems = preflight()
    if problems:
        raise SystemExit('PROMOTE_REFUSED\n' + '\n'.join(problems))
    receipt = {'status': 'PROMOTED', 'checks': 'preflight PASS', 'files': {},
               'reporting_schema_changes': {'QA_VS_JS_FINAL_A':
                   'Deprecated QA_dyn_pareto_status and QA_const_pareto_status removed; separate point/CI classes in QA_VS_JS_FINAL_CLASSIFIED replace the collapsed legacy labels.'}}
    for name in TABLE_NAMES:
        src = TABLES / f'{name}.csv'
        dest = FROZEN / f'{name}.csv'
        shutil.copyfile(src, dest)
        receipt['files'][dest.name] = {'sha256': sha256(dest), 'bytes': dest.stat().st_size, 'matches_staged': sha256(dest) == sha256(src)}
    for name in JSON_NAMES:
        src = TABLES / name
        dest = FROZEN / name
        shutil.copyfile(src, dest)
        receipt['files'][dest.name] = {'sha256': sha256(dest), 'bytes': dest.stat().st_size, 'matches_staged': sha256(dest) == sha256(src)}
    if not all(f['matches_staged'] for f in receipt['files'].values()):
        raise SystemExit('PROMOTE_VERIFY_FAILED')
    write_json(RECEIPTS / 'promotion_receipt.json', receipt)
    print('promoted', len(receipt['files']), 'files into', FROZEN)


if __name__ == '__main__':
    promote()
