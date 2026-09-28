"""Fail-closed split, CV, input and dependency contracts for the repair."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))


def split_indices(case, split):
    start,end=config()['splits'][case][split]
    return np.arange(start,end,dtype=np.int32)


def validate_keys(frame,case,split,parents):
    required={'case','time_index','parent_id'}
    if not required.issubset(frame):
        raise ValueError('Missing identity keys')
    if set(frame.case)!= {case} or frame.duplicated(['case','time_index','parent_id']).any():
        raise ValueError('Mixed case or duplicate key')
    expected=pd.MultiIndex.from_product([split_indices(case,split),parents],names=['time_index','parent_id'])
    found=pd.MultiIndex.from_frame(frame[['time_index','parent_id']])
    if len(found)!=len(expected) or not found.sort_values().equals(expected.sort_values()):
        raise ValueError('Missing, extra or doubly excluded time/parent keys')


def aggregate_cv(rows, required_folds=(1,2,3)):
    keys=['head_id','config_id']
    if rows.duplicated(keys+['fold']).any():
        raise ValueError('Duplicate CV fold')
    output=[]
    for (head,cid),g in rows.groupby(keys,sort=True):
        valid=(set(g.fold)==set(required_folds) and g.success.eq(True).all() and np.isfinite(g.score).all())
        output.append(dict(head_id=head,config_id=int(cid),fold_count=len(g),eligible=bool(valid),
                           mean_score=float(g.score.mean()) if valid else None,
                           sd_score=float(g.score.std(ddof=0)) if valid else None))
    return pd.DataFrame(output)


def forward_calibration_masks(time_indices, train_end, valid_start, valid_end):
    t=np.asarray(time_indices)
    if not train_end<=valid_start<valid_end:
        raise ValueError('Calibration intervals overlap or are reversed')
    train=t<train_end; valid=(t>=valid_start)&(t<valid_end)
    if not train.any() or not valid.any() or t[train].max()>=t[valid].min():
        raise ValueError('Insufficient forward calibration samples')
    return train,valid


def legal_inputs(frame, feature_names):
    allowed=set(config()['legal_input_columns'])
    if not set(feature_names)<=allowed:
        raise ValueError('Undeclared or label-derived feature')
    values=frame[list(feature_names)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction input')
    return values


def dependency_hash(metadata):
    required={'case','time_key_hash','parent_key_hash','terrain_version','feature_version','target_version','config_hash','source_hash','git_commit','input_hashes'}
    if not required<=set(metadata) or any(metadata[k] is None for k in required-{'git_commit'}):
        raise ValueError('Incomplete cache dependency manifest')
    return hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()
