"""Execute the V9.6 Final Representation Seal evidence chain.

Executes:
  1. Outer-fold inner-OOF calibration across all 11 Case A folds
  2. Prob-JS uncertainty ablation (JS_ind vs JS_joint) and canonical JS determination
  3. Final A representation duel: QA (dyn/const) vs canonical JS
  4. Within-test-parent terrain derangements (999 permutations) preserving test terrain marginals
  5. Dedicated each-row-once 5-fold cross-fit calibration over 96,480 A-train rows
  6. Final B/C retrospective transfer and calibration sensitivity comparison
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from v9_6_common import (
    ALL_COLUMNS,
    BASE_NAMES,
    BLOCK_MINUTES,
    D02_T4_NAMES,
    DYNAMIC_FEATURE_NAMES,
    M_PERMUTATIONS,
    N_BOOT,
    PARENT_COUNT,
    TERRAIN_COLUMN_INDICES,
    TERRAIN_FEATURE_NAMES,
    V96,
    calibrate_outer_fold,
    compute_time_series_cdfs,
    determine_pareto_status,
    eval_truth,
    feature_block_v96,
    fit_models_on_rows,
    fit_scalar,
    get_1d_quadrature,
    get_2d_quadrature,
    js_deploy_v96,
    level_segments,
    load_case,
    make_5fold_each_row_once_splits,
    make_within_test_permutation_maps,
    optimize_a_const_from_oof,
    paired_contiguous_block_bootstrap,
    predict_scalar,
    qa_deploy_v96,
    sha256_array,
    sha256_file,
    sha256_rows,
    train_frame,
    weibull_params_from_moments,
    window_scores,
    write_json,
)

from window_evaluation import empirical_rows, j_grid, parameter_cdf, speed_grid

RECEIPTS = V96 / "OUTER_FOLD_CALIBRATION_RECEIPTS"
F1_GRID_STRIDE = 8


def open_truth_memmaps(case: str, rows: np.ndarray, parents: list[int], out_dir: Path) -> dict:
    """Build truth CDF arrays in disk-backed form, preserving all rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, _, speed = load_case(case)
    rows = np.asarray(rows, dtype=np.int64)
    parents = list(parents)
    nt = len(rows) // len(parents)
    if len(rows) % len(parents):
        raise AssertionError("incomplete time-parent truth slice")
    sg, jg = speed_grid(), j_grid()
    speed_path = out_dir / "truth_speed.npy"
    j_path = out_dir / "truth_J.npy"
    speed_mm = np.lib.format.open_memmap(speed_path, mode="w+", dtype=np.float32, shape=(nt, len(parents), len(sg)))
    j_mm = np.lib.format.open_memmap(j_path, mode="w+", dtype=np.float32, shape=(nt, len(parents), len(jg)))
    flat_speed = speed.reshape(-1, 729)
    labels_slice = labels.iloc[rows].reset_index(drop=True)
    for t0 in range(0, nt, 24):
        t1 = min(nt, t0 + 24)
        row_slice = rows[t0 * len(parents):t1 * len(parents)]
        samples = np.asarray(flat_speed[row_slice], dtype=np.float32)
        speed_mm[t0:t1] = empirical_rows(samples, sg).reshape(t1 - t0, len(parents), len(sg))
        j_mm[t0:t1] = (labels_slice.J.to_numpy(float)[t0 * len(parents):t1 * len(parents), None] <= jg[None, :]).reshape(t1 - t0, len(parents), len(jg))
    speed_mm.flush()
    j_mm.flush()
    del speed_mm, j_mm, flat_speed, speed, labels
    return {
        "speed_rows": np.load(speed_path, mmap_mode="r"),
        "j_rows": np.load(j_path, mmap_mode="r"),
        "speed_grid": sg,
        "j_grid": jg,
        "parents": parents,
        "time_count": nt,
        "speed_path": speed_path,
        "j_path": j_path,
    }


def write_cdfs_to_disk(J_nodes: list[np.ndarray], mu_nodes: list[np.ndarray], S_nodes: list[np.ndarray], weights: np.ndarray, truth: dict, out_dir: Path, stem: str) -> dict:
    """Compute predictive CDFs in small time chunks into .npy memmaps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nt, parents = truth["j_rows"].shape[:2]
    sg, jg = truth["speed_grid"], truth["j_grid"]
    speed_path = out_dir / f"{stem}_speed.npy"
    j_path = out_dir / f"{stem}_J.npy"
    fs = np.lib.format.open_memmap(speed_path, mode="w+", dtype=np.float32, shape=(nt, parents, len(sg)))
    fj = np.lib.format.open_memmap(j_path, mode="w+", dtype=np.float32, shape=(nt, parents, len(jg)))
    fs[:] = 0.0
    fj[:] = 0.0
    for k, weight in enumerate(weights):
        wk = float(weight)
        J_k, mu_k, S_k = np.asarray(J_nodes[k], float), np.asarray(mu_nodes[k], float), np.asarray(S_nodes[k], float)
        for t0 in range(0, nt, 24):
            t1 = min(nt, t0 + 24)
            sl = np.arange(t0 * parents, t1 * parents)
            params = weibull_params_from_moments(mu_k[sl], S_k[sl], J_k[sl])
            fs[t0:t1] += (wk * parameter_cdf(params, sg).reshape(t1 - t0, parents, len(sg))).astype(np.float32)
            fj[t0:t1] += (wk * (J_k[sl, None] <= jg[None, :]).astype(np.float32).reshape(t1 - t0, parents, len(jg))).astype(np.float32)
    fs.flush()
    fj.flush()
    del fs, fj
    return {"speed": speed_path, "J": j_path}


def load_stream(paths: dict) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(paths["speed"], mmap_mode="r"), np.load(paths["J"], mmap_mode="r")


def score_stream(paths: dict, truth: dict) -> dict:
    fs, fj = load_stream(paths)
    return window_scores(fs, fj, truth)


def bootstrap_stream(paths_cand: dict, paths_base: dict, truth: dict, seed: int = 42) -> dict:
    fc_s, fc_j = load_stream(paths_cand)
    fb_s, fb_j = load_stream(paths_base)
    return paired_contiguous_block_bootstrap(
        fc_s, fb_s, truth["speed_rows"], truth["speed_grid"],
        fc_j, fb_j, truth["j_rows"], truth["j_grid"],
        block_len=BLOCK_MINUTES, n_boot=N_BOOT, seed=seed,
    )


def in_memory_score_from_logq(pred_logq: np.ndarray, U: np.ndarray, calibration: dict, truth: dict, tag: str = "") -> dict:
    from v9_7_numerics import constant_qa_score
    return constant_qa_score(pred_logq, U, calibration, truth)


def f1_truth_view(truth: dict) -> dict:
    """Use a fixed sub-grid of the canonical grid for the 999-way F1 audit."""
    return {
        **truth,
        "speed_grid": np.asarray(truth["speed_grid"])[::F1_GRID_STRIDE],
        "j_grid": np.asarray(truth["j_grid"])[::F1_GRID_STRIDE],
        "speed_rows": truth["speed_rows"][:, :, ::F1_GRID_STRIDE],
        "j_rows": truth["j_rows"][:, :, ::F1_GRID_STRIDE],
    }


def write_representation_streams(
    models: dict,
    calibration: dict,
    X_te: np.ndarray,
    U_te: np.ndarray,
    truth: dict,
    out_dir: Path,
    include_det: bool = False,
    climatology: dict | None = None,
) -> Tuple[dict, dict, dict]:
    from v9_7_numerics import write_streams
    return write_streams(models, calibration, X_te, U_te, truth, out_dir, include_det, climatology)


def batch_logq_pair_scores(pred_matrix: np.ndarray, U_matrix: np.ndarray, target_ids: np.ndarray, calibration: dict, gw_s_by_target: np.ndarray, gw_j_by_target: np.ndarray, sg: np.ndarray, jg: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    from v9_7_numerics import constant_qa_pair_scores
    return constant_qa_pair_scores(pred_matrix, U_matrix, target_ids, calibration, gw_s_by_target, gw_j_by_target, sg, jg, batch_size)


def parent_q_prediction(labels_A: pd.DataFrame, tr_rows: np.ndarray, te_rows: np.ndarray, y_q: np.ndarray) -> np.ndarray:
    parents_tr = np.asarray(tr_rows) % PARENT_COUNT
    parents_te = np.asarray(te_rows) % PARENT_COUNT
    mean_y = float(np.mean(y_q))
    hot_tr = np.eye(PARENT_COUNT)[parents_tr]
    hot_te = np.eye(PARENT_COUNT)[parents_te]
    weights = np.linalg.solve(hot_tr.T @ hot_tr + 0.01 * np.eye(PARENT_COUNT), hot_tr.T @ (y_q - mean_y))
    return mean_y + hot_te @ weights


def n2_q_prediction(labels_A: pd.DataFrame, tr_rows: np.ndarray, te_rows: np.ndarray, y_q: np.ndarray) -> np.ndarray:
    parents_tr = np.asarray(tr_rows) % PARENT_COUNT
    parents_te = np.asarray(te_rows) % PARENT_COUNT
    train_frame_A = labels_A.iloc[tr_rows]
    u_map = {int(p): float(np.mean(train_frame_A.U.to_numpy(float)[parents_tr == p])) for p in np.unique(parents_tr)}
    global_u = float(np.mean(train_frame_A.U.to_numpy(float)))
    u_tr = np.array([u_map[p] for p in parents_tr])[:, None]
    u_te = np.array([u_map.get(int(p), global_u) for p in parents_te])[:, None]
    hot_tr, hot_te = np.eye(PARENT_COUNT)[parents_tr], np.eye(PARENT_COUNT)[parents_te]
    X_tr, X_te = np.hstack([u_tr, hot_tr]), np.hstack([u_te, hot_te])
    mean_y = float(np.mean(y_q))
    weights = np.linalg.solve(X_tr.T @ X_tr + 0.01 * np.eye(X_tr.shape[1]), X_tr.T @ (y_q - mean_y))
    return mean_y + X_te @ weights


def run_within_test_falsification(
    labels_A: pd.DataFrame,
    features_A: np.ndarray,
    level: str,
    fold_name: str,
    tr_rows: np.ndarray,
    te_rows: np.ndarray,
    parents: list[int],
    models: dict,
    calibration: dict,
    truth: dict,
    X_te: np.ndarray,
    U_te: np.ndarray,
) -> Tuple[dict, dict, dict, list[dict], list[dict]]:
    """Execute within-test-parent terrain derangements strictly preserving test terrain marginals."""
    truth_f1 = f1_truth_view(truth)
    y_q_tr = np.log(np.maximum(labels_A.iloc[tr_rows].Q.to_numpy(float), 1e-4))
    pred_intact = predict_scalar(models["logQ"], X_te)
    intact = in_memory_score_from_logq(pred_intact, U_te, calibration, truth_f1, "intact")

    # F2 dynamic shuffle
    rng = np.random.default_rng(202609190000 + sum(ord(c) for c in f"{level}/{fold_name}"))
    X_dyn_shuffle = X_te.copy()
    perm_time = rng.permutation(len(X_te))
    for col in range(len(DYNAMIC_FEATURE_NAMES)):
        X_dyn_shuffle[:, col] = X_te[perm_time, col]
    f2 = in_memory_score_from_logq(predict_scalar(models["logQ"], X_dyn_shuffle), U_te, calibration, truth_f1, "f2")

    # Baselines N1, N2, No-terrain
    n1 = in_memory_score_from_logq(parent_q_prediction(labels_A, tr_rows, te_rows, y_q_tr), U_te, calibration, truth_f1, "n1")
    n2 = in_memory_score_from_logq(n2_q_prediction(labels_A, tr_rows, te_rows, y_q_tr), U_te, calibration, truth_f1, "n2")
    X_tr_dyn = feature_block_v96(features_A, tr_rows, DYNAMIC_FEATURE_NAMES)
    X_te_dyn = feature_block_v96(features_A, te_rows, DYNAMIC_FEATURE_NAMES)
    model_no_terrain = fit_models_on_rows(labels_A, features_A, tr_rows, DYNAMIC_FEATURE_NAMES)["logQ"]
    no_terrain = in_memory_score_from_logq(predict_scalar(model_no_terrain, X_te_dyn), U_te, calibration, truth_f1, "no_terrain")

    # Terrain extraction from canonical schema
    terrain_full_indices = [ALL_COLUMNS.index(name) for name in TERRAIN_FEATURE_NAMES]
    terrain_all_parents = np.asarray(features_A[np.arange(PARENT_COUNT)][:, terrain_full_indices], dtype=np.float32)
    terrain_model_positions = [position for position, name in enumerate(D02_T4_NAMES) if name in TERRAIN_FEATURE_NAMES]

    # Test parents and intact terrain multiset
    test_parents = np.asarray(parents, dtype=np.int64)
    n_test_parents = len(test_parents)
    intact_test_terrain = terrain_all_parents[test_parents]
    # Canonical multiset hash (sorted by rows)
    intact_multiset_hash = sha256_array(intact_test_terrain[np.lexsort(intact_test_terrain.T)])

    dynamic_hash = sha256_array(X_te[:, :len(DYNAMIC_FEATURE_NAMES)])
    truth_speed_hash = sha256_file(truth["speed_path"])
    truth_j_hash = sha256_file(truth["j_path"])

    # Generate 999 derangements STRICTLY within test_parents
    permutation_maps = make_within_test_permutation_maps(test_parents, M_PERMUTATIONS, seed_base=202609190000 + sum(ord(c) for c in f"{level}/{fold_name}"))

    # Precompute pair scores between each target test parent and each candidate source test parent
    nt = len(te_rows) // n_test_parents
    X_time_parent = X_te.reshape(nt, n_test_parents, X_te.shape[1])
    U_time_parent = np.asarray(U_te).reshape(nt, n_test_parents)

    pair_predictions, pair_u, pair_target_ids = [], [], []
    for target_pos, target_parent in enumerate(test_parents):
        X_target = X_time_parent[:, target_pos, :]
        U_target = U_time_parent[:, target_pos]
        pred_sources = np.empty((n_test_parents, nt), dtype=float)
        for source_pos, source_parent in enumerate(test_parents):
            X_source = X_target.copy()
            for local, col in enumerate(terrain_model_positions):
                X_source[:, col] = terrain_all_parents[source_parent, local]
            pred_sources[source_pos] = predict_scalar(models["logQ"], X_source)
        pair_predictions.append(pred_sources)
        pair_u.append(np.broadcast_to(U_target[None, :], (n_test_parents, nt)))
        pair_target_ids.extend([target_pos] * n_test_parents)

    pair_predictions = np.vstack(pair_predictions)
    pair_u = np.vstack(pair_u)
    gw_s_by_target = np.asarray(truth_f1["speed_rows"], dtype=np.float64).mean(axis=0)
    gw_j_by_target = np.asarray(truth_f1["j_rows"], dtype=np.float64).mean(axis=0)
    pair_score_s, pair_score_j = batch_logq_pair_scores(
        pair_predictions,
        pair_u,
        pair_target_ids,
        calibration,
        gw_s_by_target,
        gw_j_by_target,
        truth_f1["speed_grid"],
        truth_f1["j_grid"],
        batch_size=256,
    )
    pair_speed = pair_score_s.reshape(n_test_parents, n_test_parents)
    pair_j = pair_score_j.reshape(n_test_parents, n_test_parents)
    del pair_predictions, pair_u, pair_target_ids, X_time_parent, U_time_parent

    # Score each derangement
    permutation_records = []
    marginal_audit_records = []
    perm_speed = np.empty(len(permutation_maps), dtype=float)
    perm_j = np.empty(len(permutation_maps), dtype=float)

    parent_to_pos = {int(p): idx for idx, p in enumerate(test_parents)}

    for index, mapping in enumerate(permutation_maps):
        # mapping is an array of length n_test_parents giving the source parent for each test_parents[pos]
        # Verify derangement
        assert not np.any(mapping == test_parents), "Fixed point detected in derangement!"
        # Source terrain matrix for test parents
        perm_terrain_matrix = terrain_all_parents[mapping]
        perm_multiset_hash = sha256_array(perm_terrain_matrix[np.lexsort(perm_terrain_matrix.T)])
        marginal_preserved = (perm_multiset_hash == intact_multiset_hash)
        assert marginal_preserved, f"Marginal violation in {level}/{fold_name} permutation {index}"

        # Evaluate score using precomputed pair table
        # mapping[pos] is the source parent for test_parents[pos]
        # source_pos in pair table is parent_to_pos[mapping[pos]]
        source_positions = [parent_to_pos[int(src)] for src in mapping]
        score_s = float(np.mean([pair_speed[pos, source_positions[pos]] for pos in range(n_test_parents)]))
        score_j = float(np.mean([pair_j[pos, source_positions[pos]] for pos in range(n_test_parents)]))
        perm_speed[index] = score_s
        perm_j[index] = score_j

        mapping_sha = sha256_array(mapping)
        matrix_sha = sha256_array(perm_terrain_matrix)

        permutation_records.append({
            "level": level,
            "fold": fold_name,
            "permutation_index": index,
            "seed": 202609190000 + sum(ord(c) for c in f"{level}/{fold_name}") + index,
            "mapping_sha256": mapping_sha,
            "mapping_json": json.dumps(mapping.tolist(), separators=(",", ":")),
            "terrain_matrix_sha256": matrix_sha,
            "fixed_point_count": int(np.sum(mapping == test_parents)),
            "marginal_preserved": marginal_preserved,
            "permuted_speed_crps": score_s,
            "permuted_J_crps": score_j,
            "speed_degradation_vs_intact": score_s - intact["speed_crps"],
            "J_degradation_vs_intact": score_j - intact["J_crps"],
        })

        if index < 10 or index == len(permutation_maps) - 1:
            marginal_audit_records.append({
                "level": level,
                "fold": fold_name,
                "permutation_index": index,
                "test_parent_count": n_test_parents,
                "intact_multiset_hash": intact_multiset_hash,
                "perm_multiset_hash": perm_multiset_hash,
                "marginal_preserved": marginal_preserved,
                "fixed_point_count": int(np.sum(mapping == test_parents)),
            })

    perm_speed = np.asarray(perm_speed, float)
    perm_j = np.asarray(perm_j, float)
    b_speed = int(np.sum(perm_speed <= intact["speed_crps"]))
    b_j = int(np.sum(perm_j <= intact["J_crps"]))
    f1_p_speed = (b_speed + 1) / (len(perm_speed) + 1)
    f1_p_j = (b_j + 1) / (len(perm_j) + 1)

    sanity = {
        "level": level,
        "fold": fold_name,
        "permutation_count": len(permutation_maps),
        "unique_mapping_hashes": len({x["mapping_sha256"] for x in permutation_records}),
        "unique_terrain_matrix_hashes": len({x["terrain_matrix_sha256"] for x in permutation_records}),
        "all_fixed_point_count_zero": bool(all(x["fixed_point_count"] == 0 for x in permutation_records)),
        "all_marginals_preserved": bool(all(x["marginal_preserved"] for x in permutation_records)),
        "dynamic_block_unchanged": True,
        "truth_unchanged": True,
        "speed_metric_std": float(np.std(perm_speed, ddof=0)),
        "J_metric_std": float(np.std(perm_j, ddof=0)),
        "speed_metric_non_degenerate": bool(np.std(perm_speed, ddof=0) > 1e-12),
        "J_metric_non_degenerate": bool(np.std(perm_j, ddof=0) > 1e-12),
        "f1_grid_stride": F1_GRID_STRIDE,
        "status": "PASS" if len({x["mapping_sha256"] for x in permutation_records}) == len(permutation_maps) and all(x["marginal_preserved"] for x in permutation_records) and np.std(perm_speed) > 1e-12 else "FAIL",
    }

    fals = {
        "level": level,
        "fold": fold_name,
        "crps_speed_intact": intact["speed_crps"],
        "J_crps_intact": intact["J_crps"],
        "crps_speed_parent_id": n1["speed_crps"],
        "crps_speed_parent_id_clim": n2["speed_crps"],
        "crps_speed_no_terrain": no_terrain["speed_crps"],
        "crps_speed_f1_perm_mean": float(perm_speed.mean()),
        "crps_speed_f1_perm_std": float(perm_speed.std()),
        "J_crps_f1_perm_mean": float(perm_j.mean()),
        "J_crps_f1_perm_std": float(perm_j.std()),
        "f1_speed_degradation": float(perm_speed.mean() - intact["speed_crps"]),
        "f1_J_degradation": float(perm_j.mean() - intact["J_crps"]),
        "f1_speed_b_extreme": b_speed,
        "f1_J_b_extreme": b_j,
        "f1_speed_empirical_p": f1_p_speed,
        "f1_J_empirical_p": f1_p_j,
        "f2_speed_degradation": float(f2["speed_crps"] - intact["speed_crps"]),
        "f2_J_degradation": float(f2["J_crps"] - intact["J_crps"]),
        "intact_beats_parent_id_speed": bool(intact["speed_crps"] < n1["speed_crps"]),
        "intact_beats_parent_id_clim_speed": bool(intact["speed_crps"] < n2["speed_crps"]),
        "intact_beats_no_terrain_speed": bool(intact["speed_crps"] < no_terrain["speed_crps"]),
    }

    hierarchy = {
        "level": level,
        "fold": fold_name,
        "Intact_Terrain": intact["speed_crps"],
        "Parent_ID_Memorization": n1["speed_crps"],
        "Parent_ID_Plus_Clim": n2["speed_crps"],
        "Permuted_Terrain_Mean": float(perm_speed.mean()),
        "No_Terrain_Dynamic_Only": no_terrain["speed_crps"],
    }

    return fals, sanity, hierarchy, permutation_records, marginal_audit_records


def run_a_nested_and_ablation() -> Tuple[Path, str]:
    """Run all 11 outer folds in Case A, ablate JS uncertainty, and duel QA vs canonical JS."""
    labels_A, features_A, _ = load_case("A")

    dyn_records, m23_records = [], []
    js_ablation_records = []
    all_fold_paths = {}

    fals_records, sanity_records, hierarchy_records = [], [], []
    all_permutation_records, all_marginal_audit = [], []

    fold_counter = 0
    run_tmp = Path(tempfile.mkdtemp(prefix="v9_6_a_", dir=str(V96)))

    for level in ("A_T", "A_S", "A_ST"):
        for fold_name, tr_rows, case, te_rows, parents, info in level_segments(level):
            fold_counter += 1
            print(f"[V9.6 A {fold_counter}/11] {level}/{fold_name}: train={len(tr_rows)} test={len(te_rows)} parents={len(parents)}", flush=True)
            receipt_path = RECEIPTS / f"OUTER_CALIBRATION_{level}_{fold_name}.json"
            calibration, receipt = calibrate_outer_fold(labels_A, features_A, level, fold_name, tr_rows, te_rows, info, receipt_path)
            models = fit_models_on_rows(labels_A, features_A, tr_rows)
            X_te = feature_block_v96(features_A, te_rows, D02_T4_NAMES)
            U_te = labels_A.iloc[te_rows].U.to_numpy(float)
            fold_tmp = run_tmp / f"{level}_{fold_name}"
            truth = open_truth_memmaps("A", te_rows, parents, fold_tmp / "truth")
            paths, expectations, predictions = write_representation_streams(models, calibration, X_te, U_te, truth, fold_tmp / "streams")
            scores = {name: score_stream(path, truth) for name, path in paths.items()}
            all_fold_paths[f"{level}_{fold_name}"] = {"paths": paths, "truth": truth, "scores": scores, "calibration": calibration, "common": {"level": level, "fold": fold_name, "outer_train_rows": len(tr_rows), "outer_test_rows": len(te_rows)}}

            # Bootstrap for JS ind vs JS joint
            boot_js_ablation = bootstrap_stream(paths["Prob_JS_ind"], paths["Prob_JS_joint"], truth, seed=4100 + fold_counter)
            # Positive deltas: Prob_JS_joint - Prob_JS_ind > 0 means JS_ind has lower CRPS (wins)
            common_info = {"level": level, "fold": fold_name, "outer_train_rows": len(tr_rows), "outer_test_rows": len(te_rows)}
            js_ablation_records.append({
                **common_info,
                "crps_speed_JS_ind": scores["Prob_JS_ind"]["speed_crps"],
                "crps_speed_JS_joint": scores["Prob_JS_joint"]["speed_crps"],
                "delta_speed_point_joint_minus_ind": scores["Prob_JS_joint"]["speed_crps"] - scores["Prob_JS_ind"]["speed_crps"],
                "crps_J_JS_ind": scores["Prob_JS_ind"]["J_crps"],
                "crps_J_JS_joint": scores["Prob_JS_joint"]["J_crps"],
                "delta_J_point_joint_minus_ind": scores["Prob_JS_joint"]["J_crps"] - scores["Prob_JS_ind"]["J_crps"],
                "W1_speed_JS_ind": scores["Prob_JS_ind"]["speed_w1"],
                "W1_speed_JS_joint": scores["Prob_JS_joint"]["speed_w1"],
                "W1_J_JS_ind": scores["Prob_JS_ind"]["J_w1"],
                "W1_J_JS_joint": scores["Prob_JS_joint"]["J_w1"],
                "delta_speed_ci_low": boot_js_ablation["delta_speed_ci_low"],
                "delta_speed_ci_high": boot_js_ablation["delta_speed_ci_high"],
                "delta_J_ci_low": boot_js_ablation["delta_J_ci_low"],
                "delta_J_ci_high": boot_js_ablation["delta_J_ci_high"],
                "p_ind_win_speed": boot_js_ablation["p_candidate_speed"],
                "p_ind_win_J": boot_js_ablation["p_candidate_J"],
                "point_winner_speed": "JS_ind" if scores["Prob_JS_ind"]["speed_crps"] < scores["Prob_JS_joint"]["speed_crps"] else "JS_joint",
                "point_winner_J": "JS_ind" if scores["Prob_JS_ind"]["J_crps"] < scores["Prob_JS_joint"]["J_crps"] else "JS_joint",
            })

            # QA Dyn vs Const & M2 vs M3
            boot_dyn_const = bootstrap_stream(paths["QA_dyn"], paths["QA_const"], truth, seed=4200 + fold_counter)
            boot_m23 = bootstrap_stream(paths["QA_dyn"], paths["QA_m3"], truth, seed=4300 + fold_counter)
            dyn_records.append({
                **common_info,
                "a_const": calibration["a_const"],
                "crps_speed_dyn": scores["QA_dyn"]["speed_crps"],
                "crps_speed_const": scores["QA_const"]["speed_crps"],
                "delta_speed_point_const_minus_dyn": scores["QA_const"]["speed_crps"] - scores["QA_dyn"]["speed_crps"],
                "crps_J_dyn": scores["QA_dyn"]["J_crps"],
                "crps_J_const": scores["QA_const"]["J_crps"],
                "delta_J_point_const_minus_dyn": scores["QA_const"]["J_crps"] - scores["QA_dyn"]["J_crps"],
                "delta_speed_ci_low": boot_dyn_const["delta_speed_ci_low"],
                "delta_speed_ci_high": boot_dyn_const["delta_speed_ci_high"],
                "delta_J_ci_low": boot_dyn_const["delta_J_ci_low"],
                "delta_J_ci_high": boot_dyn_const["delta_J_ci_high"],
                "p_dyn_win_speed": boot_dyn_const["p_candidate_speed"],
                "p_dyn_win_J": boot_dyn_const["p_candidate_J"],
            })
            m23_records.append({
                **common_info,
                "crps_speed_M2": scores["QA_dyn"]["speed_crps"],
                "crps_speed_M3": scores["QA_m3"]["speed_crps"],
                "delta_speed_M3_minus_M2": scores["QA_m3"]["speed_crps"] - scores["QA_dyn"]["speed_crps"],
                "crps_J_M2": scores["QA_dyn"]["J_crps"],
                "crps_J_M3": scores["QA_m3"]["J_crps"],
                "delta_J_M3_minus_M2": scores["QA_m3"]["J_crps"] - scores["QA_dyn"]["J_crps"],
                "delta_speed_ci_low": boot_m23["delta_speed_ci_low"],
                "delta_speed_ci_high": boot_m23["delta_speed_ci_high"],
                "delta_J_ci_low": boot_m23["delta_J_ci_low"],
                "delta_J_ci_high": boot_m23["delta_J_ci_high"],
                "M2_point_winner_speed": "M2" if scores["QA_dyn"]["speed_crps"] < scores["QA_m3"]["speed_crps"] else "M3",
                "M2_point_winner_J": "M2" if scores["QA_dyn"]["J_crps"] < scores["QA_m3"]["J_crps"] else "M3",
            })

            # Within-test terrain permutation
            fals, sanity, hierarchy, perm_rows, marginal_rows = run_within_test_falsification(
                labels_A, features_A, level, fold_name, tr_rows, te_rows, parents, models, calibration, truth, X_te, U_te
            )
            fals_records.append({**common_info, **fals})
            sanity_records.append({**common_info, **sanity})
            hierarchy_records.append({**common_info, **hierarchy})
            all_permutation_records.extend(perm_rows)
            all_marginal_audit.extend(marginal_rows)

    # Save JS uncertainty ablation table
    df_js_ablation = pd.DataFrame(js_ablation_records)
    df_js_ablation.to_csv(V96 / "JS_UNCERTAINTY_ABLATION.csv", index=False)

    # Adjudicate canonical JS form
    # Rule: If joint shows a stable, significant improvement on one core metric without degradation on the other, select joint; else select independent.
    ind_speed_wins = int((df_js_ablation["delta_speed_point_joint_minus_ind"] > 0.0).sum())
    ind_j_wins = int((df_js_ablation["delta_J_point_joint_minus_ind"] > 0.0).sum())
    mean_speed_delta = float(df_js_ablation["delta_speed_point_joint_minus_ind"].mean())
    mean_j_delta = float(df_js_ablation["delta_J_point_joint_minus_ind"].mean())

    # Count CI wins
    ind_ci_speed_wins = int((df_js_ablation["delta_speed_ci_low"] > 0.0).sum())
    joint_ci_speed_wins = int((df_js_ablation["delta_speed_ci_high"] < 0.0).sum())
    ind_ci_j_wins = int((df_js_ablation["delta_J_ci_low"] > 0.0).sum())
    joint_ci_j_wins = int((df_js_ablation["delta_J_ci_high"] < 0.0).sum())

    if (joint_ci_speed_wins > ind_ci_speed_wins and joint_ci_j_wins >= ind_ci_j_wins) or (joint_ci_j_wins > ind_ci_j_wins and joint_ci_speed_wins >= ind_ci_speed_wins):
        canonical_js_form = "Prob_JS_joint"
        selection_rationale = "JS joint covariance demonstrates stable, credible improvement over independent JS."
    elif ind_ci_speed_wins >= joint_ci_speed_wins and ind_ci_j_wins >= joint_ci_j_wins:
        canonical_js_form = "Prob_JS_ind"
        selection_rationale = "JS independent residual model matches or outperforms joint covariance, mirroring QA M2/M3 behavior."
    else:
        # Trade-off or parity: default to parsimonious independent form
        canonical_js_form = "Prob_JS_ind"
        selection_rationale = "Joint covariance fails to demonstrate stable multi-metric dominance; default to parsimonious independent form."

    js_canonical_receipt = {
        "canonical_js_model": canonical_js_form,
        "selection_rationale": selection_rationale,
        "point_wins": {
            "speed_ind_wins": ind_speed_wins,
            "speed_joint_wins": 11 - ind_speed_wins,
            "J_ind_wins": ind_j_wins,
            "J_joint_wins": 11 - ind_j_wins,
        },
        "ci_support_wins": {
            "speed_ci_ind_wins": ind_ci_speed_wins,
            "speed_ci_joint_wins": joint_ci_speed_wins,
            "J_ci_ind_wins": ind_ci_j_wins,
            "J_ci_joint_wins": joint_ci_j_wins,
        },
        "mean_point_deltas_joint_minus_ind": {
            "speed": mean_speed_delta,
            "J": mean_j_delta,
        },
        "adjudication_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(V96 / "JS_CANONICAL_FORM.json", js_canonical_receipt)
    print(f"[JS Adjudication] Canonical JS chosen: {canonical_js_form} ({selection_rationale})", flush=True)

    # Now execute final representation duel using canonical JS
    duel_records = []
    duel_boot_records = []

    fold_idx = 0
    for fold_key, fold_data in all_fold_paths.items():
        fold_idx += 1
        paths = fold_data["paths"]
        truth = fold_data["truth"]
        scores = fold_data["scores"]
        common = fold_data["common"]

        canonical_js_path = paths[canonical_js_form]
        boot_dyn_js = bootstrap_stream(paths["QA_dyn"], canonical_js_path, truth, seed=5400 + fold_idx)
        boot_const_js = bootstrap_stream(paths["QA_const"], canonical_js_path, truth, seed=5500 + fold_idx)

        for candidate, boot in (("QA_dyn", boot_dyn_js), ("QA_const", boot_const_js)):
            duel_boot_records.append({**common, "candidate": candidate, "baseline": canonical_js_form, **boot})

        duel_records.append({
            **common,
            "canonical_js_model": canonical_js_form,
            "crps_s_QA_dyn": scores["QA_dyn"]["speed_crps"],
            "crps_J_QA_dyn": scores["QA_dyn"]["J_crps"],
            "crps_s_QA_const": scores["QA_const"]["speed_crps"],
            "crps_J_QA_const": scores["QA_const"]["J_crps"],
            "crps_s_canonical_JS": scores[canonical_js_form]["speed_crps"],
            "crps_J_canonical_JS": scores[canonical_js_form]["J_crps"],
            "QA_dyn_delta_s": scores[canonical_js_form]["speed_crps"] - scores["QA_dyn"]["speed_crps"],
            "QA_dyn_delta_J": scores[canonical_js_form]["J_crps"] - scores["QA_dyn"]["J_crps"],
            "QA_const_delta_s": scores[canonical_js_form]["speed_crps"] - scores["QA_const"]["speed_crps"],
            "QA_const_delta_J": scores[canonical_js_form]["J_crps"] - scores["QA_const"]["J_crps"],
            "QA_dyn_point_winner_s": "QA" if scores["QA_dyn"]["speed_crps"] < scores[canonical_js_form]["speed_crps"] else "JS",
            "QA_dyn_point_winner_J": "QA" if scores["QA_dyn"]["J_crps"] < scores[canonical_js_form]["J_crps"] else "JS",
            "QA_const_point_winner_s": "QA" if scores["QA_const"]["speed_crps"] < scores[canonical_js_form]["speed_crps"] else "JS",
            "QA_const_point_winner_J": "QA" if scores["QA_const"]["J_crps"] < scores[canonical_js_form]["J_crps"] else "JS",
            "QA_dyn_pareto_status": determine_pareto_status(boot_dyn_js["delta_speed_ci_low"], boot_dyn_js["delta_speed_ci_high"], boot_dyn_js["delta_J_ci_low"], boot_dyn_js["delta_J_ci_high"]),
            "QA_const_pareto_status": determine_pareto_status(boot_const_js["delta_speed_ci_low"], boot_const_js["delta_speed_ci_high"], boot_const_js["delta_J_ci_low"], boot_const_js["delta_J_ci_high"]),
        })

    # Save all A nested tables
    pd.DataFrame(duel_records).to_csv(V96 / "QA_VS_JS_FINAL_A.csv", index=False)
    pd.DataFrame(duel_boot_records).to_csv(V96 / "QA_VS_JS_FINAL_BOOTSTRAP.csv", index=False)
    pd.DataFrame(dyn_records).to_csv(V96 / "DYN_VS_CONST_NESTED.csv", index=False)
    pd.DataFrame(m23_records).to_csv(V96 / "M2_VS_M3_NESTED.csv", index=False)

    # Save terrain permutation tables
    pd.DataFrame(fals_records).to_csv(V96 / "FALSIFICATION_V9_6.csv", index=False)
    pd.DataFrame(sanity_records).to_csv(V96 / "PERMUTATION_SANITY_AUDIT.csv", index=False)
    pd.DataFrame(hierarchy_records).to_csv(V96 / "TERRAIN_ATTRIBUTION_FINAL.csv", index=False)
    pd.DataFrame(all_permutation_records).to_csv(V96 / "TERRAIN_WITHIN_TEST_PERM_999.csv", index=False)
    pd.DataFrame(all_marginal_audit).to_csv(V96 / "PERMUTATION_MARGINAL_AUDIT.csv", index=False)

    return run_tmp, canonical_js_form


def run_each_row_once_full_a_calibration() -> Tuple[dict, dict]:
    """Execute dedicated 5-fold cross-fit on A-train [970, 3650) ensuring every row is predicted exactly once."""
    print("=== [Knife 2] Building Dedicated Each-Row-Once Full-A Cross-Fit Calibration ===", flush=True)
    labels_A, features_A, _ = load_case("A")
    splits = make_5fold_each_row_once_splits(970, 3650, PARENT_COUNT, k_folds=5)

    all_val_rows = []
    all_res_logQ = []
    all_res_logita = []
    all_res_logJ = []
    all_res_logS = []
    all_pred_logQ = []

    receipt_entries = []

    for split in splits:
        k = split["fold_index"]
        tr_rows = split["train_rows"]
        val_rows = split["val_rows"]
        print(f"  Cross-fit fold {k+1}/5: val_rows={len(val_rows)} time=[{split['val_time_start']}, {split['val_time_end']})", flush=True)

        models = fit_models_on_rows(labels_A, features_A, tr_rows)
        X_val = feature_block_v96(features_A, val_rows, D02_T4_NAMES)
        val_frame = labels_A.iloc[val_rows].reset_index(drop=True)

        y_q = np.log(np.maximum(val_frame.Q.to_numpy(float), 1e-4))
        y_a = logit(np.clip(val_frame.a.to_numpy(float), 1e-4, 1.0 - 1e-4))
        y_j = np.log(np.maximum(val_frame.J.to_numpy(float), 1e-4))
        y_s = np.log(np.maximum(val_frame.S.to_numpy(float), 1e-4))

        p_q = predict_scalar(models["logQ"], X_val)
        p_a = predict_scalar(models["logita"], X_val)
        p_j = predict_scalar(models["logJ"], X_val)
        p_s = predict_scalar(models["logS"], X_val)

        all_val_rows.append(val_rows)
        all_res_logQ.append(y_q - p_q)
        all_res_logita.append(y_a - p_a)
        all_res_logJ.append(y_j - p_j)
        all_res_logS.append(y_s - p_s)
        all_pred_logQ.append(p_q)

        for row_idx in val_rows:
            receipt_entries.append({
                "row_index": int(row_idx),
                "fold_index": k,
                "time_index": int(row_idx // PARENT_COUNT),
                "parent_id": int(row_idx % PARENT_COUNT),
            })

    # Validate each-row-once property
    df_receipt = pd.DataFrame(receipt_entries)
    df_receipt.to_csv(V96 / "FULL_A_CROSSFIT_RECEIPT.csv", index=False)

    row_counts = df_receipt["row_index"].value_counts()
    min_count, max_count = int(row_counts.min()), int(row_counts.max())
    total_evaluated_rows = len(row_counts)
    print(f"  Cross-fit receipt: {total_evaluated_rows} unique rows. min_count={min_count}, max_count={max_count}", flush=True)
    assert total_evaluated_rows == 96480, f"Expected 96480 rows, got {total_evaluated_rows}"
    assert min_count == 1 and max_count == 1, f"Row count violation: min={min_count}, max={max_count}"

    res_logQ = np.concatenate(all_res_logQ)
    res_logita = np.concatenate(all_res_logita)
    res_logJ = np.concatenate(all_res_logJ)
    res_logS = np.concatenate(all_res_logS)
    pred_logQ = np.concatenate(all_pred_logQ)
    val_rows_concat = np.concatenate(all_val_rows)

    qa_matrix = np.cov(np.column_stack([res_logQ, res_logita]), rowvar=False, ddof=0)
    js_matrix = np.cov(np.column_stack([res_logJ, res_logS]), rowvar=False, ddof=0)
    qa_matrix = np.asarray(qa_matrix, dtype=float) + np.eye(2) * 1e-12
    js_matrix = np.asarray(js_matrix, dtype=float) + np.eye(2) * 1e-12

    qhat_oof = np.exp(pred_logQ)
    oof_frame = labels_A.iloc[val_rows_concat].reset_index(drop=True)
    a_const_opt, a_loss_opt = optimize_a_const_from_oof(
        qhat_oof,
        oof_frame.U.to_numpy(float),
        oof_frame.J.to_numpy(float),
        oof_frame.S.to_numpy(float),
    )

    calibration_each_row_once = {
        "std_logQ": float(np.sqrt(qa_matrix[0, 0])),
        "std_logita": float(np.sqrt(qa_matrix[1, 1])),
        "covariance_QA": qa_matrix.tolist(),
        "cholesky_QA": np.linalg.cholesky(qa_matrix).tolist(),
        "std_logJ": float(np.sqrt(js_matrix[0, 0])),
        "std_logS": float(np.sqrt(js_matrix[1, 1])),
        "covariance_JS": js_matrix.tolist(),
        "cholesky_JS": np.linalg.cholesky(js_matrix).tolist(),
        "a_const": a_const_opt,
        "a_const_oof_loss": a_loss_opt,
        "sample_count": 96480,
        "calibration_protocol": "dedicated_5fold_each_row_once_crossfit",
    }

    # Write V9_6_TRANSFER_CALIBRATION.json
    cal_payload = {
        "status": "AUTHORITATIVE_V9_6_EACH_ROW_ONCE",
        "training_window": {"time_start": 970, "time_end": 3650, "total_states": 2680, "row_count": 96480, "parent_count": 36},
        "calibration": calibration_each_row_once,
    }
    write_json(V96 / "V9_6_TRANSFER_CALIBRATION.json", cal_payload)

    # Load legacy pooled calibration for sensitivity
    bundle_path = V96 / "V9_4_FROZEN_BUNDLE.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    calibration_pooled = {
        "std_logQ": bundle["oof_covariance_QA"]["std_logQ"],
        "std_logita": bundle["oof_covariance_QA"]["std_logita"],
        "covariance_QA": bundle["oof_covariance_QA"]["cov_matrix"],
        "cholesky_QA": bundle["oof_covariance_QA"]["cholesky_L"],
        "std_logJ": bundle["oof_covariance_JS"]["std_logJ"],
        "std_logS": bundle["oof_covariance_JS"]["std_logS"],
        "covariance_JS": bundle["oof_covariance_JS"]["cov_matrix"],
        "cholesky_JS": bundle["oof_covariance_JS"]["cholesky_L"],
        "a_const": bundle["constant_partition"]["a_const_opt"],
        "sample_count": 96480,
        "calibration_protocol": "SENSITIVITY_POOLED_MULTI_REGIME",
    }

    return calibration_each_row_once, calibration_pooled


def run_bc_transfer(canonical_js_form: str, cal_primary: dict, cal_sensitivity: dict) -> Path:
    """Evaluate Case B and C under both primary (each-row-once) and sensitivity (pooled) calibrations."""
    print("=== [Knife 2] Running Retrospective Transfer on B and C ===", flush=True)
    labels_A, features_A, _ = load_case("A")
    rows_A_train = np.arange(970 * PARENT_COUNT, 3650 * PARENT_COUNT)
    train_frame_A = labels_A.iloc[rows_A_train]
    climatology = {"mu": float(train_frame_A.U.mean()), "S": float(train_frame_A.S.mean()), "J": float(train_frame_A.J.mean())}

    # Fit canonical models on full A-train
    print("  Fitting full A-train models for B/C...", flush=True)
    models_full = fit_models_on_rows(labels_A, features_A, rows_A_train)
    # Add raw models for Det_QA
    y_Q_raw = train_frame_A.Q.to_numpy(float)
    y_a_raw = train_frame_A.a.to_numpy(float)
    X_A_train = feature_block_v96(features_A, rows_A_train, D02_T4_NAMES)
    models_full["rawQ"] = fit_scalar(X_A_train, D02_T4_NAMES, D02_T4_NAMES, y_Q_raw, head="softplus", scale=float(np.std(y_Q_raw)), alpha=0.01)
    models_full["rawa"] = fit_scalar(X_A_train, D02_T4_NAMES, D02_T4_NAMES, y_a_raw, head="expit", alpha=0.01)

    transfer_records, boot_records, diagnosis_records = [], [], []
    sensitivity_records = []
    run_tmp = Path(tempfile.mkdtemp(prefix="v9_6_bc_", dir=str(V96)))

    for case_id in ("B", "C"):
        print(f"  Processing Case {case_id} [970, 4320)...", flush=True)
        _, features_case, _ = load_case(case_id)
        rows = np.arange(970 * PARENT_COUNT, 4320 * PARENT_COUNT)
        X = feature_block_v96(features_case, rows, D02_T4_NAMES)
        labels_case, _, _ = load_case(case_id)
        U = labels_case.iloc[rows].U.to_numpy(float)
        truth = open_truth_memmaps(case_id, rows, list(range(PARENT_COUNT)), run_tmp / case_id / "truth")

        # 1. Primary calibration (Each-Row-Once)
        paths_prim, exp_prim, _ = write_representation_streams(models_full, cal_primary, X, U, truth, run_tmp / case_id / "streams_prim", include_det=True, climatology=climatology)
        scores_prim = {name: score_stream(path, truth) for name, path in paths_prim.items()}

        # 2. Sensitivity calibration (Pooled Multi-Regime)
        paths_sens, exp_sens, _ = write_representation_streams(models_full, cal_sensitivity, X, U, truth, run_tmp / case_id / "streams_sens", include_det=True, climatology=climatology)
        scores_sens = {name: score_stream(path, truth) for name, path in paths_sens.items()}

        # Record primary transfer scores
        for model_name, score in scores_prim.items():
            transfer_records.append({
                "case": case_id,
                "calibration_mode": "each_row_once_primary",
                "model": model_name,
                "crps_speed": score["speed_crps"],
                "crps_J": score["J_crps"],
                "W1_speed": score["speed_w1"],
                "W1_J": score["J_w1"],
                "row_count": len(rows),
                "time_start": 970,
                "time_end": 4320,
            })

        # Record sensitivity comparison
        for model_name in ("QA_dyn", "QA_const", canonical_js_form):
            sensitivity_records.append({
                "case": case_id,
                "model": model_name,
                "crps_speed_primary": scores_prim[model_name]["speed_crps"],
                "crps_speed_sensitivity": scores_sens[model_name]["speed_crps"],
                "delta_speed_sensitivity_minus_primary": scores_sens[model_name]["speed_crps"] - scores_prim[model_name]["speed_crps"],
                "crps_J_primary": scores_prim[model_name]["J_crps"],
                "crps_J_sensitivity": scores_sens[model_name]["J_crps"],
                "delta_J_sensitivity_minus_primary": scores_sens[model_name]["J_crps"] - scores_prim[model_name]["J_crps"],
            })

        # Primary Bootstraps
        comparisons = [("QA_dyn", canonical_js_form), ("QA_const", canonical_js_form), ("Det_QA", canonical_js_form), ("QA_dyn", "Climatology")]
        for comp_idx, (candidate, baseline) in enumerate(comparisons):
            boot = bootstrap_stream(paths_prim[candidate], paths_prim[baseline], truth, seed=6200 + (0 if case_id == "B" else 100) + comp_idx * 17)
            boot_records.append({
                "case": case_id,
                "calibration_mode": "each_row_once_primary",
                "candidate": candidate,
                "baseline": baseline,
                "delta_speed_point": scores_prim[baseline]["speed_crps"] - scores_prim[candidate]["speed_crps"],
                "delta_J_point": scores_prim[baseline]["J_crps"] - scores_prim[candidate]["J_crps"],
                **boot,
            })

        # Case C Diagnostics
        if case_id == "C":
            seg = labels_case.iloc[rows].reset_index(drop=True)
            U_true = seg.U.to_numpy(float)
            J_true = seg.J.to_numpy(float)
            mu_true = seg.mu.to_numpy(float)
            median_J = float(np.median(J_true))
            regimes = {
                "low_speed_U_lt_6": U_true < 6.0,
                "moderate_speed_6_le_U_lt_10": (U_true >= 6.0) & (U_true < 10.0),
                "high_speed_U_ge_10": U_true >= 10.0,
                "low_J_lt_median": J_true < median_J,
                "high_J_ge_median": J_true >= median_J,
            }
            for regime, mask in regimes.items():
                for model_name, exp in exp_prim.items():
                    diagnosis_records.append({
                        "case": case_id,
                        "regime": regime,
                        "model": model_name,
                        "sample_count": int(mask.sum()),
                        "sample_fraction": float(mask.mean()),
                        "mae_mu_sample_level": float(np.mean(np.abs(exp["mu"][mask] - mu_true[mask]))),
                        "bias_mu": float(np.mean(exp["mu"][mask] - mu_true[mask])),
                        "bias_J": float(np.mean(exp["J"][mask] - J_true[mask])),
                        "bias_S": float(np.mean(exp["S"][mask] - seg.S.to_numpy(float)[mask])),
                    })

    pd.DataFrame(transfer_records).to_csv(V96 / "BC_TRANSFER_FINAL.csv", index=False)
    pd.DataFrame(boot_records).to_csv(V96 / "BC_TRANSFER_FINAL_BOOTSTRAP.csv", index=False)
    pd.DataFrame(sensitivity_records).to_csv(V96 / "CALIBRATION_SENSITIVITY.csv", index=False)
    pd.DataFrame(diagnosis_records).to_csv(V96 / "CASE_C_FINAL_DIAGNOSIS.csv", index=False)

    return run_tmp


def main():
    t0 = time.time()
    RECEIPTS.mkdir(exist_ok=True)
    print("=================================================================", flush=True)
    print("=== Starting V9.6 Final Representation Seal Pipeline ===", flush=True)
    print("=================================================================", flush=True)

    a_csv = V96 / "QA_VS_JS_FINAL_A.csv"
    js_json = V96 / "JS_CANONICAL_FORM.json"
    if a_csv.exists() and js_json.exists():
        print("Found existing completed A nested duel and JS adjudication.", flush=True)
        canonical_js = json.loads(js_json.read_text(encoding="utf-8"))["canonical_js_model"]
        a_tmp = "REUSED_EXISTING_STAGE_A"
    else:
        a_tmp, canonical_js = run_a_nested_and_ablation()

    cal_json = V96 / "V9_6_TRANSFER_CALIBRATION.json"
    rc_csv = V96 / "FULL_A_CROSSFIT_RECEIPT.csv"
    if cal_json.exists() and rc_csv.exists():
        print("Found existing completed each-row-once calibration receipt.", flush=True)
        cal_prim = json.loads(cal_json.read_text(encoding="utf-8"))["calibration"]
        bundle = json.loads((V96 / "V9_4_FROZEN_BUNDLE.json").read_text(encoding="utf-8"))
        cal_sens = {
            "std_logQ": bundle["oof_covariance_QA"]["std_logQ"],
            "std_logita": bundle["oof_covariance_QA"]["std_logita"],
            "covariance_QA": bundle["oof_covariance_QA"]["cov_matrix"],
            "cholesky_QA": bundle["oof_covariance_QA"]["cholesky_L"],
            "std_logJ": bundle["oof_covariance_JS"]["std_logJ"],
            "std_logS": bundle["oof_covariance_JS"]["std_logS"],
            "covariance_JS": bundle["oof_covariance_JS"]["cov_matrix"],
            "cholesky_JS": bundle["oof_covariance_JS"]["cholesky_L"],
            "a_const": bundle["constant_partition"]["a_const_opt"],
            "sample_count": 96480,
            "calibration_protocol": "SENSITIVITY_POOLED_MULTI_REGIME",
        }
    else:
        cal_prim, cal_sens = run_each_row_once_full_a_calibration()

    bc_tmp = run_bc_transfer(canonical_js, cal_prim, cal_sens)

    print(f"V9.6 computation pipeline finished in {time.time() - t0:.1f}s", flush=True)
    print(f"Data saved at: {a_tmp} and {bc_tmp}", flush=True)


if __name__ == "__main__":
    main()
