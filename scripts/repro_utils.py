from __future__ import annotations
import math, random, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ["WOW", "S-UNIWARD", "HILL", "HUGO", "MiPOD"]
SCORE_COLUMNS = {
    "WOW": "score_WOW",
    "S-UNIWARD": "score_S-UNIWARD",
    "HILL": "score_HILL",
    "HUGO": "score_HUGO",
    "MiPOD": "score_MiPOD",
}
EXPECTED_RESULTS = {
    "bossbase_04": dict(top1=42.813, top2=65.853, top3=82.280, retention=90.400,
                        mean_set=3.429, unknown=2.040, search_reduction=31.429,
                        mean_true_rank=2.157, stego_n=7500),
    "bossbase_02": dict(top1=30.107, top2=53.613, top3=72.587, retention=77.013,
                        mean_set=3.444, unknown=16.333, search_reduction=31.112,
                        mean_true_rank=2.548, stego_n=7500),
    "bows2_04": dict(top1=35.466, top2=59.260, top3=77.502, retention=91.262,
                     mean_set=3.759, unknown=1.694, search_reduction=24.827,
                     mean_true_rank=2.366, stego_n=50000),
    "bows2_02": dict(top1=24.564, top2=47.146, top3=67.148, retention=88.610,
                     mean_set=4.184, unknown=6.244, search_reduction=16.329,
                     mean_true_rank=2.758, stego_n=50000),
}
PREDICTION_LOCATIONS = {
    "bossbase_04": [
        "results/core/bossbase_04/final_predictions_stego.csv",
        "results/core/bossbase_04/final_predictions_stego.csv.gz",
    ],
    "bossbase_02": [
        "results/core/bossbase_02/final_predictions_stego.csv",
        "results/core/bossbase_02/final_predictions_stego.csv.gz",
    ],
    "bows2_04": [
        "results/core/bows2_04/external_candidate_predictions.csv",
        "results/core/bows2_04/external_candidate_predictions.csv.gz",
        "results/risk_conformal/step5_bows2_04/bows2_04_per_image_stego_predictions.csv",
        "results/risk_conformal/step5_bows2_04/bows2_04_per_image_stego_predictions.csv.gz",
    ],
    "bows2_02": [
        "results/core/bows2_02/external_candidate_predictions.csv",
        "results/core/bows2_02/external_candidate_predictions.csv.gz",
    ],
}

def canonical_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

def canonical_algorithm(value: Any) -> str:
    mapping = {"wow":"WOW", "suniward":"S-UNIWARD", "hill":"HILL",
               "hugo":"HUGO", "mipod":"MiPOD", "cover":"COVER"}
    return mapping.get(canonical_name(value), str(value).strip())

def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype(str).str.strip().str.lower().isin(
        {"1","true","yes","y","retained","included","correct","pass"}
    )

def find_column(columns, preferred):
    mapping = {canonical_name(c): c for c in columns}
    for name in preferred:
        if canonical_name(name) in mapping:
            return mapping[canonical_name(name)]
    return None

def locate_prediction(condition: str, root: Path = ROOT) -> Path:
    for rel in PREDICTION_LOCATIONS[condition]:
        path = root / rel
        if path.is_file():
            return path
    raise FileNotFoundError(f"No prediction artifact found for {condition}")

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def stego_frame(frame: pd.DataFrame) -> pd.DataFrame:
    col = find_column(frame.columns, ["is_stego"])
    if col:
        return frame.loc[parse_bool(frame[col])].copy()
    true_col = find_column(frame.columns, ["true_algo","true_algorithm","target_algorithm"])
    if true_col:
        labels = frame[true_col].map(canonical_algorithm)
        return frame.loc[labels.isin(ALGORITHMS)].copy()
    return frame.copy()

def compute_condition_metrics(frame: pd.DataFrame) -> dict:
    frame = stego_frame(frame)
    rank_col = find_column(frame.columns, ["true_rank"])
    size_col = find_column(frame.columns, ["candidate_size","candidate_set_size","set_size"])
    ret_col = find_column(frame.columns, ["candidate_contains_true","candidate_retained","retained"])
    if not rank_col or not size_col or not ret_col:
        raise ValueError("Missing true_rank/candidate_size/candidate_contains_true")
    ranks = pd.to_numeric(frame[rank_col], errors="raise")
    sizes = pd.to_numeric(frame[size_col], errors="raise")
    retained = parse_bool(frame[ret_col])
    return {
        "stego_n": int(len(frame)),
        "top1": float(ranks.le(1).mean()*100),
        "top2": float(ranks.le(2).mean()*100),
        "top3": float(ranks.le(3).mean()*100),
        "retention": float(retained.mean()*100),
        "mean_set": float(sizes.mean()),
        "unknown": float(sizes.eq(0).mean()*100),
        "search_reduction": float((5-sizes.mean())/5*100),
        "mean_true_rank": float(ranks.mean()),
    }

def condition_check(condition, observed):
    tol = dict(top1=.002, top2=.002, top3=.002, retention=.002, mean_set=.002,
               unknown=.002, search_reduction=.003, mean_true_rank=.002, stego_n=0)
    rows = []
    for metric, expected in EXPECTED_RESULTS[condition].items():
        diff = abs(float(observed[metric])-float(expected))
        rows.append(dict(condition=condition, metric=metric, observed=observed[metric],
                         expected=expected, difference=diff,
                         status="PASS" if diff <= tol[metric] else "FAIL"))
    return all(r["status"]=="PASS" for r in rows), rows

def prepare_bows2_04(root: Path = ROOT):
    path = locate_prediction("bows2_04", root)
    frame = stego_frame(read_table(path))
    true_col = find_column(frame.columns, ["true_algo","true_algorithm","target_algorithm"])
    labels = frame[true_col].map(canonical_algorithm).to_numpy()
    true_indices = np.array([ALGORITHMS.index(x) for x in labels], dtype=int)
    scores = frame[[SCORE_COLUMNS[a] for a in ALGORITHMS]].apply(
        pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    return path, frame, labels, true_indices, scores

def set_metrics(matrix, true_indices):
    matrix = np.asarray(matrix, dtype=bool)
    sizes = matrix.sum(axis=1)
    contains = matrix[np.arange(len(matrix)), true_indices]
    nonempty = sizes > 0
    risk = float((~contains[nonempty]).mean()*100) if nonempty.any() else float("nan")
    return {
        "retention_percent": float(contains.mean()*100),
        "mean_set_size": float(sizes.mean()),
        "search_reduction_percent": float((5-sizes.mean())/5*100),
        "empty_percent": float((~nonempty).mean()*100),
        "decision_coverage_percent": float(nonempty.mean()*100),
        "conditional_false_exclusion_percent": risk,
        "set_sizes": sizes,
        "contains_true": contains,
    }

def make_table8a(scores, true_indices, gamma=.900):
    metrics = set_metrics(scores >= gamma, true_indices)
    sizes, retained = metrics["set_sizes"], metrics["contains_true"]
    rows = []
    for k in range(1,6):
        admitted = (sizes >= 1) & (sizes <= k)
        rows.append({
            "max_set_k": k,
            "accepted_n": int(admitted.sum()),
            "coverage_percent": float(admitted.mean()*100),
            "conditional_retention_percent": float(retained[admitted].mean()*100),
            "conditional_false_exclusion_percent": float((~retained[admitted]).mean()*100),
            "mean_admitted_size": float(sizes[admitted].mean()),
        })
    return pd.DataFrame(rows)

def exact_mondrian(labels, true_indices, scores, alpha=.10, seed=42):
    n = len(labels)
    folds = np.full(n, -1, dtype=int)
    rng = random.Random(seed)
    for algo in ALGORITHMS:
        idx = np.where(labels == algo)[0].tolist()
        rng.shuffle(idx)
        half = len(idx)//2
        folds[idx[:half]] = 0
        folds[idx[half:]] = 1
    sets = np.zeros((n,5), dtype=bool)
    thresholds = []
    for eval_fold in [0,1]:
        cal = folds != eval_fold
        eva = folds == eval_fold
        for j, algo in enumerate(ALGORITHMS):
            vals = np.sort(1.0-scores[cal & (labels==algo), j])
            q = math.ceil((len(vals)+1)*(1-alpha))
            q = min(len(vals), max(1,q))
            threshold = float(vals[q-1])
            score_threshold = 1.0-threshold
            sets[eva,j] = scores[eva,j] >= score_threshold
            thresholds.append(dict(evaluation_fold=eval_fold, algorithm=algo,
                                   calibration_n=len(vals), quantile_index_1based=q,
                                   alpha=alpha, nonconformity_threshold=threshold,
                                   equivalent_score_threshold=score_threshold))
    return set_metrics(sets,true_indices), sets, folds, pd.DataFrame(thresholds)
