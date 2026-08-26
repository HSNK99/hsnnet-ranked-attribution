from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "core" / "bossbase_04"
VALIDATION = OUT / "validation_response_matrix_z.csv"
TEST = OUT / "test_response_matrix_z.csv"
SOURCE_STEGO = OUT / "source_evidence" / "ranked_candidate_analysis_per_sample.csv"

ALGORITHMS = ["WOW", "S-UNIWARD", "HILL", "HUGO", "MiPOD"]
Q_LOW = 10
Q_HIGH = 90
GAMMA = 0.900


def soft_range_score(z: float, low: float, high: float) -> float:
    if z < low:
        distance = low - z
    elif z > high:
        distance = z - high
    else:
        distance = 0.0
    return float(np.exp(-distance))


def build_signatures(validation: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    signatures: dict[str, dict[str, tuple[float, float]]] = {}
    rows = []
    quantiles = [5, 10, 25, 50, 75, 90, 95]

    for target in ALGORITHMS:
        subset = validation.loc[
            (validation["is_stego"] == 1)
            & (validation["true_algo"] == target)
        ]
        if len(subset) != 1500:
            raise ValueError(f"Unexpected validation count for {target}: {len(subset)}")
        signatures[target] = {}
        for dimension in ALGORITHMS:
            values = subset[f"z_{dimension}"].to_numpy(dtype=np.float64)
            q = np.percentile(values, quantiles)
            low = float(q[1])
            high = float(q[5])
            signatures[target][dimension] = (low, high)
            rows.append(
                {
                    "target_algo": target,
                    "template_dimension": dimension,
                    "q_low_used": Q_LOW,
                    "q_high_used": Q_HIGH,
                    "q05": float(q[0]),
                    "q10": low,
                    "q25": float(q[2]),
                    "q50": float(q[3]),
                    "q75": float(q[4]),
                    "q90": high,
                    "q95": float(q[6]),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "range_low": low,
                    "range_high": high,
                }
            )
    return signatures, pd.DataFrame(rows)


def recover_cover_normalization(validation: pd.DataFrame) -> dict:
    cover = validation.loc[validation["is_stego"] == 0]
    if len(cover) != 1500:
        raise ValueError(f"Unexpected validation-cover count: {len(cover)}")
    result = {}
    for algorithm in ALGORITHMS:
        p = cover[f"p_{algorithm}"].to_numpy(dtype=np.float64)
        z = cover[f"z_{algorithm}"].to_numpy(dtype=np.float64)
        design = np.column_stack([z, np.ones(len(z))])
        sigma_used, mean = np.linalg.lstsq(design, p, rcond=None)[0]
        reconstructed = mean + sigma_used * z
        max_error = float(np.max(np.abs(reconstructed - p)))
        if max_error > 1e-12:
            raise ValueError(f"Cannot recover exact normalization for {algorithm}")
        result[algorithm] = {
            "validation_cover_n": int(len(cover)),
            "mean": float(mean),
            "sigma_used": float(sigma_used),
            "reconstruction_max_abs_error": max_error,
        }
    return result


def score_test(test: pd.DataFrame, signatures: dict) -> pd.DataFrame:
    output_rows = []
    for _, row in test.iterrows():
        scores = {}
        inside_counts = {}
        for target in ALGORITHMS:
            matches = []
            inside = 0
            for dimension in ALGORITHMS:
                z = float(row[f"z_{dimension}"])
                low, high = signatures[target][dimension]
                matches.append(soft_range_score(z, low, high))
                inside += int(low <= z <= high)
            scores[target] = float(np.mean(matches))
            inside_counts[target] = inside

        # Python sorting is stable. Equal scores therefore preserve ALGORITHMS order.
        ranked = sorted(ALGORITHMS, key=lambda name: scores[name], reverse=True)
        candidates = [name for name in ranked if scores[name] >= GAMMA]
        candidate_size = len(candidates)
        is_stego = int(row["is_stego"])
        true_algo = str(row["true_algo"])
        true_rank = ranked.index(true_algo) + 1 if is_stego else np.nan
        contains_true = int(is_stego and true_algo in candidates)

        if candidate_size == 0:
            decision = "UNKNOWN_OR_COVER"
            top1 = "UNKNOWN"
            top2 = ""
        elif candidate_size == 1:
            decision = "SINGLE_CANDIDATE"
            top1 = ranked[0]
            top2 = ranked[1]
        else:
            decision = "MULTI_CANDIDATE"
            top1 = ranked[0]
            top2 = ranked[1]

        item = {
            "sample_id": row["sample_id"],
            "image_name": row["image_name"],
            "true_algo": true_algo,
            "is_stego": is_stego,
            "decision": decision,
            "top1": top1,
            "top2": top2,
            "rank1_full": ranked[0],
            "rank2_full": ranked[1],
            "rank3_full": ranked[2],
            "rank4_full": ranked[3],
            "rank5_full": ranked[4],
            "true_rank": true_rank,
            "candidate_set": ",".join(candidates),
            "candidate_size": candidate_size,
            "candidate_contains_true": contains_true,
            "top1_score": scores[ranked[0]],
            "top2_score": scores[ranked[1]],
            "score_gap": scores[ranked[0]] - scores[ranked[1]],
        }
        for target in ALGORITHMS:
            item[f"score_{target}"] = scores[target]
            item[f"hard_{target}"] = inside_counts[target] / len(ALGORITHMS)
            item[f"inside_count_{target}"] = inside_counts[target]
        output_rows.append(item)
    return pd.DataFrame(output_rows)


def summarize(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stego = predictions.loc[predictions["is_stego"] == 1].copy()
    cover = predictions.loc[predictions["is_stego"] == 0].copy()
    sizes = stego["candidate_size"]
    cover_sizes = cover["candidate_size"]

    summary = {
        "num_total": int(len(predictions)),
        "num_stego": int(len(stego)),
        "num_cover": int(len(cover)),
        "ranked_top1_accuracy": float(stego["true_rank"].le(1).mean()),
        "ranked_top2_accuracy": float(stego["true_rank"].le(2).mean()),
        "ranked_top3_accuracy": float(stego["true_rank"].le(3).mean()),
        "decision_top1_accuracy": float(
            (stego["true_rank"].eq(1) & sizes.gt(0)).mean()
        ),
        "candidate_set_accuracy": float(stego["candidate_contains_true"].mean()),
        "stego_unknown_rate": float(sizes.eq(0).mean()),
        "stego_single_candidate_rate": float(sizes.eq(1).mean()),
        "stego_multi_candidate_rate": float(sizes.gt(1).mean()),
        "stego_mean_candidate_size": float(sizes.mean()),
        "expected_search_reduction": float((5 - sizes.mean()) / 5),
        "mean_true_rank": float(stego["true_rank"].mean()),
        "cover_unknown_rate": float(cover_sizes.eq(0).mean()),
        "cover_false_attribution_rate": float(cover_sizes.gt(0).mean()),
        "cover_single_candidate_false_attribution_rate": float(cover_sizes.eq(1).mean()),
        "cover_multi_candidate_false_attribution_rate": float(cover_sizes.gt(1).mean()),
        "cover_mean_candidate_size": float(cover_sizes.mean()),
        "method": "cross_template_response_signature_candidate_attribution",
        "dataset": "BOSSBase 1.01",
        "payload": "0.4bpp",
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "gamma": GAMMA,
        "candidate_universe": ALGORITHMS,
        "ranking_before_filtering": True,
    }

    per_algorithm_rows = []
    for algorithm in ALGORITHMS:
        frame = stego.loc[stego["true_algo"] == algorithm]
        s = frame["candidate_size"]
        per_algorithm_rows.append(
            {
                "algorithm": algorithm,
                "n": int(len(frame)),
                "ranked_top1": float(frame["true_rank"].le(1).mean()),
                "ranked_top2": float(frame["true_rank"].le(2).mean()),
                "ranked_top3": float(frame["true_rank"].le(3).mean()),
                "decision_top1": float(
                    (frame["true_rank"].eq(1) & s.gt(0)).mean()
                ),
                "candidate_retention": float(frame["candidate_contains_true"].mean()),
                "unknown_rate": float(s.eq(0).mean()),
                "single_rate": float(s.eq(1).mean()),
                "multi_rate": float(s.gt(1).mean()),
                "mean_candidate_size": float(s.mean()),
                "mean_true_rank": float(frame["true_rank"].mean()),
            }
        )

    size_rows = []
    for size in range(6):
        frame = stego.loc[stego["candidate_size"] == size]
        size_rows.append(
            {
                "set_size": size,
                "n": int(len(frame)),
                "share": float(len(frame) / len(stego)),
                "retention": float(frame["candidate_contains_true"].mean()),
                "top1": float(frame["true_rank"].le(1).mean()),
                "top2": float(frame["true_rank"].le(2).mean()),
                "top3": float(frame["true_rank"].le(3).mean()),
                "mean_true_rank": float(frame["true_rank"].mean()),
            }
        )
    return summary, pd.DataFrame(per_algorithm_rows), pd.DataFrame(size_rows)


def verify_against_archived_export(predictions: pd.DataFrame) -> None:
    if not SOURCE_STEGO.is_file():
        raise FileNotFoundError(SOURCE_STEGO)
    archived = pd.read_csv(SOURCE_STEGO)
    current = predictions.loc[predictions["is_stego"] == 1].copy()
    current["ranked_list"] = current[
        ["rank1_full", "rank2_full", "rank3_full", "rank4_full", "rank5_full"]
    ].agg(",".join, axis=1)
    merged = archived.merge(
        current[["sample_id", "ranked_list", "candidate_set", "candidate_size"]],
        on="sample_id",
        suffixes=("_archived", "_regenerated"),
        validate="one_to_one",
    )
    if len(merged) != 7500:
        raise ValueError(f"Archived comparison matched {len(merged)} rather than 7500 rows")
    for field in ["ranked_list", "candidate_set", "candidate_size"]:
        left = merged[f"{field}_archived"].fillna("").astype(str)
        right = merged[f"{field}_regenerated"].fillna("").astype(str)
        mismatches = int((left != right).sum())
        if mismatches:
            raise ValueError(f"{field}: {mismatches} mismatches against archived export")


def assert_locked_results(summary: dict) -> None:
    expected = {
        "ranked_top1_accuracy": 3211 / 7500,
        "ranked_top2_accuracy": 4939 / 7500,
        "ranked_top3_accuracy": 6171 / 7500,
        "decision_top1_accuracy": 3154 / 7500,
        "candidate_set_accuracy": 6780 / 7500,
        "stego_unknown_rate": 153 / 7500,
        "stego_mean_candidate_size": 25714 / 7500,
        "cover_unknown_rate": 782 / 1500,
        "cover_false_attribution_rate": 718 / 1500,
        "cover_single_candidate_false_attribution_rate": 133 / 1500,
        "cover_multi_candidate_false_attribution_rate": 585 / 1500,
    }
    for key, value in expected.items():
        if not np.isclose(summary[key], value, atol=1e-12, rtol=0):
            raise ValueError(f"Locked-result mismatch for {key}: {summary[key]} != {value}")


def main() -> None:
    validation = pd.read_csv(VALIDATION, low_memory=False)
    test = pd.read_csv(TEST, low_memory=False)
    signatures, signature_table = build_signatures(validation)
    predictions = score_test(test, signatures)
    summary, per_algorithm, size_conditioned = summarize(predictions)

    assert_locked_results(summary)
    verify_against_archived_export(predictions)

    signature_table.to_csv(OUT / "final_cross_template_signature_table.csv", index=False)
    # Preserve the released per-image prediction exports as immutable evidence.
    # Rankings and candidate sets were recomputed and verified above.
    per_algorithm.to_csv(OUT / "final_per_algorithm.csv", index=False)
    size_conditioned.to_csv(OUT / "table_9_candidate_size_conditioned.csv", index=False)
    (OUT / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # Recompute and validate cover normalization without rewriting the released
    # platform-sensitive floating-point serialization.
    _ = recover_cover_normalization(validation)
    signature_json = {
        target: {
            dimension: {"low": low, "high": high}
            for dimension, (low, high) in dimensions.items()
        }
        for target, dimensions in signatures.items()
    }
    (OUT / "q10_q90_response_signatures.json").write_text(
        json.dumps(signature_json, indent=2), encoding="utf-8"
    )

    # Normalize generated text files to LF for cross-platform checksums.
    for output_path in list(OUT.glob("final_*")) + [
        OUT / "cover_normalization_statistics.json",
        OUT / "q10_q90_response_signatures.json",
        OUT / "table_9_candidate_size_conditioned.csv",
    ]:
        output_path.write_bytes(
            output_path.read_bytes().replace(b"\r\n", b"\n")
        )

    print("PASS | BOSSBase 0.4-bpp manuscript benchmark reproduced")
    print("PASS | 7,500 archived rankings and candidate sets match exactly")
    print(
        "Top-1/2/3 = "
        f"{summary['ranked_top1_accuracy']*100:.2f}% / "
        f"{summary['ranked_top2_accuracy']*100:.2f}% / "
        f"{summary['ranked_top3_accuracy']*100:.2f}%"
    )
    print(
        "Retention / mean set / unknown = "
        f"{summary['candidate_set_accuracy']*100:.2f}% / "
        f"{summary['stego_mean_candidate_size']:.2f} / "
        f"{summary['stego_unknown_rate']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
