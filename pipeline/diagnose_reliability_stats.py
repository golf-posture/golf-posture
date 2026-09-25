from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from pipeline.compute_reliability_stats import (
    CIRCULAR_FEATURES,
    as_float,
    circular_delta,
    read_raw_rows,
)
from pipeline.run_analysis import EVENT_NAMES, REQUIRED_FEATURES

DEFAULT_K_VALUES = [2, 3, 4, 5, 6]
KMEANS_SEED = 42
KMEANS_MAX_ITER = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose pro feature data before v2 structural changes "
        "(missing pattern, covariance condition number, cluster separability)."
    )
    parser.add_argument("--raw-csv", default="data/pro_feature_raw.csv")
    parser.add_argument("--stats-json", default="data/pro_feature_stats_reliability.json")
    parser.add_argument("--summary", default="outputs/reliability_diagnosis_summary.txt")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="Candidate cluster counts to evaluate for silhouette score.",
    )
    return parser.parse_args()


def missing_pattern_report(stats: dict) -> list[dict]:
    rows = []
    for event_index, event_name in enumerate(EVENT_NAMES):
        for feature_name in REQUIRED_FEATURES:
            item = stats[str(event_index)][feature_name]
            rows.append(
                {
                    "event_name": event_name,
                    "feature_name": feature_name,
                    "detection_rate": item["detection_rate"],
                    "missing_count": item["missing_count"],
                    "n": item["n"],
                    "total": item["total"],
                }
            )
    return rows


def event_feature_vector(row: dict, circular_centers: dict[str, float]) -> list[float] | None:
    values = []
    for feature_name in REQUIRED_FEATURES:
        value = as_float(row.get(feature_name))
        if value is None:
            return None
        if feature_name in CIRCULAR_FEATURES:
            value = circular_delta(value, circular_centers[feature_name])
        values.append(value)
    return values


def covariance_condition_numbers(rows: list[dict], stats: dict) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["event_index"])].append(row)

    results = []
    for event_index, event_name in enumerate(EVENT_NAMES):
        circular_centers = {
            feature_name: stats[str(event_index)][feature_name]["mean"]
            for feature_name in CIRCULAR_FEATURES
        }
        complete_rows = []
        for row in grouped.get(event_index, []):
            vector = event_feature_vector(row, circular_centers)
            if vector is not None:
                complete_rows.append(vector)

        n_complete = len(complete_rows)
        if n_complete < len(REQUIRED_FEATURES) + 1:
            results.append(
                {
                    "event_name": event_name,
                    "n_complete": n_complete,
                    "condition_number": None,
                }
            )
            continue

        matrix = np.asarray(complete_rows, dtype=float)
        cov = np.cov(matrix, rowvar=False)
        condition_number = float(np.linalg.cond(cov))
        assert np.isfinite(condition_number) and condition_number >= 1.0
        results.append(
            {
                "event_name": event_name,
                "n_complete": n_complete,
                "condition_number": condition_number,
            }
        )
    return results


def build_video_vectors(rows: list[dict], stats: dict) -> tuple[np.ndarray, list[str]]:
    circular_centers_by_event = {
        event_index: {
            feature_name: stats[str(event_index)][feature_name]["mean"]
            for feature_name in CIRCULAR_FEATURES
        }
        for event_index in range(len(EVENT_NAMES))
    }

    by_video: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_video[row["video_id"]][int(row["event_index"])] = row

    video_ids = []
    vectors = []
    for video_id, event_rows in by_video.items():
        if len(event_rows) != len(EVENT_NAMES):
            continue
        full_vector: list[float] = []
        is_complete = True
        for event_index in range(len(EVENT_NAMES)):
            row = event_rows.get(event_index)
            if row is None:
                is_complete = False
                break
            vector = event_feature_vector(row, circular_centers_by_event[event_index])
            if vector is None:
                is_complete = False
                break
            full_vector.extend(vector)
        if is_complete:
            video_ids.append(video_id)
            vectors.append(full_vector)

    return np.asarray(vectors, dtype=float), video_ids


def silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    n_samples = features.shape[0]
    distances = cdist(features, features)
    unique_labels = np.unique(labels)

    scores = np.zeros(n_samples)
    for i in range(n_samples):
        own_cluster = labels[i]
        own_mask = (labels == own_cluster)
        own_count = int(own_mask.sum())
        if own_count <= 1:
            scores[i] = 0.0
            continue
        a_i = distances[i, own_mask].sum() / (own_count - 1)

        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_mask = (labels == other_cluster)
            mean_dist = distances[i, other_mask].mean()
            b_i = min(b_i, mean_dist)

        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    score = float(scores.mean())
    assert -1.0 <= score <= 1.0
    return score


def cluster_separability(features: np.ndarray, k_values: list[int]) -> list[dict]:
    n_samples = features.shape[0]
    if n_samples == 0:
        return [{"k": k, "n_videos": 0, "silhouette": None} for k in k_values]

    column_std = features.std(axis=0, ddof=0)
    column_std[column_std == 0] = 1.0
    standardized = (features - features.mean(axis=0)) / column_std

    results = []
    for k in k_values:
        if n_samples <= k:
            results.append({"k": k, "n_videos": n_samples, "silhouette": None})
            continue
        rng = np.random.default_rng(KMEANS_SEED)
        _, labels = kmeans2(
            standardized, k, iter=KMEANS_MAX_ITER, minit="++", seed=rng
        )
        results.append(
            {
                "k": k,
                "n_videos": n_samples,
                "silhouette": silhouette_score(standardized, labels),
            }
        )
    return results


def write_summary(
    missing_rows: list[dict],
    covariance_rows: list[dict],
    cluster_rows: list[dict],
    n_complete_videos: int,
    summary_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Reliability diagnosis summary (v2 week 1)", ""]

    lines.append("1. Missing pattern by event x feature:")
    for row in missing_rows:
        lines.append(
            f"  {row['event_name']:20s} {row['feature_name']:20s} "
            f"detection_rate={row['detection_rate']:.3f} "
            f"missing={row['missing_count']}/{row['total']}"
        )

    lines.append("")
    lines.append("2. Covariance condition number by event (complete-case, 5 features):")
    for row in covariance_rows:
        condition_text = (
            f"{row['condition_number']:.2f}" if row["condition_number"] is not None else "n/a"
        )
        lines.append(
            f"  {row['event_name']:20s} n_complete={row['n_complete']:4d} "
            f"condition_number={condition_text}"
        )

    lines.append("")
    lines.append(f"3. Cluster separability (complete-case videos, n={n_complete_videos}):")
    for row in cluster_rows:
        silhouette_text = f"{row['silhouette']:.4f}" if row["silhouette"] is not None else "n/a"
        lines.append(
            f"  k={row['k']} n_videos={row['n_videos']} silhouette={silhouette_text}"
        )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    stats_json = Path(args.stats_json)
    summary_path = Path(args.summary)

    rows = read_raw_rows(raw_csv)
    with stats_json.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    missing_rows = missing_pattern_report(stats)
    covariance_rows = covariance_condition_numbers(rows, stats)
    video_vectors, video_ids = build_video_vectors(rows, stats)
    cluster_rows = cluster_separability(video_vectors, args.k_values)

    write_summary(missing_rows, covariance_rows, cluster_rows, len(video_ids), summary_path)
    print(f"Diagnosis summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
