from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from pipeline.compute_reliability_stats import CIRCULAR_FEATURES, as_float, circular_delta, read_raw_rows
from pipeline.run_analysis import EVENT_NAMES, REQUIRED_FEATURES

MIN_SAMPLES_FOR_DYNAMICS = 30
MAD_SCALE = 1.4826


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pro swing dynamics (event-to-event delta) statistics using "
        "robust median/MAD, for run_analysis.py's 'dynamics' output section."
    )
    parser.add_argument("--raw-csv", default="data/pro_feature_raw.csv")
    parser.add_argument("--output-json", default="data/pro_dynamics_stats.json")
    parser.add_argument("--summary", default="outputs/pro_dynamics_summary.txt")
    return parser.parse_args()


def event_pairs() -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(len(EVENT_NAMES) - 1)]


def pair_label(start_idx: int, end_idx: int) -> str:
    return f"{EVENT_NAMES[start_idx]}->{EVENT_NAMES[end_idx]}"


def compute_delta(start_value: float, end_value: float, feature_name: str) -> float:
    if feature_name in CIRCULAR_FEATURES:
        return circular_delta(end_value, start_value)
    return end_value - start_value


def robust_stats(values: list[float]) -> dict:
    center = median(values)
    mad = median([abs(v - center) for v in values])
    return {
        "median": center,
        "mad": mad,
        "scaled_mad": MAD_SCALE * mad,
        "n": len(values),
    }


def compute_dynamics_stats(rows: list[dict]) -> dict:
    by_video: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_video[row["video_id"]][int(row["event_index"])] = row

    pairs = event_pairs()
    pair_deltas: dict[tuple[int, int], dict[str, list[float]]] = {
        pair: {feature_name: [] for feature_name in REQUIRED_FEATURES} for pair in pairs
    }

    for event_rows in by_video.values():
        for start_idx, end_idx in pairs:
            start_row = event_rows.get(start_idx)
            end_row = event_rows.get(end_idx)
            if start_row is None or end_row is None:
                continue
            for feature_name in REQUIRED_FEATURES:
                start_value = as_float(start_row.get(feature_name))
                end_value = as_float(end_row.get(feature_name))
                if start_value is None or end_value is None:
                    continue
                delta = compute_delta(start_value, end_value, feature_name)
                pair_deltas[(start_idx, end_idx)][feature_name].append(delta)

    total_videos = len(by_video)
    pairs_output: dict[str, dict] = {}
    for start_idx, end_idx in pairs:
        label = pair_label(start_idx, end_idx)
        features_output = {}
        for feature_name in REQUIRED_FEATURES:
            values = pair_deltas[(start_idx, end_idx)][feature_name]
            n = len(values)
            detection_rate = n / total_videos if total_videos else 0.0
            if n < MIN_SAMPLES_FOR_DYNAMICS:
                features_output[feature_name] = {
                    "median": None,
                    "mad": None,
                    "scaled_mad": None,
                    "n": n,
                    "detection_rate": detection_rate,
                }
                continue
            stats = robust_stats(values)
            stats["detection_rate"] = detection_rate
            features_output[feature_name] = stats
        pairs_output[label] = {
            "start_event": EVENT_NAMES[start_idx],
            "end_event": EVENT_NAMES[end_idx],
            "start_event_index": start_idx,
            "end_event_index": end_idx,
            "features": features_output,
        }

    return {
        "metadata": {
            "source": "data/pro_feature_raw.csv",
            "features": REQUIRED_FEATURES,
            "circular_features": sorted(CIRCULAR_FEATURES),
            "event_names": EVENT_NAMES,
            "total_videos": total_videos,
            "min_samples_for_stats": MIN_SAMPLES_FOR_DYNAMICS,
            "delta_rule": "linear features: end - start; circular features: circular_delta(end, start).",
            "robust_stats_rule": "median/MAD (scaled by 1.4826), not mean/std -- see v2 week1/2 outlier findings.",
        },
        "pairs": pairs_output,
    }


def write_summary(dynamics_stats: dict, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Pro swing dynamics (event-to-event delta) summary", ""]
    for label, pair_data in dynamics_stats["pairs"].items():
        lines.append(f"{label}:")
        for feature_name, stats in pair_data["features"].items():
            if stats["median"] is None:
                lines.append(
                    f"  {feature_name:20s} n={stats['n']:4d} detect={stats['detection_rate']:.3f} "
                    "insufficient samples"
                )
                continue
            lines.append(
                f"  {feature_name:20s} n={stats['n']:4d} median={stats['median']:+7.2f} "
                f"scaled_mad={stats['scaled_mad']:6.2f} detect={stats['detection_rate']:.3f}"
            )
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    output_json = Path(args.output_json)
    summary_path = Path(args.summary)

    rows = read_raw_rows(raw_csv)
    dynamics_stats = compute_dynamics_stats(rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(dynamics_stats, f, indent=2, ensure_ascii=False)
    write_summary(dynamics_stats, summary_path)

    print(f"Dynamics stats JSON written: {output_json}")
    print(f"Dynamics summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
