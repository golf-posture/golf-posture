"""Compute swing-level (not per-event) sequence features from the raw feature CSV,
and build reliability-style stats for them — no re-processing of source videos needed,
since event frame_index values are already present in pro_feature_raw.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ADDRESS_IDX, TOP_IDX, IMPACT_IDX = 0, 3, 5


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def classify_reliability(std: float | None, detection_rate: float) -> tuple[str, str, str]:
    if std is None:
        return "low", "reference", "no valid values"
    if detection_rate >= 0.85 and std <= 0.3:
        return "high", "strong", "low dispersion and high coverage"
    if detection_rate >= 0.70 and std <= 0.6:
        return "medium", "normal", "moderate dispersion or coverage"
    return "low", "reference", "high dispersion or low coverage"


def compute_tempo_ratios(rows: list[dict]) -> dict[int, float]:
    by_video: dict[int, dict[int, int]] = defaultdict(dict)
    for row in rows:
        video_id = int(row["video_id"])
        event_index = int(row["event_index"])
        if event_index in (ADDRESS_IDX, TOP_IDX, IMPACT_IDX):
            frame_index = as_float(row.get("frame_index"))
            if frame_index is not None:
                by_video[video_id][event_index] = frame_index

    ratios: dict[int, float] = {}
    for video_id, frames in by_video.items():
        if ADDRESS_IDX in frames and TOP_IDX in frames and IMPACT_IDX in frames:
            backswing = frames[TOP_IDX] - frames[ADDRESS_IDX]
            downswing = frames[IMPACT_IDX] - frames[TOP_IDX]
            if backswing > 0 and downswing > 0:
                ratios[video_id] = backswing / downswing
    return ratios


def build_stats(ratios: dict[int, float], total_videos: int) -> dict:
    values = list(ratios.values())
    n = len(values)
    detection_rate = n / total_videos if total_videos else 0.0
    if not values:
        reliability, strength, reason = classify_reliability(None, detection_rate)
        return {
            "mean": None, "std": None, "median": None, "iqr": None,
            "n": 0, "total": total_videos, "detection_rate": detection_rate,
            "reliability": reliability, "feedback_strength": strength, "reliability_reason": reason,
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    med = float(median(values))
    iqr = float(percentile(values, 75) - percentile(values, 25))
    reliability, strength, reason = classify_reliability(std, detection_rate)
    return {
        "mean": mean, "std": std, "median": med, "iqr": iqr,
        "n": n, "total": total_videos, "detection_rate": detection_rate,
        "reliability": reliability, "feedback_strength": strength, "reliability_reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute swing-level sequence feature stats (e.g. tempo ratio).")
    parser.add_argument("--raw-csv", default="data/pro_feature_raw.csv")
    parser.add_argument("--output-json", default="data/pro_sequence_stats.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.raw_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total_videos = len({int(r["video_id"]) for r in rows})
    ratios = compute_tempo_ratios(rows)
    stats = {
        "tempo_ratio": build_stats(ratios, total_videos),
        "metadata": {
            "source": args.raw_csv,
            "definition": "tempo_ratio = (Top_frame - Address_frame) / (Impact_frame - Top_frame)",
            "total_videos": total_videos,
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    item = stats["tempo_ratio"]
    print(f"tempo_ratio: mean={item['mean']:.2f} std={item['std']:.2f} n={item['n']}/{total_videos} reliability={item['reliability']}")
    print(f"Written: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
