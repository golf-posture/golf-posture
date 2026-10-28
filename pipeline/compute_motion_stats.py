"""Build pro reference stats for the landmark-based motion metrics.

Reads data/pro_landmarks_raw.csv (from extract_pro_landmarks.py), computes every
metric in motion_metrics.METRIC_SPECS per pro video, and writes
data/pro_motion_stats.json (median / scaled MAD / reliability per metric).

Usage:
  pixi run python -m pipeline.compute_motion_stats
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pipeline.motion_metrics import METRIC_SPECS, RELIABILITY_RULE, build_pro_stats, compute_all

LANDMARK_NAMES = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]


def _num(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_landmarks_by_video(path: Path) -> dict[int, dict[int, dict | None]]:
    videos: dict[int, dict[int, dict | None]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            video_id = int(row["video_id"])
            event_index = int(row["event_index"])
            landmarks: dict[str, tuple[float, float] | None] = {}
            for name in LANDMARK_NAMES:
                x = _num(row.get(f"lm_{name}_x"))
                y = _num(row.get(f"lm_{name}_y"))
                landmarks[name] = (x, y) if x is not None and y is not None else None
            has_any = any(v is not None for v in landmarks.values())
            videos[video_id][event_index] = landmarks if has_any else None
    return videos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute pro stats for landmark-based motion metrics.")
    p.add_argument("--landmarks-csv", default="data/pro_landmarks_raw.csv")
    p.add_argument("--output-json", default="data/pro_motion_stats.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_landmarks_by_video(Path(args.landmarks_csv))
    total = len(videos)
    collected: dict[str, list[float]] = {spec.name: [] for spec in METRIC_SPECS}
    for lm_by_event in videos.values():
        for name, value in compute_all(lm_by_event).items():
            if value is not None:
                collected[name].append(value)

    stats = {spec.name: build_pro_stats(collected[spec.name], total, spec) for spec in METRIC_SPECS}
    stats["metadata"] = {
        "source": args.landmarks_csv,
        "total_videos": total,
        "reliability_rule": {k: {"min_detection_rate": v[0], "max_robust_cv": v[1]} for k, v in RELIABILITY_RULE.items()},
        "note": "robust_cv = scaled_MAD / max(|median|, scale_floor). Rule values are provisional team settings.",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"videos: {total}")
    print(f"{'metric':20s} {'n':>4s} {'median':>8s} {'MAD':>7s} {'p5':>7s} {'p95':>7s} {'robustCV':>8s}  reliability")
    for spec in METRIC_SPECS:
        s = stats[spec.name]
        if s.get("median") is None:
            print(f"{spec.name:20s} {s['n']:>4d}  (not enough samples)")
            continue
        print(f"{spec.name:20s} {s['n']:>4d} {s['median']:>8.2f} {s['scaled_mad']:>7.2f} {s['p5']:>7.2f} "
              f"{s['p95']:>7.2f} {s['robust_cv']:>8.2f}  {s['reliability']}")
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
