from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

from pipeline.run_analysis import EVENT_NAMES, REQUIRED_FEATURES


CIRCULAR_FEATURES = {"shoulder_tilt", "hip_tilt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pro feature reliability statistics.")
    parser.add_argument("--raw-csv", default="data/pro_feature_raw.csv")
    parser.add_argument("--output-json", default="data/pro_feature_stats_reliability.json")
    parser.add_argument("--summary", default="outputs/pro_feature_reliability_summary.txt")
    parser.add_argument("--stats-csv", default="data/pro_feature_stats_reliability.csv")
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def circular_mean(values: Iterable[float]) -> float:
    radians = np.radians(list(values))
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    return float(np.degrees(np.arctan2(sin_mean, cos_mean)))


def circular_delta(value: float, center: float) -> float:
    return float((value - center + 180.0) % 360.0 - 180.0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def classify_reliability(dispersion: float | None, detection_rate: float) -> tuple[str, str, str]:
    if dispersion is None:
        return "low", "reference", "no valid feature values"
    if detection_rate >= 0.85 and dispersion <= 10.0:
        return "high", "strong", "low dispersion and high detection rate"
    if detection_rate >= 0.70 and dispersion <= 20.0:
        return "medium", "normal", "moderate dispersion or detection rate"
    return "low", "reference", "high dispersion or low detection rate"


def compute_feature_stats(rows: list[dict], event_index: int, feature_name: str) -> dict:
    total = len(rows)
    values = [as_float(row.get(feature_name)) for row in rows]
    valid = [value for value in values if value is not None and math.isfinite(value)]
    n = len(valid)
    missing_count = total - n
    detection_rate = n / total if total else 0.0

    if not valid:
        reliability, strength, reason = classify_reliability(None, detection_rate)
        return {
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "n": 0,
            "total": total,
            "detection_rate": detection_rate,
            "missing_count": missing_count,
            "stat_type": "circular_degrees" if feature_name in CIRCULAR_FEATURES else "linear_degrees",
            "reliability": reliability,
            "feedback_strength": strength,
            "reliability_reason": reason,
        }

    if feature_name in CIRCULAR_FEATURES:
        center = circular_mean(valid)
        deltas = [circular_delta(value, center) for value in valid]
        std = float(np.std(np.asarray(deltas)))
        med = circular_mean(valid)
        q1 = percentile(deltas, 25)
        q3 = percentile(deltas, 75)
        iqr = float(q3 - q1)
        stat_type = "circular_degrees"
    else:
        center = float(np.mean(np.asarray(valid)))
        std = float(np.std(np.asarray(valid)))
        med = float(median(valid))
        iqr = float(percentile(valid, 75) - percentile(valid, 25))
        stat_type = "linear_degrees"

    reliability, strength, reason = classify_reliability(std, detection_rate)
    return {
        "mean": center,
        "std": std,
        "median": med,
        "iqr": iqr,
        "n": n,
        "total": total,
        "detection_rate": detection_rate,
        "missing_count": missing_count,
        "stat_type": stat_type,
        "reliability": reliability,
        "feedback_strength": strength,
        "reliability_reason": reason,
    }


def read_raw_rows(raw_csv: Path) -> list[dict]:
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_stats(rows: list[dict]) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["event_index"])].append(row)

    stats: dict[str, dict] = {}
    for event_index in range(len(EVENT_NAMES)):
        event_rows = grouped.get(event_index, [])
        stats[str(event_index)] = {}
        for feature_name in REQUIRED_FEATURES:
            stats[str(event_index)][feature_name] = compute_feature_stats(
                event_rows, event_index, feature_name
            )

    stats["metadata"] = {
        "source": "data/pro_feature_raw.csv",
        "features": REQUIRED_FEATURES,
        "circular_features": sorted(CIRCULAR_FEATURES),
        "event_names": EVENT_NAMES,
        "reliability_rules": {
            "high": "detection_rate >= 0.85 and dispersion_std <= 10 degrees",
            "medium": "detection_rate >= 0.70 and dispersion_std <= 20 degrees",
            "low": "otherwise; use as reference only",
        },
        "total_raw_rows": len(rows),
    }
    return stats


def write_stats_csv(stats: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_index",
        "event_name",
        "feature_name",
        "stat_type",
        "mean",
        "std",
        "median",
        "iqr",
        "n",
        "total",
        "detection_rate",
        "missing_count",
        "reliability",
        "feedback_strength",
        "reliability_reason",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event_index, event_name in enumerate(EVENT_NAMES):
            for feature_name in REQUIRED_FEATURES:
                item = stats[str(event_index)][feature_name]
                writer.writerow(
                    {
                        "event_index": event_index,
                        "event_name": event_name,
                        "feature_name": feature_name,
                        **item,
                    }
                )


def write_summary(stats: dict, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value) -> str:
        if value is None:
            return "n/a"
        return f"{value:.2f}"

    lines = [
        "Pro feature reliability summary",
        "",
        "Rules:",
        "  high   = detection_rate >= 0.85 and std <= 10 degrees",
        "  medium = detection_rate >= 0.70 and std <= 20 degrees",
        "  low    = reference only",
        "",
    ]

    flat = []
    for event_index, event_name in enumerate(EVENT_NAMES):
        for feature_name in REQUIRED_FEATURES:
            item = stats[str(event_index)][feature_name]
            flat.append((event_name, feature_name, item))

    lines.append("High reliability candidates:")
    for event_name, feature_name, item in flat:
        if item["reliability"] == "high":
            lines.append(
                f"  - {event_name} / {feature_name}: std={fmt(item['std'])}, "
                f"IQR={fmt(item['iqr'])}, detection={item['detection_rate']:.3f}"
            )

    lines.append("")
    lines.append("By event:")
    for event_index, event_name in enumerate(EVENT_NAMES):
        lines.append(f"{event_name}:")
        event_items = [
            (feature_name, stats[str(event_index)][feature_name])
            for feature_name in REQUIRED_FEATURES
        ]
        for feature_name, item in sorted(event_items, key=lambda pair: pair[1]["std"] or 9999):
            lines.append(
                f"  {feature_name:20s} {item['reliability']:6s} "
                f"std={fmt(item['std'])} IQR={fmt(item['iqr'])} "
                f"detect={item['detection_rate']:.3f} n={item['n']}"
            )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    output_json = Path(args.output_json)
    summary_path = Path(args.summary)
    stats_csv = Path(args.stats_csv)

    rows = read_raw_rows(raw_csv)
    stats = compute_stats(rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    write_stats_csv(stats, stats_csv)
    write_summary(stats, summary_path)

    print(f"Reliability JSON written: {output_json}")
    print(f"Reliability CSV written: {stats_csv}")
    print(f"Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
