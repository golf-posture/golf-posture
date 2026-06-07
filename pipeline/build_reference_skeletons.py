from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.run_analysis import (
    CIRCULAR_FEATURES,
    EVENT_NAMES,
    LANDMARK_NAMES,
    REQUIRED_FEATURES,
    analyze_pose_backend,
    circular_delta,
    compute_features,
    init_backend,
    normalize_landmarks,
    read_frame,
    repo_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-level pro reference skeletons.")
    parser.add_argument(
        "--videos-dir",
        default=r"C:\Users\zzang\Desktop\workspace\Python\golf\src\golfdb-master\learning_video\golfdb_720p_slice",
        help="Directory containing GolfDB slice videos named <id>.mp4.",
    )
    parser.add_argument(
        "--labels",
        default="event_detection/data/golfDB_labels.csv",
        help="GolfDB labels CSV with raw event frame annotations.",
    )
    parser.add_argument(
        "--feature-stats",
        default="data/pro_feature_stats_reliability.json",
        help="Feature statistics used to choose a medoid-like representative skeleton.",
    )
    parser.add_argument("--output", default="data/pro_reference_skeletons.json")
    parser.add_argument("--pose-backend", choices=["mediapipe", "yolopose"], default="mediapipe")
    parser.add_argument("--handedness", choices=["right", "left"], default="right")
    parser.add_argument("--limit", type=int, help="Optional number of videos for smoke tests.")
    return parser.parse_args()


def parse_events(value: str) -> list[int]:
    return [int(part) for part in value.split("|") if part]


def read_labels(path: Path) -> dict[int, list[int]]:
    labels: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_events = parse_events(row["events"])
            if len(raw_events) >= 10:
                labels[int(row["id"])] = raw_events
    return labels


def frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def load_feature_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def score_candidate(features: dict[str, float | None], event_stats: dict[str, Any]) -> tuple[float, int]:
    score = 0.0
    used = 0
    for feature_name in REQUIRED_FEATURES:
        value = features.get(feature_name)
        stat = event_stats.get(feature_name, {})
        mean = stat.get("mean")
        std = stat.get("std")
        if not valid_number(value) or not valid_number(mean) or not valid_number(std) or float(std) <= 0:
            continue
        if stat.get("stat_type") == "circular_degrees" or feature_name in CIRCULAR_FEATURES:
            diff = circular_delta(float(value), float(mean))
        else:
            diff = float(value) - float(mean)
        score += (diff / float(std)) ** 2
        used += 1
    return score, used


def serializable_landmarks(landmarks: dict[str, Any]) -> dict[str, list[float] | None]:
    serialized: dict[str, list[float] | None] = {}
    for name in LANDMARK_NAMES:
        value = landmarks.get(name)
        if value is None:
            serialized[name] = None
        else:
            serialized[name] = [float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else 1.0]
    return serialized


def choose_reference(candidates: list[dict[str, Any]], event_stats: dict[str, Any]) -> dict[str, Any] | None:
    if not candidates:
        return None

    scored = []
    for candidate in candidates:
        score, used = score_candidate(candidate["features"], event_stats)
        if used:
            scored.append((score / used, -used, candidate))

    if scored:
        scored.sort(key=lambda item: (item[0], item[1]))
        chosen = scored[0][2]
        chosen["selection_score"] = float(scored[0][0])
        chosen["selection_feature_count"] = int(-scored[0][1])
        return chosen

    # Fallback: choose the skeleton closest to the coordinate mean.
    names = list(LANDMARK_NAMES)
    vectors = []
    usable = []
    for candidate in candidates:
        values = []
        for name in names:
            point = candidate["landmarks"].get(name)
            if point is None:
                values.extend([np.nan, np.nan])
            else:
                values.extend([point[0], point[1]])
        vectors.append(values)
        usable.append(candidate)
    matrix = np.asarray(vectors, dtype=float)
    center = np.nanmean(matrix, axis=0)
    distances = np.nanmean((matrix - center) ** 2, axis=1)
    idx = int(np.nanargmin(distances))
    usable[idx]["selection_score"] = float(distances[idx])
    usable[idx]["selection_feature_count"] = 0
    return usable[idx]


def build_reference_skeletons(args: argparse.Namespace) -> dict[str, Any]:
    videos_dir = Path(args.videos_dir)
    labels_path = Path(args.labels)
    if not labels_path.exists():
        fallback = repo_root() / args.labels
        if fallback.exists():
            labels_path = fallback
    feature_stats_path = Path(args.feature_stats)
    if not feature_stats_path.exists():
        fallback = repo_root() / args.feature_stats
        if fallback.exists():
            feature_stats_path = fallback

    labels = read_labels(labels_path)
    stats = load_feature_stats(feature_stats_path)
    video_paths = sorted(videos_dir.glob("*.mp4"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if args.limit is not None:
        video_paths = video_paths[: args.limit]

    backend_notes: list[str] = []
    backend = init_backend(args.pose_backend, backend_notes)
    if backend is None:
        raise RuntimeError("; ".join(backend_notes) or f"Could not initialize {args.pose_backend}")

    candidates: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(EVENT_NAMES))}
    processed = 0
    try:
        for video_i, video_path in enumerate(video_paths, start=1):
            try:
                video_id = int(video_path.stem)
            except ValueError:
                continue
            raw_events = labels.get(video_id)
            if raw_events is None:
                continue
            start_frame = raw_events[0]
            event_frames = [frame - start_frame for frame in raw_events[1:-1]]
            total_frames = frame_count(video_path)

            for event_idx, event_name in enumerate(EVENT_NAMES):
                frame_index = int(event_frames[event_idx])
                if frame_index < 0 or frame_index >= total_frames:
                    continue
                frame = read_frame(video_path, frame_index)
                if frame is None:
                    continue
                result = analyze_pose_backend(backend, frame, args.handedness)
                if not result["pose_detected"]:
                    continue
                normalized = normalize_landmarks(result["landmarks"])
                if normalized is None:
                    continue
                features = compute_features(result["landmarks"], args.handedness)
                candidates[event_idx].append(
                    {
                        "event_name": event_name,
                        "source_video_id": video_id,
                        "source_video_path": str(video_path),
                        "source_frame_index": frame_index,
                        "pose_backend": args.pose_backend,
                        "landmarks": normalized,
                        "source_landmarks_pixels": serializable_landmarks(result["landmarks"]),
                        "features": features,
                    }
                )
            processed += 1
            if video_i % 25 == 0:
                print(f"processed {video_i}/{len(video_paths)} videos")
    finally:
        backend.close()

    events: dict[str, Any] = {}
    for event_idx, event_name in enumerate(EVENT_NAMES):
        chosen = choose_reference(candidates[event_idx], stats.get(str(event_idx), {}))
        if chosen is None:
            events[str(event_idx)] = {
                "event_name": event_name,
                "landmarks": {},
                "candidate_count": 0,
                "method": "feature_medoid_normalized_skeleton",
                "notes": ["No usable pro skeleton candidates for this event."],
            }
            continue
        events[str(event_idx)] = {
            "event_name": event_name,
            "method": "feature_medoid_normalized_skeleton",
            "method_reason": (
                "Uses an actual pro frame closest to event-level pro feature statistics, "
                "avoiding anatomically invalid averaged skeletons."
            ),
            "candidate_count": len(candidates[event_idx]),
            "source_video_id": chosen["source_video_id"],
            "source_video_path": chosen["source_video_path"],
            "source_frame_index": chosen["source_frame_index"],
            "pose_backend": chosen["pose_backend"],
            "selection_score": chosen.get("selection_score"),
            "selection_feature_count": chosen.get("selection_feature_count"),
            "landmarks": chosen["landmarks"],
            "features": chosen["features"],
        }

    return {
        "metadata": {
            "source_videos_dir": str(videos_dir),
            "labels_path": str(labels_path),
            "feature_stats_path": str(feature_stats_path),
            "pose_backend": args.pose_backend,
            "handedness": args.handedness,
            "method": "feature_medoid_normalized_skeleton",
            "method_reason": (
                "MVP uses a representative real pro skeleton per event instead of a pure average skeleton."
            ),
            "processed_video_count": processed,
            "event_names": EVENT_NAMES,
            "landmark_names": list(LANDMARK_NAMES),
        },
        "events": events,
    }


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    reference = build_reference_skeletons(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)
    print(f"Reference skeletons written: {output_path}")
    for event_idx, event_name in enumerate(EVENT_NAMES):
        item = reference["events"][str(event_idx)]
        print(
            f"{event_name}: candidates={item.get('candidate_count')} "
            f"source={item.get('source_video_id')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
