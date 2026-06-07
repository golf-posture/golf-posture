from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from pipeline.run_analysis import (
    EVENT_NAMES,
    REQUIRED_FEATURES,
    analyze_pose_backend,
    init_backend,
    read_frame,
    repo_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw pro pose features from GolfDB slice videos.")
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
    parser.add_argument("--output", default="data/pro_feature_raw.csv")
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


def failure_reason(pose_detected: bool, features: dict[str, float | None]) -> str:
    if not pose_detected:
        return "pose_not_detected"
    missing = [name for name in REQUIRED_FEATURES if features.get(name) is None]
    if missing:
        return "missing_features:" + "|".join(missing)
    return ""


def write_rows(args: argparse.Namespace) -> int:
    videos_dir = Path(args.videos_dir)
    labels_path = Path(args.labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = read_labels(labels_path)
    video_paths = sorted(videos_dir.glob("*.mp4"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if args.limit is not None:
        video_paths = video_paths[: args.limit]

    backend_notes: list[str] = []
    backend = init_backend(args.pose_backend, backend_notes)
    if backend is None:
        raise RuntimeError("; ".join(backend_notes) or f"Could not initialize {args.pose_backend}")

    fieldnames = [
        "video_id",
        "video_path",
        "event_index",
        "event_name",
        "frame_index",
        "pose_backend",
        "pose_detected",
        *REQUIRED_FEATURES,
        "missing_features",
        "failure_reason",
    ]

    rows_written = 0
    try:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
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

                for event_index, event_name in enumerate(EVENT_NAMES):
                    frame_index = int(event_frames[event_index])
                    base_row = {
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "event_index": event_index,
                        "event_name": event_name,
                        "frame_index": frame_index,
                        "pose_backend": args.pose_backend,
                    }
                    if frame_index < 0 or frame_index >= total_frames:
                        features = {name: None for name in REQUIRED_FEATURES}
                        row = {
                            **base_row,
                            "pose_detected": False,
                            **features,
                            "missing_features": "|".join(REQUIRED_FEATURES),
                            "failure_reason": f"frame_out_of_range:total_frames={total_frames}",
                        }
                        writer.writerow(row)
                        rows_written += 1
                        continue

                    frame = read_frame(video_path, frame_index)
                    if frame is None:
                        features = {name: None for name in REQUIRED_FEATURES}
                        pose_detected = False
                    else:
                        result = analyze_pose_backend(backend, frame, args.handedness)
                        features = result["features"]
                        pose_detected = bool(result["pose_detected"])

                    missing = [name for name in REQUIRED_FEATURES if features.get(name) is None]
                    row = {
                        **base_row,
                        "pose_detected": pose_detected,
                        **features,
                        "missing_features": "|".join(missing),
                        "failure_reason": failure_reason(pose_detected, features),
                    }
                    writer.writerow(row)
                    rows_written += 1

                if video_i % 25 == 0:
                    print(f"processed {video_i}/{len(video_paths)} videos")
    finally:
        backend.close()

    return rows_written


def main() -> int:
    args = parse_args()
    if not Path(args.labels).exists():
        fallback = repo_root() / args.labels
        if fallback.exists():
            args.labels = str(fallback)
    rows = write_rows(args)
    print(f"Raw feature CSV written: {args.output}")
    print(f"Rows written: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
