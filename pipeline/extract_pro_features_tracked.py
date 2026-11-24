"""Extract pro pose features with video-mode tracking, saving both 2D and 3D elbow angles.

Differences from extract_pro_features.py (isolated frames, 2D only, visibility >= 0.5):
  - sequential MediaPipe tracking over the whole swing (static_image_mode=False)
  - elbow angles use visibility >= 0.3, recovered from neighbouring frames when the event frame is invalid
  - elbow angle is stored in 2D (pixel) and 3D (world landmarks); the source column records how it was obtained
  - per-arm QC ingredients (min visibility, bone lengths, in-frame flag) are stored so QC rules can be
    designed offline without re-running pose estimation
Writes data/pro_feature_raw_tracked.csv; never touches data/pro_feature_raw.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2

from pipeline.extract_pro_features import read_labels
from pipeline.pose_refiner import PoseSource, refine_frames
from pipeline.pseudo_label import load_event_frames
from pipeline.refiner_cache import CachedSource
from pipeline.pose_tracking import (
    HALF_WINDOW,
    MAX_GAP,
    FrameJoints,
    angle_series,
    apply_visibility,
    arm_by_feature,
    event_frames,
    extract_tracked_frames,
    resolve_path,
    temporal_event_value,
)
from pipeline.run_analysis import EVENT_NAMES, compute_features, video_frame_count

ELBOW_MIN_VISIBILITY = 0.3
TILT_MIN_VISIBILITY = 0.5
TILT_FEATURES = ("shoulder_tilt", "hip_tilt", "spine_tilt")
ARMS = ("trail", "lead")
QC_SUFFIXES = ("min_vis", "upper_2d", "fore_2d", "upper_3d", "fore_3d", "in_frame")

FIELDNAMES = [
    "video_id", "event_index", "event_name", "frame_index", "pose_detected", "refined",
    *[f"{arm}_elbow_{suffix}" for arm in ARMS for suffix in ("angle_2d", "angle_3d", "source")],
    *TILT_FEATURES,
    *[f"{arm}_{suffix}" for arm in ARMS for suffix in QC_SUFFIXES],
    "frame_w", "frame_h",
]


# ---------------------------------------------------------------- pure helpers
def arm_geometry(frame: FrameJoints | None, arm: Sequence[str]) -> dict[str, float | None]:
    """Upper-arm and forearm lengths in pixels (2D) and world units (3D)."""
    empty = {"upper_2d": None, "fore_2d": None, "upper_3d": None, "fore_3d": None}
    if frame is None or any(name not in frame for name in arm):
        return empty
    shoulder, elbow, wrist = (frame[name] for name in arm)
    return {
        "upper_2d": math.dist(shoulder[0:2], elbow[0:2]),
        "fore_2d": math.dist(elbow[0:2], wrist[0:2]),
        "upper_3d": math.dist(shoulder[3:6], elbow[3:6]),
        "fore_3d": math.dist(elbow[3:6], wrist[3:6]),
    }


def in_frame(point: Sequence[float] | None, frame_size: tuple[int, int]) -> bool:
    if point is None:
        return False
    width, height = frame_size
    return 0 <= point[0] < width and 0 <= point[1] < height


def _arm_qc(frame: FrameJoints | None, arm: Sequence[str], frame_size: tuple[int, int]) -> dict[str, Any]:
    present = frame is not None and all(name in frame for name in arm)
    return {
        "min_vis": min(frame[name][2] for name in arm) if present else None,
        **arm_geometry(frame, arm),
        "in_frame": all(in_frame(frame[name], frame_size) for name in arm) if present else False,
    }


def _tilts(frame: FrameJoints | None, handedness: str) -> dict[str, float | None]:
    if frame is None:
        return {name: None for name in TILT_FEATURES}
    landmarks = {name: (joint[0], joint[1], joint[2]) for name, joint in frame.items()}
    features = compute_features(apply_visibility(landmarks, TILT_MIN_VISIBILITY), handedness)
    return {name: features[name] for name in TILT_FEATURES}


def build_video_rows(
    video_id: int,
    frames: Sequence[FrameJoints | None],
    frame_indices: Sequence[int],
    frame_size: tuple[int, int],
    handedness: str,
    refined_frames: frozenset[int] = frozenset(),
) -> list[dict[str, Any]]:
    """One row per event. `refined_frames` are the frames whose arm joints came from the pose refiner."""
    arms = arm_by_feature(handedness)
    arm_for = {"trail": arms["trail_elbow_angle"], "lead": arms["lead_elbow_angle"]}
    series = {
        (arm, dims): angle_series(frames, joints, dims, ELBOW_MIN_VISIBILITY)
        for arm, joints in arm_for.items()
        for dims in ("2d", "3d")
    }
    rows = []
    for event_index, frame_index in enumerate(frame_indices):
        frame = frames[frame_index] if 0 <= frame_index < len(frames) else None
        row: dict[str, Any] = {
            "video_id": video_id,
            "event_index": event_index,
            "event_name": EVENT_NAMES[event_index],
            "frame_index": frame_index,
            "pose_detected": frame is not None,
            "refined": frame_index in refined_frames,
            "frame_w": frame_size[0],
            "frame_h": frame_size[1],
            **_tilts(frame, handedness),
        }
        for arm, joints in arm_for.items():
            angle_2d, source = temporal_event_value(series[(arm, "2d")], frame_index, HALF_WINDOW, MAX_GAP)
            angle_3d, _ = temporal_event_value(series[(arm, "3d")], frame_index, HALF_WINDOW, MAX_GAP)
            row[f"{arm}_elbow_angle_2d"] = angle_2d
            row[f"{arm}_elbow_angle_3d"] = angle_3d
            row[f"{arm}_elbow_source"] = source
            row.update({f"{arm}_{key}": value for key, value in _arm_qc(frame, joints, frame_size).items()})
        rows.append(row)
    return rows


# ------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default="learning_video/golfdb_720p_slice")
    parser.add_argument("--labels", default="event_detection/data/golfDB_labels.csv")
    parser.add_argument("--output", default="data/pro_feature_raw_tracked.csv")
    parser.add_argument("--handedness", choices=["right", "left"], default="right")
    parser.add_argument("--limit", type=int, help="Only process the first N videos (smoke test).")
    parser.add_argument("--events-table", help="Take event frames from this table (video_id, event_index, "
                        "frame_index) instead of --labels, e.g. the previous data/pro_feature_raw_tracked.csv.")
    parser.add_argument("--refiner-cache", help="Replace arm joints around the events with the answers in this "
                        "pipeline.refiner_cache file (needs --events-table).")
    return parser.parse_args()


def video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return size


def process_video_at(
    video_path: Path,
    video_id: int,
    indices: list[int],
    handedness: str,
    source: PoseSource | None = None,
) -> list[dict[str, Any]]:
    """Rows of one video at the given event frames; with a `source`, arm joints around the events are refined."""
    last = min(max(indices) + HALF_WINDOW + MAX_GAP, max(0, video_frame_count(video_path) - 1))
    frames = extract_tracked_frames(video_path, last)
    refined: frozenset[int] = frozenset()
    if source is not None:
        result = refine_frames(frames, 0, video_path, dict(enumerate(indices)), source)
        if result.status != "ok":
            raise RuntimeError(f"pose refiner {result.status}")
        frames, refined = result.frames, result.refined_frames
    return build_video_rows(video_id, frames, indices, video_size(video_path), handedness, refined)


def process_video(video_path: Path, video_id: int, raw_events: list[int], handedness: str) -> list[dict[str, Any]]:
    return process_video_at(video_path, video_id, event_frames(raw_events)[: len(EVENT_NAMES)], handedness)


def table_events(table: Path) -> dict[int, list[int]]:
    """{video id: the 8 event frames in order} from a previous extraction's table; incomplete videos are left out."""
    by_video = load_event_frames(table)
    return {
        int(video): [events[event] for event in range(len(EVENT_NAMES))]
        for video, events in by_video.items()
        if all(event in events for event in range(len(EVENT_NAMES)))
    }


def main() -> int:
    args = parse_args()
    videos_dir, labels_path = resolve_path(args.videos_dir), resolve_path(args.labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.refiner_cache and not args.events_table:
        raise SystemExit("--refiner-cache needs --events-table (the cache holds answers for those event frames)")
    source = CachedSource.load(Path(args.refiner_cache)) if args.refiner_cache else None
    if args.events_table:
        events = table_events(Path(args.events_table))
        video_ids = sorted(video for video in events if (videos_dir / f"{video}.mp4").exists())
        labels: dict[int, Any] = {}
    else:
        labels = read_labels(labels_path)
        video_ids = sorted(int(p.stem) for p in videos_dir.glob("*.mp4") if p.stem.isdigit() and int(p.stem) in labels)
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    failures: list[int] = []
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for done, video_id in enumerate(video_ids, start=1):
            try:
                path = videos_dir / f"{video_id}.mp4"
                if args.events_table:
                    rows = process_video_at(path, video_id, events[video_id], args.handedness, source)
                else:
                    rows = process_video(path, video_id, labels[video_id], args.handedness)
            except Exception as exc:  # keep going; failed videos are reported, not silently dropped
                print(f"video {video_id} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                failures.append(video_id)
                continue
            writer.writerows(rows)
            f.flush()
            if done % 25 == 0:
                print(f"processed {done}/{len(video_ids)} videos", flush=True)

    print(f"written: {output}  videos ok: {len(video_ids) - len(failures)}  failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
