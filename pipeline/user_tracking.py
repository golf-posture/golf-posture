"""User-video feature extraction with video-mode tracking (the 'tracked' extraction of run_analysis).

Mirrors how data/tracked/ reference statistics were built from the pro videos:
  - elbow angles: 2D angle, visibility >= 0.3, recovered from neighbouring frames, then the same row-level
    QC as the pro data (low_vis / bone / frame / inconsistent). A flagged elbow is reported as unavailable
    rather than graded, so a wrong pose does not turn into a wrong 'Needs Improvement'.
  - tilts: event frame, visibility >= 0.5.
  - landmarks (for skeleton comparison images): event frame, visibility >= 0.5, like the static backend.
Population-based outlier flags are not applied to a single user; z-score grading covers extreme values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from pipeline.extract_pro_features_tracked import build_video_rows, video_size
from pipeline.pose_tracking import (
    HALF_WINDOW,
    MAX_GAP,
    FrameJoints,
    apply_visibility,
    extract_tracked_frames,
)
from pipeline.qc_tracked_features import ARMS, arm_flags
from pipeline.run_analysis import REQUIRED_FEATURES, video_frame_count

TRACK_WARMUP_FRAMES = 10
LANDMARK_MIN_VISIBILITY = 0.5
ELBOW_COLUMN = {"trail_elbow_angle": "trail", "lead_elbow_angle": "lead"}


def tracking_window(event_frames: Sequence[int], total_frames: int) -> tuple[int, int]:
    """First and last video frame to track: a short warm-up before the first event, a margin after the last."""
    start = max(0, min(event_frames) - TRACK_WARMUP_FRAMES)
    last = min(total_frames - 1, max(event_frames) + HALF_WINDOW + MAX_GAP)
    return start, last


def _landmarks(frame: FrameJoints | None) -> dict[str, tuple[float, float, float] | None] | None:
    if frame is None:
        return None
    points = {name: (joint[0], joint[1], joint[2]) for name, joint in frame.items()}
    return apply_visibility(points, LANDMARK_MIN_VISIBILITY)


def build_user_events(
    frames: Sequence[FrameJoints | None],
    frame_indices: Sequence[int],
    frame_size: tuple[int, int],
    handedness: str,
) -> list[dict[str, Any]]:
    """Per-event features/landmarks/QC; frame_indices index into `frames`."""
    rows = build_video_rows(0, frames, frame_indices, frame_size, handedness)
    events = []
    for row, frame_index in zip(rows, frame_indices):
        flags = {arm: arm_flags(row, arm) for arm in ARMS}
        features: dict[str, float | None] = {
            "shoulder_tilt": row["shoulder_tilt"],
            "hip_tilt": row["hip_tilt"],
            "spine_tilt": row["spine_tilt"],
        }
        for feature, arm in ELBOW_COLUMN.items():
            features[feature] = None if any(flags[arm].values()) else row[f"{arm}_elbow_angle_2d"]
        features = {name: features[name] for name in REQUIRED_FEATURES}
        frame = frames[frame_index] if 0 <= frame_index < len(frames) else None
        events.append({
            "pose_detected": bool(row["pose_detected"]),
            "features": features,
            "landmarks": _landmarks(frame),
            "computed_feature_count": sum(1 for value in features.values() if value is not None),
            "qc": {
                "trail": flags["trail"],
                "lead": flags["lead"],
                "trail_elbow_source": row["trail_elbow_source"],
                "lead_elbow_source": row["lead_elbow_source"],
            },
        })
    return events


def analyze_user_video(video_path: Path, event_frames: Sequence[int], handedness: str) -> list[dict[str, Any]]:
    start, last = tracking_window(event_frames, video_frame_count(video_path))
    frames = extract_tracked_frames(video_path, last, start_frame=start)
    indices = [frame - start for frame in event_frames]
    return build_user_events(frames, indices, video_size(video_path), handedness)
