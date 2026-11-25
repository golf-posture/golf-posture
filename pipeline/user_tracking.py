"""User-video feature extraction with video-mode tracking (the 'tracked' extraction of run_analysis).

Mirrors how data/tracked/ reference statistics were built from the pro videos:
  - elbow angles: 2D angle, visibility >= 0.3, recovered from neighbouring frames, then the same row-level
    QC as the pro data (low_vis / bone / frame / inconsistent) plus a joint-jump check against the neighbouring
    frames, with a threshold that depends on the event (qc_temporal.jump_threshold; not applied to the pro
    reference, which has no per-frame series). A flagged elbow is
    reported as unavailable rather than graded, so a wrong pose does not turn into a wrong 'Needs Improvement'.
  - arm sides: as the model reports them. The old Top/Finish rule that swapped the arms when the trail shoulder
    was not the higher one on screen (side_correction) is no longer applied.
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
    arm_by_feature,
    extract_tracked_frames,
)
from pipeline.pose_refiner import ALL_EVENTS, PoseSource, refine_frames
from pipeline.qc_temporal import ELBOW_ANGLE_FLOOR_DEG, arm_jump, elbow_angle_floor, jump_threshold
from pipeline.qc_tracked_features import ARMS, arm_flags, refined_arm_flags
from pipeline.run_analysis import REQUIRED_FEATURES, video_frame_count

EXPECTED_EVENTS = 8  # a refiner needs the full event list: the position in it is the event index
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


def _jump_flag(
    frames: Sequence[FrameJoints | None], frame_index: int, joints: Sequence[str], event_index: int
) -> bool:
    """True when an arm joint leaps away from its neighbouring frames by more than the event's threshold.

    Unmeasurable counts as not flagged.
    """
    jump = arm_jump(frames, frame_index, joints)
    return jump is not None and jump > jump_threshold(event_index)


def build_user_events(
    frames: Sequence[FrameJoints | None],
    frame_indices: Sequence[int],
    frame_size: tuple[int, int],
    handedness: str,
    refined_indices: frozenset[int] = frozenset(),
) -> list[dict[str, Any]]:
    """Per-event features/landmarks/QC; frame_indices index into `frames`.

    An event whose frame is in `refined_indices` had its arm joints replaced by the pose refiner, so its elbow
    flags come from refined_arm_flags (no MediaPipe-3D rules, lower visibility floor).
    """
    rows = build_video_rows(0, frames, frame_indices, frame_size, handedness)
    arms = arm_by_feature(handedness)
    arm_joints = {"trail": arms["trail_elbow_angle"], "lead": arms["lead_elbow_angle"]}
    events = []
    for event_index, (row, frame_index) in enumerate(zip(rows, frame_indices)):
        row_flags = refined_arm_flags if frame_index in refined_indices else arm_flags
        flags = {
            arm: {
                **row_flags(row, arm),
                "jump": _jump_flag(frames, frame_index, arm_joints[arm], event_index),
                "low_angle": elbow_angle_floor(row[f"{arm}_elbow_angle_2d"], ELBOW_ANGLE_FLOOR_DEG),
            }
            for arm in ARMS
        }
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


def _with_refiner_status(events: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
    return [{**event, "qc": {**event["qc"], "pose_refiner": status}} for event in events]


def analyze_user_video(
    video_path: Path,
    event_frames: Sequence[int],
    handedness: str,
    refiner: PoseSource | None = None,
    refine_events: frozenset[int] = ALL_EVENTS,
) -> list[dict[str, Any]]:
    """Tracked per-event features and QC; with a `refiner`, arm joints around the events come from better models.

    `event_frames` is the 8 events in order, so a position in the list is the event index. With anything else
    the refiner is skipped (the event numbers would be wrong, and Finish uses a different model), and the
    reason is recorded in qc["pose_refiner"], which is only present when a refiner was given.
    """
    start, last = tracking_window(event_frames, video_frame_count(video_path))
    frames = extract_tracked_frames(video_path, last, start_frame=start)
    status = None
    refined_indices: frozenset[int] = frozenset()
    if refiner is not None:
        if len(event_frames) == EXPECTED_EVENTS:
            refined = refine_frames(frames, start, video_path, dict(enumerate(event_frames)), refiner,
                                    enabled_events=refine_events)
            frames, status = refined.frames, refined.status
            refined_indices = frozenset(frame - start for frame in refined.refined_frames)
        else:
            status = f"skipped: expected {EXPECTED_EVENTS} events, got {len(event_frames)}"
    indices = [frame - start for frame in event_frames]
    events = build_user_events(frames, indices, video_size(video_path), handedness, refined_indices)
    return events if status is None else _with_refiner_status(events, status)
