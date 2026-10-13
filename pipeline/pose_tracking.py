"""Video-mode MediaPipe pose tracking with temporal recovery, shared by pro-data extraction and user analysis.

Why: isolated per-frame inference (static_image_mode=True) loses partly hidden arms; tracking the swing
sequentially (static_image_mode=False) keeps them. Elbow angles are computed per frame in 2D (pixels) and
3D (world landmarks); an event value is the event frame itself, else the median of nearby valid frames,
else a bounded linear interpolation (see temporal_event_value).
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pipeline.run_analysis import LANDMARK_NAMES, repo_root

# 2D landmarks as (x, y, visibility), as produced by the static backends
Landmarks = dict[str, tuple[float, float, float] | None]
# px, py, visibility, world x, world y, world z
FrameJoints = dict[str, tuple[float, float, float, float, float, float]]
Series = list[float | None]
EventValue = tuple[float | None, str]

JOINT_NAMES = tuple(LANDMARK_NAMES)
HALF_WINDOW = 2
MAX_GAP = 6


# ----------------------------------------------------------------------- utils
def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.exists() else repo_root() / value


def event_frames(raw_events: list[int]) -> list[int]:
    """GolfDB raw event frames -> frame indices inside the sliced clip (clip starts at the first label)."""
    start = raw_events[0]
    return [frame - start for frame in raw_events[1:-1]]


def apply_visibility(landmarks: Landmarks | None, min_visibility: float) -> Landmarks | None:
    if landmarks is None:
        return None
    return {
        name: (point if point is not None and point[2] >= min_visibility else None)
        for name, point in landmarks.items()
    }


# ---------------------------------------------------------------- pure helpers
def angle_at(points: Sequence[Sequence[float] | None]) -> float | None:
    """Angle in degrees at the middle point of three coordinates (any dimension)."""
    if any(p is None for p in points):
        return None
    a, b, c = (np.asarray(p, dtype=float) for p in points)
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom <= 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


def angle_series(
    frames: Sequence[FrameJoints | None], arm: Sequence[str], dims: str, min_visibility: float
) -> Series:
    if dims not in ("2d", "3d"):
        raise ValueError(f"dims must be '2d' or '3d', got {dims!r}")
    span = slice(0, 2) if dims == "2d" else slice(3, 6)
    series: Series = []
    for frame in frames:
        joints = [frame.get(name) for name in arm] if frame is not None else [None]
        if any(j is None or j[2] < min_visibility for j in joints):
            series.append(None)
        else:
            series.append(angle_at([j[span] for j in joints]))
    return series


def temporal_event_value(values: Series, idx: int, half_window: int, max_gap: int) -> EventValue:
    """Value at idx: direct, else median of +-half_window, else bounded linear interpolation."""
    if idx < 0 or idx >= len(values):
        return None, "missing"
    if values[idx] is not None:
        return values[idx], "direct"
    near = [v for v in values[max(0, idx - half_window): idx + half_window + 1] if v is not None]
    if near:
        return float(statistics.median(near)), "window"
    before = next((i for i in range(idx - 1, max(-1, idx - max_gap - 1), -1) if values[i] is not None), None)
    after = next((i for i in range(idx + 1, min(len(values), idx + max_gap + 1)) if values[i] is not None), None)
    if before is None or after is None:
        return None, "missing"
    fraction = (idx - before) / (after - before)
    return float(values[before] + (values[after] - values[before]) * fraction), "interp"


def arm_joints(side: str) -> tuple[str, str, str]:
    return (f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST")


def arm_by_feature(handedness: str) -> dict[str, tuple[str, str, str]]:
    trail, lead = ("RIGHT", "LEFT") if handedness == "right" else ("LEFT", "RIGHT")
    return {"trail_elbow_angle": arm_joints(trail), "lead_elbow_angle": arm_joints(lead)}


# ------------------------------------------------------------------ extraction
def frame_joints(result: Any, width: int, height: int) -> FrameJoints | None:
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None
    joints: FrameJoints = {}
    for name in JOINT_NAMES:
        idx = LANDMARK_NAMES[name]
        lm, wl = result.pose_landmarks.landmark[idx], result.pose_world_landmarks.landmark[idx]
        joints[name] = (lm.x * width, lm.y * height, lm.visibility, wl.x, wl.y, wl.z)
    return joints


def extract_tracked_frames(
    video_path: Path, last_frame: int, complexity: int = 1, start_frame: int = 0
) -> list[FrameJoints | None]:
    """Track frames start_frame..last_frame sequentially; list index i is video frame start_frame + i."""
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(video_path))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[FrameJoints | None] = []
    try:
        for _ in range(last_frame - start_frame + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            height, width = bgr.shape[:2]
            frames.append(frame_joints(pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), width, height))
    finally:
        pose.close()
        cap.release()
    return frames
