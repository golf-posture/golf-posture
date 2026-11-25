"""GolfPose-2D (person) keypoints as benchmark joints, and the prediction-cache layout.

Standard library only: the GolfPose model runs in its own environment (see golfpose_infer.py) and
this module is imported from both sides.

The checkpoint's 17 keypoints are in Human3.6M order, not COCO order (its dataset_meta
keypoint_id2name: 0 root, 1-3 right hip/knee/foot, 4-6 left hip/knee/foot, 7-10 spine/thorax/neck/head,
11-13 left shoulder/elbow/wrist, 14-16 right shoulder/elbow/wrist).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

GOLFPOSE_JOINT_INDEX = {
    "LEFT_SHOULDER": 11, "LEFT_ELBOW": 12, "LEFT_WRIST": 13,
    "RIGHT_SHOULDER": 14, "RIGHT_ELBOW": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 4, "RIGHT_HIP": 1,
}


def golfpose_joints(points: Sequence[Sequence[float]], scores: Sequence[float]) -> dict[str, tuple[float, ...]]:
    """Benchmark joints (px, py, visibility, 0, 0, 0); the heatmap score stands in for visibility."""
    return {name: (float(points[i][0]), float(points[i][1]), float(scores[i]), 0.0, 0.0, 0.0)
            for name, i in GOLFPOSE_JOINT_INDEX.items()}


def predictions_payload(
    backend: str, frames: Mapping[str, Mapping[int, Mapping[str, Sequence[float]] | None]]
) -> dict[str, Any]:
    """The JSON that pose_benchmark's CachedBackend replays: {video file: {frame: {joint: [...]} | None}}."""
    return {
        "backend": backend,
        "frames": {
            video: {str(frame): None if joints is None else {name: list(value) for name, value in joints.items()}
                    for frame, joints in per_frame.items()}
            for video, per_frame in frames.items()
        },
    }
