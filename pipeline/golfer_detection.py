"""Picking the golfer among YOLOv8-pose detections, and the YOLO joints as benchmark joints.

Standard library only. Shared by the benchmark (pose_benchmark.py) and the production pose source
(pose_source.py), so production code does not depend on the benchmark module.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Sequence

YOLO_JOINT_INDEX = {
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6, "LEFT_ELBOW": 7, "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9, "RIGHT_WRIST": 10, "LEFT_HIP": 11, "RIGHT_HIP": 12,
}
YOLO_TORSO = (5, 6, 11, 12)
YOLO_DEFAULT_WEIGHTS = Path("models/golfpose/yolov8x-pose.pt")
YOLO_BATCH = 8
YOLO_IMAGE_SIZE = 640
YOLO_MIN_CONFIDENCE = 0.25


def pick_golfer(boxes: Sequence[Sequence[float]], keypoints: Sequence[Sequence[Sequence[float]]]) -> int | None:
    """Index of the person most likely to be the golfer: tall box x how sure the model is of the torso points."""
    best, best_score = None, -1.0
    for i, (box, points) in enumerate(zip(boxes, keypoints)):
        torso = statistics.fmean(points[k][2] for k in YOLO_TORSO)
        score = (box[3] - box[1]) * torso
        if score > best_score:
            best, best_score = i, score
    return best


def yolo_joints(points: Sequence[Sequence[float]]) -> dict[str, tuple[float, ...]]:
    """COCO-17 keypoints (x, y, conf) as benchmark joints (px, py, visibility, 0, 0, 0)."""
    return {name: (float(points[i][0]), float(points[i][1]), float(points[i][2]), 0.0, 0.0, 0.0)
            for name, i in YOLO_JOINT_INDEX.items()}
