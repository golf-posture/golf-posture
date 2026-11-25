"""Frame sampling and keep-rules for teacher (ViTPose-H) pseudo-labels. Standard library only.

Pseudo-labels are built on unlabelled pro videos; the hand-labelled videos are never used for training.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

FINISH_EVENT = 7
DEFAULT_RADIUS = 2
# Measured on the labelled videos: at Finish the teacher is wrong 32% overall but 6.5% when it
# differs from the second model by at most this fraction of its own arm length.
FINISH_AGREEMENT = 0.15


def sample_frames(event_frames: Mapping[int, int], radius: int, frame_count: int) -> dict[int, int]:
    """{frame: event_index} for `radius` frames either side of every event, inside the video.

    A frame inside two windows belongs to the nearest event (the earlier one on a tie).
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    chosen: dict[int, tuple[int, int]] = {}
    for event, centre in sorted(event_frames.items()):
        for frame in range(centre - radius, centre + radius + 1):
            if not 0 <= frame < frame_count:
                continue
            distance = abs(frame - centre)
            if frame not in chosen or distance < chosen[frame][0]:
                chosen[frame] = (distance, event)
    return {frame: event for frame, (_, event) in sorted(chosen.items())}


def load_event_frames(table: Path, exclude: set[str] | frozenset[str] = frozenset()) -> dict[str, dict[int, int]]:
    """{video id: {event_index: frame_index}} from the pro feature table, minus excluded videos."""
    events: dict[str, dict[int, int]] = defaultdict(dict)
    with Path(table).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video = str(row["video_id"])
            if video not in exclude:
                events[video][int(row["event_index"])] = int(row["frame_index"])
    return dict(events)


# Student keypoints: both arms and both hips. FLIP_INDEX swaps left and right when an image is mirrored.
STUDENT_JOINTS = (
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
)
FLIP_INDEX = (3, 4, 5, 0, 1, 2, 7, 6)
ARM_ROLES = ("SHOULDER", "ELBOW", "WRIST")
SIDES = ("LEFT", "RIGHT")


def _arm(joints: Mapping | None, side: str):
    if not joints:
        return None
    arm = [joints.get(f"{side}_{role}") for role in ARM_ROLES]
    return None if any(j is None for j in arm) else arm


def arm_disagreement(teacher: Mapping | None, second: Mapping | None, side: str) -> float | None:
    """Worst joint distance between two models for one arm, as a fraction of the teacher's arm length."""
    a, b = _arm(teacher, side), _arm(second, side)
    if a is None or b is None:
        return None
    length = math.dist(a[0][:2], a[1][:2]) + math.dist(a[1][:2], a[2][:2])
    if length <= 0:
        return None
    return max(math.dist(p[:2], q[:2]) for p, q in zip(a, b)) / length


def yolo_pose_line(
    box: Sequence[float], joints: Mapping, width: int, height: int, hidden_sides: Sequence[str] = ()
) -> str:
    """One YOLO-pose label line (class 0, box, STUDENT_JOINTS) in 0..1 units.

    Joints of `hidden_sides` get visibility 0, so training ignores them.
    """
    x1, y1, x2, y2 = box
    parts = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
    out = ["0"] + [f"{v:.6f}" for v in parts]
    for name in STUDENT_JOINTS:
        joint = joints.get(name)
        if joint is None or name.split("_")[0] in hidden_sides:
            out += ["0.000000", "0.000000", "0"]
        else:
            out += [f"{min(max(joint[0] / width, 0.0), 1.0):.6f}", f"{min(max(joint[1] / height, 0.0), 1.0):.6f}", "2"]
    return " ".join(out)


def keep_label(event_index: int, disagreement: float | None, threshold: float = FINISH_AGREEMENT) -> bool:
    """Whether a teacher label is trusted. Only Finish needs a second model to agree."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if event_index != FINISH_EVENT:
        return True
    return disagreement is not None and disagreement <= threshold
