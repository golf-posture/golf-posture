"""Event-specific left/right correction of the tracked arm joints (Top and Finish only).

At Top (arms overhead, torso turned) and Finish the pose model often reports the golfer's right and left arms
the wrong way round, so the trail and lead elbow angles are swapped. On hand-labelled frames the golfer's trail
shoulder was the higher one on screen in 45/45 Top frames and 43/47 Finish frames (right-handers), while the
model had it the other way round in 26/45 and 23/47. Forcing "the trail shoulder is the higher one" cut the
wrong arms at Top from 58/77 to 31/77 and at Finish from 47/77 to 33/77. The same rule made Mid-downswing worse,
so it is applied to these two events only.

Thresholds were calibrated on the same labels (in-sample): Top needs no minimum height gap, Finish ignores
shoulder pairs closer than 10% of their distance, where the model's own ordering is noise. The golfers in the
labels were all right-handed; left-handers are handled by mirroring the rule, which is not verified.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from pipeline.pose_tracking import HALF_WINDOW, FrameJoints, arm_joints

TOP = 3
FINISH = 7
# event index -> minimum shoulder height gap, as a fraction of the shoulder distance, before the sides are swapped
MIN_SHOULDER_GAP: Mapping[int, float] = {TOP: 0.0, FINISH: 0.1}
# Frames around the event frame that take the same decision: the tracker keeps one identity per arm, so a
# swap spans neighbouring frames; correcting only the event frame would look like a joint jump to the QC and
# to the neighbour-based recovery of hidden values (+-1 and +-HALF_WINDOW frames).
NEIGHBOUR_RADIUS = HALF_WINDOW + 1
_SIDES = ("RIGHT", "LEFT")
_OTHER = {"RIGHT": "LEFT", "LEFT": "RIGHT"}


def needs_side_swap(frame: FrameJoints | None, handedness: str, min_gap: float) -> bool:
    """True when the trail shoulder is lower on screen than the lead one by more than min_gap.

    Gap is measured in shoulder distances; an unmeasurable frame is left alone.
    """
    if frame is None:
        return False
    trail, lead = ("RIGHT", "LEFT") if handedness == "right" else ("LEFT", "RIGHT")
    trail_shoulder, lead_shoulder = frame.get(f"{trail}_SHOULDER"), frame.get(f"{lead}_SHOULDER")
    if trail_shoulder is None or lead_shoulder is None:
        return False
    distance = math.dist(trail_shoulder[:2], lead_shoulder[:2])
    if distance <= 0:
        return False
    return (trail_shoulder[1] - lead_shoulder[1]) / distance > min_gap


def swap_arm_sides(frame: FrameJoints) -> FrameJoints:
    """A copy of the frame with the right and left shoulder, elbow and wrist exchanged (image and world coordinates)."""
    swapped = dict(frame)
    for side in _SIDES:
        for name_here, name_there in zip(arm_joints(side), arm_joints(_OTHER[side])):
            if name_there in frame:
                swapped[name_here] = frame[name_there]
            else:
                swapped.pop(name_here, None)
    return swapped


def correct_arm_sides(
    frames: Sequence[FrameJoints | None],
    frame_indices: Sequence[int],
    handedness: str,
) -> tuple[list[FrameJoints | None], list[bool]]:
    """Corrected copy of the frames and, per event, whether its sides were swapped.

    The decision is made on the event frame and applied to the frames around it; other events are untouched.
    """
    to_swap: set[int] = set()
    flags = [False] * len(frame_indices)
    for event_index, min_gap in MIN_SHOULDER_GAP.items():
        if event_index >= len(frame_indices):
            continue
        centre = frame_indices[event_index]
        if not 0 <= centre < len(frames) or not needs_side_swap(frames[centre], handedness, min_gap):
            continue
        flags[event_index] = True
        to_swap.update(range(max(0, centre - NEIGHBOUR_RADIUS), min(len(frames), centre + NEIGHBOUR_RADIUS + 1)))
    corrected = [
        swap_arm_sides(frame) if index in to_swap and frame is not None else frame
        for index, frame in enumerate(frames)
    ]
    return corrected, flags
