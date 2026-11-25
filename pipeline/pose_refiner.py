"""Replace the tracked MediaPipe arm joints around each swing event with ViTPose-H / YOLOv8x joints.

MediaPipe tracking stays the backbone of the analysis (3D world coordinates, the other landmarks, the QC and
temporal recovery all read it). Around every enabled event this module overlays only the 2D position and score of
the joints the better models return, so the rest of the pipeline sees the same `FrameJoints` shape as before.

Model per event (measured on 98 hand-labelled videos, see docs/pose_benchmark.md): ViTPose-H everywhere,
YOLOv8x at Finish. A model failure never fails the analysis: the original frames are returned with a status.

Standard library only; the models themselves live behind `PoseSource`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from pipeline.pseudo_label import DEFAULT_RADIUS, FINISH_EVENT, sample_frames

MODEL_VITPOSE = "vitpose"
MODEL_YOLOX = "yolox"
EVENT_NAMES = (
    "Address", "Toe-up", "Mid-backswing", "Top", "Mid-downswing", "Impact", "Mid-follow-through", "Finish",
)
ALL_EVENTS = frozenset(range(len(EVENT_NAMES)))
# Only the joints the benchmark labels cover (shoulder, elbow, wrist) are replaced. The models also return the
# hips, but no hip was ever validated against hand labels, so the hips stay MediaPipe's until they are.
ARM_JOINTS = frozenset(f"{side}_{part}" for side in ("LEFT", "RIGHT") for part in ("SHOULDER", "ELBOW", "WRIST"))

# px, py, visibility, world x, world y, world z (one MediaPipe landmark)
Joint = tuple[float, ...]
FrameJoints = dict[str, Joint]
# what a model returns per joint: (px, py, score)
RefinedJoints = Mapping[str, Sequence[float]]


class PoseSource(Protocol):
    """Runs the pose models on the requested video frames: {frame: model name} -> {frame: joints | None}."""

    def joints(self, video_path: Path, requests: Mapping[int, str]) -> Mapping[int, RefinedJoints | None]: ...


@dataclass(frozen=True)
class RefineResult:
    frames: list[FrameJoints | None]
    status: str  # "ok", "skipped" (nothing to ask) or "failed: <reason>"
    refined: int  # frames that actually got new joints
    refined_frames: frozenset[int] = frozenset()  # their video frame numbers


def parse_events(text: str) -> frozenset[int]:
    """Event set from "all", numbers ("3,7") or names ("Top,Finish"), mixed freely. Anything else is an error."""
    if text.strip().lower() == "all":
        return ALL_EVENTS
    by_name = {name.lower(): index for index, name in enumerate(EVENT_NAMES)}
    events = set()
    for part in text.split(","):
        token = part.strip().lower()
        if token in by_name:
            events.add(by_name[token])
        elif token.isdigit() and int(token) in ALL_EVENTS:
            events.add(int(token))
        else:
            raise ValueError(f"unknown event {part.strip()!r}; use 'all', 0-7 or names like Top, Finish")
    return frozenset(events)


def refiner_model(event_index: int) -> str:
    """Which model measures this event."""
    return MODEL_YOLOX if event_index == FINISH_EVENT else MODEL_VITPOSE


def request_frames(
    event_frames: Mapping[int, int],
    window: tuple[int, int],
    radius: int,
    enabled_events: frozenset[int],
) -> dict[int, str]:
    """{video frame: model} for `radius` frames around every enabled event, inside the tracked window.

    `window` is the first and last video frame that was tracked (inclusive).
    """
    first, last = window
    chosen = {event: frame for event, frame in event_frames.items() if event in enabled_events}
    if not chosen:
        return {}
    around = sample_frames(chosen, radius, frame_count=last + 1)
    return {frame: refiner_model(event) for frame, event in around.items() if first <= frame <= last}


def _overlay(
    frames: Sequence[FrameJoints | None],
    refined: Mapping[int, RefinedJoints | None],
    start_frame: int,
    joints: frozenset[str],
) -> tuple[list[FrameJoints | None], frozenset[int]]:
    """The frames with refined joints, and the video frame numbers that actually changed."""
    result: list[FrameJoints | None] = []
    applied: set[int] = set()
    for index, frame in enumerate(frames):
        answer = refined.get(start_frame + index)
        if frame is None or not answer:
            result.append(frame)
            continue
        merged = dict(frame)
        for name, value in answer.items():
            if name in joints and name in merged:
                x, y, score = value[0], value[1], value[2]
                merged[name] = (float(x), float(y), float(score), *merged[name][3:])
        result.append(merged)
        if merged != frame:
            applied.add(start_frame + index)
    return result, frozenset(applied)


def overlay_joints(
    frames: Sequence[FrameJoints | None],
    refined: Mapping[int, RefinedJoints | None],
    start_frame: int,
    joints: frozenset[str] = ARM_JOINTS,
) -> list[FrameJoints | None]:
    """New frames where refined joints replace position and score; 3D coordinates and other landmarks stay.

    List index i is video frame `start_frame + i`. Only the joints in `joints` (the arms by default) are replaced.
    A frame MediaPipe could not track stays empty (there is no 3D coordinate to keep, and none is invented), and a
    joint missing from a frame is not added.
    """
    return _overlay(frames, refined, start_frame, joints)[0]


def refine_frames(
    frames: Sequence[FrameJoints | None],
    start_frame: int,
    video_path: Path,
    event_frames: Mapping[int, int],
    source: PoseSource,
    radius: int = DEFAULT_RADIUS,
    enabled_events: frozenset[int] = ALL_EVENTS,
    joints: frozenset[str] = ARM_JOINTS,
    *,
    check_cancelled=None,
) -> RefineResult:
    """Overlay the models' joints around the enabled events; on any source error keep the original frames."""
    original = list(frames)
    window = (start_frame, start_frame + len(original) - 1)
    requests = request_frames(event_frames, window, radius, enabled_events)
    if not requests:
        return RefineResult(original, "skipped", 0)
    # Allow the mobile job to stop between model batches, outside the fallback handler.
    items = sorted(requests.items())
    batch_size = 5 if check_cancelled is not None else len(items)
    answers = {}
    for start in range(0, len(items), batch_size):
        if check_cancelled is not None:
            check_cancelled()
        try:
            answers.update(source.joints(video_path, dict(items[start:start + batch_size])))
        except Exception as error:
            return RefineResult(original, f"failed: {type(error).__name__}: {error}", 0)
    if check_cancelled is not None:
        check_cancelled()
    merged, applied = _overlay(original, answers, start_frame, joints)
    return RefineResult(merged, "ok", len(applied), applied)
