"""Select event frames from fixed model scores without changing model weights."""
from __future__ import annotations

import numpy as np


def ordered_event_frames(scores: np.ndarray) -> list[int]:
    """Maximize summed log scores subject to strictly increasing event frames."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Expected a finite, nonnegative frames-by-events score matrix")
    frame_count, event_count = scores.shape
    if event_count == 0:
        return []
    if frame_count < event_count:
        raise ValueError("Too few frames for distinct ordered events")
    independent = scores.argmax(axis=0)
    # Preserve existing valid choices, including their original tie handling.
    if np.all(np.diff(independent) > 0):
        return independent.tolist()
    log_scores = np.log(np.maximum(scores, np.finfo(float).tiny))
    previous = log_scores[:, 0].copy()
    parents = np.full((event_count, frame_count), -1, dtype=int)
    for event in range(1, event_count):
        current = np.full(frame_count, -np.inf)
        best_score, best_frame = -np.inf, -1
        for frame in range(event, frame_count):
            candidate = frame - 1
            if previous[candidate] > best_score:
                best_score, best_frame = previous[candidate], candidate
            current[frame] = best_score + log_scores[frame, event]
            parents[event, frame] = best_frame
        previous = current
    frames = [int(previous.argmax())]
    for event in range(event_count - 1, 0, -1):
        frames.append(int(parents[event, frames[-1]]))
    return frames[::-1]


def select_event_frames(probs: np.ndarray, mode: str = "independent", missing_events=()):
    """Return eight event frames/scores; explicitly supplied missing events remain null.

    Missing events are caller-provided information, not automatically detected.
    The ninth model class is background and is not selected as an event.
    """
    if mode not in {"independent", "ordered"}:
        raise ValueError(f"Unknown event selection mode: {mode}")
    if probs.ndim != 2 or probs.shape[1] != 9 or len(probs) == 0:
        raise ValueError("Expected nonempty SwingNet probabilities with nine classes")
    missing = set(missing_events)
    if not missing.issubset(range(8)):
        raise ValueError("Missing event indices must be between 0 and 7")
    active = [i for i in range(8) if i not in missing]
    scores = probs[:, active]
    selected = (ordered_event_frames(scores) if mode == "ordered"
                else scores.argmax(axis=0).tolist()) if active else []
    frames, confidence = [None] * 8, [None] * 8
    for event, frame in zip(active, selected):
        frames[event] = int(frame)
        confidence[event] = float(probs[frame, event])
    return frames, confidence


def gate_event_frames(frames, confidence, minimum_confidence=0.0):
    """Withhold declared missing or low-score events; scores are not calibrated accuracy."""
    if not np.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
        raise ValueError("Minimum event confidence must be between 0 and 1")
    usable, reasons = list(frames), [None] * len(frames)
    for event, (frame, score) in enumerate(zip(frames, confidence)):
        if frame is None:
            reasons[event] = "declared_missing"
        elif score is not None and (not np.isfinite(score) or score < minimum_confidence):
            usable[event] = None
            reasons[event] = "low_event_score"
    return usable, reasons
