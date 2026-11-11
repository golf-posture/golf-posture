from __future__ import annotations

import pytest

from pipeline.qc_temporal import (
    JUMP_FLOOR,
    JUMP_K,
    JUMP_MEDIAN_BY_EVENT,
    JUMP_THRESHOLD,
    jump_threshold,
)
from pipeline.user_tracking import build_user_events

FAST_EVENTS = (4, 5, 6)  # Mid-downswing, Impact, Mid-follow-through
SLOW_EVENTS = (0, 1, 2, 3, 7)


class TestJumpThreshold:
    def test_one_median_per_event(self):
        assert len(JUMP_MEDIAN_BY_EVENT) == 8

    @pytest.mark.parametrize("event", range(8))
    def test_threshold_is_k_times_the_median_but_never_below_the_floor(self, event):
        expected = max(JUMP_FLOOR, JUMP_K * JUMP_MEDIAN_BY_EVENT[event])

        assert jump_threshold(event) == pytest.approx(expected)

    def test_fast_events_tolerate_much_larger_jumps_than_slow_ones(self):
        slowest_fast = min(jump_threshold(e) for e in FAST_EVENTS)
        fastest_slow = max(jump_threshold(e) for e in SLOW_EVENTS)

        assert slowest_fast > 2 * fastest_slow

    def test_address_uses_the_floor_because_its_median_is_almost_zero(self):
        assert jump_threshold(0) == pytest.approx(JUMP_FLOOR)

    @pytest.mark.parametrize("event", [None, -1, 8, 99])
    def test_unknown_event_falls_back_to_the_single_threshold(self, event):
        assert jump_threshold(event) == JUMP_THRESHOLD


LANDMARKS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]


def _joint(px, py, vis=0.9, wx=0.0, wy=0.0, wz=0.0):
    return (px, py, vis, wx, wy, wz)


def _frame():
    """Torso 50 px (shoulders y=20, hips y=70); right arm bent, left arm straight; shoulders level."""
    frame = {name: _joint(50, 50) for name in LANDMARKS}
    frame.update({
        "RIGHT_SHOULDER": _joint(30, 20, 0.9, 0.1, 0.0, 0.0),
        "RIGHT_ELBOW": _joint(30, 40, 0.9, 0.1, 0.25, 0.0),
        "RIGHT_WRIST": _joint(50, 40, 0.9, 0.32, 0.25, 0.0),
        "LEFT_SHOULDER": _joint(70, 20, 0.9, -0.1, 0.0, 0.0),
        "LEFT_ELBOW": _joint(70, 40, 0.9, -0.1, 0.25, 0.0),
        "LEFT_WRIST": _joint(70, 60, 0.9, -0.1, 0.5, 0.0),
        "LEFT_HIP": _joint(60, 70),
        "RIGHT_HIP": _joint(40, 70),
    })
    return frame


class TestEventSpecificJumpFlag:
    EVENTS = [1, 4, 7, 10, 13, 16, 19, 22]

    def _events_with_wrist_glitch_at(self, event):
        frames = [_frame() for _ in range(24)]
        glitch = _frame()
        glitch["RIGHT_WRIST"] = _joint(75, 40, 0.9, 0.32, 0.25, 0.0)  # leaps 25 px = 0.5 torso lengths
        frames[self.EVENTS[event]] = glitch
        return build_user_events(frames, self.EVENTS, (100, 100), "right")

    def test_a_half_torso_jump_is_flagged_at_a_slow_event(self):
        event = self._events_with_wrist_glitch_at(2)[2]

        assert event["qc"]["trail"]["jump"] is True
        assert event["features"]["trail_elbow_angle"] is None

    def test_the_same_jump_is_tolerated_at_a_fast_event(self):
        event = self._events_with_wrist_glitch_at(4)[4]

        assert event["qc"]["trail"]["jump"] is False

    def test_a_huge_jump_is_still_flagged_at_a_fast_event(self):
        frames = [_frame() for _ in range(24)]
        glitch = _frame()
        glitch["RIGHT_WRIST"] = _joint(200, 40, 0.9, 0.32, 0.25, 0.0)  # 150 px = 3 torso lengths
        frames[self.EVENTS[4]] = glitch

        event = build_user_events(frames, self.EVENTS, (300, 100), "right")[4]

        assert event["qc"]["trail"]["jump"] is True
