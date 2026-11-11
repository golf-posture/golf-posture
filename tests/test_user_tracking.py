from __future__ import annotations

import pytest

from pipeline.side_correction import swap_arm_sides
from pipeline.user_tracking import build_user_events, tracking_window

FRAME_SIZE = (100, 100)
LANDMARKS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]


def _joint(px, py, vis=0.9, wx=0.0, wy=0.0, wz=0.0):
    return (px, py, vis, wx, wy, wz)


def _frame(vis: float = 0.9):
    """Right (trail) arm: 90 degree elbow in 2D and 3D. Left arm straight. Plausible bone lengths."""
    frame = {name: _joint(50, 50, vis) for name in LANDMARKS}
    frame.update({
        "RIGHT_SHOULDER": _joint(30, 20, vis, 0.1, 0.0, 0.0),
        "RIGHT_ELBOW": _joint(30, 40, vis, 0.1, 0.25, 0.0),
        "RIGHT_WRIST": _joint(50, 40, vis, 0.1 + 0.22, 0.25, 0.0),
        "LEFT_SHOULDER": _joint(70, 20, vis, -0.1, 0.0, 0.0),
        "LEFT_ELBOW": _joint(70, 40, vis, -0.1, 0.25, 0.0),
        "LEFT_WRIST": _joint(70, 60, vis, -0.1, 0.5, 0.0),
        "LEFT_HIP": _joint(60, 70, vis),
        "RIGHT_HIP": _joint(40, 70, vis),
    })
    return frame


def _events(frames, indices):
    return build_user_events(frames, indices, FRAME_SIZE, "right")


class TestTrackingWindow:
    def test_starts_before_first_event_and_ends_after_last(self):
        start, last = tracking_window([40, 60, 90], total_frames=500)

        assert start == 30
        assert last == 98

    def test_clamped_to_video_bounds(self):
        start, last = tracking_window([3, 60, 495], total_frames=500)

        assert start == 0
        assert last == 499


class TestBuildUserEvents:
    def test_returns_one_entry_per_event_with_features_and_landmarks(self):
        events = _events([_frame(), _frame()], [0, 1])

        assert len(events) == 2
        first = events[0]
        assert first["pose_detected"] is True
        assert first["features"]["trail_elbow_angle"] == pytest.approx(90.0)
        assert first["features"]["lead_elbow_angle"] == pytest.approx(180.0)
        assert set(first["landmarks"]) == set(LANDMARKS)
        assert first["computed_feature_count"] == 5

    def test_landmarks_below_half_visibility_are_dropped_but_elbow_uses_looser_threshold(self):
        event = _events([_frame(vis=0.45)], [0])[0]

        assert event["landmarks"]["RIGHT_ELBOW"] is None
        assert event["features"]["trail_elbow_angle"] == pytest.approx(90.0)
        assert event["features"]["shoulder_tilt"] is None

    def test_qc_flagged_elbow_becomes_unavailable(self):
        frame = _frame()
        frame["RIGHT_WRIST"] = _joint(50, 40, 0.9, 0.1 + 0.9, 0.25, 0.0)  # forearm 0.9m vs upper arm 0.25m

        event = _events([frame], [0])[0]

        assert event["features"]["trail_elbow_angle"] is None
        assert event["qc"]["trail"]["bone"] is True
        assert event["features"]["lead_elbow_angle"] == pytest.approx(180.0)

    def test_value_recovered_from_neighbors_is_discarded_by_qc_like_the_pro_reference(self):
        # The event frame itself has no pose, so the neighbour-derived angle is the least trustworthy kind:
        # QC (frame/low_vis flags) drops it, exactly as it did when the pro reference statistics were built.
        event = _events([_frame(), None, _frame()], [1])[0]

        assert event["pose_detected"] is False
        assert event["landmarks"] is None
        assert event["qc"]["trail_elbow_source"] == "window"
        assert event["qc"]["trail"]["frame"] is True
        assert event["features"]["trail_elbow_angle"] is None
        assert event["features"]["shoulder_tilt"] is None

    def test_smooth_arm_motion_is_not_flagged_as_a_jump(self):
        event = _events([_frame(), _frame(), _frame()], [1])[0]

        assert event["qc"]["trail"]["jump"] is False
        assert event["features"]["trail_elbow_angle"] == pytest.approx(90.0)

    def test_single_frame_glitch_in_one_arm_makes_that_elbow_unavailable(self):
        glitch = _frame()
        glitch["RIGHT_WRIST"] = _joint(90, 40, 0.9, 0.1 + 0.22, 0.25, 0.0)  # wrist leaps 40 px; torso is 50 px

        event = _events([_frame(), glitch, _frame()], [1])[0]

        assert event["qc"]["trail"]["jump"] is True
        assert event["features"]["trail_elbow_angle"] is None
        assert event["qc"]["lead"]["jump"] is False
        assert event["features"]["lead_elbow_angle"] == pytest.approx(180.0)

    def test_elbow_folded_below_the_floor_angle_becomes_unavailable(self):
        folded = _frame()
        # 2D and 3D agree on a ~16 degree elbow with plausible bone lengths, so only the floor rule can fire.
        folded["RIGHT_WRIST"] = _joint(35, 22, 0.9, 0.1 + 0.07, 0.25 - 0.22, 0.0)

        event = _events([folded], [0])[0]

        assert event["qc"]["trail"]["low_angle"] is True
        assert not any(flag for name, flag in event["qc"]["trail"].items() if name != "low_angle")
        assert event["features"]["trail_elbow_angle"] is None
        assert event["features"]["lead_elbow_angle"] == pytest.approx(180.0)

    def test_ordinary_elbow_angles_are_not_flagged_by_the_floor(self):
        event = _events([_frame()], [0])[0]

        assert event["qc"]["trail"]["low_angle"] is False
        assert event["qc"]["lead"]["low_angle"] is False

    def test_jump_cannot_be_measured_at_the_first_frame_so_nothing_is_flagged(self):
        event = _events([_frame(), _frame()], [0])[0]

        assert event["qc"]["trail"]["jump"] is False
        assert event["features"]["trail_elbow_angle"] == pytest.approx(90.0)

    def test_event_past_end_is_fully_unavailable(self):
        event = _events([_frame()], [9])[0]

        assert event["pose_detected"] is False
        assert all(value is None for value in event["features"].values())
        assert event["computed_feature_count"] == 0


class TestSideCorrection:
    EVENTS = [0, 6, 12, 18, 24, 30, 36, 42]
    TOP, MID_DOWNSWING = 3, 4

    @staticmethod
    def _ordered_frame():
        """Trail (right) shoulder higher on screen than the lead one; 90 degree trail elbow, straight lead arm."""
        frame = _frame()
        frame["RIGHT_SHOULDER"] = _joint(30, 15, 0.9, 0.1, 0.0, 0.0)
        frame["LEFT_SHOULDER"] = _joint(70, 25, 0.9, -0.1, 0.0, 0.0)
        return frame

    def _frames_with_swapped(self, event):
        frames = [self._ordered_frame() for _ in range(48)]
        for i in range(self.EVENTS[event] - 3, self.EVENTS[event] + 4):
            frames[i] = swap_arm_sides(self._ordered_frame())
        return frames

    def test_swapped_top_is_measured_on_the_right_arms(self):
        events = _events(self._frames_with_swapped(self.TOP), self.EVENTS)

        top = events[self.TOP]
        assert top["features"]["trail_elbow_angle"] == pytest.approx(90.0)
        assert top["features"]["lead_elbow_angle"] == pytest.approx(180.0)
        assert top["qc"]["side_corrected"] is True
        assert [event["qc"]["side_corrected"] for event in events].count(True) == 1

    def test_correction_does_not_look_like_a_joint_jump(self):
        top = _events(self._frames_with_swapped(self.TOP), self.EVENTS)[self.TOP]

        assert top["qc"]["trail"]["jump"] is False
        assert top["qc"]["lead"]["jump"] is False

    def test_landmarks_of_a_corrected_event_use_the_corrected_sides(self):
        top = _events(self._frames_with_swapped(self.TOP), self.EVENTS)[self.TOP]

        assert top["landmarks"]["RIGHT_SHOULDER"][:2] == (30, 15)
        assert top["landmarks"]["LEFT_SHOULDER"][:2] == (70, 25)

    def test_other_events_are_left_as_the_model_reported_them(self):
        event = _events(self._frames_with_swapped(self.MID_DOWNSWING), self.EVENTS)[self.MID_DOWNSWING]

        assert event["qc"]["side_corrected"] is False
        assert event["features"]["trail_elbow_angle"] == pytest.approx(180.0)

    def test_correct_frames_are_reported_as_not_corrected(self):
        events = _events([self._ordered_frame() for _ in range(48)], self.EVENTS)

        assert not any(event["qc"]["side_corrected"] for event in events)
