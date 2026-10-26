from __future__ import annotations

import pytest

from pipeline.pose_tracking import (
    angle_at,
    angle_series,
    apply_visibility,
    temporal_event_value,
)


def _lm(x: float, y: float, vis: float) -> tuple[float, float, float]:
    return (x, y, vis)


def _joint(px: float, py: float, vis: float, wx: float = 0.0, wy: float = 0.0, wz: float = 0.0):
    return (px, py, vis, wx, wy, wz)


class TestAngleAt:
    def test_right_angle_in_2d(self):
        assert angle_at([(1, 0), (0, 0), (0, 1)]) == pytest.approx(90.0)

    def test_straight_line_in_3d(self):
        assert angle_at([(1, 0, 0), (0, 0, 0), (-1, 0, 0)]) == pytest.approx(180.0)

    def test_none_when_any_point_missing(self):
        assert angle_at([(1, 0), None, (0, 1)]) is None

    def test_none_when_points_coincide(self):
        assert angle_at([(0, 0), (0, 0), (0, 1)]) is None


class TestAngleSeries:
    ARM = ("S", "E", "W")

    def _frame(self, vis: float):
        return {
            "S": _joint(1, 0, vis, 1, 0, 0),
            "E": _joint(0, 0, vis, 0, 0, 0),
            "W": _joint(0, 1, vis, 0, 0, 1),
        }

    def test_computes_angle_per_frame_in_2d_and_3d(self):
        frames = [self._frame(0.9)]

        assert angle_series(frames, self.ARM, "2d", 0.5) == [pytest.approx(90.0)]
        assert angle_series(frames, self.ARM, "3d", 0.5) == [pytest.approx(90.0)]

    def test_frame_invalid_when_any_joint_below_visibility(self):
        frame = self._frame(0.9)
        frame["W"] = _joint(0, 1, 0.2)

        assert angle_series([frame], self.ARM, "2d", 0.5) == [None]
        assert angle_series([frame], self.ARM, "2d", 0.1) == [pytest.approx(90.0)]

    def test_none_frame_yields_none(self):
        assert angle_series([None], self.ARM, "2d", 0.5) == [None]

    def test_rejects_unknown_dims(self):
        with pytest.raises(ValueError):
            angle_series([self._frame(0.9)], self.ARM, "4d", 0.5)


class TestTemporalEventValue:
    def test_direct_value_wins_over_neighbors(self):
        values = [10.0, 20.0, 30.0]

        assert temporal_event_value(values, 1, half_window=1, max_gap=3) == (20.0, "direct")

    def test_window_median_when_event_frame_invalid(self):
        values = [10.0, None, 30.0, 50.0]

        value, source = temporal_event_value(values, 1, half_window=2, max_gap=3)

        assert source == "window"
        assert value == pytest.approx(30.0)

    def test_interpolates_when_window_empty(self):
        values = [10.0, None, None, None, None, 50.0]

        value, source = temporal_event_value(values, 2, half_window=1, max_gap=3)

        assert source == "interp"
        assert value == pytest.approx(10.0 + (50.0 - 10.0) * 2 / 5)

    def test_none_when_gap_exceeds_max(self):
        values = [10.0, None, None, None, None, None, None, 50.0]

        assert temporal_event_value(values, 3, half_window=1, max_gap=2) == (None, "missing")

    def test_none_when_only_one_side_available(self):
        values = [None, None, None, 40.0]

        assert temporal_event_value(values, 1, half_window=0, max_gap=5) == (None, "missing")

    def test_none_when_index_out_of_range(self):
        assert temporal_event_value([1.0], 5, half_window=1, max_gap=2) == (None, "missing")


class TestApplyVisibility:
    def test_drops_joints_below_threshold(self):
        landmarks = {"A": _lm(1, 2, 0.4), "B": _lm(3, 4, 0.6)}

        result = apply_visibility(landmarks, 0.5)

        assert result["A"] is None
        assert result["B"] == _lm(3, 4, 0.6)

    def test_keeps_joint_exactly_at_threshold(self):
        result = apply_visibility({"A": _lm(1, 2, 0.5)}, 0.5)

        assert result["A"] is not None

    def test_does_not_mutate_input(self):
        landmarks = {"A": _lm(1, 2, 0.1)}

        apply_visibility(landmarks, 0.5)

        assert landmarks["A"] == _lm(1, 2, 0.1)

    def test_returns_none_when_no_pose(self):
        assert apply_visibility(None, 0.5) is None
