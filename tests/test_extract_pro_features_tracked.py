from __future__ import annotations

import pytest

from pipeline.extract_pro_features_tracked import (
    FIELDNAMES,
    arm_geometry,
    build_video_rows,
    in_frame,
)

FRAME_SIZE = (100, 100)


def _joint(px, py, vis=0.9, wx=0.0, wy=0.0, wz=0.0):
    return (px, py, vis, wx, wy, wz)


def _frame(vis: float = 0.9):
    """Right arm forms a 90 degree elbow in both 2D and 3D; left arm is straight."""
    return {
        "RIGHT_SHOULDER": _joint(30, 20, vis, 1, 0, 0),
        "RIGHT_ELBOW": _joint(30, 40, vis, 0, 0, 0),
        "RIGHT_WRIST": _joint(50, 40, vis, 0, 1, 0),
        "LEFT_SHOULDER": _joint(70, 20, vis, -1, 0, 0),
        "LEFT_ELBOW": _joint(70, 40, vis, -2, 0, 0),
        "LEFT_WRIST": _joint(70, 60, vis, -3, 0, 0),
        "LEFT_HIP": _joint(60, 70, vis),
        "RIGHT_HIP": _joint(40, 70, vis),
    }


class TestArmGeometry:
    def test_returns_2d_and_3d_bone_lengths(self):
        frame = {
            "S": _joint(0, 0, 0.9, 0, 0, 0),
            "E": _joint(3, 4, 0.9, 0, 0.3, 0),
            "W": _joint(3, 10, 0.9, 0.4, 0.3, 0),
        }

        geometry = arm_geometry(frame, ("S", "E", "W"))

        assert geometry["upper_2d"] == pytest.approx(5.0)
        assert geometry["fore_2d"] == pytest.approx(6.0)
        assert geometry["upper_3d"] == pytest.approx(0.3)
        assert geometry["fore_3d"] == pytest.approx(0.4)

    def test_returns_none_values_when_frame_missing(self):
        geometry = arm_geometry(None, ("S", "E", "W"))

        assert all(value is None for value in geometry.values())


class TestInFrame:
    def test_inside(self):
        assert in_frame((10.0, 10.0), FRAME_SIZE) is True

    def test_outside_or_missing(self):
        assert in_frame((-1.0, 10.0), FRAME_SIZE) is False
        assert in_frame((10.0, 100.0), FRAME_SIZE) is False
        assert in_frame(None, FRAME_SIZE) is False


class TestBuildVideoRows:
    def _rows(self, frames, frame_indices):
        return build_video_rows(
            video_id=7,
            frames=frames,
            frame_indices=frame_indices,
            frame_size=FRAME_SIZE,
            handedness="right",
        )

    def test_row_has_exactly_the_declared_columns(self):
        rows = self._rows([_frame()], [0])

        assert set(rows[0]) == set(FIELDNAMES)

    def test_direct_row_records_2d_and_3d_elbow_angles(self):
        row = self._rows([_frame()], [0])[0]

        assert row["trail_elbow_source"] == "direct"
        assert row["trail_elbow_angle_2d"] == pytest.approx(90.0)
        assert row["trail_elbow_angle_3d"] == pytest.approx(90.0)
        assert row["lead_elbow_angle_2d"] == pytest.approx(180.0)
        assert row["pose_detected"] is True
        assert row["video_id"] == 7

    def test_missing_event_frame_is_recovered_from_neighbors(self):
        rows = self._rows([_frame(), None, _frame()], [0, 1])

        recovered = rows[1]
        assert recovered["pose_detected"] is False
        assert recovered["trail_elbow_source"] == "window"
        assert recovered["trail_elbow_angle_2d"] == pytest.approx(90.0)

    def test_low_visibility_arm_is_missing_but_tilts_use_stricter_threshold(self):
        row = self._rows([_frame(vis=0.4)], [0])[0]

        assert row["trail_elbow_source"] == "direct"  # 0.4 >= 0.3 elbow threshold
        assert row["shoulder_tilt"] is None  # 0.4 < 0.5 tilt threshold

    def test_very_low_visibility_arm_has_no_value(self):
        row = self._rows([_frame(vis=0.1)], [0])[0]

        assert row["trail_elbow_source"] == "missing"
        assert row["trail_elbow_angle_2d"] is None
        assert row["trail_elbow_angle_3d"] is None

    def test_event_frame_past_end_of_video_yields_missing_row(self):
        row = self._rows([_frame()], [5])[0]

        assert row["pose_detected"] is False
        assert row["trail_elbow_source"] == "missing"

    def test_records_qc_ingredients(self):
        row = self._rows([_frame()], [0])[0]

        assert row["trail_min_vis"] == pytest.approx(0.9)
        assert row["trail_upper_2d"] == pytest.approx(20.0)
        assert row["trail_fore_2d"] == pytest.approx(20.0)
        assert row["trail_in_frame"] is True
        assert row["frame_w"] == 100 and row["frame_h"] == 100
