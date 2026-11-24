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


class TestProcessVideoAt:
    EVENTS = [10, 20, 30, 40, 50, 60, 70, 80]

    @pytest.fixture(autouse=True)
    def fake_tracking(self, monkeypatch):
        from pipeline import extract_pro_features_tracked as module

        monkeypatch.setattr(module, "video_frame_count", lambda path: 200)
        monkeypatch.setattr(module, "video_size", lambda path: FRAME_SIZE)
        monkeypatch.setattr(module, "extract_tracked_frames", lambda path, last: [_frame() for _ in range(last + 1)])

    class Source:
        def __init__(self, answer):
            self.answer = answer

        def joints(self, video_path, requests):
            return {frame: self.answer for frame in requests}

    def test_without_a_source_nothing_is_refined(self):
        from pathlib import Path

        from pipeline.extract_pro_features_tracked import process_video_at

        rows = process_video_at(Path("8.mp4"), 8, self.EVENTS, "right")

        assert [row["refined"] for row in rows] == [False] * 8

    def test_with_a_source_the_event_rows_use_its_joints(self):
        from pathlib import Path

        from pipeline.extract_pro_features_tracked import process_video_at

        straight = {"RIGHT_SHOULDER": (30.0, 20.0, 0.8), "RIGHT_ELBOW": (30.0, 40.0, 0.8), "RIGHT_WRIST": (30.0, 60.0, 0.8)}

        rows = process_video_at(Path("8.mp4"), 8, self.EVENTS, "right", self.Source(straight))

        assert [row["refined"] for row in rows] == [True] * 8
        assert rows[3]["trail_elbow_angle_2d"] == pytest.approx(180.0)
        assert rows[3]["trail_elbow_angle_3d"] == pytest.approx(90.0)  # MediaPipe's 3D is kept

    def test_a_source_that_fails_fails_the_video(self):
        from pathlib import Path

        from pipeline.extract_pro_features_tracked import process_video_at

        class Broken:
            def joints(self, video_path, requests):
                raise OSError("no gpu")

        with pytest.raises(RuntimeError, match="failed"):
            process_video_at(Path("8.mp4"), 8, self.EVENTS, "right", Broken())

    def test_table_events_keeps_only_videos_with_all_eight_events(self, tmp_path):
        from pipeline.extract_pro_features_tracked import table_events

        lines = ["video_id,event_index,frame_index"]
        lines += [f"8,{event},{10 * event}" for event in range(8)] + [f"9,{event},{event}" for event in range(7)]
        table = tmp_path / "t.csv"
        table.write_text("\n".join(lines), encoding="utf-8")

        assert table_events(table) == {8: [0, 10, 20, 30, 40, 50, 60, 70]}


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

    def test_refined_column_marks_the_events_the_refiner_measured(self):
        rows = build_video_rows(7, [_frame(), _frame()], [0, 1], FRAME_SIZE, "right", refined_frames=frozenset({1}))

        assert [row["refined"] for row in rows] == [False, True]

    def test_refined_column_defaults_to_false(self):
        assert self._rows([_frame()], [0])[0]["refined"] is False

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
