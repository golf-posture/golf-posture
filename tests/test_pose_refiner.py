"""Overlay of ViTPose-H / YOLOv8x arm joints onto the tracked MediaPipe frames."""

from pathlib import Path

import pytest

from pipeline.pose_refiner import (
    ARM_JOINTS,
    MODEL_VITPOSE,
    MODEL_YOLOX,
    overlay_joints,
    parse_events,
    refine_frames,
    refiner_model,
    request_frames,
)

VIDEO = Path("clip.mp4")
WORLD = (0.1, 0.2, 0.3)


def _frame(offset=0.0):
    """A MediaPipe-like frame: arm joints, a head landmark, world coordinates."""
    return {
        "LEFT_SHOULDER": (10 + offset, 10, 0.9, *WORLD),
        "LEFT_ELBOW": (20 + offset, 20, 0.9, *WORLD),
        "LEFT_WRIST": (30 + offset, 30, 0.9, *WORLD),
        "NOSE": (50 + offset, 5, 0.9, *WORLD),
    }


class FakeSource:
    def __init__(self, answers=None, error=None):
        self.answers = answers or {}
        self.error = error
        self.calls = []

    def joints(self, video_path, requests):
        self.calls.append((video_path, dict(requests)))
        if self.error:
            raise self.error
        return {frame: self.answers.get(frame) for frame in requests}


class TestRefinerModel:
    def test_finish_uses_yolox_and_other_events_use_vitpose(self):
        assert refiner_model(7) == MODEL_YOLOX
        assert [refiner_model(e) for e in range(7)] == [MODEL_VITPOSE] * 7


class TestRequestFrames:
    def test_window_around_every_enabled_event(self):
        requests = request_frames({3: 100, 7: 200}, window=(0, 299), radius=2, enabled_events=frozenset({3, 7}))
        assert sorted(requests) == [98, 99, 100, 101, 102, 198, 199, 200, 201, 202]
        assert requests[100] == MODEL_VITPOSE and requests[200] == MODEL_YOLOX

    def test_disabled_events_are_not_requested(self):
        requests = request_frames({3: 100, 7: 200}, window=(0, 299), radius=2, enabled_events=frozenset({3}))
        assert sorted(requests) == [98, 99, 100, 101, 102]

    def test_frames_outside_the_tracked_window_are_dropped(self):
        requests = request_frames({3: 100}, window=(99, 101), radius=2, enabled_events=frozenset({3}))
        assert sorted(requests) == [99, 100, 101]

    def test_no_events_means_no_requests(self):
        assert request_frames({}, window=(0, 10), radius=2, enabled_events=frozenset(range(8))) == {}


class TestOverlayJoints:
    def test_replaces_position_and_score_and_keeps_world_coordinates(self):
        frames = [_frame(), _frame()]
        result = overlay_joints(frames, {11: {"LEFT_WRIST": (99.0, 88.0, 0.7)}}, start_frame=10)
        assert result[1]["LEFT_WRIST"] == (99.0, 88.0, 0.7, *WORLD)
        assert result[1]["LEFT_ELBOW"] == frames[1]["LEFT_ELBOW"]
        assert result[1]["NOSE"] == frames[1]["NOSE"]
        assert result[0] == frames[0]

    def test_does_not_modify_the_input_frames(self):
        frames = [_frame()]
        before = {name: tuple(value) for name, value in frames[0].items()}
        overlay_joints(frames, {0: {"LEFT_WRIST": (1.0, 2.0, 0.5)}}, start_frame=0)
        assert frames[0] == before

    def test_frame_without_a_tracked_pose_stays_empty(self):
        result = overlay_joints([None], {0: {"LEFT_WRIST": (1.0, 2.0, 0.5)}}, start_frame=0)
        assert result == [None]

    def test_none_answer_and_out_of_range_frames_are_ignored(self):
        frames = [_frame()]
        result = overlay_joints(frames, {0: None, 5: {"LEFT_WRIST": (1.0, 2.0, 0.5)}}, start_frame=0)
        assert result == frames

    def test_joint_missing_from_the_frame_is_not_invented(self):
        frames = [{"NOSE": (1.0, 1.0, 0.9, *WORLD)}]
        result = overlay_joints(frames, {0: {"LEFT_WRIST": (1.0, 2.0, 0.5)}}, start_frame=0)
        assert "LEFT_WRIST" not in result[0]


class TestRefineFrames:
    ANSWERS = {100: {"LEFT_WRIST": (1.0, 2.0, 0.6)}, 101: None}

    def _run(self, source, enabled=frozenset({3})):
        frames = [_frame() for _ in range(10)]
        return frames, refine_frames(frames, 98, VIDEO, {3: 100}, source, radius=2, enabled_events=enabled)

    def test_overlays_answered_frames_and_reports_how_many(self):
        source = FakeSource(self.ANSWERS)
        frames, result = self._run(source)
        assert result.status == "ok" and result.refined == 1
        assert result.frames[2]["LEFT_WRIST"][:3] == (1.0, 2.0, 0.6)
        assert result.frames[3] == frames[3]

    def test_asks_the_source_once_with_the_per_frame_models(self):
        source = FakeSource(self.ANSWERS)
        self._run(source)
        assert len(source.calls) == 1
        assert source.calls[0] == (VIDEO, {98: MODEL_VITPOSE, 99: MODEL_VITPOSE, 100: MODEL_VITPOSE,
                                           101: MODEL_VITPOSE, 102: MODEL_VITPOSE})

    def test_source_failure_returns_the_original_frames_with_a_reason(self):
        frames, result = self._run(FakeSource(error=RuntimeError("worker died")))
        assert result.frames == frames
        assert result.refined == 0
        assert result.status.startswith("failed") and "worker died" in result.status

    def test_nothing_to_request_does_not_call_the_source(self):
        source = FakeSource()
        frames, result = self._run(source, enabled=frozenset())
        assert source.calls == []
        assert result.status == "skipped" and result.frames == frames


class TestParseEvents:
    def test_all(self):
        assert parse_events("all") == frozenset(range(8))

    def test_numbers(self):
        assert parse_events("3,7") == frozenset({3, 7})

    def test_names_are_case_insensitive_and_may_mix_with_numbers(self):
        assert parse_events("top, Finish") == frozenset({3, 7})
        assert parse_events("Mid-backswing,0") == frozenset({2, 0})

    @pytest.mark.parametrize("text", ["", "9", "-1", "swing", "3,,7x"])
    def test_invalid_values_are_rejected(self, text):
        with pytest.raises(ValueError):
            parse_events(text)


class TestJointSelection:
    ANSWER = {"LEFT_WRIST": (99.0, 88.0, 0.7), "LEFT_HIP": (1.0, 2.0, 0.6)}

    @staticmethod
    def _frame_with_hip():
        return {**_frame(), "LEFT_HIP": (60.0, 70.0, 0.9, *WORLD)}

    def test_only_arm_joints_are_overlaid_by_default(self):
        frame = self._frame_with_hip()
        result = overlay_joints([frame], {0: self.ANSWER}, start_frame=0)
        assert result[0]["LEFT_WRIST"][:3] == (99.0, 88.0, 0.7)
        assert result[0]["LEFT_HIP"] == frame["LEFT_HIP"]

    def test_the_joint_set_can_be_widened(self):
        frame = self._frame_with_hip()
        result = overlay_joints([frame], {0: self.ANSWER}, start_frame=0, joints=ARM_JOINTS | {"LEFT_HIP"})
        assert result[0]["LEFT_HIP"] == (1.0, 2.0, 0.6, *WORLD)

    def test_arm_joint_set_is_both_shoulders_elbows_and_wrists(self):
        assert ARM_JOINTS == {f"{side}_{part}" for side in ("LEFT", "RIGHT")
                              for part in ("SHOULDER", "ELBOW", "WRIST")}

    def test_a_frame_where_nothing_changed_is_not_counted_as_refined(self):
        frames = [self._frame_with_hip()]
        source = FakeSource({98: {"LEFT_HIP": (1.0, 2.0, 0.6)}})
        result = refine_frames(frames, 98, VIDEO, {3: 100}, source, radius=2, enabled_events=frozenset({3}))
        assert result.status == "ok" and result.refined == 0 and result.frames == frames
