"""analyze_user_video with a pose refiner: MediaPipe tracking stays, the refiner overlays arm joints (models faked)."""

from __future__ import annotations

import pytest

from pipeline import user_tracking
from pipeline.pose_refiner import MODEL_VITPOSE, MODEL_YOLOX
from pipeline.user_tracking import analyze_user_video

VIDEO = "clip.mp4"
EVENT_FRAMES = [40, 50, 60, 70, 80, 90, 100, 110]
FRAME_SIZE = (100, 100)
LANDMARKS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]


def _joint(px, py, vis=0.9, wx=0.0, wy=0.0, wz=0.0):
    return (px, py, vis, wx, wy, wz)


def _frame():
    """Right arm: 90 degree elbow in 2D and 3D; left arm straight."""
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


SHIFTED_RIGHT_ARM = {  # the same 90 degree arm, moved 5 px to the right
    "RIGHT_SHOULDER": (35.0, 20.0, 0.9), "RIGHT_ELBOW": (35.0, 40.0, 0.9), "RIGHT_WRIST": (55.0, 40.0, 0.9),
}


class FakeSource:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def joints(self, video_path, requests):
        self.calls.append(dict(requests))
        if self.error:
            raise self.error
        return {frame: self.answer for frame in requests}


@pytest.fixture(autouse=True)
def fake_tracking(monkeypatch):
    """MediaPipe tracking and video metadata replaced by plain frames."""
    monkeypatch.setattr(user_tracking, "video_frame_count", lambda path: 300)
    monkeypatch.setattr(user_tracking, "video_size", lambda path: FRAME_SIZE)

    def tracked(path, last_frame, complexity=1, start_frame=0, crop=None):
        return [_frame() for _ in range(last_frame - start_frame + 1)]

    monkeypatch.setattr(user_tracking, "extract_tracked_frames", tracked)


def _analyze(source=None, event_frames=EVENT_FRAMES, **options):
    return analyze_user_video(VIDEO, event_frames, "right", refiner=source, **options)


class TestWithoutRefiner:
    def test_output_has_no_refiner_entry(self):
        events = _analyze()
        assert len(events) == 8
        assert all("pose_refiner" not in event["qc"] for event in events)
        assert events[3]["landmarks"]["RIGHT_WRIST"] == pytest.approx((50.0, 40.0, 0.9))


class TestWithRefiner:
    def test_event_landmarks_use_the_refined_joints(self):
        events = _analyze(FakeSource(SHIFTED_RIGHT_ARM))
        assert events[3]["landmarks"]["RIGHT_WRIST"] == pytest.approx((55.0, 40.0, 0.9))
        assert events[3]["landmarks"]["LEFT_WRIST"] == pytest.approx((70.0, 60.0, 0.9))  # not in the answer
        assert all(event["qc"]["pose_refiner"] == "ok" for event in events)

    def test_asks_for_every_event_window_with_the_per_event_model(self):
        source = FakeSource(SHIFTED_RIGHT_ARM)
        _analyze(source)
        requests = source.calls[0]
        assert requests[70] == MODEL_VITPOSE and requests[110] == MODEL_YOLOX
        assert set(range(38, 43)) <= set(requests)

    def test_only_the_enabled_events_are_refined(self):
        source = FakeSource(SHIFTED_RIGHT_ARM)
        events = _analyze(source, refine_events=frozenset({3}))
        assert sorted(source.calls[0]) == [68, 69, 70, 71, 72]
        assert events[3]["landmarks"]["RIGHT_WRIST"] == pytest.approx((55.0, 40.0, 0.9))
        assert events[0]["landmarks"]["RIGHT_WRIST"] == pytest.approx((50.0, 40.0, 0.9))


class TestFallbacks:
    def test_model_failure_keeps_the_mediapipe_result_and_records_why(self):
        baseline = _analyze()
        events = _analyze(FakeSource(error=RuntimeError("no gpu")))
        assert events[3]["landmarks"] == baseline[3]["landmarks"]
        assert events[3]["features"] == baseline[3]["features"]
        assert events[3]["qc"]["pose_refiner"].startswith("failed") and "no gpu" in events[3]["qc"]["pose_refiner"]

    def test_anything_but_eight_events_skips_the_refiner(self):
        source = FakeSource(SHIFTED_RIGHT_ARM)
        events = _analyze(source, event_frames=EVENT_FRAMES[:7])
        assert source.calls == []
        assert len(events) == 7
        assert events[0]["qc"]["pose_refiner"] == "skipped: expected 8 events, got 7"


STRAIGHT_RIGHT_ARM = {  # 180 degrees in 2D while MediaPipe's 3D still says 90: the 3D rules must not veto it
    "RIGHT_SHOULDER": (30.0, 20.0, 0.9), "RIGHT_ELBOW": (30.0, 40.0, 0.9), "RIGHT_WRIST": (30.0, 60.0, 0.9),
}
LOW_SCORE_RIGHT_ARM = {name: (x, y, 0.35) for name, (x, y, _) in STRAIGHT_RIGHT_ARM.items()}


class TestRefinedQc:
    def test_refined_arm_is_not_vetoed_by_mediapipes_3d(self):
        events = _analyze(FakeSource(STRAIGHT_RIGHT_ARM))
        assert events[3]["qc"]["trail"]["inconsistent"] is False
        assert events[3]["features"]["trail_elbow_angle"] == pytest.approx(180.0)

    def test_low_model_score_between_the_two_thresholds_is_kept(self):
        events = _analyze(FakeSource(LOW_SCORE_RIGHT_ARM))
        assert events[3]["qc"]["trail"]["low_vis"] is False

    def test_unrefined_event_with_a_low_score_is_still_flagged(self, monkeypatch):
        def low_visibility_frame():
            frame = _frame()
            frame["RIGHT_ELBOW"] = _joint(30, 40, 0.35, 0.1, 0.25, 0.0)
            return frame

        monkeypatch.setattr(user_tracking, "extract_tracked_frames",
                            lambda path, last, complexity=1, start_frame=0, crop=None:
                            [low_visibility_frame() for _ in range(last - start_frame + 1)])
        events = _analyze(FakeSource(STRAIGHT_RIGHT_ARM), refine_events=frozenset({3}))
        assert events[0]["qc"]["trail"]["low_vis"] is True   # not refined: the 0.4 threshold applies
        assert events[3]["qc"]["trail"]["low_vis"] is False  # refined with score 0.9


class TestArmsOnly:
    def test_hips_in_the_models_answer_do_not_replace_the_tracked_hips(self):
        answer = {**SHIFTED_RIGHT_ARM, "LEFT_HIP": (90.0, 90.0, 0.9), "RIGHT_HIP": (10.0, 90.0, 0.9)}
        events = _analyze(FakeSource(answer))
        assert events[3]["landmarks"]["RIGHT_WRIST"] == pytest.approx((55.0, 40.0, 0.9))
        assert events[3]["landmarks"]["LEFT_HIP"] == pytest.approx((60.0, 70.0, 0.9))
        assert events[3]["landmarks"]["RIGHT_HIP"] == pytest.approx((40.0, 70.0, 0.9))
