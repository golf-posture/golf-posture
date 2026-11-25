"""Model answers computed once (on a GPU machine) and replayed as a PoseSource: models faked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.pose_refiner import MODEL_VITPOSE, MODEL_YOLOX, refine_frames
from pipeline.refiner_cache import CachedSource, build_cache, event_requests

EVENTS = {0: 40, 1: 50, 2: 60, 3: 70, 4: 80, 5: 90, 6: 100, 7: 110}
ARM = {"RIGHT_SHOULDER": (30.0, 20.0, 0.9), "RIGHT_ELBOW": (30.0, 40.0, 0.9), "RIGHT_WRIST": (50.0, 40.0, 0.9)}


class TestEventRequests:
    def test_every_event_window_with_the_per_event_model(self):
        requests = event_requests(EVENTS, last_frame=200)

        assert requests[70] == MODEL_VITPOSE and requests[110] == MODEL_YOLOX
        assert set(range(68, 73)) <= set(requests)
        assert len(requests) == 8 * 5

    def test_frames_past_the_end_of_the_video_are_dropped(self):
        assert max(event_requests(EVENTS, last_frame=111)) == 111


class TestCachedSource:
    def _source(self):
        return CachedSource({"8": {"70": {name: list(value) for name, value in ARM.items()}, "71": None}})

    def test_returns_the_stored_joints_for_the_video_stem(self):
        answer = self._source().joints(Path("learning_video/x/8.mp4"), {70: MODEL_VITPOSE})

        assert answer[70]["RIGHT_WRIST"] == [50.0, 40.0, 0.9]

    def test_frame_without_an_answer_or_not_stored_is_none(self):
        answer = self._source().joints(Path("8.mp4"), {71: MODEL_VITPOSE, 99: MODEL_VITPOSE})

        assert answer == {71: None, 99: None}

    def test_unknown_video_answers_none_for_every_frame(self):
        assert self._source().joints(Path("9.mp4"), {70: MODEL_VITPOSE}) == {70: None}

    def test_load_reads_the_saved_file(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"videos": {"8": {"70": {"RIGHT_WRIST": [1, 2, 0.5]}}}}), encoding="utf-8")

        assert CachedSource.load(path).joints(Path("8.mp4"), {70: MODEL_VITPOSE})[70]["RIGHT_WRIST"] == [1, 2, 0.5]

    def test_works_as_the_pose_source_of_refine_frames(self):
        frames = [{"RIGHT_SHOULDER": (0.0, 0.0, 0.1, 1.0, 2.0, 3.0), "RIGHT_ELBOW": (0.0, 0.0, 0.1, 0.0, 0.0, 0.0),
                   "RIGHT_WRIST": (0.0, 0.0, 0.1, 0.0, 0.0, 0.0)} for _ in range(120)]
        source = CachedSource({"8": {str(frame): {n: list(v) for n, v in ARM.items()} for frame in range(68, 73)}})

        result = refine_frames(frames, 0, Path("8.mp4"), {3: 70}, source, enabled_events=frozenset({3}))

        assert result.status == "ok" and result.refined == 5 and result.refined_frames == frozenset(range(68, 73))
        assert result.frames[70]["RIGHT_WRIST"] == (50.0, 40.0, 0.9, 0.0, 0.0, 0.0)


class FakeGolfSource:
    def __init__(self):
        self.calls = []

    def joints(self, video_path, requests):
        self.calls.append((Path(video_path).name, dict(requests)))
        return {frame: {n: v for n, v in ARM.items()} for frame in requests}


class TestBuildCache:
    EVENTS_BY_VIDEO = {"8": EVENTS, "9": {k: v + 5 for k, v in EVENTS.items()}}

    def test_stores_the_answers_of_every_video(self, tmp_path):
        output = tmp_path / "cache.json"
        source = FakeGolfSource()

        build_cache(self.EVENTS_BY_VIDEO, tmp_path, source, output, frame_count=lambda path: 300)

        saved = json.loads(output.read_text(encoding="utf-8"))["videos"]
        assert set(saved) == {"8", "9"} and len(saved["8"]) == 40
        assert saved["8"]["70"]["RIGHT_WRIST"] == [50.0, 40.0, 0.9]

    def test_videos_already_in_the_file_are_not_computed_again(self, tmp_path):
        output = tmp_path / "cache.json"
        build_cache({"8": EVENTS}, tmp_path, FakeGolfSource(), output, frame_count=lambda path: 300)
        source = FakeGolfSource()

        build_cache(self.EVENTS_BY_VIDEO, tmp_path, source, output, frame_count=lambda path: 300)

        assert [name for name, _ in source.calls] == ["9.mp4"]

    def test_a_video_that_fails_is_reported_and_skipped(self, tmp_path):
        class Flaky(FakeGolfSource):
            def joints(self, video_path, requests):
                if Path(video_path).stem == "8":
                    raise OSError("unreadable")
                return super().joints(video_path, requests)

        failed = build_cache(self.EVENTS_BY_VIDEO, tmp_path, Flaky(), tmp_path / "c.json", frame_count=lambda p: 300)

        assert failed == ["8"]
        assert set(json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["videos"]) == {"9"}

    @pytest.mark.parametrize("limit,expected", [(1, 1), (0, 2)])
    def test_limit_caps_the_number_of_videos(self, tmp_path, limit, expected):
        build_cache(self.EVENTS_BY_VIDEO, tmp_path, FakeGolfSource(), tmp_path / "c.json",
                    frame_count=lambda path: 300, limit=limit)

        assert len(json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["videos"]) == expected
