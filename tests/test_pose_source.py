"""GolfPoseSource: golfer detection, ViTPose-H and YOLOv8x joints for the requested frames (models faked)."""

from pathlib import Path

import numpy as np
import pytest

from pipeline.golfpose_joints import GOLFPOSE_JOINT_INDEX
from pipeline.pose_refiner import MODEL_VITPOSE, MODEL_YOLOX
from pipeline.pose_source import GolfPoseSource, refiner_from_options
from pipeline.vitpose_runtime import HEATMAP_SIZE, bbox_center_scale, predict_keypoints

VIDEO = Path("clip.mp4")
BOX = [100.0, 50.0, 200.0, 250.0]
YOLO_POINTS = [[float(i), float(i) * 2, 0.9] for i in range(17)]
ARM_JOINTS = set(GOLFPOSE_JOINT_INDEX)


def _heatmaps():
    """One 17-channel set where channel c peaks at a different place."""
    width, height = HEATMAP_SIZE
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([np.exp(-((xs - (5 + c)) ** 2 + (ys - (10 + 2 * c)) ** 2) / 8.0) for c in range(17)]).astype(np.float32)


class Fakes:
    """Records what the source asked the collaborators for."""

    def __init__(self, detections=None, unreadable=()):
        self.detections = detections if detections is not None else {}
        self.unreadable = set(unreadable)
        self.read_calls, self.detect_calls, self.net_batches = [], [], []

    def read_frames(self, video_path, frames):
        self.read_calls.append((video_path, list(frames)))
        return {f: np.full((300, 400, 3), f % 255, dtype=np.uint8) for f in frames if f not in self.unreadable}

    def detect(self, images):
        self.detect_calls.append(len(images))
        return [self.detections.get(int(image[0, 0, 0])) for image in images]

    def heatmaps(self, batch):
        self.net_batches.append(batch.shape)
        return np.stack([_heatmaps() for _ in range(batch.shape[0])])


def _source(fakes, batch_size=5):
    return GolfPoseSource(fakes.read_frames, fakes.detect, fakes.heatmaps, batch_size=batch_size)


class TestVitposeFrames:
    def test_joints_come_from_the_decoded_heatmaps_in_image_pixels(self):
        fakes = Fakes({10: (BOX, YOLO_POINTS)})
        answer = _source(fakes).joints(VIDEO, {10: MODEL_VITPOSE})[10]
        center, scale = bbox_center_scale(BOX)
        keypoints, scores = predict_keypoints(_heatmaps(), center, scale)
        assert set(answer) == ARM_JOINTS
        for name, channel in GOLFPOSE_JOINT_INDEX.items():
            assert answer[name][:2] == pytest.approx(tuple(keypoints[channel]), abs=1e-3)
            assert answer[name][2] == pytest.approx(float(scores[channel]), abs=1e-4)

    def test_frames_are_batched(self):
        fakes = Fakes({f: (BOX, YOLO_POINTS) for f in range(5)})
        _source(fakes, batch_size=2).joints(VIDEO, {f: MODEL_VITPOSE for f in range(5)})
        assert [shape[0] for shape in fakes.net_batches] == [2, 2, 1]
        assert fakes.net_batches[0][1:] == (3, 256, 192)


class TestYoloxFrames:
    def test_joints_come_from_the_yolo_keypoints_without_running_vitpose(self):
        fakes = Fakes({20: (BOX, YOLO_POINTS)})
        answer = _source(fakes).joints(VIDEO, {20: MODEL_YOLOX})[20]
        assert set(answer) == ARM_JOINTS
        assert answer["LEFT_WRIST"] == (9.0, 18.0, 0.9)  # COCO keypoint 9
        assert fakes.net_batches == []


class TestMixedAndMissing:
    def test_one_read_and_one_detection_pass_for_both_models(self):
        fakes = Fakes({10: (BOX, YOLO_POINTS), 20: (BOX, YOLO_POINTS)})
        answer = _source(fakes).joints(VIDEO, {10: MODEL_VITPOSE, 20: MODEL_YOLOX})
        assert set(answer) == {10, 20} and all(answer.values())
        assert fakes.read_calls == [(VIDEO, [10, 20])] and fakes.detect_calls == [2]

    def test_frame_without_a_golfer_is_none_for_both_models(self):
        fakes = Fakes({})
        answer = _source(fakes).joints(VIDEO, {10: MODEL_VITPOSE, 20: MODEL_YOLOX})
        assert answer == {10: None, 20: None}
        assert fakes.net_batches == []

    def test_unreadable_frame_is_none_and_not_detected(self):
        fakes = Fakes({10: (BOX, YOLO_POINTS)}, unreadable={20})
        answer = _source(fakes).joints(VIDEO, {10: MODEL_VITPOSE, 20: MODEL_VITPOSE})
        assert answer[20] is None and answer[10] is not None
        assert fakes.detect_calls == [1]

    def test_no_requests_touches_nothing(self):
        fakes = Fakes()
        assert _source(fakes).joints(VIDEO, {}) == {}
        assert fakes.read_calls == []

    def test_unknown_model_name_is_rejected(self):
        with pytest.raises(ValueError):
            _source(Fakes({10: (BOX, YOLO_POINTS)})).joints(VIDEO, {10: "hrnet"})


class TestRefinerFromOptions:
    def test_off_loads_nothing_and_leaves_no_note(self, tmp_path):
        assert refiner_from_options("none", tmp_path / "missing.ts") == (None, {})

    def test_missing_model_file_is_reported_not_raised(self, tmp_path):
        source, note = refiner_from_options("golfpose", tmp_path / "missing.ts")
        assert source is None
        assert note["mode"] == "golfpose" and "not found" in note["status"]

    def test_existing_model_file_gives_a_source_without_loading_it(self, tmp_path):
        model = tmp_path / "vitpose.ts"
        model.write_bytes(b"not a real model")
        source, note = refiner_from_options("golfpose", model)
        assert isinstance(source, GolfPoseSource)
        assert note == {"mode": "golfpose"}
