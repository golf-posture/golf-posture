"""The production `PoseSource`: golfer detection (YOLOv8x), ViTPose-H and YOLOv8x arm joints for chosen frames.

One process, no mmpose: the golfer is found with YOLOv8x-pose, then
  - frames asked for "vitpose" are cropped on that box and measured by the exported ViTPose-H (vitpose_export.py),
  - frames asked for "yolox" use YOLOv8x's own keypoints.
Both return the benchmark joints (shoulders, elbows, wrists, hips) as (px, py, score) in original-frame pixels.

`GolfPoseSource` takes its three heavy collaborators as plain callables, so the logic is testable without a GPU;
`load_golfpose_source` builds the real ones. Models load on first use, not when the source is created.

    python -m pipeline.pose_source --boxes outputs/golfpose/boxes.json --model vitpose --output cache.json
writes a prediction cache of every frame in boxes.json, to be scored by pose_benchmark --predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from pipeline.golfer_detection import (
    YOLO_BATCH,
    YOLO_DEFAULT_WEIGHTS,
    YOLO_IMAGE_SIZE,
    YOLO_MIN_CONFIDENCE,
    pick_golfer,
    yolo_joints,
)
from pipeline.golfpose_joints import golfpose_joints, predictions_payload
from pipeline.pose_refiner import MODEL_VITPOSE, MODEL_YOLOX
from pipeline.vitpose_runtime import crop_for_model, model_input, predict_keypoints

DEFAULT_VITPOSE = Path("outputs/golfpose/vitpose_huge.ts")
VITPOSE_BATCH = 5

Detection = tuple[Sequence[float], Sequence[Sequence[float]]]  # golfer box xyxy, COCO-17 (x, y, conf)
ArmJoints = dict[str, tuple[float, float, float]]
ReadFrames = Callable[[Path, Sequence[int]], Mapping[int, np.ndarray]]
Detect = Callable[[Sequence[np.ndarray]], Sequence[Optional[Detection]]]
Heatmaps = Callable[[np.ndarray], np.ndarray]  # (N, 3, 256, 192) -> (N, 17, 64, 48)


def _arm_joints(joints: Mapping[str, Sequence[float]]) -> ArmJoints:
    return {name: (float(value[0]), float(value[1]), float(value[2])) for name, value in joints.items()}


class GolfPoseSource:
    """Implements pose_refiner.PoseSource on top of three collaborators (frame reader, detector, ViTPose network)."""

    def __init__(self, read_frames: ReadFrames, detect: Detect, heatmaps: Heatmaps, batch_size: int = VITPOSE_BATCH):
        self._read_frames = read_frames
        self._detect = detect
        self._heatmaps = heatmaps
        self._batch_size = batch_size

    def joints(self, video_path: Path, requests: Mapping[int, str]) -> dict[int, Optional[ArmJoints]]:
        unknown = set(requests.values()) - {MODEL_VITPOSE, MODEL_YOLOX}
        if unknown:
            raise ValueError(f"unknown pose model(s): {sorted(unknown)}")
        result: dict[int, Optional[ArmJoints]] = {frame: None for frame in requests}
        if not requests:
            return result
        images = dict(self._read_frames(video_path, sorted(requests)))
        readable = [frame for frame in sorted(requests) if frame in images]
        if not readable:
            return result
        found = dict(zip(readable, self._detect([images[frame] for frame in readable])))
        vitpose_frames = []
        for frame in readable:
            detection = found[frame]
            if detection is None:
                continue
            if requests[frame] == MODEL_YOLOX:
                result[frame] = _arm_joints(yolo_joints(detection[1]))
            else:
                vitpose_frames.append(frame)
        self._run_vitpose(result, vitpose_frames, images, found)
        return result

    def _run_vitpose(self, result: dict[int, Optional[ArmJoints]], frames: Sequence[int],
                     images: Mapping[int, np.ndarray], found: Mapping[int, Optional[Detection]]) -> None:
        for start in range(0, len(frames), self._batch_size):
            chunk = frames[start:start + self._batch_size]
            crops = [crop_for_model(images[frame], found[frame][0]) for frame in chunk]
            heatmaps = self._heatmaps(np.stack([model_input(crop) for crop, _, _ in crops]))
            for frame, (_, center, scale), maps in zip(chunk, crops, heatmaps):
                keypoints, scores = predict_keypoints(maps, center, scale)
                result[frame] = _arm_joints(golfpose_joints(keypoints, scores))


# ----------------------------------------------------------- the real collaborators
def read_video_frames(video_path: Path, frames: Sequence[int]) -> dict[int, np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    images: dict[int, np.ndarray] = {}
    try:
        for frame in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if ok:
                images[frame] = image
    finally:
        capture.release()
    return images


class YoloGolferDetector:
    """YOLOv8-pose golfer detection, batched. The model loads on the first call."""

    def __init__(self, weights: Path, device: str):
        self._weights, self._device, self._model = weights, device, None

    def __call__(self, images: Sequence[np.ndarray]) -> list[Optional[Detection]]:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights))
        found: list[Optional[Detection]] = []
        for start in range(0, len(images), YOLO_BATCH):
            outputs = self._model(list(images[start:start + YOLO_BATCH]), imgsz=YOLO_IMAGE_SIZE,
                                  conf=YOLO_MIN_CONFIDENCE, device=self._device, verbose=False)
            found.extend(self._golfer(output) for output in outputs)
        return found

    @staticmethod
    def _golfer(output: Any) -> Optional[Detection]:
        if output.keypoints is None or len(output.boxes) == 0:
            return None
        boxes = output.boxes.xyxy.cpu().numpy().tolist()
        points = output.keypoints.data.cpu().numpy().tolist()
        chosen = pick_golfer(boxes, points)
        return None if chosen is None else (boxes[chosen], points[chosen])


class VitposeNetwork:
    """The exported ViTPose-H TorchScript module. It loads on the first call."""

    def __init__(self, path: Path, device: str):
        self._path, self._device, self._model = path, device, None

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            self._model = torch.jit.load(str(self._path), map_location=self._device).eval()
        with torch.no_grad():
            return self._model(torch.from_numpy(batch).to(self._device)).float().cpu().numpy()


def default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_golfpose_source(
    vitpose_path: Path = DEFAULT_VITPOSE,
    yolo_weights: Path = YOLO_DEFAULT_WEIGHTS,
    device: Optional[str] = None,
    batch_size: int = VITPOSE_BATCH,
) -> GolfPoseSource:
    device = device or default_device()
    return GolfPoseSource(read_video_frames, YoloGolferDetector(yolo_weights, device),
                          VitposeNetwork(vitpose_path, device), batch_size=batch_size)


def refiner_from_options(mode: str, vitpose_path: Path) -> tuple[Optional[GolfPoseSource], dict[str, str]]:
    """The pose source for run_analysis's --pose-refiner, and the note to record in the result ({} when off).

    A model that cannot be set up never fails the analysis: the source is None and the note says why.
    """
    if mode != "golfpose":
        return None, {}
    path = Path(vitpose_path)
    if not path.exists():
        return None, {"mode": mode, "status": f"failed: model file not found: {path}"}
    try:
        return load_golfpose_source(path), {"mode": mode}
    except Exception as error:
        return None, {"mode": mode, "status": f"failed: {type(error).__name__}: {error}"}


# ------------------------------------------------------------------ smoke test CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boxes", type=Path, required=True, help="boxes.json: only its video paths and frame lists are used")
    parser.add_argument("--model", choices=(MODEL_VITPOSE, MODEL_YOLOX), required=True)
    parser.add_argument("--vitpose", type=Path, default=DEFAULT_VITPOSE)
    parser.add_argument("--yolo-weights", type=Path, default=YOLO_DEFAULT_WEIGHTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    videos = json.loads(args.boxes.read_text(encoding="utf-8"))["videos"]
    source = load_golfpose_source(args.vitpose, args.yolo_weights, args.device)
    frames: dict[str, dict[int, Optional[dict[str, tuple[float, ...]]]]] = {}
    for name, entry in sorted(videos.items()):
        wanted = {int(frame): args.model for frame in entry["boxes"]}
        answer = source.joints(Path(entry["path"]), wanted)
        frames[name] = {frame: None if joints is None else {n: (*v, 0.0, 0.0, 0.0) for n, v in joints.items()}
                        for frame, joints in answer.items()}
        sys.stdout.write(f"{name}: {len(wanted)} frames\n")
        sys.stdout.flush()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions_payload(f"pose_source_{args.model}", frames)), encoding="utf-8")
    sys.stdout.write(f"saved: {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
