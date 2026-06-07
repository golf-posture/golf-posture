from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EVENT_NAMES = [
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish",
]

REQUIRED_FEATURES = [
    "trail_elbow_angle",
    "lead_elbow_angle",
    "shoulder_tilt",
    "hip_tilt",
    "spine_tilt",
]

CIRCULAR_FEATURES = {"shoulder_tilt", "hip_tilt"}
RELIABILITY_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.35, "unknown": 0.5}

LANDMARK_NAMES = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}

YOLO_LANDMARK_NAMES = {
    "NOSE": 0,
    "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9,
    "RIGHT_WRIST": 10,
    "LEFT_HIP": 11,
    "RIGHT_HIP": 12,
    "LEFT_KNEE": 13,
    "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15,
    "RIGHT_ANKLE": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golf posture feedback pipeline.")
    parser.add_argument("--video", required=True, help="Input swing mp4 path.")
    parser.add_argument(
        "--checkpoint",
        help="SwingNet checkpoint path. Defaults to event_detection/models/swingnet_1800.pth.tar.",
    )
    parser.add_argument(
        "--pro-stats",
        default="data/pro_angle_stats.json",
        help="Path to pro angle statistics JSON.",
    )
    parser.add_argument(
        "--reliability-stats",
        default="data/pro_feature_stats_reliability.json",
        help="Optional reliability stats JSON. Used when the file exists.",
    )
    parser.add_argument(
        "--reference-skeletons",
        default="data/pro_reference_skeletons.json",
        help="Optional pro reference skeleton JSON for user/pro comparison images.",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis_result",
        help="Output prefix. Writes <prefix>.json and <prefix>.txt.",
    )
    parser.add_argument("--handedness", choices=["right", "left"], default="right")
    parser.add_argument(
        "--pose-backend",
        choices=["mediapipe", "yolopose", "auto", "hybrid"],
        default="mediapipe",
    )
    parser.add_argument(
        "--mock-events",
        action="store_true",
        help="Use evenly spaced mock event frames for integration testing.",
    )
    parser.add_argument("--seq-length", type=int, default=64)
    return parser.parse_args()


def output_paths(output: str) -> tuple[Path, Path, Path, Path]:
    base = Path(output)
    if base.suffix.lower() == ".json":
        json_path = base
        txt_path = base.with_suffix(".txt")
        frame_dir = base.with_suffix("").with_name(base.stem + "_frames")
        skeleton_dir = base.with_suffix("").with_name(base.stem + "_skeletons")
    else:
        json_path = Path(str(base) + ".json")
        txt_path = Path(str(base) + ".txt")
        frame_dir = Path(str(base) + "_frames")
        skeleton_dir = Path(str(base) + "_skeletons")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    return json_path, txt_path, frame_dir, skeleton_dir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pro stats JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_stats(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_stats(path)


def video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    return count


def read_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def mock_event_frames(video_path: Path) -> tuple[list[int], list[float | None]]:
    count = video_frame_count(video_path)
    events = [int(round(x)) for x in np.linspace(0, max(count - 1, 0), len(EVENT_NAMES))]
    return events, [None] * len(EVENT_NAMES)


class SampleVideo:
    def __init__(self, path: Path, input_size: int = 160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")

        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ratio = self.input_size / max(height, width)
        new_size = (int(height * ratio), int(width * ratio))
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            bordered = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[0.406 * 255, 0.456 * 255, 0.485 * 255],
            )
            images.append(cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB))
        cap.release()

        labels = np.zeros(len(images))
        sample = {"images": np.asarray(images), "labels": np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def detect_swingnet_events(
    video_path: Path,
    checkpoint_path: Path,
    seq_length: int,
) -> tuple[list[int], list[float]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SwingNet checkpoint not found: {checkpoint_path}. "
            "Pass --checkpoint or place swingnet_1800.pth.tar in event_detection/models."
        )

    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError(
            "SwingNet inference requires torch and torchvision. "
            "Install them or run with --mock-events for integration testing."
        ) from exc

    event_dir = repo_root() / "event_detection"
    sys.path.insert(0, str(event_dir))
    try:
        from dataloader import Normalize, ToTensor
        from model import EventDetector
    finally:
        if sys.path[0] == str(event_dir):
            sys.path.pop(0)

    dataset = SampleVideo(
        video_path,
        transform=transforms.Compose(
            [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    probs = None
    with torch.no_grad():
        for sample in loader:
            images = sample["images"]
            batch = 0
            while batch * seq_length < images.shape[1]:
                start = batch * seq_length
                end = (batch + 1) * seq_length
                image_batch = images[:, start:end]
                logits = model(image_batch.to(device))
                batch_probs = F.softmax(logits.data, dim=1).cpu().numpy()
                probs = batch_probs if probs is None else np.append(probs, batch_probs, 0)
                batch += 1

    if probs is None:
        raise RuntimeError(f"SwingNet produced no probabilities for {video_path}")

    events = np.argmax(probs, axis=0)[:-1]
    confidence = [float(probs[event_frame, event_idx]) for event_idx, event_frame in enumerate(events)]
    return [int(e) for e in events], confidence


class MediaPipePoseBackend:
    name = "mediapipe"

    def __init__(self):
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )

    def extract(self, frame_bgr) -> dict[str, tuple[float, float, float] | None] | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        raw = result.pose_landmarks.landmark
        landmarks = {}
        for name, idx in LANDMARK_NAMES.items():
            point = raw[idx]
            if point.visibility < 0.5:
                landmarks[name] = None
            else:
                landmarks[name] = (
                    float(point.x * width),
                    float(point.y * height),
                    float(point.visibility),
                )
        return landmarks

    def close(self):
        self._pose.close()


class YoloPoseBackend:
    name = "yolopose"

    def __init__(self):
        from ultralytics import YOLO

        self._model = YOLO("yolov8n-pose.pt")

    def extract(self, frame_bgr) -> dict[str, tuple[float, float, float] | None] | None:
        results = self._model(frame_bgr, verbose=False)
        if not results:
            return None
        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) > 0:
            person_idx = int(result.boxes.conf.detach().cpu().numpy().argmax())
        else:
            person_idx = 0

        xy = result.keypoints.xy[person_idx].detach().cpu().numpy()
        if result.keypoints.conf is not None:
            conf = result.keypoints.conf[person_idx].detach().cpu().numpy()
        else:
            conf = np.ones((xy.shape[0],), dtype=float)

        landmarks = {}
        for name, idx in YOLO_LANDMARK_NAMES.items():
            if idx >= xy.shape[0] or float(conf[idx]) < 0.3:
                landmarks[name] = None
            else:
                landmarks[name] = (float(xy[idx][0]), float(xy[idx][1]), float(conf[idx]))
        return landmarks

    def close(self):
        pass


def calculate_angle(p1, p2, p3) -> float | None:
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom <= 1e-8:
        return None
    cos = np.dot(v1, v2) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compute_features(landmarks: dict | None, handedness: str) -> dict[str, float | None]:
    features = {name: None for name in REQUIRED_FEATURES}
    if landmarks is None:
        return features

    if handedness == "right":
        trail, lead = "RIGHT", "LEFT"
    else:
        trail, lead = "LEFT", "RIGHT"

    def get(name: str):
        return landmarks.get(name)

    features["trail_elbow_angle"] = calculate_angle(
        get(f"{trail}_SHOULDER"), get(f"{trail}_ELBOW"), get(f"{trail}_WRIST")
    )
    features["lead_elbow_angle"] = calculate_angle(
        get(f"{lead}_SHOULDER"), get(f"{lead}_ELBOW"), get(f"{lead}_WRIST")
    )

    ls = get("LEFT_SHOULDER")
    rs = get("RIGHT_SHOULDER")
    lh = get("LEFT_HIP")
    rh = get("RIGHT_HIP")

    if ls and rs:
        features["shoulder_tilt"] = float(np.degrees(np.arctan2(ls[1] - rs[1], rs[0] - ls[0])))
    if lh and rh:
        features["hip_tilt"] = float(np.degrees(np.arctan2(lh[1] - rh[1], rh[0] - lh[0])))
    if ls and rs and lh and rh:
        mid_s = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        mid_h = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        dy = mid_h[1] - mid_s[1]
        dx = mid_h[0] - mid_s[0]
        features["spine_tilt"] = float(np.degrees(np.arctan2(dx, dy)))

    return features


def grade_from_z(z_score: float | None) -> str:
    if z_score is None:
        return "unavailable"
    magnitude = abs(z_score)
    if magnitude <= 1.0:
        return "Excellent"
    if magnitude <= 2.0:
        return "Good"
    return "Needs Improvement"


def circular_delta(value: float, center: float) -> float:
    return float((value - center + 180.0) % 360.0 - 180.0)


def feedback_for_event(
    event_idx: int,
    features: dict[str, float | None],
    stats: dict[str, Any],
    reliability_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_stats = stats.get(str(event_idx), {})
    event_reliability_stats = (reliability_stats or {}).get(str(event_idx), {})
    feedback = {}
    for feature_name in REQUIRED_FEATURES:
        value = features.get(feature_name)
        base_stat = event_stats.get(feature_name)
        reliability_stat = event_reliability_stats.get(feature_name)
        stat = reliability_stat or base_stat
        reliability = (reliability_stat or {}).get("reliability", "unknown")
        feedback_strength = (reliability_stat or {}).get("feedback_strength", "normal")
        reliability_reason = (reliability_stat or {}).get(
            "reliability_reason", "reliability stats not available"
        )
        item = {
            "value": value,
            "pro_mean": None,
            "pro_std": None,
            "z_score": None,
            "grade": "unavailable",
            "reliability": reliability,
            "feedback_strength": feedback_strength,
            "reliability_reason": reliability_reason,
            "weighted_score": None,
            "message": "",
        }
        if value is None:
            item["message"] = "Feature unavailable because required landmarks were not detected."
        elif stat is None:
            item["message"] = "Pro reference value is missing for this event and feature."
        else:
            mean = stat.get("mean")
            std = stat.get("std")
            item["pro_mean"] = mean
            item["pro_std"] = std
            if mean is None or std is None or std <= 0:
                item["message"] = "Pro reference std is missing or zero."
            else:
                if stat.get("stat_type") == "circular_degrees" or feature_name in CIRCULAR_FEATURES:
                    diff = circular_delta(value, mean)
                else:
                    diff = value - mean
                z_score = float(diff / std)
                item["z_score"] = z_score
                item["grade"] = grade_from_z(z_score)
                base_score = max(0.0, 100.0 - abs(z_score) * 30.0)
                item["weighted_score"] = float(base_score * RELIABILITY_WEIGHT.get(reliability, 0.5))
                if reliability == "low":
                    item["feedback_strength"] = "reference"
                item["message"] = (
                    f"{feature_name}: {item['grade']} "
                    f"(value={value:.2f}, pro_mean={mean:.2f}, pro_std={std:.2f}, "
                    f"z={z_score:.2f}, reliability={item['reliability']}, "
                    f"strength={item['feedback_strength']})"
                )
                if reliability == "low" and item["grade"] == "Needs Improvement":
                    item["message"] += " Low reliability: treat this as a reference signal, not a strong correction."
        feedback[feature_name] = item
    return feedback


def count_features(features: dict[str, float | None]) -> int:
    return sum(1 for name in REQUIRED_FEATURES if features.get(name) is not None)


def select_backend(
    requested_backend: str,
    media_result: dict[str, Any] | None,
    yolo_result: dict[str, Any] | None,
) -> str:
    if requested_backend == "mediapipe":
        return "mediapipe"
    if requested_backend == "yolopose":
        return "yolopose"

    media_count = count_features(media_result["features"]) if media_result else -1
    yolo_count = count_features(yolo_result["features"]) if yolo_result else -1
    if yolo_count > media_count:
        return "yolopose"
    return "mediapipe"


def init_backend(name: str, notes: list[str]):
    try:
        if name == "mediapipe":
            return MediaPipePoseBackend()
        if name == "yolopose":
            return YoloPoseBackend()
    except Exception as exc:
        notes.append(f"{name} backend unavailable: {type(exc).__name__}: {exc}")
    return None


def analyze_pose_backend(backend, frame, handedness: str) -> dict[str, Any]:
    landmarks = backend.extract(frame) if backend else None
    features = compute_features(landmarks, handedness)
    return {
        "pose_detected": landmarks is not None,
        "landmarks": landmarks,
        "features": features,
        "computed_feature_count": count_features(features),
    }


def safe_frame_name(event_idx: int, event_name: str) -> str:
    safe = event_name.lower().replace(" ", "_").replace("-", "_")
    return f"{event_idx}_{safe}.jpg"


SKELETON_EDGES = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]


def load_reference_skeletons(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_stats(path)


def reference_event(reference_skeletons: dict[str, Any] | None, event_idx: int) -> dict[str, Any] | None:
    if not reference_skeletons:
        return None
    events = reference_skeletons.get("events", reference_skeletons)
    return events.get(str(event_idx)) or events.get(EVENT_NAMES[event_idx])


def normalize_landmarks(
    landmarks: dict[str, tuple[float, float, float] | list[float] | None] | None,
) -> dict[str, list[float] | None] | None:
    if not landmarks:
        return None

    def point(name: str):
        value = landmarks.get(name)
        if value is None or len(value) < 2:
            return None
        return float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else 1.0

    ls = point("LEFT_SHOULDER")
    rs = point("RIGHT_SHOULDER")
    lh = point("LEFT_HIP")
    rh = point("RIGHT_HIP")
    visible = [point(name) for name in LANDMARK_NAMES if point(name) is not None]
    if len(visible) < 4:
        return None

    if lh and rh:
        center = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    else:
        center = (
            (min(p[0] for p in visible) + max(p[0] for p in visible)) / 2.0,
            (min(p[1] for p in visible) + max(p[1] for p in visible)) / 2.0,
        )

    scale_candidates = []
    if ls and rs:
        scale_candidates.append(float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
    if lh and rh:
        scale_candidates.append(float(np.hypot(lh[0] - rh[0], lh[1] - rh[1])))
    if ls and rs and lh and rh:
        mid_s = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        mid_h = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        scale_candidates.append(float(np.hypot(mid_s[0] - mid_h[0], mid_s[1] - mid_h[1])))
    scale_candidates.append(max(max(p[0] for p in visible) - min(p[0] for p in visible), 1.0))
    scale_candidates.append(max(max(p[1] for p in visible) - min(p[1] for p in visible), 1.0))
    scale = max(scale_candidates)
    if scale <= 1e-6:
        return None

    normalized: dict[str, list[float] | None] = {}
    for name in LANDMARK_NAMES:
        p = point(name)
        if p is None:
            normalized[name] = None
        else:
            normalized[name] = [(p[0] - center[0]) / scale, (p[1] - center[1]) / scale, p[2]]
    return normalized


def _pixel_point(value):
    if value is None or len(value) < 2:
        return None
    return int(round(float(value[0]))), int(round(float(value[1])))


def draw_pixel_skeleton(
    frame_bgr,
    landmarks: dict[str, tuple[float, float, float] | list[float] | None] | None,
    color: tuple[int, int, int],
    thickness: int = 2,
    radius: int = 4,
) -> None:
    if not landmarks:
        return
    for start, end in SKELETON_EDGES:
        p1 = _pixel_point(landmarks.get(start))
        p2 = _pixel_point(landmarks.get(end))
        if p1 and p2:
            cv2.line(frame_bgr, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
    for name in LANDMARK_NAMES:
        p = _pixel_point(landmarks.get(name))
        if p:
            cv2.circle(frame_bgr, p, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def normalized_to_canvas(
    skeletons: list[dict[str, list[float] | None]],
    width: int,
    height: int,
    margin: int = 48,
) -> dict[str, float]:
    points = []
    for skeleton in skeletons:
        for value in skeleton.values():
            if value is not None and len(value) >= 2:
                points.append((float(value[0]), float(value[1])))
    if not points:
        return {"scale": 1.0, "offset_x": width / 2, "offset_y": height / 2}

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((width - margin * 2) / span_x, (height - margin * 2) / span_y)
    return {
        "scale": scale,
        "offset_x": (width - (min_x + max_x) * scale) / 2,
        "offset_y": (height - (min_y + max_y) * scale) / 2,
    }


def draw_normalized_skeleton(
    canvas,
    landmarks: dict[str, list[float] | None],
    transform: dict[str, float],
    color: tuple[int, int, int],
    thickness: int = 2,
    radius: int = 4,
) -> None:
    def cv_point(name: str):
        value = landmarks.get(name)
        if value is None or len(value) < 2:
            return None
        x = float(value[0]) * transform["scale"] + transform["offset_x"]
        y = float(value[1]) * transform["scale"] + transform["offset_y"]
        return int(round(x)), int(round(y))

    for start, end in SKELETON_EDGES:
        p1 = cv_point(start)
        p2 = cv_point(end)
        if p1 and p2:
            cv2.line(canvas, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
    for name in LANDMARK_NAMES:
        p = cv_point(name)
        if p:
            cv2.circle(canvas, p, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def write_skeleton_comparison_image(
    frame_bgr,
    user_landmarks: dict[str, tuple[float, float, float] | list[float] | None] | None,
    reference: dict[str, Any] | None,
    output_path: Path,
    event_name: str,
) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    user_normalized = normalize_landmarks(user_landmarks)
    reference_landmarks = (reference or {}).get("landmarks")

    overlay = frame_bgr.copy()
    draw_pixel_skeleton(overlay, user_landmarks, color=(0, 80, 255))
    target_height = 480
    scale = target_height / max(overlay.shape[0], 1)
    overlay = cv2.resize(overlay, (int(overlay.shape[1] * scale), target_height))

    comparison = np.full((target_height, 480, 3), 255, dtype=np.uint8)
    cv2.putText(comparison, event_name, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.putText(comparison, "user", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
    cv2.putText(comparison, "pro reference", (110, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 150, 50), 2)

    notes = []
    skeletons = []
    if user_normalized:
        skeletons.append(user_normalized)
    else:
        notes.append("user skeleton unavailable")
    if reference_landmarks:
        skeletons.append(reference_landmarks)
    else:
        notes.append("pro reference unavailable")

    if skeletons:
        transform = normalized_to_canvas(skeletons, comparison.shape[1], comparison.shape[0], margin=70)
        if reference_landmarks:
            draw_normalized_skeleton(comparison, reference_landmarks, transform, color=(30, 150, 50), thickness=2, radius=4)
        if user_normalized:
            draw_normalized_skeleton(comparison, user_normalized, transform, color=(0, 80, 255), thickness=2, radius=4)

    for i, note in enumerate(notes):
        cv2.putText(comparison, note, (24, 110 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)

    cv2.putText(overlay, "User keyframe", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    merged = np.concatenate([overlay, comparison], axis=1)
    cv2.imwrite(str(output_path), merged)
    return not notes, "; ".join(notes)


def write_outputs(result: dict[str, Any], json_path: Path, txt_path: Path) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [
        "Golf posture analysis summary",
        f"Video: {result.get('video_path')}",
        f"Mock events: {result.get('mock_events')}",
        f"Handedness: {result.get('handedness')}",
        f"Pose backend request: {result.get('pose_backend')}",
    ]
    if result.get("event_frame_dir"):
        lines.append(f"Event frames: {result.get('event_frame_dir')}")
    if result.get("skeleton_comparison_dir"):
        lines.append(f"Skeleton comparisons: {result.get('skeleton_comparison_dir')}")
    if result.get("errors"):
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result["errors"])
    comparison = result.get("pose_backend_comparison", {})
    if comparison:
        lines.append(f"Selected backend: {comparison.get('selected_backend')}")
        lines.append(f"Selection reason: {comparison.get('selection_reason')}")

    for event_name, event in result.get("events", {}).items():
        lines.append("")
        lines.append(
            f"{event_name}: frame={event.get('frame_index')} "
            f"pose={event.get('pose_detected')} backend={event.get('selected_pose_backend')}"
        )
        for feature_name, item in event.get("feedback", {}).items():
            z = item.get("z_score")
            z_text = "n/a" if z is None or math.isnan(z) else f"{z:.2f}"
            lines.append(
                f"  - {feature_name}: {item.get('grade')} "
                f"value={item.get('value')} z={z_text} "
                f"reliability={item.get('reliability')} strength={item.get('feedback_strength')}"
            )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    video_path = Path(args.video)
    stats_path = Path(args.pro_stats)
    reliability_stats_path = Path(args.reliability_stats)
    reference_skeletons_path = Path(args.reference_skeletons)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else repo_root() / "event_detection" / "models" / "swingnet_1800.pth.tar"
    )

    result: dict[str, Any] = {
        "video_path": str(video_path),
        "mock_events": bool(args.mock_events),
        "handedness": args.handedness,
        "pose_backend": args.pose_backend,
        "checkpoint_path": None if args.mock_events else str(checkpoint_path),
        "pro_stats_path": str(stats_path),
        "reliability_stats_path": str(reliability_stats_path),
        "reference_skeletons_path": str(reference_skeletons_path),
        "reference_skeletons_available": False,
        "events": {},
        "pose_backend_comparison": {
            "mediapipe": {"attempted": False, "pose_detected_events": 0, "notes": []},
            "yolopose": {"attempted": False, "pose_detected_events": 0, "notes": []},
            "selected_backend": None,
            "selection_reason": "",
        },
        "errors": [],
    }

    if not video_path.exists():
        result["errors"].append(f"Input video not found: {video_path}")
        return result, 1

    try:
        stats = load_stats(stats_path)
    except Exception as exc:
        result["errors"].append(str(exc))
        return result, 1
    reliability_stats = load_optional_stats(reliability_stats_path)
    reference_skeletons = load_reference_skeletons(reference_skeletons_path)
    if reference_skeletons is None:
        result["errors"].append(
            f"Pro reference skeletons not found: {reference_skeletons_path}. "
            "Skeleton comparison images will include the user skeleton only."
        )
    else:
        result["reference_skeletons_available"] = True

    try:
        if args.mock_events:
            event_frames, confidence = mock_event_frames(video_path)
        else:
            event_frames, confidence = detect_swingnet_events(video_path, checkpoint_path, args.seq_length)
    except Exception as exc:
        result["errors"].append(str(exc))
        return result, 1

    attempt_media = args.pose_backend in {"mediapipe", "auto", "hybrid"}
    attempt_yolo = args.pose_backend in {"yolopose", "auto", "hybrid"}
    media_backend = None
    yolo_backend = None
    if attempt_media:
        result["pose_backend_comparison"]["mediapipe"]["attempted"] = True
        media_backend = init_backend("mediapipe", result["pose_backend_comparison"]["mediapipe"]["notes"])
    if attempt_yolo:
        result["pose_backend_comparison"]["yolopose"]["attempted"] = True
        yolo_backend = init_backend("yolopose", result["pose_backend_comparison"]["yolopose"]["notes"])

    selected_counts = {"mediapipe": 0, "yolopose": 0}
    _, _, frame_dir, skeleton_dir = output_paths(args.output)
    result["event_frame_dir"] = str(frame_dir)
    result["skeleton_comparison_dir"] = str(skeleton_dir)

    try:
        for event_idx, event_name in enumerate(EVENT_NAMES):
            frame_index = int(event_frames[event_idx])
            frame = read_frame(video_path, frame_index)
            skeleton_path = skeleton_dir / safe_frame_name(event_idx, event_name)
            reference = reference_event(reference_skeletons, event_idx)
            if frame is None:
                result["events"][event_name] = {
                    "frame_index": frame_index,
                    "confidence": confidence[event_idx],
                    "pose_detected": False,
                    "selected_pose_backend": None,
                    "features": {name: None for name in REQUIRED_FEATURES},
                    "feedback": feedback_for_event(
                        event_idx,
                        {name: None for name in REQUIRED_FEATURES},
                        stats,
                        reliability_stats,
                    ),
                    "pose_candidates": {},
                    "selected_landmarks": None,
                    "notes": ["Could not read event frame."],
                    "event_frame_path": str(frame_dir / safe_frame_name(event_idx, event_name)),
                    "skeleton_comparison_path": None,
                    "reference_skeleton_available": reference is not None,
                }
                continue

            cv2.imwrite(str(frame_dir / safe_frame_name(event_idx, event_name)), frame)

            media_result = analyze_pose_backend(media_backend, frame, args.handedness) if media_backend else None
            yolo_result = analyze_pose_backend(yolo_backend, frame, args.handedness) if yolo_backend else None

            if media_result and media_result["pose_detected"]:
                result["pose_backend_comparison"]["mediapipe"]["pose_detected_events"] += 1
            if yolo_result and yolo_result["pose_detected"]:
                result["pose_backend_comparison"]["yolopose"]["pose_detected_events"] += 1

            selected_backend = select_backend(args.pose_backend, media_result, yolo_result)
            selected = media_result if selected_backend == "mediapipe" else yolo_result
            if selected is None:
                selected = {
                    "pose_detected": False,
                    "features": {name: None for name in REQUIRED_FEATURES},
                    "computed_feature_count": 0,
                }
            selected_counts[selected_backend] = selected_counts.get(selected_backend, 0) + 1

            result["events"][event_name] = {
                "frame_index": frame_index,
                "confidence": confidence[event_idx],
                "pose_detected": bool(selected["pose_detected"]),
                "selected_pose_backend": selected_backend,
                "selected_landmarks": selected.get("landmarks"),
                "features": selected["features"],
                "feedback": feedback_for_event(event_idx, selected["features"], stats, reliability_stats),
                "pose_candidates": {
                    "mediapipe": media_result,
                    "yolopose": yolo_result,
                },
                "event_frame_path": str(frame_dir / safe_frame_name(event_idx, event_name)),
                "skeleton_comparison_path": str(skeleton_path),
                "reference_skeleton_available": reference is not None,
                "reference_skeleton_source": (reference or {}).get("source_video_id"),
            }
            ok, note = write_skeleton_comparison_image(
                frame,
                selected.get("landmarks"),
                reference,
                skeleton_path,
                event_name,
            )
            if note:
                result["events"][event_name].setdefault("notes", []).append(note)
            result["events"][event_name]["skeleton_comparison_complete"] = ok
    finally:
        if media_backend:
            media_backend.close()
        if yolo_backend:
            yolo_backend.close()

    if args.pose_backend == "hybrid":
        selected_backend = "hybrid"
        reason = "Hybrid mode selected the per-event backend with more computable required features."
    elif args.pose_backend == "auto":
        selected_backend = "yolopose" if selected_counts.get("yolopose", 0) > selected_counts.get("mediapipe", 0) else "mediapipe"
        reason = f"Auto mode selected by event count: {selected_counts}."
    else:
        selected_backend = args.pose_backend
        reason = f"Requested backend '{args.pose_backend}' was used."

    result["pose_backend_comparison"]["selected_backend"] = selected_backend
    result["pose_backend_comparison"]["selection_reason"] = reason
    return result, 0


def main() -> int:
    args = parse_args()
    json_path, txt_path, _, _ = output_paths(args.output)
    result, exit_code = run(args)
    write_outputs(result, json_path, txt_path)
    print(f"JSON result: {json_path}")
    print(f"Text summary: {txt_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
