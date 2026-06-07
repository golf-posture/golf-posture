"""Extract MediaPipe pose overlays for GolfDB event frames.

This script reads GolfDB label CSV rows, opens the corresponding preprocessed
``videos_160/<id>.mp4`` clips, extracts the 8 key event frames, runs MediaPipe
Pose on each frame, and writes debug overlay images with the 13 landmarks used
by this project.

It is intentionally independent from ``event_detection`` code so pose analysis
can evolve as a separate module.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


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

PROJECT_LANDMARKS = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

PROJECT_POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
]


@dataclass(frozen=True)
class GolfDbRow:
    annotation_id: int
    youtube_id: str
    player: str
    slow: int
    raw_events: list[int]

    @property
    def key_events(self) -> list[int]:
        return self.raw_events[1:-1]

    @property
    def start_frame(self) -> int:
        return self.raw_events[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MediaPipe pose overlay images for GolfDB event frames."
    )
    parser.add_argument(
        "--labels",
        default="event_detection/data/golfDB_labels.csv",
        help="Path to GolfDB labels CSV.",
    )
    parser.add_argument(
        "--videos-dir",
        default="event_detection/data/videos_160",
        help="Directory containing preprocessed <id>.mp4 clips.",
    )
    parser.add_argument(
        "--output-dir",
        default="pose_analysis/debug_overlays",
        help="Directory where overlay images are written.",
    )
    parser.add_argument(
        "--landmarks-csv",
        default="pose_analysis/pose_landmarks_sample.csv",
        help="CSV path for extracted landmark coordinates. Set empty to disable.",
    )
    parser.add_argument(
        "--model",
        default="pose_analysis/models/pose_landmarker_full.task",
        help="Path to MediaPipe Pose Landmarker .task model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of label rows to process when --ids is not provided.",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        help="Specific GolfDB annotation IDs to process.",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        choices=EVENT_NAMES,
        help="Optional subset of event names to process.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum pose detection confidence.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=1,
        help="Skeleton line thickness in overlay images.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Landmark point radius in overlay images. Use 0 to hide points.",
    )
    return parser.parse_args()


def parse_pipe_ints(value: str) -> list[int]:
    return [int(part) for part in value.split("|") if part != ""]


def read_label_rows(labels_path: Path, limit: int, ids: set[int] | None) -> list[GolfDbRow]:
    rows: list[GolfDbRow] = []
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotation_id = int(row["id"])
            if ids is not None and annotation_id not in ids:
                continue
            raw_events = parse_pipe_ints(row["events"])
            if len(raw_events) < 10:
                print(f"Skipping id={annotation_id}: expected 10 raw events, got {len(raw_events)}")
                continue
            rows.append(
                GolfDbRow(
                    annotation_id=annotation_id,
                    youtube_id=row.get("youtube_id", ""),
                    player=row.get("player", ""),
                    slow=int(row.get("slow", 0)),
                    raw_events=raw_events,
                )
            )
            if ids is None and len(rows) >= limit:
                break
    return rows


def selected_events(event_names: list[str], requested: Iterable[str] | None) -> list[tuple[int, str]]:
    requested_set = set(requested) if requested else None
    selected = []
    for idx, event_name in enumerate(event_names):
        if requested_set is None or event_name in requested_set:
            selected.append((idx, event_name))
    return selected


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        cap.release()
        raise RuntimeError(
            f"Frame index out of range for {video_path.name}: {frame_idx} "
            f"(total_frames={total_frames})"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def draw_title(frame, text: str) -> None:
    height, width = frame.shape[:2]
    bar_height = max(28, height // 8)
    cv2.rectangle(frame, (0, 0), (width, bar_height), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        (8, min(bar_height - 8, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_pose_overlay(
    frame,
    landmarks,
    min_visibility: float = 0.2,
    line_thickness: int = 1,
    point_radius: int = 1,
) -> None:
    height, width = frame.shape[:2]
    points: dict[int, tuple[int, int] | None] = {}
    for landmark_idx in PROJECT_LANDMARKS:
        if landmark_idx >= len(landmarks):
            continue
        landmark = landmarks[landmark_idx]
        visibility = getattr(landmark, "visibility", 1.0)
        if visibility < min_visibility:
            points[landmark_idx] = None
            continue
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        points[landmark_idx] = (x, y)

    for start_idx, end_idx in PROJECT_POSE_CONNECTIONS:
        start = points.get(start_idx)
        end = points.get(end_idx)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, (80, 220, 120), line_thickness, cv2.LINE_AA)

    for point in points.values():
        if point is None:
            continue
        if point_radius > 0:
            cv2.circle(frame, point, point_radius, (0, 60, 255), thickness=-1, lineType=cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    labels_path = repo_root / args.labels
    videos_dir = repo_root / args.videos_dir
    output_dir = repo_root / args.output_dir
    landmarks_csv = repo_root / args.landmarks_csv if args.landmarks_csv else None
    model_path = repo_root / args.model

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}\n"
            "Download pose_landmarker_full.task into pose_analysis/models/ first."
        )

    # Import lazily so --help and syntax checks do not require MediaPipe.
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    output_dir.mkdir(parents=True, exist_ok=True)
    if landmarks_csv:
        landmarks_csv.parent.mkdir(parents=True, exist_ok=True)

    ids = set(args.ids) if args.ids else None
    rows = read_label_rows(labels_path, args.limit, ids)
    if not rows:
        print("No label rows selected.")
        return 1

    event_selection = selected_events(EVENT_NAMES, args.events)
    landmark_rows: list[dict[str, object]] = []
    overlays_written = 0

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_detection_confidence,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for row in rows:
            video_path = videos_dir / f"{row.annotation_id}.mp4"
            if not video_path.exists():
                print(f"Skipping id={row.annotation_id}: missing video {video_path}")
                continue

            video_output_dir = output_dir / str(row.annotation_id)
            video_output_dir.mkdir(parents=True, exist_ok=True)

            for event_idx, event_name in event_selection:
                raw_event_frame = row.key_events[event_idx]
                relative_frame = raw_event_frame - row.start_frame
                try:
                    frame_bgr = read_frame(video_path, relative_frame)
                except RuntimeError as exc:
                    print(f"Skipping id={row.annotation_id} {event_name}: {exc}")
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect(mp_image)
                overlay = frame_bgr.copy()

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    draw_pose_overlay(
                        overlay,
                        landmarks,
                        line_thickness=args.line_thickness,
                        point_radius=args.point_radius,
                    )
                    for landmark_idx, landmark_name in PROJECT_LANDMARKS.items():
                        if landmark_idx >= len(landmarks):
                            continue
                        landmark = landmarks[landmark_idx]
                        landmark_rows.append(
                            {
                                "id": row.annotation_id,
                                "youtube_id": row.youtube_id,
                                "player": row.player,
                                "slow": row.slow,
                                "event_index": event_idx,
                                "event_name": event_name,
                                "raw_event_frame": raw_event_frame,
                                "relative_frame": relative_frame,
                                "landmark_index": landmark_idx,
                                "landmark_name": landmark_name,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": landmark.visibility,
                            }
                        )
                else:
                    print(f"No pose detected: id={row.annotation_id}, event={event_name}")

                title = (
                    f"id={row.annotation_id} | {event_idx}_{event_name} | "
                    f"frame={relative_frame} | {row.player}"
                )
                draw_title(overlay, title)
                safe_event_name = event_name.replace(" ", "_").replace("-", "_")
                output_path = video_output_dir / (
                    f"{event_idx}_{safe_event_name}_f{relative_frame}.jpg"
                )
                cv2.imwrite(str(output_path), overlay)
                overlays_written += 1

    if landmarks_csv and landmark_rows:
        with landmarks_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(landmark_rows[0].keys()))
            writer.writeheader()
            writer.writerows(landmark_rows)

    print(f"selected_videos: {len(rows)}")
    print(f"overlays_written: {overlays_written}")
    if landmarks_csv:
        print(f"landmarks_csv: {landmarks_csv}")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
