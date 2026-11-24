"""Pose-refiner answers computed once on a GPU machine and replayed anywhere as a `PoseSource`.

Rebuilding the pro reference statistics runs the refiner over every pro video (342 videos x 8 events x 5 frames).
ViTPose-H and YOLOv8x need a GPU for that to take minutes instead of hours, and the GPU build of torch lives in a
different environment from MediaPipe. So the models run first and save their answers; the MediaPipe side
(extract_pro_features_tracked --refiner-cache) then reads them through `CachedSource`.

    # in the GPU environment (torch + ultralytics + opencv, no MediaPipe needed)
    python -m pipeline.refiner_cache --table data/pro_feature_raw_tracked.csv --output outputs/refiner_cache_pro.json

The file is rewritten as it grows and videos already in it are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeline.pose_refiner import ALL_EVENTS, PoseSource, request_frames
from pipeline.pseudo_label import DEFAULT_RADIUS, load_event_frames

DEFAULT_TABLE = Path("data/pro_feature_raw_tracked.csv")
DEFAULT_VIDEOS_DIR = Path("learning_video/golfdb_720p_slice")
SAVE_EVERY = 10  # videos


def event_requests(events: Mapping[int, int], last_frame: int, radius: int = DEFAULT_RADIUS) -> dict[int, str]:
    """{frame: model} around every event of one video, up to the video's last frame."""
    return request_frames(events, (0, last_frame), radius, ALL_EVENTS)


class CachedSource:
    """PoseSource that answers from a saved cache; a video or frame that was never stored answers None."""

    def __init__(self, videos: Mapping[str, Mapping[str, Any]]):
        self._videos = videos

    @classmethod
    def load(cls, path: Path) -> "CachedSource":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["videos"])

    def joints(self, video_path: Path, requests: Mapping[int, str]) -> dict[int, Any]:
        stored = self._videos.get(Path(video_path).stem, {})
        return {frame: stored.get(str(frame)) for frame in requests}


def _save(path: Path, videos: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"videos": videos}), encoding="utf-8")


def build_cache(
    events_by_video: Mapping[str, Mapping[int, int]],
    videos_dir: Path,
    source: PoseSource,
    output: Path,
    frame_count: Callable[[Path], int],
    limit: int = 0,
) -> list[str]:
    """Run the source over every video and save its answers; returns the videos that failed."""
    videos: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))["videos"] if output.exists() else {}
    failed: list[str] = []
    chosen = list(events_by_video.items())[:limit] if limit else list(events_by_video.items())
    for done, (video_id, events) in enumerate(chosen, 1):
        if video_id in videos:
            continue
        path = Path(videos_dir) / f"{video_id}.mp4"
        try:
            requests = event_requests(events, frame_count(path) - 1)
            answers = source.joints(path, requests)
        except Exception as error:  # one unreadable video must not stop the other 341
            sys.stderr.write(f"video {video_id} failed: {type(error).__name__}: {error}\n")
            failed.append(video_id)
            continue
        videos[video_id] = {str(frame): None if joints is None else {n: list(v) for n, v in joints.items()}
                            for frame, joints in answers.items()}
        if done % SAVE_EVERY == 0:
            _save(output, videos)
            sys.stdout.write(f"{done}/{len(chosen)} videos\n")
            sys.stdout.flush()
    _save(output, videos)
    return failed


def video_frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="CSV with video_id, event_index, frame_index")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vitpose", type=Path, default=None, help="exported ViTPose-H (default: pose_source's)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N videos (0 = all)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from pipeline.pose_source import DEFAULT_VITPOSE, load_golfpose_source

    args = parse_args(argv)
    source = load_golfpose_source(args.vitpose or DEFAULT_VITPOSE)
    failed = build_cache(load_event_frames(args.table), args.videos_dir, source, args.output,
                         video_frame_count, args.limit)
    sys.stdout.write(f"saved: {args.output}  failed: {failed}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
