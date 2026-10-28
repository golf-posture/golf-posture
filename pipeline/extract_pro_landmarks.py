"""Extract per-event landmark coordinates (and the usual angle features) for the
342 GolfDB pro videos WITHOUT keeping the videos on disk.

Each video is downloaded to a temp file, its 8 labelled event frames are run
through the pose backend, the result rows are appended to the output CSV, and
the temp file is deleted before the next video. Peak disk use is one video.

The script can resume: video_ids already present in the output CSV are skipped.

Usage (from the repo root, inside the pixi env):

  # videos live in a private GitHub repo -> give a read-only token
  set GITHUB_TOKEN=github_pat_xxx            (Windows cmd)
  $env:GITHUB_TOKEN="github_pat_xxx"         (PowerShell)
  pixi run python -m pipeline.extract_pro_landmarks --limit 5      # smoke test
  pixi run python -m pipeline.extract_pro_landmarks                # all videos

  # or, if the videos are already on disk somewhere:
  pixi run python -m pipeline.extract_pro_landmarks --videos-dir D:\\golfdb_720p_slice

Output: data/pro_landmarks_raw.csv  (does NOT overwrite data/pro_feature_raw.csv)
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from pipeline.run_analysis import (
    EVENT_NAMES,
    LANDMARK_NAMES,
    REQUIRED_FEATURES,
    analyze_pose_backend,
    init_backend,
    repo_root,
)

DEFAULT_REPO = "golf-posture/golfdb-master"
DEFAULT_BRANCH = "main"
DEFAULT_PREFIX = "learning_video/golfdb_720p_slice"

LANDMARK_COLUMNS = [f"lm_{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract landmark coordinates for GolfDB pro videos (streaming).")
    p.add_argument("--labels", default="event_detection/data/golfDB_labels.csv")
    p.add_argument("--output", default="data/pro_landmarks_raw.csv")
    p.add_argument("--videos-dir", help="Use local videos named <id>.mp4 instead of downloading.")
    p.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo that holds the videos (owner/name).")
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="Folder inside the repo.")
    p.add_argument("--token-env", default="GITHUB_TOKEN", help="Env var holding a read-only GitHub token.")
    p.add_argument("--handedness", choices=["right", "left"], default="right")
    p.add_argument("--pose-backend", choices=["mediapipe", "yolopose"], default="mediapipe")
    p.add_argument("--limit", type=int, help="Only process the first N videos (smoke test).")
    return p.parse_args()


def read_labels(path: Path) -> dict[int, list[int]]:
    labels: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = [int(part) for part in row["events"].split("|") if part]
            if len(raw) >= 10:
                labels[int(row["id"])] = raw
    return labels


def already_done(output: Path) -> set[int]:
    if not output.exists():
        return set()
    with output.open("r", encoding="utf-8", newline="") as f:
        return {int(r["video_id"]) for r in csv.DictReader(f) if r.get("video_id")}


def download_video(video_id: int, args: argparse.Namespace, token: str, dest: Path) -> None:
    url = f"https://api.github.com/repos/{args.repo}/contents/{args.prefix}/{video_id}.mp4?ref={args.branch}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "golf-posture-extractor",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def open_capture(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, 0
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def read_frame_from(cap, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def landmark_cells(landmarks) -> dict[str, float | str]:
    cells: dict[str, float | str] = {}
    for name in LANDMARK_NAMES:
        point = (landmarks or {}).get(name)
        cells[f"lm_{name}_x"] = "" if point is None else point[0]
        cells[f"lm_{name}_y"] = "" if point is None else point[1]
    return cells


def process_video(video_id: int, video_path: Path, raw_events: list[int], backend, args) -> list[dict]:
    start_frame = raw_events[0]
    event_frames = [frame - start_frame for frame in raw_events[1:-1]]
    cap, total_frames = open_capture(video_path)
    rows: list[dict] = []
    try:
        for event_index, event_name in enumerate(EVENT_NAMES):
            frame_index = int(event_frames[event_index])
            base = {
                "video_id": video_id,
                "event_index": event_index,
                "event_name": event_name,
                "frame_index": frame_index,
                "pose_backend": args.pose_backend,
            }
            empty = {name: "" for name in REQUIRED_FEATURES}
            empty_lm = landmark_cells(None)
            if cap is None or frame_index < 0 or frame_index >= total_frames:
                reason = "video_unreadable" if cap is None else f"frame_out_of_range:total_frames={total_frames}"
                rows.append({**base, "pose_detected": False, "frame_width": "", "frame_height": "",
                             **empty, **empty_lm, "failure_reason": reason})
                continue
            frame = read_frame_from(cap, frame_index)
            if frame is None:
                rows.append({**base, "pose_detected": False, "frame_width": "", "frame_height": "",
                             **empty, **empty_lm, "failure_reason": "frame_read_failed"})
                continue
            height, width = frame.shape[:2]
            result = analyze_pose_backend(backend, frame, args.handedness)
            features = {k: ("" if v is None else v) for k, v in result["features"].items()}
            rows.append({
                **base,
                "pose_detected": bool(result["pose_detected"]),
                "frame_width": width,
                "frame_height": height,
                **{name: features.get(name, "") for name in REQUIRED_FEATURES},
                **landmark_cells(result["landmarks"]),
                "failure_reason": "" if result["pose_detected"] else "pose_not_detected",
            })
    finally:
        if cap is not None:
            cap.release()
    return rows


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        labels_path = repo_root() / args.labels
    labels = read_labels(labels_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(output)
    video_ids = [vid for vid in sorted(labels) if vid not in done]
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    token = os.environ.get(args.token_env, "")
    if not args.videos_dir and not token:
        raise SystemExit(
            f"No local --videos-dir and no token in env var {args.token_env}. "
            "Create a read-only GitHub token and set it, or pass --videos-dir."
        )

    notes: list[str] = []
    backend = init_backend(args.pose_backend, notes)
    if backend is None:
        raise SystemExit("; ".join(notes) or f"Could not initialize {args.pose_backend}")

    fieldnames = [
        "video_id", "event_index", "event_name", "frame_index", "pose_backend", "pose_detected",
        "frame_width", "frame_height", *REQUIRED_FEATURES, *LANDMARK_COLUMNS, "failure_reason",
    ]
    new_file = not output.exists() or output.stat().st_size == 0
    print(f"{len(done)} videos already done, {len(video_ids)} to process.")

    processed = skipped = 0
    started = time.time()
    try:
        with output.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            for i, video_id in enumerate(video_ids, start=1):
                tmp_path = None
                try:
                    if args.videos_dir:
                        video_path = Path(args.videos_dir) / f"{video_id}.mp4"
                        if not video_path.exists():
                            skipped += 1
                            print(f"[{i}/{len(video_ids)}] {video_id}: missing local file, skipped")
                            continue
                    else:
                        fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
                        os.close(fd)
                        tmp_path = Path(tmp_name)
                        try:
                            download_video(video_id, args, token, tmp_path)
                        except urllib.error.HTTPError as exc:
                            skipped += 1
                            print(f"[{i}/{len(video_ids)}] {video_id}: download failed (HTTP {exc.code}), skipped")
                            if exc.code in (401, 403):
                                print("  -> check the token permissions / rate limit; rerun later to resume.")
                            continue
                        video_path = tmp_path
                    rows = process_video(video_id, video_path, labels[video_id], backend, args)
                    writer.writerows(rows)
                    f.flush()
                    processed += 1
                    if processed % 10 == 0 or i == len(video_ids):
                        rate = (time.time() - started) / processed
                        print(f"[{i}/{len(video_ids)}] processed={processed} skipped={skipped} "
                              f"~{rate:.1f}s/video, ~{rate * (len(video_ids) - i) / 60:.0f} min left")
                finally:
                    if tmp_path is not None and tmp_path.exists():
                        tmp_path.unlink()
    finally:
        backend.close()

    print(f"Done. processed={processed} skipped={skipped}. Output: {output}")
    if skipped:
        print("Some videos were skipped. Rerun the same command to retry only the missing ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
