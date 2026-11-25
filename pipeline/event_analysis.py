"""Event detection and posture feedback on the source video timeline."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from pipeline.event_refinement import detect_refined_events
from pipeline.event_feedback import feedback_for_frames

MAX_SECONDS = 30
EVENT_NAMES = ("Address", "Toe-up", "Mid-backswing", "Top", "Mid-downswing",
               "Impact", "Mid-follow-through", "Finish")


class AnalysisCancelled(Exception):
    pass


class InvalidVideo(ValueError):
    pass


def probe_video(path: Path, check):
    # Prefer the ffprobe shipped in this Python environment, including on Windows.
    candidates = (Path(sys.prefix) / "Library/bin/ffprobe.exe",
                  Path(sys.prefix) / "bin/ffprobe")
    executable = next((str(p) for p in candidates if p.exists()), shutil.which("ffprobe"))
    if not executable:
        raise RuntimeError("ffprobe is required; run pixi install")
    check()
    metadata = _probe(executable, path, "format=start_time,duration:stream=start_time,duration", check)
    try:
        duration = float(metadata["format"]["duration"])
        origin = float(metadata["format"].get("start_time", 0))
        if not metadata.get("streams") or not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration")
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidVideo("영상의 길이를 확인할 수 없습니다.") from exc
    if duration > MAX_SECONDS:
        raise InvalidVideo("30초 이하의 영상을 선택해주세요.")
    data = _probe(executable, path, "frame=best_effort_timestamp_time,duration_time", check)
    try:
        frames = data["frames"]
        times = [float(f["best_effort_timestamp_time"]) - origin for f in frames]
        if len(times) < 8 or any(not math.isfinite(t) or t < 0 or t >= duration for t in times):
            raise ValueError("timestamps")
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("timestamp order")
        # Seek inside each frame's display interval, avoiding boundary rounding.
        ends = times[1:] + [min(duration, times[-1] + float(frames[-1].get("duration_time", duration - times[-1])))]
        targets = [(start + end) / 2 for start, end in zip(times, ends)]
        if any(not start <= target < duration for start, target in zip(times, targets)):
            raise ValueError("frame duration")
        return duration, targets, times
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidVideo("영상의 프레임 시각을 확인할 수 없습니다.") from exc


def _probe(executable, path, entries, check):
    command = [executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
               entries, "-of", "json", str(path)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        try:
            while True:
                check()
                try:
                    output, _ = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.kill()
            process.communicate()
            raise
    check()
    if process.returncode:
        raise InvalidVideo("영상을 디코딩할 수 없습니다. MP4 또는 MOV 파일을 확인해주세요.")
    try:
        return json.loads(output)
    except ValueError as exc:
        raise InvalidVideo("영상 정보를 읽을 수 없습니다.") from exc


def skeleton_result(points, width, height, frame_times, check):
    """Keep reliable observations only; never invent positions for missing joints."""
    if len(points) != len(frame_times):
        raise InvalidVideo("관절과 영상의 프레임 수가 일치하지 않습니다.")
    frames = []
    any_visible = False
    for pose in points:
        check()
        joints = []
        for x, y, confidence in pose:
            visible = (math.isfinite(float(x)) and math.isfinite(float(y)) and
                       math.isfinite(float(confidence)) and confidence >= .4 and
                       0 <= x <= width and 0 <= y <= height)
            joints.append([round(float(x) / width, 5), round(float(y) / height, 5)] if visible else None)
            any_visible |= bool(visible)
        frames.append(joints)
    if not any_visible:
        return None
    return {"format": "coco17", "width": width, "height": height,
            "frame_times": frame_times, "frames": frames}


def analyze_events(path: Path, checkpoint: Path, update, check):
    update(stage="preparing")
    duration, times, frame_times = probe_video(path, check)
    check()
    update(total_frames=len(times), processed_frames=0)
    processed = 0
    subject_pose = None

    def capture_pose(points, width, height):
        nonlocal subject_pose
        subject_pose = (points, width, height)

    def progress(stage, count, total):
        nonlocal processed
        check()
        if count > len(times) or (total is not None and total != len(times)):
            raise InvalidVideo("디코딩한 프레임 수와 영상 시각 정보가 일치하지 않습니다.")
        update(stage=stage, processed_frames=count)
        if stage == "detecting":
            processed = count

    frames, scores, details = detect_refined_events(
        path, checkpoint, on_progress=progress, check_cancelled=check, on_subject_pose=capture_pose)
    check()
    if processed != len(times) or details["frames"] != len(times):
        raise InvalidVideo("디코딩한 프레임 수와 영상 시각 정보가 일치하지 않습니다.")
    events = [{"id": i, "name": name, "frame_index": frame,
               "time_seconds": times[frame] if frame is not None else None,
               "model_score": scores[i], "status": "candidate" if frame is not None else "withheld"}
              for i, (name, frame) in enumerate(zip(EVENT_NAMES, frames))]
    skeleton = skeleton_result(*subject_pose, frame_times, check) if subject_pose is not None else None
    update(stage="feedback")
    feedback = feedback_for_frames(path, frames, check)
    update(stage="finalizing")
    return {"duration_seconds": duration, "total_frames": len(times), "events": events,
            "frame_times": frame_times, "last_frame_seek_seconds": times[-1], "skeleton": skeleton,
            "feedback": feedback}


def apply_event_frames(path: Path, frames, update, check):
    """Resolve the submitted event IDs on the uploaded video's presentation timeline."""
    update(stage="preparing_frames")
    duration, targets, starts = probe_video(path, check)
    total = len(targets)
    update(total_frames=total)
    events = []
    for index, (name, frame) in enumerate(zip(EVENT_NAMES, frames, strict=True)):
        check()
        if frame is not None and not 0 <= frame < total:
            raise InvalidVideo("영상 범위를 벗어난 이벤트 프레임입니다.")
        events.append({"id": index, "name": name, "frame_index": frame,
                       "time_seconds": targets[frame] if frame is not None else None,
                       "status": "candidate" if frame is not None else "withheld", "model_score": None})
    update(stage="feedback")
    feedback = feedback_for_frames(path, frames, check)
    update(stage="finalizing", processed_frames=total)
    return {"duration_seconds": duration, "total_frames": total, "frame_times": starts,
            "last_frame_seek_seconds": targets[-1], "events": events, "feedback": feedback}
