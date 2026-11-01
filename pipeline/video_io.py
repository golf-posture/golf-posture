"""Decode model and pose inputs with the same metadata-aware FFmpeg runtime."""
from __future__ import annotations

from contextlib import closing
from itertools import islice
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def iter_video_frames(video_path: Path):
    # Older OpenCV FFmpeg builds ignore BT.709 and convert as BT.601.
    # Keep source color metadata and frame order; do not resample to a fixed FPS.
    with closing(imageio_ffmpeg.read_frames(
        str(video_path), pix_fmt="bgr24", output_params=["-fps_mode", "passthrough"]
    )) as frames:
        metadata = next(frames)
        width, height = metadata["size"]
        for frame in frames:
            yield np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3).copy()


def read_video_frame(video_path: Path, frame_index: int):
    with closing(iter_video_frames(video_path)) as frames:
        for index, frame in enumerate(frames):
            if index == frame_index:
                return frame
    return None


def iter_video_batches(video_path: Path, batch_size: int, frame_start=0, frame_count=None):
    """Read consecutive batches from a source-frame interval and close its decoder."""
    stop = None if frame_count is None else frame_start + frame_count
    with closing(iter_video_frames(video_path)) as frames:
        selected = islice(frames, frame_start, stop)
        while batch := list(islice(selected, batch_size)):
            yield batch
