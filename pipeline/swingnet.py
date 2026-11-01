"""Shared SwingNet preprocessing, model loading and fixed-window inference."""
from contextlib import closing, contextmanager
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from .event_selection import select_event_frames
from .video_io import iter_video_frames


def prepare_swingnet_frame(img: np.ndarray, input_size: int = 160) -> np.ndarray:
    """Letterbox a BGR frame with the ImageNet mean, then convert to RGB."""
    height, width = img.shape[:2]
    ratio = input_size / max(height, width)
    new_height, new_width = int(height * ratio), int(width * ratio)
    pad_h, pad_w = input_size - new_height, input_size - new_width
    resized = cv2.resize(img, (new_width, new_height))
    bordered = cv2.copyMakeBorder(
        resized, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT, value=[0.406 * 255, 0.456 * 255, 0.485 * 255],
    )
    return cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB)


class SampleVideo:
    """Single-video dataset retained for existing experiment callers."""

    def __init__(self, path: Path, input_size: int = 160, transform=None):
        self.path, self.input_size, self.transform = path, input_size, transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        images = [prepare_swingnet_frame(frame, self.input_size) for frame in iter_video_frames(self.path)]
        sample = {"images": np.asarray(images), "labels": np.zeros(len(images))}
        return self.transform(sample) if self.transform else sample


def normalized(frames):
    import torch

    x = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2)).float().div(255.)
    return x.sub_(torch.tensor([.485, .456, .406])[None, :, None, None]).div_(
        torch.tensor([.229, .224, .225])[None, :, None, None])


@contextmanager
def fp32():
    """Deterministic refinement/training precision; restore the caller's flags."""
    import torch

    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled, benchmark=False,
            benchmark_limit=torch.backends.cudnn.benchmark_limit,
            deterministic=True, allow_tf32=False,
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def load_model(checkpoint, device):
    import torch
    from event_detection.model import EventDetector

    model = EventDetector(pretrain=False, width_mult=1., lstm_layers=1,
                          lstm_hidden=256, bidirectional=True, dropout=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval()


def detect_swingnet_events(video_path: Path, checkpoint_path: Path, seq_length: int,
                           selection: str = "ordered", missing_events=()):
    """Score consecutive windows without retaining the entire decoded video."""
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SwingNet checkpoint not found: {checkpoint_path}. "
            "Pass --checkpoint or place swingnet_1800.pth.tar in event_detection/models."
        )
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "SwingNet inference requires torch. "
            "Install it or run with --mock-events for integration testing."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    probabilities = []
    # Baseline precision preserves benchmark/deterministic/matmul settings.
    # The stricter fp32() policy belongs to the opt-in refinement path.
    with closing(iter_video_frames(video_path)) as frames, torch.no_grad(), torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=torch.backends.cudnn.benchmark,
        benchmark_limit=torch.backends.cudnn.benchmark_limit,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=False,
    ):
        prepared = (prepare_swingnet_frame(frame) for frame in frames)
        while batch := list(islice(prepared, seq_length)):
            images = normalized(batch)
            # Preserve the legacy channels-last frame layout and its numerics.
            logits = model(images[None].to(device))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    if not probabilities:
        raise RuntimeError(f"SwingNet produced no probabilities for {video_path}")
    return select_event_frames(np.concatenate(probabilities), selection, missing_events)
