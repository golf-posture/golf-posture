"""Opt-in SwingNet subject crop and grouped-training phase refinement.

The public SwingNet weights stay fixed. Scores reported to callers are raw
centered SwingNet class probabilities, not calibrated ranker confidence.
"""
import hashlib
import json
from contextlib import closing
from pathlib import Path
import time

from .event_refinement_core import anchor_base, centered_scores, predict_one
from .event_selection import select_event_frames
from .event_subject import select_golfer, square_crop
from .swingnet import fp32, load_model, normalized, prepare_swingnet_frame
from .video_io import iter_video_batches


CHECKPOINT_SHA256 = '6331e303a9e86d0c19f183899f958bf2a71cf5a7070d46899e25e1ac877b23d4'
RANKER_SHA256 = '5e4cfc9fd6e1d52033b12cc933e12af67713166d1076cca2559a2b58c3fd529f'
ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def decode_refined_events(centered, points, boxes, model, missing_events=(), *, check_cancelled=None):
    """Decode complete swings; explicitly missing events use active-class order."""
    check = check_cancelled or (lambda: None)
    check()
    if missing_events:
        frames, confidence = select_event_frames(centered, 'ordered', missing_events)
        check()
        return frames, confidence, 'centered_ordered_declared_missing'
    base = anchor_base(centered)
    check()
    frames, _ = predict_one(centered, points, boxes, base, model, check_cancelled=check)
    confidence = [float(centered[t, e]) for e, t in enumerate(frames)]
    return frames, confidence, 'phase_ranker_v1'


def detect_refined_events(video_path, checkpoint_path, missing_events=(), *, device=None,
                          frame_start=0, frame_count=None, on_progress=None, check_cancelled=None,
                          on_subject_pose=None):
    """Return source-video frame indices, model scores and auditable diagnostics.

    Decode twice to avoid retaining all native-resolution video frames. The
    optional interval is also used by the time-aligned evaluation runner.
    on_progress(stage, completed_frames, total_frames) reports per-pass work;
    total_frames is unknown until the first pass ends unless frame_count is set.
    Cancellation is cooperative between model batches and refinement steps.
    on_subject_pose(points, width, height) receives the selected golfer's COCO17
    observations in decoded, uncropped video coordinates after inference.
    """
    check = check_cancelled or (lambda: None)
    progress = on_progress or (lambda stage, completed, total: None)
    check()
    if frame_start < 0 or (frame_count is not None and frame_count <= 0):
        raise ValueError('Expected a nonnegative start and a positive frame count')
    if not set(missing_events).issubset(range(8)):
        raise ValueError('Missing event indices must be between 0 and 7')
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError('Phase refinement requires the verified public SwingNet checkpoint')
    ranker_path = Path(__file__).with_name('models') / 'swingnet_phase_ranker_v1.json'
    if file_sha256(ranker_path) != RANKER_SHA256:
        raise ValueError('Phase refinement model checksum mismatch')
    check()
    pose_path = ROOT / 'yolov8n-pose.pt'
    if not pose_path.exists():
        raise FileNotFoundError(f'Event refinement pose weights not found: {pose_path}')

    import torch
    from ultralytics import YOLO

    started = time.perf_counter()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)
    check()
    ranker = json.loads(ranker_path.read_text(encoding='utf-8'))
    pose = YOLO(str(pose_path))
    check()
    detections = []
    width = height = None
    with fp32(), torch.inference_mode():
        progress('locating_golfer', 0, frame_count)
        check()
        with closing(iter_video_batches(video_path, 16, frame_start, frame_count)) as batches:
            for batch in batches:
                check()
                height, width = batch[0].shape[:2]
                results = pose.predict(batch, device=device, imgsz=640, conf=.25,
                                       half=False, verbose=False, save=False)
                for result in results:
                    check()
                    detections.append({'boxes': result.boxes.xyxy.cpu().tolist(),
                                       'confidence': result.boxes.conf.cpu().tolist(),
                                       'points': result.keypoints.data.cpu().tolist()})
                check()
                progress('locating_golfer', len(detections), frame_count)
                check()
        if not detections:
            raise ValueError('Video interval has no readable frames')
        if frame_count is not None and len(detections) != frame_count:
            raise ValueError(f'Decoded {len(detections)}, expected {frame_count} frames')
        total = len(detections)
        progress('selecting_golfer', total, total)
        check()
        points, boxes, subject = select_golfer(detections, width, height, check_cancelled=check)
        check()
        crop = square_crop(points, boxes, subject, width, height)
        check()
        left, top, right, bottom = crop['xyxy']
        parts = []
        processed = 0
        progress('extracting_features', 0, total)
        check()
        with closing(iter_video_batches(video_path, 64, frame_start, frame_count)) as batches:
            for batch in batches:
                check()
                images = [prepare_swingnet_frame(frame[top:bottom, left:right]) for frame in batch]
                x = normalized(images).to(device)
                check()
                parts.append(model.cnn(x).mean(3).mean(2).cpu())
                check()
                processed += len(batch)
                progress('extracting_features', processed, total)
                check()
        features = torch.cat(parts)
        if len(features) != len(detections):
            raise ValueError('Frame count changed between the pose and SwingNet passes')
        progress('detecting', 0, total)
        centered = centered_scores(model, features, device, check_cancelled=check,
                                   on_progress=lambda count: progress('detecting', count, total))
        progress('refining', total, total)
        frames, confidence, selection = decode_refined_events(
            centered, points, boxes, ranker, missing_events, check_cancelled=check)
        check()
    frames = [None if f is None else f + frame_start for f in frames]
    metadata = {'method': selection, 'device': str(device), 'frames': len(detections),
                'frame_start': frame_start, 'indices': 'zero_based_source_video_frames',
                'subject': subject, 'crop': crop, 'input_size': 160, 'context_frames': 64,
                'anchor_events': [4, 5, 6] if selection == 'phase_ranker_v1' else [],
                'score_source': 'centered_SwingNet_class_probability_at_selected_frame',
                'score_is_calibrated_accuracy': False, 'checkpoint_sha256': CHECKPOINT_SHA256,
                'ranker_sha256': RANKER_SHA256, 'pose_sha256': file_sha256(pose_path),
                'seconds': time.perf_counter() - started}
    check()
    if on_subject_pose is not None:
        on_subject_pose(points, width, height)
        check()
    return frames, confidence, metadata
