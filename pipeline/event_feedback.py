"""Build mobile feedback from the exact event slots resolved by the upload job."""
import logging
from pathlib import Path

from pipeline.app_feedback import build_app_payload
from pipeline.golfer_detection import YOLO_DEFAULT_WEIGHTS
from pipeline.pose_source import DEFAULT_VITPOSE, load_golfpose_source
from pipeline.run_analysis import (
    EVENT_NAMES, add_derived_features, evaluate_sequence_features, evaluate_swing_motion,
    feedback_for_event, load_stats, mahalanobis_combined_score, repo_root,
)
from pipeline.user_tracking import analyze_user_video


def feedback_for_frames(path: Path, frames, check):
    check()
    root = repo_root()
    data = root / "data"
    vitpose, yolo = root / DEFAULT_VITPOSE, root / YOLO_DEFAULT_WEIGHTS
    source = None
    if any(frame is None for frame in frames):
        reason = "missing_events"
    elif not vitpose.is_file() or not yolo.is_file():
        reason = "missing_weights"
    else:
        reason = "no_refined_frames"
        try:
            source = load_golfpose_source(vitpose, yolo)
        except Exception:
            logging.getLogger(__name__).exception("Could not initialize GolfPose")
            reason = "inference_failed"
    handedness = "right"
    poses = analyze_user_video(path, frames, handedness, refiner=source,
                               check_cancelled=check, legacy_fallback=True)
    check()
    qc = poses[0]["qc"]
    refined_count = qc.get("refined_frames", 0)
    if qc.get("pose_refiner", "").startswith("failed:"):
        reason = "inference_failed"
        logging.getLogger(__name__).warning("GolfPose fallback: %s", qc["pose_refiner"])
    reference = data / "refined" if refined_count else data
    angles = load_stats(data / "pro_angle_stats.json")
    reliability = load_stats(reference / "pro_feature_stats_reliability.json")
    covariance = load_stats(reference / "pro_feature_covariance.json")
    events = {}
    for i, (name, frame, pose) in enumerate(zip(EVENT_NAMES, frames, poses, strict=True)):
        check()
        features = add_derived_features(pose["features"])
        events[name] = {
            "frame_index": frame, "features": features, "selected_landmarks": pose["landmarks"],
            "pose_detected": pose["pose_detected"],
            "feedback": feedback_for_event(i, features, angles, reliability),
            "combined_score": mahalanobis_combined_score(features, i, covariance),
        }
    analysis = {
        "events": events, "handedness": handedness, "mock_events": False,
        "sequence_features": evaluate_sequence_features(frames, load_stats(data / "pro_sequence_stats.json")),
    }
    evaluate_swing_motion(analysis, load_stats(data / "pro_motion_stats.json"),
                          load_stats(reference / "pro_dynamics_stats.json"))
    feedback = build_app_payload(analysis)
    feedback["meta"]["pose_refiner"] = {
        "status": "applied" if refined_count else "fallback",
        "model": "golfpose" if refined_count else "mediapipe",
        "reference": "refined" if refined_count else "legacy",
        "refined_frames": refined_count,
        "refined_events": {name: frame for name, frame, pose in zip(EVENT_NAMES, frames, poses, strict=True)
                           if pose["qc"].get("refined_event")},
        "reason": None if refined_count else reason,
    }
    return feedback
