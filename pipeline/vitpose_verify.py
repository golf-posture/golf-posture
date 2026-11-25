"""Check that the exported ViTPose-H gives the same keypoints as the mmpose run it came from.

Needs only torch, OpenCV and numpy (no mmpose):
    python pipeline/vitpose_verify.py --model models/golfpose/vitpose_huge.ts \
        --boxes outputs/golfpose/boxes.json \
        --reference outputs/golfpose/pred_golfpose_vitpose_gpu.json \
        --output outputs/golfpose/pred_vitpose_ts.json

--reference is a prediction cache made by golfpose_infer.py (mmpose). The summary compares the benchmark joints
(shoulders, elbows, wrists, hips) in pixels. --output is a prediction cache for pose_benchmark, so the benchmark
metrics can be compared too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.golfpose_joints import golfpose_joints, predictions_payload  # noqa: E402
from pipeline.vitpose_runtime import crop_for_model, model_input, predict_keypoints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, default=Path("outputs/golfpose/boxes.json"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="only the first N frames (0 = all)")
    return parser.parse_args()


def run_model(model, device: str, image_bgr: np.ndarray, box: list[float]):
    import torch

    crop, center, scale = crop_for_model(image_bgr, box)
    tensor = torch.from_numpy(model_input(crop)).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = model(tensor)[0].float().cpu().numpy()
    return predict_keypoints(heatmaps, center, scale)


def summarise(differences: list[float], score_differences: list[float], without_reference: int,
              without_result: int) -> str:
    d = np.asarray(differences)
    s = np.asarray(score_differences)
    return (f"joints compared: {d.size}\n"
            f"position diff px: mean {d.mean():.4f}  p99 {np.percentile(d, 99):.4f}  max {d.max():.4f}\n"
            f"score diff: mean {s.mean():.5f}  max {s.max():.5f}\n"
            f"frames only the exported model found: {without_reference}, only mmpose found: {without_result}")


def main() -> int:
    import cv2
    import torch

    args = parse_args()
    model = torch.jit.load(str(args.model), map_location=args.device).eval()
    videos = json.loads(args.boxes.read_text(encoding="utf-8"))["videos"]
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["frames"]
    frames: dict[str, dict[int, dict | None]] = {}
    differences: list[float] = []
    score_differences: list[float] = []
    without_reference = without_result = done = 0
    started = time.time()
    for name, entry in sorted(videos.items()):
        if args.limit and done >= args.limit:
            break
        capture = cv2.VideoCapture(entry["path"])
        per_frame: dict[int, dict | None] = {}
        for frame_text, box in entry["boxes"].items():
            if args.limit and done >= args.limit:
                break
            per_frame[int(frame_text)] = None
            expected = reference.get(name, {}).get(frame_text)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_text))
            ok, image = capture.read()
            if box is None or not ok:
                without_result += expected is not None
                continue
            keypoints, scores = run_model(model, args.device, image, box)
            joints = golfpose_joints(keypoints, scores)
            per_frame[int(frame_text)] = joints
            done += 1
            if expected is None:
                without_reference += 1
                continue
            for joint, value in joints.items():
                ref = expected[joint]
                differences.append(float(np.hypot(value[0] - ref[0], value[1] - ref[1])))
                score_differences.append(abs(value[2] - ref[2]))
        capture.release()
        frames[name] = per_frame
    sys.stdout.write(f"frames run: {done} in {time.time() - started:.0f}s (torch {torch.__version__})\n")
    sys.stdout.write(summarise(differences, score_differences, without_reference, without_result) + "\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(predictions_payload("vitpose_ts", frames)), encoding="utf-8")
        sys.stdout.write(f"saved: {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
