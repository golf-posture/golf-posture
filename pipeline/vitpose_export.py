"""Export GolfPose ViTPose-H (person) as one TorchScript file that needs no mmpose to run.

Run once, in the environment that already has mmpose (outputs/golfpose/venv):
    outputs/golfpose/venv/Scripts/python.exe pipeline/vitpose_export.py \
        --config outputs/golfpose/golfpose_person_ViTPose_huge_full.py \
        --checkpoint outputs/golfpose/golfpose_person_ViTPose_huge_clean.pth \
        --output outputs/golfpose/vitpose_huge.ts

The exported module takes (N, 3, 256, 192) float BGR 0..255 crops (vitpose_runtime.model_input) and returns the
flip-test averaged heatmaps (N, 17, 64, 48). It does the BGR->RGB conversion, normalisation and the flip test
itself, as mmpose's TopdownPoseEstimator does (test_cfg: flip_test, flip_mode=heatmap, shift_heatmap=False).
Check the result with vitpose_verify.py before using it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.vitpose_runtime import FLIP_INDICES, INPUT_SIZE, MEAN_RGB, STD_RGB  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_wrapper(model):
    import torch
    from torch import nn

    class ExportedViTPose(nn.Module):
        def __init__(self, pose_model) -> None:
            super().__init__()
            self.model = pose_model
            self.register_buffer("mean", torch.tensor(MEAN_RGB).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(STD_RGB).view(1, 3, 1, 1))
            # a buffer, not a Python list: indexing with a list bakes the tracing device into the file
            self.register_buffer("flip_index", torch.tensor(FLIP_INDICES, dtype=torch.long))

        def heatmaps(self, x):
            return self.model.head.forward(self.model.extract_feat(x))

        def forward(self, bgr):
            x = bgr.flip(1)  # BGR -> RGB
            x = (x - self.mean) / self.std
            plain = self.heatmaps(x)
            mirrored = self.heatmaps(x.flip(-1)).flip(-1).index_select(1, self.flip_index)
            return (plain + mirrored) * 0.5

    return ExportedViTPose(model).eval()


def main() -> int:
    import torch
    from mmpose.apis import init_model

    args = parse_args()
    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    wrapper = build_wrapper(model).to(args.device)
    example = torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0], device=args.device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    sys.stdout.write(f"saved: {args.output} ({args.output.stat().st_size / 1e9:.2f} GB, torch {torch.__version__})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
