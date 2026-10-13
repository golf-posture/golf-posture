"""Quality-control flags for data/pro_feature_raw_tracked.csv.

Flags never delete rows; they mark values that should be treated as missing downstream.
Rules were fixed from the value distributions and visual review, before any clustering was run.

Row-level, per arm (apply to that arm's elbow angles only):
  low_vis       min visibility over shoulder/elbow/wrist < LOW_VIS_THRESHOLD
  bone          3D forearm/upper-arm ratio outside BONE_RATIO_RANGE, or a 3D bone shorter than MIN_BONE_3D
  frame         a joint lies outside the image
  inconsistent  |2D angle - 3D angle| > MAX_2D_3D_DIFF_DEG
Feature-level (per event, across videos):
  outlier_<column>  robust z (median/MAD; circular for shoulder/hip tilt) > OUTLIER_Z
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from pipeline.compute_reliability_stats import circular_delta, circular_mean

LOW_VIS_THRESHOLD = 0.4
BONE_RATIO_RANGE = (0.5, 2.0)
MIN_BONE_3D = 0.1
MAX_2D_3D_DIFF_DEG = 60.0
OUTLIER_Z = 4.0
MIN_VALUES_FOR_OUTLIER = 5
MIN_OUTLIER_SCALE_DEG = 1.0
MAD_TO_SIGMA = 1.4826

ARMS = ("trail", "lead")
ARM_FLAGS = ("low_vis", "bone", "frame", "inconsistent")
OUTLIER_COLUMNS = (
    "trail_elbow_angle_2d", "trail_elbow_angle_3d", "lead_elbow_angle_2d", "lead_elbow_angle_3d",
    "shoulder_tilt", "hip_tilt", "spine_tilt",
)
CIRCULAR_COLUMNS = {"shoulder_tilt", "hip_tilt"}
FINISH_EVENT_INDEX = 7
# At Finish the shoulder/hip line is nearly parallel to the camera axis, so the 2D left/right order
# flips between swings: the distribution is genuinely two-modal (~64% near +-180, ~36% near 0), not noise.
OUTLIER_EXEMPT = {("shoulder_tilt", FINISH_EVENT_INDEX), ("hip_tilt", FINISH_EVENT_INDEX)}


def outlier_exempt(column: str, event_index: int) -> bool:
    return (column, event_index) in OUTLIER_EXEMPT


def _num(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def bone_length_flag(upper_3d: float | None, fore_3d: float | None) -> bool:
    upper, fore = _num(upper_3d), _num(fore_3d)
    if upper is None or fore is None:
        return False
    if upper < MIN_BONE_3D or fore < MIN_BONE_3D:
        return True
    ratio = fore / upper
    return bool(ratio < BONE_RATIO_RANGE[0] or ratio > BONE_RATIO_RANGE[1])


def robust_outlier_flags(
    values: Sequence[float | None],
    circular: bool,
    z_threshold: float = OUTLIER_Z,
    min_values: int = MIN_VALUES_FOR_OUTLIER,
) -> list[bool]:
    flags = [False] * len(values)
    valid = [(i, _num(v)) for i, v in enumerate(values) if _num(v) is not None]
    if len(valid) < min_values:
        return flags
    numbers = [v for _, v in valid]
    center = circular_mean(numbers) if circular else statistics.median(numbers)
    deltas = [circular_delta(v, center) if circular else v - center for v in numbers]
    median_delta = statistics.median(deltas)
    mad = statistics.median([abs(d - median_delta) for d in deltas])
    scale = max(MAD_TO_SIGMA * mad, MIN_OUTLIER_SCALE_DEG)
    for (index, _), delta in zip(valid, deltas):
        flags[index] = bool(abs(delta - median_delta) / scale > z_threshold)
    return flags


def arm_flags(row: Mapping[str, Any], arm: str) -> dict[str, bool]:
    if row.get(f"{arm}_elbow_source") == "missing":
        return {name: False for name in ARM_FLAGS}
    min_vis = _num(row.get(f"{arm}_min_vis"))
    angle_2d, angle_3d = _num(row.get(f"{arm}_elbow_angle_2d")), _num(row.get(f"{arm}_elbow_angle_3d"))
    return {
        "low_vis": bool(min_vis is not None and min_vis < LOW_VIS_THRESHOLD),
        "bone": bone_length_flag(row.get(f"{arm}_upper_3d"), row.get(f"{arm}_fore_3d")),
        "frame": not bool(row.get(f"{arm}_in_frame")),
        "inconsistent": bool(
            angle_2d is not None and angle_3d is not None and abs(angle_2d - angle_3d) > MAX_2D_3D_DIFF_DEG
        ),
    }


# --------------------------------------------------------------------- table
def build_qc_table(tracked: pd.DataFrame) -> pd.DataFrame:
    qc = tracked[["video_id", "event_index"]].copy()
    for arm in ARMS:
        flags = pd.DataFrame([arm_flags(row, arm) for row in tracked.to_dict("records")], index=tracked.index)
        for name in ARM_FLAGS:
            qc[f"{arm}_flag_{name}"] = flags[name]
    for column in OUTLIER_COLUMNS:
        qc[f"outlier_{column}"] = False
    for event_index, group in tracked.groupby("event_index"):
        for column in OUTLIER_COLUMNS:
            if outlier_exempt(column, event_index):
                continue
            flags = robust_outlier_flags(group[column].tolist(), circular=column in CIRCULAR_COLUMNS)
            qc.loc[group.index, f"outlier_{column}"] = flags
    return qc


def clean_tracked(tracked: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """Copy of `tracked` with QC-flagged values set to NaN (arm flags hit that arm's elbow angles in
    both 2D and 3D; outlier flags hit a single column)."""
    cleaned = tracked.copy()
    for arm in ARMS:
        bad = qc[[f"{arm}_flag_{name}" for name in ARM_FLAGS]].any(axis=1)
        for dims in ("2d", "3d"):
            column = f"{arm}_elbow_angle_{dims}"
            cleaned.loc[bad | qc[f"outlier_{column}"], column] = float("nan")
    for column in ("shoulder_tilt", "hip_tilt", "spine_tilt"):
        cleaned.loc[qc[f"outlier_{column}"], column] = float("nan")
    return cleaned


def summarize_flags(qc: pd.DataFrame, tracked: pd.DataFrame) -> str:
    lines = [f"rows: {len(qc)}"]
    for arm in ARMS:
        valid = tracked[f"{arm}_elbow_source"] != "missing"
        lines.append(f"[{arm}] valid elbow rows: {int(valid.sum())}")
        for name in ARM_FLAGS:
            count = int(qc.loc[valid, f"{arm}_flag_{name}"].sum())
            lines.append(f"  {name:<13} {count:>5} ({count / max(int(valid.sum()), 1):.1%})")
    for column in OUTLIER_COLUMNS:
        count = int(qc[f"outlier_{column}"].sum())
        lines.append(f"outlier_{column:<22} {count:>5}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/pro_feature_raw_tracked.csv")
    parser.add_argument("--output", default="data/pro_feature_qc_tracked.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tracked = pd.read_csv(args.input)
    qc = build_qc_table(tracked)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(args.output, index=False)
    print(summarize_flags(qc, tracked))
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
