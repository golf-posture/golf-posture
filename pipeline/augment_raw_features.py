"""Add derived swing-quality features (xfactor) to an existing raw feature CSV
without re-running pose extraction on the source videos.

xfactor = circular difference between shoulder_tilt and hip_tilt at the same
event/frame. This is the classic golf biomechanics "X-factor" (shoulder-hip
separation angle) and only needs the two angle columns we already extracted.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def circular_delta(value: float, center: float) -> float:
    return float((value - center + 180.0) % 360.0 - 180.0)


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add xfactor column to raw feature CSV.")
    parser.add_argument("--input", default="data/pro_feature_raw.csv")
    parser.add_argument("--output", default="data/pro_feature_raw.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.input).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "xfactor" not in fieldnames:
        # insert right after hip_tilt for readability
        idx = fieldnames.index("hip_tilt") + 1 if "hip_tilt" in fieldnames else len(fieldnames)
        fieldnames = fieldnames[:idx] + ["xfactor"] + fieldnames[idx:]

    added = 0
    for row in rows:
        shoulder = as_float(row.get("shoulder_tilt"))
        hip = as_float(row.get("hip_tilt"))
        if shoulder is not None and hip is not None:
            row["xfactor"] = circular_delta(shoulder, hip)
            added += 1
        else:
            row["xfactor"] = ""

    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"xfactor computed for {added}/{len(rows)} rows")
    print(f"Written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
