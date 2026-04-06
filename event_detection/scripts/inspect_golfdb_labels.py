import argparse
from typing import List, Sequence, Tuple

import pandas as pd

EVENT_NAMES = [
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish",
]

ORIGINAL_COLUMNS = [
    "id",
    "youtube_id",
    "player",
    "sex",
    "club",
    "view",
    "slow",
    "events",
    "bbox",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect GolfDB labels from a pickle file."
    )
    parser.add_argument(
        "--data-file",
        default="data/golfDB.pkl",
        help="Path to pickle file (e.g. data/golfDB.pkl, data/train_split_1.pkl)",
    )
    parser.add_argument(
        "--id",
        dest="annotation_ids",
        type=int,
        nargs="+",
        help="Show only these annotation IDs",
    )
    parser.add_argument("--split", type=int, help="Filter by split index")
    parser.add_argument("--club", help="Filter by club type, e.g. driver")
    parser.add_argument("--view", help="Filter by camera view, e.g. face-on")
    parser.add_argument("--limit", type=int, default=10, help="Rows to print")
    parser.add_argument(
        "--show-event-names",
        action="store_true",
        help="Show 8 key events with human-readable names",
    )
    parser.add_argument(
        "--save-csv",
        default="",
        help="Optional output CSV path for filtered rows",
    )
    parser.add_argument(
        "--export-all-csv",
        default="data/golfDB_labels.csv",
        help="Output CSV path for exported rows after filters are applied. Set empty string to disable.",
    )
    return parser.parse_args()


def split_events(events: Sequence[int]) -> Tuple[List[int], List[int]]:
    raw = [int(x) for x in events]
    # GolfDB raw events are typically 10 values: [start, 8 key events, end].
    if len(raw) >= 10:
        key_events = raw[1:-1]
    elif len(raw) == 8:
        key_events = raw
    else:
        key_events = []
    return raw, key_events


def build_event_name_text(key_events: List[int]) -> str:
    pairs = []
    for i, frame_idx in enumerate(key_events):
        name = EVENT_NAMES[i] if i < len(EVENT_NAMES) else f"Event{i}"
        pairs.append(f"{name}:{frame_idx}")
    return " | ".join(pairs)


def event_column_name(idx: int, event_name: str) -> str:
    cleaned = event_name.lower().replace("-", "_").replace(" ", "_")
    return f"event_{idx}_{cleaned}"


def split_bbox(bbox: Sequence[float]) -> List[float]:
    return [float(x) for x in bbox]


def safe_value(row: pd.Series, key: str):
    if key not in row:
        return ""
    value = row[key]
    if pd.isna(value):
        return ""
    return value


def build_output_df(source_df: pd.DataFrame, include_named_events: bool) -> pd.DataFrame:
    rows = []
    for _, row in source_df.iterrows():
        raw_events, key_events = split_events(row["events"])
        bbox = split_bbox(row["bbox"]) if "bbox" in row else []
        out = {
            "id": int(safe_value(row, "id")) if safe_value(row, "id") != "" else "",
            "video": safe_value(row, "video"),
            "youtube_id": safe_value(row, "youtube_id"),
            "player": safe_value(row, "player"),
            "sex": safe_value(row, "sex"),
            "club": safe_value(row, "club"),
            "view": safe_value(row, "view"),
            "slow": safe_value(row, "slow"),
            "events": "|".join(str(x) for x in raw_events),
            "bbox": "|".join(f"{x:.8f}" for x in bbox),
            "split": safe_value(row, "split"),
            "events_raw_len": len(raw_events),
            "key_event_count": len(key_events),
            "event_start": raw_events[0] if raw_events else "",
            "event_end": raw_events[-1] if raw_events else "",
            "events_raw": "|".join(str(x) for x in raw_events),
            "events_8": "|".join(str(x) for x in key_events),
            "bbox_x": bbox[0] if len(bbox) > 0 else "",
            "bbox_y": bbox[1] if len(bbox) > 1 else "",
            "bbox_w": bbox[2] if len(bbox) > 2 else "",
            "bbox_h": bbox[3] if len(bbox) > 3 else "",
        }

        for i, event_name in enumerate(EVENT_NAMES, start=1):
            col = event_column_name(i, event_name)
            out[col] = key_events[i - 1] if i - 1 < len(key_events) else ""

        if include_named_events:
            out["events_8_named"] = build_event_name_text(key_events)

        rows.append(out)

    return pd.DataFrame(rows)


def build_export_df(source_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in source_df.iterrows():
        raw_events, _ = split_events(row["events"])
        bbox = split_bbox(row["bbox"]) if "bbox" in row else []
        rows.append(
            {
                "id": int(safe_value(row, "id")) if safe_value(row, "id") != "" else "",
                "youtube_id": safe_value(row, "youtube_id"),
                "player": safe_value(row, "player"),
                "sex": safe_value(row, "sex"),
                "club": safe_value(row, "club"),
                "view": safe_value(row, "view"),
                "slow": safe_value(row, "slow"),
                "events": "|".join(str(x) for x in raw_events),
                "bbox": "|".join(f"{x:.8f}" for x in bbox),
                "split": safe_value(row, "split"),
            }
        )
    return pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)


def main() -> int:
    args = parse_args()

    source_df = pd.read_pickle(args.data_file)

    filtered_df = source_df
    if args.annotation_ids:
        filtered_df = filtered_df[filtered_df["id"].astype(int).isin(args.annotation_ids)]
    if args.split is not None and "split" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["split"].astype(int) == args.split]
    if args.club and "club" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["club"].astype(str).str.lower() == args.club.lower()
        ]
    if args.view and "view" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["view"].astype(str).str.lower() == args.view.lower()
        ]

    if filtered_df.empty:
        print("No rows matched your filters.")
        return 1

    out_df = build_output_df(filtered_df, include_named_events=args.show_event_names)

    if args.export_all_csv:
        export_df = build_export_df(filtered_df)
        export_df.to_csv(args.export_all_csv, index=False)
        print(
            f"Saved export CSV: {args.export_all_csv} (rows: {len(export_df)})"
        )

    print(f"source: {args.data_file}")
    print(f"matched_rows: {len(out_df)}")
    print("event length distribution:")
    print(out_df["events_raw_len"].value_counts().sort_index().to_string())
    print()

    print(out_df.head(args.limit).to_string(index=False))

    if args.save_csv:
        export_df = build_export_df(filtered_df)
        export_df.to_csv(args.save_csv, index=False)
        print(f"\nSaved filtered CSV: {args.save_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
