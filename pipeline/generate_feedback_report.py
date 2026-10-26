from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


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

REQUIRED_FEATURES = [
    "trail_elbow_angle",
    "lead_elbow_angle",
    "shoulder_tilt",
    "hip_tilt",
    "spine_tilt",
]

EVENT_KO = {
    "Address": "어드레스",
    "Toe-up": "토업",
    "Mid-backswing": "백스윙 중간",
    "Top": "탑",
    "Mid-downswing": "다운스윙 중간",
    "Impact": "임팩트",
    "Mid-follow-through": "팔로우스루 중간",
    "Finish": "피니시",
}

FEATURE_KO = {
    "trail_elbow_angle": "트레일 팔꿈치 각도",
    "lead_elbow_angle": "리드 팔꿈치 각도",
    "shoulder_tilt": "어깨 기울기",
    "hip_tilt": "골반 기울기",
    "spine_tilt": "척추 기울기",
}

GRADE_PRIORITY = {"Needs Improvement": 3, "Good": 2, "Excellent": 0, "unavailable": 0}
RELIABILITY_PRIORITY = {"high": 3, "medium": 2, "unknown": 1, "low": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Korean feedback from analysis JSON.")
    parser.add_argument("--analysis-json", required=True, help="Path to run_analysis result JSON.")
    parser.add_argument("--output-dir", help="Directory for feedback_table.csv and feedback_ko.txt.")
    parser.add_argument(
        "--prefix",
        help="Optional filename prefix for legacy prefix-mode outputs.",
    )
    return parser.parse_args()


def load_analysis(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ordered_event_items(events: dict[str, Any]):
    used = set()
    for event_name in EVENT_NAMES:
        if event_name in events:
            used.add(event_name)
            yield event_name, events[event_name]
    for event_name, event in events.items():
        if event_name not in used:
            yield event_name, event


def display_grade(item: dict[str, Any]) -> str:
    grade = item.get("grade") or "unavailable"
    reliability = item.get("reliability") or "unknown"
    if reliability == "low" and grade != "Excellent":
        return f"{grade}*"
    return grade


def build_table_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for event_name, event in ordered_event_items(data.get("events", {})):
        row = {"Event": event_name}
        feedback = event.get("feedback", {})
        for feature_name in REQUIRED_FEATURES:
            row[feature_name] = display_grade(feedback.get(feature_name, {}))
        rows.append(row)
    return rows


def write_table_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Event", *REQUIRED_FEATURES]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: list[dict[str, str]]) -> list[str]:
    fieldnames = ["Event", *REQUIRED_FEATURES]
    widths = {}
    for field in fieldnames:
        values = [len(str(row.get(field, ""))) for row in rows]
        widths[field] = max([len(field), *values])
    lines = []
    header = "  ".join(field.ljust(widths[field]) for field in fieldnames)
    lines.append(header)
    lines.append("  ".join("-" * widths[field] for field in fieldnames))
    for row in rows:
        lines.append("  ".join(str(row.get(field, "")).ljust(widths[field]) for field in fieldnames))
    return lines


def iter_feedback_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for event_name, event in ordered_event_items(data.get("events", {})):
        for feature_name in REQUIRED_FEATURES:
            item = event.get("feedback", {}).get(feature_name, {})
            grade = item.get("grade") or "unavailable"
            reliability = item.get("reliability") or "unknown"
            z_score = item.get("z_score")
            abs_z = abs(z_score) if isinstance(z_score, (int, float)) and math.isfinite(z_score) else 0.0
            items.append(
                {
                    "event_name": event_name,
                    "feature_name": feature_name,
                    "grade": grade,
                    "reliability": reliability,
                    "feedback_strength": item.get("feedback_strength", "normal"),
                    "z_score": z_score,
                    "abs_z": abs_z,
                    "value": item.get("value"),
                    "raw": item,
                }
            )
    return items


def importance_key(item: dict[str, Any]) -> tuple[int, int, float]:
    return (
        GRADE_PRIORITY.get(item["grade"], 0),
        RELIABILITY_PRIORITY.get(item["reliability"], 1),
        item["abs_z"],
    )


def select_core_items(items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in items
        if item["reliability"] != "low" and item["grade"] in {"Needs Improvement", "Good"}
    ]
    candidates.sort(key=importance_key, reverse=True)
    return candidates[:limit]


def select_reference_items(items: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in items
        if item["reliability"] == "low" and item["grade"] != "Excellent"
    ]
    candidates.sort(key=importance_key, reverse=True)
    return candidates[:limit]


def sentence_for_core(item: dict[str, Any]) -> str:
    event_ko = EVENT_KO.get(item["event_name"], item["event_name"])
    feature_ko = FEATURE_KO.get(item["feature_name"], item["feature_name"])
    z_text = f" z-score는 {item['z_score']:.2f}입니다." if isinstance(item["z_score"], (int, float)) else ""

    if item["grade"] == "Needs Improvement":
        sentence = f"{event_ko} 단계에서 {feature_ko}가 프로 기준과 크게 차이가 있습니다."
    else:
        sentence = f"{event_ko} 단계에서 {feature_ko}가 프로 기준과 약간 차이가 있습니다."

    if item["reliability"] == "high":
        sentence += " 이 항목은 프로 선수들 사이에서도 공통성이 높은 지표이므로 우선적으로 확인하는 것이 좋습니다."
    elif item["reliability"] == "medium":
        sentence += " 일반적인 점검 항목으로 참고할 수 있습니다."
    else:
        sentence += " 기준 신뢰도 정보가 충분하지 않으므로 보조 지표로 확인하는 것이 좋습니다."
    return sentence + z_text


def sentence_for_reference(item: dict[str, Any]) -> str:
    event_ko = EVENT_KO.get(item["event_name"], item["event_name"])
    feature_ko = FEATURE_KO.get(item["feature_name"], item["feature_name"])
    return (
        f"{event_ko} 단계의 {feature_ko}는 프로 기준과 차이가 있으나, "
        "프로 선수 간 개인차가 큰 항목이므로 참고용으로만 보는 것이 적절합니다."
    )


def iter_combined_score_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for event_name, event in ordered_event_items(data.get("events", {})):
        combined = event.get("combined_score")
        if not combined:
            continue
        items.append(
            {
                "event_name": event_name,
                "grade": combined.get("grade") or "unavailable",
                "percentile": combined.get("deviation_percentile"),
                "distance": combined.get("distance"),
            }
        )
    return items


def select_combined_score_items(items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    candidates = [item for item in items if item["grade"] in {"Needs Improvement", "Good"}]
    candidates.sort(key=lambda item: item["percentile"] or 0.0, reverse=True)
    return candidates[:limit]


def sentence_for_combined(item: dict[str, Any]) -> str:
    event_ko = EVENT_KO.get(item["event_name"], item["event_name"])
    percentile = item["percentile"]
    percentile_text = f"{percentile:.1f}%" if isinstance(percentile, (int, float)) else "알 수 없음"
    degree = "매우 드문" if item["grade"] == "Needs Improvement" else "다소 특이한"
    return (
        f"{event_ko} 단계는 개별 항목은 대체로 양호하지만, 5개 항목의 조합이 프로 데이터 분포 기준 "
        f"이례도 {percentile_text}로 {degree} 패턴입니다. (이례도 수치가 높을수록 프로 데이터에서 벗어난 조합입니다.)"
    )


def iter_dynamics_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    dynamics = data.get("dynamics") or {}
    for pair_label, pair_data in dynamics.items():
        features = pair_data.get("features", pair_data) if isinstance(pair_data, dict) else {}
        for feature_name in REQUIRED_FEATURES:
            item = features.get(feature_name) if isinstance(features, dict) else None
            if not item:
                continue
            grade = item.get("grade") or "unavailable"
            reliability = item.get("reliability") or "unknown"
            robust_z = item.get("robust_z")
            abs_z = abs(robust_z) if isinstance(robust_z, (int, float)) and math.isfinite(robust_z) else 0.0
            items.append(
                {
                    "pair_label": pair_label,
                    "feature_name": feature_name,
                    "grade": grade,
                    "reliability": reliability,
                    "robust_z": robust_z,
                    "abs_z": abs_z,
                }
            )
    return items


def select_core_dynamics_items(items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    candidates = [
        item for item in items if item["reliability"] != "low" and item["grade"] in {"Needs Improvement", "Good"}
    ]
    candidates.sort(key=importance_key, reverse=True)
    return candidates[:limit]


def select_reference_dynamics_items(items: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    candidates = [item for item in items if item["reliability"] == "low" and item["grade"] != "Excellent"]
    candidates.sort(key=importance_key, reverse=True)
    return candidates[:limit]


def _has_final_consonant(word: str) -> bool:
    if not word:
        return False
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _pair_transition_phrase(pair_label: str) -> str:
    start_name, _, end_name = pair_label.partition("->")
    start_ko = EVENT_KO.get(start_name, start_name)
    end_ko = EVENT_KO.get(end_name, end_name)
    particle = "으로" if _has_final_consonant(end_ko) else "로"
    return f"{start_ko}에서 {end_ko}{particle} 이어지는"


def sentence_for_dynamics_core(item: dict[str, Any]) -> str:
    transition = _pair_transition_phrase(item["pair_label"])
    feature_ko = FEATURE_KO.get(item["feature_name"], item["feature_name"])
    z_text = f" robust_z는 {item['robust_z']:.2f}입니다." if isinstance(item["robust_z"], (int, float)) else ""

    if item["grade"] == "Needs Improvement":
        sentence = f"{transition} 구간의 {feature_ko} 변화량이 프로 기준과 크게 차이가 있습니다."
    else:
        sentence = f"{transition} 구간의 {feature_ko} 변화량이 프로 기준과 약간 차이가 있습니다."

    if item["reliability"] == "high":
        sentence += " 이 구간은 프로 선수들 사이에서도 변화 패턴이 일관적인 지표이므로 확인하는 것이 좋습니다."
    elif item["reliability"] == "medium":
        sentence += " 일반적인 점검 항목으로 참고할 수 있습니다."
    return sentence + z_text


def sentence_for_dynamics_reference(item: dict[str, Any]) -> str:
    transition = _pair_transition_phrase(item["pair_label"])
    feature_ko = FEATURE_KO.get(item["feature_name"], item["feature_name"])
    return (
        f"{transition} 구간의 {feature_ko} 변화량은 프로 기준과 차이가 있으나, "
        "프로 선수 간 개인차가 큰 구간이므로 참고용으로만 보는 것이 적절합니다."
    )


def summary_sentence(counts: Counter, total: int) -> str:
    needs = counts.get("Needs Improvement", 0)
    good = counts.get("Good", 0)
    unavailable = counts.get("unavailable", 0)
    if needs:
        return f"총 {total}개 항목 중 교정이 필요한 항목이 {needs}개 확인되었습니다."
    if good:
        return f"총 {total}개 항목 중 큰 교정 항목은 없고, 일부 점검 항목이 {good}개 확인되었습니다."
    if unavailable:
        return f"총 {total}개 항목 중 일부는 관절 검출 실패로 평가할 수 없었습니다."
    return f"총 {total}개 항목이 전반적으로 프로 기준과 안정적으로 일치합니다."


def build_korean_feedback(
    data: dict[str, Any],
    rows: list[dict[str, str]],
    analysis_json: Path,
    table_csv: Path,
    feedback_ko: Path,
) -> str:
    items = iter_feedback_items(data)
    counts = Counter(item["grade"] for item in items)
    total = len(items)
    core_items = select_core_items(items)
    reference_items = select_reference_items(items)

    lines = [
        "[전체 요약]",
        (
            f"Excellent {counts.get('Excellent', 0)}개, Good {counts.get('Good', 0)}개, "
            f"Needs Improvement {counts.get('Needs Improvement', 0)}개, "
            f"unavailable {counts.get('unavailable', 0)}개입니다."
        ),
        summary_sentence(counts, total),
    ]
    if counts.get("unavailable", 0):
        lines.append("일부 항목은 필요한 관절이 검출되지 않아 평가할 수 없습니다.")

    lines.extend(["", "[자세 평가 요약표]"])
    lines.extend(format_table(rows))
    lines.append("")
    lines.append("* = 프로 선수 간 개인차가 큰 항목으로 참고용 지표")

    lines.extend(["", "[핵심 피드백]"])
    if core_items:
        for idx, item in enumerate(core_items, start=1):
            lines.append(f"{idx}. {sentence_for_core(item)}")
    else:
        lines.append("우선적으로 교정해야 할 항목은 확인되지 않았습니다.")

    lines.extend(["", "[참고 지표]"])
    if reference_items:
        for item in reference_items:
            lines.append(f"- {sentence_for_reference(item)}")
    else:
        lines.append("- 참고용으로 분리할 low reliability 항목은 없습니다.")

    combined_all = iter_combined_score_items(data)
    combined_flagged = select_combined_score_items(combined_all)
    lines.extend(["", "[결합 스코어 참고 - 항목 조합 이상 탐지]"])
    if combined_flagged:
        for item in combined_flagged:
            lines.append(f"- {sentence_for_combined(item)}")
    elif combined_all:
        lines.append("- 결합 스코어 관점에서 특별히 이례적인 이벤트는 없습니다.")
    else:
        lines.append("- 이 결과에는 결합 스코어 정보가 포함되어 있지 않습니다.")

    dynamics_all = iter_dynamics_items(data)
    dynamics_core = select_core_dynamics_items(dynamics_all)
    dynamics_reference = select_reference_dynamics_items(dynamics_all)
    lines.extend(["", "[스윙 다이내믹스 참고 - 이벤트 간 변화량]"])
    if dynamics_core or dynamics_reference:
        for item in dynamics_core:
            lines.append(f"- {sentence_for_dynamics_core(item)}")
        for item in dynamics_reference:
            lines.append(f"- {sentence_for_dynamics_reference(item)}")
    elif dynamics_all:
        lines.append("- 스윙 다이내믹스 관점에서 특별히 확인이 필요한 구간은 없습니다.")
    else:
        lines.append("- 이 결과에는 스윙 다이내믹스 정보가 포함되어 있지 않습니다.")

    lines.extend(
        [
            "",
            "[결과 파일]",
            f"- JSON 상세 결과: {analysis_json}",
            f"- 기본 텍스트 요약: {data.get('text_summary_path')}",
            f"- 표 CSV: {table_csv}",
            f"- 한국어 피드백: {feedback_ko}",
            f"- keyframe 이미지 폴더: {data.get('event_frame_dir')}",
            f"- skeleton 비교 이미지 폴더: {data.get('skeleton_comparison_dir')}",
        ]
    )
    return "\n".join(lines) + "\n"


def report_paths(output_dir: Path, filename_prefix: str | None = None) -> dict[str, Path]:
    if filename_prefix:
        return {
            "table_csv": output_dir / f"{filename_prefix}_feedback_table.csv",
            "feedback_ko": output_dir / f"{filename_prefix}_feedback_ko.txt",
        }
    return {
        "table_csv": output_dir / "feedback_table.csv",
        "feedback_ko": output_dir / "feedback_ko.txt",
    }


def generate_feedback_report(
    analysis_json: Path | str,
    output_dir: Path | str | None = None,
    filename_prefix: str | None = None,
) -> dict[str, Path]:
    analysis_path = Path(analysis_json)
    output_path = Path(output_dir) if output_dir else analysis_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    paths = report_paths(output_path, filename_prefix)
    data = load_analysis(analysis_path)
    rows = build_table_rows(data)
    write_table_csv(rows, paths["table_csv"])

    feedback_text = build_korean_feedback(
        data,
        rows,
        analysis_path,
        paths["table_csv"],
        paths["feedback_ko"],
    )
    paths["feedback_ko"].write_text(feedback_text, encoding="utf-8")
    return paths


def main() -> int:
    args = parse_args()
    paths = generate_feedback_report(
        args.analysis_json,
        args.output_dir,
        filename_prefix=args.prefix,
    )
    print(f"Feedback table written: {paths['table_csv']}")
    print(f"Korean feedback written: {paths['feedback_ko']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
