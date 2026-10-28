"""General-user swing feedback (non-expert wording).

Reads the same analysis result dict that ``generate_feedback_report`` reads and
builds a *structured* feedback object first (``build_user_feedback``) and then
renders it as Korean text (``render_user_feedback_text``).  The structured
object is what a mobile app can later consume as cards, so the text renderer
is just one view of it.

Nothing here changes how scores are computed.  It only decides
  * what to praise,
  * which few things to fix first (TOP N, one item per body part, not per event),
  * what is only worth a side note,
  * when the analysis itself should be treated as unreliable.

Tunable rules live in the constants right below so they can be reviewed in one place.
"""
from __future__ import annotations

import math
from typing import Any

EVENT_NAMES = [
    "Address", "Toe-up", "Mid-backswing", "Top",
    "Mid-downswing", "Impact", "Mid-follow-through", "Finish",
]
EVENT_KO = {
    "Address": "어드레스", "Toe-up": "토업", "Mid-backswing": "백스윙 중간", "Top": "탑",
    "Mid-downswing": "다운스윙 중간", "Impact": "임팩트",
    "Mid-follow-through": "팔로우스루 중간", "Finish": "피니시",
}

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
TOP_N = 3
MAX_PRAISE = 5
MAX_SIDE_NOTES = 4
RELIABLE = ("high", "medium")

# A tempo ratio outside this range is treated as an event-detection failure, not
# as a swing flaw (pro tempo: median 3.7, 87% within 2.5-5.0).
TEMPO_PLAUSIBLE = (1.0, 8.0)

# Arms move fastest here, so a few frames of event error change the angle a lot.
FAST_EVENTS = {"Mid-downswing", "Impact"}

# Angle features grouped by body part, so one body part = one card.
FAMILIES: dict[str, dict[str, Any]] = {
    "elbow": {"features": ["lead_elbow_angle", "trail_elbow_angle"], "label": "팔꿈치 각도", "top": True},
    "spine": {"features": ["spine_tilt"], "label": "척추 기울기", "top": True},
    "shoulder_tilt": {"features": ["shoulder_tilt"], "label": "어깨선 기울기", "top": False},
    "hip_tilt": {"features": ["hip_tilt"], "label": "골반선 기울기", "top": False},
}
ARM_KO = {"lead_elbow_angle": "타깃 쪽(리드) 팔꿈치", "trail_elbow_angle": "뒤쪽(트레일) 팔꿈치"}

# metric -> {"label", "low": (problem, tip) , "high": (problem, tip), "ok", "up"}
# {val} = user's value, {med} = pro median.
MOTION_TEXT: dict[str, dict[str, Any]] = {
    "shoulder_turn_top": {
        "label": "백스윙 어깨 회전",
        "low": ("백스윙에서 어깨가 프로 중앙값({med})보다 덜 돌아갔어요({val}).",
                "등이 타깃을 향한다는 느낌으로 어깨를 끝까지 돌려 보세요."),
        "high": ("백스윙에서 어깨를 프로보다 많이 돌렸어요({val}).", ""),
        "ok": "백스윙에서 어깨 회전이 프로 범위 안이에요({val}).",
        "up": "백스윙에서 어깨를 충분히 돌려요({val}). 프로 상위권 수준이에요.",
    },
    "hip_turn_top": {
        "label": "탑에서 골반 회전",
        "low": ("탑에서 골반이 프로보다 덜 돌아갔어요({val}).", "하체도 자연스럽게 따라 돌아가는지 확인해 보세요."),
        "high": ("탑에서 골반이 프로보다 많이 돌아갔어요({val}).", "하체를 더 단단히 잡고 어깨와의 꼬임을 만들어 보세요."),
        "ok": "탑에서 골반 회전이 프로 범위 안이에요({val}).",
        "up": "탑에서 골반 회전이 프로 범위 안이에요({val}).",
    },
    "xfactor_turn_top": {
        "label": "어깨와 골반의 꼬임",
        "low": ("어깨와 골반의 꼬임이 프로보다 작아요({val}, 프로 중앙값 {med}).",
                "하체는 단단히 두고 어깨를 더 깊이 돌려 보세요."),
        "high": ("어깨와 골반의 꼬임이 프로보다 커요({val}).", ""),
        "ok": "어깨와 골반의 꼬임이 프로 범위 안이에요({val}).",
        "up": "어깨와 골반의 꼬임이 커서({val}) 힘을 모을 수 있는 자세예요.",
    },
    "head_movement": {
        "label": "머리 위치",
        "low": ("", ""),
        "high": ("스윙 중 머리가 프로보다 많이 움직여요.", "공을 끝까지 바라보며 머리 위치를 고정해 보세요."),
        "ok": "스윙 중 머리 위치가 안정적이에요.",
        "up": "스윙 중 머리 위치가 안정적이에요.",
    },
    "weight_shift": {
        "label": "체중 이동",
        "low": ("체중 이동이 프로보다 적어요({val}, 프로 중앙값 {med}).",
                "다운스윙에서 타깃 쪽 발로 체중을 옮기는 느낌을 연습해 보세요."),
        "high": ("골반이 프로보다 옆으로 많이 이동해요({val}).", "몸이 옆으로 밀리는 것(스웨이)은 아닌지 확인해 보세요."),
        "ok": "체중 이동이 프로와 비슷한 수준이에요.",
        "up": "체중 이동이 프로와 비슷한 수준이에요.",
    },
    "hip_height_change": {
        "label": "자세 높이 유지",
        "low": ("", ""),
        "high": ("임팩트에서 골반이 프로보다 많이 올라와요.", "일어서지 않고 자세 높이를 유지해 보세요."),
        "ok": "스윙 중 자세 높이가 잘 유지돼요.",
        "up": "스윙 중 자세 높이가 잘 유지돼요.",
    },
    "stance_change": {
        "label": "발 위치",
        "low": ("", ""),
        "high": ("스윙 중 발 간격이 프로보다 많이 변해요.", "발이 밀리지 않게 안정적으로 버텨 보세요."),
        "ok": "발 위치가 안정적이에요.",
        "up": "발 위치가 안정적이에요.",
    },
}
MOTION_ORDER = ["shoulder_turn_top", "xfactor_turn_top", "hip_turn_top", "weight_shift",
                "head_movement", "hip_height_change", "stance_change"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _span_text(event_names: list[str]) -> str:
    """['Mid-downswing','Impact'] -> '다운스윙 중간~임팩트'; non-contiguous -> joined by ', '; 5+ events -> '스윙 전반'."""
    idx = sorted({EVENT_NAMES.index(e) for e in event_names if e in EVENT_NAMES})
    if not idx:
        return ""
    if len(idx) >= 5:
        return "스윙 전반"
    segments: list[list[int]] = [[idx[0]]]
    for i in idx[1:]:
        if i == segments[-1][-1] + 1:
            segments[-1].append(i)
        else:
            segments.append([i])
    parts = []
    for seg in segments:
        first, last = EVENT_KO[EVENT_NAMES[seg[0]]], EVENT_KO[EVENT_NAMES[seg[-1]]]
        parts.append(first if len(seg) == 1 else f"{first}~{last}")
    return ", ".join(parts)


def _angle_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for event_name in EVENT_NAMES:
        feedback = ((data.get("events") or {}).get(event_name) or {}).get("feedback") or {}
        for feature, item in feedback.items():
            records.append({
                "event": event_name,
                "feature": feature,
                "grade": item.get("grade") or "unavailable",
                "reliability": item.get("reliability") or "unknown",
                "z": item.get("z_score"),
            })
    return records


def _family_of(feature: str) -> str | None:
    for name, fam in FAMILIES.items():
        if feature in fam["features"]:
            return name
    return None


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _family_cards(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (top_candidates, side_candidates). One card per body part."""
    top, side = [], []
    for fam_name, fam in FAMILIES.items():
        reliable = [r for r in records
                    if r["feature"] in fam["features"] and r["reliability"] in RELIABLE and _finite(r["z"])]
        strong = [r for r in reliable if r["grade"] == "Needs Improvement"]
        weak = [r for r in reliable if r["grade"] == "Good"]
        used = strong if (strong and fam["top"]) else strong + weak
        if not used:
            continue
        severity = max(abs(r["z"]) for r in used)
        span = _span_text([r["event"] for r in used])
        fast = any(r["event"] in FAST_EVENTS for r in used)
        card: dict[str, Any] = {
            "key": fam_name,
            "title": f"{fam['label']} ({span})",
            "severity": severity,
            "tier": 2 if strong else 1,
            "caveat": "",
            "advice": "",
        }
        if fam_name == "elbow":
            parts = []
            for feature in fam["features"]:
                rows = [r for r in used if r["feature"] == feature]
                if not rows:
                    continue
                lead = max(rows, key=lambda r: abs(r["z"]))
                shape = "더 펴져" if lead["z"] > 0 else "더 굽혀져"
                parts.append(f"{ARM_KO[feature]}가 {_span_text([r['event'] for r in rows])}에서 프로보다 {shape} 있어요.")
            card["message"] = " ".join(parts)
            card["advice"] = "슬로모션으로 프로 영상과 나란히 비교해 보세요."
        elif fam_name == "spine":
            card["message"] = f"척추 기울기가 {span}에서 프로와 달라요."
            card["advice"] = "상체 각도가 스윙 중에 달라지지 않는지 확인해 보세요."
        else:
            card["message"] = f"{fam['label']}가 {span}에서 프로와 달라요."
            card["advice"] = "카메라 각도의 영향을 받는 항목이라 참고용으로만 봐 주세요."
        if fast and fam["top"]:
            card["caveat"] = "이 구간은 동작이 빨라 분석 오차가 생기기 쉬워요. 다른 각도에서 한 번 더 촬영해 비교해 보면 더 정확해요."
        (top if (strong and fam["top"]) else side).append(card)
    return top, side


def _motion_cards(motion: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    top, side, praise = [], [], []
    for name in MOTION_ORDER:
        item = motion.get(name)
        text = MOTION_TEXT.get(name)
        if not item or not text or item.get("value") is None or not _finite(item.get("penalty_z")):
            continue
        penalty = item["penalty_z"]
        reliable = item.get("reliability") in RELIABLE
        fmt = {"val": item.get("value_text") or "", "med": item.get("pro_median_text") or ""}
        if penalty > 1.0:
            problem, tip = text.get(item.get("direction") or "", ("", ""))
            if not problem:
                continue
            card = {
                "key": name, "title": text["label"], "message": problem.format(**fmt),
                "advice": tip, "caveat": "", "severity": penalty,
                "tier": 2 if item.get("grade") == "Needs Improvement" else 1,
            }
            (top if reliable else side).append(card)
        elif reliable:
            up = _finite(item.get("robust_z")) and abs(item["robust_z"]) > 1.0
            praise.append(text["up" if up else "ok"].format(**fmt))
    return top, side, praise


def _angle_stability_praise(records: list[dict[str, Any]]) -> list[str]:
    sentences = []
    labels = {"spine": "척추 기울기", "shoulder_tilt": "어깨선 기울기", "hip_tilt": "골반선 기울기"}
    stable = []
    for fam_name, label in labels.items():
        rows = [r for r in records if r["feature"] in FAMILIES[fam_name]["features"]
                and r["reliability"] in RELIABLE and r["grade"] != "unavailable"]
        # every evaluated event must be Excellent, otherwise it contradicts the side notes
        if len(rows) >= 5 and all(r["grade"] == "Excellent" for r in rows):
            stable.append(label)
    if stable:
        sentences.append(f"{', '.join(stable)}가 스윙 내내 안정적이에요.")
    return sentences


def _tempo(data: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Return (praise, side_card, analysis_note)."""
    item = (data.get("sequence_features") or {}).get("tempo_ratio")
    if not item:
        return None, None, None
    value, mean = item.get("value"), item.get("pro_mean")
    if not _finite(value):
        return None, None, "백스윙·다운스윙 구간을 구분하지 못해 스윙 템포는 평가에서 제외했어요."
    low, high = TEMPO_PLAUSIBLE
    if not (low <= value <= high):
        return None, None, "백스윙·다운스윙 구간 구분이 불확실해 스윙 템포는 평가에서 제외했어요."
    if not _finite(mean):
        return None, None, None
    grade = item.get("grade")
    if grade == "Excellent":
        return f"스윙 리듬이 프로 평균과 가까워요(백스윙:다운스윙 = {value:.1f}:1, 프로 평균 {mean:.1f}:1).", None, None
    if grade in ("Good", "Needs Improvement"):
        shape = "백스윙이 긴" if value > mean else "백스윙이 짧은"
        return None, {
            "key": "tempo", "title": "스윙 리듬",
            "message": f"스윙 리듬이 {value:.1f}:1로 프로 평균({mean:.1f}:1)보다 {shape} 편이에요.",
            "advice": "템포는 선수마다 개인차가 커서 참고용으로 봐 주세요.", "caveat": "",
            "severity": 0.0, "tier": 0,
        }, None
    return None, None, None


def _count_reliable(records: list[dict[str, Any]], motion: dict[str, Any]) -> tuple[int, int]:
    total = ok = 0
    for r in records:
        if r["grade"] == "unavailable":
            continue
        if r["reliability"] not in RELIABLE and r["grade"] != "Excellent":
            continue  # unreliable and flagged -> not counted either way
        total += 1
        ok += r["grade"] == "Excellent"
    for item in motion.values():
        if item.get("value") is None or item.get("grade") in (None, "unavailable"):
            continue
        if item.get("reliability") not in RELIABLE:
            continue
        total += 1
        ok += item.get("grade") == "Excellent"
    return ok, total


def _tone(fraction: float, n_top: int) -> str:
    if n_top == 0 and fraction >= 0.85:
        return "전체적으로 아주 좋은 스윙이에요."
    if fraction >= 0.8:
        return "전체적으로 안정적인 스윙이에요."
    if fraction >= 0.6:
        return "좋은 점도 많지만, 다듬으면 더 좋아질 부분이 있어요."
    return "다듬으면 크게 좋아질 부분이 여럿 있어요."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_user_feedback(data: dict[str, Any]) -> dict[str, Any]:
    records = _angle_records(data)
    motion = data.get("motion_metrics") or {}

    fam_top, fam_side = _family_cards(records)
    mot_top, mot_side, mot_praise = _motion_cards(motion)
    tempo_praise, tempo_side, tempo_note = _tempo(data)

    ranked = sorted(fam_top + mot_top, key=lambda c: (c["tier"], c["severity"]), reverse=True)
    top = ranked[:TOP_N]
    leftovers = ranked[TOP_N:]
    side = sorted(leftovers + fam_side + mot_side, key=lambda c: (c["tier"], c["severity"]), reverse=True)
    if tempo_side:
        side.append(tempo_side)
    side = side[:MAX_SIDE_NOTES]

    praise = _angle_stability_praise(records) + mot_praise
    if tempo_praise:
        praise.append(tempo_praise)
    praise = praise[:MAX_PRAISE]

    ok, total = _count_reliable(records, motion)
    fraction = ok / total if total else 0.0

    notes: list[str] = []
    if tempo_note:
        notes.append(tempo_note)
    n_unavail = sum(r["grade"] == "unavailable" for r in records)
    if n_unavail >= 3:
        notes.append("일부 자세는 관절이 잘 보이지 않아 평가하지 못했어요. 전신이 정면에서 보이게 촬영하면 더 정확해요.")
    n_ref = sum(r["reliability"] not in RELIABLE and r["grade"] not in ("Excellent", "unavailable") for r in records)
    if n_ref:
        notes.append("프로 선수들 사이에서도 개인차가 큰 일부 항목은 참고용이라 결과에서 제외했어요.")
    notes.append("정면 영상 기준의 2D 추정이라 회전각은 근사값이에요.")

    return {
        "overview": {
            "tone": _tone(fraction, len(top)) if total else "분석할 수 있는 항목이 부족했어요.",
            "within_pro_range": ok,
            "analyzed": total,
        },
        "praise": praise,
        "fix_first": top,
        "side_notes": side,
        "analysis_notes": notes,
    }


def render_user_feedback_text(feedback: dict[str, Any]) -> str:
    ov = feedback["overview"]
    lines = ["[이번 스윙 한눈에 보기]", ov["tone"]]
    if ov["analyzed"]:
        lines.append(f"분석한 자세·동작 {ov['analyzed']}개 중 {ov['within_pro_range']}개가 프로 범위 안에 있어요.")

    lines += ["", "[잘하고 있는 점]"]
    lines += [f"- {s}" for s in feedback["praise"]] or ["- 이번 분석에서는 뚜렷하게 뽑을 만한 강점을 찾지 못했어요."]

    top = feedback["fix_first"]
    lines += ["", f"[먼저 고쳐볼 점 TOP {len(top)}]" if len(top) > 1 else "[먼저 확인해 볼 점]" if top else "[먼저 고쳐볼 점]"]
    if not top:
        lines.append("눈에 띄게 고쳐야 할 부분은 없어요.")
    for i, card in enumerate(top, start=1):
        lines.append(f"{i}. {card['title']}")
        lines.append(f"   {card['message']}")
        if card.get("advice"):
            lines.append(f"   → {card['advice']}")
        if card.get("caveat"):
            lines.append(f"   ※ {card['caveat']}")

    if feedback["side_notes"]:
        lines += ["", "[함께 보면 좋은 점]"]
        for card in feedback["side_notes"]:
            tail = f" {card['advice']}" if card.get("advice") else ""
            lines.append(f"- {card['message']}{tail}")

    lines += ["", "[분석 참고]"]
    lines += [f"- {s}" for s in feedback["analysis_notes"]]
    return "\n".join(lines) + "\n"
