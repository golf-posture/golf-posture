"""Feedback shaped for the mobile app (``feedback_app.json``).

This is a separate view of the same analysis result that ``feedback_ko.txt`` and
``feedback_user.txt`` are made from.  Nothing here changes those two outputs: it only
reads the result dict (and a few helpers of ``user_feedback``) and builds the card
layout of the app's result screen (see docs/app_feedback_schema.md).

Where the cards come from (one pool, one score, one ordering):
  * angle features   -> one card per body part (elbow, spine), grouped over events
  * motion metrics   -> one card per metric (weight shift, shoulder turn, ...)
  * swing dynamics   -> merged into the body part's card (change between two events)
Scores (all 0-100, one scale):
  * item score   : piecewise-linear on |z| (SCORE_POINTS) -- gentle inside the pro range, steep outside
  * card score   : item score of the card's worst item (lowest score first)
  * event score  : reliability-weighted mean of the item scores that belong to the event
                   (and, in parallel, a score from the Mahalanobis combined score of that event)
  * posture score: event scores averaged with EVENT_WEIGHTS
Praise cards (the "잘한 점" tab) come from the same data and never contradict an issue.

All tunable rules are the constants right below.
"""
from __future__ import annotations

import re
from statistics import NormalDist
from typing import Any

try:  # package-style import (python -m pipeline.run_analysis)
    from pipeline.user_feedback import (
        EVENT_NAMES, FAMILIES, FAST_EVENTS, MOTION_ORDER, MOTION_TEXT, RELIABLE, TEMPO_PLAUSIBLE,
        _angle_records, _finite, _tempo,
    )
except ImportError:  # script-style import from inside pipeline/
    from user_feedback import (
        EVENT_NAMES, FAMILIES, FAST_EVENTS, MOTION_ORDER, MOTION_TEXT, RELIABLE, TEMPO_PLAUSIBLE,
        _angle_records, _finite, _tempo,
    )

SCHEMA_VERSION = 3

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
MAX_ISSUES = 5            # issue cards on the "자세 진단" tab; fewer if fewer qualify, never padded
MAX_PRAISE = 12           # cards on the "잘한 점" tab (4 body parts + 7 motion metrics + tempo = every possible card)

# Item score: piecewise-linear on |z| (구간 선형).  Inside the pro range (|z| <= 1) it stays 85+,
# "Needs Improvement" (|z| > 2) is below 50, and it reaches 0 at |z| = 4.
SCORE_POINTS = ((0.0, 100.0), (1.0, 85.0), (2.0, 50.0), (3.0, 15.0), (4.0, 0.0))
STATUS_CUTS = (50, 68)    # score < 50 priority, < 68 adjust, else recheck (temporary cut points)

# How much one item counts inside its event's score (a missing item is left out, never scored 0).
ITEM_RELIABILITY_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.2}
ITEM_RELIABILITY_DEFAULT = 0.4
# How much each event counts in the posture score (temporary starting values, to be confirmed with a coach).
EVENT_WEIGHTS = {
    "Address": 1.0, "Toe-up": 0.6, "Mid-backswing": 0.8, "Top": 1.2,
    "Mid-downswing": 1.0, "Impact": 1.5, "Mid-follow-through": 0.7, "Finish": 0.5,
}
STATUS_LABEL = {"priority": "먼저 확인해요", "adjust": "조금 더 안정적으로", "recheck": "한 번 더 확인해요"}

# Advice wording.  Direction hints ("more straight / more bent") and practice tips are simple, generic
# statements; each can be switched off without touching anything else.
USE_DIRECTION_HINTS = True
USE_DRILLS = True
USE_DETAILED_ADVICE = True    # "why it matters", a check list and a goal on top of the short advice
DRILL_BELOW_SCORE = 68    # cards scoring at or above this get no practice tip (they are only "check once more")
PHASE = {"Address": "setup", "Toe-up": "setup", "Mid-backswing": "setup", "Top": "swing",
         "Mid-downswing": "swing", "Impact": "swing", "Mid-follow-through": "finish", "Finish": "finish"}

MOTION_PENALTY_MIN = 1.0  # a motion metric becomes an issue above this penalty z (same as the text feedback)
DYNAMICS_GRADE = "Needs Improvement"   # swing-dynamics findings qualify only at this grade

EVENT_UI = {
    "Address": "어드레스", "Toe-up": "토업", "Mid-backswing": "백스윙", "Top": "탑",
    "Mid-downswing": "다운스윙", "Impact": "임팩트", "Mid-follow-through": "팔로스루", "Finish": "피니시",
}
# Body parts that can become an issue card (tilt families depend too much on the camera angle).
ISSUE_FAMILIES = {
    "elbow": {"topic": "팔꿈치 각도 차이", "who_both": "양쪽 팔꿈치",
              "advice": "슬로모션으로 프로 영상과 나란히 비교해 보세요."},
    "spine": {"topic": "상체 기울기 차이", "who_both": "상체 기울기",
              "advice": "상체 각도가 스윙 중에 달라지지 않는지 확인해 보세요."},
}
ARM_SHORT = {"lead_elbow_angle": "타깃 쪽 팔꿈치", "trail_elbow_angle": "뒤쪽 팔꿈치"}
FEATURE_LABEL = {**ARM_SHORT, "spine_tilt": "상체 기울기"}
# metric -> (event shown on the card, title noun). Whole-swing metrics sit at the event they end at.
MOTION_UI = {
    "shoulder_turn_top": ("Top", "어깨 회전"), "xfactor_turn_top": ("Top", "어깨·골반 꼬임"),
    "hip_turn_top": ("Top", "골반 회전"), "weight_shift": ("Impact", "체중 이동"),
    "head_movement": ("Impact", "머리 위치"), "hip_height_change": ("Impact", "골반 높이"),
    "stance_change": ("Impact", "발 간격"),
}
STABLE_LABEL = {"elbow": "팔꿈치 각도", "spine": "상체 기울기", "shoulder_tilt": "어깨선 기울기", "hip_tilt": "골반선 기울기"}
KEEP_TEXT = "지금 느낌을 기억해 두고, 다음 스윙에서도 같은 느낌을 유지해 보세요."

# Praise wording: what the good result means, and how to keep it.  General statements, no direction claims.
PRAISE_MEANING = {
    "shoulder_turn_top": "백스윙에서 어깨를 충분히 돌리면 몸 전체를 써서 힘을 모을 준비가 된 거예요.",
    "hip_turn_top": "골반이 알맞게 따라 돌아가면 하체와 상체가 함께 움직이는 바탕이 돼요.",
    "xfactor_turn_top": "어깨와 골반 사이에 꼬임이 있으면 다운스윙에서 힘을 풀어내기 좋은 상태예요.",
    "weight_shift": "체중이 알맞게 이동하면 임팩트에서 몸의 힘을 공으로 전달하기 쉬워요.",
    "head_movement": "머리 위치가 안정적이면 공을 일정하게 맞히기 좋은 기준이 유지돼요.",
    "hip_height_change": "자세 높이가 유지되면 임팩트까지 같은 거리와 각도로 클럽을 보내기 쉬워요.",
    "stance_change": "발 위치가 안정적이면 스윙 중 몸을 받쳐 주는 기반이 흔들리지 않아요.",
    "stable_spine": "상체 기울기가 일정하면 스윙 내내 같은 기준선에서 클럽을 휘두를 수 있어요.",
    "stable_shoulder_tilt": "어깨선이 일정하게 유지되면 몸의 정렬이 스윙 끝까지 흐트러지지 않았다는 뜻이에요.",
    "stable_hip_tilt": "골반선이 일정하면 하체가 안정적으로 받쳐 주고 있다는 신호예요.",
    "stable_elbow": "팔꿈치 각도가 프로 범위 안이면 팔을 쓰는 방식이 스윙 전반에서 안정적이에요.",
    "tempo": "백스윙에 시간을 들이고 다운스윙을 이어 가는 리듬이 프로와 비슷하다는 뜻이에요.",
}
PRAISE_KEEP = {
    "shoulder_turn_top": "촬영할 때마다 어깨가 같은 깊이로 돌아가는지 비교해 보면 이 감각을 유지하기 쉬워요.",
    "hip_turn_top": "백스윙에서 하체가 자연스럽게 따라 돌아가는 느낌을 기억해 두세요.",
    "xfactor_turn_top": "다음 촬영에서도 어깨와 골반 사이의 간격이 비슷하게 만들어지는지 비교해 보세요.",
    "weight_shift": "다운스윙에서 체중이 타깃 쪽으로 자연스럽게 옮겨가는 느낌을 기억해 두세요.",
    "head_movement": "공이 놓인 자리를 끝까지 바라보는 습관을 유지해 보세요.",
    "hip_height_change": "임팩트까지 자세 높이를 유지하는 느낌을 기억해 두세요.",
    "stance_change": "발 위치가 밀리지 않는 지금의 셋업을 매번 같게 만들어 보세요.",
    "stable_spine": "어드레스에서 만든 상체 라인을 스윙 끝까지 유지하는 감각을 기억해 두세요.",
    "stable_shoulder_tilt": "촬영 각도가 달라지면 값이 달라질 수 있으니, 같은 위치에서 찍어 비교해 보세요.",
    "stable_hip_tilt": "촬영 각도가 달라지면 값이 달라질 수 있으니, 같은 위치에서 찍어 비교해 보세요.",
    "stable_elbow": "지금 팔을 쓰는 느낌을 기억해 두고 다음 스윙에서도 같게 해 보세요.",
    "tempo": "리듬은 컨디션에 따라 달라지니, 여러 번 촬영해 평균적인 템포를 확인해 보세요.",
}
ROTATION_METRICS = {"shoulder_turn_top", "hip_turn_top", "xfactor_turn_top"}
ROTATION_CAUTION = "회전각은 정면 2D 영상에서 어깨·골반 폭의 변화로 계산한 근사값이라 실제와 조금 다를 수 있어요."
CAMERA_CAUTION = "카메라 각도의 영향을 받는 항목이라 같은 위치에서 촬영해 비교해 보는 게 좋아요."
FAST_CAUTION = "이 구간은 동작이 빨라 분석 오차가 생기기 쉬워요. 다른 각도에서 한 번 더 촬영해 비교해 보면 더 정확해요."
EMPTY_STATE = {"title": "발견된 교정 항목이 없어요", "message": "이번 스윙에서 주요 자세 문제가 발견되지 않았어요."}
NOTICE_TEXT = {
    "tempo_excluded": "이번 영상은 동작 구분이 불확실해 일부 결과는 참고용이에요.",
    "many_unavailable": "일부 자세는 관절이 잘 보이지 않아 평가하지 못했어요. 전신이 정면에서 보이게 촬영해 보세요.",
    "mock_events": "이벤트 위치를 임의로 지정한 테스트 결과예요.",
}
NOTICE_ORDER = ["tempo_excluded", "many_unavailable", "mock_events"]
ANALYSIS_NOTE_2D = "정면 영상 기준의 2D 추정이라 회전각은 근사값이에요."


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def item_score(severity: float) -> float:
    """0-100 score of one item from its |z| (piecewise linear through SCORE_POINTS)."""
    z = abs(float(severity))
    for (x0, y0), (x1, y1) in zip(SCORE_POINTS, SCORE_POINTS[1:]):
        if z <= x1:
            return y0 + (y1 - y0) * (z - x0) / (x1 - x0)
    return 0.0


def score_from_severity(severity: float) -> int:
    return int(round(item_score(severity)))


def percentile_to_score(deviation_percentile: float) -> float:
    """Combined (Mahalanobis) score -> same scale: the |z| a normal variable would need to be as unusual."""
    p = min(max(float(deviation_percentile), 0.0), 99.99)
    return item_score(NormalDist().inv_cdf(0.5 + p / 200.0))


def status_from_score(score: int) -> dict[str, str]:
    low, mid = STATUS_CUTS
    code = "priority" if score < low else "adjust" if score < mid else "recheck"
    return {"code": code, "label": STATUS_LABEL[code]}


def _num(x: Any, digits: int = 1) -> float | None:
    return round(float(x), digits) if _finite(x) else None


def _particle_ro(word: str) -> str:
    """'탑' -> '으로', '임팩트' -> '로' (direction particle after an event name)."""
    code = ord(word[-1]) if word else 0
    return "으로" if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0 else "로"


# Each body item gets a "section" so the app can show headings above the long feedback.
SECTION_TITLES = {
    "diagnosis": "무엇이 다른가요", "why": "왜 중요한가요", "check": "이렇게 확인해 보세요", "practice": "연습 방법",
    "goal": "목표", "note": "참고", "good": "잘한 점", "keep": "유지하는 방법",
}
ISSUE_SECTION = {"score": "diagnosis", "evidence": "diagnosis", "direction": "diagnosis", "also": "diagnosis",
                 "change": "diagnosis", "pattern": "diagnosis", "meaning": "why", "advice": "check", "check": "check",
                 "drill": "practice", "goal": "goal", "caution": "note"}
PRAISE_SECTION = {"score": "good", "evidence": "good", "praise": "good", "detail": "good", "meaning": "why",
                  "keep": "keep", "caution": "note"}


def _with_sections(body: list[dict[str, Any]], table: dict[str, str]) -> list[dict[str, Any]]:
    return [{**b, "section": table.get(b["kind"], "note")} for b in body]


def _particle_eul(word: str) -> str:
    """'차이' -> '를', '체중 이동' -> '을' (object particle)."""
    code = ord(word[-1]) if word else 0
    return "을" if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0 else "를"


def _score_level(score: int) -> str:
    if score >= 95:
        return "프로 중앙값과 거의 같은 수준이에요."
    if score >= 90:
        return "프로 분포의 한가운데에 가까워요."
    return "프로 범위 안에서 안정적이에요."


def _praise_body(key: str, score_text: str, evidence: str, sentence: str, detail: str = "", caution: str = "") -> list[dict[str, str]]:
    """score -> evidence -> praise -> meaning -> detail -> keep -> caution."""
    body = [{"kind": "score", "text": score_text}]
    for kind, text in (("evidence", evidence), ("praise", sentence), ("meaning", PRAISE_MEANING.get(key, "")),
                       ("detail", detail), ("keep", PRAISE_KEEP.get(key, KEEP_TEXT)), ("caution", caution)):
        if text:
            body.append({"kind": kind, "text": text})
    return body


def _span_ui(events: list[str]) -> str:
    """Short event span with the app's event names: ['Mid-downswing','Impact'] -> '다운스윙~임팩트'."""
    idx = sorted({EVENT_NAMES.index(e) for e in events if e in EVENT_NAMES})
    if not idx:
        return ""
    if len(idx) >= 5:
        return "스윙 전반"
    segments: list[list[int]] = [[idx[0]]]
    for i in idx[1:]:
        segments[-1].append(i) if i == segments[-1][-1] + 1 else segments.append([i])
    parts = [EVENT_UI[EVENT_NAMES[s[0]]] if len(s) == 1 else f"{EVENT_UI[EVENT_NAMES[s[0]]]}~{EVENT_UI[EVENT_NAMES[s[-1]]]}"
             for s in segments]
    return ", ".join(parts)


def _event_ref(event: str | None) -> dict[str, str] | None:
    return {"key": event, "label": EVENT_UI.get(event, event)} if event else None


def _media(data: dict[str, Any], *events: str | None, primary: str | None = None) -> dict[str, Any]:
    """keyframe/skeleton of ``primary`` (default: the first event), plus the keyframes of all given events."""
    infos = [((data.get("events") or {}).get(e) or {}) if e else {} for e in events]
    first = ((data.get("events") or {}).get(primary) or {}) if primary else (infos[0] if infos else {})
    return {
        "keyframe": first.get("event_frame_path"),
        "skeleton_compare": first.get("skeleton_comparison_path"),
        "keyframes": [i.get("event_frame_path") for i in infos if i.get("event_frame_path")],
    }


def _dynamics_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten data['dynamics'] into (pair, feature, grade, reliability, robust_z, delta, pro_median_delta)."""
    out = []
    for pair, pair_data in (data.get("dynamics") or {}).items():
        features = pair_data.get("features", pair_data) if isinstance(pair_data, dict) else {}
        start, _, end = pair.partition("->")
        if start not in EVENT_NAMES or end not in EVENT_NAMES:
            continue
        for feature, item in (features or {}).items():
            if not isinstance(item, dict):
                continue
            out.append({
                "pair": pair, "start": start, "end": end, "feature": feature,
                "grade": item.get("grade") or "unavailable",
                "reliability": item.get("reliability") or "unknown",
                "z": item.get("robust_z"), "delta": item.get("delta"), "pro_median": item.get("pro_median_delta"),
            })
    return out


# ---------------------------------------------------------------------------
# Items, event scores, posture score
# ---------------------------------------------------------------------------
def _score_items(data: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every measured item with the event it belongs to.

    angle    : 5 angle features x 8 events (the event itself)
    motion   : 7 whole-swing metrics (the event they end at, see MOTION_UI)
    dynamics : change between two consecutive events, 5 features x 7 pairs (the pair's end event)
    """
    items: list[dict[str, Any]] = []
    for r in records:
        if _finite(r["z"]):
            items.append({"kind": "angle", "id": r["feature"], "owner": r["event"],
                          "severity": abs(r["z"]), "reliability": r["reliability"]})
    motion = data.get("motion_metrics") or {}
    for name in MOTION_ORDER:
        item = motion.get(name)
        if item and item.get("value") is not None and _finite(item.get("penalty_z")):
            items.append({"kind": "motion", "id": name, "owner": MOTION_UI[name][0],
                          "severity": max(0.0, item["penalty_z"]), "reliability": item.get("reliability")})
    for d in _dynamics_items(data):
        if _finite(d["z"]):
            items.append({"kind": "dynamics", "id": f"{d['pair']}:{d['feature']}", "owner": d["end"],
                          "severity": abs(d["z"]), "reliability": d["reliability"]})
    for it in items:
        it["score"] = item_score(it["severity"])
        it["weight"] = ITEM_RELIABILITY_WEIGHT.get(it["reliability"], ITEM_RELIABILITY_DEFAULT)
    return items


def _event_scores(data: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for event in EVENT_NAMES:
        mine = [i for i in items if i["owner"] == event]
        wsum = sum(i["weight"] for i in mine)
        combined = (((data.get("events") or {}).get(event) or {}).get("combined_score") or {}).get("deviation_percentile")
        out[event] = {
            "items": sum(i["weight"] * i["score"] for i in mine) / wsum if wsum > 0 else None,
            "combined": percentile_to_score(combined) if _finite(combined) else None,
            "item_count": len(mine),
        }
    return out


def _weighted_event_mean(scores: dict[str, dict[str, Any]], key: str) -> float | None:
    num = den = 0.0
    for event, entry in scores.items():
        if entry[key] is not None:
            num += EVENT_WEIGHTS[event] * entry[key]
            den += EVENT_WEIGHTS[event]
    return num / den if den else None


def _round_or_none(x: float | None) -> int | None:
    return int(round(x)) if x is not None else None


# ---------------------------------------------------------------------------
# Issue candidates
# ---------------------------------------------------------------------------
ELBOW_ADVICE = {
    "setup": "준비 자세의 팔 모양이 이후 스윙의 출발점이 돼요. 거울이나 영상으로 이 순간의 팔 모양을 프로 자세와 비교해 보세요.",
    "swing": "이 구간은 팔이 빠르게 움직여요. 슬로모션으로 팔이 펴지고 굽어지는 타이밍을 프로 영상과 나란히 비교해 보세요.",
    "finish": "임팩트 이후 팔이 어떻게 마무리되는지 프로 영상과 비교해 보세요.",
}
ELBOW_DIRECTION_ADVICE = {
    "straighter": "팔에 힘이 들어가 너무 곧게 펴져 있지 않은지 확인해 보세요.",
    "bent": "팔이 너무 접혀 있지 않은지 확인해 보세요.",
}
SPINE_ADVICE = {
    "setup": "거울이나 영상으로 준비 자세의 상체 라인을 프로와 나란히 비교해 보세요.",
    "swing": "스윙 중 상체 기울기가 어드레스 때와 크게 달라지지 않는지 슬로모션으로 확인해 보세요.",
    "finish": "임팩트 이후에도 상체 라인이 크게 흔들리지 않는지, 피니시 자세를 프로 영상과 비교해 보세요.",
}
# metric -> direction -> (extra advice used when the base tip is empty, practice tip).  {trail_side} = 오른쪽/왼쪽.
MOTION_ADVICE = {
    "weight_shift": {
        "high": ("", "{trail_side} 골반 바깥에 벽이나 의자를 가볍게 대고, 백스윙 동안 골반이 닿은 곳에서 떨어지지 않게 돌려 보세요."),
        "low": ("", "다운스윙에서 타깃 쪽 발로 체중을 옮기고, 피니시에서 타깃 쪽 발 하나로 균형을 잡는 연습을 해 보세요."),
    },
    "shoulder_turn_top": {
        "high": ("탑에서 클럽이 지면과 평행에 가까운지 영상으로 확인해 보세요. 너무 크게 돌아가지 않았는지 점검하면 좋아요.",
                 "탑에서 멈추고 클럽이 지면과 평행에 가까운지 거울로 확인해 보세요."),
        "low": ("", "클럽을 가슴 앞에 가로로 안고 어깨를 천천히 돌려, 등이 타깃을 향하는지 확인해 보세요."),
    },
    "hip_turn_top": {
        "high": ("", "탑에서 하체를 단단히 잡고 어깨를 먼저 돌린다는 느낌으로 천천히 백스윙해 보세요."),
        "low": ("", "양팔을 가슴에 모으고 어깨와 골반을 함께 천천히 돌려, 하체가 자연스럽게 따라 돌아가는지 확인해 보세요."),
    },
    "xfactor_turn_top": {
        "high": ("탑에서 몸의 꼬임이 편안한 범위인지 확인해 보세요. 무리하게 돌리면 균형이 흔들릴 수 있어요.",
                 "느린 백스윙으로 균형을 유지하며 편안하게 돌아가는 범위를 확인해 보세요."),
        "low": ("", "하체는 고정한 채 어깨만 돌리는 느린 백스윙으로 어깨와 골반 사이의 간격을 익혀 보세요."),
    },
    "head_movement": {
        "high": ("", "거울 앞에서 공이 놓인 자리를 계속 바라보며 빈 스윙을 해, 머리가 좌우로 움직이지 않는지 확인해 보세요."),
    },
    "hip_height_change": {
        "high": ("", "어드레스의 무릎 각도를 유지한 채 천천히 스윙하며, 임팩트에서 일어서지 않는지 거울로 확인해 보세요."),
    },
    "stance_change": {
        "high": ("", "바닥에 발 위치를 표시하고 스윙한 뒤, 발이 표시에서 벗어났는지 확인해 보세요."),
    },
}


def _motion_evidence(item: dict[str, Any]) -> str:
    rng, med, val = item.get("pro_range_text"), item.get("pro_median_text"), item.get("value_text")
    if val and rng:
        return f"내 값은 {val}, 프로는 대부분 {rng}예요" + (f"(중앙값 {med})." if med else ".")
    if val and med:
        return f"내 값은 {val}, 프로 중앙값은 {med}예요."
    return ""


def _fmt_deg(x: float, signed: bool = False) -> str:
    """Whole degrees; a value that rounds to zero is shown as 0 (never -0)."""
    n = int(round(x))
    return "0°" if n == 0 else (f"{n:+d}°" if signed else f"{n}°")


def _pattern_text(events: list[str]) -> str | None:
    distinct = sorted({e for e in events if e in EVENT_NAMES}, key=EVENT_NAMES.index)
    if len(distinct) >= 3:
        return f"{_span_ui(distinct)}에서 반복돼요. 한 순간의 흔들림보다 스윙 흐름의 차이일 수 있어요."
    if len(distinct) == 1:
        return f"{EVENT_UI[distinct[0]]} 한 곳에서만 나타나요. 그 순간의 타이밍을 중심으로 확인해 보세요."
    return None


OPPOSITE_MIN_DEG = 3.0    # both changes must be at least this big before "opposite direction" is claimed


def _dyn_move(w: dict[str, Any]) -> str | None:
    """'opposite' / 'bigger' / 'smaller' / None: how the change between two events compares with the pros."""
    if not (_finite(w.get("delta")) and _finite(w.get("pro_median"))):
        return None
    if (w["delta"] * w["pro_median"] < 0 and abs(w["delta"]) >= OPPOSITE_MIN_DEG
            and abs(w["pro_median"]) >= OPPOSITE_MIN_DEG):
        return "opposite"
    return "bigger" if abs(w["delta"]) >= abs(w["pro_median"]) else "smaller"


def _elbow_motion(delta: float) -> str:
    """Elbow angle up = arm straightens, down = arm bends."""
    return "펴졌어요" if delta > 0 else "굽혀졌어요"


def _dyn_evidence(w: dict[str, Any]) -> str:
    if not (_finite(w.get("delta")) and _finite(w.get("pro_median"))):
        return ""
    mine, pro = abs(w["delta"]), abs(w["pro_median"])
    if _dyn_move(w) == "opposite":
        if w.get("feature") in ARM_SHORT and USE_DIRECTION_HINTS:
            pro_word = "펴지는데" if w["pro_median"] > 0 else "굽혀지는데"
            return f"프로는 이 구간에서 팔꿈치가 {round(pro)}° {pro_word}, 내 스윙은 {round(mine)}° {_elbow_motion(w['delta'])}."
        return f"프로와 반대 방향으로 변했어요(프로 {round(pro)}°, 내 스윙 {round(mine)}°)."
    pro_part = "프로는 이 구간에서 거의 변하지 않는데" if round(pro) == 0 else f"프로는 이 구간에서 보통 {round(pro)}° 정도 변하는데"
    mine_part = "내 스윙은 거의 변하지 않았어요." if round(mine) == 0 else f"내 스윙은 {round(mine)}° 변했어요."
    return f"{pro_part}, {mine_part}"


def _dyn_advice(w: dict[str, Any], subject: str) -> str:
    if _dyn_move(w) == "opposite":
        if w.get("feature") in ARM_SHORT and USE_DIRECTION_HINTS:
            return "이 구간에서 팔이 프로와 반대로 움직여요. 슬로모션으로 팔이 펴지고 굽어지는 타이밍을 프로 영상과 맞춰 비교해 보세요."
        return "이 구간에서 프로와 반대 방향으로 움직여요. 프로 영상과 이 구간을 나란히 놓고 어떻게 움직이는지 비교해 보세요."
    if _dyn_move(w) == "bigger":
        return f"이 구간에서 {subject}가 프로보다 많이 변해요. 슬로모션으로 변화가 커지는 순간을 확인해 보세요."
    return f"이 구간에서 프로는 {subject}가 변하는데 내 스윙은 변화가 작아요. 프로 영상과 이 구간을 나란히 놓고 어떻게 움직이는지 비교해 보세요."


ELBOW_MEANING = {
    "setup": "어드레스부터 백스윙 초반의 팔 모양은 이후 스윙의 궤도와 타이밍의 출발점이 돼요.",
    "swing": "탑에서 임팩트까지는 팔이 가장 빠르게 움직이는 구간이라, 팔을 펴고 굽히는 타이밍이 공에 전달되는 힘과 방향에 영향을 줄 수 있어요.",
    "finish": "임팩트 이후 팔을 어떻게 마무리하는지는 스윙을 끝까지 일정하게 이어 가는 데 도움이 돼요.",
}
ELBOW_CHECK = {
    ("setup", "straighter"): ["클럽을 들어 올릴 때 팔이 너무 곧게 펴져 있지 않은지 확인해 보세요.",
                              "팔만 들지 않고 어깨와 몸통이 함께 움직이는지 확인해 보세요."],
    ("setup", "bent"): ["팔꿈치가 너무 접혀 있지 않은지 확인해 보세요.",
                        "손과 클럽이 몸에서 너무 가까워지지 않는지 확인해 보세요."],
    ("setup", None): ["준비 자세에서 팔이 편안한 모양인지 확인해 보세요.",
                      "팔만 움직이지 않고 어깨와 몸통이 함께 움직이는지 확인해 보세요."],
    ("swing", "straighter"): ["임팩트 직전에 팔이 너무 일찍 곧게 펴지지 않는지 확인해 보세요.",
                              "팔에 힘이 들어가 어깨가 올라가지 않는지 확인해 보세요."],
    ("swing", "bent"): ["임팩트 구간에서 팔이 접힌 채로 남아 있지 않은지 확인해 보세요.",
                        "팔이 몸 쪽에 붙은 채 내려오지 않는지 확인해 보세요."],
    ("swing", None): ["팔이 펴지고 굽혀지는 순서가 프로 영상과 비슷한지 확인해 보세요.",
                      "팔에 불필요한 힘이 들어가지 않는지 확인해 보세요."],
    ("finish", "straighter"): ["팔로스루에서 팔이 너무 곧게 굳어 있지 않은지 확인해 보세요.",
                               "피니시에서 팔이 흔들리지 않는지 확인해 보세요."],
    ("finish", "bent"): ["팔로스루에서 팔이 접힌 채 멈추지 않는지 확인해 보세요.",
                         "피니시에서 팔이 몸에 붙어 있지 않은지 확인해 보세요."],
    ("finish", None): ["팔로스루에서 팔이 자연스럽게 뻗어 나가는지 확인해 보세요.",
                       "피니시에서 팔이 흔들리지 않는지 확인해 보세요."],
}
SPINE_MEANING = {
    "setup": "어드레스의 상체 기울기는 스윙 내내 클럽을 휘두르는 기준선이 돼요.",
    "swing": "스윙 중 상체 기울기가 일정하면 공과의 거리와 타이밍을 맞추기 쉬워요.",
    "finish": "마무리에서 상체 라인이 유지되면 스윙을 끝까지 안정적으로 이어 갔다는 뜻이에요.",
}
SPINE_CHECK = {
    "setup": ["상체가 한쪽으로 쏠려 있지 않은지 확인해 보세요.", "머리 위치가 양발 가운데 근처인지 확인해 보세요."],
    "swing": ["백스윙에서 상체가 어드레스 때보다 크게 일어서거나 기울지 않는지 확인해 보세요.",
              "다운스윙에서 상체가 공 쪽으로 무너지지 않는지 확인해 보세요."],
    "finish": ["피니시에서 몸이 한쪽으로 크게 쏠리지 않는지 확인해 보세요.",
               "체중이 타깃 쪽 발에 실린 채 균형을 잡고 있는지 확인해 보세요."],
}
DYN_CHECK = {   # move -> checks for the change between two events (any body part)
    "bigger": ["이 구간에서 움직임이 한꺼번에 커지지 않는지 확인해 보세요.", "머리와 상체 라인이 처음 위치에서 크게 벗어나지 않는지 확인해 보세요."],
    "smaller": ["프로는 이 구간에서 움직이는데 내 스윙은 몸이 굳어 있지 않은지 확인해 보세요.", "팔과 몸통이 함께 움직이는지 확인해 보세요."],
    "opposite": ["이 구간에서 프로와 반대 방향으로 움직이고 있지 않은지 영상으로 확인해 보세요.",
                 "몸통과 팔 중 무엇이 먼저 움직이는지 프로 영상과 비교해 보세요."],
}
# metric -> direction -> (why it matters, two checks, extra goal sentence)
MOTION_DETAIL = {
    "weight_shift": {
        "high": ("체중이 옆으로 많이 이동하면 몸의 중심이 흔들려 공을 맞히는 타이밍이 달라질 수 있어요.",
                 ["백스윙에서 골반이 뒤쪽으로 밀려나지 않는지 확인해 보세요.", "어드레스 때보다 머리와 몸 전체가 옆으로 움직이지 않는지 확인해 보세요."], ""),
        "low": ("체중 이동이 적으면 임팩트에서 몸의 힘을 공으로 전달하기 어려울 수 있어요.",
                ["다운스윙에서 체중이 타깃 쪽 발로 옮겨가는지 확인해 보세요.", "피니시에서 타깃 쪽 발에 체중이 실려 있는지 확인해 보세요."], ""),
    },
    "shoulder_turn_top": {
        "high": ("어깨가 너무 크게 돌아가면 탑에서 자세가 흐트러지거나 스윙이 커질 수 있어요.",
                 ["탑에서 클럽이 지면과 평행에 가까운지 확인해 보세요.", "하체가 함께 너무 많이 돌아가지 않는지 확인해 보세요."],
                 "무리하게 줄이기보다 편안하게 돌아가는 범위에서 조절해 보세요."),
        "low": ("어깨 회전이 부족하면 몸 전체의 힘을 모으기 어려워 팔에 의존하기 쉬워요.",
                ["탑에서 등이 타깃 쪽을 향하는지 확인해 보세요.", "어깨가 돌기 전에 팔만 먼저 올라가지 않는지 확인해 보세요."],
                "무리하게 돌리기보다 편안하게 돌아가는 범위에서 조금씩 늘려 보세요."),
    },
    "hip_turn_top": {
        "high": ("골반이 너무 많이 돌아가면 어깨와의 간격이 줄어 힘을 모으기 어려울 수 있어요.",
                 ["탑에서 하체가 단단히 버티고 있는지 확인해 보세요.", "어깨가 먼저 돌아가는 느낌이 있는지 확인해 보세요."], ""),
        "low": ("골반이 덜 돌아가면 어깨만 돌아가 몸의 연결이 끊기기 쉬워요.",
                ["백스윙에서 골반이 어깨와 함께 따라 돌아가는지 확인해 보세요.", "무릎과 골반이 너무 굳어 있지 않은지 확인해 보세요."],
                "무리하게 돌리기보다 편안하게 돌아가는 범위에서 조금씩 늘려 보세요."),
    },
    "xfactor_turn_top": {
        "high": ("어깨와 골반 사이의 꼬임이 너무 크면 균형이 흔들릴 수 있어요.",
                 ["탑에서 균형을 유지하고 있는지 확인해 보세요.", "어깨나 허리에 무리한 힘이 들어가지 않는지 확인해 보세요."],
                 "무리하게 줄이기보다 편안하게 돌아가는 범위에서 조절해 보세요."),
        "low": ("어깨와 골반 사이의 꼬임이 작으면 다운스윙에서 풀어낼 힘이 적을 수 있어요.",
                ["탑에서 어깨가 골반보다 더 많이 돌아가 있는지 확인해 보세요.", "하체가 어깨와 함께 따라 돌아가지 않는지 확인해 보세요."],
                "무리하게 키우기보다 편안하게 돌아가는 범위에서 조금씩 늘려 보세요."),
    },
    "head_movement": {
        "high": ("머리 위치가 많이 움직이면 공과의 거리가 달라져 일정하게 맞히기 어려워요.",
                 ["백스윙에서 머리가 좌우로 밀리지 않는지 확인해 보세요.", "공이 놓인 자리를 끝까지 바라보고 있는지 확인해 보세요."], ""),
    },
    "hip_height_change": {
        "high": ("임팩트에서 골반이 올라가면 공과의 거리가 달라져 임팩트가 흔들릴 수 있어요.",
                 ["다운스윙에서 무릎이 펴지며 일어서지 않는지 확인해 보세요.", "어드레스 때의 무릎 각도가 임팩트까지 유지되는지 확인해 보세요."], ""),
    },
    "stance_change": {
        "high": ("스윙 중 발이 밀리면 몸을 받치는 기반이 흔들려 균형을 유지하기 어려워요.",
                 ["스윙 중 발바닥이 지면에서 미끄러지거나 들리지 않는지 확인해 보세요.", "체중이 발 안쪽으로만 모이지 않는지 확인해 보세요."], ""),
    },
}
REPEAT_STEP = "5회 반복한 뒤 스윙을 다시 촬영해 점수가 올랐는지 확인해요."


def _goal_text(score: int, extra: str = "") -> str:
    low, mid = STATUS_CUTS
    pro = int(round(item_score(1.0)))
    if score < low:
        base = f"다음 촬영에서는 먼저 {mid}점 이상('한 번 더 확인' 단계)을 목표로 해 보세요. {pro}점 이상이면 프로 범위 안이에요."
    elif score < mid:
        base = f"다음 촬영에서는 {pro}점 이상(프로 범위 안)을 목표로 해 보세요."
    else:
        base = "프로 범위에 가까워요. 다음 촬영에서도 같은 결과가 나오는지 확인해 보세요."
    return f"{base} {extra}".strip()


def _dyn_steps(w: dict[str, Any]) -> list[str]:
    span = f"{EVENT_UI[w['start']]}~{EVENT_UI[w['end']]}"
    middle = ("클럽 없이 같은 속도로 따라 하며 움직임이 한꺼번에 커지지 않게 해요." if _dyn_move(w) == "bigger"
              else "클럽 없이 같은 속도로 천천히 따라 하며 움직이는 순서를 프로 영상과 맞춰 봐요.")
    return [f"프로 영상의 {span} 구간을 0.5배속으로 반복해서 봐요.", middle, REPEAT_STEP]


def _elbow_steps(event: str, direction: str | None) -> list[str]:
    ev = EVENT_UI[event]
    tail = {"straighter": "팔꿈치에 약간 여유가 있는 모양인지 확인해요.",
            "bent": "팔이 자연스럽게 뻗어 있는지 확인해요."}.get(direction or "", "팔 모양을 프로 자세와 비교해요.")
    return [f"거울 앞에서 {ev}의 팔 모양을 만들고 사진으로 남겨 프로 자세와 비교해요.",
            f"클럽 없이 천천히 스윙하며 {ev}에서 멈추고, {tail}", REPEAT_STEP]


def _spine_steps(phase: str) -> list[str]:
    if phase == "setup":
        return ["어드레스 자세를 정면에서 촬영해요.", "스윙 후 같은 위치에서 다시 촬영해 상체 라인을 겹쳐 비교해요.",
                "5회 반복해 어드레스 라인이 매번 같은지 확인해요."]
    return ["거울 앞에서 클럽 없이 천천히 스윙해요.", "머리와 상체 라인이 처음 위치에서 크게 벗어나지 않는지 확인해요.",
            "10회 반복한 뒤 정면에서 촬영해 비교해요."]


def _part_label(feature: str) -> str:
    return f"{ARM_SHORT[feature]} 각도" if feature in ARM_SHORT else FEATURE_LABEL.get(feature, "각도")


def _evidence_angle(r: dict[str, Any]) -> str:
    value, mean, std = r.get("value"), r.get("pro_mean"), r.get("pro_std")
    if not (_finite(value) and _finite(mean)):
        return ""
    tail = f"(프로는 보통 ±{std:.0f}° 이내)" if _finite(std) and std > 0 else ""
    return f"내 값은 {value:.0f}°, 프로 평균은 {mean:.0f}°예요{tail}."


def _family_body(fam_name, angle, worst_angle, worst_dyn, sev_angle, sev_dyn, card_score):
    """diagnosis (score, evidence, direction, also, change, pattern) -> why -> check -> practice -> goal.

    Direction hints are only made where the measure has a plain geometric meaning (elbow angle: larger = more
    straight).  Everything else compares sizes (how much it changes) and never says which way."""
    body: list[dict[str, Any]] = []
    noun = "팔꿈치" if fam_name == "elbow" else "상체"
    what = "각도" if fam_name == "elbow" else "기울기"
    subject = f"{noun} {what}"

    def add(kind, text, **extra):
        if text:
            body.append({"kind": kind, "text": text, **extra})

    def dyn_label(w):
        end_ko = EVENT_UI[w["end"]]
        return f"{EVENT_UI[w['start']]}에서 {end_ko}{_particle_ro(end_ko)} 가는 동안 {subject} 변화"

    angle_primary = worst_angle is not None and sev_angle >= sev_dyn
    if angle_primary:
        r = worst_angle
        phase = PHASE[r["event"]]
        score = score_from_severity(r["z"])
        add("score", f"{EVENT_UI[r['event']]}에서 {_part_label(r['feature'])} 점수가 {score}점으로 가장 낮아요.")
        add("evidence", _evidence_angle(r))
        direction = None
        if fam_name == "elbow" and USE_DIRECTION_HINTS and _finite(r["z"]):
            direction = "straighter" if r["z"] > 0 else "bent"
            word = "더 펴져" if direction == "straighter" else "더 굽혀져"
            add("direction", f"{ARM_SHORT[r['feature']]}가 프로보다 {word} 있어요.")
        others = sorted((x for x in angle if x is not r), key=lambda x: -abs(x["z"]))[:2]
        if others:
            parts = ", ".join(f"{EVENT_UI[x['event']]} {ARM_SHORT.get(x['feature'], FEATURE_LABEL.get(x['feature'], ''))}({score_from_severity(x['z'])}점)"
                              for x in others)
            add("also", f"같은 문제가 {parts}에서도 보여요.")
        if worst_dyn is not None:
            add("change", f"{dyn_label(worst_dyn)}도 점수가 {score_from_severity(worst_dyn['z'])}점으로 낮아요.")
        add("pattern", _pattern_text([x["event"] for x in angle]))
        if USE_DETAILED_ADVICE:
            add("meaning", (ELBOW_MEANING if fam_name == "elbow" else SPINE_MEANING)[phase])
        add("advice", (ELBOW_ADVICE if fam_name == "elbow" else SPINE_ADVICE)[phase])
        if USE_DETAILED_ADVICE:
            for text in (ELBOW_CHECK[(phase, direction)] if fam_name == "elbow" else SPINE_CHECK[phase]):
                add("check", text)
        elif direction:
            add("advice", ELBOW_DIRECTION_ADVICE[direction])
        if USE_DRILLS and card_score < DRILL_BELOW_SCORE:
            steps = _elbow_steps(r["event"], direction) if fam_name == "elbow" else _spine_steps(phase)
            for i, text in enumerate(steps, start=1):
                add("drill", text, step=i)
        worst_item = {"kind": "angle", "event": _event_ref(r["event"]), "label": _part_label(r["feature"]), "score": score}
    else:
        w = worst_dyn
        phase = PHASE[w["end"]]
        score = score_from_severity(w["z"])
        add("score", f"{dyn_label(w)} 점수가 {score}점이에요.")
        add("evidence", _dyn_evidence(w))
        if worst_angle is not None:
            r = worst_angle
            add("also", f"{EVENT_UI[r['event']]}에서 {_part_label(r['feature'])} 점수도 {score_from_severity(r['z'])}점으로 낮아요.")
        if USE_DETAILED_ADVICE:
            add("meaning", (ELBOW_MEANING if fam_name == "elbow" else SPINE_MEANING)[phase])
        add("advice", _dyn_advice(w, subject))
        if USE_DETAILED_ADVICE:
            for text in DYN_CHECK[_dyn_move(w) or "smaller"]:
                add("check", text)
        if USE_DRILLS and card_score < DRILL_BELOW_SCORE:
            for i, text in enumerate(_dyn_steps(w), start=1):
                add("drill", text, step=i)
        worst_item = {"kind": "dynamics", "event": _event_ref(w["end"]), "label": f"{noun} {what} 변화", "score": score}
    if USE_DETAILED_ADVICE:
        add("goal", _goal_text(card_score))
    return body, worst_item


def _family_candidates(data: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dyn_all = _dynamics_items(data)
    cards = []
    for fam_name, spec in ISSUE_FAMILIES.items():
        features = FAMILIES[fam_name]["features"]
        angle = [r for r in records if r["feature"] in features and r["reliability"] in RELIABLE
                 and _finite(r["z"]) and r["grade"] == "Needs Improvement"]
        dyn = [d for d in dyn_all if d["feature"] in features and d["reliability"] != "low"
               and d["grade"] == DYNAMICS_GRADE and _finite(d["z"])]
        if not angle and not dyn:
            continue
        worst_angle = max(angle, key=lambda r: abs(r["z"])) if angle else None
        worst_dyn = max(dyn, key=lambda d: abs(d["z"])) if dyn else None
        sev_angle = abs(worst_angle["z"]) if worst_angle else -1.0
        sev_dyn = abs(worst_dyn["z"]) if worst_dyn else -1.0
        severity = max(sev_angle, sev_dyn)
        rep_event = worst_angle["event"] if sev_angle >= sev_dyn else worst_dyn["end"]

        arms = [f for f in features if any(r["feature"] == f for r in angle)]
        who = spec["who_both"] if fam_name == "spine" or len(arms) != 1 else ARM_SHORT[arms[0]]
        if fam_name == "elbow" and len(arms) == 2:
            who = spec["who_both"]
        if angle:
            span = _span_ui([r["event"] for r in angle])
            subtitle = f"{span}에서 {who} 각도가 프로와 달라요." if fam_name == "elbow" else f"{span}에서 {who}가 프로와 달라요."
        else:
            w = worst_dyn
            end_ko = EVENT_UI[w["end"]]
            what = "각도" if fam_name == "elbow" else "기울기"
            noun = "팔꿈치" if fam_name == "elbow" else "상체"
            subtitle = f"{EVENT_UI[w['start']]}에서 {end_ko}{_particle_ro(end_ko)} 가는 동안 {noun} {what} 변화가 프로와 달라요."

        body, worst_item = _family_body(fam_name, angle, worst_angle, worst_dyn, sev_angle, sev_dyn,
                                        score_from_severity(severity))
        events_used = sorted({r["event"] for r in angle} | ({worst_dyn["start"], worst_dyn["end"]} if worst_dyn else set()),
                             key=EVENT_NAMES.index)
        fast = any(e in FAST_EVENTS for e in ([r["event"] for r in angle] + ([worst_dyn["end"]] if worst_dyn else [])))
        if fast:
            body.append({"kind": "caution", "text": FAST_CAUTION})

        stats = []
        for feature in features:
            rows = [r for r in angle if r["feature"] == feature]
            if rows:
                lead = max(rows, key=lambda r: abs(r["z"]))
                stats.append({"label": FEATURE_LABEL[feature], "at": EVENT_UI[lead["event"]],
                              "value_text": f"{lead['value']:.0f}°" if _finite(lead.get("value")) else None,
                              "pro_text": f"{lead['pro_mean']:.0f}°" if _finite(lead.get("pro_mean")) else None,
                              "range_text": None, "tail_pct": None})
        if worst_dyn:
            w = worst_dyn
            stats.append({"label": f"{EVENT_UI[w['start']]}→{EVENT_UI[w['end']]} 변화량", "at": EVENT_UI[w["end"]],
                          "value_text": _fmt_deg(w["delta"], signed=True) if _finite(w["delta"]) else None,
                          "pro_text": _fmt_deg(w["pro_median"], signed=True) if _finite(w["pro_median"]) else None,
                          "range_text": None, "tail_pct": None})

        pair_events = [worst_dyn["start"], worst_dyn["end"]] if (worst_dyn and sev_dyn > sev_angle) else [rep_event]
        cards.append({
            "id": fam_name, "severity": severity, "event": rep_event, "topic": spec["topic"],
            "subtitle": subtitle, "body": body, "stats": stats, "events": events_used, "worst_item": worst_item,
            "sources": (["angle"] if angle else []) + (["dynamics"] if worst_dyn else []),
            "media_events": pair_events,
        })
    return cards


def _motion_candidates(motion: dict[str, Any], handedness: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (issue candidates, praise cards)."""
    trail_side = "왼쪽" if str(handedness or "").lower().startswith("l") else "오른쪽"
    issues, praise = [], []
    for name in MOTION_ORDER:
        item, text = motion.get(name), MOTION_TEXT.get(name)
        if not item or not text or item.get("value") is None or not _finite(item.get("penalty_z")):
            continue
        penalty = item["penalty_z"]
        reliable = item.get("reliability") in RELIABLE
        fmt = {"val": item.get("value_text") or "", "med": item.get("pro_median_text") or ""}
        event, noun = MOTION_UI[name]
        stat = {"label": text["label"], "at": EVENT_UI[event], "value_text": item.get("value_text"),
                "pro_text": item.get("pro_median_text"), "range_text": item.get("pro_range_text"),
                "tail_pct": _num(item.get("tail_pct"))}
        if penalty > MOTION_PENALTY_MIN:
            problem, tip = text.get(item.get("direction") or "", ("", ""))
            if not problem or not reliable:
                continue
            score = score_from_severity(penalty)
            body = [{"kind": "score", "text": f"{noun} 점수가 {score}점이에요."}]
            if _motion_evidence(item):
                body.append({"kind": "evidence", "text": _motion_evidence(item)})
            direction = item.get("direction") or ""
            extra, drill = MOTION_ADVICE.get(name, {}).get(direction, ("", ""))
            why, checks, goal_extra = MOTION_DETAIL.get(name, {}).get(direction, ("", [], ""))
            if USE_DETAILED_ADVICE and why:
                body.append({"kind": "meaning", "text": why})
            if tip or extra:
                body.append({"kind": "advice", "text": tip or extra})
            if USE_DETAILED_ADVICE:
                body += [{"kind": "check", "text": t} for t in checks]
            if USE_DRILLS and drill and score < DRILL_BELOW_SCORE:
                steps = ["거울 앞에서 어드레스 자세를 잡아요.", re.sub(r"\s보세요\.$", "요.", drill.format(trail_side=trail_side)), REPEAT_STEP]
                body += [{"kind": "drill", "text": t, "step": i} for i, t in enumerate(steps, start=1)]
            if USE_DETAILED_ADVICE:
                rng, med = item.get("pro_range_text"), item.get("pro_median_text")
                aim = f"값으로는 프로 대부분 범위({rng}) 안이 기준이에요." if rng else (f"값으로는 프로 중앙값({med})에 가까워지는 게 방향이에요." if med else "")
                body.append({"kind": "goal", "text": _goal_text(score, " ".join(x for x in (aim, goal_extra) if x))})
            issues.append({"id": name, "severity": penalty, "event": event, "topic": f"{noun} 차이",
                           "subtitle": problem.format(**fmt), "body": body, "stats": [stat],
                           "events": [event], "sources": ["motion"], "media_events": [event],
                           "worst_item": {"kind": "motion", "event": _event_ref(event), "label": noun, "score": score}})
        elif reliable:
            up = _finite(item.get("robust_z")) and abs(item["robust_z"]) > 1.0
            sentence = text["up" if up else "ok"].format(**fmt)
            score = score_from_severity(max(0.0, penalty))
            body = _praise_body(
                name, f"{noun} 점수가 {score}점으로, {_score_level(score)}", _motion_evidence(item), sentence,
                caution=ROTATION_CAUTION if up and name in ROTATION_METRICS else "")
            praise.append({"id": name, "severity": max(0.0, penalty), "event": event, "title": f"{EVENT_UI[event]} {noun}",
                           "subtitle": sentence, "kind": "motion", "body": body})
    return issues, praise


def _stable_praise(records: list[dict[str, Any]], blocked: set[str]) -> list[dict[str, Any]]:
    cards = []
    for fam_name, label in STABLE_LABEL.items():
        if fam_name in blocked:
            continue
        rows = [r for r in records if r["feature"] in FAMILIES[fam_name]["features"]
                and r["reliability"] in RELIABLE and r["grade"] != "unavailable"]
        if len(rows) >= 5 and all(r["grade"] == "Excellent" for r in rows):
            scored = [r for r in rows if _finite(r["z"])]
            best = min(scored, key=lambda r: abs(r["z"])) if scored else rows[0]
            worst = max((abs(r["z"]) for r in scored), default=0.0)
            sentence = f"어드레스부터 피니시까지 {label}가 일정하게 유지돼요."
            low = score_from_severity(worst)
            worst_row = max(scored, key=lambda r: abs(r["z"])) if scored else best
            best_score = score_from_severity(abs(best["z"])) if scored else None
            if not scored:
                detail = ""
            elif best_score - low <= 3 or best["event"] == worst_row["event"]:
                detail = f"평가된 구간이 {low}~{best_score}점으로 고르게 안정적이에요."
            else:
                detail = (f"가장 안정적인 구간은 {EVENT_UI[best['event']]}({best_score}점)이고, "
                          f"가장 낮은 구간도 {EVENT_UI[worst_row['event']]}에서 {low}점이에요.")
            body = _praise_body(
                f"stable_{fam_name}", f"{label} 점수가 평가된 모든 이벤트에서 {low}점 이상이에요.",
                f"평가된 {len(rows)}개 항목이 모두 프로 범위 안이에요.", sentence, detail=detail,
                caution=CAMERA_CAUTION if fam_name in ("shoulder_tilt", "hip_tilt") else "")
            cards.append({"id": f"stable_{fam_name}", "severity": worst, "event": best["event"], "title": f"{label} 안정",
                          "subtitle": sentence, "kind": "stable", "body": body})
    return cards


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _issue_card(rank: int, c: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    score = score_from_severity(c["severity"])
    return {
        "rank": rank, "id": c["id"], "score": score, "status": status_from_score(score),
        "event": _event_ref(c["event"]), "title": f"{EVENT_UI[c['event']]} {c['topic']}", "topic": c["topic"],
        "subtitle": c["subtitle"], "body": _with_sections(c["body"], ISSUE_SECTION), "worst_item": c.get("worst_item"),
        "stats": c["stats"], "sources": c["sources"],
        "media": _media(data, *c["media_events"], primary=c["event"]),
    }


def _praise_card(c: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": c["id"], "title": c["title"], "subtitle": c["subtitle"], "score": score_from_severity(c["severity"]),
        "event": _event_ref(c["event"]), "body": _with_sections(c.get("body", []), PRAISE_SECTION),
        "media": _media(data, c["event"]),
    }


def _praise_header(praise: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, str] | None:
    """One line for the top of the praise tab: how many, and what to look at next."""
    if not praise:
        return None
    message = f"잘한 점 {len(praise)}가지를 찾았어요."
    if issues:
        title = issues[0]["title"]
        message += f" 이 부분은 그대로 유지하고, '{title}'{_particle_eul(title)} 먼저 확인해 보세요."
    else:
        message += " 이번 스윙에서는 교정할 항목이 발견되지 않았어요."
    return {"title": "이번 스윙에서 잘한 점", "message": message}


def _tempo_summary(data: dict[str, Any]) -> dict[str, Any]:
    item = (data.get("sequence_features") or {}).get("tempo_ratio") or {}
    value, mean = item.get("value"), item.get("pro_mean")
    low, high = TEMPO_PLAUSIBLE
    ok = _finite(value) and low <= value <= high
    return {"status": "ok" if ok else "excluded", "ratio": _num(value) if ok else None,
            "text": f"{value:.1f} : 1" if ok else "-", "caption": "백스윙 : 다운스윙", "pro_mean": _num(mean)}


def build_app_payload(data: dict[str, Any]) -> dict[str, Any]:
    """The JSON the app renders. Field meanings: docs/app_feedback_schema.md."""
    records = _angle_records_with_values(data)
    motion = data.get("motion_metrics") or {}

    fam = _family_candidates(data, records)
    mot_issues, mot_praise = _motion_candidates(motion, data.get("handedness"))
    ranked = sorted(fam + mot_issues, key=lambda c: (-c["severity"], c["id"]))
    issues = [_issue_card(i, c, data) for i, c in enumerate(ranked[:MAX_ISSUES], start=1)]

    blocked = {c["id"] for c in ranked}                      # a praised body part must not also be an issue
    praise_items = _stable_praise(records, blocked) + mot_praise
    tempo_praise, _tempo_side, tempo_note = _tempo(data)
    if tempo_praise:
        tempo = _tempo_summary(data)
        evidence = (f"내 리듬은 {tempo['text']}, 프로 평균은 {tempo['pro_mean']:.1f} : 1이에요."
                    if tempo["status"] == "ok" and tempo["pro_mean"] is not None else "")
        body = _praise_body("tempo", "", evidence, tempo_praise,
                            detail="백스윙이 다운스윙보다 약 {:.1f}배 길다는 뜻이에요.".format(tempo["ratio"]) if tempo["ratio"] else "")
        body = [b for b in body if b["kind"] != "score" or b["text"]]
        praise_items.append({"id": "tempo", "severity": 0.0, "event": None, "title": "스윙 리듬",
                             "subtitle": tempo_praise, "kind": "tempo", "body": body})
    praise_items.sort(key=lambda c: c["severity"])           # strongest praise first (stable sort keeps the order otherwise)
    praise = [_praise_card(c, data) for c in praise_items[:MAX_PRAISE]]

    items = _score_items(data, records)
    event_scores = _event_scores(data, items)
    posture = _weighted_event_mean(event_scores, "items")
    posture_alt = _weighted_event_mean(event_scores, "combined")
    n_unavail = sum(r["grade"] == "unavailable" for r in records)
    reasons = (["tempo_excluded"] if tempo_note else []) + (["many_unavailable"] if n_unavail >= 3 else []) \
        + (["mock_events"] if data.get("mock_events") else [])
    reason = next((r for r in NOTICE_ORDER if r in reasons), None)
    notes = ([tempo_note] if tempo_note else []) + [ANALYSIS_NOTE_2D]

    by_event: dict[str, list[str]] = {}
    for card in issues:
        if card["event"]:
            by_event.setdefault(card["event"]["key"], []).append(card["id"])
    events = [{"key": n, "label": EVENT_UI[n], "frame_index": ((data.get("events") or {}).get(n) or {}).get("frame_index"),
               "score": {"items": _round_or_none(event_scores[n]["items"]), "combined": _round_or_none(event_scores[n]["combined"]),
                         "weight": EVENT_WEIGHTS[n], "item_count": event_scores[n]["item_count"]},
               "issue_ids": by_event.get(n, []), "media": _media(data, n)} for n in EVENT_NAMES]

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {"handedness": data.get("handedness"), "mock_events": bool(data.get("mock_events"))},
        "state": "issues" if issues else "no_issues",
        "reliability": "low" if reasons else "normal",
        "notice": {"code": reason, "message": NOTICE_TEXT[reason]} if reason else None,
        "summary": {
            "posture_score": _round_or_none(posture),
            "posture_score_basis": "event_weighted_items",
            "posture_score_alt": {"basis": "combined_mahalanobis", "value": _round_or_none(posture_alt)},
            "event_weights": dict(EVENT_WEIGHTS),
            "analyzed_items": len(items),
            "tempo": _tempo_summary(data),
        },
        "issues": issues,
        "empty_state": None if issues else EMPTY_STATE,
        "praise": praise,
        "praise_header": _praise_header(praise, issues),
        "section_titles": dict(SECTION_TITLES),
        "events": events,
        "analysis_notes": notes,
    }


def _angle_records_with_values(data: dict[str, Any]) -> list[dict[str, Any]]:
    """user_feedback's angle records plus the raw value / pro mean needed for the numbers on a card."""
    records = _angle_records(data)
    for r in records:
        item = (((data.get("events") or {}).get(r["event"]) or {}).get("feedback") or {}).get(r["feature"]) or {}
        r["value"], r["pro_mean"], r["pro_std"] = item.get("value"), item.get("pro_mean"), item.get("pro_std")
    return records
