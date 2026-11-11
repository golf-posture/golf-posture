"""Feedback shaped for the mobile app (``feedback_app.json``).

This is a separate view of the same analysis result that ``feedback_ko.txt`` and
``feedback_user.txt`` are made from.  Nothing here changes those two outputs: it only
reads the result dict (and a few helpers of ``user_feedback``) and builds the card
layout of the app's result screen (see docs/app_feedback_schema.md).

Where the cards come from (one pool, one score, one ordering):
  * angle features   -> one card per body part (elbow, spine), grouped over events
  * motion metrics   -> one card per metric (weight shift, shoulder turn, ...)
  * swing dynamics   -> merged into the body part's card (change between two events)
A card's score is ``100 - 30 x |z|`` of its worst component, lowest score first.
Praise cards (the "잘한 점" tab) come from the same data and never contradict an issue.

All tunable rules are the constants right below.
"""
from __future__ import annotations

from typing import Any

try:  # package-style import (python -m pipeline.run_analysis)
    from pipeline.user_feedback import (
        EVENT_NAMES, FAMILIES, FAST_EVENTS, MOTION_ORDER, MOTION_TEXT, RELIABLE, TEMPO_PLAUSIBLE,
        _angle_records, _count_reliable, _finite, _tempo,
    )
except ImportError:  # script-style import from inside pipeline/
    from user_feedback import (
        EVENT_NAMES, FAMILIES, FAST_EVENTS, MOTION_ORDER, MOTION_TEXT, RELIABLE, TEMPO_PLAUSIBLE,
        _angle_records, _count_reliable, _finite, _tempo,
    )

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
MAX_ISSUES = 5            # issue cards on the "자세 진단" tab; fewer if fewer qualify, never padded
MAX_PRAISE = 8            # cards on the "잘한 점" tab

# 0-100 score of one finding: same slope as the per-feature score in run_analysis / motion_metrics.
SCORE_PER_Z = 30.0        # |z| 1 -> 70, |z| 2 -> 40
STATUS_CUTS = (40, 55)    # score < 40 priority, < 55 adjust, else recheck (temporary cut points)
STATUS_LABEL = {"priority": "먼저 확인해요", "adjust": "조금 더 안정적으로", "recheck": "한 번 더 확인해요"}

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
STABLE_LABEL = {"spine": "상체 기울기", "shoulder_tilt": "어깨선 기울기", "hip_tilt": "골반선 기울기"}
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
def score_from_severity(severity: float) -> int:
    return int(max(0, min(100, round(100.0 - SCORE_PER_Z * severity))))


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


def _media(data: dict[str, Any], *events: str | None) -> dict[str, Any]:
    """keyframe/skeleton of the first event, plus the keyframes of all given events."""
    infos = [((data.get("events") or {}).get(e) or {}) if e else {} for e in events]
    first = infos[0] if infos else {}
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
# Issue candidates
# ---------------------------------------------------------------------------
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

        body = []
        if angle and worst_dyn:
            end_ko = EVENT_UI[worst_dyn["end"]]
            body.append({"kind": "change",
                         "text": f"{EVENT_UI[worst_dyn['start']]}에서 {end_ko}{_particle_ro(end_ko)} 가는 동안의 변화도 프로와 달라요."})
        body.append({"kind": "advice", "text": spec["advice"]})
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
                          "value_text": f"{w['delta']:+.0f}°" if _finite(w["delta"]) else None,
                          "pro_text": f"{w['pro_median']:+.0f}°" if _finite(w["pro_median"]) else None,
                          "range_text": None, "tail_pct": None})

        pair_events = [worst_dyn["start"], worst_dyn["end"]] if (worst_dyn and sev_dyn > sev_angle) else [rep_event]
        cards.append({
            "id": fam_name, "severity": severity, "event": rep_event, "topic": spec["topic"],
            "subtitle": subtitle, "body": body, "stats": stats, "events": events_used,
            "sources": (["angle"] if angle else []) + (["dynamics"] if worst_dyn else []),
            "media_events": pair_events,
        })
    return cards


def _motion_candidates(motion: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (issue candidates, praise cards)."""
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
            body = [{"kind": "advice", "text": tip}] if tip else []
            issues.append({"id": name, "severity": penalty, "event": event, "topic": f"{noun} 차이",
                           "subtitle": problem.format(**fmt), "body": body, "stats": [stat],
                           "events": [event], "sources": ["motion"], "media_events": [event]})
        elif reliable:
            up = _finite(item.get("robust_z")) and abs(item["robust_z"]) > 1.0
            praise.append({"id": name, "severity": max(0.0, penalty), "event": event, "title": f"{EVENT_UI[event]} {noun}",
                           "subtitle": text["up" if up else "ok"].format(**fmt), "kind": "motion"})
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
            cards.append({"id": f"stable_{fam_name}", "severity": worst, "event": best["event"], "title": f"{label} 안정",
                          "subtitle": f"어드레스부터 피니시까지 {label}가 일정하게 유지돼요.", "kind": "stable"})
    return cards


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _issue_card(rank: int, c: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    score = score_from_severity(c["severity"])
    return {
        "rank": rank, "id": c["id"], "score": score, "status": status_from_score(score),
        "event": _event_ref(c["event"]), "title": f"{EVENT_UI[c['event']]} {c['topic']}", "topic": c["topic"],
        "subtitle": c["subtitle"], "body": c["body"], "stats": c["stats"], "sources": c["sources"],
        "media": _media(data, *c["media_events"]),
    }


def _praise_card(c: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": c["id"], "title": c["title"], "subtitle": c["subtitle"], "score": score_from_severity(c["severity"]),
        "event": _event_ref(c["event"]), "media": _media(data, c["event"]),
    }


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
    mot_issues, mot_praise = _motion_candidates(motion)
    ranked = sorted(fam + mot_issues, key=lambda c: (-c["severity"], c["id"]))
    issues = [_issue_card(i, c, data) for i, c in enumerate(ranked[:MAX_ISSUES], start=1)]

    blocked = {c["id"] for c in ranked}                      # a praised body part must not also be an issue
    praise_items = _stable_praise(records, blocked) + mot_praise
    tempo_praise, _tempo_side, tempo_note = _tempo(data)
    if tempo_praise:
        praise_items.append({"id": "tempo", "severity": 0.0, "event": None, "title": "스윙 리듬",
                             "subtitle": tempo_praise, "kind": "tempo"})
    praise_items.sort(key=lambda c: c["severity"])           # strongest praise first (stable sort keeps the order otherwise)
    praise = [_praise_card(c, data) for c in praise_items[:MAX_PRAISE]]

    ok, total = _count_reliable(records, motion)
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
               "issue_ids": by_event.get(n, []), "media": _media(data, n)} for n in EVENT_NAMES]

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {"handedness": data.get("handedness"), "mock_events": bool(data.get("mock_events"))},
        "state": "issues" if issues else "no_issues",
        "reliability": "low" if reasons else "normal",
        "notice": {"code": reason, "message": NOTICE_TEXT[reason]} if reason else None,
        "summary": {
            "posture_score": round(10.0 * ok / total, 1) if total else None,
            "posture_score_basis": "within_pro_ratio",
            "score_inputs": {"within_pro": ok, "analyzed": total},
            "tempo": _tempo_summary(data),
        },
        "issues": issues,
        "empty_state": None if issues else EMPTY_STATE,
        "praise": praise,
        "events": events,
        "analysis_notes": notes,
    }


def _angle_records_with_values(data: dict[str, Any]) -> list[dict[str, Any]]:
    """user_feedback's angle records plus the raw value / pro mean needed for the numbers on a card."""
    records = _angle_records(data)
    for r in records:
        item = (((data.get("events") or {}).get(r["event"]) or {}).get("feedback") or {}).get(r["feature"]) or {}
        r["value"], r["pro_mean"] = item.get("value"), item.get("pro_mean")
    return records
