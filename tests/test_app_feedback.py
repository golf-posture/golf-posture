"""Tests for pipeline.app_feedback (feedback_app.json) and for keeping feedback_ko / feedback_user separate."""
import copy
import json
from pathlib import Path

import pytest

from pipeline import generate_feedback_report as gfr
import pipeline.app_feedback as app_feedback
from pipeline.app_feedback import (
    EVENT_WEIGHTS,
    MAX_ISSUES,
    MAX_PRAISE,
    build_app_payload,
    item_score,
    percentile_to_score,
    score_from_severity,
    status_from_score,
)
from pipeline.user_feedback import EVENT_NAMES, build_user_feedback, render_user_feedback_text

FEATURES = ["trail_elbow_angle", "lead_elbow_angle", "shoulder_tilt", "hip_tilt", "spine_tilt"]
MOTION = ["shoulder_turn_top", "xfactor_turn_top", "hip_turn_top", "weight_shift",
          "head_movement", "hip_height_change", "stance_change"]


def _fb(grade="Excellent", z=0.2, reliability="high"):
    return {"grade": grade, "z_score": z, "reliability": reliability, "value": 100.0, "pro_mean": 95.0, "pro_std": 5.0}


def _data():
    events = {}
    for idx, name in enumerate(EVENT_NAMES):
        events[name] = {
            "frame_index": 10 * (idx + 1),
            "features": {f: 1.0 for f in FEATURES},
            "feedback": {f: _fb() for f in FEATURES},
            "event_frame_path": f"frames/{idx}.jpg",
            "skeleton_comparison_path": f"skeletons/{idx}.jpg",
        }
    motion = {
        name: {"value": 1.0, "value_text": "1.0", "pro_median_text": "1.0", "pro_range_text": "0.5~1.5",
               "penalty_z": 0.1, "robust_z": 0.1, "direction": "high", "grade": "Excellent",
               "reliability": "high", "tail_pct": 40.0}
        for name in MOTION
    }
    return {
        "events": events, "motion_metrics": motion, "dynamics": {},
        "sequence_features": {"tempo_ratio": {"value": 2.8, "pro_mean": 3.9, "grade": "Excellent"}},
        "handedness": "right", "mock_events": False,
    }


def _elbow_problem(data):
    data["events"]["Impact"]["feedback"]["lead_elbow_angle"] = _fb("Needs Improvement", 2.6)
    data["events"]["Impact"]["feedback"]["trail_elbow_angle"] = _fb("Needs Improvement", -2.3)
    data["events"]["Mid-downswing"]["feedback"]["lead_elbow_angle"] = _fb("Needs Improvement", 2.1)
    return data


def _motion_problem(data, name, penalty, grade="Needs Improvement", direction="high"):
    data["motion_metrics"][name].update({"penalty_z": penalty, "grade": grade, "direction": direction,
                                         "value_text": "0.34배", "pro_median_text": "0.23배"})
    return data


def _dynamics(data, pair, feature, z, grade="Needs Improvement", reliability="high", delta=12.0, pro=3.0):
    data["dynamics"].setdefault(pair, {"features": {}})["features"][feature] = {
        "delta": delta, "pro_median_delta": pro, "robust_z": z, "grade": grade, "reliability": reliability}
    return data


# ---- scoring -------------------------------------------------------------------------------------
def test_item_score_is_piecewise_linear_through_the_anchor_points():
    assert [item_score(z) for z in (0, 1, 2, 3, 4)] == [100, 85, 50, 15, 0]
    assert item_score(0.5) == pytest.approx(92.5)
    assert item_score(1.5) == pytest.approx(67.5)
    assert item_score(-2.5) == pytest.approx(32.5)            # sign does not matter
    assert item_score(9.0) == 0
    assert score_from_severity(1.0) == 85                     # inside the pro range stays high


def test_status_rules():
    assert status_from_score(49)["code"] == "priority"
    assert status_from_score(50)["code"] == "adjust"
    assert status_from_score(67)["code"] == "adjust"
    assert status_from_score(68)["code"] == "recheck"


def test_combined_percentile_maps_onto_the_same_scale():
    assert percentile_to_score(0) == pytest.approx(100)
    assert percentile_to_score(68.27) == pytest.approx(85, abs=0.5)     # like |z| = 1
    assert percentile_to_score(95.45) == pytest.approx(50, abs=0.5)     # like |z| = 2
    assert percentile_to_score(99.99) < 15


# ---- issue cards ---------------------------------------------------------------------------------
def test_angle_card_has_title_subtitle_detail_and_image_of_worst_event():
    payload = build_app_payload(_elbow_problem(_data()))
    assert payload["state"] == "issues"
    card = payload["issues"][0]
    assert card["title"] == "임팩트 팔꿈치 각도 차이"                 # "<problem event> <summary>", no direction
    assert card["subtitle"] == "다운스윙~임팩트에서 양쪽 팔꿈치 각도가 프로와 달라요."
    assert card["event"] == {"key": "Impact", "label": "임팩트"}
    assert card["score"] == score_from_severity(2.6) == 29 and card["status"]["code"] == "priority"
    assert card["worst_item"] == {"kind": "angle", "event": {"key": "Impact", "label": "임팩트"},
                                  "label": "타깃 쪽 팔꿈치 각도", "score": 29}
    assert card["media"]["keyframe"] == "frames/5.jpg"
    kinds = [b["kind"] for b in card["body"]]
    assert kinds == ["score", "evidence", "direction", "also", "advice", "advice", "drill", "caution"]
    body = {b["kind"]: b["text"] for b in card["body"]}
    assert body["score"] == "임팩트에서 타깃 쪽 팔꿈치 각도 점수가 29점으로 가장 낮아요."
    assert body["evidence"] == "내 값은 100°, 프로 평균은 95°예요(프로는 보통 ±5° 이내)."
    assert "임팩트 뒤쪽 팔꿈치" in body["also"] and "다운스윙 타깃 쪽 팔꿈치" in body["also"]
    assert body["direction"] == "타깃 쪽 팔꿈치가 프로보다 더 펴져 있어요."
    assert body["drill"] == "연습: 빈 스윙을 천천히 하며 임팩트에서 멈추고, 팔꿈치에 약간 여유가 있는 모양인지 거울로 확인해 보세요."
    advice = [b["text"] for b in card["body"] if b["kind"] == "advice"]
    assert advice[1] == "팔에 힘이 들어가 너무 곧게 펴져 있지 않은지 확인해 보세요."
    assert card["sources"] == ["angle"]


def test_cards_are_sorted_by_score_lowest_first_and_ranks_are_sequential():
    data = _elbow_problem(_data())
    _motion_problem(data, "weight_shift", 3.0)
    _motion_problem(data, "head_movement", 1.5)
    issues = build_app_payload(data)["issues"]
    assert [c["rank"] for c in issues] == [1, 2, 3]
    scores = [c["score"] for c in issues]
    assert scores == sorted(scores)
    assert issues[0]["id"] == "weight_shift"


def test_at_most_five_issue_cards():
    data = _elbow_problem(_data())
    for name, z in zip(["weight_shift", "head_movement", "hip_height_change", "stance_change", "hip_turn_top"],
                       [3.0, 2.8, 2.6, 2.4, 2.2]):
        _motion_problem(data, name, z)
    data["events"]["Address"]["feedback"]["spine_tilt"] = _fb("Needs Improvement", 2.5)
    payload = build_app_payload(data)
    assert len(payload["issues"]) == MAX_ISSUES == 5


def test_fewer_candidates_are_not_padded_and_none_means_no_issues():
    one = build_app_payload(_motion_problem(_data(), "weight_shift", 1.4))
    assert len(one["issues"]) == 1 and one["state"] == "issues"
    none = build_app_payload(_data())
    assert none["issues"] == [] and none["state"] == "no_issues"
    assert none["empty_state"]["title"] == "발견된 교정 항목이 없어요"


def test_unreliable_candidates_never_become_cards():
    data = _data()
    data["events"]["Impact"]["feedback"]["lead_elbow_angle"] = _fb("Needs Improvement", 3.0, reliability="low")
    data["motion_metrics"]["weight_shift"].update({"penalty_z": 3.0, "reliability": "low", "direction": "high"})
    assert build_app_payload(data)["issues"] == []


def test_swing_dynamics_merge_into_the_body_part_card():
    data = _elbow_problem(_data())
    _dynamics(data, "Address->Top", "lead_elbow_angle", 1.5)          # weaker than the angle finding
    card = build_app_payload(data)["issues"][0]
    assert card["id"] == "elbow" and card["sources"] == ["angle", "dynamics"]
    assert any(b["kind"] == "change" for b in card["body"])
    assert any("변화량" in s["label"] for s in card["stats"])


def test_dynamics_only_finding_becomes_a_card_with_both_event_images():
    data = _dynamics(_data(), "Address->Top", "spine_tilt", 2.7)
    card = build_app_payload(data)["issues"][0]
    assert card["id"] == "spine" and card["sources"] == ["dynamics"]
    assert card["event"]["key"] == "Top"
    assert card["subtitle"] == "어드레스에서 탑으로 가는 동안 상체 기울기 변화가 프로와 달라요."
    assert card["media"]["keyframes"] == ["frames/0.jpg", "frames/3.jpg"]
    score = score_from_severity(2.7)
    assert card["body"][0]["text"] == f"어드레스에서 탑으로 가는 동안 상체 기울기 변화 점수가 {score}점이에요."
    assert card["body"][1]["text"] == "프로는 이 구간에서 보통 3° 정도 변하는데, 내 스윙은 12° 변했어요."
    assert card["worst_item"]["kind"] == "dynamics" and card["worst_item"]["score"] == score


def test_low_reliability_or_good_grade_dynamics_are_ignored():
    data = _dynamics(_data(), "Address->Top", "spine_tilt", 2.7, reliability="low")
    _dynamics(data, "Top->Impact", "spine_tilt", 1.5, grade="Good")
    assert build_app_payload(data)["issues"] == []


# ---- praise tab ----------------------------------------------------------------------------------
def test_praise_cards_exist_and_never_contradict_an_issue():
    data = _motion_problem(_data(), "weight_shift", 2.0)
    data["events"]["Address"]["feedback"]["spine_tilt"] = _fb("Needs Improvement", 2.5)
    payload = build_app_payload(data)
    ids = {c["id"] for c in payload["praise"]}
    assert "weight_shift" not in ids and "stable_spine" not in ids
    assert "shoulder_turn_top" in ids and "tempo" in ids
    assert 0 < len(payload["praise"]) <= MAX_PRAISE
    scores = [c["score"] for c in payload["praise"]]
    assert scores == sorted(scores, reverse=True)                     # strongest praise first
    card = next(c for c in payload["praise"] if c["id"] == "shoulder_turn_top")
    assert card["title"] == "탑 어깨 회전" and card["score"] >= 70 and card["media"]["keyframe"] == "frames/3.jpg"


# ---- summary / notice ----------------------------------------------------------------------------
def test_tempo_and_posture_score():
    summary = build_app_payload(_data())["summary"]
    assert summary["tempo"]["text"] == "2.8 : 1" and summary["tempo"]["caption"] == "백스윙 : 다운스윙"
    assert 95 <= summary["posture_score"] <= 100 and summary["posture_score_basis"] == "event_weighted_items"
    assert summary["event_weights"] == EVENT_WEIGHTS


def test_tempo_out_of_range_is_excluded_with_notice():
    data = _data()
    data["sequence_features"]["tempo_ratio"]["value"] = 39.25
    payload = build_app_payload(data)
    assert payload["summary"]["tempo"] == {"status": "excluded", "ratio": None, "text": "-",
                                           "caption": "백스윙 : 다운스윙", "pro_mean": 3.9}
    assert payload["reliability"] == "low" and payload["notice"]["code"] == "tempo_excluded"


def test_mock_events_raise_a_notice():
    data = _data()
    data["mock_events"] = True
    assert build_app_payload(data)["notice"]["code"] == "mock_events"


def test_events_for_the_full_swing_tab_link_to_issue_cards():
    payload = build_app_payload(_elbow_problem(_data()))
    rows = {e["key"]: e for e in payload["events"]}
    assert list(rows) == EVENT_NAMES
    assert rows["Impact"]["frame_index"] == 60 and rows["Impact"]["issue_ids"] == ["elbow"]
    assert rows["Address"]["issue_ids"] == []


# ---- event scores, weights, reliability, combined ------------------------------------------------
def _all_bad_at(data, event, z=3.0):
    for f in FEATURES:
        data["events"][event]["feedback"][f] = _fb("Needs Improvement", z)
    return data


def test_event_weights_make_an_impact_problem_cost_more_than_a_finish_problem():
    impact = build_app_payload(_all_bad_at(_data(), "Impact"))["summary"]["posture_score"]
    finish = build_app_payload(_all_bad_at(_data(), "Finish"))["summary"]["posture_score"]
    assert impact < finish < build_app_payload(_data())["summary"]["posture_score"]


def test_low_reliability_items_pull_an_event_score_less():
    high, low = _data(), _data()
    high["events"]["Address"]["feedback"]["spine_tilt"] = _fb("Needs Improvement", 3.0, "high")
    low["events"]["Address"]["feedback"]["spine_tilt"] = _fb("Needs Improvement", 3.0, "low")
    score = lambda d: {e["key"]: e for e in build_app_payload(d)["events"]}["Address"]["score"]["items"]
    assert score(low) > score(high)


def test_missing_items_are_left_out_not_scored_zero():
    data = _data()
    data["events"]["Finish"]["feedback"] = {f: {"grade": "unavailable", "z_score": None, "reliability": "low"}
                                            for f in FEATURES}
    payload = build_app_payload(data)
    finish = {e["key"]: e for e in payload["events"]}["Finish"]
    assert finish["score"]["items"] is None and finish["score"]["item_count"] == 0
    assert payload["summary"]["posture_score"] >= 95                 # not dragged down by the missing event


def test_motion_and_dynamics_items_belong_to_their_events():
    data = _dynamics(_motion_problem(_data(), "weight_shift", 3.0), "Top->Mid-downswing", "spine_tilt", 3.0)
    rows = {e["key"]: e for e in build_app_payload(data)["events"]}
    assert rows["Impact"]["score"]["item_count"] == 5 + 4              # 5 angles + 4 motion metrics at impact
    assert rows["Top"]["score"]["item_count"] == 5 + 3                 # 5 angles + 3 motion metrics at top
    assert rows["Mid-downswing"]["score"]["item_count"] == 5 + 1       # 5 angles + the dynamics item ending here


def test_combined_score_runs_in_parallel_with_the_item_based_score():
    data = _data()
    for name in EVENT_NAMES:
        data["events"][name]["combined_score"] = {"deviation_percentile": 20.0}
    data["events"]["Impact"]["combined_score"] = {"deviation_percentile": 99.0}
    payload = build_app_payload(data)
    impact = {e["key"]: e for e in payload["events"]}["Impact"]["score"]
    assert impact["combined"] < 40 < impact["items"]                   # the two views can disagree
    alt = payload["summary"]["posture_score_alt"]
    assert alt["basis"] == "combined_mahalanobis" and alt["value"] < payload["summary"]["posture_score"] + 1


def test_motion_card_sentences_give_score_evidence_and_advice():
    card = build_app_payload(_motion_problem(_data(), "weight_shift", 3.0))["issues"][0]
    kinds = [b["kind"] for b in card["body"]]
    assert kinds[:2] == ["score", "evidence"]
    assert card["body"][0]["text"] == "체중 이동 점수가 15점이에요."
    assert card["body"][1]["text"] == "내 값은 0.34배, 프로는 대부분 0.5~1.5예요(중앙값 0.23배)."
    assert card["worst_item"] == {"kind": "motion", "event": {"key": "Impact", "label": "임팩트"},
                                  "label": "체중 이동", "score": 15}


# ---- advice (richer, data-driven) -----------------------------------------------------------------
def _texts(card, kind):
    return [b["text"] for b in card["body"] if b["kind"] == kind]


def test_bent_elbow_gets_the_opposite_direction_hint():
    data = _data()
    data["events"]["Impact"]["feedback"]["lead_elbow_angle"] = _fb("Needs Improvement", -2.6)
    card = build_app_payload(data)["issues"][0]
    assert _texts(card, "direction") == ["타깃 쪽 팔꿈치가 프로보다 더 굽혀져 있어요."]
    assert _texts(card, "advice")[1] == "팔이 너무 접혀 있지 않은지 확인해 보세요."
    assert "팔이 자연스럽게 뻗어 있는지" in _texts(card, "drill")[0]


def test_spine_advice_follows_the_phase_and_never_names_a_direction():
    data = _data()
    data["events"]["Address"]["feedback"]["spine_tilt"] = _fb("Needs Improvement", 2.5)
    card = build_app_payload(data)["issues"][0]
    assert _texts(card, "direction") == []
    assert _texts(card, "pattern") == ["어드레스 한 곳에서만 나타나요. 그 순간의 타이밍을 중심으로 확인해 보세요."]
    assert _texts(card, "advice")[0].startswith("어드레스 때의 상체 기울기가 이후 스윙의 기준이 돼요.")
    assert _texts(card, "drill")[0].startswith("연습: 어드레스 자세를 정면에서 촬영해 두고")
    late = _data()
    late["events"]["Impact"]["feedback"]["spine_tilt"] = _fb("Needs Improvement", 2.5)
    assert _texts(build_app_payload(late)["issues"][0], "advice")[0].startswith("스윙 중 상체 기울기가 어드레스 때와")


def test_a_problem_that_repeats_is_described_as_a_swing_pattern():
    data = _data()
    for event in ("Address", "Top", "Impact"):
        data["events"][event]["feedback"]["spine_tilt"] = _fb("Needs Improvement", 2.5)
    card = build_app_payload(data)["issues"][0]
    assert _texts(card, "pattern") == ["어드레스, 탑, 임팩트에서 반복돼요. 한 순간의 흔들림보다 스윙 흐름의 차이일 수 있어요."]


def test_dynamics_advice_compares_sizes_and_never_prints_minus_zero():
    small = _dynamics(_data(), "Mid-follow-through->Finish", "spine_tilt", 2.4, delta=-0.3, pro=-14.0)
    card = build_app_payload(small)["issues"][0]
    assert _texts(card, "evidence") == ["프로는 이 구간에서 보통 14° 정도 변하는데, 내 스윙은 거의 변하지 않았어요."]
    assert "변화가 작아요" in _texts(card, "advice")[0]
    assert _texts(card, "drill")[0] == "연습: 프로 영상의 팔로스루~피니시 구간을 반복해서 보고, 같은 속도로 천천히 따라 해 보세요."
    assert card["stats"][0]["value_text"] == "0°"                      # -0.3 rounds to 0, not "-0°"
    assert "-0°" not in json.dumps(build_app_payload(small), ensure_ascii=False)

    big = _dynamics(_data(), "Top->Mid-downswing", "lead_elbow_angle", 2.4, delta=30.0, pro=8.0)
    card = build_app_payload(big)["issues"][0]
    assert _texts(card, "evidence") == ["프로는 이 구간에서 보통 8° 정도 변하는데, 내 스윙은 30° 변했어요."]
    assert "프로보다 많이 변해요" in _texts(card, "advice")[0]


def test_motion_cards_get_a_practice_tip_only_when_they_need_attention():
    severe = build_app_payload(_motion_problem(_data(), "weight_shift", 3.0))["issues"][0]
    assert _texts(severe, "advice") == ["몸이 옆으로 밀리는 것(스웨이)은 아닌지 확인해 보세요."]
    assert _texts(severe, "drill")[0].startswith("연습: 오른쪽 골반 바깥에 벽이나 의자를")
    mild = build_app_payload(_motion_problem(_data(), "weight_shift", 1.2))["issues"][0]
    assert _texts(mild, "drill") == []                                  # "check once more" cards stay short


def test_practice_tip_uses_the_trail_side_of_the_golfer():
    data = _motion_problem(_data(), "weight_shift", 3.0)
    data["handedness"] = "left"
    assert _texts(build_app_payload(data)["issues"][0], "drill")[0].startswith("연습: 왼쪽 골반 바깥에")


def test_empty_tip_falls_back_to_a_generic_hint():
    card = build_app_payload(_motion_problem(_data(), "shoulder_turn_top", 2.6, direction="high"))["issues"][0]
    assert _texts(card, "advice") == ["탑에서 클럽이 지면과 평행에 가까운지 영상으로 확인해 보세요. 너무 크게 돌아가지 않았는지 점검하면 좋아요."]


def test_direction_hints_and_drills_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(app_feedback, "USE_DIRECTION_HINTS", False)
    monkeypatch.setattr(app_feedback, "USE_DRILLS", False)
    card = build_app_payload(_elbow_problem(_data()))["issues"][0]
    kinds = [b["kind"] for b in card["body"]]
    assert "direction" not in kinds and "drill" not in kinds
    assert kinds.count("advice") == 1
    motion = build_app_payload(_motion_problem(_data(), "weight_shift", 3.0))["issues"][0]
    assert _texts(motion, "drill") == []


def _praise(payload, key):
    return next(c for c in payload["praise"] if c["id"] == key)


def test_praise_body_explains_what_it_means_and_how_to_keep_it():
    payload = build_app_payload(_data())
    card = _praise(payload, "weight_shift")
    kinds = [b["kind"] for b in card["body"]]
    assert kinds == ["score", "evidence", "praise", "meaning", "keep"]
    body = {b["kind"]: b["text"] for b in card["body"]}
    assert body["score"].startswith("체중 이동 점수가") and "프로" in body["score"]
    assert body["meaning"] == "체중이 알맞게 이동하면 임팩트에서 몸의 힘을 공으로 전달하기 쉬워요."
    assert body["keep"] == "다운스윙에서 체중이 타깃 쪽으로 자연스럽게 옮겨가는 느낌을 기억해 두세요."   # not the generic line


def test_every_praise_kind_has_its_own_keep_tip():
    keys = set(app_feedback.PRAISE_MEANING)
    assert keys == set(app_feedback.PRAISE_KEEP)
    assert set(app_feedback.MOTION_UI) | {"tempo"} <= keys


def test_score_level_wording_follows_the_score():
    assert app_feedback._score_level(98).startswith("프로 중앙값과 거의")
    assert app_feedback._score_level(92).startswith("프로 분포의 한가운데")
    assert app_feedback._score_level(86).startswith("프로 범위 안에서")


def test_stable_part_praise_names_the_best_and_weakest_event():
    data = _data()
    data["events"]["Impact"]["feedback"]["spine_tilt"] = _fb("Excellent", 0.9)      # still inside the pro range
    data["events"]["Top"]["feedback"]["spine_tilt"] = _fb("Excellent", 0.0)
    card = _praise(build_app_payload(data), "stable_spine")
    detail = next(b["text"] for b in card["body"] if b["kind"] == "detail")
    assert detail == f"가장 안정적인 구간은 탑(100점)이고, 가장 낮은 구간도 임팩트에서 {score_from_severity(0.9)}점이에요."
    even = _praise(build_app_payload(_data()), "stable_spine")
    assert next(b["text"] for b in even["body"] if b["kind"] == "detail").startswith("평가된 구간이")


def test_praise_cautions_only_where_they_apply():
    data = _data()
    data["motion_metrics"]["shoulder_turn_top"].update({"robust_z": 1.4})            # "up": beyond the pro median
    payload = build_app_payload(data)
    assert any(b["kind"] == "caution" for b in _praise(payload, "shoulder_turn_top")["body"])
    assert not any(b["kind"] == "caution" for b in _praise(payload, "weight_shift")["body"])
    assert any(b["kind"] == "caution" for b in _praise(payload, "stable_hip_tilt")["body"])     # camera dependent
    assert not any(b["kind"] == "caution" for b in _praise(payload, "stable_spine")["body"])


def test_praise_tab_header_points_to_the_first_issue():
    payload = build_app_payload(_elbow_problem(_data()))
    header = payload["praise_header"]
    assert header["title"] == "이번 스윙에서 잘한 점"
    assert header["message"].startswith(f"잘한 점 {len(payload['praise'])}가지를 찾았어요.")
    assert "'임팩트 팔꿈치 각도 차이'를 먼저 확인해 보세요." in header["message"]
    clean = build_app_payload(_data())["praise_header"]["message"]
    assert clean.endswith("이번 스윙에서는 교정할 항목이 발견되지 않았어요.")


def test_no_praise_means_no_header():
    data = _data()
    for event in data["events"].values():
        event["feedback"] = {f: _fb("Needs Improvement", 3.0) for f in FEATURES}
    for name in MOTION:
        _motion_problem(data, name, 3.0)
    data["sequence_features"]["tempo_ratio"]["value"] = 39.0
    payload = build_app_payload(data)
    assert payload["praise"] == [] and payload["praise_header"] is None


def test_object_particle_follows_the_last_syllable():
    assert app_feedback._particle_eul("차이") == "를"
    assert app_feedback._particle_eul("체중 이동") == "을"


def test_praise_cards_are_part_of_the_payload():
    payload = build_app_payload(_data())
    assert payload["praise"], "a clean swing must produce praise cards"
    assert all({"id", "title", "subtitle", "score", "event", "body", "media"} <= set(c) for c in payload["praise"])


def test_praise_cards_carry_a_full_body_like_the_issue_cards():
    cards = {c["id"]: c for c in build_app_payload(_data())["praise"]}
    motion = cards["shoulder_turn_top"]
    assert [b["kind"] for b in motion["body"]] == ["score", "evidence", "praise", "meaning", "keep"]
    assert motion["body"][0]["text"].startswith(f"어깨 회전 점수가 {motion['score']}점으로,")
    assert motion["body"][1]["text"].startswith("내 값은 1.0")
    stable = cards["stable_spine"]
    assert [b["kind"] for b in stable["body"]] == ["score", "evidence", "praise", "meaning", "detail", "keep"]
    assert "모든 이벤트에서" in stable["body"][0]["text"]
    tempo = cards["tempo"]
    assert tempo["body"][0]["text"] == "내 리듬은 2.8 : 1, 프로 평균은 3.9 : 1이에요."
    assert [b["kind"] for b in tempo["body"]] == ["evidence", "praise", "meaning", "detail", "keep"]


def test_elbow_angle_can_be_praised_when_it_is_not_an_issue():
    ids = {c["id"] for c in build_app_payload(_data())["praise"]}
    assert "stable_elbow" in ids
    ids = {c["id"] for c in build_app_payload(_elbow_problem(_data()))["praise"]}
    assert "stable_elbow" not in ids                                  # an elbow issue and elbow praise never coexist


# ---- contract & separation ----------------------------------------------------------------------
def test_payload_is_strict_json_and_does_not_mutate_the_input():
    data = _dynamics(_elbow_problem(_data()), "Address->Top", "lead_elbow_angle", 1.5)
    before = copy.deepcopy(data)
    payload = build_app_payload(data)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert data == before
    assert payload["schema_version"] == 3


def _analysis_file(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_report_writes_app_json_and_keeps_the_two_existing_files_identical(tmp_path, monkeypatch):
    data = _dynamics(_motion_problem(_elbow_problem(_data()), "weight_shift", 1.6), "Address->Top", "spine_tilt", 2.2)
    analysis = _analysis_file(tmp_path, data)

    with_app = gfr.generate_feedback_report(analysis, tmp_path / "with_app")
    assert with_app["feedback_app_json"].exists()

    def boom(_data):
        raise RuntimeError("app payload failed")
    monkeypatch.setattr(gfr, "build_app_payload", boom)
    without_app = gfr.generate_feedback_report(analysis, tmp_path / "without_app")
    assert not without_app["feedback_app_json"].exists()           # failure is contained ...
    for key in ("feedback_ko", "feedback_user", "feedback_user_json", "table_csv"):
        a = with_app[key].read_text(encoding="utf-8").replace(str(tmp_path / "with_app"), "<OUT>")
        b = without_app[key].read_text(encoding="utf-8").replace(str(tmp_path / "without_app"), "<OUT>")
        assert a == b                                                  # only the output folder differs


def test_feedback_user_text_is_still_rendered_by_user_feedback_alone(tmp_path):
    data = _elbow_problem(_data())
    paths = gfr.generate_feedback_report(_analysis_file(tmp_path, data), tmp_path / "out")
    expected = render_user_feedback_text(build_user_feedback(data))
    assert paths["feedback_user"].read_text(encoding="utf-8") == expected
