"""Tests for pipeline.app_feedback (feedback_app.json) and for keeping feedback_ko / feedback_user separate."""
import copy
import json
from pathlib import Path

import pytest

from pipeline import generate_feedback_report as gfr
from pipeline.app_feedback import (
    MAX_ISSUES,
    MAX_PRAISE,
    build_app_payload,
    score_from_severity,
    status_from_score,
)
from pipeline.user_feedback import EVENT_NAMES, build_user_feedback, render_user_feedback_text

FEATURES = ["trail_elbow_angle", "lead_elbow_angle", "shoulder_tilt", "hip_tilt", "spine_tilt"]
MOTION = ["shoulder_turn_top", "xfactor_turn_top", "hip_turn_top", "weight_shift",
          "head_movement", "hip_height_change", "stance_change"]


def _fb(grade="Excellent", z=0.2, reliability="high"):
    return {"grade": grade, "z_score": z, "reliability": reliability, "value": 100.0, "pro_mean": 95.0}


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


def _dynamics(data, pair, feature, z, grade="Needs Improvement", reliability="high"):
    data["dynamics"].setdefault(pair, {"features": {}})["features"][feature] = {
        "delta": 12.0, "pro_median_delta": 3.0, "robust_z": z, "grade": grade, "reliability": reliability}
    return data


# ---- scoring -------------------------------------------------------------------------------------
def test_score_and_status_rules():
    assert [score_from_severity(z) for z in (0, 1, 2, 9)] == [100, 70, 40, 0]
    assert status_from_score(39)["code"] == "priority"
    assert status_from_score(40)["code"] == "adjust"
    assert status_from_score(55)["code"] == "recheck"


# ---- issue cards ---------------------------------------------------------------------------------
def test_angle_card_has_title_subtitle_detail_and_image_of_worst_event():
    payload = build_app_payload(_elbow_problem(_data()))
    assert payload["state"] == "issues"
    card = payload["issues"][0]
    assert card["title"] == "임팩트 팔꿈치 각도 차이"                 # "<problem event> <summary>", no direction
    assert card["subtitle"] == "다운스윙~임팩트에서 양쪽 팔꿈치 각도가 프로와 달라요."
    assert card["event"] == {"key": "Impact", "label": "임팩트"}
    assert card["score"] == 22 and card["status"]["code"] == "priority"
    assert card["media"]["keyframe"] == "frames/5.jpg"
    kinds = [b["kind"] for b in card["body"]]
    assert "advice" in kinds and "caution" in kinds                   # impact is a fast event
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
    assert summary["posture_score"] == 10.0 and summary["posture_score_basis"] == "within_pro_ratio"


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


# ---- contract & separation ----------------------------------------------------------------------
def test_payload_is_strict_json_and_does_not_mutate_the_input():
    data = _dynamics(_elbow_problem(_data()), "Address->Top", "lead_elbow_angle", 1.5)
    before = copy.deepcopy(data)
    payload = build_app_payload(data)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert data == before
    assert payload["schema_version"] == 2


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
