"""Pseudo-label frame sampling and filtering rules."""

import csv

import pytest

from pipeline.pseudo_label import (
    FINISH_EVENT,
    FLIP_INDEX,
    STUDENT_JOINTS,
    arm_disagreement,
    yolo_pose_line,
    keep_label,
    load_event_frames,
    sample_frames,
)


class TestSampleFrames:
    def test_window_around_each_event(self):
        frames = sample_frames({3: 100}, radius=2, frame_count=500)
        assert frames == {98: 3, 99: 3, 100: 3, 101: 3, 102: 3}

    def test_clipped_to_video_length(self):
        frames = sample_frames({0: 1, 7: 199}, radius=2, frame_count=200)
        assert min(frames) == 0 and max(frames) == 199

    def test_overlapping_windows_keep_nearest_event(self):
        frames = sample_frames({3: 100, 4: 103}, radius=2, frame_count=500)
        assert frames[101] == 3 and frames[102] == 4

    def test_tie_goes_to_the_earlier_event(self):
        frames = sample_frames({3: 100, 4: 102}, radius=2, frame_count=500)
        assert frames[101] == 3


class TestLoadEventFrames:
    def _write(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_id", "event_index", "event_name", "frame_index"])
            writer.writerows(rows)

    def test_groups_by_video_and_skips_excluded(self, tmp_path):
        table = tmp_path / "events.csv"
        self._write(table, [(8, 0, "Address", 29), (8, 3, "Top", 90), (9, 3, "Top", 70)])
        result = load_event_frames(table, exclude={"9"})
        assert result == {"8": {0: 29, 3: 90}}


class TestKeepLabel:
    def test_non_finish_events_are_kept_unchecked(self):
        assert keep_label(event_index=3, disagreement=None)
        assert keep_label(event_index=4, disagreement=9.0)

    def test_finish_needs_agreement(self):
        assert keep_label(FINISH_EVENT, 0.15)
        assert not keep_label(FINISH_EVENT, 0.16)

    def test_finish_without_second_opinion_is_dropped(self):
        assert not keep_label(FINISH_EVENT, None)

    def test_negative_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            keep_label(FINISH_EVENT, 0.1, threshold=-1)


def _arm(side, shoulder, elbow, wrist):
    return {f"{side}_SHOULDER": [*shoulder, 1.0], f"{side}_ELBOW": [*elbow, 1.0], f"{side}_WRIST": [*wrist, 1.0]}


class TestArmDisagreement:
    def test_identical_arms_have_zero_disagreement(self):
        arm = _arm("LEFT", (0, 0), (10, 0), (20, 0))
        assert arm_disagreement(arm, arm, "LEFT") == 0

    def test_worst_joint_over_arm_length(self):
        teacher = _arm("LEFT", (0, 0), (10, 0), (20, 0))
        second = _arm("LEFT", (0, 0), (10, 0), (20, 5))
        assert arm_disagreement(teacher, second, "LEFT") == pytest.approx(5 / 20)

    def test_missing_arm_has_no_disagreement_value(self):
        arm = _arm("LEFT", (0, 0), (10, 0), (20, 0))
        assert arm_disagreement(arm, None, "LEFT") is None
        assert arm_disagreement(arm, arm, "RIGHT") is None

    def test_zero_length_arm_is_undefined(self):
        flat = _arm("LEFT", (1, 1), (1, 1), (1, 1))
        assert arm_disagreement(flat, flat, "LEFT") is None


class TestFlipIndex:
    def test_flip_is_an_involution_that_swaps_sides(self):
        assert [FLIP_INDEX[FLIP_INDEX[i]] for i in range(8)] == list(range(8))
        for i, j in enumerate(FLIP_INDEX):
            assert STUDENT_JOINTS[i].split("_", 1)[1] == STUDENT_JOINTS[j].split("_", 1)[1]
            assert STUDENT_JOINTS[i].split("_")[0] != STUDENT_JOINTS[j].split("_")[0]


class TestYoloPoseLine:
    JOINTS = {name: [100.0, 50.0, 1.0] for name in STUDENT_JOINTS}

    def test_layout_and_normalisation(self):
        fields = yolo_pose_line([0, 0, 200, 100], self.JOINTS, 200, 100).split()
        assert len(fields) == 5 + 3 * 8
        assert fields[0] == "0"
        assert [float(v) for v in fields[1:5]] == [0.5, 0.5, 1.0, 1.0]
        assert fields[5:8] == ["0.500000", "0.500000", "2"]

    def test_hidden_side_gets_zero_visibility(self):
        fields = yolo_pose_line([0, 0, 200, 100], self.JOINTS, 200, 100, hidden_sides=("LEFT",)).split()
        visibility = fields[7::3]
        assert visibility == ["0", "0", "0", "2", "2", "2", "0", "2"]

    def test_missing_joint_is_hidden(self):
        joints = {k: v for k, v in self.JOINTS.items() if k != "LEFT_WRIST"}
        assert yolo_pose_line([0, 0, 200, 100], joints, 200, 100).split()[7::3][2] == "0"

    def test_coordinates_are_clamped_into_the_image(self):
        joints = dict(self.JOINTS, LEFT_SHOULDER=[-5.0, 500.0, 1.0])
        fields = yolo_pose_line([0, 0, 200, 100], joints, 200, 100).split()
        assert fields[5:7] == ["0.000000", "1.000000"]
