from __future__ import annotations

import math

from pipeline.qc_tracked_features import (
    arm_flags,
    bone_length_flag,
    outlier_exempt,
    robust_outlier_flags,
)

FINISH_EVENT = 7


class TestOutlierExempt:
    def test_finish_shoulder_and_hip_tilt_are_exempt_because_bimodal(self):
        assert outlier_exempt("shoulder_tilt", FINISH_EVENT) is True
        assert outlier_exempt("hip_tilt", FINISH_EVENT) is True

    def test_other_events_and_features_are_not_exempt(self):
        assert outlier_exempt("shoulder_tilt", 3) is False
        assert outlier_exempt("spine_tilt", FINISH_EVENT) is False
        assert outlier_exempt("trail_elbow_angle_2d", FINISH_EVENT) is False


class TestBoneLengthFlag:
    def test_plausible_ratio_not_flagged(self):
        assert bone_length_flag(0.25, 0.22) is False
        assert bone_length_flag(0.25, 0.14) is False  # ratio 0.56, both bones >= 0.1

    def test_flags_extreme_ratio_either_direction(self):
        assert bone_length_flag(0.25, 0.60) is True  # ratio 2.4
        assert bone_length_flag(0.30, 0.12) is True  # ratio 0.4

    def test_flags_collapsed_bone(self):
        assert bone_length_flag(0.05, 0.05) is True

    def test_missing_lengths_not_flagged(self):
        assert bone_length_flag(None, 0.2) is False
        assert bone_length_flag(float("nan"), 0.2) is False


class TestRobustOutlierFlags:
    def test_flags_value_far_from_median(self):
        values = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8, 160.0]

        flags = robust_outlier_flags(values, circular=False)

        assert flags == [False] * 7 + [True]

    def test_missing_values_are_never_flagged_and_do_not_shift_center(self):
        values = [100.0, None, 101.0, float("nan"), 99.0, 100.5, 99.5]

        flags = robust_outlier_flags(values, circular=False)

        assert not any(flags)

    def test_circular_values_wrap_around_180(self):
        values = [179.0, -179.0, 178.0, -178.0, 180.0, -180.0, 0.0]

        flags = robust_outlier_flags(values, circular=True)

        assert flags == [False] * 6 + [True]

    def test_constant_values_not_flagged(self):
        assert not any(robust_outlier_flags([50.0] * 10, circular=False))

    def test_too_few_values_not_flagged(self):
        assert robust_outlier_flags([1.0, 500.0], circular=False) == [False, False]


def _row(**overrides):
    base = {
        "trail_elbow_source": "direct",
        "trail_min_vis": 0.9,
        "trail_upper_3d": 0.25,
        "trail_fore_3d": 0.22,
        "trail_in_frame": True,
        "trail_elbow_angle_2d": 120.0,
        "trail_elbow_angle_3d": 125.0,
    }
    return {**base, **overrides}


class TestArmFlags:
    def test_clean_row_has_no_flags(self):
        assert not any(arm_flags(_row(), "trail").values())

    def test_low_visibility(self):
        assert arm_flags(_row(trail_min_vis=0.35), "trail")["low_vis"] is True
        assert arm_flags(_row(trail_min_vis=0.4), "trail")["low_vis"] is False

    def test_bone_rule(self):
        assert arm_flags(_row(trail_fore_3d=0.6), "trail")["bone"] is True

    def test_out_of_frame(self):
        assert arm_flags(_row(trail_in_frame=False), "trail")["frame"] is True

    def test_inconsistent_2d_3d(self):
        flags = arm_flags(_row(trail_elbow_angle_2d=170.0, trail_elbow_angle_3d=100.0), "trail")

        assert flags["inconsistent"] is True

    def test_missing_elbow_is_not_flagged(self):
        row = _row(trail_elbow_source="missing", trail_min_vis=0.1, trail_in_frame=False)

        assert not any(arm_flags(row, "trail").values())

    def test_nan_inputs_do_not_raise(self):
        row = _row(trail_min_vis=math.nan, trail_elbow_angle_2d=math.nan)

        assert arm_flags(row, "trail")["inconsistent"] is False
