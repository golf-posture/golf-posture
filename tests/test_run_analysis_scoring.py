from __future__ import annotations

import numpy as np
import pytest

from pipeline.run_analysis import (
    REQUIRED_FEATURES,
    circular_delta,
    dynamics_feedback_for_pair,
    grade_from_combined_percentile,
    grade_from_z,
    mahalanobis_combined_score,
)


class TestGradeFromZ:
    @pytest.mark.parametrize(
        "z_score, expected_grade",
        [
            (None, "unavailable"),
            (0.0, "Excellent"),
            (1.0, "Excellent"),
            (-1.0, "Excellent"),
            (1.5, "Good"),
            (2.0, "Good"),
            (2.1, "Needs Improvement"),
            (-3.0, "Needs Improvement"),
        ],
    )
    def test_grades_by_absolute_z_score_thresholds(self, z_score, expected_grade):
        # Act
        grade = grade_from_z(z_score)

        # Assert
        assert grade == expected_grade


class TestGradeFromCombinedPercentile:
    @pytest.mark.parametrize(
        "percentile, expected_grade",
        [
            (None, "unavailable"),
            (50.0, "Excellent"),
            (68.27, "Excellent"),
            (90.0, "Good"),
            (95.45, "Good"),
            (99.0, "Needs Improvement"),
        ],
    )
    def test_grades_by_deviation_percentile_thresholds(self, percentile, expected_grade):
        # Act
        grade = grade_from_combined_percentile(percentile)

        # Assert
        assert grade == expected_grade


class TestCircularDelta:
    def test_wraps_around_180_degree_boundary(self):
        # Act
        delta = circular_delta(179.0, -179.0)

        # Assert
        assert delta == pytest.approx(-2.0)

    def test_returns_plain_difference_within_range(self):
        # Act
        delta = circular_delta(10.0, 5.0)

        # Assert
        assert delta == pytest.approx(5.0)


def _identity_covariance_stats(event_idx: int, mean_vector: list[float]) -> dict:
    dimension = len(mean_vector)
    return {
        "events": {
            str(event_idx): {
                "feature_order": REQUIRED_FEATURES,
                "mean": mean_vector,
                "precision": np.eye(dimension).tolist(),
                "shrinkage_alpha": 0.0,
            }
        }
    }


class TestMahalanobisCombinedScore:
    def test_returns_zero_distance_when_features_match_the_mean_exactly(self):
        # Arrange
        mean_vector = [0.0, 0.0, 90.0, 0.0, 0.0]
        features = dict(zip(REQUIRED_FEATURES, mean_vector))
        covariance_stats = _identity_covariance_stats(0, mean_vector)

        # Act
        result = mahalanobis_combined_score(features, 0, covariance_stats)

        # Assert
        assert result["distance"] == pytest.approx(0.0)
        assert result["grade"] == "Excellent"

    def test_reports_unavailable_when_a_required_feature_is_missing(self):
        # Arrange
        mean_vector = [0.0, 0.0, 90.0, 0.0, 0.0]
        features = dict(zip(REQUIRED_FEATURES, mean_vector))
        features[REQUIRED_FEATURES[0]] = None
        covariance_stats = _identity_covariance_stats(0, mean_vector)

        # Act
        result = mahalanobis_combined_score(features, 0, covariance_stats)

        # Assert
        assert result["distance"] is None
        assert result["grade"] == "unavailable"

    def test_reports_unavailable_when_covariance_stats_are_missing_for_the_event(self):
        # Arrange
        features = dict(zip(REQUIRED_FEATURES, [0.0] * len(REQUIRED_FEATURES)))

        # Act
        result = mahalanobis_combined_score(features, 0, None)

        # Assert
        assert result["distance"] is None
        assert result["grade"] == "unavailable"


class TestDynamicsFeedbackForPair:
    def test_grades_delta_against_pro_median_and_scaled_mad(self):
        # Arrange
        start_features = {"trail_elbow_angle": 10.0}
        end_features = {"trail_elbow_angle": 15.0}
        dynamics_stats = {
            "pairs": {
                "Address->Toe-up": {
                    "features": {
                        "trail_elbow_angle": {
                            "median": 5.0,
                            "scaled_mad": 1.0,
                            "reliability": "high",
                            "feedback_strength": "strong",
                        }
                    }
                }
            }
        }

        # Act
        feedback = dynamics_feedback_for_pair(start_features, end_features, 0, 1, dynamics_stats)

        # Assert
        item = feedback["trail_elbow_angle"]
        assert item["delta"] == pytest.approx(5.0)
        assert item["robust_z"] == pytest.approx(0.0)
        assert item["grade"] == "Excellent"

    def test_marks_unavailable_when_one_side_of_the_pair_has_no_value(self):
        # Arrange
        start_features = {"trail_elbow_angle": None}
        end_features = {"trail_elbow_angle": 15.0}

        # Act
        feedback = dynamics_feedback_for_pair(start_features, end_features, 0, 1, None)

        # Assert
        item = feedback["trail_elbow_angle"]
        assert item["delta"] is None
        assert item["grade"] == "unavailable"
