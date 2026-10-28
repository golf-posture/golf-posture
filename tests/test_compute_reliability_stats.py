from __future__ import annotations

import numpy as np
import pytest

from pipeline.compute_reliability_stats import (
    center_on_population_mean,
    circular_delta,
    circular_mean,
    classify_reliability,
    event_feature_vector,
    sample_covariance,
    shrink_covariance,
    trim_outliers,
)


class TestClassifyReliability:
    def test_returns_low_when_dispersion_is_none(self):
        # Arrange
        dispersion = None
        detection_rate = 1.0

        # Act
        reliability, strength, reason = classify_reliability(dispersion, detection_rate)

        # Assert
        assert reliability == "low"
        assert strength == "reference"
        assert reason == "no valid feature values"

    def test_returns_high_when_detection_and_dispersion_both_meet_threshold(self):
        # Arrange
        dispersion = 10.0
        detection_rate = 0.85

        # Act
        reliability, strength, _ = classify_reliability(dispersion, detection_rate)

        # Assert
        assert reliability == "high"
        assert strength == "strong"

    def test_returns_medium_when_only_moderate_thresholds_are_met(self):
        # Arrange
        dispersion = 15.0
        detection_rate = 0.70

        # Act
        reliability, strength, _ = classify_reliability(dispersion, detection_rate)

        # Assert
        assert reliability == "medium"
        assert strength == "normal"

    def test_returns_low_when_dispersion_exceeds_medium_threshold(self):
        # Arrange
        dispersion = 25.0
        detection_rate = 0.90

        # Act
        reliability, strength, _ = classify_reliability(dispersion, detection_rate)

        # Assert
        assert reliability == "low"
        assert strength == "reference"


class TestCircularHelpers:
    def test_circular_delta_wraps_around_positive_boundary(self):
        # Arrange
        value = 179.0
        center = -179.0

        # Act
        delta = circular_delta(value, center)

        # Assert
        assert delta == pytest.approx(-2.0)

    def test_circular_mean_of_values_straddling_wrap_boundary(self):
        # Arrange
        values = [179.0, -179.0]

        # Act
        mean = circular_mean(values)

        # Assert
        assert abs(mean) == pytest.approx(180.0)


class TestCenterOnPopulationMean:
    def test_subtracts_mean_for_linear_features_and_leaves_circular_features_untouched(self):
        # Arrange
        vector = [10.0, 90.0]
        feature_names = ["trail_elbow_angle", "shoulder_tilt"]
        mean_vector = [5.0, 90.0]

        # Act
        centered = center_on_population_mean(vector, feature_names, mean_vector)

        # Assert
        assert centered == [5.0, 90.0]


class TestEventFeatureVector:
    def test_returns_none_when_a_required_feature_is_missing(self):
        # Arrange
        row = {"trail_elbow_angle": "10.0"}
        circular_centers = {"shoulder_tilt": 90.0, "hip_tilt": 0.0}

        # Act
        vector = event_feature_vector(row, circular_centers, ["trail_elbow_angle", "shoulder_tilt"])

        # Assert
        assert vector is None

    def test_converts_circular_feature_to_delta_from_its_center(self):
        # Arrange
        row = {"shoulder_tilt": "95.0"}
        circular_centers = {"shoulder_tilt": 90.0}

        # Act
        vector = event_feature_vector(row, circular_centers, ["shoulder_tilt"])

        # Assert
        assert vector == pytest.approx([5.0])


class TestSampleCovariance:
    def test_uncorrelated_normal_data_yields_near_diagonal_covariance(self):
        # Arrange
        rng = np.random.default_rng(42)
        centered = rng.normal(scale=1.0, size=(500, 2))
        centered -= centered.mean(axis=0)

        # Act
        cov = sample_covariance(centered)

        # Assert
        assert cov.shape == (2, 2)
        assert cov[0, 1] == pytest.approx(0.0, abs=0.2)
        assert cov[0, 0] == pytest.approx(1.0, rel=0.3)


class TestShrinkCovariance:
    def test_leaves_well_conditioned_covariance_unchanged(self):
        # Arrange
        cov = np.eye(3) * 4.0

        # Act
        shrunk, before, after, alpha = shrink_covariance(cov)

        # Assert
        assert alpha == 0.0
        assert before == after
        assert np.allclose(shrunk, cov)

    def test_shrinks_ill_conditioned_covariance_toward_target_condition_number(self):
        # Arrange
        cov = np.diag([100.0, 1.0, 0.01])  # condition number = 10000, well above the trigger

        # Act
        shrunk, before, after, alpha = shrink_covariance(cov)

        # Assert
        assert before > 50.0
        assert alpha > 0.0
        assert after < before
        assert after <= 30.0 + 1e-6 or alpha == pytest.approx(0.5)


class TestTrimOutliers:
    def test_removes_rows_far_outside_chi2_threshold(self):
        # Arrange
        rng = np.random.default_rng(0)
        inliers = rng.normal(scale=1.0, size=(200, 5))
        inliers -= inliers.mean(axis=0)
        outliers = np.full((3, 5), 50.0)
        matrix = np.vstack([inliers, outliers])

        # Act
        kept, n_removed = trim_outliers(matrix)

        # Assert
        assert n_removed >= 3
        assert kept.shape[0] == matrix.shape[0] - n_removed

    def test_keeps_all_rows_when_sample_is_below_the_minimum_complete_case_size(self):
        # Arrange
        matrix = np.random.default_rng(1).normal(size=(5, 5))

        # Act
        kept, n_removed = trim_outliers(matrix)

        # Assert
        assert n_removed == 0
        assert kept.shape[0] == 5
