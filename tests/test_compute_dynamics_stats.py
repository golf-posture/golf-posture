from __future__ import annotations

import pytest

from pipeline.compute_dynamics_stats import (
    MIN_SAMPLES_FOR_DYNAMICS,
    compute_delta,
    compute_dynamics_stats,
    event_pairs,
    pair_label,
    robust_stats,
)
from pipeline.run_analysis import EVENT_NAMES


class TestEventPairs:
    def test_produces_consecutive_pairs_covering_all_events(self):
        # Act
        pairs = event_pairs()

        # Assert
        assert pairs[0] == (0, 1)
        assert pairs[-1] == (len(EVENT_NAMES) - 2, len(EVENT_NAMES) - 1)
        assert len(pairs) == len(EVENT_NAMES) - 1

    def test_pair_label_joins_event_names_with_arrow(self):
        # Arrange
        start_idx, end_idx = 0, 1

        # Act
        label = pair_label(start_idx, end_idx)

        # Assert
        assert label == f"{EVENT_NAMES[0]}->{EVENT_NAMES[1]}"


class TestComputeDelta:
    def test_linear_feature_uses_plain_subtraction(self):
        # Act
        delta = compute_delta(10.0, 25.0, "trail_elbow_angle")

        # Assert
        assert delta == pytest.approx(15.0)

    def test_circular_feature_wraps_around_boundary(self):
        # Act
        delta = compute_delta(-179.0, 179.0, "shoulder_tilt")

        # Assert
        assert delta == pytest.approx(-2.0)


class TestRobustStats:
    def test_computes_median_and_scaled_mad_for_known_values_with_an_outlier(self):
        # Arrange
        values = [1.0, 2.0, 3.0, 4.0, 100.0]

        # Act
        stats = robust_stats(values)

        # Assert
        assert stats["median"] == pytest.approx(3.0)
        assert stats["mad"] == pytest.approx(1.0)
        assert stats["scaled_mad"] == pytest.approx(1.4826)
        assert stats["n"] == 5


def _make_row(video_id: str, event_index: int, **feature_values: float) -> dict:
    row = {"video_id": video_id, "event_index": str(event_index)}
    row.update({name: str(value) for name, value in feature_values.items()})
    return row


class TestComputeDynamicsStats:
    def test_marks_pair_feature_low_reliability_when_below_minimum_samples(self):
        # Arrange
        rows = [
            _make_row("video_1", 0, trail_elbow_angle=10.0),
            _make_row("video_1", 1, trail_elbow_angle=20.0),
        ]

        # Act
        result = compute_dynamics_stats(rows)

        # Assert
        pair = result["pairs"][f"{EVENT_NAMES[0]}->{EVENT_NAMES[1]}"]
        stat = pair["features"]["trail_elbow_angle"]
        assert stat["n"] == 1
        assert stat["median"] is None
        assert stat["reliability"] == "low"

    def test_computes_robust_stats_once_minimum_sample_size_is_reached(self):
        # Arrange
        rows = []
        for i in range(MIN_SAMPLES_FOR_DYNAMICS):
            rows.append(_make_row(f"video_{i}", 0, trail_elbow_angle=10.0))
            rows.append(_make_row(f"video_{i}", 1, trail_elbow_angle=15.0))

        # Act
        result = compute_dynamics_stats(rows)

        # Assert
        pair = result["pairs"][f"{EVENT_NAMES[0]}->{EVENT_NAMES[1]}"]
        stat = pair["features"]["trail_elbow_angle"]
        assert stat["n"] == MIN_SAMPLES_FOR_DYNAMICS
        assert stat["median"] == pytest.approx(5.0)
        assert stat["reliability"] == "high"

    def test_skips_pair_when_one_of_the_two_events_is_missing_for_a_video(self):
        # Arrange
        rows = [_make_row("video_1", 0, trail_elbow_angle=10.0)]

        # Act
        result = compute_dynamics_stats(rows)

        # Assert
        pair = result["pairs"][f"{EVENT_NAMES[0]}->{EVENT_NAMES[1]}"]
        stat = pair["features"]["trail_elbow_angle"]
        assert stat["n"] == 0
