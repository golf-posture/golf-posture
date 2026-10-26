from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.run_analysis import effective_extraction, resolve_stats_path


class TestEffectiveExtraction:
    def test_tracked_is_used_with_mediapipe(self):
        assert effective_extraction("tracked", "mediapipe") == ("tracked", None)

    @pytest.mark.parametrize("backend", ["yolopose", "auto", "hybrid"])
    def test_tracked_falls_back_to_static_for_other_backends_with_a_note(self, backend):
        mode, note = effective_extraction("tracked", backend)

        assert mode == "static"
        assert backend in note

    @pytest.mark.parametrize("backend", ["mediapipe", "yolopose", "auto", "hybrid"])
    def test_static_is_always_static(self, backend):
        assert effective_extraction("static", backend) == ("static", None)


class TestResolveStatsPath:
    def test_explicit_value_always_wins(self):
        assert resolve_stats_path("custom/x.json", "tracked", "pro_dynamics_stats.json") == Path("custom/x.json")

    def test_legacy_reference_lives_in_data_directory(self):
        assert resolve_stats_path(None, "legacy", "pro_dynamics_stats.json") == Path("data/pro_dynamics_stats.json")

    def test_tracked_reference_lives_in_tracked_directory(self):
        assert resolve_stats_path(None, "tracked", "pro_dynamics_stats.json") == Path(
            "data/tracked/pro_dynamics_stats.json"
        )
