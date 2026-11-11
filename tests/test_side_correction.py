from __future__ import annotations

import copy

import pytest

from pipeline.side_correction import (
    correct_arm_sides,
    needs_side_swap,
    swap_arm_sides,
)

TOP = 3
FINISH = 7


def _joint(px, py, vis=0.9, wx=0.0, wy=0.0, wz=0.0):
    return (px, py, vis, wx, wy, wz)


def _frame(right_shoulder_y=15.0, left_shoulder_y=25.0):
    """Golfer's trail (right) shoulder higher on screen than the lead (left) one: the anatomically correct order."""
    return {
        "NOSE": _joint(50, 10, wx=0.5),
        "RIGHT_SHOULDER": _joint(30, right_shoulder_y, wx=0.1),
        "RIGHT_ELBOW": _joint(30, 40, wx=0.2),
        "RIGHT_WRIST": _joint(50, 40, wx=0.3),
        "LEFT_SHOULDER": _joint(70, left_shoulder_y, wx=-0.1),
        "LEFT_ELBOW": _joint(70, 40, wx=-0.2),
        "LEFT_WRIST": _joint(70, 60, wx=-0.3),
        "LEFT_HIP": _joint(60, 70),
        "RIGHT_HIP": _joint(40, 70),
    }


def _mislabelled(frame):
    """The same pose as the model reports it when it has the left and right arms the wrong way round."""
    return swap_arm_sides(frame)


class TestNeedsSideSwap:
    def test_correct_order_needs_no_swap(self):
        assert needs_side_swap(_frame(), "right", min_gap=0.0) is False

    def test_trail_shoulder_lower_on_screen_needs_swap(self):
        assert needs_side_swap(_mislabelled(_frame()), "right", min_gap=0.0) is True

    def test_level_shoulders_are_left_alone(self):
        assert needs_side_swap(_frame(20.0, 20.0), "right", min_gap=0.0) is False

    def test_gap_below_minimum_is_left_alone(self):
        # shoulders 40 px apart horizontally, 4 px vertically: gap = 0.1 of the shoulder distance
        frame = _mislabelled(_frame(18.0, 22.0))

        assert needs_side_swap(frame, "right", min_gap=0.2) is False
        assert needs_side_swap(frame, "right", min_gap=0.05) is True

    def test_left_handed_golfer_mirrors_the_rule(self):
        # for a left-hander the trail arm is the left one, which should be the higher shoulder
        left_handed_correct = _frame(right_shoulder_y=25.0, left_shoulder_y=15.0)

        assert needs_side_swap(left_handed_correct, "left", min_gap=0.0) is False
        assert needs_side_swap(_frame(), "left", min_gap=0.0) is True

    @pytest.mark.parametrize("missing", ["RIGHT_SHOULDER", "LEFT_SHOULDER"])
    def test_missing_shoulder_is_left_alone(self, missing):
        frame = _mislabelled(_frame())
        del frame[missing]

        assert needs_side_swap(frame, "right", min_gap=0.0) is False

    def test_missing_frame_is_left_alone(self):
        assert needs_side_swap(None, "right", min_gap=0.0) is False


class TestSwapArmSides:
    def test_exchanges_shoulder_elbow_and_wrist_with_their_world_coordinates(self):
        frame = _frame()

        swapped = swap_arm_sides(frame)

        for part in ("SHOULDER", "ELBOW", "WRIST"):
            assert swapped[f"RIGHT_{part}"] == frame[f"LEFT_{part}"]
            assert swapped[f"LEFT_{part}"] == frame[f"RIGHT_{part}"]

    def test_leaves_head_and_hips_untouched_and_does_not_mutate_input(self):
        frame = _frame()
        before = copy.deepcopy(frame)

        swapped = swap_arm_sides(frame)

        assert frame == before
        for name in ("NOSE", "LEFT_HIP", "RIGHT_HIP"):
            assert swapped[name] == frame[name]

    def test_swapping_twice_restores_the_frame(self):
        frame = _frame()

        assert swap_arm_sides(swap_arm_sides(frame)) == frame

    def test_joint_missing_on_one_side_moves_with_the_arm(self):
        frame = _frame()
        del frame["LEFT_WRIST"]

        swapped = swap_arm_sides(frame)

        assert "RIGHT_WRIST" not in swapped
        assert swapped["LEFT_WRIST"] == frame["RIGHT_WRIST"]


class TestCorrectArmSides:
    EVENTS = [0, 6, 12, 18, 24, 30, 36, 42]

    def _frames(self, wrong_events=()):
        frames = [_frame() for _ in range(48)]
        for event in wrong_events:
            centre = self.EVENTS[event]
            for i in range(centre - 3, centre + 4):
                if 0 <= i < len(frames):
                    frames[i] = _mislabelled(_frame())
        return frames

    def test_correct_frames_pass_through_unchanged(self):
        frames = self._frames()

        corrected, flags = correct_arm_sides(frames, self.EVENTS, "right")

        assert corrected == frames
        assert flags == [False] * 8

    def test_swapped_top_is_corrected_with_its_neighbours(self):
        frames = self._frames(wrong_events=[TOP])

        corrected, flags = correct_arm_sides(frames, self.EVENTS, "right")

        assert flags[TOP] is True
        assert sum(flags) == 1
        assert corrected == self._frames()

    def test_swapped_finish_is_corrected_when_the_gap_is_large_enough(self):
        frames = self._frames(wrong_events=[FINISH])

        corrected, flags = correct_arm_sides(frames, self.EVENTS, "right")

        assert flags[FINISH] is True
        assert corrected == self._frames()

    def test_finish_with_a_tiny_shoulder_gap_is_left_alone(self):
        frames = [_mislabelled(_frame(19.0, 21.0)) for _ in range(48)]

        corrected, flags = correct_arm_sides(frames, self.EVENTS, "right")

        assert flags[FINISH] is False

    def test_top_is_corrected_even_when_the_shoulders_are_nearly_level(self):
        frames = [_mislabelled(_frame(19.0, 21.0)) for _ in range(48)]

        _, flags = correct_arm_sides(frames, self.EVENTS, "right")

        assert flags[TOP] is True

    def test_other_events_are_never_corrected(self):
        frames = self._frames(wrong_events=[2, 4, 5])

        corrected, flags = correct_arm_sides(frames, self.EVENTS, "right")

        assert flags == [False] * 8
        assert corrected == frames

    def test_event_frame_without_a_pose_is_skipped(self):
        frames = self._frames(wrong_events=[TOP])
        frames[self.EVENTS[TOP]] = None

        corrected, flags = correct_arm_sides(frames, self.EVENTS, "right")

        assert flags[TOP] is False
        assert corrected == frames

    def test_input_frames_are_not_mutated(self):
        frames = self._frames(wrong_events=[TOP])
        before = copy.deepcopy(frames)

        correct_arm_sides(frames, self.EVENTS, "right")

        assert frames == before

    def test_event_near_the_end_of_the_video_is_clipped(self):
        frames = [_mislabelled(_frame()) for _ in range(10)]
        events = [0, 1, 2, 3, 4, 5, 6, 9]

        corrected, flags = correct_arm_sides(frames, events, "right")

        assert flags[FINISH] is True
        assert corrected[9] == _frame()

    def test_overlapping_blocks_are_swapped_only_once(self):
        frames = [_mislabelled(_frame()) for _ in range(20)]
        events = [0, 1, 2, 8, 9, 10, 11, 12]  # Top (8) and Finish (12) blocks overlap

        corrected, flags = correct_arm_sides(frames, events, "right")

        assert flags[TOP] is True and flags[FINISH] is True
        assert all(corrected[i] == _frame() for i in range(5, 16))
