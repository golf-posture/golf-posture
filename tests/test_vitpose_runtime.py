"""ViTPose-H top-down pre/post-processing without mmpose (ported from mmpose 1.3.2, UDP)."""

import numpy as np
import pytest

from pipeline.vitpose_runtime import (
    FLIP_INDICES,
    HEATMAP_SIZE,
    INPUT_SIZE,
    bbox_center_scale,
    crop_for_model,
    decode_heatmaps,
    flip_heatmaps,
    to_image_space,
    udp_warp_matrix,
)


def _gaussian_heatmaps(points, sigma=2.0):
    width, height = HEATMAP_SIZE
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2)) for x, y in points]).astype(np.float32)


class TestBboxCenterScale:
    def test_padding_and_aspect_ratio_for_a_tall_box(self):
        center, scale = bbox_center_scale([100, 50, 200, 250])
        assert center == pytest.approx([150, 150])
        # padded 125 x 250; 125 < 250 * 0.75, so the width grows to 187.5
        assert scale == pytest.approx([187.5, 250])

    def test_wide_box_grows_the_height(self):
        _, scale = bbox_center_scale([0, 0, 400, 100])
        assert scale == pytest.approx([500, 500 / 0.75])


class TestUdpWarpMatrix:
    def test_bbox_center_lands_on_the_crop_centre(self):
        center, scale = bbox_center_scale([100, 50, 200, 250])
        matrix = udp_warp_matrix(center, scale, INPUT_SIZE)
        x, y = matrix @ np.array([center[0], center[1], 1.0])
        assert (x, y) == pytest.approx(((INPUT_SIZE[0] - 1) / 2, (INPUT_SIZE[1] - 1) / 2))


class TestCropForModel:
    def test_shape_and_centre_pixel(self):
        image = np.zeros((300, 200, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(200, dtype=np.uint8)[None, :]
        crop, center, scale = crop_for_model(image, [50, 50, 150, 150])
        assert crop.shape == (INPUT_SIZE[1], INPUT_SIZE[0], 3)
        assert abs(int(crop[crop.shape[0] // 2, crop.shape[1] // 2, 0]) - 100) <= 1
        assert center == pytest.approx([100, 100])


class TestDecodeHeatmaps:
    def test_recovers_a_subpixel_peak(self):
        points = [(20.3, 31.7)] * 17
        keypoints, scores = decode_heatmaps(_gaussian_heatmaps(points))
        width, height = HEATMAP_SIZE
        expected = (20.3 / (width - 1) * INPUT_SIZE[0], 31.7 / (height - 1) * INPUT_SIZE[1])
        assert keypoints.shape == (17, 2) and scores.shape == (17,)
        assert keypoints[0] == pytest.approx(expected, abs=0.5)
        assert scores[0] == pytest.approx(1.0, abs=0.05)

    def test_does_not_modify_its_input(self):
        heatmaps = _gaussian_heatmaps([(10.0, 10.0)] * 17)
        before = heatmaps.copy()
        decode_heatmaps(heatmaps)
        assert np.array_equal(heatmaps, before)

    def test_empty_heatmap_has_zero_score(self):
        _, scores = decode_heatmaps(np.zeros((17, HEATMAP_SIZE[1], HEATMAP_SIZE[0]), dtype=np.float32))
        assert scores.tolist() == [0.0] * 17


class TestToImageSpace:
    def test_crop_centre_maps_back_to_the_bbox_centre(self):
        center, scale = bbox_center_scale([100, 50, 200, 250])
        point = to_image_space(np.array([[INPUT_SIZE[0] / 2, INPUT_SIZE[1] / 2]]), center, scale)
        assert point[0] == pytest.approx(center)


class TestFlipHeatmaps:
    def test_flip_swaps_left_right_channels_and_mirrors_x(self):
        heatmaps = np.arange(17 * 2 * 3, dtype=np.float32).reshape(17, 2, 3)
        flipped = flip_heatmaps(heatmaps)
        assert np.array_equal(flipped[1], heatmaps[FLIP_INDICES[1]][:, ::-1])
        assert np.array_equal(flipped[0], heatmaps[0][:, ::-1])

    def test_flipping_twice_restores_the_input(self):
        heatmaps = np.random.default_rng(0).random((17, 4, 5)).astype(np.float32)
        assert np.array_equal(flip_heatmaps(flip_heatmaps(heatmaps)), heatmaps)
