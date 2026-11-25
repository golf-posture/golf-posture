"""ViTPose-H (GolfPose `person`) top-down pre/post-processing without mmpose.

Ported from mmpose 1.3.2 (Apache-2.0) so a server only needs the exported network (see vitpose_export.py),
not the mmpose stack: the bbox -> UDP crop (`GetBBoxCenterScale` + `TopdownAffine(use_udp=True)`), the
`UDPHeatmap` decode with DARK refinement, the heatmap flip used by `flip_test`, and the mapping back to
image pixels. Only numpy and OpenCV are needed here.

The exported network takes the crop as float BGR 0..255 of shape (N, 3, 256, 192) and returns the
flip-averaged heatmaps (N, 17, 64, 48); normalisation and the flip test are inside it.

Keypoints are in Human3.6M order (see golfpose_joints.py). Equivalence to mmpose is checked by
vitpose_verify.py on the labelled frames.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

INPUT_SIZE = (192, 256)  # (w, h) of the network input
HEATMAP_SIZE = (48, 64)  # (W, H) of its output
BBOX_PADDING = 1.25
BLUR_KERNEL = 11  # DARK modulation, matches the training sigma of 2
FLIP_INDICES = (0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13)
MEAN_RGB = (123.675, 116.28, 103.53)
STD_RGB = (58.395, 57.12, 57.375)


def bbox_center_scale(
    box_xyxy: Sequence[float], padding: float = BBOX_PADDING, input_size: tuple[int, int] = INPUT_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Padded bbox as (center, scale), with the scale reshaped to the network's aspect ratio."""
    box = np.asarray(box_xyxy, dtype=np.float32)
    scale = (box[2:] - box[:2]) * padding
    center = (box[2:] + box[:2]) * 0.5
    aspect = input_size[0] / input_size[1]
    width, height = scale
    scale = np.array([width, width / aspect] if width > height * aspect else [height * aspect, height],
                     dtype=np.float32)
    return center, scale


def udp_warp_matrix(center: np.ndarray, scale: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """2x3 affine matrix of the unbiased (UDP) crop, rotation 0."""
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = scale_x
    matrix[0, 2] = scale_x * (-center[0] + 0.5 * scale[0])
    matrix[1, 1] = scale_y
    matrix[1, 2] = scale_y * (-center[1] + 0.5 * scale[1])
    return matrix


def crop_for_model(
    image_bgr: np.ndarray, box_xyxy: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The (256, 192, 3) BGR crop for one golfer box, with the center and scale needed to map back."""
    center, scale = bbox_center_scale(box_xyxy)
    matrix = udp_warp_matrix(center, scale, INPUT_SIZE)
    crop = cv2.warpAffine(image_bgr, matrix, INPUT_SIZE, flags=cv2.INTER_LINEAR)
    return crop, center, scale


def flip_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Mirror (K, H, W) heatmaps horizontally and swap the left/right keypoint channels."""
    return heatmaps[list(FLIP_INDICES), :, ::-1]


def _gaussian_blur(heatmaps: np.ndarray, kernel: int) -> np.ndarray:
    border = (kernel - 1) // 2
    count, height, width = heatmaps.shape
    for k in range(count):
        origin_max = np.max(heatmaps[k])
        padded = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
        padded[border:-border, border:-border] = heatmaps[k]
        padded = cv2.GaussianBlur(padded, (kernel, kernel), 0)
        heatmaps[k] = padded[border:-border, border:-border]
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def _refine_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """Sub-pixel refinement of integer peaks by a second-order fit on the log of the blurred heatmaps."""
    count = keypoints.shape[0]
    height, width = heatmaps.shape[1:]
    heatmaps = _gaussian_blur(heatmaps, BLUR_KERNEL)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)
    padded = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    index = keypoints[:, 0] + 1 + (keypoints[:, 1] + 1) * (width + 2)
    index = (index + (width + 2) * (height + 2) * np.arange(0, count)).astype(int).reshape(-1, 1)
    centre = padded[index]
    right, down = padded[index + 1], padded[index + width + 2]
    down_right, up_left = padded[index + width + 3], padded[index - width - 3]
    left, up = padded[index - 1], padded[index - 2 - width]

    derivative = np.concatenate([0.5 * (right - left), 0.5 * (down - up)], axis=1).reshape(count, 2, 1)
    dxx = right - 2 * centre + left
    dyy = down - 2 * centre + up
    dxy = 0.5 * (down_right - right - down + centre + centre - left - up + up_left)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(count, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    return keypoints - np.einsum("imn,ink->imk", hessian, derivative).squeeze()


def decode_heatmaps(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(K, H, W) heatmaps -> keypoints (K, 2) in network-input pixels and scores (K,). The input is left intact."""
    work = np.array(heatmaps, dtype=np.float32, copy=True)
    count, height, width = work.shape
    flat = work.reshape(count, -1)
    y_locs, x_locs = np.unravel_index(np.argmax(flat, axis=1), shape=(height, width))
    peaks = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    scores = np.amax(flat, axis=1)
    peaks[scores <= 0.0] = -1
    refined = _refine_dark_udp(peaks, work)
    heat_w, heat_h = HEATMAP_SIZE
    return refined / [heat_w - 1, heat_h - 1] * INPUT_SIZE, scores


def to_image_space(keypoints: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Network-input pixels -> original image pixels."""
    return keypoints / np.asarray(INPUT_SIZE, dtype=np.float32) * scale + center - 0.5 * scale


def model_input(crop_bgr: np.ndarray) -> np.ndarray:
    """(256, 192, 3) uint8 BGR crop -> float32 (3, 256, 192) BGR 0..255, as the exported network expects."""
    return np.ascontiguousarray(crop_bgr.transpose(2, 0, 1), dtype=np.float32)


def predict_keypoints(heatmaps: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One golfer: (K, H, W) network heatmaps -> image-pixel keypoints (K, 2) and scores (K,)."""
    keypoints, scores = decode_heatmaps(heatmaps)
    return to_image_space(keypoints, center, scale), scores
