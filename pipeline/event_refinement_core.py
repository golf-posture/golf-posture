"""Frozen centered-context and phase-ranker inference primitives."""
import numpy as np
import torch
from .event_selection import ordered_event_frames

OFFSETS = np.array([-16, -8, -4, -2, 0, 2, 4, 8, 16])
REFINED = [0, 1, 2, 3, 7]


def centered_scores(model, features, device="cpu", batch_size=32):
    n = len(features)
    length = min(n, 64)
    starts = np.maximum(0, np.minimum(np.arange(n) - 32, n - length))
    unique = np.unique(starts)
    scores = np.zeros((n, 9), dtype=np.float32)
    for first in range(0, len(unique), batch_size):
        part = unique[first:first + batch_size]
        x = torch.stack([features[s:s + length] for s in part]).to(device)
        hidden = model.init_hidden(len(part), device)
        y, _ = model.rnn(x, hidden)
        y = model.lin(y).softmax(-1).cpu().numpy()
        frames = np.flatnonzero((starts >= part[0]) & (starts <= part[-1]))
        scores[frames] = y[np.searchsorted(part, starts[frames]), frames - starts[frames]]
    assert np.isfinite(scores).all() and np.allclose(scores.sum(1), 1, atol=2e-5)
    return scores


def features(centered, points, boxes, base, use_pose):
    n = len(centered)
    stride = max(1, min(8, round((base[6] - base[4]) / 8)))
    rows = np.clip(np.arange(n)[:, None] + OFFSETS[None] * stride, 0, n - 1)
    parts = [np.log(np.clip(centered, 1e-4, 1.))[rows].reshape(n, -1)]
    if use_pose:
        good_box = np.isfinite(boxes).all(1) & (boxes[:, 3] > boxes[:, 1])
        body = float(np.median(boxes[good_box, 3] - boxes[good_box, 1])) if good_box.any() else 1.
        ids = [9, 10, 11, 12]
        valid = np.isfinite(points[:, ids]).all((1, 2)) & (points[:, ids, 2] >= .4).all(1)
        xy = (points[:, [9, 10], :2].mean(1) - points[:, [11, 12], :2].mean(1)) / body
        xy = np.where(valid[:, None], np.clip(xy, -2, 2), 0.)
        # No interpolation: missing observations have a separate validity bit.
        before = np.maximum(np.arange(n) - stride, 0)
        motion_valid = valid & valid[before]
        speed = np.where(motion_valid, np.linalg.norm(xy - xy[before], axis=1), 0.)
        pose = np.column_stack([xy, speed, valid.astype(float), motion_valid.astype(float)])
        parts.append(pose[rows].reshape(n, -1))
    result = np.column_stack(parts).astype(np.float64)
    assert np.isfinite(result).all()
    return result


def anchor_base(centered):
    if len(centered) < 8:
        raise ValueError("Eight ordered events require at least eight frames")
    # Reserve four pre-downswing events and one post-follow-through event.
    anchors = (np.asarray(ordered_event_frames(centered[4:-1, 4:7])) + 4).tolist()
    # Other slots only satisfy the experimental ranker's input shape. The
    # ranker reads slots4..6 exclusively and predicts all other events afresh.
    return [0, 1, 2, 3, *anchors, len(centered) - 1]


def candidates(base, n, event):
    return np.arange(base[4]) if event < 4 else np.arange(base[6] + 1, n)


def predict_one(centered, points, boxes, base, model):
    x = features(centered, points, boxes, base, True)
    scores = np.zeros((len(centered), 8), float)
    for event in [4, 5, 6]:
        scores[base[event], event] = 1.
    for event in REFINED:
        allowed = candidates(base, len(centered), event)
        fit = model["events"][str(event)]
        z = (x[allowed] - np.asarray(fit["mean"])) / np.asarray(fit["scale"])
        logits = z @ np.asarray(fit["weights"])
        if model["residual"]:
            logits += np.log(np.maximum(centered[allowed, event], 1e-7))
        scores[allowed, event] = np.exp(np.maximum(logits - logits.max(), -100))
    predicted = ordered_event_frames(scores)
    assert predicted[4:7] == list(base[4:7]) and np.all(np.diff(predicted) > 0)
    return predicted, scores
