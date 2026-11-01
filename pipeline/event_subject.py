"""Persistent golfer selection and whole-clip fixed crop. No event labels."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def track_people(detections):
    tracks = []
    for t, frame in enumerate(detections):
        boxes = np.asarray(frame["boxes"], dtype=float).reshape(-1, 4)
        active = [i for i, tr in enumerate(tracks) if t - tr[-1][0] <= 5]
        costs = np.full((len(active), len(boxes)), 100., dtype=float)
        for a, index in enumerate(active):
            old = np.asarray(tracks[index][-1][1]["box"])
            oh = max(old[3] - old[1], 1)
            for b, box in enumerate(boxes):
                bh = max(box[3] - box[1], 1)
                distance = np.linalg.norm((old[:2] + old[2:] - box[:2] - box[2:]) / 2) / max(oh, bh)
                ratio = bh / oh
                if distance < .4 and .5 < ratio < 2:
                    costs[a, b] = distance + .2 * abs(np.log(ratio))
        matched = {}
        if costs.size:
            for a, b in zip(*linear_sum_assignment(costs)):
                if costs[a, b] >= 100:
                    continue
                matched[int(b)] = active[a]
        for b in range(len(boxes)):
            value = {"box": frame["boxes"][b], "points": frame["points"][b], "confidence": frame["confidence"][b]}
            if b in matched:
                tracks[matched[b]].append((t, value))
            else:
                tracks.append([(t, value)])
    return tracks


def select_golfer(detections, width, height):
    count = len(detections)
    tracks = track_people(detections)
    ranking = []
    for i, tr in enumerate(tracks):
        boxes = np.asarray([v["box"] for _, v in tr])
        points = np.asarray([v["points"] for _, v in tr])
        body = float(np.median(boxes[:, 3] - boxes[:, 1]))
        # A close spectator truncated by the image is not a complete golfer.
        torso = (points[:, [5, 6, 11, 12], 2] >= .4).all(axis=1)
        legs = (points[:, [13, 14, 15, 16], 2] >= .35).all(axis=1)
        full = torso & legs
        wrists = (points[:, [9, 10], 2] >= .4).all(axis=1) & torso
        excursion = 0.
        if wrists.sum() >= 8 and body > 0:
            xy = points[wrists][:, [9, 10], :2].mean(1) - points[wrists][:, [11, 12], :2].mean(1)
            span = np.quantile(xy, .95, axis=0) - np.quantile(xy, .05, axis=0)
            excursion = float(np.linalg.norm(span) / body)
        coverage = len(tr) / max(count, 1)
        full_fraction = float(full.mean())
        score = coverage * (body / max(height, 1)) ** 2 * full_fraction ** 2 * (.25 + .75 * np.clip(excursion / .6, 0, 1))
        eligible = coverage >= .8 and full_fraction >= .6 and body >= .2 * height
        ranking.append({"track": i, "score": float(score) if eligible else 0., "coverage": coverage,
                        "full_body_fraction": full_fraction, "body_height": body,
                        "wrist_excursion": excursion, "eligible": bool(eligible)})
    ranking.sort(key=lambda r: r["score"], reverse=True)
    points = np.full((count, 17, 3), np.nan, np.float32)
    points[:, :, 2] = 0
    boxes = np.full((count, 4), np.nan, np.float32)
    detail = {"ranking": ranking, "accepted": False, "reason": "no_eligible_persistent_full_body_track"}
    if not ranking or ranking[0]["score"] <= 0:
        return points, boxes, detail
    best = ranking[0]
    ratio = ranking[1]["score"] / best["score"] if len(ranking) > 1 else 0.
    detail["runner_up_ratio"] = ratio
    if ratio > .8:
        detail["reason"] = "similar_subject_scores"
        return points, boxes, detail
    for t, v in tracks[best["track"]]:
        points[t] = v["points"]
        boxes[t] = v["box"]
    detail.update(accepted=True, reason="persistent_full_body_scale_and_motion", selected_track=best["track"])
    return points, boxes, detail


def square_crop(points, boxes, subject, width, height):
    full = [0, 0, width, height]
    result = {"xyxy": full, "applied": False, "reason": "subject_not_accepted", "club_coverage_verified": False}
    if not subject["accepted"]:
        return result
    valid = np.isfinite(boxes).all(1)
    body = float(np.median(boxes[valid, 3] - boxes[valid, 1]))
    # Keep the shorter image dimension. Landscape removes horizontal background;
    # portrait removes vertical background using exactly the same rule.
    center = np.median((boxes[valid, :2] + boxes[valid, 2:]) / 2, axis=0)
    side = max(min(width, height), int(np.ceil(1.75 * body)))
    cw, ch = min(width, side), min(height, side)
    left = max(0, min(width - cw, int(round(center[0] - cw / 2))))
    top = max(0, min(height - ch, int(round(center[1] - ch / 2))))
    right, bottom = left + cw, top + ch
    # Every confident selected-person joint must remain in the crop.
    visible = points[:, :, 2] >= .4
    if visible.any():
        left = max(0, min(left, int(np.floor(np.min(points[:, :, 0][visible]) - .1 * body))))
        right = min(width, max(right, int(np.ceil(np.max(points[:, :, 0][visible]) + .1 * body))))
        top = max(0, min(top, int(np.floor(np.min(points[:, :, 1][visible]) - .1 * body))))
        bottom = min(height, max(bottom, int(np.ceil(np.max(points[:, :, 1][visible]) + .1 * body))))
    area = (right - left) * (bottom - top) / (width * height)
    if area >= .92:
        result["reason"] = "little_background_to_remove"
        return result
    result.update(xyxy=[left, top, right, bottom], applied=True, reason="whole_clip_fixed_long_axis_crop",
                  body_height=body, center_xy=center.tolist(), area_fraction=area)
    return result
