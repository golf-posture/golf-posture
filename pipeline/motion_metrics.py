"""Landmark-based swing motion metrics.

Everything here works on *landmarks per event*, so the exact same functions are
used (a) to build the pro reference stats from the 342 GolfDB videos and
(b) to score a user's video. That keeps pro and user numbers comparable.

Input shape for every metric:
    lm_by_event: {event_index: {LANDMARK_NAME: (x, y[, visibility]) | None} | None}
Event indices follow EVENT_NAMES: 0=Address, 3=Top, 5=Impact.

Assumption: face-on (front) camera. Several metrics (turn angles, weight
shift) are 2D approximations that are only meaningful from the front.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

ADDRESS, TOP, IMPACT = 0, 3, 5

# ---------------------------------------------------------------------------
# Reliability / grading rules -- kept in ONE place so they can be swapped for
# the team's shared rule later.
# ---------------------------------------------------------------------------
# reliability = high   if detection_rate >= 0.85 and robust_cv <= 0.25
#               medium if detection_rate >= 0.70 and robust_cv <= 0.50
#               low    otherwise
# robust_cv = scaled_MAD / max(|median|, scale_floor)
RELIABILITY_RULE = {"high": (0.85, 0.25), "medium": (0.70, 0.50)}
FEEDBACK_STRENGTH = {"high": "strong", "medium": "normal", "low": "reference"}
GRADE_Z_CUTS = (1.0, 2.0)  # |z| <= 1 Excellent, <= 2 Good, else Needs Improvement
SCORE_PER_Z = 30.0  # score = 100 - |z| * 30 (same shape as the existing angle score)
MIN_PRO_SAMPLES = 30


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _pt(lm: dict | None, name: str) -> np.ndarray | None:
    if not lm:
        return None
    value = lm.get(name)
    if value is None:
        return None
    return np.array([float(value[0]), float(value[1])])


def _mid(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _dist(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def torso_length(lm: dict | None) -> float | None:
    """Distance between mid-shoulder and mid-hip. Used to remove camera scale."""
    mid_s = _mid(_pt(lm, "LEFT_SHOULDER"), _pt(lm, "RIGHT_SHOULDER"))
    mid_h = _mid(_pt(lm, "LEFT_HIP"), _pt(lm, "RIGHT_HIP"))
    length = _dist(mid_s, mid_h)
    if length is None or length < 1e-6:
        return None
    return length


def _width_ratio(lm: dict | None, left: str, right: str) -> float | None:
    """Apparent left-right width divided by torso length (scale free)."""
    width = _dist(_pt(lm, left), _pt(lm, right))
    torso = torso_length(lm)
    if width is None or torso is None:
        return None
    return width / torso


def _turn_degrees(lm_by_event: dict, left: str, right: str) -> float | None:
    """Rotation estimated from width foreshortening in a front view.

    width(theta) ~ width0 * |cos(theta)|, so theta ~ arccos(width / width0)
    with the Address frame as the 0-degree reference. 2D approximation.
    """
    at_address = _width_ratio(lm_by_event.get(ADDRESS), left, right)
    at_top = _width_ratio(lm_by_event.get(TOP), left, right)
    if at_address is None or at_top is None or at_address < 1e-6:
        return None
    ratio = min(1.0, max(0.0, at_top / at_address))
    return float(math.degrees(math.acos(ratio)))


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def shoulder_turn_top(lm_by_event: dict) -> float | None:
    return _turn_degrees(lm_by_event, "LEFT_SHOULDER", "RIGHT_SHOULDER")


def hip_turn_top(lm_by_event: dict) -> float | None:
    return _turn_degrees(lm_by_event, "LEFT_HIP", "RIGHT_HIP")


def xfactor_turn_top(lm_by_event: dict) -> float | None:
    shoulder = shoulder_turn_top(lm_by_event)
    hip = hip_turn_top(lm_by_event)
    if shoulder is None or hip is None:
        return None
    return shoulder - hip


def head_movement(lm_by_event: dict) -> float | None:
    """Nose displacement Address -> Impact, in torso lengths."""
    address = lm_by_event.get(ADDRESS)
    torso = torso_length(address)
    moved = _dist(_pt(address, "NOSE"), _pt(lm_by_event.get(IMPACT), "NOSE"))
    if torso is None or moved is None:
        return None
    return moved / torso


def _hip_center(lm: dict | None) -> np.ndarray | None:
    return _mid(_pt(lm, "LEFT_HIP"), _pt(lm, "RIGHT_HIP"))


def weight_shift(lm_by_event: dict) -> float | None:
    """Horizontal hip-center travel Address -> Impact, in torso lengths.

    Absolute value: GolfDB has no handedness label, so the direction toward the
    target cannot be fixed. Reverse pivots therefore look like a normal shift.
    """
    address = lm_by_event.get(ADDRESS)
    torso = torso_length(address)
    start = _hip_center(address)
    end = _hip_center(lm_by_event.get(IMPACT))
    if torso is None or start is None or end is None:
        return None
    return abs(float(end[0] - start[0])) / torso


def hip_height_change(lm_by_event: dict) -> float | None:
    """Hip-center rise Address -> Impact in torso lengths (positive = standing up)."""
    address = lm_by_event.get(ADDRESS)
    torso = torso_length(address)
    start = _hip_center(address)
    end = _hip_center(lm_by_event.get(IMPACT))
    if torso is None or start is None or end is None:
        return None
    return float(start[1] - end[1]) / torso  # image y grows downward


def stance_change(lm_by_event: dict) -> float | None:
    """Relative change of the ankle-to-ankle distance Address -> Impact."""
    start = _dist(_pt(lm_by_event.get(ADDRESS), "LEFT_ANKLE"), _pt(lm_by_event.get(ADDRESS), "RIGHT_ANKLE"))
    end = _dist(_pt(lm_by_event.get(IMPACT), "LEFT_ANKLE"), _pt(lm_by_event.get(IMPACT), "RIGHT_ANKLE"))
    if start is None or end is None or start < 1e-6:
        return None
    return abs(end - start) / start


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    name: str
    label_ko: str
    compute: Callable[[dict], float | None]
    fmt: str  # python format for the value in Korean text
    bad_direction: str  # "high" | "low" | "both": which deviation is a problem
    scale_floor: float  # smallest scale that is meaningful (for robust CV)
    advice_high: str
    advice_low: str


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        "shoulder_turn_top", "탑에서의 어깨 회전각", shoulder_turn_top, "{:.0f}°", "low", 20.0,
        "프로보다 어깨를 더 많이 회전했습니다.",
        "프로보다 어깨 회전이 부족합니다. 백스윙에서 어깨를 더 충분히 돌려주는 것이 좋습니다.",
    ),
    MetricSpec(
        "hip_turn_top", "탑에서의 골반 회전각", hip_turn_top, "{:.0f}°", "both", 20.0,
        "프로보다 골반이 많이 회전했습니다. 하체를 더 단단히 잡아주는 것이 좋습니다.",
        "프로보다 골반 회전이 적습니다.",
    ),
    MetricSpec(
        "xfactor_turn_top", "탑에서의 어깨-골반 회전 분리각(X-factor)", xfactor_turn_top, "{:.0f}°", "low", 15.0,
        "프로보다 어깨와 골반의 회전 차이가 큽니다.",
        "프로보다 어깨와 골반의 회전 차이가 작아 몸통의 꼬임이 부족합니다.",
    ),
    MetricSpec(
        "head_movement", "머리 움직임(어드레스→임팩트, 몸통 길이 대비)", head_movement, "{:.2f}배", "high", 0.05,
        "프로보다 머리가 많이 움직입니다. 스윙 중 머리 위치를 고정하는 것이 좋습니다.",
        "프로보다 머리가 덜 움직입니다.",
    ),
    MetricSpec(
        "weight_shift", "체중 이동(골반 좌우 이동량, 어드레스→임팩트, 몸통 길이 대비)", weight_shift, "{:.2f}배", "both", 0.05,
        "프로보다 골반이 많이 이동했습니다. 과도한 흔들림(스웨이)이 아닌지 확인해보세요.",
        "프로보다 골반의 좌우 이동이 적습니다. 체중 이동이 부족할 수 있습니다.",
    ),
    MetricSpec(
        "hip_height_change", "골반 높이 변화(어드레스→임팩트, 몸통 길이 대비, 올라가면 +)", hip_height_change, "{:+.2f}배", "high", 0.05,
        "임팩트에서 프로보다 골반이 많이 올라옵니다. 자세가 일어서지 않도록 유지하는 것이 좋습니다.",
        "프로보다 골반 높이가 더 낮게 유지됩니다.",
    ),
    MetricSpec(
        "stance_change", "스탠스 폭 변화율(어드레스→임팩트)", stance_change, "{:.0%}", "high", 0.05,
        "스윙 중 발 간격이 프로보다 많이 변합니다. 발이 밀리지 않게 안정적으로 유지하는 것이 좋습니다.",
        "프로보다 발 간격이 더 안정적입니다.",
    ),
]
SPEC_BY_NAME = {spec.name: spec for spec in METRIC_SPECS}


def format_value(spec: "MetricSpec", value: float) -> str:
    text = spec.fmt.format(value)
    if text.startswith("-") and float(text.strip("-+%°배").replace("%", "") or 0) == 0.0:
        text = text[1:]  # "-0.00배" -> "0.00배"
    return text


def compute_all(lm_by_event: dict) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for spec in METRIC_SPECS:
        try:
            value = spec.compute(lm_by_event)
        except Exception:  # a broken landmark must never crash the analysis
            value = None
        values[spec.name] = None if value is None or not math.isfinite(value) else float(value)
    return values


# ---------------------------------------------------------------------------
# Pro statistics
# ---------------------------------------------------------------------------
def _bimodality_coefficient(values: np.ndarray) -> float | None:
    n = len(values)
    if n < 10:
        return None
    centered = values - values.mean()
    m2 = float(np.mean(centered**2))
    if m2 < 1e-12:
        return None
    skew = float(np.mean(centered**3)) / m2**1.5
    kurt = float(np.mean(centered**4)) / m2**2 - 3.0
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (skew**2 + 1.0) / denom if denom > 0 else None


def classify_reliability(detection_rate: float, robust_cv: float | None) -> tuple[str, str]:
    if robust_cv is None:
        return "low", "no usable distribution"
    high_rate, high_cv = RELIABILITY_RULE["high"]
    med_rate, med_cv = RELIABILITY_RULE["medium"]
    if detection_rate >= high_rate and robust_cv <= high_cv:
        return "high", "low relative dispersion and high coverage"
    if detection_rate >= med_rate and robust_cv <= med_cv:
        return "medium", "moderate relative dispersion or coverage"
    return "low", "high relative dispersion or low coverage"


def build_pro_stats(values: list[float], total: int, spec: MetricSpec) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    detection_rate = n / total if total else 0.0
    if n < MIN_PRO_SAMPLES:
        return {
            "n": n, "total": total, "detection_rate": detection_rate, "median": None, "scaled_mad": None,
            "reliability": "low", "feedback_strength": "reference",
            "reliability_reason": f"fewer than {MIN_PRO_SAMPLES} valid pro samples",
        }
    median = float(np.median(arr))
    scaled_mad = float(np.median(np.abs(arr - median)) * 1.4826)
    robust_cv = scaled_mad / max(abs(median), spec.scale_floor)
    reliability, reason = classify_reliability(detection_rate, robust_cv)
    p5, p25, p75, p95 = (float(x) for x in np.percentile(arr, [5, 25, 75, 95]))
    bc = _bimodality_coefficient(arr)
    quantiles = [float(x) for x in np.percentile(arr, np.linspace(0, 100, 21))]  # 0%,5%,...,100%
    return {
        "n": n,
        "total": total,
        "detection_rate": detection_rate,
        "median": median,
        "scaled_mad": scaled_mad,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "robust_cv": robust_cv,
        "quantiles": quantiles,  # every 5% -> used to say "N of 100 pros ..." in user-facing text
        "bimodality_coefficient": bc,  # diagnostic only; > ~0.555 hints at two clusters
        "reliability": reliability,
        "feedback_strength": FEEDBACK_STRENGTH[reliability],
        "reliability_reason": reason,
    }


# ---------------------------------------------------------------------------
# Scoring a user's video
# ---------------------------------------------------------------------------
LEVEL_KO = {"Excellent": "양호", "Good": "주의", "Needs Improvement": "개선 필요", "unavailable": "평가 불가"}


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def share_of_pros_at_or_below(value: float, stat: dict[str, Any], robust_z: float | None) -> float | None:
    """Fraction (0-1) of the pro videos whose value is <= `value`.

    Uses the stored empirical quantiles when present; otherwise falls back to a
    normal approximation of the robust z-score.
    """
    quantiles = stat.get("quantiles")
    if quantiles and len(quantiles) >= 3:
        grid = np.linspace(0.0, 1.0, len(quantiles))
        return float(np.interp(value, np.asarray(quantiles, dtype=float), grid))
    if robust_z is not None:
        return _normal_cdf(robust_z)
    return None


def grade_from_penalty(penalty: float | None) -> str:
    if penalty is None:
        return "unavailable"
    if penalty <= GRADE_Z_CUTS[0]:
        return "Excellent"
    if penalty <= GRADE_Z_CUTS[1]:
        return "Good"
    return "Needs Improvement"


def evaluate_motion_metrics(
    lm_by_event: dict,
    motion_stats: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    values = compute_all(lm_by_event)
    results: dict[str, dict[str, Any]] = {}
    for spec in METRIC_SPECS:
        value = values[spec.name]
        stat = (motion_stats or {}).get(spec.name) or {}
        median = stat.get("median")
        mad = stat.get("scaled_mad")
        item: dict[str, Any] = {
            "label_ko": spec.label_ko,
            "value": value,
            "value_text": None if value is None else format_value(spec, value),
            "pro_median": median,
            "pro_median_text": None if median is None else format_value(spec, median),
            "pro_scaled_mad": mad,
            "robust_z": None,
            "penalty_z": None,
            "direction": None,
            "grade": "unavailable",
            "score": None,
            "reliability": stat.get("reliability", "unknown"),
            "feedback_strength": stat.get("feedback_strength", "normal"),
            "reliability_reason": stat.get("reliability_reason", "motion stats not available"),
            "bad_direction": spec.bad_direction,
            "advice": "",
            "level_ko": LEVEL_KO["unavailable"],
            "percentile": None,
            "tail_pct": None,
            "tail_side": None,
            "pro_range_text": None,
        }
        if stat.get("p5") is not None and stat.get("p95") is not None:
            item["pro_range_text"] = f"{format_value(spec, stat['p5'])}~{format_value(spec, stat['p95'])}"
        if value is None:
            item["reliability_reason"] = "required landmarks were not detected in the Address/Top/Impact frames"
        elif median is not None and mad is not None and mad > 1e-9:
            z = (value - median) / mad
            item["robust_z"] = float(z)
            item["direction"] = "high" if z > 0 else "low"
            if spec.bad_direction == "high":
                penalty = max(0.0, z)
            elif spec.bad_direction == "low":
                penalty = max(0.0, -z)
            else:
                penalty = abs(z)
            item["penalty_z"] = float(penalty)
            item["grade"] = grade_from_penalty(penalty)
            item["level_ko"] = LEVEL_KO[item["grade"]]
            item["score"] = round(max(0.0, 100.0 - SCORE_PER_Z * penalty), 1)
            cdf = share_of_pros_at_or_below(value, stat, z)
            if cdf is not None:
                item["percentile"] = round(cdf * 100.0, 1)
                above_median = value >= median
                item["tail_side"] = "high" if above_median else "low"
                item["tail_pct"] = round((1.0 - cdf if above_median else cdf) * 100.0, 1)
            if penalty > GRADE_Z_CUTS[0]:
                item["advice"] = spec.advice_high if z > 0 else spec.advice_low
        results[spec.name] = item
    return results


def landmarks_by_event_from_result(events: dict[str, Any], event_names: list[str]) -> dict[int, dict | None]:
    """Build lm_by_event from run_analysis' result['events'] dict."""
    lm_by_event: dict[int, dict | None] = {}
    for index, name in enumerate(event_names):
        lm_by_event[index] = (events.get(name) or {}).get("selected_landmarks")
    return lm_by_event
