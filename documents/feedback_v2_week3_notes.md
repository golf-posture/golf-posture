# v2 3주차 작업 노트 — 스윙 다이내믹스 (문구 계층 제외)

작성일: 2026-09-25
브랜치: `feature/feedback-v2-week2`
범위: 문구 계층(B2/B3/B4)은 이번 3주차에서 제외, 이벤트 간 delta 다이내믹스 통계/연결만 진행.

## 구현된 것

- `pipeline/compute_dynamics_stats.py` 신규: `pro_feature_raw.csv`를 `video_id`로 묶어 연속 이벤트쌍 7개(Address→Toe-up ... Mid-follow-through→Finish) × 5 feature = 35 조합의 delta를 계산.
  - linear feature: `end - start`, circular feature(shoulder_tilt/hip_tilt): `circular_delta(end, start)`.
  - **median/MAD 기반 강건 통계**를 처음부터 사용(2주차에서 뒤늦게 고쳤던 방식을 이번엔 선반영) — `scaled_mad = 1.4826 * median(|x - median|)`.
  - 표본 수가 `MIN_SAMPLES_FOR_DYNAMICS=30` 미만인 조합은 `null`로 표시(현재 데이터에서는 전부 76~341 사이라 해당 없음).
  - 결과를 `data/pro_feature_covariance.json`과 별도인 신규 `data/pro_dynamics_stats.json`에 저장.
- `pipeline/run_analysis.py`
  - `dynamics_feedback_for_pair()` 신규: 사용자의 두 이벤트 feature 값(둘 다 검출된 경우만) → delta → robust_z(median/MAD 기준) → 기존 `grade_from_z`(|z|<=1/<=2) 등급 언어 그대로 재사용.
  - 결과 JSON에 **최상위 신규 섹션** `result["dynamics"]`(이벤트별 `Address->Toe-up` 등 7개 키)로 추가. 기존 `events.*.feedback`/`events.*.combined_score`는 전혀 변경 없음 — additive-only.
  - `--dynamics-stats` CLI 옵션 신규(기본값 `data/pro_dynamics_stats.json`), 없으면 각 구간이 `"unavailable"`로 graceful degrade.

## 회귀 확인

`raw_videos/` 6개 영상 전부 재실행:
- 기존 `feedback.<feature>.{z_score, grade, pro_mean, pro_std, value}` 40개 조합 x 6영상, byte-level 완전히 동일(mismatch 0건).
- `combined_score` 필드 그대로 유지.
- 신규 `dynamics` 섹션이 7개 구간 x 5 feature 모두 정상 채워짐, `errors: []` 유지.

## 재검증 (2026-09-25) — 이변량(bimodal) 구간 보정

3주차 구현을 "제대로 되었는지 한번 더 확인"하는 과정에서, 342명 pro 데이터를 그 자신의 통계로 재채점하는 self-consistency 방식으로 각 (event-pair, feature) 조합의 등급 분포를 점검했다.

- **발견**: `Mid-follow-through->Finish` 구간의 `shoulder_tilt` delta가 심하게 miscalibrate됨 — Excellent 56.7% / Good 4.1% / Needs Improvement 38.6% (다른 구간은 대략 Excellent 63-68% / Good 22-27% / Needs 7-11%).
- **원인 조사**: raw delta 히스토그램 확인 결과, circular wraparound 버그가 아니라 **진짜 이변량(bimodal) 분포**였음 — 다수 클러스터(-20~-60°)와 소수 클러스터(+120~+180°, 전체의 약 32%)가 실제로 존재. README에 이미 기록되어 있던 "Finish 단계 feature는 개인차가 크다"는 노트와 일치하는, 실제 스윙 다양성.
- **수정**: 새 통계 로직을 만드는 대신, v1부터 있던 `classify_reliability()`(`compute_reliability_stats.py`)를 재사용해 각 (event-pair, feature) 조합에 reliability 등급(high/medium/low)을 매김.
  - `pipeline/compute_dynamics_stats.py`: `classify_reliability`를 import, `n>=30`/`n<30` 두 분기 모두에서 `reliability`/`feedback_strength`/`reliability_reason`을 stats dict에 추가.
  - `pipeline/run_analysis.py`의 `dynamics_feedback_for_pair()`: 결과 item에 `reliability`(기본값 `"unknown"`)/`feedback_strength`(기본값 `"normal"`)를 전파하고, `reliability == "low"`이면서 등급이 `Needs Improvement`일 때만 메시지 끝에 "참고용 신호일 뿐 강한 교정 지시로 보지 말라"는 경고 문장을 덧붙임.
- **검증 결과**: `Mid-follow-through->Finish`가 `shoulder_tilt=low`, `hip_tilt=medium`, `spine_tilt=high` reliability로 표시됨 — 앞서 발견한 miscalibration 심각도와 정확히 일치(shoulder_tilt가 가장 심했고 low로 분류됨).
- **회귀 재확인**: `raw_videos/` 6개 영상 전부 재실행, 기존 `feedback.<feature>.{z_score, grade, pro_mean, pro_std, value}` 필드 byte-level 동일(mismatch 0건), `combined_score` 그대로 유지, `errors: []` 유지.

## 남은 범위 (별도 계획으로 이후 진행)

- 문구 계층(B2/B3/B4, `generate_feedback_report.py`)은 이번 3주차 계획에서 제외됨 — 다이내믹스/결합 스코어 결과를 실제 한국어 피드백 문장에 반영하는 작업은 아직 남아 있음. (reliability/feedback_strength 필드를 문구 생성 시 반영하는 것도 이 작업에 포함될 것.)
- 정식 pytest 스위트는 v2 계획서 일정대로 4주차 예정.
