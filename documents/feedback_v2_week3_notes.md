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

## 남은 범위 (별도 계획으로 이후 진행)

- 문구 계층(B2/B3/B4, `generate_feedback_report.py`)은 이번 3주차 계획에서 제외됨 — 다이내믹스/결합 스코어 결과를 실제 한국어 피드백 문장에 반영하는 작업은 아직 남아 있음.
- 정식 pytest 스위트는 v2 계획서 일정대로 4주차 예정.
