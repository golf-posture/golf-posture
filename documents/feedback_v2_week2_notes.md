# v2 2주차 작업 노트 — Mahalanobis 결합 스코어 (+ 클러스터링 스킵 확정)

작성일: 2026-09-25
브랜치: `feature/feedback-v2-week2` (`feature/feedback-v2`에서 분기)

## 구현된 것

### 1. Mahalanobis 결합 스코어 (v2 계획 항목 1)

- `pipeline/compute_reliability_stats.py`
  - `compute_covariance_stats()` 신규: 이벤트별 5-feature(circular_delta 보정 포함) complete-case 공분산 행렬 계산.
  - `shrink_covariance()` 신규: 조건수가 `SHRINKAGE_TRIGGER_CONDITION=50`을 넘으면 `(1-α)·Σ + α·avg_variance·I` 형태로 축소, 조건수가 `SHRINKAGE_TARGET_CONDITION=30` 이하가 되거나 `α`가 `SHRINKAGE_MAX_ALPHA=0.5`에 도달할 때까지 `α`를 `0.05` 단위로 증가.
  - 결과를 `data/pro_feature_covariance.json`에 저장(기존 `pro_feature_stats_reliability.json`은 변경 없음, 신규 파일로 추가).
  - 실제 실행 결과 (`α`가 데이터 기반으로 자동 결정됨, 하드코딩한 "3개 이벤트" 지정 없이 조건수 임계값만으로 판단):

    | 이벤트 | 조건수(적용 전) | 조건수(적용 후) | α |
    |---|---|---|---|
    | Address | 53.08 | 22.17 | 0.10 |
    | Toe-up | 26.34 | 26.34 | 0.00 |
    | Mid-backswing | 281.35 | 22.58 | 0.10 |
    | Top | 32.04 | 32.04 | 0.00 |
    | Mid-downswing | 310.79 | 25.91 | 0.10 |
    | Impact | 22.27 | 22.27 | 0.00 |
    | Mid-follow-through | 47.27 | 47.27 | 0.00 |
    | Finish | 540.64 | 23.86 | 0.10 |

- `pipeline/run_analysis.py`
  - `mahalanobis_combined_score()` 신규: 5-feature 벡터(circular_delta 적용) → precision 행렬로 마할라노비스 거리 계산 → `scipy.stats.chi2.cdf(df=5)`로 백분위 환산 → `grade_from_combined_percentile()`로 등급화(68.27%/95.45% 경계 — 기존 `grade_from_z`의 |z|<=1/|z|<=2가 의미하는 1σ/2σ 양측 커버리지와 동일한 기준을 다변량으로 확장한 것).
  - 이벤트 dict에 `combined_score` 필드로 신규 추가. **기존 `feedback.<feature>.z_score/grade/pro_mean/pro_std/value`는 전혀 변경하지 않음** — 추가 전용(additive-only).
  - `--covariance-stats` CLI 옵션 신규(기본값 `data/pro_feature_covariance.json`), 파일이 없으면 `combined_score`가 `"unavailable"`로 graceful degrade (기존 reliability_stats 로딩 패턴과 동일).

### 2. 클러스터링 (v2 계획 항목 3) — 최종 스킵

1주차 진단(40차원, complete-case 13명, 실루엣 0.09~0.14)에 이어 2주차에 `trail_elbow_angle` 제외 32차원으로 재진단했을 때 처음에는 k=2에서 실루엣 0.4574가 나와 유의미해 보였으나, **실제로는 이상치 2명(video_id 160, 440)을 분리한 결과**였음을 검증 과정에서 발견(중심으로부터 거리가 중앙값의 2~3배). 이상치를 제거하고 재계산하면 실루엣이 0.06~0.11로 붕괴 — 진짜 클러스터 구조는 없음.

**최종 결정**: 3번(클러스터링)은 이번 4주 범위에서 완전히 스킵. `pipeline/cluster_pro_features.py`(검증 중 작성했던 스크립트)와 그 결과 파일은 삭제함 — 사용되지 않을 코드를 남기지 않기 위함. 스켈레톤 비교는 기존 전역 medoid(`data/pro_reference_skeletons.json`)를 그대로 사용. 상세 근거는 `documents/feedback_v2_week1_diagnosis.md`의 "2주차 재진단" 절 참고.

원본 342개 영상(`learning_video/golfdb_720p_slice`, 확보 완료)은 이번엔 클러스터별 skeleton 생성에 쓰이지 않았지만, 향후 pose 오검출 이상치를 골라내는 QC 작업이나 다른 재추출 작업에는 계속 사용 가능.

## 회귀 확인

`raw_videos/`의 6개 영상(longform/shorts/shorts2/test1/test2/test3) 전부를 새 파이프라인(`--covariance-stats` 옵션 포함)으로 재실행해 1주차 베이스라인(`documents/baseline_eval/2026-09-25/`)과 비교:

- 기존 `feedback.<feature>.{z_score, grade, pro_mean, pro_std, value}` 40개 조합 x 6영상, 전부 **완전히 동일**(byte-level 비교, mismatch 0건).
- 6개 영상 전부 신규 `combined_score` 필드가 정상적으로 추가됨, `errors: []` 유지.
- 정식 pytest 스위트는 v2 계획서 일정대로 4주차에 작성 예정(1주차와 동일한 유예 사유).

## 남은 백로그

- pose 오검출로 보이는 이상치 프레임(예: video 160/440 특정 이벤트)에 대한 `extract_pro_features.py` QC 규칙 — 이번 범위 밖.
- 클러스터링(3번)은 이상치 정제 없이는 재시도해도 같은 문제가 반복될 가능성이 높음 — 향후 재검토 시 이상치 제거를 선행 단계로 포함해야 함.
