# v2 2주차 작업 노트 — Mahalanobis 결합 스코어 (+ 클러스터링 스킵 확정)

작성일: 2026-09-25
브랜치: `feature/feedback-v2-week2` (`feature/feedback-v2`에서 분기)

## 구현된 것

### 1. Mahalanobis 결합 스코어 (v2 계획 항목 1)

- `pipeline/compute_reliability_stats.py`
  - `compute_covariance_stats()` 신규: 이벤트별 5-feature(circular_delta 보정 포함) complete-case 공분산 행렬 계산.
  - `shrink_covariance()` 신규: 조건수가 `SHRINKAGE_TRIGGER_CONDITION=50`을 넘으면 `(1-α)·Σ + α·avg_variance·I` 형태로 축소, 조건수가 `SHRINKAGE_TARGET_CONDITION=30` 이하가 되거나 `α`가 `SHRINKAGE_MAX_ALPHA=0.5`에 도달할 때까지 `α`를 `0.05` 단위로 증가.
  - 결과를 `data/pro_feature_covariance.json`에 저장(기존 `pro_feature_stats_reliability.json`은 변경 없음, 신규 파일로 추가).
  - 실제 실행 결과(아래 "1.5" 절의 이상치 트리밍까지 반영된 최종 값, `α`는 데이터 기반 자동 결정):

    | 이벤트 | n_complete | 이상치 제거 | 조건수(적용 전) | 조건수(적용 후) | α |
    |---|---|---|---|---|---|
    | Address | 336 | 24 | 8.57 | 8.57 | 0.00 |
    | Toe-up | 281 | 31 | 35.51 | 35.51 | 0.00 |
    | Mid-backswing | 120 | 12 | 132.72 | 20.58 | 0.10 |
    | Top | 126 | 11 | 201.87 | 27.88 | 0.10 |
    | Mid-downswing | 137 | 11 | 300.56 | 20.73 | 0.15 |
    | Impact | 316 | 30 | 10.98 | 10.98 | 0.00 |
    | Mid-follow-through | 294 | 28 | 11.67 | 11.67 | 0.00 |
    | Finish | 127 | 4 | 717.14 | 28.06 | 0.10 |

    (트리밍 전 초기 조건수는 훨씬 컸다 — 예: Address 53.08, Finish 540.64. 이상치 제거만으로도 대부분 이벤트의 조건수가 크게 개선됐고, 그 위에 shrinkage가 나머지를 안정화한다.)

- `pipeline/run_analysis.py`
  - `mahalanobis_combined_score()` 신규: 5-feature 벡터(circular_delta 적용) → precision 행렬로 마할라노비스 거리 계산 → `scipy.stats.chi2.cdf(df=5)`로 `deviation_percentile` 환산 → `grade_from_combined_percentile()`로 등급화(68.27%/95.45% 경계 — 기존 `grade_from_z`의 |z|<=1/|z|<=2가 의미하는 1σ/2σ 양측 커버리지와 동일한 기준을 다변량으로 확장한 것).
  - 이벤트 dict에 `combined_score` 필드로 신규 추가. **기존 `feedback.<feature>.z_score/grade/pro_mean/pro_std/value`는 전혀 변경하지 않음** — 추가 전용(additive-only).
  - `--covariance-stats` CLI 옵션 신규(기본값 `data/pro_feature_covariance.json`), 파일이 없으면 `combined_score`가 `"unavailable"`로 graceful degrade (기존 reliability_stats 로딩 패턴과 동일).

### 1.5. 구현 직후 자체 재검토로 발견/수정한 버그 (같은 날)

구현 완료 후 `/code-review`로 다시 점검하다가, **공분산이 이상치에 심하게 오염되어 있다는 심각한 문제**를 발견해 바로 수정했다.

- **문제**: `compute_covariance_stats`가 일반 `np.cov`로 공분산을 계산했는데, 342명 데이터에는 이미 클러스터링 검증 중 확인한 pose 오검출 이상치(video 160, 440 등)가 섞여 있음. 342명을 자기 자신의 통계로 재채점(self-consistency check)했을 때 이론상 median percentile ≈ 50%, Excellent 비율 ≈ 68.3%가 나와야 하는데, 실제로는 **median 7.6~27.2%, Excellent 83~93%**로 나옴 — 이상치가 공분산을 부풀려 대다수 사람의 거리가 인위적으로 짜부라짐. 결합 스코어가 의도(조합 이상 탐지)와 반대로 너무 관대하게 작동하고 있었음.
- **부가 문제**: 공분산은 "complete-case 표본 자체의 평균"으로 암묵적 중심을 잡는데(`np.cov` 내부 동작), Mahalanobis 계산은 "전체 모집단 평균"으로 중심을 잡아 — 결측이 특정 패턴을 띠는 Finish 이벤트에서 두 평균이 최대 57.7도까지 벌어짐.
- **수정**:
  - 공분산 계산 전 모든 행을 **모집단 평균으로 명시적으로 센터링**(`center_on_population_mean`) — 공분산과 Mahalanobis 계산이 같은 기준점을 쓰도록 통일 (평균 불일치 문제 해결).
  - **이상치 트리밍 2회 반복** 추가(`trim_outliers`): 공분산을 1차 적합 → 그 공분산 기준 마할라노비스 거리가 카이제곱(5) 97.5백분위수를 넘는 행 제거 → 재적합, 총 2회. shrinkage는 트리밍 이후 단계에서 그대로 적용.
  - `data/pro_feature_covariance.json`에 `n_outliers_removed` 필드 신규 추가(투명성 확보).
  - `combined_score.percentile` → `combined_score.deviation_percentile`로 필드명 변경(값이 클수록 "더 이상치"라는 뜻인데 "percentile이 높다=더 좋다"로 오해하기 쉬워 명확화, 이번 주 신규 필드라 하위호환 영향 없음).
- **수정 후 재검증**: 8개 이벤트 median이 26.7~58.4%(수정 전 7.6~27.2%), Excellent 비율 59.8~80.3%(수정 전 83~93%, 이론값 68.3%)로 이론적 캘리브레이션에 크게 근접. 회귀 확인(기존 z_score/grade 불변)도 재실행해 재확인함.

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
