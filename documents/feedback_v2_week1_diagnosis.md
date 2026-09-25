# v2 1주차 데이터 진단 결과 및 결정 게이트

작성일: 2026-09-25
근거 스크립트: `pipeline/diagnose_reliability_stats.py`
원본 출력: `outputs/reliability_diagnosis_summary.txt` (읽기 전용, git 미추적 `outputs/` 하위)
입력 데이터: `data/pro_feature_raw.csv`(342명 x 8이벤트), `data/pro_feature_stats_reliability.json` — 변경 없음

## 1. 결측 패턴 (event x feature)

전체적으로 `shoulder_tilt`/`hip_tilt`/`spine_tilt`는 대부분 이벤트에서 detection_rate 0.98 이상으로 안정적. 문제는 `trail_elbow_angle`과 `lead_elbow_angle`에 집중됨:

| 이벤트 | trail_elbow_angle detection_rate | 비고 |
|---|---|---|
| Mid-backswing | 0.374 | 결측 214/342 |
| Top | 0.380 | 결측 212/342 |
| Mid-downswing | 0.433 | 결측 194/342 |
| Finish | 0.784 | 결측 74/342 |
| Finish (lead_elbow_angle) | 0.444 | 결측 190/342 |

즉 결측이 완전 무작위(MCAR)가 아니라 **특정 이벤트(백스윙~다운스윙 중간 구간)의 trail_elbow_angle에 집중**되어 있음. 이는 아래 3번(클러스터링) 결과에 직접적인 영향을 준다.

## 2. 이벤트별 공분산 조건수 (Mahalanobis 적용 가능성)

| 이벤트 | n_complete (5-feature complete-case) | 조건수 |
|---|---|---|
| Address | 336 | 53.08 |
| Toe-up | 281 | 26.34 |
| Mid-backswing | 120 | 281.35 |
| Top | 126 | 32.04 |
| Mid-downswing | 137 | 310.79 |
| Impact | 316 | 22.27 |
| Mid-follow-through | 294 | 47.27 |
| Finish | 127 | 540.64 |

**판단**: 모든 조건수가 유한하고 극단적 병리 수준(1e6+)은 아니라 Mahalanobis 자체는 8개 이벤트 전부에서 계산 가능. 다만 `Mid-backswing`(281), `Mid-downswing`(311), `Finish`(541)는 조건수가 상대적으로 크고 `n_complete`도 120~137로 작아(1번의 결측 집중이 원인) 역행렬이 불안정할 수 있음 → **이 3개 이벤트만 Ledoit-Wolf shrinkage 또는 정규화를 적용**, 나머지 5개 이벤트(22~53)는 표본 공분산을 그대로 사용해도 무방.

## 3. 클러스터 분리도 (실루엣 스코어)

8이벤트 x 5feature = 40차원 벡터를 **모든 값이 존재하는 video만** 사용해 구성한 결과, 342명 중 **완전한 케이스는 13명뿐**(1번 결측 패턴이 전 이벤트에 걸쳐 누적된 결과).

| k | n_videos | silhouette |
|---|---|---|
| 2 | 13 | 0.1154 |
| 3 | 13 | 0.0866 |
| 4 | 13 | 0.1195 |
| 5 | 13 | 0.1400 |
| 6 | 13 | 0.1102 |

**판단**: 두 가지 문제가 겹쳐 있음.
1. 표본 수(13명)가 342명 중 3.8%에 불과해 어떤 k를 골라도 통계적으로 신뢰하기 어려움.
2. 표본 수를 무시하고 봐도 실루엣 스코어가 전 구간 0.09~0.14로, 통상 "약한 구조"로도 보기 어려운 수준(참고: 0.5 이상이 뚜렷한 분리, 0.25 미만은 사실상 무구조로 간주하는 것이 일반적인 경험적 기준).

원인은 1번과 동일 — `trail_elbow_angle`이 특정 이벤트에서 결측률이 높아, "8이벤트 전부 결측 없음"을 요구하는 현재 complete-case 방식이 표본을 거의 다 걸러낸다.

## 결정 게이트

v2 계획서(67행) 기준: "클러스터 분리가 뚜렷하지 않으면 3번(클러스터링)을 스킵하고 1, 2번만 진행".

- **1번(Mahalanobis)**: 예정대로 2주차 진행. 단 Mid-backswing/Mid-downswing/Finish 3개 이벤트에는 shrinkage 적용을 필수 작업 항목으로 추가.
- **2번(이벤트 간 delta 다이내믹스)**: 이번 진단과 무관하게 예정대로 3주차 진행 (delta는 이벤트 쌍별로 계산되므로 40차원 전체 complete-case가 아니라 이벤트 쌍 단위 결측 처리로 충분 — 1번 결측 패턴표 참고해 쌍별 n을 그때 재확인).
- **3번(프로 유형 클러스터링)**: **현재 방식(40차원 전체 complete-case)으로는 스킵.** 실루엣 스코어가 낮고 표본이 13명뿐이라 뚜렷한 근거가 없음. 다만 원인이 "클러스터 구조 자체의 부재"가 아니라 "`trail_elbow_angle` 결측이 만든 complete-case 표본 축소"일 가능성이 커서, 2주차 착수 전에 아래 대안 중 하나로 재시도 여부를 재확인한다:
  - (a) `trail_elbow_angle`을 클러스터링 feature에서 제외하고 나머지 4개 feature x 8이벤트(32차원)로 재계산 — 결측이 덜한 조합이라 표본이 크게 늘어날 것으로 예상.
  - (b) 이벤트별 부분 벡터(예: Address/Impact처럼 detection_rate가 높은 이벤트만) 기준으로 클러스터링 범위 축소.
  - 위 재시도도 실루엣이 낮으면 3번은 이번 4주 범위에서 완전히 제외하고 1, 2, 4번만 진행.

## 2주차 재진단 (2026-09-25) — 대안 (a) 결과, 이후 이상치 검증으로 정정됨

1차 재진단: `pipeline/diagnose_reliability_stats.py --exclude-features trail_elbow_angle` 실행 결과 표본이 13명 → 110명으로 늘고 k=2에서 실루엣 0.4574가 나와, 처음에는 "뚜렷한 2개 유형 구조"로 판단하고 `pipeline/cluster_pro_features.py`로 실제 클러스터 배정까지 실행했다.

**그런데 실행 결과 k=2 분할이 2명(video_id 160, 440) vs 108명으로 극단적으로 불균형**하게 나와 이상치 여부를 검증했다:

- 110명의 32차원 표준화 벡터에서 "중심으로부터의 거리"를 계산하니 중앙값 4.26 대비 상위 10명(565, 19, 440, 160, 442, 657, 845, 20, 257, 789)이 8.5~14.3으로 2~3배 떨어져 있음.
- video 160/440의 원본 값을 확인하니 `Finish`/`Impact` 등 일부 이벤트에서 `shoulder_tilt`/`hip_tilt`/`spine_tilt`가 다른 341명과 확연히 다른 범위(예: 160의 Finish `shoulder_tilt=71.5`, 대다수는 -150~180 범위) — 스윙 스타일 차이라기보다 해당 프레임에서의 pose 오검출(각도 계산 이상치)로 보임.
- 이상치를 거리 임계값 7/8/9로 각각 제거하고(93/97/101명 남음) k=2~4로 재클러스터링한 결과 **실루엣이 0.06~0.11로 전 구간 붕괴** — 이상치를 빼면 뚜렷한 군집 구조가 없다는 뜻.

**최종 결론**: 1차 실루엣 0.4574는 "두 스윙 유형"이 아니라 **이상치 2명을 분리한 결과**였다. 이상치를 제외한 나머지 표본은 원래 40차원 진단(0.09~0.14)과 마찬가지로 유의미한 클러스터 구조가 없음. v2 계획서 결정 게이트("분리가 뚜렷하지 않으면 3번 스킵") 기준으로 **3번(프로 유형 클러스터링)은 이번 4주 범위에서 완전히 스킵**한다. 검증 과정에서 만들었던 `pipeline/cluster_pro_features.py`와 그 결과 파일(`data/pro_cluster_assignments.json`, `data/pro_feature_stats_by_cluster.json`)은 스킵 결정에 따라 삭제했다(사용되지 않는 코드를 남기지 않기 위함).

## 다음 확인 사항

- ~~3번(클러스터링) 재시도 방향~~ → 이상치 검증까지 마치고 **완전 스킵**으로 최종 확정. 스켈레톤 비교는 기존 전역 medoid(`data/pro_reference_skeletons.json`)를 그대로 유지.
- Mid-backswing/Mid-downswing/Finish의 shrinkage 적용 방식(Ledoit-Wolf vs 단순 정규화)은 2주차 구현 시 결정.
- (백로그) pose 오검출로 보이는 이상치 프레임(예: video 160/440의 특정 이벤트)은 `extract_pro_features.py` 재추출 시 QC 규칙 추가를 검토할 만함 — 이번 4주 범위 밖.
