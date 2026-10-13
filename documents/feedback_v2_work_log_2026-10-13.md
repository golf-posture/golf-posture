# 2026-10-13 작업 기록 — 프로 유형 분류 재시도, 팔꿈치 결측 해결, 사용자 추출 추적 방식 전환

브랜치 `feature/feedback-v2-week2`. 이 문서는 하루 동안 한 일을 순서대로 기록하고, 결과·결정·재현 방법·남은 일을 정리한다. 클러스터링과 kNN의 상세는 `documents/feedback_v2_clustering_retry.md`에 있다.

## 0. 한눈에 보기

| 질문 | 결과 |
|---|---|
| 클러스터링으로 프로 유형을 분류할 수 있나? | **아니오.** 결측·이상치를 해결한 데이터에서도 48개 설정 중 0개 통과 (최고 실루엣 0.188, 기준 0.25) |
| 클러스터 없이 "나와 비슷한 프로"와 비교(kNN)하면 나은가? | **이득이 작아 구현하지 않음.** 전역 대비 최대 +2.0% (사전 기준 3%) |
| 팔꿈치 결측은 해결할 수 있나? | **예.** 영상 모드 추적으로 trail 결측 27.8% → 3.1%, 40차원 complete-case 13명 → 246명 |
| 사용자 영상 추출은? | **추적 방식으로 전환** (`--extraction tracked`, 기본). **채점 기준은 기존 통계를 그대로 유지** (`--reference legacy`, 기본). QC 정제된 추적 기준(`data/tracked/`)은 선택 옵션(`--reference tracked`)으로만 보존 |
| 부수적으로 발견한 것 | 기존 기준 통계가 이상치 때문에 관대함(자기 \|z\|≤1 비율 85.9% vs 기대 68.3%). 추적 기준은 69.5%로 보정되지만 표준편차가 크게 좁아져(중앙값 0.67배) 등급이 엄격해지므로 **기본값으로 바꾸지 않음**. 영상이 독립 표본이 아님(고유 youtube_id 160개) |

테스트: 작업 시작 시 43개 → 현재 **129개 모두 통과** (실험 모듈 정리 후; 정리 전 154개).

## 1. 출발점

사용자 질문: "클러스터링으로 프로 유형을 분류하는 것에 실패했었는데 다시 시도해볼 수 있을까?"

이전 실패 기록(`feedback_v2_week1_diagnosis.md`, `feedback_v2_week2_notes.md`):
- 40차원 complete-case 클러스터링은 `trail_elbow_angle` 결측 때문에 표본이 13명으로 붕괴, 실루엣 0.09~0.14.
- trail 팔꿈치를 제외한 32차원은 k=2 실루엣 0.457로 유망해 보였으나, 실제로는 pose 오검출 이상치 2명(video 160, 440)을 분리한 결과였고 제거 후 0.06~0.11로 붕괴.
- 결론: "클러스터 구조 없음 + 이상치 오염 + 결측".

이번 접근: 결측과 이상치를 **먼저** 해결한 뒤, 같은 질문을 **결과를 보기 전에 고정한 기준**으로 다시 검증한다.

## 2. 진행 순서

### 2.1 결측 원인 분석

- 결측은 포즈 검출 실패가 아니었다 (`pose_not_detected`는 2736행 중 14건). 팔꿈치 각도(어깨-팔꿈치-손목 3점)가 MediaPipe `visibility < 0.5` 컷오프(`run_analysis.py`)로 버려진 것이었다.
- `MediaPipePoseBackend`가 `static_image_mode=True`라 이벤트 프레임 8장을 서로 무관한 이미지처럼 추론했다. 앞뒤 프레임의 맥락이 없었다.
- 결측은 trail 팔꿈치의 Mid-backswing/Top/Mid-downswing에 집중. 원인은 (1) 실제 가림과 (2) 보이는데 모델이 확신하지 못함이 섞여 있었다 (v196 오버레이: 팔이 뚜렷이 보이는데 기존 설정에서 결측).

### 2.2 실험 1 — 임계값과 모델 (50개 영상, 실험 스크립트는 정리되어 삭제됨)

| 조건 | trail 결측 | lead 결측 | 새로 채운 값의 모델 간 불일치 (새 값/기존 값) |
|---|---|---|---|
| 기존 (complexity 1, vis 0.5) | 22.8% | 10.5% | — |
| vis 0.3 | 11.3% | 5.2% | 0.18 / 0.09 (약 2배) |
| vis 0.2 | 7.2% | 3.5% | 0.24 / 0.09 |
| complexity 2 (heavy), vis 0.5 | 42.2% | 29.5% | — |
| complexity 2, vis 0.3 | 30.7% | 18.5% | 0.28 / 0.09 |

결론: 임계값만 낮추면 결측은 주나 노이즈가 늘고, heavy 모델은 visibility를 더 보수적으로 보고해 오히려 나빠진다 → 폐기.

### 2.3 실험 2 — 영상 모드 추적 + 시간축 복구 + 2D/3D (50개 영상, 실험 스크립트는 정리되어 삭제됨; 추적·각도·복구 로직은 `pose_tracking.py`로 이전)

| 조건 | trail 결측 | lead 결측 | 40차원 complete | 32차원(trail 제외) complete |
|---|---|---|---|---|
| 기존 (정지 프레임, 2D, vis 0.5) | 22.8% | 10.5% | 2 | 18 |
| 영상 모드, 이벤트 프레임만 (vis 0.5) | 10.5% | 6.2% | 14 | 25 |
| + 시간축 복구 (vis 0.5) | 7.2% | 5.2% | 18 | 29 |
| **+ 시간축 복구 (vis 0.3)** | **1.5%** | **2.0%** | **36** | **41** |

- **개선의 거의 전부는 영상 모드 추적 자체에서 나왔다.** 시간축 복구(±2프레임 중앙값, ±6프레임 선형보간)로 채운 값은 13개(vis 0.5)/5개(vis 0.3)뿐이다.
- 새로 채워진 값 36장을 접촉 시트로 육안 채점: 그럴듯함 약 24, 명백히 틀림 약 5, 판단 불가 약 7. 틀린 사례는 주로 얼굴 클로즈업/팔이 프레임 밖.
- 2D와 3D(world landmark) 각도: 중앙값 15.3° 차이, 상위 10%는 35.9° 이상, 14.7%는 30° 초과 어긋남, 상관 0.76. 2D 투영의 foreshortening 왜곡 가설과 일치하나, 3D가 더 정확한지는 정답이 없어 확인하지 못했다.

### 2.4 342개 전체 재추출 (`pipeline/extract_pro_features_tracked.py`)

영상 모드 + vis 0.3 + 시간축 복구, **2D와 3D 팔꿈치 각도를 모두 저장**, 값의 출처(`direct|window|interp|missing`)와 QC 재료(팔 3관절 최소 visibility, 2D/3D 상완·전완 길이, 프레임 안 여부)를 함께 저장. 출력 `data/pro_feature_raw_tracked.csv` (2736행, 실패 영상 0).

| 항목 | 기존 | 추적 방식 |
|---|---|---|
| trail 팔꿈치 결측 | 760/2736 (27.8%) | 85 (3.1%) |
| lead 팔꿈치 결측 | 322 (11.8%) | 47 (1.7%) |
| 40차원 complete-case | 13명 | 246명 |
| 32차원(trail 제외) complete | 110명 | 293명 |
| 24차원(tilt만) complete | 329명 | 338명 |
| pose 검출 실패 | 14행 | 4행 |

이벤트별 trail 결측: Mid-backswing 62.6% → 7.6%, Top 62.0% → 11.4%, Mid-downswing 56.7% → 3.5%, Finish 21.6% → 1.2%.

### 2.5 QC 규칙 설계 (`pipeline/qc_tracked_features.py`)

결과(클러스터링)를 보기 **전에** 분포와 시각 확인만으로 고정했다. 플래그는 행을 지우지 않고 "결측으로 취급할 값"을 표시한다.

| 규칙 | 조건 | 걸린 비율 (유효 행 기준) |
|---|---|---|
| `low_vis` | 팔 3관절 최소 visibility < 0.4 | trail 5.8%, lead 2.2% |
| `bone` | 3D 전완/상완 비 ∉ [0.5, 2.0] 또는 3D 뼈 < 0.1 | trail 4.4%, lead 2.0% |
| `frame` | 관절이 프레임 밖 | 각 3건 |
| `inconsistent` | \|2D−3D 각도\| > 60° | trail 3.7%, lead 3.0% |
| `outlier_<col>` | 이벤트별 robust z(median/MAD, tilt는 원형) > 4 | 아래 |

- Finish의 `shoulder_tilt`/`hip_tilt`는 이상치 규칙에서 **제외**: 분포가 이봉(약 64% ±180° 근처, 36% 0° 근처)이라 정상 변이를 이상치로 오판했다 (제외 전 shoulder 189건, hip 131건 → 제외 후 65건, 49건).
- `inconsistent` 플래그의 약 2/3(150개 중 102개)가 Finish에 집중 (팔이 머리 뒤로 접힘).
- 시각 확인 (범주별 12장, 육안): **플래그 없음 12/12 그럴듯함**. `low_vis`만 걸린 12장 중 약 6 그럴듯/1 틀림/5 의심, `bone`만은 절반 안팎이 틀리거나 의심, `inconsistent`만은 대부분 Finish. 규칙은 의도대로 구별하나 표본이 작고 주관적이다. `inconsistent`는 2D가 맞고 3D가 틀린 경우도 버리는 보수적 규칙.

### 2.6 클러스터링 진단 (`pipeline/diagnose_clustering.py`) — 통과 0/48

설계(실행 전 고정): 특징 세트 3종(`elbow2d`, `elbow3d` = 팔꿈치 이벤트 0~6 + tilt 36차원, `tilt_only` 22차원) × 표본 2종(`normal` = slow==0만, `all`) × 알고리즘 2종(k-means 10 init, Ward) × k=2~5. 전처리: 원형 tilt 선형화 → z-score → 분산 80% PCA(최대 10). scikit-learn은 추가하지 않고 scipy/numpy만 사용.

통과 기준(모두 충족): 실루엣 ≥ 0.25, `youtube_id` 그룹 부트스트랩 평균 ARI ≥ 0.6, 최소 클러스터 ≥ 15, 실루엣 > 열 셔플 널 95퍼센타일.

| 세트 | n (독립 그룹) | 최고 실루엣 | 최고 ARI | 통과 |
|---|---|---|---|---|
| elbow2d, normal | 69 (64) | 0.150 | 0.667 | 0 |
| elbow2d, all | 112 (78) | 0.161 | 0.633 | 0 |
| elbow3d, normal | 73 (67) | 0.158 | 0.635 | 0 |
| elbow3d, all | 124 (84) | 0.160 | 0.712 | 0 |
| tilt_only, normal | 149 (132) | 0.187 | 0.767 | 0 |
| tilt_only, all | 290 (144) | 0.188 | 0.836 | 0 |

- k=2에서 ARI가 높은(최대 0.84) 설정이 있으나 실루엣이 낮은 채라, 이산 유형이 아니라 연속 분포를 가장 큰 분산 방향으로 안정적으로 이등분한 결과로 보인다 (추정, 직접 검증하지 않음).
- 널(열별 셔플)은 특징 간 상관까지 파괴하는 약한 검정이라 통과는 구조의 증거가 아니다.
- **결론: 이산 프로 유형은 이 데이터로 지지되지 않는다. v2 항목 3(클러스터별 medoid/통계)은 구현하지 않는다.**

### 2.7 연속 개인화(kNN) 검증 (실험 스크립트는 정리되어 삭제됨)

"나와 가장 가까운 프로 K명"을 기준으로 채점하면 전역 기준보다 나은지 시험. 기존 정지 프레임 데이터(`data/pro_feature_raw.csv`, 사용자 파이프라인과 같은 추출)로 `youtube_id` 그룹 단위 leave-one-out. **이웃은 Address 특징으로만 선택하고 이후 이벤트(1~7)를 채점**해 "가까운 사람으로 뽑아서 정상으로 보이는" 순환을 피했다. 무작위 K 대조군 포함.

| K | 전역 대비 개선 [95% CI] | 무작위 K 대비 | \|z\|≤1 커버리지 | 판정 |
|---|---|---|---|---|
| 15 | −0.5% [−2.0, +1.0] | +3.5% (CI>0) | 0.579 | 미달 |
| 30 | +1.2% [+0.2, +2.1] | +2.9% (CI>0) | 0.619 | 미달 |
| 60 | +2.0% [+1.4, +2.8] | +2.7% (CI>0) | 0.639 | 미달 (기준 3%) |

유사도는 약한 정보를 담지만(무작위보다 유의하게 나음) 전역 대비 이득이 최대 2%라 복잡도를 정당화하지 못한다. **구현하지 않고 전역 기준과 전역 medoid를 유지한다.** 한계: Address 5개 특징만 사용, 체형 정보는 라벨에 없음.

### 2.8 사용자 추출을 추적 방식으로 전환

요청: "사용자 추출 추적방식으로 바꾸고 …"

**기준 통계 결정 경위 (정정 기록)**: 기준 통계(신뢰도/공분산/다이내믹스)는 모두 기존 정지 프레임 데이터 `data/pro_feature_raw.csv`로 만들어져 있다. 처음에는 "사용자 추출만 바꾸면 분포가 어긋난다"고 판단해 QC 정제된 추적 데이터로 기준 통계를 새로 만들고(`data/tracked/`) 추적 모드의 기본 기준으로 삼았다. **이는 요청 범위를 넘어선 결정이었고, 확인 없이 진행한 것이 잘못이었다.**

이후 비교로 그 근거가 약했음이 드러났다. 기준의 **평균은 거의 안 움직였고**(중앙값 0.9°) 달라진 것은 **표준편차**(40쌍 중 28쌍이 20% 넘게 감소, 중앙값 0.67배)인데, 이는 추적 방식 때문이 아니라 **QC로 이상치를 제거했기 때문**이다. 즉 추출 방식 전환과 기준 재생성은 서로 독립적인 결정이었다. 그래서 최종 결정은 다음과 같다.
- **채점 기준은 기존 `data/*.json`을 기본으로 유지** (`--reference legacy`, 기본값). 기준 선택은 추출 방식과 독립된 별도 옵션이다.
- QC 정제된 추적 기준(`data/tracked/`)은 삭제하지 않고 `--reference tracked` 선택 옵션으로 보존한다. 더 엄격한 채점이 사용자에게 더 유용한지 검증되지 않았기 때문에 기본값으로 쓰지 않는다.
- 사용자 팔꿈치 QC(의심스러운 값은 `unavailable`)는 기준이 아니라 사용자 값에 대한 안전장치이므로 유지한다.

변경 내용:
- `pipeline/pose_tracking.py` (신규): 추적·각도·시간축 복구 로직을 실험 모듈에서 정식 모듈로 승격. 13개 랜드마크 전부 저장(스켈레톤 그리기용). `start_frame` 지원(긴 영상에서 처음부터 추적하지 않음).
- `pipeline/user_tracking.py` (신규): 사용자 영상의 추적 결과 → 이벤트별 특징. 팔꿈치는 2D 각도 + 프로 기준과 같은 행 단위 QC(`low_vis/bone/frame/inconsistent`)를 적용하고, 걸리면 `unavailable`(잘못된 포즈가 잘못된 "개선 필요"가 되지 않도록). tilt는 vis ≥ 0.5, 랜드마크는 vis ≥ 0.5. 첫 이벤트 10프레임 전부터 마지막 이벤트 + 8프레임까지만 추적.
- `pipeline/run_analysis.py`: `--extraction {tracked,static}` 추가 (기본 `tracked`), 별도 옵션 `--reference {legacy,tracked}` 추가 (기본 `legacy` = 기존 `data/*.json`, `resolve_stats_path`). 두 옵션은 서로 독립이다. `tracked` 추출은 mediapipe 백엔드에서만 동작하고 `yolopose/auto/hybrid`는 `static`으로 자동 전환 (`effective_extraction`, 결과 JSON의 `extraction`/`extraction_note`/`reference`에 기록). 추적 실패는 조용히 되돌리지 않고 `errors`에 기록 후 종료. `MediaPipePoseBackend`에 `model_complexity`, `min_visibility` 옵션 추가 (기본값은 기존과 동일).
- 시간축 복구 값(window/interp)은 이벤트 프레임의 `min_vis`가 낮거나 포즈가 없어 QC에 항상 걸려 버려진다. 프로 기준 통계에서도 똑같이 버려졌으므로 **일관된 동작**이며, 개선은 추적 자체에서 나온다는 2.3의 결론과 같다.

## 3. 검증

### 3.1 단위 테스트
129개 통과 (실험 모듈 정리 후). 신규: 추적/각도/복구(`test_pose_tracking`), 재추출 헬퍼, 추적 추출 행 생성, QC 규칙, 클러스터링 수치 함수(ARI, silhouette, PCA, k-means, 원형 선형화, 그룹 부트스트랩), kNN 헬퍼, 사용자 추적, 추출 방식 분기, 기준 통계 입력.

### 3.2 베이스라인 6개 영상 end-to-end (`--pose-backend mediapipe`)

#### 최종 기본 설정: 추적 추출 + 기존 기준 (`--extraction tracked --reference legacy`)
기본값으로 6개 영상을 다시 실행해 확인했다. 결과 JSON에 `extraction: tracked`, `reference: legacy`가 기록되고 사용 통계는 `data/pro_feature_stats_reliability.json`, `data/pro_feature_covariance.json`, `data/pro_dynamics_stats.json`(기존 파일)이다. 6개 영상 모두 `errors: 0`.

정지 방식 결과와 비교한 등급 차이 (6영상 × 8이벤트 × 5특징 = 240항목, **기준은 동일**):

| 구분 | 항목 수 |
|---|---|
| 등급이 달라진 항목 | **37 (15%)** |
| 　값이 생기거나 사라져서 (QC로 팔꿈치가 `unavailable`이 되거나, 정지 방식이 놓친 값이 채워짐) | 19 |
| 　값 자체가 움직여서 | 18 |
| 　방향: 더 관대 / 더 엄격 | 21 / 16 (한쪽으로 치우치지 않음) |

기준을 그대로 둔 덕분에, 이 37개는 순수하게 "추출 방식과 QC의 효과"이다.

#### (참고) 첫 시도: 기준까지 `tracked`로 바꾼 경우 (`--reference tracked`) — 기본값으로 채택하지 않음
12번 실행(정지/추적 × 6영상) 모두 `errors: 0`, 8/8 이벤트 pose 검출. SwingNet 이벤트 프레임은 두 모드가 동일.

| 영상 | static (Exc/Good/NeedsImp/unavail) | tracked (Exc/Good/NeedsImp/unavail) |
|---|---|---|
| longform | 28/9/0/3 | 25/10/3/2 |
| shorts | 31/5/3/1 | 20/9/8/3 |
| shorts2 | 35/3/0/2 | 32/7/0/1 |
| test1 | 24/7/5/4 | 20/9/5/6 |
| test2 | 30/8/0/2 | 27/10/3/0 |
| test3 | 33/5/0/2 | 20/15/4/1 |

- 특징 가용성은 거의 같다 (226/240 vs 227/240): **사용자 영상은 프레임이 안정적이라 정지 방식에서도 팔꿈치가 잘 잡혀, 추적의 이득은 사용자 쪽에서는 작다.** 추적 전환의 사용자 쪽 의미는 가려진 팔의 안정성과 QC 안전장치 정도이다.
- 같은 영상·이벤트에서 두 모드 특징 값의 중앙값 차이: trail 팔꿈치 3.8°, lead 3.3°, shoulder 2.6°, hip 1.4°, spine 0.6°. 즉 **같은 영상의 측정값은 두 방식이 비슷하다.**
- 기준까지 `tracked`로 바꾸면 등급이 더 엄격해졌다 ("Needs Improvement" 증가). 원인을 값 변화와 기준 변화로 분해했다 (같은 항목을 (값, 기준) 4가지 조합으로 채점):

| 구분 | 항목 수 (240 중) |
|---|---|
| 정지 대비 등급이 달라진 항목 | 82 (34%) |
| 　기준 변화만으로 바뀜 / 값 변화만으로 바뀜 / 둘 다 단독으로 바뀜 / 함께 작용해야 바뀜 | 35 / 25 / 7 / 15 |
| 　방향: 더 엄격 / 더 관대 | 65 / 17 |
| 각 요인을 단독 적용했을 때 등급이 바뀌는 항목 (값 / 기준) | 37 / 62 |

→ **기준 변화가 더 큰 몫**이고 (표준편차가 좁아져 같은 값이 더 엄격해짐), 값 변화도 작지 않으며 둘이 맞물린다 (기준이 좁은 곳에서는 몇 도 차이가 등급을 바꿈). 이 때문에 기준은 기본값으로 바꾸지 않았다. 원인 확인:

| 기준 통계 | 자기 데이터에 적용 시 \|z\|≤1 | \|z\|≤2 | 신뢰도 high/medium/low |
|---|---|---|---|
| 기존 (`data/`) | **85.9%** | 97.1% | 9 / 14 / 17 |
| 추적 (`data/tracked/`) | **69.5%** | 94.9% | 26 / 7 / 7 |
| 정규분포 기대값 | 68.3% | 95.4% | — |

기존 기준은 이상치가 표준편차를 부풀려 등급이 관대했고(Mahalanobis에서 확인한 현상과 같은 원인), 추적 기준은 등급 체계가 전제하는 보정에 맞는다. 따라서 엄격해진 것이 기존 관대함의 교정일 가능성이 있다. 다만 **단정할 수 없다**: 정답 데이터가 없어 "더 정확한 채점"인지 검증하지 못했고, tilt처럼 표준편차가 2~3°까지 좁아진 항목은 두 추출 방식 간 측정 차이(hip 1.4°, shoulder 2.6°)와 비슷한 크기라 **측정 노이즈 수준에 가까워 과하게 엄격할 위험**이 있다. 사용자에게 유용한 엄격함인지는 미검증이며, 이는 기본 기준(legacy 유지)과 분리된 별도 결정이다.

### 3.3 한국어 리포트
`generate_feedback_report`가 추적 모드 결과 JSON에서 정상 동작 (test2로 확인).

## 4. 새로 확인된 사실과 위험

1. **영상이 독립 표본이 아니다.** 342개 영상 = 고유 `youtube_id` 160개, 선수 107명. 145개 ID가 일반 속도(slow=0)와 슬로모션(slow=1)을 함께 갖는다 (같은 스윙 중복). 기존 Mahalanobis/다이내믹스/신뢰도 통계(`data/`, `data/tracked/` 모두)가 342를 독립 표본으로 가정한다 → 표준편차/공분산 추정이 과대평가일 수 있다. 영향 크기는 측정하지 않았다.
2. **Finish의 tilt는 이봉 분포** (shoulder_tilt 약 64% ±180° 근처 / 36% 0° 근처). 피니시 때 어깨선이 카메라 축과 정렬되어 2D 좌우 순서가 뒤집히는 기하학적 한계로 보인다 (성별과 무관, 같은 영상 내에서도 갈림; 직접 검증하지 않은 추정). 이 구간 tilt의 z-score는 신뢰도가 낮다.
3. **전원 오른손잡이로 가정**한다 (trail = 오른팔). 라벨에 손잡이 정보가 없어 왼손잡이가 섞여 있는지 모른다. 미검증.
4. 전부 face-on, driver 영상이라 시점/클럽 차이는 이번 결과와 무관.
5. 2D vs 3D 각도가 크게 어긋나는 행이 14.7%. 사용자 채점은 2D만 쓴다 (3D는 저장만, 미사용).

## 5. 변경 파일 목록

작업 전부터 있던 변경(내가 건드리지 않음): `.gitignore`, `pixi.toml`, `pixi.lock`, `documents/feedback_improvement_plan_v2.md`, `pipeline/generate_feedback_report.py`, `conftest.py`, `documents/feedback_v2_week4_notes.md`, `documents/v2_after_eval/`, `learning_video/`.

### 수정
| 파일 | 내용 |
|---|---|
| `pipeline/run_analysis.py` | `--extraction`(기본 tracked), `--reference`(기본 legacy), `effective_extraction`, `resolve_stats_path`, 추적 분기, 결과 JSON `extraction`/`extraction_note`/`reference`, `MediaPipePoseBackend` 옵션 인자 |
| `README.md` | "추출 방식 (`--extraction`)" 및 기준(`--reference`) 절 추가 |

### 신규 코드
| 파일 | 역할 |
|---|---|
| `pipeline/pose_tracking.py` | 영상 모드 추적, 각도, 시간축 복구, 공용 헬퍼 |
| `pipeline/user_tracking.py` | 사용자 영상 추적 → 이벤트별 특징 + QC |
| `pipeline/extract_pro_features_tracked.py` | 프로 342영상 추적 재추출 (2D/3D, 출처, QC 재료) |
| `pipeline/qc_tracked_features.py` | QC 플래그, `clean_tracked` |
| `pipeline/export_tracked_stats_input.py` | QC 정제 데이터 → 기존 통계 스크립트 입력 스키마 |
| `pipeline/diagnose_clustering.py` | 사전 고정 기준 클러스터링 진단 |

### 신규 테스트 (`tests/`)
`test_pose_tracking.py`, `test_extract_pro_features_tracked.py`, `test_qc_tracked_features.py`, `test_diagnose_clustering.py`, `test_user_tracking.py`, `test_run_analysis_extraction.py`, `test_tracked_stats_input.py`

### 신규 데이터
| 파일 | 내용 |
|---|---|
| `data/pro_feature_raw_tracked.csv` | 추적 재추출 원본 (2736행, 2D/3D 팔꿈치 + QC 재료) |
| `data/pro_feature_qc_tracked.csv` | QC 플래그 |
| `data/tracked/pro_feature_raw.csv` | QC 정제, 기존 스키마 (trail 2302 / lead 2476 / shoulder 2667 / hip 2683 / spine 2708 유효, 2736 중) |
| `data/tracked/pro_feature_stats_reliability.{json,csv}`, `pro_feature_covariance.json`, `pro_dynamics_stats.json` | 추적 기준 통계 (**선택 옵션** `--reference tracked`용으로 보존, 기본값 아님) |

기존 `data/pro_feature_raw.csv`, `data/pro_feature_stats_reliability.json`, `data/pro_feature_covariance.json`, `data/pro_dynamics_stats.json`, `data/pro_reference_skeletons.json`은 변경하지 않았다.

### 문서
`documents/feedback_v2_clustering_retry.md` (클러스터링·kNN 상세), 이 문서.

### 산출물 (git 미추적 `outputs/`)
`reextract_experiment/`, `temporal_experiment/`, `qc_review/` (+ 일회성 `outputs/qc_review.py`), `clustering_diagnosis/`, `knn_experiment/`, `extraction_compare/{static,tracked}/<영상>/`, `tracked/`

## 6. 사용법과 재현

### 사용자 영상 분석
```bash
# 기본 = 추적 추출(--extraction tracked) + 기존 기준(--reference legacy), mediapipe 백엔드
pixi run python -m pipeline.run_analysis --video raw_videos/test2.mp4 \
  --checkpoint event_detection/models/swingnet_1800.pth.tar --output outputs/test2

# 기존 방식 재현 (정지 프레임 추출; 기준은 기본값 legacy라 그대로)
pixi run python -m pipeline.run_analysis --video raw_videos/test2.mp4 \
  --checkpoint event_detection/models/swingnet_1800.pth.tar --extraction static

# (선택) QC 정제된 추적 기준 통계 사용 — 표준편차가 좁아져 등급이 더 엄격해짐
pixi run python -m pipeline.run_analysis --video raw_videos/test2.mp4 \
  --checkpoint event_detection/models/swingnet_1800.pth.tar --reference tracked
```

### tracked 기준 통계 생성
`learning_video/golfdb_720p_slice`(342개 원본 슬라이스)가 필요하다.
```bash
pixi run python -m pipeline.extract_pro_features_tracked      # → data/pro_feature_raw_tracked.csv (소요 시간은 측정하지 않음)
pixi run python -m pipeline.qc_tracked_features               # → data/pro_feature_qc_tracked.csv
pixi run python -m pipeline.export_tracked_stats_input        # → data/tracked/pro_feature_raw.csv
pixi run python -m pipeline.compute_reliability_stats \
  --raw-csv data/tracked/pro_feature_raw.csv \
  --output-json data/tracked/pro_feature_stats_reliability.json \
  --summary outputs/tracked/pro_feature_reliability_summary.txt \
  --stats-csv data/tracked/pro_feature_stats_reliability.csv \
  --covariance-json data/tracked/pro_feature_covariance.json
pixi run python -m pipeline.compute_dynamics_stats \
  --raw-csv data/tracked/pro_feature_raw.csv \
  --output-json data/tracked/pro_dynamics_stats.json \
  --summary outputs/tracked/pro_dynamics_summary.txt
```

### 진단·테스트
```bash
pixi run python -m pipeline.diagnose_clustering                          # 클러스터링 진단 (48 설정)
pixi run python -m pytest                                                # 129개
```

실험 1·2·kNN 스크립트(`experiment_*.py`)는 정리해 삭제했다. 결과와 설계는 이 문서 2.2·2.3·2.7절과 `feedback_v2_clustering_retry.md`에 남아 있고, 재사용되는 로직(추적, 각도, 시간축 복구)은 `pose_tracking.py`에 있다. 스크립트는 한 번도 커밋되지 않아 되살릴 수 없다.

## 7. 한계와 미해결

- 추적 모드의 사용자 영상 처리 시간은 측정하지 않았다 (6개 영상 12회 실행은 모두 완료).
- 3D 팔꿈치 각도는 저장만 하고 채점에는 쓰지 않는다. 2D/3D 중 무엇이 더 정확한지는 정답 데이터가 없어 판단하지 못했다.
- 참조 스켈레톤(`data/pro_reference_skeletons.json`)은 기존 정지 방식의 전역 medoid 그대로다 (재생성하지 않음).
- 2026-09-25 베이스라인 스냅샷은 정지 방식 + 기존 기준(`--pose-backend hybrid`)으로 만든 것이다. 기본 설정도 기준은 같지만 추출 방식이 달라 등급이 일부 다르다 (같은 6영상 기준 정지 대비 240항목 중 37개, 15%). 스냅샷과 엄밀히 비교하려면 `--extraction static`을 쓰거나 추적 모드로 새 스냅샷을 만든다. `v2_after_eval`은 열어보지 않아 조건을 확인하지 못했다.
- 기존 기준은 이상치 때문에 관대한 편이다 (자기 일관성 |z|≤1 85.9%, 기대 68.3%). 이를 교정할지, 좁은 tilt 표준편차에 측정 노이즈 기반 하한을 둘지는 이번 전환과 분리된 **별도 결정**이며 미검증이다.
- `hybrid`/`auto`/`yolopose`는 추적 모드가 적용되지 않는다 (YOLO 값이 QC 없이 섞여 기준과 어긋나기 때문).
- 시각 확인은 모두 육안, 표본 12~36장 수준이다.
- 통계는 342를 독립 표본으로 가정한다 (4.1).

## 8. 결정이 필요한 항목

**결정 완료**: 기본 = 추적 추출 + 기존 기준 + 사용자 팔꿈치 QC 유지. `data/tracked/` 통계는 `--reference tracked` 선택 옵션으로 보존.

결정 사항 (2026-10-13):

- **실험 모듈 정리**: `experiment_reextract_pose.py`, `experiment_temporal_pose.py`, `experiment_knn_personalization.py`와 그 테스트를 삭제했다. 재사용 로직의 테스트는 `tests/test_pose_tracking.py`로 옮겼다.
- **슬로모션 중복 영상**: 기준 통계에 반영하지 않는다 (4.1의 위험은 인지한 채로 유지).
- **왼손잡이 처리**: 일단 하지 않는다 (전원 오른손잡이 가정 유지).
- **새 베이스라인 스냅샷**: `documents/baseline_eval/2026-10-13/` (추적 추출 + 기존 기준, `--pose-backend mediapipe`).
- **커밋 범위**: 채점 경로 코드·테스트·문서는 커밋함. `data/tracked/` 등은 미정.

남은 결정:

1. **커밋 범위(나머지)**: `data/tracked/`, 추적 재추출 CSV 2개, `diagnose_clustering.py`, `export_tracked_stats_input.py`, `feedback_v2_clustering_retry.md`. `data/` 신규 파일이 `.gitignore`에 없어 추적 대상이 된다. `--reference tracked`는 `data/tracked/`가 필요하다.
2. 추적 모드를 `hybrid` 등으로 확장할지.
3. 기존 기준의 관대함 교정 / tilt 표준편차 하한은 별도 결정.
