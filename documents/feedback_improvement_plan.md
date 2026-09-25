# 골프 스윙 피드백 신뢰도 개선 - 구현 계획

작성일: 2026-09-15
상태: 계획 확정 대기 (구현 착수 전, 사용자 확인 대기 중)
근거 문서: `documents/feedback_improvement_ideas.md`

## 범위

- 대안 **A(통계 비교 방식)**와 **B(피드백 문구 방식)**의 조합만 다룬다.
- 대안 **C(사용자 영상 프레임 노이즈 완화)는 이번 범위에서 완전히 제외**한다.
- 342명 프로 스냅샷(`data/pro_feature_raw.csv`)만 사용, 신규 데이터 확보나 LLM 등 외부 의존성 도입 없음.

## A와 B의 관계

같은 문제를 해결하는 두 개의 서로 다른 레이어이며, 대체 관계가 아니라 파이프라인상 순차적으로 이어지는 단계다.

| 구분 | A (통계 비교 방식) | B (피드백 문구 방식) |
|---|---|---|
| 해결하는 문제 | 문제 1: z-score 통계 자체가 부실함 (이상치 취약, 정규분포 가정) | 문제 2: 피드백 문구가 기계적/일반적 |
| 다루는 대상 | 숫자 계산 — 사용자 값이 프로 342명 분포에서 어디에 위치하는지 판정 | 문장 생성 — 계산된 숫자를 자연어로 표현 |
| 코드 위치 | `compute_reliability_stats.py:compute_feature_stats`, `run_analysis.py:grade_from_z`, `feedback_for_event` | `generate_feedback_report.py:sentence_for_core`, `sentence_for_reference` |
| 입력 → 출력 | 프로 342명 raw feature → median/MAD, percentile → 등급(Excellent/Good/Needs Improvement) | 등급 + percentile 숫자 → 한국어 문장 |

A가 먼저 계산되어 "이 값은 상위 20%다 / 등급은 Good이다"를 만들면, B가 그 결과를 받아 표현 방식을 결정한다.

## 코드 그라운딩 (확인 완료)

| 항목 | 위치 | 현재 동작 |
|---|---|---|
| 프로 통계 계산 | `pipeline/compute_reliability_stats.py:65-120` | `mean`/`std`/`median`/`iqr`를 이벤트×feature별로 이미 계산 중 (median은 이미 있음, MAD는 없음) |
| z-score/등급 판정 | `pipeline/run_analysis.py:487-495`, `:538-563` | `(value-mean)/std`, `abs(z)<=1/2` 임계값으로 등급 판정 |
| 원형 feature 처리 | `run_analysis.py:498-499`, `compute_reliability_stats.py:38-47` | shoulder_tilt/hip_tilt는 이미 circular delta로 별도 처리됨 (재사용 가능) |
| 문구 생성 | `pipeline/generate_feedback_report.py:177-202` | `sentence_for_core`/`sentence_for_reference`가 등급 2분기 + reliability 3분기 고정 템플릿만 사용 |
| 테스트 | `event_detection/test_*.py`만 존재 | pipeline/ 쪽 통계·피드백 로직에 대한 단위테스트 없음 — 신규 추가 필요 |

## 추천 조합

### A: A1 (Robust z-score) + A2 (백분위)

- **내부 등급 판정(A1)**: `grade_from_z` 계산식을 `(value - median) / (1.4826 * MAD)`로 교체. 기존 `|z|<=1.0 / <=2.0` 임계값 체계는 유지해 변경 범위를 최소화.
- **사용자 표시(A2)**: 정렬된 프로 원본값 배열 기준 백분위 순위. 분포 가정이 전혀 없어 가장 직관적이고 B3과 자연스럽게 결합.
- A3(IQR 펜스)는 3단계 등급화에 추가 임계값이 필요해 제외, A4(Winsorized z-score)는 이상치 강건성이 A1보다 약해 제외.
- **구현 주의점**: 백분위 계산에는 요약 통계만으론 부족하고 이벤트×feature별 정렬된 원본값 배열(약 342×8×5 ≈ 13,680 float)이 stats JSON에 추가로 필요. `np.searchsorted`로 계산 가능(scipy 신규 의존성 불필요).

### B: B2 (편차 크기별 문장 강도) + B3 (정직한 순위 문구) + B4 (템플릿 다양화)

- B3을 문장의 사실적 뼈대로 사용: "프로 342명 중 상위/하위 X%에 해당합니다".
- B2로 표현 강도(약간/뚜렷하게/크게)를 백분위 구간에 따라 자동 선택하고, 실제 각도 차이(도 단위)를 함께 명시.
- B4로 같은 등급·구간이라도 문장 변형을 2~3개 순환시켜 기계적 반복 인상 제거.
- **B1(도메인 지식 매핑 테이블)은 보류**: "골프 지식이 틀리면 신뢰도가 오히려 하락, 검수 부담 큼"이라는 리스크가 이 작업의 목적(신뢰도 개선)과 충돌. 골프 도메인 전문가 검수 프로세스가 갖춰지기 전까지 별도 단계로 분리.

## 확신 여부에 대한 판단

- **A(통계 방식)**: median/MAD가 mean/std보다 이상치에 강하다는 건 수학적으로 증명된 사실이나, 실제로 "더 나은 결과"가 되려면 `pro_feature_raw.csv`에 등급 판정을 왜곡할 만한 이상치가 실존해야 함 — 이건 아직 확인되지 않음.
- **B(문구 방식)**: 등급/수치 계산의 정확도를 바꾸는 게 아니라 표현 방식만 바꾸는 것이라, 코드로 "더 좋다"를 증명할 수 없음. 결국 사용자 체감의 문제이며 이 프로젝트에는 정답 라벨/평가 지표가 없어 정량적 비교가 불가능.
- 따라서 A의 실익을 코드 작성 전에 데이터로 먼저 확인하기 위해 **Phase 1 진단 단계**를 추가함.

## 구현 단계

### Phase 0 — 현재 방식 베이스라인 스냅샷 저장 (신규, Phase 1 진단보다 먼저 실행)

**목적**: A/B 변경 전 현재 파이프라인의 실제 출력(z-score, 등급, 피드백 문구)을 고정된 샘플 세트로 저장해, 이후 변경 전/후 출력을 diff로 비교할 수 있게 한다. 정답 라벨이 없어 "더 낫다"를 자동 판정할 수는 없으므로, 이 스냅샷은 사람이 눈으로 변화를 검토하기 위한 근거 자료 용도다.

**작업**:
1. 고정된 입력 세트 확정 — 가능하면 실제 사용자 영상 샘플 몇 개, 어려우면 `run_analysis.py --mock-events`로 생성한 목 이벤트를 사용해 현재 코드로 `analysis.json`(현재 `z_score`/`grade` 필드 포함) 생성.
2. 각 `analysis.json`에 대해 `generate_feedback_report.py --analysis-json ... --output-dir ...`를 실행해 `feedback_table.csv`/`feedback_ko.txt` 생성.
3. 위 산출물(`analysis.json`, `feedback_table.csv`, `feedback_ko.txt`)과 당시 사용한 `pro_feature_stats_reliability.json`을 `documents/baseline_eval/<날짜>/` 아래에 그대로 복사해 보관. 파일명에 입력 샘플 식별자 포함.
4. 어떤 샘플과 커밋 해시로 생성했는지 `documents/baseline_eval/<날짜>/README.md`에 1~2줄로 기록해 재현성 확보.

**완료 기준**: Phase 2~4(통계/등급/문구 계층) 변경이 끝난 뒤 동일한 샘플 세트로 파이프라인을 재실행해 이 베이스라인과 diff — 등급이 바뀐 항목, 문구가 바뀐 항목을 사람이 검토할 수 있는 상태가 되면 통과. 읽기 전용 작업이며 기존 코드/출력에는 영향 없음.

### Phase 1 — 진단 (Phase 0 다음, Phase 2보다 먼저 실행)

**목적**: A1(robust z-score)로 바꿀 실익이 실제로 있는지 코드 작성 전에 데이터로 확인.

**작업**:
1. 신규 스크립트 `pipeline/diagnose_reliability_stats.py` 작성 (기존 `compute_reliability_stats.py`의 `compute_feature_stats`, `circular_delta` 재사용).
2. `data/pro_feature_raw.csv`의 이벤트×feature(8×5=40개 조합)별로:
   - 기존 방식: mean/std 기반 outlier 여부 확인 (`|z|>3` 등)
   - 개선 방식: median/MAD 기반 modified z-score (`0.6745*(x-median)/MAD`, 임계값 `>3.5`)로 outlier 여부 확인
   - 두 기준의 outlier 탐지 개수 차이 비교
3. 342명 각각에 대해 현재 등급(mean/std z-score) vs robust 등급(median/MAD z-score)을 모두 계산해 등급이 바뀌는 샘플 수/비율을 40개 조합별로 집계.
4. 결과를 `outputs/reliability_diagnosis_summary.txt`로 출력 (조합별 outlier 개수, 등급 변경 비율, 가장 많이 바뀐 상위 5개 조합).

**판단 기준(Decision Gate)**:
- 등급 변경 비율이 유의미(예: 특정 조합에서 5% 이상) → A1 도입 근거 확보, Phase 2 그대로 진행.
- 거의 모든 조합에서 등급 변화가 미미(<1%) → A1을 "이론적 안전장치"로만 규정하고 진행할지, A2(백분위)만으로 단순화할지 재확인 필요.
- 읽기 전용 분석이며 기존 `pro_feature_stats_reliability.json`이나 파이프라인 동작에는 영향 없음.

### Phase 2 — 통계 계산 계층 (`compute_reliability_stats.py`)
- `compute_feature_stats`에 robust 통계 추가: MAD(`1.4826` 스케일 포함) 계산, 기존 median 재사용.
- 이벤트×feature별 정렬된 원본값 배열을 stats JSON에 추가 필드로 포함 (circular feature는 `circular_delta` 적용 후 정렬).
- `write_stats_csv`/`write_summary`는 요약 통계만 다루므로 영향 최소, robust 컬럼만 추가.

### Phase 3 — 등급 판정 계층 (`run_analysis.py`)
- `grade_from_z` → robust z 기반으로 계산식 교체, 함수명/임계값 체계는 유지.
- `feedback_for_event`에 백분위 순위 계산 추가 (`np.searchsorted` 기반), `item`에 `percentile_rank` 필드 신설.
- 기존 `z_score`, `pro_mean`, `pro_std` 필드는 하위 호환을 위해 유지, 표시용 우선순위는 percentile로 전환.

### Phase 4 — 문구 생성 계층 (`generate_feedback_report.py`)
- `sentence_for_core`/`sentence_for_reference`를 B2+B3+B4 로직으로 교체: percentile 구간 → 강도 수식어 매핑, 각도 차이 삽입, 템플릿 변형 순환 선택.
- `FEATURE_KO`/`EVENT_KO`는 그대로 재사용.

### Phase 5 — 테스트 (신규)
- pipeline 쪽에 pytest 단위테스트가 전혀 없으므로 신규 작성 필요 (AAA 패턴).
- MAD 계산, robust z 등급 경계값, percentile 계산(정상/이상치 포함 케이스), 문구 강도 매핑에 대한 단위테스트.
- 기존 파이프라인 end-to-end는 목 데이터(`--mock-events`)로 회귀 확인.

## 리스크

| 리스크 | 가능성 | 완화 방안 |
|---|---|---|
| stats JSON에 원본 배열 추가로 파일 크기/로딩 비용 증가 | 낮음 | 13k개 float 수준으로 무시 가능한 크기 |
| robust z 임계값(1.0/2.0)이 MAD 스케일에서 기존과 다른 등급 분포를 만들 수 있음 | 중간 | Phase 1 진단으로 사전 확인, 필요 시 Phase 2 완료 후 임계값 재조정 |
| percentile 계산 시 표본이 적은 이벤트×feature 조합에서 순위가 거칠어짐 | 낮음 (문서에서도 "현재는 문제 적음"으로 확인) | 표본 n이 임계치(예: 30) 미만이면 percentile 대신 fallback 문구 사용 |
| 문구 계층 변경이 기존 CSV/텍스트 출력 포맷에 영향 | 낮음 | `build_table_rows`/`write_table_csv`는 grade만 사용하므로 영향 없음, `feedback_ko.txt` 문구만 변경 |

## 복잡도: Medium

Phase 2~4 로직 자체는 순수 함수 수준 변경이라 작지만, 기존 z-score 필드와의 하위 호환·회귀 검증(Phase 5)과 Phase 1 진단 결과에 따른 재확인 체크포인트, 그리고 Phase 0 베이스라인 스냅샷 준비가 있어 중간 난이도로 평가.

## 다음 확인 사항

- Phase 1 진단 결과를 본 뒤 Phase 2~5 진행 여부/세부 조정을 다시 확인한다.
- B1(도메인 지식 매핑 테이블) 포함 여부는 골프 도메인 전문가 검수 프로세스가 마련되면 재논의한다.
