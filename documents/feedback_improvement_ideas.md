# 골프 스윙 피드백 신뢰도 개선 - 대안 정리

작성일: 2026-09-15
상태: 아이디어 정리 단계 (구현 전, 방식 미확정)

## 배경 / 문제 정의

현재 피드백 로직(`pipeline/run_analysis.py:feedback_for_event`, `pipeline/compute_reliability_stats.py`)은
프로 342명의 이벤트별 각도 feature 평균/표준편차와 사용자 값을 비교하는 z-score 방식이다.

```
z_score = (사용자값 - 프로평균) / 프로표준편차
|z| <= 1.0 -> Excellent, <= 2.0 -> Good, 그 외 -> Needs Improvement
```

지목된 핵심 문제 2가지 (우선순위 순):

1. **z-score 통계 자체가 부실함** - feature별 독립 비교(상관관계 무시), 정규분포 가정(원형 feature인
   shoulder/hip tilt에도 동일 적용), 이상치(포즈 추정 오류가 섞인 프로 샘플)에 취약.
2. **피드백 문구가 기계적/일반적** - `generate_feedback_report.py`의 `sentence_for_core` /
   `sentence_for_reference`가 고정 템플릿에 z-score 숫자만 끼워 넣는 수준.

부차적으로 논의된 항목(우선순위는 아니지만 근본 원인과 연결됨):

3. 단일 이벤트 프레임 1장만 사용 -> pose 추정 오차/occlusion이 그대로 각도 계산에 반영됨.

## 참고한 자료

- `documents/Applying_Pose_Estimation_to_Predict_Amateur_Golf_Swing_Performance_Using_Edge_Processing.pdf`
  (Kim et al., 2020, IEEE Access) - Savitzky-Golay 필터로 pose 시계열 스무딩, 손목/골반 교차 등
  다중 신호로 이벤트 검출, P1-P7 구간 궤적으로 스윙 경로(훅/페이드) 예측.
- `documents/KCI_FI002927622.pdf` (Park, 2023, JKSCI) - 프로 평균이 아니라 개별 프로 스윙과
  DTW(Dynamic Time Warping)로 궤적 유사도를 측정, 스윙을 구간(Address-Top, Top-Impact,
  Impact-Finish)으로 나눠 비교.
- 웹 검색: robust z-score(median/MAD, scale 1.4826)와 백분위 기반 채점은 이상치에 강하고 정규분포를
  가정하지 않는 업계 표준 단순 기법으로 확인됨(표본 30개 이상이면 안정적 - 현재 이벤트당 300명대라 충분).

## 리소스 제약 (중요)

- 342개 프로 원본 슬라이스 영상(`golfdb_720p_slice/`)과 SwingNet 체크포인트
  (`swingnet_1800.pth.tar`, `mobilenet_v2.pth.tar`)는 git에도 이 로컬 머신에도 없음
  (팀원 PC 경로만 코드에 하드코딩되어 있었음, `.gitignore`로 제외됨).
- 현재 남은 프로 데이터는 이벤트별 스냅샷 feature뿐(`data/pro_feature_raw.csv`, 342명 x 8이벤트),
  프레임 단위 연속 궤적 데이터는 없음.
- GolfDB 자체는 공개 데이터셋이라 재확보는 가능하나(SwingNet도 이걸로 학습됨), 342개 영상을
  재다운로드/재슬라이스/재처리하는 작업은 상당한 리소스가 필요함.
- SwingNet 체크포인트는 사용자가 별도로 보유, 곧 배치 예정.

## 제외한 방식 (난이도/리소스 문제로 배제)

- **DTW 기반 궤적 유사도 비교** (논문2 방식 온전 구현) - 342개 프로 영상 재확보 + 프레임 단위 pose
  재추출 + Savitzky-Golay 스무딩 + 구간별 DTW 계산까지 필요. 리스크와 작업량이 큼.
- **다변량 공분산(Mahalanobis distance) 결합** - 5~7개 feature를 공분산 고려해 하나의 결합
  점수로 묶는 방식. 통계적으로는 더 정교하지만 구현/검증 복잡도가 높음.
- **LLM 기반 문구 생성 레이어** - 구조화된 수치를 LLM에 넣어 자연어로 다듬는 방식. 외부 의존성,
  지연시간, 할루시네이션(잘못된 수치 서술) 방지용 가드레일이 추가로 필요함.

## 대안 A - 통계 비교 방식 (z-score 대체/보완)

기존 `pro_feature_raw.csv`(342명 x 8이벤트, 이미 보유)만으로 적용 가능. 신규 데이터 불필요.

| ID | 방식 | 설명 | 장점 | 단점 |
|----|------|------|------|------|
| A1 | Robust z-score | `(값 - median) / (1.4826 * MAD)` | 이상치에 강함, 계산 간단 | 여전히 "표준편차형" 해석이라 일반인에게 덜 직관적 |
| A2 | 백분위 순위 | 342명 분포 내 사용자 값의 순위를 %로 환산 | 분포 가정 전혀 없음, "상위 20%" 식으로 가장 직관적, 구현 극히 쉬움(`scipy.stats.percentileofscore` 등) | 표본이 적은 조합에서는 순위가 거칠어질 수 있음(현재는 문제 적음) |
| A3 | IQR 펜스 | Q1-1.5*IQR ~ Q3+1.5*IQR 안이면 정상, 벗어나면 개선 필요 | 통계학 표준 이상치 판정 기준, 설명하기 쉬움 | 3단계 등급으로 세분화하려면 추가 임계값 필요 |
| A4 | Winsorized z-score | 프로 데이터 상하위 2%만 clip 후 기존 mean/std z-score 그대로 사용 | 기존 코드 변경 최소, 이상치 영향만 완화 | median/MAD보다 이상치 강건성은 약함 |

관련 코드 위치: `pipeline/compute_reliability_stats.py:compute_feature_stats`,
`pipeline/run_analysis.py:grade_from_z`, `feedback_for_event`.

## 대안 B - 피드백 문구 방식 (기계적 느낌 해소)

| ID | 방식 | 설명 | 장점 | 단점 |
|----|------|------|------|------|
| B1 | 도메인 지식 매핑 테이블 | (이벤트, feature, 편차방향)별 원인/영향 한 줄 설명 사전 작성 | 가장 "코칭"답게 느껴짐 | 골프 지식이 틀리면 신뢰도가 오히려 하락, 검수 부담 큼 |
| B2 | 편차 크기별 문장 강도 자동 조절 | 백분위/robust z 크기 구간별로 "약간/뚜렷하게/크게" 등 표현 자동 선택 + 실제 각도 차이(도) 명시 | 데이터만으로 자동 생성, 도메인 지식 오류 리스크 없음 | 원인 설명까지는 제공하지 못함 |
| B3 | 정직한 순위 비교 문구 | "프로 342명 중 상위/하위 X%에 해당합니다" 식으로 순위만 명시, 원인 해석 자제 | 통계적으로 정확, 과장 없음 | 일부 사용자에게는 여전히 코칭처럼 안 느껴질 수 있음 |
| B4 | 템플릿 다양화 | 같은 등급이라도 여러 문장 변형 중 선택 | 구현 매우 쉬움, "기계적 반복" 인상 제거 | 근본적인 정보량 자체는 그대로 |

관련 코드 위치: `pipeline/generate_feedback_report.py:sentence_for_core`,
`sentence_for_reference`, `FEATURE_KO`/`EVENT_KO`.

## 대안 C - 사용자 영상 프레임 노이즈 완화 (선택 사항, 우선순위 낮음)

| ID | 방식 | 설명 | 장점 | 단점 |
|----|------|------|------|------|
| C1 | 윈도우 median | 이벤트 프레임 +-3~5프레임의 각도값 median 사용 | 구현 쉬움, 튄 값 제거 | 여러 프레임 pose 추출로 처리 시간 소폭 증가 |
| C2 | 신뢰도 가중 평균 | 각 프레임의 landmark visibility 점수로 가중평균 | 기존 visibility 값 재활용 | 여전히 여러 프레임 필요 |
| C3 | 최고 신뢰도 프레임 선택 | +-N 프레임 중 pose confidence가 가장 높은 1장만 사용 | 가장 단순(평균/필터 계산 없음), 처리비용 영향 적음 | 실제 이벤트 시점과 살짝 어긋날 수 있음 |
| C4 | 백엔드 합치도 체크 | MediaPipe/YOLO 결과가 일정 오차 이내로 일치할 때만 신뢰 | 이미 두 백엔드를 다 돌리고 있어 추가 영상처리 불필요 | 원래도 잘 안 맞는 landmark(예: 척추)엔 효과 적음 |

관련 코드 위치: `pipeline/run_analysis.py:analyze_pose_backend`, 이벤트 루프(`run` 함수).

## 다음에 결정할 것

- A(통계 방식) 중 어떤 조합을 쓸지 (내부 계산용과 사용자 표시용을 다르게 가져가도 됨,
  예: 내부는 A1, 화면 표시는 A2).
- B(문구 방식) 중 어떤 조합을 쓸지 (B2+B4 기본 조합 + 시간 되면 B1 추가하는 방향이 논의됨).
- C(프레임 노이즈 완화)를 이번 범위에 포함할지, 나중으로 미룰지.
- 이 문서는 구현 착수 전 결정을 내릴 때 참고용으로 유지. 실제 구현 시작 시 `/plan`으로
  다시 세부 Task를 쪼갤 것.
