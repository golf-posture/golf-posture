# 포즈 좌표 정확도 벤치마크

사람이 직접 찍은 팔 관절 라벨과 비교해서 포즈 추정 백엔드의 좌표 정확도를 재는 도구와, 지금까지 시험한 개선 방법의 결과를 기록합니다. 코드는 `pipeline/pose_benchmark.py`, `pose_crop.py`, `pose_review.py`입니다.

> **측정 스크립트는 Git에 없습니다.** 이 문서의 수치를 만든 `pose_benchmark`, `pose_review`, `pose_crop`, `golfpose_boxes`, `golfpose_infer`, `qc_refined_eval`, `qc_refined_report`, `qc_disagreement`, `shadow_compare`(와 증류 실험용 `pseudo_*`, `student_*`)는 실행에 필요하지 않아 `.gitignore`에 올려 두었고 개발 PC에만 있습니다. 운영에 쓰는 `pose_refiner`, `pose_source`, `vitpose_runtime` 등은 추적됩니다. 아래 명령은 그 스크립트가 있는 로컬 환경 기준입니다.

## 요약

| 시도 | 결과 | 판정 |
|---|---|---|
| **`model_complexity=2`** (MediaPipe heavy) | 틀린 팔 24.0% → **18.4%** (-5.6%p), PCK@0.1 0.456 → **0.541** (+0.085) | **유의미한 개선** |
| 골퍼 crop (YOLO + `event_subject`) | 모든 지표의 95% 신뢰구간이 0을 포함 | 효과 없음 |
| 역방향 추적 | 단독은 차이 없음, Top은 더 나쁨. oracle 상한도 Top에서 이득 없음 | Top·Finish에는 효과 없음 |
| **YOLOv8x-pose** (다른 계열, 프레임 단독) | 틀린 팔 18.6% → **9.1%**, Finish 59.6% → **14.1%**. Top은 47.7%로 c2와 같은 팔을 틀림(겹침 72%) | **Finish는 해결, Top은 미해결** |
| **GolfPose ViTPose-H** (골프 데이터 학습) | 틀린 팔 18.6% → **6.3%**, Top 58.1% → **17.4%**. Finish는 32.3% | **Top 해결**, Finish는 YOLOv8x가 더 좋음. 서버 전용(637M 파라미터) |
| 증류: ViTPose-H → YOLOv8n 8점 학생 | 틀린 팔 7.9~8.2%, Top 23~24%, Finish 40~42%. 교사보다 나쁨 | 서버 계산이 확정되어 **채택하지 않음** |
| **서버 조합: ViTPose-H + Finish만 YOLOv8x** | 틀린 팔 18.4% → **4.3%**, Top 17.4%, Finish **14.1%** | **현재 최선** (선택에 쓴 평가 영상을 그대로 써서 약간 낙관적) |

- 개선은 어려운 이벤트(Top, Mid-backswing, Mid-downswing)에 몰려 있고, 원래 잘 맞는 Address, Toe-up, Impact에는 변화가 없습니다.
- **Top(58%)과 Finish(60%)는 `complexity=2`에서도 절반 이상 틀립니다.** 모델 크기만으로는 풀리지 않는 구간이고, 다른 계열 모델과 골프 데이터로 학습한 모델에서 풀렸습니다(아래 "다른 계열 모델", "GolfPose").
- **계산은 서버에서 합니다(2026-11-24 확정).** 그래서 모바일 추론 비용 때문에 큰 모델을 피할 이유가 없고, 증류로 작은 모델을 만들 필요도 없습니다. 서버 추론 속도는 ViTPose-H가 RTX 3070에서 프레임당 약 0.28초입니다(466프레임 129초, 모델 로딩 포함).

## 데이터와 지표

**라벨** (`tests/fixtures/human_arm_labels*.json`, 원본 프레임 픽셀 좌표, 못 본 관절은 null)

| 파일 | 프레임 수 | 영상 |
|---|---|---|
| `human_arm_labels.json` (user) | 48 | `raw_videos/` |
| `human_arm_labels_pro.json` (pro) | 47 | `learning_video/golfdb_720p_slice/` |
| `human_arm_labels_selfmade.json` (selfmade) | 376 | `learning_video/selfmade_slice/` (`outputs/selfmade_label_map.json`으로 매핑) |

좌우 라벨이 뒤바뀌었을 수 있어 표시된 3프레임(`shorts_3`, `shorts2_7`, `longform_2`)은 제외합니다. 세 관절(어깨, 팔꿈치, 손목)이 모두 라벨된 팔만 채점해서 **870개 팔, 영상 100개**가 평가 대상입니다. 라벨 파일과 영상은 `.gitignore`에 있어 Git에 없고, 로컬에서만 재현됩니다.

**지표** (모든 오차는 사람이 찍은 해당 팔의 길이 = 어깨-팔꿈치 + 팔꿈치-손목 으로 나눈 값)

| 지표 | 정의 |
|---|---|
| PCK@0.1 / 0.2 | 관절 오차가 팔 길이의 10% / 20% 이내인 관절의 비율. 검출 못 한 관절은 miss로 셈 |
| 틀린 팔 비율 | 검출된 팔 중 최악 관절 오차가 팔 길이의 35%를 넘는 비율 (`qc_evaluate`와 같은 기준) |
| 좌우 뒤바뀜 | 틀린 팔이면서 모델의 반대쪽 팔이 사람 라벨과 맞는 경우의 비율 |
| coverage | 세 관절을 모두 낸 팔의 비율 |
| mean NME | 검출된 관절의 평균 정규화 오차 |

**기본은 원시 출력입니다.** QC와 Top/Finish 좌우 보정(`side_correction`)을 적용하기 전의 백엔드 출력을 `--min-visibility 0`(모든 출력 채점)으로 측정합니다. `--side-correction`을 주면 운영 경로의 Top/Finish 좌우 규칙을 이벤트 프레임에 적용한 뒤 채점합니다.

## 실행 방법

```powershell
# 측정 (결과 JSON 저장)
pixi run --locked python -m pipeline.pose_benchmark --backend mediapipe --complexity 2 --output outputs/pose_benchmark/mp_c2.json

# 두 결과 비교: 차이와 95% 신뢰구간(영상 단위 bootstrap 1000회)
pixi run --locked python -m pipeline.pose_benchmark --compare outputs/pose_benchmark/mp_c1.json outputs/pose_benchmark/mp_c2.json

# 두 결과를 완벽히 융합했을 때의 상한(라벨을 보고 팔마다 더 맞는 쪽 선택)
pixi run --locked python -m pipeline.pose_benchmark --oracle outputs/pose_benchmark/mp_c1.json outputs/pose_benchmark/mp_rev_c1.json

# 운영의 좌우 규칙(side_correction)을 적용한 뒤 채점
pixi run --locked python -m pipeline.pose_benchmark --backend mediapipe --complexity 2 --side-correction --output outputs/pose_benchmark/mp_c2_sc.json

# 실패한 팔을 눈으로 확인: 사람 라벨과 모델 결과를 나란히 그린 HTML 갤러리
# (--side-correction은 갤러리를 만든 결과와 같은 값으로 줘야 하고, 갤러리 제목에 적용 여부가 표시됩니다)
pixi run --locked python -m pipeline.pose_review --result outputs/pose_benchmark/mp_c2_sc.json --backend mediapipe --complexity 2 --side-correction --kind wrong --events Top,Finish --output outputs/pose_benchmark/review_c2_sc_wrong
```

백엔드: `mediapipe`(운영 경로와 같은 연속 추적), `mediapipe_crop`(골퍼 crop), `mediapipe_reverse`(역방향 추적), `yolo_pose`(YOLOv8-pose, `--yolo-weights`로 모델 선택). 다른 환경에서 돌린 모델은 `--predictions 캐시.json`으로 채점합니다(GolfPose, 증류 학생. 각각 `pipeline/golfpose_infer.py`, `pipeline/student_infer.py`가 캐시를 만듭니다). `--overlap A B`는 두 결과가 같은 팔에서 틀리는 정도를 보여줍니다. 새 백엔드는 `Backend` 프로토콜(`predict(video_path, frame_indices)`)을 구현하고 `BACKENDS`에 등록하면 됩니다.

## 결과 (2026-11-23, MediaPipe 0.10.21, Windows CPU)

### `complexity=1`(기준) 대비 `complexity=2`

| 지표 | c1 | c2 | 변화 [95% CI] |
|---|---|---|---|
| PCK@0.1 | 0.456 | 0.541 | **+0.085** [+0.061, +0.108] |
| PCK@0.2 | 0.710 | 0.785 | **+0.074** [+0.048, +0.102] |
| 틀린 팔 비율 | 24.0% | 18.4% | **-5.6%p** [-9.6, -1.8] |
| 좌우 뒤바뀜 | 9.0% | 6.8% | **-2.2%p** [-4.0, -0.4] |
| mean NME | 0.217 | 0.184 | -0.032 [-0.064, -0.005] |
| coverage | 0.997 | 0.998 | +0.001 [-0.005, +0.008] |

비교 대상은 두 실행이 모두 채점한 팔이 있는 98개 영상입니다.

**이벤트별** (틀린 팔 비율 / PCK@0.1, c1 → c2)

| 이벤트 | 틀린 팔 | PCK@0.1 | 좌우 뒤바뀜 |
|---|---|---|---|
| Address | 5.4% → 3.6% | 0.884 → 0.884 | 0% → 0% |
| Toe-up | 5.2% → 3.4% | 0.741 → 0.767 | 0% → 0% |
| Mid-backswing | 22.0% → **12.7%** | 0.395 → 0.559 | 10.2% → 5.1% |
| **Top** | 70.9% → **58.1%** | 0.058 → **0.190** | 47.7% → 41.9% |
| Mid-downswing | 20.7% → **8.9%** | 0.321 → 0.503 | 2.7% → 0.9% |
| Impact | 3.6% → 5.3% | 0.681 → 0.705 | 0% → 0% |
| Mid-follow-through | 15.9% → 10.6% | 0.277 → 0.404 | 0% → 0% |
| **Finish** | 64.6% → **59.6%** | 0.148 → 0.182 | 22.2% → 16.2% |

**데이터셋별** (틀린 팔 비율 / PCK@0.1, c1 → c2): user 15.6% → 10.4% / 0.593 → 0.662, pro 27.9% → 20.9% / 0.461 → 0.570, selfmade 24.4% → 19.0% / 0.440 → 0.524. 세 데이터셋 모두 같은 방향입니다.

### 좌우 보정(`side_correction`) 적용 후

운영 경로의 Top/Finish 좌우 규칙("trail 어깨가 화면에서 더 높다")을 이벤트 프레임에 적용하고(`--side-correction`) 같은 지표로 다시 쟀습니다. 규칙은 `pipeline/side_correction.py` 그대로입니다.

| | 틀린 팔 | 좌우 뒤바뀜 | PCK@0.1 | PCK@0.2 |
|---|---|---|---|---|
| c1 원시 | 24.0% | 9.0% | 0.456 | 0.710 |
| c1 + 보정 | 19.4% | 4.4% | 0.467 | 0.738 |
| c2 원시 | 18.4% | 6.8% | 0.541 | 0.785 |
| **c2 + 보정** | **15.6%** | **3.9%** | **0.545** | **0.805** |

- 보정 효과(95% CI): c1은 틀린 팔 -4.6%p [-6.1, -3.2], c2는 -2.9%p [-4.4, -1.3]. 강한 모델이 이미 좌우를 더 잘 맞히므로 보정의 이득은 c2에서 더 작습니다.
- 보정 후 c1 대비 c2: PCK@0.1 +0.078 [+0.054, +0.101], 틀린 팔 -3.8%p [-8.0, -0.0]. 보정을 적용해도 `complexity=2`의 이득은 남습니다.
- 규칙의 효과는 Top과 Finish에만 있고 나머지 6개 이벤트는 그대로입니다.

| 이벤트 | 모델 | 틀린 팔 (원시 → 보정) | 좌우 뒤바뀜 (원시 → 보정) |
|---|---|---|---|
| Top | c1 | 70.9% → 40.7% | 47.7% → 17.4% |
| Top | c2 | 58.1% → 41.9% | 41.9% → 25.6% |
| Finish | c1 | 64.6% → 50.5% | 22.2% → 8.1% |
| Finish | c2 | 59.6% → 48.5% | 16.2% → 5.1% |

**보정 후에도 Top은 약 42%, Finish는 약 49%가 틀립니다.** 남은 오류는 이벤트마다 성격이 다릅니다.
- Finish(c2): 틀린 팔 48.5% 중 뒤바뀜은 5.1%뿐이라 약 43%p가 위치 자체의 오류입니다. 좌우 규칙으로는 더 줄일 수 없습니다.
- Top(c2): 보정 후에도 뒤바뀜이 25.6%로 남습니다. 어깨 위치가 양쪽 다 약해서(오차가 팔 길이의 35%를 넘는 비율이 오른쪽 50%, 왼쪽 56%) 어깨 높이 규칙의 입력 자체가 부정확하기 때문일 수 있습니다. (확인 중)
- 이 코드 주석의 "Top 58/77 → 31/77"(약 40%)과 c1 보정 후 Top 40.7%가 일치해서, 벤치마크가 운영 동작을 재현합니다.

### Top/Finish 좌우 규칙 비교: y 규칙 대 x 규칙

현재 운영 규칙(y: trail 어깨가 화면에서 더 높다)과 새 후보(x: 이미지에서 x가 작은 어깨의 팔이 LEFT)를 `--side-rule`로 비교했습니다. x 규칙은 Top과 Finish에서 몸이 90° 넘게 돌아가 오른팔이 이미지 오른쪽에 온다는 사실에 기반하며(사람 라벨에서 92~98%), 영상이 좌우반전되어도, 골퍼의 손잡이와 상관없이 같은 형태로 성립합니다.

**사람 라벨에서 규칙이 성립하는 비율** (어깨 기준)

| 이벤트 | x 규칙 | y 규칙 |
|---|---|---|
| Top (50프레임) | 92% | 100% |
| Finish (60프레임) | 95% | 87% |

**모델 출력에서의 틀린 팔 비율 / 좌우 뒤바뀜**

| 모델 | 규칙 | 전체 | Top | Finish | 사용자 영상 6개 |
|---|---|---|---|---|---|
| c1 | 보정 없음 | 24.0% / 9.0% | 70.9% / 47.7% | 64.6% / 22.2% | 15.6% / 5.2% |
| c1 | y (운영) | 19.4% / 4.4% | 40.7% / 17.4% | 50.5% / 8.1% | 15.6% / 5.2% |
| c1 | **x** | **18.0% / 3.0%** | **32.6% / 9.3%** | **45.5% / 3.0%** | 22.1% / 11.7% |
| c1 | Top은 y, Finish는 x | 18.8% / 3.8% | 40.7% / 17.4% | 45.5% / 3.0% | 18.2% / 7.8% |
| c2 | 보정 없음 | 18.4% / 6.8% | 58.1% / 41.9% | 59.6% / 16.2% | 10.4% / 3.9% |
| c2 | y (운영) | 15.6% / 3.9% | 41.9% / 25.6% | 48.5% / 5.1% | 9.1% / 2.6% |
| c2 | **x** | **14.4% / 2.8%** | **31.4% / 15.1%** | 47.5% / 4.0% | 14.3% / 7.8% |
| c2 | Top은 y, Finish는 x | 15.4% / 3.8% | 41.9% / 25.6% | 47.5% / 4.0% | 11.7% / 5.2% |

**y 대비 x의 틀린 팔 비율 변화** (95% CI, 영상 단위 bootstrap, 음수면 x가 좋음)

| 대상 | c1 | c2 |
|---|---|---|
| 전체 팔 | -1.4%p [-3.0, +0.1] | -1.2%p [-2.7, +0.5] |
| Top + Finish, 모든 영상 | -6.5%p [-13.8, +1.0] | -5.4%p [-12.1, +1.8] |
| Top만 | -8.1%p [-21.0, +3.5] | -10.5%p [-23.3, +1.2] |
| Finish만 | -5.1%p [-11.5, +1.1] | -1.0%p [-6.0, +4.1] |
| Top + Finish, 사용자 영상 제외 | **-10.1%p [-15.7, -4.3]** | **-8.3%p [-14.5, -2.4]** |
| Top + Finish, 사용자 영상만 (16개 팔, 영상 6개) | +31.2%p [-20.0, +66.7] | +25.0%p [-20.0, +63.2] |

- 프로 스윙 영상에서는 x가 y보다 확실히 낫습니다. 모델이 어깨 높이(y)를 부정확하게 잡아서 y 규칙의 입력이 흔들리기 때문에, 라벨에서는 y가 더 정확한데도 모델 출력에서는 x가 Top에서도 앞섭니다.
- 사용자 영상에서는 x가 나빠 보입니다. 라벨에서 x가 틀린 `test1_7`, `test2_7`, `test3_3`, `test2_3`이 모두 사용자 영상이고, 아마추어는 회전이 덜해서 순서가 Address와 같게 남는 경우가 있는 것으로 보입니다. 다만 표본이 16개 팔(영상 6개)뿐이라 신뢰구간이 매우 넓어 단정할 수 없습니다.
- 전체 팔 기준 차이는 신뢰구간이 0을 포함합니다. 프로 영상이 대부분이라 프로에서의 이득이 평균에 반영된 결과입니다.
- **운영 코드(`side_correction.py`)는 바꾸지 않았습니다.** 실제 대상이 아마추어라면 결정에 아마추어 영상의 Top과 Finish 라벨이 더 필요합니다.
- 보조 단서로 "손에서 가장 가까운 어깨가 lead 어깨"도 라벨에서 시험했습니다. Finish에서 55/58(95%, 사용자 영상 4/5), Top에서 80%로, x 규칙보다 낫다고 할 근거는 없습니다.

### 효과 없던 시도

**골퍼 crop** (`mediapipe_crop`, c1 기준): YOLOv8n-pose로 골퍼를 찾아 `event_subject`의 고정 crop 안에서 MediaPipe를 돌렸습니다. 100개 영상 중 65개에만 crop이 적용됐고(나머지는 골퍼 선택 거부 또는 제거할 배경 부족), 모든 지표의 신뢰구간이 0을 포함합니다.

| 지표 | 변화 [95% CI] |
|---|---|
| PCK@0.1 | +0.011 [-0.003, +0.030] |
| 틀린 팔 비율 | -1.5%p [-5.1, +1.3] |
| 좌우 뒤바뀜 | +0.5%p [-0.3, +1.4] |

Top(틀린 팔 70.9%)과 Finish(64.6% → 65.7%)는 그대로입니다. 세로 영상은 골퍼가 이미 화면을 크게 채워 crop이 거의 줄어들지 않았습니다.

**역방향 추적** (`mediapipe_reverse`, c1 기준): 프레임을 끝에서부터 거꾸로 추적했습니다.

| 지표 | 변화 [95% CI] |
|---|---|
| PCK@0.1 | +0.004 [-0.017, +0.028] |
| 틀린 팔 비율 | -0.1%p [-3.8, +2.8] |

Top은 오히려 나빠졌습니다(틀린 팔 70.9% → 82.6%, 좌우 뒤바뀜 47.7% → 58.1%). 정방향과 역방향 중 팔마다 더 맞는 쪽을 라벨로 고른 oracle 상한도 Top에서 70.9% → 69.8%로 이득이 없고, Finish 64.6% → 58.6%, Mid-downswing 20.7% → 11.6%, 전체 24.0% → 19.2%입니다. oracle은 두 추정 중 좋은 쪽을 고르는 값이라 오류가 독립적이기만 해도 이득이 생기는 낙관적 상한입니다. 그럼에도 Top에서 이득이 없다는 것은 두 방향이 같은 오류를 낸다는 뜻입니다.

## 해석

- **Top의 좌우 뒤바뀜은 한 프레임의 실수가 아니라 구간 상태입니다.** selfmade 추적 캐시(이벤트 ±8프레임)에서 라벨과 비교할 수 있었던 Top 프레임 32건 중 30건이 이벤트 프레임에서 좌우가 뒤바뀌어 있었고, 그중 95%는 ±8프레임 전체에 같은 틀린 배정이 이어졌습니다(이웃 프레임이 뒤집혀 있는 비율은 0~3%). 이웃 프레임의 중앙값이나 보간(`temporal_event_value`)은 값이 비었을 때만 쓰이고, 뒤바뀐 프레임도 visibility가 높아 `direct`로 통과하므로 이를 고치지 못합니다. (이 분석의 "뒤바뀜"은 "반대 배정이 사람 라벨에 더 가깝다"는 느슨한 정의라 위 표의 좌우 뒤바뀜 수치와 직접 비교하면 안 됩니다.)
- 좌우 뒤바뀜을 제외해도 오류가 남습니다. c1 기준 Top은 틀린 팔 70.9% 중 뒤바뀜이 47.7%라 약 23%p, Finish는 64.6% 중 22.2%라 약 42%p가 위치 자체가 틀린 경우입니다.
- 기존 `side_correction`은 Top과 Finish에서 "trail 어깨가 더 높다"는 규칙으로 틀린 팔을 줄입니다(코드 주석 기준 Top 58/77 → 31/77). 이 규칙은 라벨로 임계값을 맞춘 in-sample 규칙이고 오른손잡이만 검증됐습니다.

## 개선 방법의 전제 재검증 (2026-11-23)

다음 개선 방법을 고르기 전에, 각 방법의 전제를 기존 결과로 직접 확인했습니다.

**1. 같은 계열 모델의 변형들은 같은 팔에서 틀립니다** (한 실행이 틀렸을 때 다른 실행도 틀릴 확률, 팔 기준)

| 비교 | Top | Finish | Mid-* |
|---|---|---|---|
| c1 → c2 (모델 크기만 다름) | 74% | 72% | 37% |
| c1 → 역방향 추적 | 98% | 91% | 63% |
| c1 → 골퍼 crop | 97% | 98% | 79% |

두 실행의 오류가 독립이면 이 값은 각 실행의 틀린 비율과 같아야 합니다(예: c2는 Top에서 58%). 역방향과 crop은 사실상 같은 오류를 내서, 이 변형들을 섞어도(앙상블, TTA 포함) 얻을 게 거의 없습니다. 팔마다 좋은 쪽을 라벨로 고르는 oracle도 c2 단독 대비 Top 58.1%에서 52.3%, Finish 59.6%에서 46.5%, Mid-* 10.8%에서 7.3%로, 상한조차 크지 않습니다.

**2. 모델은 자기 오류를 모릅니다** (c2 기준, "이 팔이 틀렸는가"를 맞히는 AUC, 0.5는 정보 없음)

| 이벤트 | 낮은 visibility | c1과 c2의 불일치 |
|---|---|---|
| Top | 0.57 | 0.64 |
| Finish | **0.44** | 0.65 |
| Mid-* | 0.54 | 0.77 |
| Address, Toe-up, Impact | 0.80 | 0.86 |

가장 많이 틀리는 Top과 Finish에서 visibility는 정보가 없거나(Finish는 0.5 미만) 오히려 반대입니다. 모델 간 불일치는 약하게만 신호가 됩니다. 라벨 없이 여러 후보 중 맞는 것을 고르기 어렵다는 뜻입니다.

**3. 기하 제약은 모델 출력에서 약한 탐지기입니다** (c2 기준, 임계값을 라벨의 5·95번째 백분위수로 잡은 in-sample 값이라 낙관적)

| 이벤트 | 제약 | 틀린 팔을 잡는 비율 | 맞는 팔을 잘못 거르는 비율 |
|---|---|---|---|
| Top | C1 손목이 어깨보다 아래 | 6% | 6% |
| Top | C4 팔 뼈 길이 비율 | 20% | 17% |
| Top | C2 두 손목 거리 | 36% | 25% |
| Top | 하나라도 위반 | 52% | 36% |
| Finish | C1 | 8% | 5% |
| Finish | C4 | 17% | 10% |
| Finish | C2 | 88% | 42% |

라벨 통계로는 C1~C3가 유망해 보였지만 모델 출력에서는 C1이 거의 쓸모없고, 효과가 있는 것은 Finish의 C2뿐입니다(그것도 맞는 팔의 42%를 잘못 거릅니다). 제약으로 후보를 고르는 방식은 근거가 약합니다.

**4. 이웃 프레임의 중앙값(시간 평활화)은 오히려 해롭습니다** (selfmade 추적 캐시, 틀린 팔 비율: 이벤트 프레임 값 → ±2프레임 중앙값 → ±4프레임 중앙값)

| 이벤트 | 이벤트 프레임 | ±2 | ±4 |
|---|---|---|---|
| Top | 74.7% | 73.4% | 74.7% |
| Finish | 61.0% | 62.3% | 61.0% |
| Mid-* | 20.4% | 23.7% | 39.6% |
| Address, Toe-up, Impact | 3.9% | 5.0% | 19.2% |

Top과 Finish는 변화가 없고(오류가 구간 전체에 이어짐), 움직임이 큰 이벤트는 평균이 실제 동작을 뭉개서 나빠집니다.

**결론**: 같은 모델의 변형, 시간 평활화, 제약 기반 선택은 모두 근거가 약하거나 해롭습니다. 남은 길은 오류가 독립에 가까운 다른 계열의 모델을 쓰거나, 골프 데이터로 모델 자체를 바꾸는 것이었고, 아래에서 둘 다 시험했습니다.

## 다른 계열 모델: YOLOv8-pose (2026-11-23)

`yolo_pose` 백엔드는 라벨된 프레임마다 YOLOv8-pose를 단독으로 돌립니다(추적 없음, imgsz 640, conf 0.25). 여러 사람 중 골퍼는 "박스 높이 × 몸통 4점 평균 신뢰도"가 가장 큰 사람으로 고릅니다. 원시 출력, 870팔 기준입니다.

| 모델 | 전체 | Top | Finish | 좌우 뒤바뀜(전체) |
|---|---|---|---|---|
| c2 (MediaPipe heavy) | 18.6% | 58.1% | 59.6% | 6.8% |
| **YOLOv8x-pose** | **9.1%** | 47.7% | **14.1%** | 3.2% |
| YOLOv8n-pose | 19.3% | 53.5% | 54.5% | — |

- **이득은 계열이 아니라 모델 크기에서 옵니다.** 같은 계열의 작은 모델(YOLOv8n)은 c2와 비슷합니다.
- Top은 YOLOv8x도 47.7%로, c2가 틀린 팔과 겹치는 비율이 72%입니다. Finish는 겹침이 작아 14.1%까지 내려갑니다.
- 두 결과가 같은 팔에서 틀리는 정도는 `--overlap A B`로 봅니다(한 모델이 틀렸을 때 다른 모델도 틀릴 확률).

## 골프 데이터 모델: GolfPose-2D (2026-11-24)

GolfPose-2D(ICPR 2024, `MingHanLee/GolfPose`)의 `person` 체크포인트를 같은 프레임과 같은 지표로 평가했습니다. 모델은 top-down이라 박스가 필요해서, YOLOv8x가 고른 골퍼 박스를 그대로 넘겼습니다.

| 모델 | 전체 | Top | Finish | 좌우 뒤바뀜(전체) |
|---|---|---|---|---|
| GolfPose HRNet-w48 (입력 288×384) | 20.7% | 51.2% | 47.5% | 3.0% |
| **GolfPose ViTPose-H** (입력 192×256, 637.2M) | **6.3%** | **17.4%** | 32.3% | **0.9%** |

- **ViTPose-H는 Top을 58.1%에서 17.4%로 줄였습니다.** 어떤 모델도 풀지 못한 구간이 풀렸습니다. 같은 계열의 HRNet은 c2 수준이라, 골프 데이터보다 모델 용량이 더 크게 작용한 것으로 보입니다(이 둘을 분리해서 시험하지는 않았습니다).
- 틀리는 팔의 겹침(한 모델이 틀릴 때 다른 모델도 틀릴 확률)은 Top에서 ViTPose-H 대 c2 26%, 대 YOLOv8x 32%, 대 HRNet 25%로, 계열이 다르면 오류가 크게 독립입니다. Finish에서는 ViTPose-H 대 YOLOv8x가 86%입니다. 이 값은 ViTPose-H가 더 많이 틀려서 높게 나옵니다.
- **키포인트 순서가 COCO가 아닙니다.** `person` 체크포인트는 Human3.6M 순서입니다(0 root, 1~3 오른쪽 hip/knee/foot, 4~6 왼쪽 hip/knee/foot, 7 spine, 8 thorax, 9 neck, 10 head, 11~13 왼쪽 shoulder/elbow/wrist, 14~16 오른쪽 shoulder/elbow/wrist). 체크포인트 `dataset_meta`의 `keypoint_id2name`으로 확인했고, `pipeline/golfpose_joints.py`에 매핑이 있습니다. COCO 순서로 읽으면 좌우와 관절이 모두 틀립니다.
- DEKR 체크포인트는 bottom-up이고 컴파일된 `mmcv` `DeformConv2d`가 필요해서 평가하지 못했습니다.
- **실행 환경이 둘로 나뉩니다.** mmpose 1.3.2는 Python 3.13 venv(`outputs/golfpose/venv`)에서 돌고, 박스는 pixi에서 만들어 JSON으로 넘깁니다. `xtcocotools`와 `mmdet`은 이 환경에 설치할 수 없어서 필요한 부분만 임시 shim으로 대체했고, `mmpose/models/heads/__init__.py`는 `EDPoseHead` import를 try/except로 감쌌습니다(CPU와 GPU 출력이 0.164px 이내로 같다는 것만 확인했고, shim이 결과를 바꾸지 않는다는 것은 가정입니다). 체크포인트는 `weights_only=True`로 읽히도록 텐서와 plain 메타만 담은 사본(`*_clean.pth`)을 따로 만들었습니다.

```powershell
# 1. pixi: 라벨된 프레임의 골퍼 박스 (YOLOv8x)
pixi run --locked python -m pipeline.golfpose_boxes --yolo-weights models/golfpose/yolov8x-pose.pt --output outputs/golfpose/boxes.json
# 2. venv: GolfPose 추론 -> 예측 캐시
outputs/golfpose/venv/Scripts/python.exe pipeline/golfpose_infer.py --device cuda:0 --config outputs/golfpose/golfpose_person_ViTPose_huge_full.py --checkpoint outputs/golfpose/golfpose_person_ViTPose_huge_clean.pth --output outputs/golfpose/pred_golfpose_vitpose_gpu.json
# 3. pixi: 캐시 채점
pixi run --locked python -m pipeline.pose_benchmark --predictions outputs/golfpose/pred_golfpose_vitpose_gpu.json --output outputs/pose_benchmark/golfpose_vit.json
```

**라이선스는 미확인입니다.** GolfPose 데이터셋은 자체 EULA(상용 모델 학습 허용, 원본 재배포·판매 금지)이고 이메일 승인이 필요합니다. 체크포인트 자체는 비밀번호 없이 받을 수 있었지만, 체크포인트의 라이선스는 확인하지 못했습니다. 상용 서비스에 쓰기 전에 확인해야 합니다.

## 합의 필터: 두 모델이 맞으면 믿을 수 있는가 (2026-11-24)

교사(ViTPose-H)의 라벨을 신뢰할 수 있는 경우를 가려내기 위해, 교사와 YOLOv8x가 같은 팔에서 얼마나 어긋나는지(최악 관절 거리 ÷ 교사의 팔 길이)를 보고, 어긋남이 작은 팔만 남겼을 때 교사가 틀리는 비율을 라벨로 재 봤습니다(라벨은 평가에만 쓰고 선택에는 쓰지 않았습니다).

| 이벤트 | 필터 없음 | 어긋남 ≤0.15 | 어긋남 ≤0.2 |
|---|---|---|---|
| Finish | 32.3% | **6.5%** (46% 유지) | 8.9% (57% 유지) |
| Top | 17.4% | 16.7~25% (개선 없음) | — |
| Mid-* | 1.2% | 1.2% | — |

- **Finish에서는 두 모델이 합의하면 거의 맞습니다.** 틀린 팔 32.3%가 6.5%로 줄고 절반 가까이 남습니다. 이 값은 "틀렸을 가능성이 높은 팔을 표시"하는 데 쓸 수 있습니다.
- Top에서는 합의해도 오류가 줄지 않습니다. 두 모델이 같은 방향으로 틀리는 경우가 있다는 뜻입니다.

## 증류 시도: ViTPose-H를 작은 모델로 (2026-11-24, 채택하지 않음)

처음에는 온디바이스를 가정해, ViTPose-H(교사)의 출력을 라벨로 삼아 YOLOv8n-pose(학생)를 학습했습니다.

- 가짜 라벨: 라벨이 없는 프로 영상 295개(평가용 47개 제외)의 이벤트 ±2프레임, 11,023프레임. 학생 키포인트는 팔 6점과 엉덩이 2점입니다(`pipeline/pseudo_label.py`의 `STUDENT_JOINTS`, 좌우 반전용 `FLIP_INDEX`). 검증은 영상 단위로 10%를 떼어 학습 모니터링에만 썼고, 최종 평가는 손 라벨 100개 영상에서만 했습니다.
- v1은 Finish에서 YOLOv8x와 0.15 넘게 어긋난 팔(67%)을 학습에서 제외했고, v2는 Finish 라벨도 교사 그대로 썼습니다(`--no-filter`). 둘 다 YOLOv8n-pose, 60 에폭입니다.

| | 전체 | Top | Finish | PCK@0.1 (c2 대비) |
|---|---|---|---|---|
| c2 | 18.6% | 58.1% | 59.6% | — |
| 학생 v1 | 7.9% | 24.4% | 40.4% | -4.2%p [-7.6, -0.4] |
| 학생 v2 | 8.2% | 23.3% | 42.4% | -1.5%p [-5.2, +2.3] |
| 교사 ViTPose-H | 6.3% | 17.4% | 32.3% | — |

- c2 대비 Top은 v1 -33.7%p [-45.0, -23.1], v2 -34.9%p [-45.3, -25.0], Finish는 v1 -19.2%p [-34.3, -4.1], v2 -17.2%p [-31.4, -2.8]입니다(영상 단위 부트스트랩 95% CI). **학생은 c2보다 확실히 좋았지만 교사를 넘지 못했습니다.**
- Finish를 고치려던 가설("어려운 Finish 포즈를 필터로 빼서 학생이 못 배운다")은 기각됐습니다. v2-v1은 Finish +2.0%p [-8.2, +11.1], Top -1.2%p [-9.0, +6.2]로 차이가 없습니다. 원인은 필터가 아니라 모델 용량이나 Finish 포즈의 난이도일 가능성이 크지만, 분리해서 확인하지는 않았습니다.
- 사용자 영상(77팔)에서는 v2가 11.7%로 c2(10.4%)보다 나빠, 아마추어에 대한 일반화가 확인되지 않았습니다.
- 서버에서 계산하기로 하여 증류의 목적(모바일 추론)이 사라졌습니다. 코드(`pseudo_*.py`, `student_*.py`)와 학습 결과는 남겨 두었습니다.

## 서버 조합: ViTPose-H + Finish만 YOLOv8x (2026-11-24)

계산이 서버에서 이루어지므로 큰 모델을 이벤트별로 골라 씁니다. 이벤트는 파이프라인이 이미 알고 있는 값이라 라벨을 쓰는 규칙이 아닙니다. 98개 영상, 870팔, 원시 출력의 틀린 팔 비율입니다.

| 방식 | 전체 | Top | Finish | selfmade | pro | user |
|---|---|---|---|---|---|---|
| c2 | 18.4% | 58.1% | 59.6% | 19.0% | 20.9% | 10.4% |
| ViTPose-H 단독 | 6.3% | 17.4% | 32.3% | 6.9% | 5.8% | 1.3% |
| YOLOv8x 단독 | 9.1% | 47.7% | 14.1% | 10.5% | 3.5% | 2.6% |
| 학생 v2 | 8.2% | 23.3% | 42.4% | 8.2% | 4.7% | 11.7% |
| **ViTPose-H + Finish만 YOLOv8x** | **4.3%** | 17.4% | **14.1%** | **4.7%** | 4.7% | **0.0%** |

c2 대비(95% CI): 전체 **-14.1%p [-18.0, -10.8]**, Top -40.7%p [-52.7, -29.5], Finish -45.5%p [-58.9, -31.3]. ViTPose-H 단독 대비 전체 -2.1%p [-3.0, -1.1]입니다.

**남은 오류의 상한.** 조합이 틀리는 팔은 37개(4.3%)입니다. 5개 모델(c2, ViTPose-H, YOLOv8x, HRNet, 학생 v2) 중 하나라도 맞히는 팔을 전부 맞힌다고 가정하는 이론상 상한(라벨을 보고 고르는 값)도 15개(1.7%)이고, Top은 15개에서 6개(7.0%), Finish는 14개에서 8개(8.1%)입니다. 정답 없이 모델을 고르는 방법은 앞서 확인했듯 마땅한 신호가 없어서, 모델을 더 찾는 것보다 얻을 몫이 작다고 판단하고 여기서 멈췄습니다.

**주의**
- **운영 기준선은 c2가 아니라 c1입니다.** `user_tracking.analyze_user_video`는 `extract_tracked_frames`를 `complexity` 지정 없이 불러서 기본값 1을 씁니다(코드에서 2를 넘기는 곳은 없음). c1의 원시 출력은 틀린 팔 24.0%, Top 70.9%, Finish 64.6%이고, 운영의 y 규칙을 적용하면 19.4%, 40.7%, 50.5%입니다. 위 표의 c2는 "MediaPipe를 한 단계 키웠을 때"의 참고값이며, 운영 대비 개선폭은 표보다 큽니다.
- Finish에 YOLOv8x를 고른 것은 같은 평가 라벨을 본 뒤의 선택이라 4.3%는 약간 낙관적입니다. 새 영상을 구할 수 없어 따로 검증하지 못했습니다. 두 모델이 Finish에서 14.1% 대 32.3%로 차이가 커서 선택이 뒤집힐 가능성은 낮다고 봅니다.
- Top은 17.4%로 여전히 가장 약합니다. 평가 팔이 86개뿐이라 신뢰구간이 ±10%p 안팎입니다.
- 불확실성 표시(합의 필터에서 두 모델이 어긋나면 "자세 인식이 불확실"로 알리는 방식)는 효과가 있을 것으로 보이나, 앱이 받아 쓸 수 있는지 확인되지 않아 구현하지 않았습니다.

## 운영 통합: 새 모델로 팔 관절 덮어쓰기 (2026-11-24)

분석 파이프라인에 서버 조합을 연결했습니다. 기본은 꺼져 있습니다.

```powershell
# 새 모델로 팔 관절을 덮어쓰고(이벤트 8개 전부), 새 기준 통계로 채점
pixi run --locked python -m pipeline.run_analysis --video <영상.mp4> --pose-refiner golfpose --reference refined
# 일부 이벤트만: --pose-refiner-events Top,Finish   (이름 또는 0-7 번호, 기본 all)
# 모델 파일: --vitpose-model models/golfpose/vitpose_huge.ts
```

- MediaPipe 추적이 뼈대입니다(3D 좌표, 다른 랜드마크, 시간 QC). 각 이벤트 ±2프레임에서 **어깨·팔꿈치·손목의 2D 위치와 점수만** 새 모델 값으로 바꿉니다. 엉덩이는 엉덩이 라벨로 검증한 적이 없어서 MediaPipe 값을 그대로 씁니다.
- 모델은 ViTPose-H이고 Finish만 YOLOv8x입니다. mmpose 없이 TorchScript로 내보낸 파일을 씁니다(`docs/vitpose_server.md`). 모델이 실패하거나 이벤트가 8개가 아니면 MediaPipe 결과를 그대로 쓰고 `qc.pose_refiner`에 사유를 남깁니다.
- **Top/Finish 좌우 규칙(`side_correction`)은 분석에서 뺐습니다.** 새 모델 답에는 규칙이 도움이 되지 않았습니다(98개 영상의 틀린 팔: 규칙 없음 Top 15·Finish 14, 규칙 적용 16·17). 모듈은 이 문서의 근거를 재현하는 벤치마크용으로만 남겼습니다. 모델이 실패해 MediaPipe로 폴백하면 이 규칙도 없습니다.

### QC 재보정

라벨된 영상 50개(이벤트 8개가 모두 라벨된 사용자 6 + selfmade 44, 743팔)로 기본 경로와 새 모델 경로의 QC 입력을 측정했습니다(`pipeline.qc_refined_eval`, `pipeline.qc_refined_report`).

| | 기본(MediaPipe) | 새 모델 |
|---|---|---|
| 틀린 팔 | 141 | 35 (고친 팔 118, 망친 팔 12) |
| 현재 QC가 잡은 틀린 팔 | 40 | 10 |
| 현재 QC가 버린 정상 팔 | 15/602 | 44/708 (6.2%) |

새 모델 팔에서는 `bone`과 `inconsistent`가 MediaPipe 3D 좌표를 읽는데 3D는 바뀌지 않아서, 새 2D 팔이 맞아도 MediaPipe의 실패를 이유로 버렸습니다(버린 34개 중 20개는 기본 경로에서 틀렸던 팔). 새 모델의 점수는 틀린 팔과 정상 팔을 가르지 못합니다(중앙값 0.75 대 0.78). 그래서 덮어쓴 팔에만 `refined_arm_flags`를 씁니다: `bone`·`inconsistent` 끔, `low_vis` 기준 0.4→0.3. 같은 데이터에서 틀린 팔 1개를 잡고 정상 팔 3개(0.4%)를 버립니다(같은 라벨로 정하고 확인한 값이라 약간 낙관적입니다). 기본 경로 규칙은 그대로입니다.

**남은 틀린 팔 35개는 QC로 못 거릅니다.** 31개가 Top(15)과 Finish(16)입니다. ViTPose-H와 YOLOv8x의 팔 불일치(`pipeline.qc_disagreement`)는 어느 임계값에서도 순손해였고(Top >0.3: 틀린 팔 4/16 포착·정상 21/70 버림, Finish >0.3: 10/17 포착·정상 20/82 버림), 좌우 뒤바뀜 신호도 33개 중 7개만 잡고 정상 25개를 버렸습니다. 맞는 팔에서도 두 모델이 팔 길이의 0.1~0.3만큼 다르기 때문입니다.

### 프로 기준 통계 재생성

342개 프로 영상을 새 모델로 다시 재서 `data/refined/`에 만들었습니다(`data/`와 `data/tracked/`는 그대로).

```powershell
# 1. GPU 환경(venv): 모델 답을 한 번 계산해 캐시 (약 26분)
outputs\golfpose\venv\Scripts\python.exe -m pipeline.refiner_cache --output outputs/refined_pro/refiner_cache.json
# 2. pixi: 캐시와 MediaPipe 추적을 합쳐 추출 (약 20분), QC, 통계 입력 변환, 통계
pixi run --locked python -m pipeline.extract_pro_features_tracked --events-table data/pro_feature_raw_tracked.csv --refiner-cache outputs/refined_pro/refiner_cache.json --output outputs/refined_pro/pro_feature_raw_tracked.csv
pixi run --locked python -m pipeline.qc_tracked_features --input outputs/refined_pro/pro_feature_raw_tracked.csv --output outputs/refined_pro/pro_feature_qc_tracked.csv
pixi run --locked python -m pipeline.export_tracked_stats_input --tracked outputs/refined_pro/pro_feature_raw_tracked.csv --qc outputs/refined_pro/pro_feature_qc_tracked.csv --output data/refined/pro_feature_raw.csv
pixi run --locked python -m pipeline.compute_reliability_stats --raw-csv data/refined/pro_feature_raw.csv --output-json data/refined/pro_feature_stats_reliability.json --stats-csv data/refined/pro_feature_stats_reliability.csv --covariance-json data/refined/pro_feature_covariance.json
pixi run --locked python -m pipeline.compute_dynamics_stats --raw-csv data/refined/pro_feature_raw.csv --output-json data/refined/pro_dynamics_stats.json
```

모션 통계, 대표 스켈레톤, 템포 통계는 다시 만들지 않았습니다.

| 항목 | 기존(`data/tracked`) | 새 기준(`data/refined`) |
|---|---|---|
| Mid-downswing 팔꿈치 표준편차 (trail / lead) | 30.7 / 13.4 | 17.7 / 8.3 |
| Mid-follow-through 팔꿈치 표준편차 | 12.4 / 11.0 | 6.9 / 5.9 |
| Mid-backswing trail 팔꿈치 표준편차 | 19.2 | 12.9 |
| Finish trail 팔꿈치 측정 영상 수 | 223 | 342 |
| Finish 어깨 기울기 | 쌍봉(평균 178.5, 표준편차 91.0) | 한쪽(평균 2.8, 표준편차 27.4) |
| Top 어깨 기울기 | 평균 176.2, 표준편차 11.5 | 평균 56.5, 표준편차 69.6 |

팔꿈치 신뢰도는 Top·Mid-downswing에서 낮음→보통으로 올랐습니다(Finish 팔꿈치는 그대로 낮음). Finish 어깨 기울기의 쌍봉은 풀렸습니다(|기울기|>120 이 63%→2%). 반대로 **Top 어깨 기울기가 갈라졌습니다**(|기울기|>120 이 95%→34%, <60 이 59%, 340개 중 214개 영상에서 MediaPipe와 방향이 바뀜). 좌우 규칙을 지운 영향으로 보이며, 통계 스크립트가 이 항목을 `low/reference`로 낮춰 강한 피드백에는 쓰이지 않습니다. 원인은 분포와 라벨 비율로 추정한 것이고 해당 영상을 눈으로 확인하지는 않았습니다. `hip_tilt`는 MediaPipe 값 그대로라 Finish 쌍봉(표준편차 82)이 남아 있습니다.

### 섀도 비교 (라벨된 영상 50개, 정답 없음)

같은 영상을 기본 경로와 새 경로로 분석해 `feedback_app.json`을 비교했습니다(`pipeline.shadow_compare`). 두 조합을 봤습니다.

| | 새 모델 + 기존 통계(legacy) | 새 모델 + 새 통계(refined) |
|---|---|---|
| 평가 불가 특징 | 92 → 29 | 92 → 29 |
| 신뢰도 낮음→보통 | 8편 | 8편 |
| 점수 변화 중앙값(범위) | -0.4 (-1.9 ~ +3.2) | -2.5 (-4.7 ~ +1.2) |
| 프로 범위 안 항목 평균 | 32.4 → 30.8 | 32.4 → 24.9 |
| 채점 항목 수 평균 | 38.1 → 38.1 | 38.1 → 41.9 |
| 이슈 카드 / 칭찬 카드 바뀐 영상 | 38 / 33 | 37 / 33 |
| 상태 변경 | 4편 | 4편 |

새 통계는 분포가 좁아져서 채점이 더 엄격해지고 점수가 약 2.5점 내려갑니다. 정답이 없는 비교라 "내려간 점수가 더 정확하다"고 말할 수는 없습니다. 앱 점수 환산(`SCORE_PER_Z`, 임시 `STATUS_CUTS`)은 이전 분포에 맞춘 값이라, 새 통계를 쓰려면 이 값을 다시 정해야 합니다.

## 한계

- 라벨은 어깨, 팔꿈치, 손목뿐이라 몸통과 하체 정확도는 알 수 없습니다. 정면 영상 위주이고 라벨된 골퍼는 오른손잡이가 대부분입니다.
- 별도 표기가 없는 수치는 원시 출력입니다. `side_correction` 적용 후 수치는 위 절에 있고, 시간축 QC(`qc_temporal`)와 값 복구는 아직 반영하지 않았습니다. 보정은 이벤트 프레임에만 적용하며 오른손잡이 기준입니다.
- **추론 비용은 서버 기준으로만 알고 있습니다.** ViTPose-H는 RTX 3070에서 프레임당 약 0.28초입니다. 계산은 서버에서 하기로 했지만, 운영 서버의 GPU와 동시 요청 수에 따른 처리량은 재지 않았습니다.
- **출시 전 확인이 남았습니다.** GolfPose 체크포인트의 상용 라이선스와 ultralytics(AGPL-3.0)를 확인하지 못했습니다. 서버의 torch 버전 호환(내보낼 때 2.11, CPU 검증은 2.10), 리눅스, 서버 GPU 속도도 재지 않았습니다.
- 서버 조합은 라벨된 프레임에서의 정확도(위 절)와 라벨된 영상 50개의 섀도 비교로만 확인했습니다. 섀도 비교에는 정답이 없고, 새 통계로 바뀌는 앱 점수가 더 정확한지는 모릅니다.
- 신뢰구간은 영상 단위 resampling이라 같은 영상 안의 프레임 상관을 반영합니다. 라벨이 selfmade 영상 47개에 많이 몰려 있어(707/870 팔) 전체 수치는 selfmade 영향이 큽니다.
- 결과 JSON은 `outputs/`(Git 제외)에 저장되므로, 위 수치를 다시 얻으려면 같은 영상과 라벨이 로컬에 있어야 합니다.
