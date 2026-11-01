# golf-posture

골프 스윙 영상에서 SwingNet으로 8개 이벤트를 검출하고, 자세와 동작을 프로 기준 통계와 비교해 한국어 피드백을 생성하는 Python 분석 파이프라인입니다.

현재 지원 대상으로 삼는 입력은 **정면(face-on), 단일 화면, 드라이버 한 스윙 영상**입니다. 기본 분석은 MediaPipe 연속 추적과 QC를 사용하며, 이벤트 검출 개선 기능은 `--event-refinement subject-phase`로 선택합니다. 기본값은 `none`입니다.

## 빠른 시작

아래 명령은 PowerShell 기준입니다. 분석 명령은 항상 저장소 루트에서 실행합니다.

### 1. 환경 설치

Git과 Pixi를 설치한 뒤 저장소를 받습니다. 이미 받은 저장소라면 해당 폴더로 이동해 설치 명령부터 실행하면 됩니다.

```powershell
git clone https://github.com/golf-posture/golf-posture.git
Set-Location golf-posture
pixi install --locked
pixi run --locked python -m pipeline.run_analysis --help
```

환경은 [pixi.toml](pixi.toml)과 [pixi.lock](pixi.lock)으로 관리합니다. Python 3.12, PyTorch, torchvision, OpenCV, MediaPipe 0.10.21, Ultralytics 등을 설치합니다. 환경 정의에는 `win-64`와 `linux-64`가 있으며, 아래 최근 실제 실행 검증은 **Windows CPU**에서 수행했습니다.

### 2. 모델 가중치 준비

대형 가중치와 입력 영상은 Git에 포함하지 않습니다. 별도로 받은 가중치를 다음 위치에 배치합니다.

| 파일 | 용도 | 필요 조건 |
|---|---|---|
| `event_detection/models/swingnet_1800.pth.tar` | SwingNet 이벤트 검출 | 실제 이벤트 검출 시 필수 |
| `yolov8n-pose.pt` | 골퍼 선택·관절 관측, YOLO Pose | `subject-phase` 또는 YOLO Pose 사용 시 필요 |
| `pipeline/models/swingnet_phase_ranker_v1.json` | 이벤트 후보정 모델 | `subject-phase`에 필요하며 **Git에 포함됨** |

SwingNet 가중치는 [GolfDB 원본 저장소](https://github.com/wmcnally/golfdb)의 공개 [체크포인트](https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view)를 사용합니다. `event_detection/models/` 폴더가 없으면 생성합니다.

YOLO Pose 가중치는 저장소 루트에서 다음 명령으로 준비할 수 있습니다. 최초 다운로드에는 인터넷 연결이 필요합니다.

```powershell
pixi run --locked python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

완성된 SwingNet 체크포인트를 읽을 때는 `mobilenet_v2.pth.tar`가 필요하지 않습니다. `EventDetector(pretrain=True)`를 직접 사용하는 별도 학습 작업에만 필요합니다.

`subject-phase`는 SwingNet 체크포인트와 후보정 JSON의 SHA256을 [event_refinement.py](pipeline/event_refinement.py)의 값과 비교합니다. 이는 검증된 모델과 파일 내용이 같은지 확인하는 검사이며, 불일치하면 실행을 중단합니다. 후보정 JSON의 바이트는 [.gitattributes](.gitattributes)로 보존하므로 자동 줄바꿈 변환이나 재저장을 피합니다.

### 3. 영상 분석

`$video`를 실제 영상 경로로 바꿉니다. 다음 두 명령은 각각 기본 모드와 개선 모드이며 결과를 다른 폴더에 저장합니다.

```powershell
$video = "C:/videos/swing.mp4"

# 기본 이벤트 검출 + MediaPipe 추적 + 기존 프로 기준 피드백
pixi run --locked python -m pipeline.run_analysis --video $video --output "outputs/swing_baseline/"

# 골퍼 중심 영역과 문맥 기반 이벤트 후보정 사용
pixi run --locked python -m pipeline.run_analysis --video $video --event-refinement subject-phase --output "outputs/swing_subject_phase/"
```

`--output` 경로 끝에 `/`를 붙이면 폴더로 저장됩니다. 일반 사용자용 결과는 각 폴더의 `feedback_user.txt`부터 확인하면 됩니다.

이벤트 검출은 `torch.cuda.is_available()`에 따라 CUDA 또는 CPU를 자동 선택합니다. CUDA는 학습뿐 아니라 이벤트 추론에도 사용되지만, 이 설정이 모든 자세 처리 단계를 GPU로 실행한다는 뜻은 아닙니다.

## 분석 흐름과 이벤트 검출 개선

```text
한 스윙 영상
  → SwingNet 이벤트 검출 / 선택적 subject-phase 후보정
  → 시간 순서에 맞는 이벤트 선택, 명시적 누락·점수 기준 보류 처리
  → MediaPipe 연속 추적 또는 프레임별 Pose 추출
  → 자세 feature·템포·동작 지표 계산
  → 프로 기준 비교, QC와 reliability 반영
  → JSON·한국어 피드백·이벤트 이미지·스켈레톤 비교 저장
```

이벤트 순서는 다음과 같습니다. 결과의 `frame_index`는 **입력 영상의 0부터 시작하는 프레임 번호**입니다.

| 번호 | 이벤트 |
|---:|---|
| 0 | Address |
| 1 | Toe-up |
| 2 | Mid-backswing |
| 3 | Top |
| 4 | Mid-downswing |
| 5 | Impact |
| 6 | Mid-follow-through |
| 7 | Finish |

`subject-phase`는 영상 전체에서 골퍼를 선택해 고정 영역을 잡고, 160×160 입력과 각 후보 프레임 주변의 64프레임 문맥으로 SwingNet 점수를 계산합니다. Mid-downswing·Impact·Mid-follow-through를 기준점으로 삼아 나머지 5개 이벤트를 고정된 선형 모델로 후보정합니다. 공개 SwingNet 가중치를 재학습하지 않으며, 이후의 자세 추적·QC·피드백·기준 통계는 같은 파이프라인을 사용합니다.

이 옵션은 **실제 이벤트 검출, `--event-selection ordered`, `--seq-length 64`** 조합에서만 사용할 수 있습니다. 위 두 옵션은 기본값이므로 보통 별도 지정하지 않아도 됩니다. `--mock-events`와는 함께 사용할 수 없습니다.

### 누락 이벤트와 낮은 점수 처리

영상에서 Finish가 잘린 사실을 알고 있다면 다음처럼 지정합니다.

```powershell
pixi run --locked python -m pipeline.run_analysis --video $video --event-refinement subject-phase --missing-events Finish --output "outputs/swing_missing_finish/"
```

- `--missing-events`는 사용자가 알고 있는 누락을 전달하는 옵션이며, 누락을 자동 판별하는 기능은 아닙니다.
- `--min-event-confidence 0.5`처럼 지정하면 선택된 프레임의 모델 점수가 기준보다 낮은 이벤트를 보류합니다. 기본값 `0`은 점수 필터를 끕니다. 이 점수는 정확도로 보정된 신뢰도가 아닙니다.
- 누락·보류 이벤트는 `frame_index: null`로 남고 자세 평가에서 제외됩니다. Address·Top·Impact 중 하나가 없으면 템포도 `unavailable`입니다.
- 누락 이벤트를 지정한 `subject-phase`는 완전한 8이벤트용 선형 후보정 대신, 관측 대상 이벤트의 문맥 점수와 시간 순서로 프레임을 선택합니다.

## 주요 실행 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--event-refinement` | `none` | `subject-phase`로 이벤트 후보정 선택 |
| `--event-selection` | `ordered` | 프레임 순서를 강제. `independent`는 이벤트별 독립 선택 |
| `--seq-length` | `64` | SwingNet 문맥 길이. `subject-phase`는 64 고정 |
| `--pose-backend` | `mediapipe` | `mediapipe`, `yolopose`, `auto`, `hybrid` |
| `--extraction` | `tracked` | 연속 추적·가려진 팔 값 복구·QC. `static`은 이벤트별 독립 추론 |
| `--reference` | `legacy` | 기본 `data/` 통계. `tracked`는 별도 준비한 `data/tracked/` 통계 |
| `--handedness` | `right` | 왼손잡이는 `left` 지정 |
| `--checkpoint` | 기본 SwingNet 경로 | 다른 경로의 체크포인트를 직접 지정 |
| `--missing-events` | 없음 | 영상에서 빠진 이벤트 이름을 하나 이상 지정 |
| `--min-event-confidence` | `0.0` | 선택 프레임의 모델 점수 기준 보류 |
| `--mock-events` | 사용 안 함 | 균등 분할된 테스트 이벤트로 실행. SwingNet 가중치 불필요 |

`tracked` 자세 추출은 `--pose-backend mediapipe`에서만 동작합니다. `yolopose`, `auto`, `hybrid`를 지정하면 `static`으로 처리되며 결과의 `extraction`과 `extraction_note`에 기록됩니다. `hybrid`는 이벤트별로 계산 가능한 feature 수 등을 비교해 backend를 선택합니다.

`--extraction`과 `--reference`는 독립적입니다. 따라서 기본 실행은 **추적해서 얻은 사용자 자세를 기존 프로 통계로 평가**합니다. `--reference tracked`는 추적 재추출·QC 후 만든 별도 통계가 필요하며, 이 통계는 Git에 포함하지 않습니다. 준비되지 않은 상태에서 지정하면 결과의 `errors`에 안내가 남습니다.

`--reference`가 바꾸는 경로는 reliability·covariance·dynamics 통계입니다. `--reliability-stats`, `--covariance-stats`, `--dynamics-stats`를 직접 지정하면 해당 경로가 우선합니다. 기본 각도·스켈레톤·motion·sequence 통계는 각각 `--pro-stats`, `--reference-skeletons`, `--motion-stats`, `--sequence-stats`로 지정합니다.

SwingNet 없이 연결만 확인하려면 다음 명령을 사용합니다. 실제 입력 영상과 자세 분석 환경은 필요하며, 결과의 `mock_events`는 `true`가 됩니다.

```powershell
pixi run --locked python -m pipeline.run_analysis --video $video --mock-events --output "outputs/swing_mock/"
```

## 피드백이 사용하는 정보

| 구분 | 내용 |
|---|---|
| 기본 자세 feature | `trail_elbow_angle`, `lead_elbow_angle`, `shoulder_tilt`, `hip_tilt`, `spine_tilt` |
| 추가 정보 | `xfactor`, 이벤트 간 각도 변화(`dynamics`), 백스윙/다운스윙 템포 |
| 동작 지표 | 어깨·골반 회전, 머리 움직임, 체중 이동 관련 2D 관절 지표 |
| 결합 평가 | feature 간 상관관계를 반영하는 Mahalanobis 기반 평가 |
| 일반 사용자 피드백 | 잘한 점, 우선 확인할 점, 참고 사항, 분석 신뢰도 안내 |

오른손잡이는 왼팔이 lead, 오른팔이 trail이며 왼손잡이는 반대입니다. `shoulder_tilt`와 `hip_tilt`는 원형 각도로 비교해 -180°/+180° 경계의 차이를 보정합니다.

기본 각도 통계는 평균·표준편차·중앙값·IQR·관측 수·검출률·누락 수와 reliability를 포함합니다. 각도 feature의 reliability 기준은 다음과 같습니다. 동작 지표에는 별도의 분산 기준이 적용됩니다.

| Reliability | 각도 통계 조건 | 피드백 강도 |
|---|---|---|
| `high` | 검출률 ≥ 0.85, 표준편차 ≤ 10° | `strong` |
| `medium` | 검출률 ≥ 0.70, 표준편차 ≤ 20° | `normal` |
| `low` | 나머지 | `reference` |

기준 스켈레톤은 이벤트별로 feature 통계에 가까운 실제 프로 프레임을 고르는 **feature-medoid** 방식입니다. 관절을 독립적으로 평균 낸 자세가 아닙니다. 현재 파이프라인은 이 기준과 사용자 스켈레톤의 비교 이미지를 생성합니다.

프로 통계와의 거리나 reliability는 **자세 교정의 정확도**를 뜻하지 않습니다. 정면 영상의 회전·체중 이동 지표는 2D 근사이며, 가림·이벤트 선택 오류·촬영 각도에 영향을 받습니다. 계산할 수 없는 항목은 `unavailable`로 표시합니다.

## 출력 결과

`--output`을 생략하면 `outputs/<영상이름>/`에 저장됩니다. 예를 들어 `10.mp4`는 `outputs/10/10_result.json`을 만듭니다.

`--output "outputs/swing_subject_phase/"`를 지정한 경우:

```text
outputs/swing_subject_phase/
  swing_subject_phase_result.json
  swing_subject_phase_summary.txt
  feedback_table.csv
  feedback_ko.txt
  feedback_user.txt
  feedback_user.json
  frames/
  skeletons/
```

- `*_result.json`: 이벤트 프레임·모델 점수·자세·feature·피드백·템포·동작 지표·오류와 실행 정책.
- `*_summary.txt`: 이벤트별 분석 요약.
- `feedback_table.csv`: 8개 이벤트 × 5개 기본 자세 feature의 등급 표.
- `feedback_ko.txt`: 상세 한국어 피드백, 템포·동작 등 추가 평가.
- `feedback_user.txt`, `feedback_user.json`: 일반 사용자를 위한 문장과 구조화된 피드백.
- `frames/`: 검출된 이벤트의 원본 프레임 이미지.
- `skeletons/`: 사용자 관절 overlay와 사용자/프로 기준 스켈레톤 비교 이미지.

누락·보류 이벤트가 있으면 해당 이미지가 없을 수 있습니다. `events`의 `frame_index`, `pose_detected`, `feedback`과 최상위 `errors`를 함께 확인합니다. 후보정의 골퍼·영역·모델 해시·장치 정보는 `event_detection_policy.refinement`에 기록됩니다.

기존 prefix 저장 방식도 지원합니다. 예를 들어 존재하지 않는 경로 `--output "outputs/custom_result"`를 주고 그 이름이 입력 영상 이름과 다르면 `outputs/custom_result.json`, `outputs/custom_result_frames/` 등의 형태로 저장됩니다. 폴더 저장을 확실히 지정하려면 끝에 `/`를 붙입니다.

분석 JSON에서 피드백만 다시 생성할 수 있습니다.

```powershell
pixi run --locked python -m pipeline.generate_feedback_report --analysis-json "outputs/swing_subject_phase/swing_subject_phase_result.json" --output-dir "outputs/swing_subject_phase/"
```

## 최근 실행 검증

2026-11-01, main의 `0ebb3f4`에서 GolfDB `10.mp4`로 실제 분석을 확인했습니다. 입력은 시각 검수를 통과한 정면 드라이버 영상이며 170프레임, 29.97fps, 1024×720입니다. 환경은 Windows CPU, PyTorch 2.10.0이며 CUDA는 사용할 수 없는 상태였습니다.

| 모드 | 실행 시간 | 이벤트 | 자세 검출 | 분석 오류 |
|---|---:|---:|---:|---|
| 기본 `none` | 21.46초 | 8/8 | 8/8 | 없음 |
| `subject-phase` | 40.30초 | 8/8 | 8/8 | 없음 |

두 모드 모두 종료코드 0, JSON·요약·한국어/사용자 피드백 생성을 확인했습니다. 각 모드의 이벤트·스켈레톤 이미지 16장이 정상적으로 열렸고, 후보정 Top·Finish 비교 이미지는 직접 확인했습니다. 시간은 해당 PC·영상의 관측값입니다. 이 검증은 **실행 정상 여부**에 대한 것이며 독립 정확도 평가나 Linux/CUDA 실행 검증은 아닙니다. 같은 main의 `tests/` 131개 통과도 확인했습니다.

실행 로그에는 torch 2.10 / torchvision 0.26 조합 경고와 MediaPipe 경고가 있었지만 위 두 분석은 완료됐습니다. 모든 경고가 무해하다는 뜻은 아니므로 실패 시 종료코드·로그·결과 JSON의 `errors`를 함께 확인합니다.

팀의 기존 로컬 데이터 배치가 저장소 옆 `src/golfdb-master/`에 있는 경우, 저장소 루트에서 다음과 같이 재현할 수 있습니다. 영상은 Git에 포함되지 않으므로 별도로 보유해야 합니다.

```powershell
$video = "../src/golfdb-master/learning_video/golfdb_720p_slice/10.mp4"
pixi run --locked python -m pipeline.run_analysis --video $video --event-refinement subject-phase --output "outputs/golfdb10_subject_phase/"
```

검증 원본과 생성 결과는 로컬 `outputs/runtime_check_golfdb10_20261101/`에 보관하며 Git에는 올리지 않습니다.

## 저장소 구성

| 경로 | 역할 |
|---|---|
| `pipeline/run_analysis.py` | 분석 CLI와 전체 파이프라인 |
| `pipeline/swingnet.py` | SwingNet 전처리·모델 로딩·기본 추론 |
| `pipeline/event_selection.py` | 이벤트 시간 순서·누락·보류 처리 |
| `pipeline/event_refinement.py`, `event_refinement_core.py`, `event_subject.py` | 골퍼 선택·고정 영역·문맥 점수·선형 후보정 |
| `pipeline/video_io.py` | 영상 프레임 읽기 |
| `pipeline/user_tracking.py`, `pose_tracking.py` | 사용자/기준 영상의 연속 자세 추적과 관절 값 복구 |
| `pipeline/qc_tracked_features.py`, `qc_temporal.py` | 관절 관측 품질과 시간축 QC |
| `pipeline/motion_metrics.py` | 관절 기반 동작 지표 계산·평가 |
| `pipeline/generate_feedback_report.py`, `user_feedback.py` | 표·상세 피드백·일반 사용자 피드백 |
| `pipeline/models/swingnet_phase_ranker_v1.json` | Git에 포함한 고정 후보정 모델 |
| `event_detection/model.py`, `MobileNetV2.py`, `dataloader.py` | SwingNet 모델·backbone·데이터 전처리 구성요소 |
| `data/pro_*.json` | 실행에 사용하는 프로 기준 통계·스켈레톤 |
| `tests/` | 통계·등급·추적·QC·분석 연결 회귀 테스트 |
| `pixi.toml`, `pixi.lock` | 실행 환경 정의·고정 버전 |
| `.gitignore`, `.gitattributes` | 로컬 산출물 제외·후보정 모델 바이트 보존 |

실행에 사용하는 기준 데이터는 다음과 같습니다.

| 파일 | 내용 |
|---|---|
| `data/pro_angle_stats.json` | 기본 각도 평균·표준편차, reliability 기준이 없을 때 사용하는 기준 |
| `data/pro_feature_stats_reliability.json` | feature별 분포·검출률·reliability |
| `data/pro_feature_covariance.json` | feature 결합 평가용 공분산/정밀도 |
| `data/pro_dynamics_stats.json` | 이벤트 간 각도 변화 통계 |
| `data/pro_sequence_stats.json` | 스윙 전체 템포 통계 |
| `data/pro_motion_stats.json` | 관절 기반 동작 지표 통계 |
| `data/pro_reference_skeletons.json` | 이벤트별 대표 프로 스켈레톤 |

영상, 대형 가중치, `output/`·`outputs/`, raw CSV, 개인 실험·학습 자료는 Git에 포함하지 않습니다. 과거의 `event_detection/train.py`, `event_detection/eval.py`, `event_detection/data/`, `pose_analysis/`, `golf_yb_yolopose/` 등은 현재 배포하는 실행 파일 목록에 포함되지 않습니다. 로컬에 남아 있더라도 일반 분석의 필수 파일은 아닙니다.

## 기준 통계 개발과 테스트

기준 통계는 팀의 GolfDB/Colab 실험을 바탕으로 만들었습니다. 기존 342개 slice 기반 통계는 새 자세 판별 신경망을 학습한 결과가 아니라 관측 feature의 분포입니다. **342개라는 과거 수집 수를 현재 시각 검수된 적격 영상 수로 해석하지 않습니다.**

일반 영상 분석은 저장소에 포함된 JSON을 그대로 사용하므로 아래 개발 작업이 필요하지 않습니다. 기준을 다시 만들려면 영상·GolfDB 라벨·시각 검수 기록을 별도로 준비하고, 단일 화면의 정면 드라이버로 확인된 입력만 선별해야 합니다. 클립과 라벨의 시간축도 확인해야 하며, 기존 추출 스크립트가 이 범위 검수를 자동 수행한다고 가정하면 안 됩니다.

| 개발 모듈 | 용도 |
|---|---|
| `pipeline.extract_pro_features` | 정답 이벤트 프레임의 raw feature CSV |
| `pipeline.extract_pro_features_tracked` | 연속 추적·관절 품질 정보가 포함된 CSV |
| `pipeline.extract_pro_landmarks` | 프로 이벤트 관절 좌표 추출 |
| `pipeline.augment_raw_features` | raw CSV의 xfactor 확장 |
| `pipeline.compute_reliability_stats` | reliability·공분산 통계 |
| `pipeline.compute_dynamics_stats` | 이벤트 간 변화 통계 |
| `pipeline.compute_sequence_features` | 템포 등 스윙 전체 기준 |
| `pipeline.compute_motion_stats` | 관절 기반 동작 기준 |
| `pipeline.build_reference_skeletons` | feature-medoid 기준 스켈레톤 |

각 모듈의 `--help`에서 입력·출력 옵션을 확인할 수 있습니다. 새 결과는 별도 로컬 경로에 저장하고 검증 후 기준 교체 여부를 결정합니다.

```powershell
pixi run --locked python -m pipeline.extract_pro_features --help
pixi run --locked python -m pipeline.compute_reliability_stats --help
pixi run --locked python -m pytest tests -q
```

`--mock-events` 실행이나 단위 테스트 통과는 실제 이벤트 정확도를 증명하지 않습니다. 이벤트 개선의 기존 GolfDB 결과도 반복 사용 데이터의 회귀 근거로 다루며, 다른 사람·촬영 환경에서의 일반화와 자세 피드백 정확도는 별도로 검증해야 합니다.
