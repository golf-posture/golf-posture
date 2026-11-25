# golf-posture

골프 스윙 영상에서 SwingNet으로 8개 이벤트를 검출하고, 자세와 동작을 프로 기준 통계와 비교해 한국어 피드백을 생성하는 Python 분석 파이프라인입니다.

현재 지원 대상으로 삼는 입력은 **정면(face-on), 단일 화면, 드라이버 한 스윙 영상**입니다. CLI 분석은 MediaPipe 연속 추적과 QC를 사용하며, 이벤트 검출 개선 기능은 `--event-refinement subject-phase`로 선택합니다(CLI 기본값 `none`). 앱 연결용 이벤트 API는 `subject-phase`를 사용합니다.

## 빠른 시작

아래 명령은 PowerShell 기준입니다. 분석 명령은 항상 저장소 루트에서 실행합니다.

앱을 연결하려면 1·2단계를 마치고 [앱 연결용 이벤트 API](#앱-연결용-이벤트-api)를 실행하세요.

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
| `yolov8n-pose.pt` | 골퍼 선택·관절 관측, YOLO Pose | 앱 API, CLI `subject-phase` 또는 YOLO Pose 사용 시 필요 |
| `models/golfpose/vitpose_huge.ts`, `models/golfpose/yolov8x-pose.pt` | 이벤트 주변 팔 관절 보정 | 앱 자동 분석·프레임 재계산에서 GolfPose 사용 시 필요 |
| `pipeline/models/swingnet_phase_ranker_v1.json` | 이벤트 후보정 모델 | `subject-phase`에 필요하며 **Git에 포함됨** |

SwingNet 가중치는 [GolfDB 원본 저장소](https://github.com/wmcnally/golfdb)의 공개 [체크포인트](https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view)를 사용합니다. `event_detection/models/` 폴더가 없으면 생성합니다.

YOLO Pose 가중치는 저장소 루트에서 다음 명령으로 준비할 수 있습니다. 최초 다운로드에는 인터넷 연결이 필요합니다.

```powershell
pixi run --locked python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

완성된 SwingNet 체크포인트를 읽을 때는 `mobilenet_v2.pth.tar`가 필요하지 않습니다. `EventDetector(pretrain=True)`를 직접 사용하는 별도 학습 작업에만 필요합니다.

`subject-phase`는 SwingNet 체크포인트와 후보정 JSON의 SHA256을 [event_refinement.py](pipeline/event_refinement.py)의 값과 비교합니다. 이는 검증된 모델과 파일 내용이 같은지 확인하는 검사이며, 불일치하면 실행을 중단합니다. 후보정 JSON의 바이트는 [.gitattributes](.gitattributes)로 보존하므로 자동 줄바꿈 변환이나 재저장을 피합니다.

### 3. CLI로 영상 분석

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

## 앱 연결용 이벤트 API

위 가중치를 준비한 뒤 저장소 루트에서 서버를 실행합니다. iPhone과 Expo 연결 방법은 [KUGOLF 앱 README](https://github.com/golf-posture/kugolf-app#readme)를 참고하세요.

```powershell
pixi run --locked start
```

`start`는 `pixi.toml`에 등록된 API 실행 명령입니다. 기존과 같이 `0.0.0.0:8000`에서 worker 1개로 실행하며 자동 재시작은 사용하지 않습니다. 종료할 때는 `Ctrl+C`를 누릅니다.

Expo와 백엔드를 같은 PC에서 실행하면 앱은 Expo PC의 LAN IPv4 주소와 `8000` 포트로 자동 연결합니다. 앱의 `EXPO_PUBLIC_API_URL`은 비워 두고, 다른 PC·배포 서버·API 터널을 사용할 때만 직접 지정합니다. PC에서 상태 확인은 `http://localhost:8000/health`, API 명세는 `http://localhost:8000/docs`입니다. 이 서버는 같은 네트워크에서 확인하는 로컬 데모용입니다.

| 요청 | 역할 |
|---|---|
| `POST /event-jobs` | 업로드 전에 작업 ID 발급 |
| `PUT /event-jobs/{id}/video` | multipart의 `file`로 영상 한 개 업로드. 수동 재계산은 `event_frames` 필드에 이벤트 ID 순서의 JSON 배열도 전송 |
| `GET /event-jobs/{id}` | 상태·현재 단계·처리 프레임 수·완료 결과 조회 |
| `POST /event-jobs/{id}/cancel` | 작업 취소 요청; `cancelled`가 실제 정리 완료 상태 |

- MP4/MOV, 최대 30초·200MB(200,000,000바이트). 추론은 한 번에 한 작업만 실행하며 다음 작업은 대기합니다.
- API 내부에서 `subject-phase`를 사용합니다. 골퍼 중심 고정 영역·중심 64프레임 문맥·고정된 선형 후보정과 시간 순서를 적용하며, 실행 명령과 CLI 기본값 `none`은 그대로입니다. 확정된 이벤트 프레임에서 MediaPipe로 관절을 추적하고 팀원 채점 기준으로 자세와 리듬을 평가합니다. 완료 응답의 `result.feedback`는 [앱 피드백 v3](docs/app_feedback_schema.md) 형식이며 자동 분석과 수동 재계산에서 같습니다. 현재 앱 입력은 오른손잡이 기준입니다.
- 앱의 자동 분석과 프레임 재계산은 같은 GolfPose 팔 보정 모델을 사용합니다. [서버 의존성 문서](docs/server_dependencies.md)의 체크섬과 일치하는 `vitpose_huge.ts`와 `yolov8x-pose.pt`를 `SnapshotModel.zip`에서 꺼내 저장소의 `models/golfpose/`에 함께 배치하세요. 8개 이벤트의 ±2프레임에서 양팔을 보정하고 피니시에는 YOLOv8x를 사용합니다. 보정된 프레임이 있으면 `data/refined/`의 기준으로 채점합니다. 이벤트 누락, 모델 파일 누락, 추론 실패 또는 보정 프레임 0개일 때 앱은 기존 MediaPipe·legacy 채점을 사용합니다. 완료 응답의 `result.feedback.meta.pose_refiner`에서 적용 상태, 기준 통계, 실제 보정 프레임 수와 보정된 이벤트 위치(0부터 시작)를 확인할 수 있습니다. ViTPose-H 가중치는 서버 프로세스에서 재사용합니다. 사용자 영상의 관절 좌표는 작업마다 다시 계산합니다.
- `locating_golfer`(골퍼 관측), `extracting_features`(CNN 특징 추출), `detecting`(중심 문맥 점수 계산)의 진행률은 각각 `processed_frames / total_frames`입니다. 단계마다 0부터 시작하며 전체 예상 시간의 비율이 아닙니다. 준비·골퍼 영역 설정·후보정·자세 피드백 계산(`feedback`)·결과 정리에는 퍼센트가 없고, 전체 완료는 `status=completed`로 판단합니다.
- 결과의 `events`는 8개 ID(0~7), 0기준 `frame_index`, 초 단위 `time_seconds`, `model_score`, `status`를 제공합니다. 재생 시각은 원본 프레임의 표시 구간 안쪽이며 가변 FPS도 실제 타임스탬프로 계산합니다. 모델 점수는 정확도 백분율이 아닙니다.
- 최상위 `frame_times`는 관절 검출 여부와 관계없이 프레임별 시작 시각을 초 단위로 제공합니다. 수동 재계산은 같은 영상을 다시 읽고 8개 이벤트의 지정 프레임에서 재생 시각을 구합니다. 각 값은 영상 범위의 0기준 정수 또는 미지정 `null`이며 이벤트 간 역순과 중복을 허용합니다. 이 경로는 자동 검출을 재실행하지 않으며 수동 결과의 `model_score`는 `null`입니다.
- 결과의 `skeleton`은 골퍼 선택에서 이미 추출한 COCO17 관절을 추가 추론 없이 제공합니다. 원본 영상 크기(`width`, `height`), 실제 프레임 시작 시각(`frame_times`), 프레임별 정규화 좌표(`frames`, 관절당 `[x, y]`)를 포함하며 불확실한 관절은 `null`입니다. 유효 관절이 전혀 없으면 `skeleton=null`입니다. 앱은 풀스윙 영상 왼쪽 상단에서 표시를 켜고 끌 수 있습니다.
- 취소 시 현재 실행 중인 모델 연산은 끝날 수 있지만 다음 추론 묶음은 시작하지 않습니다. `cancelling` 이후 파일·디코더 정리가 끝나야 `cancelled`가 됩니다. 이미 완료된 작업에 대한 취소는 완료 상태를 유지합니다.
- 영상 임시 파일은 완료·실패·취소 후 삭제합니다. 결과와 사용하지 않은 작업 ID는 30분간 메모리에 보관하며 서버 재시작 후 복구되지 않습니다. **서버 worker는 1개로 유지합니다.**

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

## 앱 분석 검증

Windows CPU에서 같은 정면 드라이버 영상(GolfDB 10번)으로 자동 분석과 수동 재계산을 확인했습니다. 아래 프레임 번호는 앱에 표시되는 1기준 값입니다.

| 입력 | GolfPose 보정 프레임 수 | 자세 점수 | 교정 / 칭찬 |
|---|---:|---:|---:|
| 자동 분석 · 탑 105 | 35 | 8.7 / 10 | 0 / 5 |
| 탑을 113으로 수정 후 재계산 | 31 | 7.3 / 10 | 5 / 1 |

두 요청 모두 `result.feedback.meta.pose_refiner.status`가 `applied`였습니다. 앱 API는 `output/`과 `outputs/` 디렉터리가 없는 환경에서도 같은 피드백을 생성했습니다.

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
| `api/`, `pipeline/event_analysis.py`, `pipeline/event_feedback.py` | 영상 업로드 작업, 자동 분석·수동 재계산과 앱 피드백 |
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
| `data/refined/` | `--reference refined`용 기준 통계(신뢰도·공분산·동역학). 프로 영상을 ViTPose-H/YOLOv8x 팔 관절로 다시 잰 값 |

`--pose-refiner golfpose`는 이벤트 주변의 팔 관절(어깨·팔꿈치·손목)만 ViTPose-H(Finish는 YOLOv8x)로 바꿔 정확도를 높이는 선택 기능입니다(기본 꺼짐, 서버용, 모델 파일은 Git 제외). 새 모델과 함께 쓸 때는 `--reference refined`를 같이 주는 것을 전제로 만들었습니다. 측정 근거, QC 재보정, 통계 재생성 순서는 [docs/pose_benchmark.md](docs/pose_benchmark.md), 서버 의존성(패키지, 모델 파일, 라이선스)은 [docs/server_dependencies.md](docs/server_dependencies.md), ViTPose-H 실행은 [docs/vitpose_server.md](docs/vitpose_server.md)에 있습니다. CLI의 새 경로는 Top/Finish 좌우 보정을 사용하지 않으며, 앱의 legacy 대체 경로는 이전 앱 기준을 유지합니다.

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
