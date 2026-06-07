# golf-posture

골프 스윙 영상을 입력하면 GolfDB/SwingNet으로 8개 스윙 이벤트를 찾고, 각 이벤트 프레임에서 pose landmark를 추출한 뒤 프로 선수 기준 feature와 비교해 자세 피드백을 생성하는 졸업 프로젝트입니다.

현재 MVP는 학습이 아니라 **inference 파이프라인**과 **프로 공통 feature 기반 피드백 기준 개선**에 초점을 둡니다.

## 프로젝트가 하는 일

입력 영상 한 개를 다음 순서로 분석합니다.

```text
사용자 스윙 영상
-> SwingNet 이벤트 검출
-> 8개 이벤트 프레임 추출
-> MediaPipe Pose 또는 YOLO Pose 관절 추출
-> 각도 feature 계산
-> 프로 기준 통계와 비교
-> JSON / 텍스트 피드백 저장
```

검출하는 8개 이벤트는 다음과 같습니다.

```text
0 Address
1 Toe-up
2 Mid-backswing
3 Top
4 Mid-downswing
5 Impact
6 Mid-follow-through
7 Finish
```

MVP에서 사용하는 필수 feature는 다음 5개입니다.

```text
trail_elbow_angle
lead_elbow_angle
shoulder_tilt
hip_tilt
spine_tilt
```

기본 handedness는 오른손잡이입니다.

```text
오른손잡이: lead arm = left arm, trail arm = right arm
왼손잡이: lead arm = right arm, trail arm = left arm
```

## 핵심 개선점

기존에는 프로 선수 평균/표준편차와 사용자 값을 단순 비교했습니다. 하지만 프로마다 자세 차이가 있고, 어떤 feature는 공통성이 낮아 강한 교정 기준으로 쓰기 어렵습니다.

그래서 342개 GolfDB 720p slice 영상에서 event-feature별 raw 값을 다시 추출하고, 다음 통계를 계산했습니다.

```text
mean
std
median
IQR
n
detection_rate
missing_count
reliability
feedback_strength
```

특히 `shoulder_tilt`, `hip_tilt`는 -180도/+180도 경계에서 값이 튀는 circular angle 문제를 보정합니다. 예를 들어 178도와 -175도는 실제로 비슷한 방향이므로 353도 차이가 아니라 약 7도 차이로 처리합니다.

reliability 기준은 다음과 같습니다.

```text
high   = detection_rate >= 0.85 and std <= 10 degrees
medium = detection_rate >= 0.70 and std <= 20 degrees
low    = otherwise
```

피드백 강도는 reliability에 따라 달라집니다.

```text
high   -> strong feedback
medium -> normal feedback
low    -> reference only
```

현재 분석 결과에서는 `spine_tilt`가 대부분의 이벤트에서 high reliability로 나타났고, `Mid-downswing / hip_tilt`도 high reliability입니다. 반대로 `Top`, `Finish`의 팔꿈치/어깨/골반 일부 feature는 개인차가 커서 참고 지표로 다루는 것이 안전합니다.

## 실행 환경

이 저장소는 `pixi`로 Python 환경을 관리합니다. 현재 `linux-64`, `win-64`를 모두 지원하도록 설정되어 있습니다.

필수 환경:

```text
Python 3.12
pixi
OpenCV
PyTorch / torchvision
MediaPipe
Ultralytics YOLO
```

설치:

```bash
pixi install
```

실행:

```bash
pixi run python -m pipeline.run_analysis --help
```

## 모델 가중치

모델 가중치는 git에 포함하지 않습니다. 아래 위치에 직접 배치해야 합니다.

```text
event_detection/
  mobilenet_v2.pth.tar
  models/
    swingnet_1800.pth.tar
```

역할:

```text
mobilenet_v2.pth.tar
-> SwingNet 내부 CNN backbone 초기화에 사용

swingnet_1800.pth.tar
-> GolfDB 기반 8개 스윙 이벤트 검출에 사용
```

원래 README에서 사용하던 참고 링크:

- MobileNetV2: https://github.com/tonylins/pytorch-mobilenet-v2
- SwingNet checkpoint: https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view

## 사용자 영상 분석 실행

기본 실행:

```bash
pixi run python -m pipeline.run_analysis \
  --video path/to/sample.mp4 \
  --output outputs/sample_result \
  --handedness right \
  --pose-backend mediapipe
```

SwingNet checkpoint를 명시하는 경우:

```bash
pixi run python -m pipeline.run_analysis \
  --video path/to/sample.mp4 \
  --checkpoint event_detection/models/swingnet_1800.pth.tar \
  --pro-stats data/pro_angle_stats.json \
  --reliability-stats data/pro_feature_stats_reliability.json \
  --output outputs/sample_result \
  --handedness right \
  --pose-backend hybrid
```

checkpoint 없이 통합 테스트만 하는 경우:

```bash
pixi run python -m pipeline.run_analysis \
  --video path/to/sample.mp4 \
  --output outputs/sample_mock \
  --handedness right \
  --mock-events
```

`--mock-events`를 사용하면 실제 SwingNet 검출 결과가 아니라 균등 분할된 테스트 이벤트를 사용합니다. 이 경우 결과 JSON에 `"mock_events": true`가 기록됩니다.

## 출력 결과

`--output outputs/sample_result`로 실행하면 다음이 생성됩니다.

```text
outputs/sample_result.json
outputs/sample_result.txt
outputs/sample_result_frames/
```

JSON에는 이벤트별로 다음 정보가 들어갑니다.

```json
{
  "video_path": "...",
  "mock_events": false,
  "handedness": "right",
  "events": {
    "Address": {
      "frame_index": 14,
      "pose_detected": true,
      "selected_pose_backend": "mediapipe",
      "features": {
        "trail_elbow_angle": 166.6,
        "lead_elbow_angle": 176.0,
        "shoulder_tilt": -173.7,
        "hip_tilt": 178.2,
        "spine_tilt": -1.6
      },
      "feedback": {
        "spine_tilt": {
          "value": -1.6,
          "pro_mean": 3.1,
          "pro_std": 4.4,
          "z_score": -1.07,
          "grade": "Good",
          "reliability": "high",
          "feedback_strength": "strong",
          "weighted_score": 1.07,
          "message": "..."
        }
      }
    }
  }
}
```

## 프로 feature reliability 재생성

342개 GolfDB 720p slice 영상에서 raw feature CSV를 다시 생성하려면:

```bash
pixi run python -m pipeline.extract_pro_features \
  --videos-dir "C:\Users\zzang\Desktop\workspace\Python\golf\src\golfdb-master\learning_video\golfdb_720p_slice" \
  --labels event_detection\data\golfDB_labels.csv \
  --output data\pro_feature_raw.csv \
  --pose-backend mediapipe \
  --handedness right
```

raw CSV에서 reliability 통계를 계산하려면:

```bash
pixi run python -m pipeline.compute_reliability_stats \
  --raw-csv data\pro_feature_raw.csv \
  --output-json data\pro_feature_stats_reliability.json \
  --stats-csv data\pro_feature_stats_reliability.csv \
  --summary outputs\pro_feature_reliability_summary.txt
```

## 파일별 설명

### 루트 파일

```text
README.md
```

프로젝트 목적, 실행 방법, 파일 구조, 모델 가중치 배치 방법을 설명하는 문서입니다.

```text
pixi.toml
```

pixi 환경 설정 파일입니다. Python, OpenCV, PyTorch, MediaPipe, Ultralytics 의존성을 정의합니다. Windows와 WSL/Linux 환경을 모두 대상으로 합니다.

```text
pixi.lock
```

pixi가 실제로 해결한 패키지 버전을 고정하는 lock 파일입니다. 팀원 환경에서 같은 버전 조합을 재현하는 데 사용합니다.

```text
.gitignore
```

모델 가중치, 영상, 이미지, pixi 환경, 출력 결과, pose debug overlay처럼 git에 올리면 안 되는 파일을 제외합니다.

### pipeline

```text
pipeline/run_analysis.py
```

현재 MVP의 메인 entrypoint입니다. 사용자 영상을 입력받아 SwingNet 이벤트 검출, pose 추출, feature 계산, 프로 기준 비교, reliability 반영 피드백 저장까지 수행합니다.

지원 pose backend:

```text
mediapipe
yolopose
auto
hybrid
```

`hybrid`는 이벤트별로 계산 가능한 feature 수가 더 많은 backend를 선택합니다.

```text
pipeline/extract_pro_features.py
```

342개 GolfDB slice 영상에서 이벤트별 raw feature를 추출해 CSV로 저장합니다. 각 row는 특정 영상의 특정 이벤트 프레임에서 계산된 5개 feature와 누락 여부를 담습니다.

```text
pipeline/compute_reliability_stats.py
```

`pro_feature_raw.csv`를 읽어 event-feature별 통계를 계산합니다. circular angle 보정, detection_rate 계산, reliability 분류, 요약 파일 생성을 담당합니다.

```text
pipeline/__init__.py
```

`pipeline`을 Python module로 실행하기 위한 패키지 초기화 파일입니다.

### data

```text
data/pro_angle_stats.json
```

기존 실험에서 만든 프로 기준 angle 통계입니다. event-feature별 `mean`, `std`, `n`을 포함합니다. reliability 파일이 없을 때 fallback 기준으로 사용할 수 있습니다.

```text
data/pro_feature_raw.csv
```

342개 프로 slice 영상에서 새로 추출한 raw feature 데이터입니다. 총 2736 rows이며, 342개 영상 x 8개 이벤트 구조입니다.

주요 컬럼:

```text
video_id
video_path
event_index
event_name
frame_index
pose_backend
pose_detected
trail_elbow_angle
lead_elbow_angle
shoulder_tilt
hip_tilt
spine_tilt
missing_features
failure_reason
```

```text
data/pro_feature_stats_reliability.json
```

현재 피드백에서 우선 사용하는 개선된 프로 기준 통계입니다. `mean`, `std`, `median`, `IQR`, `detection_rate`, `reliability`, `feedback_strength`를 포함합니다.

```text
data/pro_feature_stats_reliability.csv
```

`pro_feature_stats_reliability.json`과 같은 내용을 표 형태로 확인하기 위한 CSV입니다.

### event_detection

```text
event_detection/model.py
```

SwingNet 이벤트 검출 모델 정의입니다. MobileNetV2 backbone과 LSTM을 사용합니다. 현재는 Windows/CPU 환경에서도 checkpoint를 읽을 수 있도록 `map_location="cpu"`와 파일 기준 상대 경로를 적용했습니다.

```text
event_detection/MobileNetV2.py
```

SwingNet의 CNN backbone으로 사용하는 MobileNetV2 구현입니다.

```text
event_detection/dataloader.py
```

GolfDB 학습/평가용 데이터를 읽고 전처리하는 dataloader입니다.

```text
event_detection/train.py
```

SwingNet 학습 스크립트입니다. 현재 MVP 범위에서는 재학습하지 않습니다.

```text
event_detection/eval.py
```

학습된 SwingNet checkpoint를 평가하는 스크립트입니다.

```text
event_detection/test_video.py
```

로컬 테스트 영상에 대해 SwingNet 이벤트 검출을 실행하는 기존 테스트 스크립트입니다.

```text
event_detection/test_video_colab.py
```

Colab 환경에서 테스트 영상을 실행하기 위한 기존 스크립트입니다.

```text
event_detection/test_youtube_colab.py
```

Colab에서 YouTube 또는 외부 영상을 복사해 SwingNet 이벤트 검출을 테스트하는 스크립트입니다.

```text
event_detection/util.py
```

이벤트 검출 결과 처리와 보조 계산에 사용하는 utility 함수입니다.

```text
event_detection/data/golfDB_labels.csv
```

GolfDB annotation을 CSV로 정리한 파일입니다. 각 영상의 10개 raw frame annotation을 포함하며, 가운데 8개가 스윙 이벤트로 사용됩니다.

```text
event_detection/data/golfDB.mat
```

GolfDB 원본 annotation 데이터입니다.

```text
event_detection/data/generate_splits.py
```

GolfDB train/validation split을 생성하는 기존 스크립트입니다.

```text
event_detection/data/preprocess_videos.py
```

GolfDB 영상을 학습/평가용으로 전처리하는 기존 스크립트입니다.

```text
event_detection/scripts/inspect_golfdb_labels.py
```

GolfDB label CSV 구조와 annotation을 확인하기 위한 보조 스크립트입니다.

### pose_analysis

```text
pose_analysis/scripts/extract_mediapipe_pose_overlays.py
```

GolfDB 이벤트 프레임에 MediaPipe pose skeleton을 overlay하는 실험용 스크립트입니다. 프로젝트에서 쓰는 13개 주요 landmark만 표시하도록 구성되어 있습니다.

생성 결과는 기본적으로 `pose_analysis/debug_overlays/`에 저장되며, 이 폴더는 git에 포함하지 않습니다.

```text
pose_analysis/models/pose_landmarker_full.task
```

MediaPipe Pose Landmarker 모델 파일입니다. 모델 파일은 git에 포함하지 않습니다.

```text
pose_analysis/pose_landmarks_sample.csv
```

pose overlay 실험 중 추출한 landmark sample CSV입니다. 생성 산출물이므로 git에 포함하지 않습니다.

### golf_yb_yolopose

```text
golf_yb_yolopose/yolopose_skeleton.py
```

팀원이 작성한 YOLO Pose skeleton overlay 비교용 코드입니다. 현재 MVP 파이프라인과는 별도 실험 코드입니다.

```text
golf_yb_yolopose/yolopose_data_label.py
```

YOLO Pose 기반 데이터 라벨링 또는 skeleton 실험을 위한 보조 코드입니다.

### outputs

```text
outputs/
```

분석 결과 JSON, 텍스트 요약, 이벤트 프레임 이미지, reliability summary가 저장되는 폴더입니다. 실행 결과 산출물이므로 git에 포함하지 않습니다.

## 현재 검증된 명령

문법 확인:

```bash
pixi run python -m py_compile \
  pipeline/run_analysis.py \
  pipeline/extract_pro_features.py \
  pipeline/compute_reliability_stats.py
```

프로 raw feature 생성:

```bash
pixi run python -m pipeline.extract_pro_features \
  --videos-dir "C:\Users\zzang\Desktop\workspace\Python\golf\src\golfdb-master\learning_video\golfdb_720p_slice" \
  --labels event_detection\data\golfDB_labels.csv \
  --output data\pro_feature_raw.csv \
  --pose-backend mediapipe \
  --handedness right
```

reliability 통계 생성:

```bash
pixi run python -m pipeline.compute_reliability_stats \
  --raw-csv data\pro_feature_raw.csv \
  --output-json data\pro_feature_stats_reliability.json \
  --stats-csv data\pro_feature_stats_reliability.csv \
  --summary outputs\pro_feature_reliability_summary.txt
```

사용자 영상 분석:

```bash
pixi run python -m pipeline.run_analysis \
  --video path/to/sample.mp4 \
  --pro-stats data\pro_angle_stats.json \
  --reliability-stats data\pro_feature_stats_reliability.json \
  --output outputs/sample_result \
  --handedness right \
  --pose-backend hybrid
```

## 개발 범위와 주의점

- 이번 MVP는 SwingNet 학습 또는 pose 모델 학습을 하지 않습니다.
- SwingNet checkpoint가 없으면 실제 이벤트 검출은 실행할 수 없습니다.
- `--mock-events`는 통합 테스트용이며 실제 이벤트 검출 결과로 해석하면 안 됩니다.
- YOLO Pose는 사용자 영상 분석에서 사용할 수 있지만, 현재 reliability 기준 데이터는 MediaPipe 기반 raw feature로 생성되었습니다.
- 영상 파일, 모델 가중치, overlay 이미지, 실행 출력은 git에 올리지 않습니다.
