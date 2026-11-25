# 서버 의존성 안내

서버에서 이 프로젝트를 돌리려면 무엇이 필요한지 한 곳에 모은 문서입니다. 서버 담당자가 설치와 배포를 준비할 때 보세요. 작성일 2026-11-24. **아래 "확인한 환경"은 개발 PC(Windows 10) 한 곳에서만 검증했습니다.** 리눅스와 GPU 서버는 검증하지 못했고, 그런 항목은 "미확인"으로 적었습니다.

## 한눈에

| 구분 | 필요한 것 | Git에 있나 |
|---|---|---|
| 패키지 환경 | Python 3.12 + [`pixi.toml`](../pixi.toml) / `pixi.lock`의 패키지 | 있음 |
| 이벤트 검출 가중치 | `event_detection/models/swingnet_1800.pth.tar` (63MB) | 없음 (직접 준비) |
| 기본 골퍼 추적 가중치 | 저장소 루트의 `yolov8n-pose.pt` (앱 API의 골퍼 선택·관절 관측) | 없음 ([준비 방법](../README.md#빠른-시작)) |
| 골퍼 검출 가중치 | `models/golfpose/yolov8x-pose.pt` (139MB) | 없음 (별도 배치) |
| 팔 관절 모델 | `models/golfpose/vitpose_huge.ts` (2.55GB) | 없음 (따로 배포) |
| 기준 통계 | `data/*.json`, `data/refined/*.json` | 있음 |

앱 API는 자동 분석과 재계산 때 `models/golfpose/`에서 두 가중치를 읽습니다. CLI에서는 `--pose-refiner golfpose`를 선택할 때 사용합니다. 앱에서 가중치가 누락되거나 보정이 실패하면 MediaPipe 결과를 사용하고 `result.feedback.meta.pose_refiner`에 이유를 기록합니다.

## 1. 패키지 환경

[`pixi.toml`](../pixi.toml)이 정의하고 `pixi.lock`이 버전을 고정합니다. 설치는 저장소 루트에서 `pixi install --locked`입니다.

| 패키지 | 버전 | 쓰임 |
|---|---|---|
| python | 3.12 | |
| numpy | >=1.26, <2 (확인: 1.26.4) | 2.x는 쓰지 않습니다 |
| pandas, scipy | 2.3.x, 1.16 이상 | 기준 통계 계산, QC |
| opencv | >=4.12, <5 (확인: 4.13.0) | 영상 읽기, 크롭 |
| pytorch, torchvision | `*` (확인: torch 2.10.0, **CPU 빌드**) | SwingNet, ViTPose-H 실행 |
| mediapipe | ==0.10.21 (PyPI) | 자세 추적의 뼈대 |
| ultralytics | >=8.4.60, <9 (확인: 8.4.60) | 골퍼 검출(YOLOv8x-pose), Finish 팔 관절 |
| imageio-ffmpeg | ==0.6.0 (PyPI) | 영상 입출력 |
| ffmpeg | >=8, <9 | API에서 ffprobe로 업로드 영상 검사 |
| fastapi, uvicorn, python-multipart | `pixi.lock`에 고정 | 분석 API 실행과 영상 업로드 |
| pytest, pytest-cov | 8.x, 5.x | 테스트용 |
| libgles, libegl | 리눅스에서만 | mediapipe가 요구 |

### GPU 서버에서 꼭 확인할 것

- **이 PC의 pixi 환경은 CPU 빌드 torch입니다.** `pytorch = "*"`가 GPU 서버에서도 CUDA 빌드로 잡힌다는 보장은 없습니다. 설치 후 아래로 확인하세요. `None`이면 CPU 빌드입니다.
  ```bash
  pixi run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```
  CUDA 빌드로 바꾸는 방법(pixi의 `system-requirements`나 `pytorch-gpu` 지정 등)은 서버 환경에 맞춰 정해야 하고, 여기서는 시험하지 않았습니다.
- **torch 버전.** `vitpose_huge.ts`는 torch 2.11.0+cu128에서 내보냈고, torch 2.10.0(CPU, Python 3.12)에서 로딩과 추론이 되는 것을 확인했습니다. 그 밖의 버전은 모릅니다. 서버 torch로 [`vitpose_server.md`](vitpose_server.md)의 동등성 검증을 반드시 돌리세요.
- torchvision이 torch와 짝이 맞지 않는다는 경고(`torchvision 0.26 / torch 2.10`)가 이 PC에서 나옵니다. 분석은 정상 동작했습니다.
- 개발 중에는 GPU용으로 별도 가상환경(Python 3.13, torch 2.11+cu128, ultralytics)을 따로 썼지만, **운영에는 필요하지 않습니다.** 운영은 pixi 환경 한 프로세스에서 MediaPipe, ultralytics, TorchScript ViTPose-H를 모두 부릅니다. mmpose, mmcv, mmengine은 필요 없습니다.

## 2. 모델과 가중치 파일

| 파일 | 크기 | 용도 | 준비 방법 |
|---|---|---|---|
| `event_detection/models/swingnet_1800.pth.tar` | 63MB | 이벤트 8개 검출(SwingNet) | [GolfDB 원본 저장소](https://github.com/wmcnally/golfdb)의 공개 체크포인트. 폴더가 없으면 만듭니다 |
| `models/golfpose/yolov8x-pose.pt` | 139MB | 골퍼 박스와 Finish 팔 관절 | `SnapshotModel.zip`의 파일을 배치 |
| `models/golfpose/vitpose_huge.ts` | 2.55GB | ViTPose-H(TorchScript) | 개발 PC에서 만든 파일을 서버에 복사. 만드는 법은 [`vitpose_server.md`](vitpose_server.md)와 `pipeline/vitpose_export.py`(mmpose 환경에서 한 번만) |
| `event_detection/models/mobilenet_v2.pth.tar` | 14MB | SwingNet 학습 때만 | **운영에 필요 없음** |

### 체크섬 (2026-11-24, 개발 PC 기준)

받은 파일이 온전한지 서버에서 `sha256sum <파일>`(Windows는 `Get-FileHash <파일>`)로 계산해 비교하세요. 바이트 수도 같이 적습니다.

| 파일 | 바이트 | SHA256 |
|---|---|---|
| `vitpose_huge.ts` | 2,549,838,458 | `cb13e88b854538a54c374d57033d83a435c16c3115fe87f7ee172235264a6dd7` |
| `swingnet_1800.pth.tar` | 63,280,059 | `6331e303a9e86d0c19f183899f958bf2a71cf5a7070d46899e25e1ac877b23d4` |
| `yolov8x-pose.pt` | 139,420,105 | `7a9bf952c2fc1c07d4a1da00dbb4cfab5ed71f89cbe8824d51a8f7a61cee5825` |

`yolov8x-pose.pt`를 ultralytics에서 별도로 내려받으면 ultralytics 버전에 따라 새 버전이 받아질 수 있고, 그러면 값이 다를 수 있습니다. 값이 달라도 모델이 정상 동작하면 문제는 아니며, 동일 결과가 필요하면 위 값의 파일을 직접 전달해 쓰세요. `vitpose_huge.ts`를 다시 내보내면(체크포인트 변경 등) 값이 바뀌므로 이 표도 갱신해야 합니다.

CLI에서 SwingNet과 ViTPose-H의 경로를 바꿀 때는 각각 `--checkpoint`, `--vitpose-model`을 사용합니다. YOLOv8x의 기본 경로는 `pipeline/golfer_detection.py`의 `YOLO_DEFAULT_WEIGHTS`에 지정돼 있습니다.

## 3. 기준 데이터 (Git에 포함)

| 경로 | 쓰는 옵션 | 내용 |
|---|---|---|
| `data/pro_*.json` | 기본(`--reference legacy`) | 기존 기준 통계(MediaPipe로 잰 값) |
| `data/refined/pro_feature_stats_reliability.json`, `pro_feature_covariance.json`, `pro_dynamics_stats.json` | `--reference refined` | 같은 프로 영상 342개를 ViTPose-H/YOLOv8x 팔 관절로 다시 잰 기준 |

새 모델(`--pose-refiner golfpose`)과 새 기준(`--reference refined`)은 함께 쓰도록 만들었습니다. 새 기준으로 바꾸면 채점이 더 엄격해져서 점수가 내려갑니다([`pose_benchmark.md`](pose_benchmark.md)의 섀도 비교). 어느 쪽을 기본으로 할지는 정해지지 않았습니다.

## 4. 실행 예

```bash
# 기본 (MediaPipe만, 가장 가벼움)
pixi run --locked python -m pipeline.run_analysis --video swing.mp4 --output outputs/swing/

# 팔 관절을 ViTPose-H(Finish는 YOLOv8x)로 보정 + 새 기준 통계
pixi run --locked python -m pipeline.run_analysis --video swing.mp4 --output outputs/swing/ \
    --pose-refiner golfpose --reference refined
```

앱이 읽는 결과는 `outputs/swing/feedback_app.json`입니다. `--pose-refiner`는 이벤트가 8개 모두 검출될 때만 동작합니다.

## 5. 하드웨어와 속도

개발 PC(RTX 3070 8GB, i5-13400F)에서 잰 값입니다. 서버에서는 재지 않았습니다.

| 실행 | 값 |
|---|---|
| ViTPose-H, GPU | 프레임당 약 0.14초, 5장 배치에서 GPU 메모리 최대 2.7GB(모델 약 2.5GB 포함) |
| ViTPose-H, CPU | 프레임당 약 1초 |
| 영상 1개의 모델 추론 | 40프레임(이벤트 8개 × ±2프레임): GPU 약 6초, CPU 약 40초 |
| 전체 분석(MediaPipe 추적 포함, CPU) | 기본 약 13초, `--pose-refiner` 켠 경우 약 60초 |

동시 요청 처리량, 모델 상주(메모리에 계속 올려 둠) 방식의 이득은 재지 않았습니다. 지금 코드는 분석을 부를 때마다 프로세스가 시작되고 모델을 처음 쓸 때 로딩합니다(2.55GB 파일 읽기 포함).

## 6. 라이선스 (출시 전 확인 필요)

| 구성요소 | 상태 |
|---|---|
| **GolfPose 체크포인트**(ViTPose-H) | **상용 사용 조건을 확인하지 못했습니다.** 학습 데이터는 EULA상 상용 모델 학습이 가능하다고 읽었지만 원본 재배포는 금지입니다 |
| **ultralytics**(YOLOv8) | **AGPL-3.0.** 서버로 서비스를 제공하면 소스 공개 의무가 생길 수 있어 검토가 필요합니다 |
| mmpose 코드를 옮긴 전·후처리 (`pipeline/vitpose_runtime.py`) | Apache-2.0 |
| mediapipe, opencv, torch | 일반적으로 Apache-2.0 / BSD 계열이나 이 프로젝트에서 따로 확인하지 않았습니다 |
| SwingNet(GolfDB) 가중치 | 공개 체크포인트이며 사용 조건은 확인하지 않았습니다 |

## 7. 설치 후 점검 순서

1. `pixi install --locked` 후 `pixi run --locked python -m pytest tests -q`. 이 PC에서는 전부 통과했습니다.
2. 위 "GPU 서버에서 꼭 확인할 것"의 torch 확인 명령.
3. 모델 파일을 두고 [`vitpose_server.md`](vitpose_server.md)의 `vitpose_verify.py` 동등성 검증(기준 캐시와 라벨 영상이 있는 환경에서).
4. 영상 한 개로 두 실행 예를 돌려 `feedback_app.json`이 나오는지, `--pose-refiner` 쪽 결과 JSON의 `pose_refiner.status`가 실패 메시지가 아닌지 확인.

## 8. 아직 모르는 것

- 리눅스 동작(`pixi.toml`에 `linux-64`와 mediapipe용 `libgles`·`libegl`을 적어 두었지만 실제로 돌려 보지 못했습니다).
- 서버 GPU에서의 처리량과 동시 요청, 서버 torch 버전과의 호환.
- 위 라이선스 두 건.
