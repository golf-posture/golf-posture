# ViTPose-H 서버 실행 (mmpose 없이)

GolfPose ViTPose-H(`person`)를 서버에서 쓰기 위한 실행 방법입니다. 모델을 TorchScript 파일 하나로 내보내서 **mmpose, mmcv, mmengine, 임시 shim 없이 `torch`, `numpy`, `opencv`만으로** 돌립니다. 정확도 근거는 [pose_benchmark.md](pose_benchmark.md)에 있습니다. 서버 전체 의존성(패키지, 모델 파일, 라이선스)은 [server_dependencies.md](server_dependencies.md)를 보세요. 작성일 2026-11-24.

## 구성

| 항목 | 위치 | 비고 |
|---|---|---|
| 모델 파일 `vitpose_huge.ts` | `outputs/golfpose/` (Git 제외) | 2.55GB, 서버에 따로 배포해야 합니다 |
| 전처리·후처리 | `pipeline/vitpose_runtime.py` | numpy와 OpenCV만 사용 (mmpose 1.3.2 코드를 옮김, Apache-2.0) |
| 내보내기 | `pipeline/vitpose_export.py` | mmpose 환경에서 **한 번만** 실행. 체크포인트가 바뀔 때만 다시 |
| 동등성 검증 | `pipeline/vitpose_verify.py` | mmpose 없이 실행. 서버에서 설치 확인용으로 씁니다 |

입력은 `(N, 3, 256, 192)` float BGR 0..255 크롭(`vitpose_runtime.model_input`), 출력은 flip test가 적용된 히트맵 `(N, 17, 64, 48)`입니다. BGR→RGB, 정규화, flip test는 모델 안에 들어 있습니다. 키포인트 순서는 COCO가 아니라 Human3.6M입니다([`golfpose_joints.py`](../pipeline/golfpose_joints.py)).

## 서버 요구사항

- Python 3.12 또는 3.13, `torch`, `numpy`, `opencv-python`. **내보낼 때 torch 2.11.0+cu128을 썼고, torch 2.10.0(CPU, Python 3.12)에서 로딩과 추론이 되는 것을 확인했습니다.** 다른 버전은 확인하지 않았으니 서버의 torch로 아래 검증을 반드시 돌려야 합니다.
- 속도가 필요하면 NVIDIA GPU와 CUDA 빌드 torch. GPU 메모리는 모델 약 2.5GB에 활성값을 더해 5프레임 배치에서 최대 2.7GB였습니다. 이 프로젝트의 pixi 환경은 이 PC에서 CPU 빌드가 잡혀서, GPU 서버에서는 CUDA 빌드가 설치되는지 확인해야 합니다.
- 디스크 2.55GB(모델 파일). 모델 파일은 저장소에 없습니다.
- 리눅스는 검증하지 못했습니다(이 문서의 모든 측정은 Windows 10에서 했습니다).

## 설치 확인(동등성 검증)

기준 예측 캐시와 박스 파일(`outputs/golfpose/pred_golfpose_vitpose_gpu.json`, `boxes.json`)과 라벨 영상이 있는 환경에서:

```powershell
python pipeline/vitpose_verify.py --model outputs/golfpose/vitpose_huge.ts `
    --reference outputs/golfpose/pred_golfpose_vitpose_gpu.json --device cuda:0
```

기대 결과(2026-11-24, 466프레임, 3,728개 관절):

| 환경 | 위치 차이 | 비고 |
|---|---|---|
| GPU, torch 2.11 | 평균 0.0000px, **최대 0.0008px** | 벤치마크 지표가 mmpose와 같음(틀린 팔 6.3%, Top 17.4%, Finish 32.3%) |
| CPU, torch 2.10, Python 3.12 | 팔 길이 대비 중앙값 0.1%, 99번째 백분위 1.8%, 최대 7.5% | 60프레임 표본. 큰 차이는 점수가 낮은 불확실한 관절에서만 났고, 팔 길이 5% 초과는 0.28% |

GPU에서 위치 차이가 0.01px를 넘거나 지표가 달라지면 torch 버전이나 CUDA 문제를 먼저 의심하세요.

## 속도(이 PC: RTX 3070, i5-13400F 16스레드)

| 실행 | 처리량 |
|---|---|
| GPU, 단일 크롭 | 466프레임 63초 (영상 읽기와 로딩 포함, 프레임당 약 0.14초) |
| GPU, 5장 배치 | 한 장씩과 히트맵 차이 최대 0.0003 |
| CPU | 60프레임 64초 (프레임당 약 1초) |

영상 1개당 이벤트 8개 × ±2프레임이면 40프레임이므로, GPU는 약 6초, CPU는 약 40초입니다. 운영 서버에서의 처리량과 동시 요청은 재지 않았습니다.

## 주의

- **라이선스 미확인.** GolfPose 체크포인트의 상용 사용 조건을 확인하지 못했습니다. 상용 서비스에 쓰기 전에 확인이 필요합니다.
- 히트맵이 전부 0이면 디코딩에서 0으로 나누는 경고가 나고 좌표가 NaN이 됩니다(mmpose 원본과 같은 동작). 호출하는 쪽에서 점수 0인 관절을 걸러야 합니다.
- 내보낸 파일에는 장치가 고정되어 있지 않습니다(처음 내보냈을 때 `cuda:0`이 박혀 CPU에서 실패했고, 내보내기 코드를 고쳐 해결했습니다). 다시 내보낼 때는 `vitpose_export.py`를 그대로 쓰세요.
- 체크포인트를 바꾸면 `vitpose_export.py`로 다시 내보낸 뒤 위 검증을 다시 해야 합니다.
