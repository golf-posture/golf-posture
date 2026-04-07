# golf-posture

## WSL 환경 설정

이 프로젝트는 `WSL2 + Ubuntu` 환경에서 실행하는 것을 기준으로 합니다.

### 1. WSL 설치

PowerShell에서 아래 명령을 실행합니다.

```powershell
wsl --install -d Ubuntu
```

설치가 끝나면 재부팅하고, Ubuntu를 처음 실행해서 Linux 사용자 이름과 비밀번호를 설정합니다.

### 2. WSL 버전 확인

PowerShell에서 아래 명령으로 Ubuntu가 WSL2로 설치되었는지 확인합니다.

```powershell
wsl -l -v
```

WSL2 설치에 대한 전체 가이드는 https://learn.microsoft.com/en-us/windows/wsl/install 을 참조해주세요.

### 3. Ubuntu 기본 패키지 설치

Ubuntu 터미널에서 아래 명령을 실행합니다.

```bash
sudo apt update
sudo apt install -y git curl
```

### 4. 저장소 클론

WSL 홈 디렉터리 아래 작업 폴더를 만들고 저장소를 클론합니다.

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/golf-posture/golf-posture.git
cd golf-posture
```

### 5. Pixi 설치

이 프로젝트는 Python 가상환경 대신 `pixi`를 사용합니다.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

설치 후 새 셸을 열거나 아래를 실행합니다.

```bash
source ~/.bashrc
pixi --version
```

### 6. 프로젝트 환경 설치

저장소 루트에서 아래 명령을 실행합니다.

```bash
pixi install
```

### 7. 실행 확인

프로젝트에서 정의한 태스크가 있다면 아래처럼 실행합니다.

```bash
pixi run <task-name>
```

전체 설치 가이드는 https://pixi.prefix.dev/latest/installation 에서 확인할 수 있습니다.

## 모델 가중치 다운로드

이벤트 감지 실행을 위해 아래 파일을 다운로드하여 지정된 위치에 배치합니다. (git에 포함되지 않음)

| 파일 | 다운로드 | 배치 위치 |
|------|----------|----------|
| MobileNetV2 사전학습 가중치 | [mobilenet_v2.pth.tar](https://github.com/tonylins/pytorch-mobilenet-v2) | `event_detection/` |
| SwingNet 학습 완료 가중치 | [swingnet_1800.pth.tar](https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view?usp=sharing) | `event_detection/models/` |

```
event_detection/
├── mobilenet_v2.pth.tar        ← 여기
├── models/
│     └── swingnet_1800.pth.tar ← 여기
```

> Colab 사용 시 Google Drive에 업로드 후 노트북에서 복사하여 사용합니다.

## YouTube 영상으로 이벤트 감지 테스트

YouTube 골프 스윙 영상을 다운로드하여 SwingNet 모델로 이벤트 감지를 테스트할 수 있습니다.

### 1. 적합한 영상 조건

| 조건 | 설명 |
|------|------|
| 화면 비율 | 가로(16:9) 영상 (세로/Shorts ❌) |
| 인원 | 골퍼 1명만 나오는 영상 |
| 앵글 | 정면(face-on) 또는 후방(down-the-line) |
| 구간 | 한 번의 풀 스윙이 보이는 2~4초 구간 |

> GolfDB 모델은 TV 중계 스타일의 가로 영상으로 학습되었기 때문에, 세로 영상이나 여러 앵글이 섞인 영상에서는 정확도가 떨어집니다.

### 2. 영상 다운로드

브라우저 확장 프로그램(Video Downloader 등)을 이용하여 **MP4** 형식으로 다운로드합니다.

### 3. Google Drive 업로드

다운로드한 영상을 Google Drive에 업로드합니다.

```
Google Drive/
└── event_detection/
    └── youtube/
        └── 영상파일.mp4
```

### 4. Colab에서 실행

```python
# 셀 1: 환경 설정
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf golf-posture
!git clone https://github.com/golf-posture/golf-posture.git

# 모델 가중치 복사
%cd /content/golf-posture/event_detection
!cp /content/drive/MyDrive/event_detection/mobilenet_v2.pth.tar .
!mkdir -p models
!cp /content/drive/MyDrive/event_detection/models/swingnet_1800.pth.tar models/
```

```python
# 셀 2: 영상 복사 및 이벤트 감지 실행
!cp /content/drive/MyDrive/event_detection/youtube/영상파일.mp4 .
!python test_youtube_colab.py --file 영상파일.mp4 --start 시작초 --end 끝초
```

- `--start` / `--end`: 스윙 구간의 시작/끝 시간(초)
- 결과는 `output/영상파일_시작s-끝s/` 폴더에 저장됩니다

### 실행 예시

```python
!cp /content/drive/MyDrive/event_detection/youtube/videoplayback.mp4 .
!python test_youtube_colab.py --file videoplayback.mp4 --start 3.0 --end 7.0
```

8개 이벤트(Address → Finish) 프레임이 감지되어 이미지로 저장되고, `events_summary.png`로 요약 이미지가 출력됩니다.
