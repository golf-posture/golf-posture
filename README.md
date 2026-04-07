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

WSL2 설치에 대한 전체 가이드는 https://learn.microsoft.com/en-us/windows/wsl/install 을 참조하세요 .

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

## 권장 사항

- 프로젝트 파일은 `C:\...` 아래가 아니라 WSL 내부 경로인 `~/workspace/...` 아래에 두는 것을 권장합니다.
- Windows 탐색기에서 WSL 경로를 열려면 아래 주소를 사용하면 됩니다.

```text
\\wsl.localhost\Ubuntu\home\<사용자이름>\workspace
```
