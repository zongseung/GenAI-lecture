# Energy-LLM

전력 수요 데이터를 기반으로 LLM을 활용한 분석 및 예측 프로젝트입니다.  
패키지 관리는 **[uv](https://github.com/astral-sh/uv)** 로 수행합니다.  

---

## 요구 사항

- Windows 10 이상  
- Python 3.10+ (uv가 관리)  
- [uv 설치](https://github.com/astral-sh/uv#installation) 
  - Windows PowerShell에서 설치 (권장):
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```
  - 다른 환경 설치 방법은 [공식 문서](https://github.com/astral-sh/uv#installation) 참고

- Docker (선택 사항, 컨테이너 실행 시 필요)  

---

## 설치 방법
git이 설치되지 않았다면 무조건 설치해주셔야 합니다.
### 1. 프로젝트 클론
```powershell
git clone https://github.com/zongseung/GenAI-lecture.git
cd energy-llm
```

### 2. uv 환경 설정
uv를 이용하면 가상환경 생성 및 패키지 설치를 간단히 할 수 있습니다.

```powershell
uv sync  # pyproject.toml 및 uv.lock 기반 패키지 설치
```

> ⚠️ Windows에서는 `venv\Scripts\activate`로 가상환경 활성화하세요.
```powershell
.\.venv\Scripts\activate
```

### 3. 실행
```powershell
python app.py
```

---

## 데이터 파일

`power_demand_final.csv` 파일이 필요합니다.  
해당 파일은 `/data` 디렉토리에 위치시켜주세요:

```plaintext
energy-llm/
├── .venv/                # 가상환경 (uv sync 시 자동 생성)
├── generated/            # 결과물 저장 디렉토리
├── src/                  # 소스 코드
│   ├── agents/
│   ├── utils/
│   └── workflow/
├── static/               # 정적 파일 (이미지 등)
│   ├── detailed_workflow.png
│   └── langgraph_workflow.png
├── app.py                # 메인 실행 스크립트 Streamlit
├── config.py             # 설정 파일
├── test.ipynb            # 기존 demand 데이터 sqlite 로 생성
├── .env                  # 환경 변수
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── power_demand_final.csv
└── power_demand.db

```

---

## Docker 사용법

Docker를 이용하면 Windows 환경과 상관없이 동일한 실행 환경을 보장할 수 있습니다.  

### 1. 이미지 빌드
```powershell
docker build -t energy-llm .
```

### 2. 컨테이너 실행
```powershell
docker run --rm -it -v ${PWD}\data:/app/data energy-llm
```

> ⚠️ Windows PowerShell에서는 경로 구분자(`\`)와 `${PWD}`를 위와 같이 사용하세요.  
> Git Bash 또는 WSL에서는 `$(pwd):/app/data` 로 사용합니다.

---

## 패키지 공유 (uv 활용)

uv는 패키지를 PyPI에 배포하거나 wheel/tarball 파일로 배포할 수 있습니다.  

### wheel 빌드
```powershell
uv build
```

생성된 패키지는 `dist/` 디렉토리에 위치합니다.  

### 설치 (공유된 패키지)
```powershell
uv pip install dist/energy_llm-0.1.0-py3-none-any.whl
```

---

## 개발 가이드

- 의존성 추가:  
  ```powershell
  uv add requests
  ```
- 의존성 제거:  
  ```powershell
  uv remove requests
  ```
