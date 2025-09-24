import os
import re
import textwrap
import unicodedata
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import sqlite3

from langchain_openai import ChatOpenAI
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None

from langchain.schema import HumanMessage, SystemMessage


# --------- Helpers ---------
def _strip_code_fences(s: str) -> str:
    """```python ... ``` 형태로 감싸져 오면 내부만 추출."""
    if not s:
        return s
    pattern = r"^```(?:\w+)?\s*(.*?)\s*```$"
    m = re.match(pattern, s, flags=re.DOTALL)
    return m.group(1) if m else s

def _extract_section(text: str, tag: str) -> str:
    """
    LLM 출력에서 특정 섹션만 안전하게 추출.
    tag="CODE"면 <<<CODE>>> ... <<<END-CODE>>> 사이를 반환.
    """
    if not text:
        return ""
    pattern = rf"<<<{tag}>>>\s*(.*?)\s*<<<END-{tag}>>>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else "").strip()

def _versioned_filename(prefix: str, ext: str) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}.{ext}"

def _safe_unicode(text: str) -> str:
    """유니코드 문자 정규화 및 ASCII 안전 처리"""
    if not text:
        return text
    # 유니코드 정규화
    text = unicodedata.normalize('NFKC', text)
    # 특수 유니코드 문자 치환
    replacements = {
        '\u201c': '"', '\u201d': '"',
        '\u2018': "'", '\u2019': "'",
        '\u2013': '-', '\u2014': '-',
        '\u2026': '...', '\u00a0': ' '
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text


# --------- Agent ----------
class MLEngineerAgent:
    """
    사용자가 요청하면:
      1) LLM이 PyTorch 기반의 '충분히 긴' 실행 가능 코드 생성 (주석으로 설명)
      2) 코드와 requirements.txt를 분리 출력 → 파일 저장
      3) 성찰(2회)로 코드 품질/안정성/재현성 강화
    """
    def __init__(
        self,
        db_path: str,
        openai_api_key: str,
        backend: str = "openai",
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
    ):
        self.db_path = db_path
        backend = (backend or "openai").lower()

        if backend == "ollama" and ChatOllama is not None:
            self.llm = ChatOllama(
                model=ollama_model or "llama3",
                base_url=ollama_base_url or "http://localhost:11434",
                temperature=0.2,
            )
        else:
            self.llm = ChatOpenAI(
                model=openai_model,
                openai_api_key=openai_api_key,  # ✅ 올바른 인자명
                temperature=0.2,
            )

        # 일관된 규칙: "설명은 주석으로만", "코드/요구사항은 정해진 섹션으로만 출력"
        self.system_prompt = textwrap.dedent("""
        당신은 딥러닝/머신러닝 **코드 생성 전문가**입니다.
        반드시 **실행 가능한 PyTorch 기반 Python 스크립트**와 **requirements.txt**를 생성합니다.

        출력 규칙(아주 중요):
        - **설명은 코드 주석(# ...)으로만** 작성하고, 텍스트 설명은 절대 출력하지 마세요.
        - 아래의 섹션 태그를 정확히 지켜 출력하세요.
          1) 코드 섹션:
             <<<CODE>>>
             # 여기에 실행 가능한 Python 코드 (하나의 .py 파일로 완결)
             <<<END-CODE>>>
          2) 요구사항 섹션:
             <<<REQS>>>
             # 여기에 requirements.txt 내용 (패키지 줄바꿈 나열)
             <<<END-REQS>>>

        데이터 스키마 (SQLite 'my_table'):
        - time (TEXT, ISO8601 문자열)  → pandas에서 datetime으로 파싱
        - humanity (REAL, 습도), temperature (REAL, 기온)
        - spring/summer/autumn/winter (INTEGER, 계절 더미)
        - weekday/weekend (INTEGER)
        - is_holiday_dummies (INTEGER), holiday_name (TEXT)
        - target: "power demand(MW)" (REAL, 전력수요)

        코드 필수 요건:
        - 구조: load_data() → build_dataset/dataloader → build_model() → train() → evaluate() → visualize() → save_model()
        - 시드 고정(재현성): random/np/torch 모두
        - 장치: GPU 있으면 cuda, 없으면 cpu 자동 선택
        - 전처리: 결측치 처리, 스케일링(StandardScaler 등)
        - 데이터 분리: train/valid/test
        - 학습: 배치/에폭, 옵티마이저/스케줄러(선택), 조기종료(선택)
        - 평가: MAE, MSE, RMSE, R2 등
        - 시각화: 학습 곡선 및 "예측 vs 실제" 그래프 → ./results/ 하위에 저장
        - 모델 저장: ./models/ 하위에 저장 (os.makedirs 사용)
        - 로깅: logging 모듈로 주요 단계/지표 출력
        - 상대경로 사용, 경로 생성 보장(os.makedirs)
        - 코드 길이는 충분히 상세하게 (짧은 토이 예제 금지)

        requirements.txt 생성 지침:
        - torch, torchmetrics(선택), numpy, pandas, scikit-learn, matplotlib, sqlite3(표준 라이브러리 제외), joblib 등 필요 패키지만
        - seaborn은 선택. 최소 의존으로 구성
        - 정확한 패키지명을 줄바꿈으로 나열 (버전 고정은 선택)
        """)

    # --- metadata: DB 컬럼 가져오기 (fallback 포함) ---
    def _get_table_columns(self) -> List[str]:
        fallback = [
            "time", "humanity", "temperature", "power demand(MW)",
            "holiday_name", "weekday", "weekend",
            "spring", "summer", "autumn", "winter",
            "is_holiday_dummies",
        ]
        try:
            if not os.path.exists(self.db_path):
                return fallback
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='my_table'")
            if not cur.fetchone():
                conn.close()
                return fallback
            cur.execute("PRAGMA table_info(my_table)")
            rows = cur.fetchall()
            conn.close()
            cols = [r[1] for r in rows] if rows else []
            return cols or fallback
        except Exception:
            return fallback

    # --- 1st draft generation ---
    def _gen_initial(self, request: str, columns: List[str]) -> str:
        prompt = textwrap.dedent(f"""
        사용자 요청: {request}

        데이터베이스: SQLite '{self.db_path}' → 'my_table'
        사용 가능한 컬럼: {columns}

        아래 출력 규칙을 반드시 따르세요:
        - 코드만 주석으로 설명, 텍스트 설명 금지
        - 코드는 PyTorch 기반으로, 충분히 상세하고 구조화된 스크립트로 작성
        - 코드 섹션과 요구사항 섹션을 정확히 구분해 태그로 출력

        출력 형식:
        <<<CODE>>>
        # 실행 가능한 Python 코드
        <<<END-CODE>>>
        <<<REQS>>>
        # requirements.txt 내용
        <<<END-REQS>>>
        """)
        resp = self.llm.invoke([SystemMessage(content=self.system_prompt),
                                HumanMessage(content=prompt)])
        return _safe_unicode(resp.content or "")

    # --- reflection round ---
    def _reflect(self, previous_code: str, focus: str) -> str:
        prompt = textwrap.dedent(f"""
        다음은 이전에 생성한 코드 섹션입니다:
        ```python
        {previous_code}
        ```

        지시사항:
        - 아래 개선 포인트에 초점을 맞춰 **완성된 코드**와 **requirements.txt**를 다시 출력하세요.
        - 설명은 코드 주석으로만 작성. 텍스트 설명 금지.
        - 반드시 섹션 태그를 지켜 출력:
          <<<CODE>>>
          # 개선된 전체 Python 코드
          <<<END-CODE>>>
          <<<REQS>>>
          # requirements.txt
          <<<END-REQS>>>

        개선 포인트:
        {focus}
        """)
        resp = self.llm.invoke([SystemMessage(content=self.system_prompt),
                                HumanMessage(content=prompt)])
        return _safe_unicode(resp.content or "")

    def _save_files(self, code_text: str, reqs_text: str) -> Tuple[str, str, str]:
        os.makedirs("generated", exist_ok=True)
        filename = _versioned_filename("generated_model", "py")
        code_path = os.path.join("generated", filename)
        reqs_path = os.path.join("generated", "requirements.txt")
        status = "성공"
        try:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_text.rstrip() + "\n")
            with open(reqs_path, "w", encoding="utf-8") as f:
                f.write(reqs_text.rstrip() + "\n")
        except Exception as e:
            status = f"실패: {e}"
        return code_path, reqs_path, status

    # --- Public API ---
    def create_model(self, request: str) -> Dict[str, Any]:
        columns = self._get_table_columns()

        # 1) 초기 생성
        draft = self._gen_initial(request, columns)
        code_1 = _extract_section(_strip_code_fences(draft), "CODE")
        reqs_1 = _extract_section(_strip_code_fences(draft), "REQS")

        # 2) 성찰 1: 품질/모범사례/데이터처리/성능/시각화
        focus_1 = textwrap.dedent("""
        1) 코드 품질/모듈성/가독성 향상
        2) PyTorch 모범사례(디바이스, 최적화, 배치, 체크포인트)
        3) 데이터 전처리/스케일링/시계열 처리 강화
        4) 성능(학습 루프/메모리 효율/조기 종료)
        5) 시각화(학습 곡선/예측 vs 실제) 개선 및 저장
        """)
        draft2 = self._reflect(code_1, focus_1)
        code_2 = _extract_section(_strip_code_fences(draft2), "CODE")
        reqs_2 = _extract_section(_strip_code_fences(draft2), "REQS")

        # 3) 성찰 2: 안정성/로깅/하이퍼파라미터/재현성/사용성
        focus_2 = textwrap.dedent("""
        1) 실행 안정성(예외 처리, 경로 검증)
        2) 로깅(logging) 및 진행 상황 출력
        3) 하이퍼파라미터 기본값과 CLI 인자 처리(optional)
        4) 시드 고정 및 재현성
        5) 결과물 저장 구조(./models, ./results) 엄격 준수
        """)
        draft3 = self._reflect(code_2, focus_2)
        code_final = _extract_section(_strip_code_fences(draft3), "CODE") or code_2 or code_1
        reqs_final = _extract_section(_strip_code_fences(draft3), "REQS") or reqs_2 or reqs_1

        # 빈 섹션 방어
        if not code_final.strip():
            code_final = "# Fallback: 코드 생성 실패\nprint('Code generation failed')\n"
        if not reqs_final.strip():
            # 가장 최소 세트
            reqs_final = "\n".join([
                "torch",
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                # sqlite3는 표준 라이브러리의 sqlite3 모듈이므로 별도 패키지 불필요
            ])

        # 저장
        code_path, reqs_path, save_status = self._save_files(code_final, reqs_final)

        # 사용자 안내
        analysis = textwrap.dedent(f"""
        🎯 **딥러닝 코드 생성 완료 (Reflection x2)**

        📁 코드 파일: `{code_path}`
        📄 requirements: `{reqs_path}`
        💾 저장 상태: {save_status}

        🔄 개선 과정
        - 초안 → 성찰①(품질/모범사례/성능/시각화) → 성찰②(안정성/로깅/재현성/저장구조)

        🚀 실행 방법
        ```bash
        pip install -r {reqs_path}
        python {code_path}
        ```

        결과물은 `./results/`(그래프/리포트), 모델은 `./models/`에 저장됩니다.
        """)

        return {
            "agent_type": "ml_engineer",
            "request": request,
            "generated_code_path": code_path,
            "requirements_path": reqs_path,
            "analysis": analysis,
            "save_status": save_status,
            "raw_code_preview": code_final[:4000],  # 필요시 확인용
        }
