import sqlite3
import pandas as pd
import json
import unicodedata
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None
from langchain.schema import HumanMessage, SystemMessage


class SQLQueryRunner:
    """SQLite 쿼리 실행기 (안전/일관 dict 반환)."""
    def __init__(self, db_path: str):
        self.db_path = db_path

    @staticmethod
    def _normalize_sql(q: str) -> str:
        if not isinstance(q, str):
            return q
        q = unicodedata.normalize('NFKC', q)
        repl = {'\u201c':'"', '\u201d':'"', '\u2018':"'", '\u2019':"'"}
        for k, v in repl.items():
            q = q.replace(k, v)
        return q

    def run(self, query: str, limit_rows: Optional[int] = None) -> Dict[str, Any]:
        """SQL 실행 (limit_rows=None이면 전체 반환)."""
        try:
            query = self._normalize_sql(query)
            if not query.lower().strip().startswith("select"):
                return {"success": False, "error": "Only SELECT statements are allowed.", "query": query}

            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)

            if limit_rows:
                data = df.head(limit_rows).to_dict(orient="records")
            else:
                data = df.to_dict(orient="records")

            return {"success": True, "row_count": len(df), "data": data}
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}


class SQLAnalystAgent:
    """
    사용자가 자연어로 묻기 → LLM이 SQL 생성 → DB 실행 → 결과 해석
    """
    def __init__(self, db_path: str, openai_api_key: str,
                 backend: str = "openai",
                 ollama_base_url: Optional[str] = None,
                 ollama_model: Optional[str] = None,
                 openai_model: str = "gpt-4o-mini"):
        self.db_path = db_path
        self.runner = SQLQueryRunner(db_path)
        backend = (backend or "openai").lower()
        if backend == "ollama" and ChatOllama is not None:
            self.llm = ChatOllama(
                model=ollama_model or "llama3",
                base_url=ollama_base_url or "http://localhost:11434",
                temperature=0.1
            )
        else:
            self.llm = ChatOpenAI(model=openai_model,
                                  openai_api_key=openai_api_key,
                                  temperature=0.2)

        # ⚡️ 월별 그룹핑 지침 강화
        self.sql_system = """
당신은 전력수요 데이터 SQL 전문가입니다.
- 사용할 테이블: my_table(
  time TEXT, humanity REAL, temperature REAL, "power demand(MW)" REAL,
  holiday_name TEXT, weekday INTEGER, weekend INTEGER,
  spring INTEGER, summer INTEGER, autumn INTEGER, winter INTEGER,
  is_holiday_dummies INTEGER
)
- 반드시 SELECT 문만 작성 (INSERT/UPDATE/DELETE 금지)

기간 필터링 가이드 (시간 컬럼은 "time", 데이터 범위: 2019-01-01 ~ 2024-10-31):
- "최근 1주일": WHERE datetime("time") >= datetime('2024-10-31', '-7 days')
- "최근 한달/30일": WHERE datetime("time") >= datetime('2024-10-31', '-30 days')
- "최근 3개월": WHERE datetime("time") >= datetime('2024-10-31', '-3 months')
- "최근 1년": WHERE datetime("time") >= datetime('2024-10-31', '-1 year')
- "어제": WHERE date("time") = date('2024-10-31', '-1 day')
- "오늘": WHERE date("time") = date('2024-10-31')
- 기간 명시 없으면 사용자 요청에 맞게 쿼리문을 사용

⚠️ 월별/연별 분석 요청 시:
- 반드시 strftime('%Y-%m', time) 또는 strftime('%Y', time)으로 그룹핑하세요.

유용한 분석 예시:
- 수요 추세/피크: AVG/MAX/MIN "power demand(MW)" 와 시간대별, 월별, 연별 그룹핑
- 요일/주말/공휴일 영향: weekday, weekend, holiday_name 별 집계
- 계절/기온 영향: season 컬럼들(spring/summer/autumn/winter), temperature(기온), humanity(습도)와의 상관
- 불필요한 설명 없이 실행 가능한 SQL만 출력
"""

        self.interpret_system = """
당신은 전력수요 데이터 분석 전문가입니다.
SQL 결과를 보고 전력수요 관점에서 다음 형식으로 답변하세요:

📊 **분석 요약**: (핵심 인사이트)
📈 **수요 동향/피크**: (최근 추세, 최대/최소, 시간대별/월별/연별 패턴)
🌡️ **기온/계절 영향**: (Temperature 및 계절 더미와 수요의 관계)
📅 **요일/주말/휴일 효과**: (weekday/weekend/holiday 영향)
💡 **운영 인사이트**: (수요관리/예측에 유용한 제안)

친근하고 전문적인 한국어로 작성하세요.
"""

    def _gen_sql(self, user_request: str) -> str:
        resp = self.llm.invoke([
            SystemMessage(content=self.sql_system),
            HumanMessage(content=user_request)
        ])
        sql = (resp.content or "").strip()
        if sql.lower().startswith("```sql"):
            sql = sql[6:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()

    def analyze(self, request: str, limit_rows: Optional[int] = None) -> Dict[str, Any]:
        sql = self._gen_sql(request)
        result = self.runner.run(sql, limit_rows=limit_rows)

        analysis = ""
        if result.get("success") and result.get("data"):
            data = result["data"]
            data_summary = {
                "총 데이터 수": result.get("row_count", 0),
                "샘플 데이터 (상위 5행)": data[:5],
                "데이터 종류": list(data[0].keys()) if data else []
            }

            prompt = f"""
사용자 질문: {request}
실행된 SQL: {sql}
데이터 요약: {json.dumps(data_summary, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 전문적이고 구체적인 전력수요 분석을 해주세요.
숫자와 트렌드를 구체적으로 언급하며 분석해주세요.
"""
            resp = self.llm.invoke([
                SystemMessage(content=self.interpret_system),
                HumanMessage(content=prompt)
            ])
            analysis = unicodedata.normalize('NFKC', resp.content or "")
            # 특수 유니코드 문자 치환
            replacements = {
                '\u201c': '"', '\u201d': '"',
                '\u2018': "'", '\u2019': "'",
                '\u2013': '-', '\u2014': '-',
                '\u2026': '...', '\u00a0': ' '
            }
            for unicode_char, ascii_char in replacements.items():
                analysis = analysis.replace(unicode_char, ascii_char)
        elif result.get("success") and not result.get("data"):
            analysis = "쿼리는 성공했지만 조건에 맞는 데이터가 없습니다."
        else:
            analysis = f"데이터 조회 중 오류가 발생했습니다: {result.get('error', '알 수 없는 오류')}"

        pretty_json = json.dumps(result.get("data", []), ensure_ascii=False, indent=2)
        human_result_text = (
            f"Query executed successfully:\n{pretty_json}"
            if result.get("success")
            else f"Error executing query: {result.get('error')}"
        )

        return {
            "request": request,
            "sql_query": sql,
            "analysis": analysis,
            "raw_data": [{"query": sql, "result": human_result_text}]
        }
