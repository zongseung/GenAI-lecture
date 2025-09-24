from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from .sql_analyst import SQLAnalystAgent
from .ml_engineer import MLEngineerAgent

ROUTING_SYSTEM_PROMPT = """
당신은 사용자 요청을 분석하여 적절한 에이전트로 라우팅하는 지능형 수퍼바이저입니다.

🎯 **라우팅 대상 에이전트:**

1️⃣ **SQL_ANALYST** - 데이터 조회, 분석, 통계 작업
   - 데이터베이스 쿼리가 필요한 모든 작업
   - 통계 분석, 데이터 탐색, 트렌드 분석
   - 특정 기간/조건의 데이터 조회
   - 예: "최근 30일 전력수요는?", "최고 수요가 언제?", "계절별 패턴 분석"

2️⃣ **ML_ENGINEER** - 머신러닝 모델, 코드 생성 작업  
   - 예측 모델 개발, 알고리즘 구현
   - 파이썬 코드 생성, 모델 훈련 스크립트
   - 딥러닝, 머신러닝 관련 모든 작업
   - 예: "LSTM 모델 코드 생성", "예측 알고리즘 개발", "회귀 모델 구현"

3️⃣ **BOTH** - 복합 작업 (데이터 분석 + 모델링)
   - 데이터 분석 후 모델링까지 필요한 경우
   - 분석 결과를 바탕으로 예측 모델 개발
   - 예: "데이터 분석하고 예측 모델도 만들어줘", "패턴 분석 후 AI 모델 개발"

4️⃣ **GENERAL** - 일반적인 질문이나 처리 불가능한 요청
   - 인사말, 일반 상식 질문
   - 시스템 정보 문의
   - 에이전트 범위를 벗어나는 요청

📋 **출력 형식:** 반드시 다음 중 하나만 출력하세요:
- SQL_ANALYST
- ML_ENGINEER  
- BOTH
- GENERAL

⚠️ **중요:** 결정 이유나 추가 설명 없이 에이전트 이름만 출력하세요.
"""

class SupervisorAgent:
    def __init__(self, db_path: str, openai_api_key: str,
                 backend: str = "openai",
                 ollama_base_url: str | None = None,
                 sql_model: str = "gpt-4o-mini",
                 ml_model: str = "gpt-4o-mini",
                 ollama_sql_model: str | None = None,
                 ollama_ml_model: str | None = None):
        self.db_path = db_path
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key, temperature=0.0)
        
        # SQL 에이전트 초기화 - 오류나면 이쪽 코드를 사용
        if backend == "ollama":
            self.sql_agent = SQLAnalystAgent(
                db_path=db_path,
                openai_api_key=openai_api_key,
                backend=backend,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_sql_model or sql_model,
            )
        else:
            self.sql_agent = SQLAnalystAgent(
                db_path=db_path,
                openai_api_key=openai_api_key,
                backend=backend,
                openai_model=sql_model,
            )
        
        # ML 에이전트 초기화  
        if backend == "ollama":
            self.ml_agent = MLEngineerAgent(
                db_path=db_path,
                openai_api_key=openai_api_key,
                backend=backend,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_ml_model or ml_model,
            )
        else:
            self.ml_agent = MLEngineerAgent(
                db_path=db_path,
                openai_api_key=openai_api_key,
                backend=backend,
                openai_model=ml_model,
            )

    def _intelligent_routing(self, user_prompt: str) -> str:
        """LLM을 사용한 지능형 라우팅 결정"""
        try:
            routing_prompt = f"""
            사용자 요청: "{user_prompt}"
            
            위 요청을 분석하여 적절한 에이전트를 선택하세요.
            """
            
            response = self.llm.invoke([
                SystemMessage(content=ROUTING_SYSTEM_PROMPT),
                HumanMessage(content=routing_prompt)
            ])
            
            # 응답에서 에이전트 이름 추출
            agent_decision = response.content.strip().upper()
            
            # 유효한 에이전트인지 확인
            valid_agents = ["SQL_ANALYST", "ML_ENGINEER", "BOTH", "GENERAL"]
            if agent_decision in valid_agents:
                return agent_decision
            else:
                # 부분 매칭 시도
                for agent in valid_agents:
                    if agent in agent_decision:
                        return agent
                # 폴백: 키워드 기반 간단 분류
                return self._fallback_routing(user_prompt)
                
        except Exception as e:
            print(f"라우팅 LLM 오류: {e}")
            return self._fallback_routing(user_prompt)
    
    def _fallback_routing(self, user_prompt: str) -> str:
        """폴백 라우팅 (키워드 기반)"""
        text = user_prompt.lower()
        
        # 복합 작업 키워드 (우선 체크)
        both_keywords = ["분석하고", "분석 후", "그리고", "또한", "다음에", "이어서", "모델도", "코드도"]
        if any(k in text for k in both_keywords):
            return "BOTH"
        
        # ML 키워드
        ml_keywords = ["모델", "예측", "코드", "lstm", "딥러닝", "머신러닝", "알고리즘", "생성"]
        if any(k in text for k in ml_keywords):
            return "ML_ENGINEER"
        
        # SQL 키워드  
        sql_keywords = ["조회", "분석", "최근", "언제", "통계", "데이터", "수요", "전력", "최대", "최소"]
        if any(k in text for k in sql_keywords):
            return "SQL_ANALYST"
        
        return "GENERAL"

    def process_request(self, user_prompt: str) -> Dict[str, Any]:
        """
        🧠 LLM 기반 지능형 라우팅 → 에이전트 실행 → 결과 반환
        """
        # 1단계: LLM으로 라우팅 결정
        agent_decision = self._intelligent_routing(user_prompt)
        
        # 2단계: 결정된 에이전트 실행
        result: Dict[str, Any] = {
            "route_decision": agent_decision.lower(),
            "agent_used": "none",
            "final_response": "요청을 처리할 수 없습니다."
        }
        
        if agent_decision == "SQL_ANALYST":
            # SQL 분석가 실행
            sql_res = self.sql_agent.analyze(user_prompt)
            final = sql_res.get("analysis") or "분석 결과를 찾았습니다."
            result.update({
                "route": "sql",
                "sql_analysis": sql_res,
                "final_response": final,
                "agent_used": "sql_analyst"
            })
            
        elif agent_decision == "ML_ENGINEER":
            # ML 엔지니어 실행
            ml_res = self.ml_agent.create_model(user_prompt)
            final = ml_res.get("analysis") or "코드가 생성되었습니다."
            result.update({
                "route": "ml", 
                "ml_results": ml_res,
                "final_response": final,
                "agent_used": "ml_engineer"
            })
            
        elif agent_decision == "BOTH":
            # 복합 작업: SQL 분석 → ML 모델링
            # 먼저 SQL 분석 실행
            sql_res = self.sql_agent.analyze(user_prompt)
            
            # SQL 결과를 바탕으로 ML 요청 생성
            ml_request = f"""
            다음 데이터 분석 결과를 바탕으로 예측 모델을 개발해주세요:
            
            원본 요청: {user_prompt}
            데이터 분석 결과: {sql_res.get('analysis', '')}
            
            이 분석을 바탕으로 적절한 머신러닝 모델 코드를 생성해주세요.
            """
            
            ml_res = self.ml_agent.create_model(ml_request)
            
            # 통합 응답 생성
            combined_response = f"""
            🔍 **1단계: 데이터 분석 결과**
            {sql_res.get('analysis', '')}
            
            🧠 **2단계: ML 모델 개발 결과** 
            {ml_res.get('analysis', '')}
            """
            
            result.update({
                "route": "both",
                "sql_analysis": sql_res,
                "ml_results": ml_res,
                "final_response": combined_response,
                "agent_used": "both",
                "collaboration": True
            })
            
        else:  # GENERAL
            # 일반 응답
            result.update({
                "route": "general",
                "final_response": """
                안녕하세요! 전력수요 분석 AI 어시스턴트입니다. 🔌

                다음과 같은 작업을 도와드릴 수 있습니다:

                📊 **데이터 분석:**
                • "최근 30일 전력수요 추세는?"
                • "여름철 피크 수요가 언제인가요?"
                • "계절별 전력수요 패턴 분석"

                🧠 **AI 모델 개발:**
                • "LSTM으로 전력수요 예측 모델 만들어줘"
                • "시계열 예측 알고리즘 코드 생성"
                • "회귀 모델로 수요 예측"

                🔄 **복합 작업:**
                • "데이터 분석하고 예측 모델도 개발해줘"

                어떤 도움이 필요하신가요?
                """,
                "agent_used": "general"
            })

        return result
