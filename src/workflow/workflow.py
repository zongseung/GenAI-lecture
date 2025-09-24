"""
🚀 LangGraph 기반 멀티에이전트 워크플로우 (LLM Intelligent Routing + Fallback)

구조: Router → Agent Executor → Synthesizer (3노드 선형 플로우)
- Router: OpenAI LLM 기반 intelligent routing (실패 시 키워드 fallback)
- Agent Executor: SQL / ML 라우팅에 맞는 에이전트 실행
- Synthesizer: 최종 결과 종합 및 응답 생성
"""
from typing import Dict, Any
import logging
import unicodedata
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 스트리밍 지원 추가
try:
    from langchain_teddynote.messages import stream_response
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    print("⚠️ langchain_teddynote 미설치 - 스트리밍 비활성화")

from .state import AgentState, create_initial_state
from ..agents.sql_analyst import SQLAnalystAgent
from ..agents.ml_engineer import MLEngineerAgent

logger = logging.getLogger(__name__)


def _sanitize_unicode(text: str) -> str:
    """유니코드 문자 정규화 및 ASCII 안전 처리"""
    if not isinstance(text, str):
        return str(text)
    
    # 유니코드 정규화
    text = unicodedata.normalize('NFKC', text)
    
    # 특수 유니코드 문자 치환 (ASCII 안전)
    replacements = {
        '\u201c': '"', '\u201d': '"',  # 스마트 따옴표
        '\u2018': "'", '\u2019': "'",  # 스마트 따옴표
        '\u2013': '-', '\u2014': '-',  # 대시
        '\u2026': '...', '\u00a0': ' '  # 생략부호, 공백
    }
    
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    return text


class EnergyLLMWorkflow:
    """🚀 전력수요 AI 워크플로우 (3노드 구조: Router → Executor → Synthesizer)"""
    
    def __init__(
        self,
        db_path: str,
        openai_api_key: str,
        backend: str = "openai",
        ollama_base_url: str = None,
        sql_model: str = "gpt-4o-mini",
        ml_model: str = "gpt-4o-mini",
        ollama_sql_model: str = None,
        ollama_ml_model: str = None,
    ):
        self.db_path = db_path
        self.openai_api_key = openai_api_key
        self.backend = backend
        self.ollama_base_url = ollama_base_url
        
        # 모델 설정
        self.models = {
            "sql_model": sql_model,
            "ml_model": ml_model,
            "ollama_sql_model": ollama_sql_model,
            "ollama_ml_model": ollama_ml_model,
        }
        
        # 에이전트 초기화
        self._init_agents()
        
        # Supervisor LLM 초기화 (OpenAI + Ollama 지원)
        self.supervisor_llm = None
        self.ollama_llm = None
        
        # OpenAI LLM (우선)
        if openai_api_key:
            try:
                self.supervisor_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=openai_api_key,
                    temperature=0.0
                )
                print("✅ OpenAI Supervisor LLM 초기화 완료")
            except Exception as e:
                print(f"❌ OpenAI LLM 초기화 실패: {e}")
        
        # Ollama LLM (백업)
        if ollama_base_url:
            try:
                from langchain_community.chat_models import ChatOllama
                self.ollama_llm = ChatOllama(
                    model=ollama_sql_model or "llama3.1:8b",
                    base_url=ollama_base_url,
                    temperature=0.0
                )
                print("✅ Ollama Supervisor LLM 초기화 완료")
            except Exception as e:
                print(f"❌ Ollama LLM 초기화 실패: {e}")
        
        if not self.supervisor_llm and not self.ollama_llm:
            print("⚠️ 경고: LLM이 초기화되지 않음. 키워드 기반 폴백만 사용됩니다.")
        
        # LangGraph 워크플로우 구성
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _init_agents(self):
        """SQL/ML 에이전트 초기화"""
        # SQL 분석가
        if self.backend == "ollama":
            self.sql_agent = SQLAnalystAgent(
                db_path=self.db_path,
                openai_api_key=self.openai_api_key,
                backend=self.backend,
                ollama_base_url=self.ollama_base_url,
                ollama_model=self.models["ollama_sql_model"] or self.models["sql_model"],
            )
        else:
            self.sql_agent = SQLAnalystAgent(
                db_path=self.db_path,
                openai_api_key=self.openai_api_key,
                backend=self.backend,
                openai_model=self.models["sql_model"],
            )
        
        # ML 엔지니어
        if self.backend == "ollama":
            self.ml_agent = MLEngineerAgent(
                db_path=self.db_path,
                openai_api_key=self.openai_api_key,
                backend=self.backend,
                ollama_base_url=self.ollama_base_url,
                ollama_model=self.models["ollama_ml_model"] or self.models["ml_model"],
            )
        else:
            self.ml_agent = MLEngineerAgent(
                db_path=self.db_path,
                openai_api_key=self.openai_api_key,
                backend=self.backend,
                openai_model=self.models["ml_model"],
            )
    
    def _build_workflow(self) -> StateGraph:
        """간단한 LangGraph 워크플로우: supervisor → {coder, researcher}"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("coder", self._coder_node)
        workflow.add_node("researcher", self._researcher_node)
        
        # 시작점 설정
        workflow.set_entry_point("supervisor")
        
        # 조건부 라우팅: supervisor에서 결정
        workflow.add_conditional_edges(
            "supervisor",
            self._route_condition,
            {
                "coder": "coder",
                "researcher": "researcher", 
                "FINISH": END
            }
        )
        
        # 각 에이전트에서 다시 supervisor로 (순환 가능)
        workflow.add_edge("coder", "supervisor")
        workflow.add_edge("researcher", "supervisor")
        
        return workflow
    
    # ========== Conditional Routing Function ==========
    def _route_condition(self, state: AgentState) -> str:
        """조건부 라우팅 결정 함수"""
        route = state.get("route_decision", "FINISH")
        print(f"🔀 라우팅 결정: {route}")
        return route
    
    # ========== Supervisor Node ==========
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """🎯 Supervisor: LLM 기반 지능형 라우팅 및 최종 종합"""
        print(f"👨‍💼 Supervisor 시작: {state['user_request'][:50]}...")
        state["agent_sequence"].append("supervisor")

        # 현재 상태 분석
        has_sql_result = bool(state.get("sql_results"))
        has_ml_result = bool(state.get("ml_results"))
        iteration_count = state.get("iteration_count", 0)
        
        # 다중작업 지원: 사용자가 원하는 모든 작업이 완료되었는지 체크
        user_wants_both = self._user_wants_both_tasks(state["user_request"])
        
        # 작업 완료 조건 체크
        work_complete = False
        if user_wants_both:
            # 둘 다 원하는 경우: 둘 다 완료되어야 함
            work_complete = has_sql_result and has_ml_result
        else:
            # 하나만 원하는 경우: 해당 작업만 완료되면 됨  
            work_complete = has_sql_result or has_ml_result
        
        # 최대 반복 횟수 체크 또는 작업 완료시 종료
        if work_complete or iteration_count >= 5:
            final_response = self._generate_final_response(state)
            if final_response:
                state["final_response"] = final_response
                state["route_decision"] = "FINISH"
                print(f"🏁 작업 완료 - 종료 (SQL: {has_sql_result}, ML: {has_ml_result})")
                return state
        
        # 다음 작업 결정
        route = self._intelligent_routing(
            user_request=state["user_request"],
            has_sql_result=has_sql_result,
            has_ml_result=has_ml_result,
            iteration_count=iteration_count
        )
        
        state["iteration_count"] = iteration_count + 1
        state["route_decision"] = route
        print(f"🧠 Supervisor 결정: {route} (반복: {state['iteration_count']})")
        return state
    
    def _generate_final_response(self, state: AgentState) -> str:
        """최종 응답 생성"""
        responses = []
        
        # SQL 분석 결과
        if state.get("sql_analysis"):
            responses.append(f"📊 **데이터 분석 결과**\n{state['sql_analysis']}")
        
        # ML 코드 결과  
        if state.get("ml_analysis"):
            responses.append(f"🤖 **코드 생성 결과**\n{state['ml_analysis']}")
        
        # 에러가 있는 경우
        if state.get("errors") and not responses:
            error_msg = "\n".join([_sanitize_unicode(err) for err in state["errors"]])
            return f"❌ **오류 발생**\n\n{error_msg}\n\n💡 다시 시도해보세요."
        
        # 정상 응답
        if responses:
            return "\n\n".join(responses)
        
        return "죄송합니다. 요청을 처리할 수 없었습니다."
    
    def _user_wants_both_tasks(self, user_request: str) -> bool:
        """사용자가 데이터 분석과 코드 생성을 모두 원하는지 판단"""
        text = user_request.lower()
        
        # 다중작업 키워드
        both_keywords = [
            "분석하고", "분석 후", "그리고", "또한", "다음에", "이어서", 
            "모델도", "코드도", "분석과 코드", "데이터와 모델", "둘 다", "모두"
        ]
        
        # 코드/모델 관련 키워드
        coder_keywords = ["모델", "코드", "예측", "lstm", "딥러닝", "머신러닝", "생성", "구현", "알고리즘"]
        # 데이터 분석 관련 키워드  
        researcher_keywords = ["분석", "조회", "데이터", "통계", "트렌드", "패턴", "최근", "평균", "최대", "최소"]
        
        has_both_signal = any(k in text for k in both_keywords)
        has_coder = any(k in text for k in coder_keywords)
        has_researcher = any(k in text for k in researcher_keywords)
        
        # 명시적으로 둘 다 언급했거나, 둘 다 키워드가 있으면서 연결어가 있으면
        return has_both_signal or (has_coder and has_researcher)
    
    def _stream_llm_response(self, llm, messages, use_streaming=True):
        """LLM 응답 스트리밍 (ChatOllama 우선 지원)"""
        if not use_streaming or not STREAMING_AVAILABLE:
            return llm.invoke(messages)
        
        try:
            # ChatOllama인 경우 스트리밍 사용
            if hasattr(llm, 'base_url'):  # Ollama 모델 감지
                return stream_response(llm, messages)
            else:
                # OpenAI는 기본 invoke 사용
                return llm.invoke(messages)
        except Exception as e:
            print(f"스트리밍 실패, 기본 모드로 전환: {e}")
            return llm.invoke(messages)
    
    def _intelligent_routing(self, user_request: str, has_sql_result: bool, 
                           has_ml_result: bool, iteration_count: int) -> str:
        """LLM 기반 지능형 라우팅 결정"""
        
        # 무한 루프 방지
        if iteration_count >= 3:
            return "FINISH"
            
        # 컨텍스트 구성
        context = []
        if has_sql_result:
            context.append("- 이미 데이터 분석이 완료됨")
        if has_ml_result:
            context.append("- 이미 코드 생성이 완료됨")
        
        context_text = "\n".join(context) if context else "- 아직 수행된 작업 없음"
        
        routing_prompt = f"""
        당신은 전력수요 분석 시스템의 지능형 수퍼바이저입니다.
        사용자의 요청을 분석하고 현재 상황을 고려하여 다음 단계를 결정하세요.

        **사용자 요청:** {user_request}
        
        **현재 상황:**
        {context_text}
        
        **선택 가능한 다음 단계:**
        1. **researcher** - 전력수요 데이터 조회, 분석, 통계, 트렌드 탐색
        2. **coder** - 머신러닝 모델 코드 생성, 예측 알고리즘 구현
        3. **FINISH** - 작업 완료 (사용자 요구사항이 충족된 경우)
        
        **판단 기준:**
        - 사용자가 데이터 분석을 원하면 → researcher
        - 사용자가 코드/모델 생성을 원하면 → coder  
        - 요구사항이 이미 충족되었으면 → FINISH
        - 복합 요청인 경우 우선순위가 높은 것부터
        
        반드시 researcher, coder, FINISH 중 하나만 출력하세요.
        """

        try:
            # OpenAI 모델 우선 시도
            if self.supervisor_llm:
                messages = [
                    SystemMessage(content=routing_prompt),
                    HumanMessage(content=_sanitize_unicode(user_request))
                ]
                response = self._stream_llm_response(self.supervisor_llm, messages, use_streaming=False)
                route = _sanitize_unicode(response.content or "").strip().upper()
                
                # 유효성 검증
                valid_routes = {"RESEARCHER", "CODER", "FINISH"}
                if route in valid_routes:
                    return route.lower() if route != "FINISH" else "FINISH"
        
        except Exception as e:
            print(f"OpenAI 라우팅 실패: {_sanitize_unicode(str(e))}")
        
        # Ollama 백업 시도 (스트리밍 지원)
        try:
            if hasattr(self, 'ollama_llm') and self.ollama_llm:
                messages = [
                    SystemMessage(content=routing_prompt),
                    HumanMessage(content=_sanitize_unicode(user_request))
                ]
                response = self._stream_llm_response(self.ollama_llm, messages, use_streaming=True)
                route = _sanitize_unicode(response.content or "").strip().upper()
                
                valid_routes = {"RESEARCHER", "CODER", "FINISH"}
                if route in valid_routes:
                    return route.lower() if route != "FINISH" else "FINISH"
                    
        except Exception as e:
            print(f"Ollama 라우팅 실패: {_sanitize_unicode(str(e))}")
        
        # 폴백: 키워드 기반 라우팅
        return self._fallback_routing(user_request, has_sql_result, has_ml_result)
    
    def _fallback_routing(self, user_request: str, has_sql_result: bool, has_ml_result: bool) -> str:
        """폴백 라우팅 로직 (다중작업 지원)"""
        text = user_request.lower()
        
        # 사용자가 다중작업을 원하는지 체크
        user_wants_both = self._user_wants_both_tasks(user_request)
        
        # 완료 조건 체크
        if user_wants_both:
            # 둘 다 원하는 경우: 둘 다 완료되어야 종료
            if has_sql_result and has_ml_result:
                return "FINISH"
        else:
            # 하나만 원하는 경우: 해당 작업 완료시 종료
            if has_sql_result or has_ml_result:
                return "FINISH"
        
        # 코드/모델 관련 키워드
        coder_keywords = ["모델", "코드", "예측", "lstm", "딥러닝", "머신러닝", "생성", "구현", "알고리즘"]
        # 데이터 분석 관련 키워드  
        researcher_keywords = ["분석", "조회", "데이터", "통계", "트렌드", "패턴", "최근", "평균", "최대", "최소"]
        
        has_coder = any(k in text for k in coder_keywords)
        has_researcher = any(k in text for k in researcher_keywords)
        
        # 다중작업인 경우 순서대로 처리
        if user_wants_both or (has_coder and has_researcher):
            if not has_sql_result:
                return "researcher"  # 데이터 분석 먼저
            elif not has_ml_result:
                return "coder"       # 그 다음 코드 생성
            else:
                return "FINISH"
        
        # 단일 작업인 경우
        elif has_coder and not has_ml_result:
            return "coder"
        elif has_researcher and not has_sql_result:
            return "researcher"
        else:
            # 기본값: researcher 우선
            return "researcher" if not has_sql_result else "FINISH"
    
    # ========== Researcher Node ==========
    def _researcher_node(self, state: AgentState) -> AgentState:
        """📊 Researcher: 데이터 분석 및 연구"""
        print("📊 Researcher 실행 시작")
        state["agent_sequence"].append("researcher")
        
        try:
            result = self.sql_agent.analyze(state["user_request"])
            state["sql_query"] = result.get("sql_query")
            state["sql_results"] = result
            state["sql_analysis"] = _sanitize_unicode(result.get("analysis", ""))
            print("✅ Researcher 작업 완료 - Supervisor로 복귀")
            
        except Exception as e:
            safe_error = _sanitize_unicode(str(e))
            print(f"❌ Researcher 오류: {safe_error}")
            state["errors"].append(f"데이터 분석 실패: {safe_error}")
        
        return state
    
    # ========== Coder Node ==========
    def _coder_node(self, state: AgentState) -> AgentState:
        """🤖 Coder: 코드 생성 및 모델링"""
        print("🤖 Coder 실행 시작")
        state["agent_sequence"].append("coder")
        
        try:
            result = self.ml_agent.create_model(state["user_request"])
            state["ml_code_path"] = result.get("generated_code_path")
            state["ml_results"] = result
            state["ml_analysis"] = _sanitize_unicode(result.get("analysis", ""))
            print("✅ Coder 작업 완료 - Supervisor로 복귀")
            
        except Exception as e:
            safe_error = _sanitize_unicode(str(e))
            print(f"❌ Coder 오류: {safe_error}")
            state["errors"].append(f"코드 생성 실패: {safe_error}")
        
        return state
    
    # ========== Public API ==========
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """사용자 요청 처리"""
        import time
        start_time = time.time()
        
        initial_state = create_initial_state(
            user_request=user_request,
            backend=self.backend,
            models=self.models
        )
        
        try:
            result = self.app.invoke(initial_state)
            execution_time = time.time() - start_time
            
            return {
                "final_response": result.get("final_response"),
                "agent_sequence": result.get("agent_sequence", []),
                "sql_results": result.get("sql_results"),
                "ml_results": result.get("ml_results"),
                "route_decision": result.get("route_decision"),
                "collaboration": result.get("needs_collaboration", False),
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"워크플로우 실행 오류: {e}")
            return {
                "final_response": f"❌ 시스템 오류가 발생했습니다: {e}",
                "agent_sequence": ["error"],
                "errors": [str(e)]
            }
