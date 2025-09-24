"""
LangGraph 멀티에이전트 시스템의 상태 정의
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """에이전트 간 공유 상태"""
    
    # 사용자 입력
    user_request: str
    original_request: str
    
    # 라우팅 정보
    route_decision: Optional[str]  # "sql", "ml", "both", "none"
    next_agent: Optional[str]      # 다음에 실행할 에이전트
    
    # SQL 분석 관련
    sql_query: Optional[str]
    sql_results: Optional[Dict[str, Any]]
    sql_analysis: Optional[str]
    
    # ML 관련
    ml_request: Optional[str]
    ml_code_path: Optional[str]
    ml_results: Optional[Dict[str, Any]]
    ml_analysis: Optional[str]
    
    # 협력 워크플로우
    needs_collaboration: bool
    collaboration_context: Optional[str]
    sql_for_ml: bool  # SQL 결과를 ML에서 사용할지
    ml_insights_for_sql: bool  # ML 인사이트를 SQL 분석에 반영할지
    
    # 최종 결과
    final_response: Optional[str]
    agent_sequence: List[str]  # 실행된 에이전트 순서
    
    # 메타데이터
    backend: str  # "openai" or "ollama"
    models: Dict[str, str]  # 사용된 모델 정보
    iteration_count: int  # 반복 횟수 (무한 루프 방지)
    
    # 오류 처리
    errors: List[str]
    warnings: List[str]


def create_initial_state(
    user_request: str,
    backend: str = "openai",
    models: Optional[Dict[str, str]] = None
) -> AgentState:
    """초기 상태 생성"""
    return AgentState(
        user_request=user_request,
        original_request=user_request,
        route_decision=None,
        next_agent=None,
        sql_query=None,
        sql_results=None,
        sql_analysis=None,
        ml_request=None,
        ml_code_path=None,
        ml_results=None,
        ml_analysis=None,
        needs_collaboration=False,
        collaboration_context=None,
        sql_for_ml=False,
        ml_insights_for_sql=False,
        final_response=None,
        agent_sequence=[],
        backend=backend,
        models=models or {},
        iteration_count=0,
        errors=[],
        warnings=[]
    )
