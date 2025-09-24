"""
전력수요 데이터 분석 멀티에이전트 챗봇 시스템 - Streamlit UI
"""
import streamlit as st
import os
import locale
import unicodedata
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import urllib.request as _urlreq
import urllib.error as _urlerr

# ===== UTF-8 환경 강제 (출력 인코딩 오류 방지) =====
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")
try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except Exception:
    pass

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="⚡ 전력수요 AI 분석 시스템",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 스타일 =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card { background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚡ 전력수요 AI 분석 챗봇</h1>', unsafe_allow_html=True)

# ===== 설정/가져오기 =====
import sys
sys.path.append('src')

from config import OPENAI_API_KEY, DATABASE_PATH
from src.agents.supervisor import SupervisorAgent
from src.workflow import EnergyLLMWorkflow
# LangGraph 시각화 기능 제거 (렌더링 오류 회피)
from src.utils.graph_visualizer import create_workflow_images
from src.utils.mermaid_visualizer import (
    render_langgraph_workflow,
    create_mermaid_workflow_images,
    create_langgraph_native_workflow,
)

def safe_str(text):
    """UI 표시 안정화를 위한 정규화(데이터 가공 금지)."""
    if not isinstance(text, str):
        try: 
            text = str(text)
        except Exception: 
            return ""
    
    # 강제 UTF-8 인코딩/디코딩으로 ASCII 오류 방지
    try:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        text = str(text).encode('ascii', errors='replace').decode('ascii')
    
    # 유니코드 정규화
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception:
        pass
    
    # 특수 유니코드 문자 치환
    repl = {
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'", 
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u00a0': ' ',
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"'
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    
    return text

# ===== GPU/CUDA 감지 유틸 =====
def detect_cuda_status():
    info = {"available": False, "via": [], "devices": []}
    # 1) PyTorch
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            info["available"] = True
            info["via"].append("torch")
            try:
                cnt = _torch.cuda.device_count()
                info["devices"] = [
                    _torch.cuda.get_device_name(i) for i in range(cnt)
                ]
            except Exception:
                pass
    except Exception:
        pass
    # 2) nvidia-smi
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=_sp.DEVNULL,
            timeout=1.5,
            text=True,
        ).strip()
        if out:
            info["available"] = True
            info["via"].append("nvidia-smi")
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            info["devices"].extend(lines)
    except Exception:
        pass
    # 3) /dev 검사
    try:
        import os as _os
        devs = [p for p in _os.listdir("/dev") if p.startswith("nvidia")]  # nvidia0, nvidiactl 등
        if devs:
            info["available"] = True
            info["via"].append("/dev/nvidia*")
    except Exception:
        pass
    return info

if not OPENAI_API_KEY:
    st.error("⚠️ 환경변수 또는 .env 에 OPENAI_API_KEY를 설정하세요.")
    st.stop()

# ===== LangGraph 워크플로우 초기화 =====
@st.cache_resource
def initialize_workflow():
    """기본 LangGraph 워크플로우 초기화"""
    try:
        return EnergyLLMWorkflow(
            db_path=DATABASE_PATH,
            openai_api_key=OPENAI_API_KEY
        )
    except Exception as e:
        st.error(f"LangGraph 워크플로우 초기화 오류: {e}")
        return None

@st.cache_resource
def get_workflow_with_backend(backend: str, ollama_url: str | None, 
                             sql_model: str, ml_model: str,
                             ollama_sql_model: str | None = None, 
                             ollama_ml_model: str | None = None):
    """백엔드별 LangGraph 워크플로우 생성"""
    try:
        return EnergyLLMWorkflow(
            db_path=DATABASE_PATH,
            openai_api_key=OPENAI_API_KEY,
            backend=backend,
            ollama_base_url=ollama_url,
            sql_model=sql_model,
            ml_model=ml_model,
            ollama_sql_model=ollama_sql_model,
            ollama_ml_model=ollama_ml_model,
        )
    except Exception as e:
        st.error(f"백엔드별 LangGraph 워크플로우 초기화 오류: {e}")
        return None

default_workflow = initialize_workflow()
if not default_workflow:
    st.stop()

# ===== 레거시 Supervisor 초기화 =====
@st.cache_resource
def initialize_supervisor():
    try:
        return SupervisorAgent(db_path=DATABASE_PATH, openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"에이전트 초기화 오류: {e}")
        return None

supervisor = initialize_supervisor()
if not supervisor:
    st.stop()

# 백엔드 옵션별 Supervisor 초기화 (캐시)
@st.cache_resource
def get_supervisor_with_backend(backend: str, ollama_url: str | None, 
                               sql_model: str, ml_model: str,
                               ollama_sql_model: str | None = None, 
                               ollama_ml_model: str | None = None):
    try:
        return SupervisorAgent(
            db_path=DATABASE_PATH,
            openai_api_key=OPENAI_API_KEY,
            backend=backend,
            ollama_base_url=ollama_url,
            sql_model=sql_model,
            ml_model=ml_model,
            ollama_sql_model=ollama_sql_model,
            ollama_ml_model=ollama_ml_model,
        )
    except Exception as e:
        st.error(f"에이전트 초기화 오류(백엔드): {e}")
        return None

# 데이터 로딩
@st.cache_data
def load_head_data(db_path: str, limit: int = 1000) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM my_table ORDER BY \"time\" DESC LIMIT {limit}", conn)
        conn.close()
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        st.error(f"데이터 로딩 오류: {e}")
        return pd.DataFrame()

# 전체 데이터 수 확인을 위한 함수
@st.cache_data
def get_total_count(db_path: str) -> int:
    try:
        conn = sqlite3.connect(db_path)
        result = conn.execute("SELECT COUNT(*) FROM my_table").fetchone()
        conn.close()
        return result[0] if result else 0
    except Exception:
        return 0

data = load_head_data(DATABASE_PATH, 1000)
total_count = get_total_count(DATABASE_PATH)

# ===== 사이드바 =====
with st.sidebar:
    st.markdown("**🔑 API 설정**")
    st.success("✅ API 키 설정됨" if OPENAI_API_KEY else "❌ API 키 없음")

    st.markdown("---")
    st.markdown("**🤖 사용 가능한 에이전트**")
    st.markdown("• **SQL 분석가**: 데이터 탐색/분석")
    st.markdown("• **ML 엔지니어**: 모델 코드 생성")
    st.markdown("• **수퍼바이저**: 라우팅/통합")

    st.markdown("---")
    st.markdown("**LLM 백엔드**")
    backend = st.selectbox("백엔드 선택", ["OpenAI", "Ollama"], index=0)
    
    # 에이전트별 모델 설정
    st.markdown("**에이전트별 모델 설정**")
    
    if backend == "OpenAI":
        sql_model = st.selectbox("SQL 분석가 모델", 
                                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], 
                                index=0, key="sql_openai_model")
        ml_model = st.selectbox("ML 엔지니어 모델", 
                               ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], 
                               index=0, key="ml_openai_model")
        ollama_url = None
        ollama_sql_model = None
        ollama_ml_model = None
    else:  # Ollama
        ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
        sql_model = st.text_input("SQL 분석가 Ollama 모델", value="llama3.1:8b", key="sql_ollama_model")
        ml_model = st.text_input("ML 엔지니어 Ollama 모델", value="qwen2.5:72b", key="ml_ollama_model")
        ollama_sql_model = sql_model
        ollama_ml_model = ml_model

    # Ollama 연결 사전 점검 및 폴백
    effective_backend = "openai"
    if backend == "Ollama" and ollama_url:
        try:
            _ = _urlreq.urlopen(f"{ollama_url.rstrip('/')}/api/tags", timeout=1.5)
            st.success("Ollama 연결 성공")
            effective_backend = "ollama"
        except Exception:
            st.warning("Ollama에 연결할 수 없어 OpenAI로 폴백합니다. (Base URL 확인)")
            effective_backend = "openai"

    # CUDA 감지 (다중 전략)
    cuda = detect_cuda_status()
    if cuda.get("available"):
        devices = cuda.get("devices") or []
        via = ", ".join(cuda.get("via", [])) or "unknown"
        label = " | ".join(devices) if devices else "GPU Detected"
        st.success(f"CUDA 감지됨 ({via}) - {label}")
    else:
        st.info("CUDA 미감지 (CPU 모드)")

    st.markdown("---")
    st.markdown("**🔄 LangGraph 워크플로우**")
    
    workflow_option = st.radio(
        "시각화 방식 선택:",
        ["🎯 LangGraph 네이티브", "📊 Mermaid 다이어그램", "🖼️ Graphviz 이미지"],
        horizontal=True
    )
    
    if st.button("🔄 워크플로우 생성"):
        with st.spinner("LangGraph 다이어그램 생성 중..."):
            try:
                if workflow_option == "🎯 LangGraph 네이티브":
                    result = create_langgraph_native_workflow(default_workflow)
                    if result.get("success"):
                        st.success("✅ LangGraph 네이티브 워크플로우 생성 완료")
                    else:
                        st.error(f"❌ 네이티브 워크플로우 오류: {result.get('error')}")
                elif workflow_option == "📊 Mermaid 다이어그램":
                    result = create_mermaid_workflow_images()
                    if result.get("success"):
                        st.success("✅ Mermaid 워크플로우 생성 완료")
                    else:
                        st.error(f"❌ Mermaid 오류: {result.get('error')}")
                else:
                    results = create_workflow_images()
                    if results.get("basic_workflow"):
                        st.success("✅ Graphviz 이미지 생성 완료")
                        st.image(results["basic_workflow"], caption="LangGraph 멀티에이전트 워크플로우", use_column_width=True)
                    else:
                        st.warning("⚠️ graphviz 설치를 확인해주세요.")
            except Exception as e:
                st.error(f"❌ 워크플로우 생성 오류: {e}")

    st.markdown("---")
    st.markdown("**💡 예시 질문**")
    st.markdown("**📊 데이터 분석:**")
    st.markdown("• 최근 1주일 전력수요 추세와 피크는?")
    st.markdown("• 기온(ta)과 전력수요의 상관관계 분석")
    st.markdown("• 요일/주말/공휴일에 따른 평균 수요 비교")
    st.markdown("• 어제 가장 수요가 높았던 시간은?")
    st.markdown("**🧠 모델 코드 생성:**")
    st.markdown("• LSTM으로 전력수요 예측 코드 생성해줘")
    st.markdown("• scikit-learn 회귀 기반 수요 예측 모델")
    st.markdown("• XGBoost 수요 예측 코드 작성해줘")

# ===== 상단 메트릭 =====
if not data.empty:
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("총 데이터 수", f"{total_count:,}", help=f"표시된 데이터: {len(data):,}개")
    with c2:
        if 'power demand(MW)' in data.columns:
            latest_demand = data['power demand(MW)'].iloc[0]
            st.metric("최신 수요(MW)", f"{latest_demand:,.0f}")
        else:
            st.metric("최신 수요(MW)", "N/A")
    with c3:
        if 'power demand(MW)' in data.columns and len(data) > 1 and data['power demand(MW)'].iloc[1] != 0:
            demand_change = (data['power demand(MW)'].iloc[0] - data['power demand(MW)'].iloc[1]) / data['power demand(MW)'].iloc[1] * 100
            st.metric("수요 변동률", f"{demand_change:.2f}%", delta=f"{demand_change:.2f}%")
        else:
            st.metric("수요 변동률", "N/A")
    with c4:
        if 'ta' in data.columns:
            st.metric("최신 기온(°C)", f"{data['temperature'].iloc[0]:.1f}")
        else:
            st.metric("최신 기온(°C)", "N/A")

st.markdown("---")

# ===== 채팅 상태 =====
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요! ⚡ 전력수요 데이터 분석 어시스턴트입니다.\n원하시는 분석을 말씀해 주세요. (예: 어제 피크 수요 시간은?)"
    }]

# 기존 메시지 출력
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(safe_str(m["content"]))

# ===== 입력 처리 =====
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(safe_str(prompt))

    with st.chat_message("assistant"):
        with st.spinner("AI 에이전트가 분석 중입니다..."):
            try:
                # LangGraph 워크플로우 사용
                sel_backend = effective_backend
                if sel_backend == "ollama":
                    workflow = get_workflow_with_backend(
                        sel_backend, ollama_url, sql_model, ml_model, 
                        ollama_sql_model, ollama_ml_model
                    ) or default_workflow
                else:
                    workflow = get_workflow_with_backend(
                        sel_backend, None, sql_model, ml_model
                    ) or default_workflow
                
                # LangGraph 워크플로우 실행
                result = workflow.process_request(prompt)

                # 최종 응답 가져오기
                response = result.get("final_response", "죄송합니다. 분석 결과를 가져올 수 없습니다.")
                response = safe_str(response)
                
                # 동적 워크플로우 차트 표시
                agent_sequence = result.get("agent_sequence", [])
                route_decision = result.get("route_decision", "")
                collaboration = result.get("collaboration", False)

                if agent_sequence:
                    with st.expander("🔄 **실행된 워크플로우 경로**", expanded=False):
                        render_langgraph_workflow(
                            agent_sequence=agent_sequence,
                            route_decision=route_decision,
                            collaboration=False,
                            show_detailed=False
                        )

                    if "sql_analyst" in agent_sequence:
                        st.markdown("🔍 **SQL 분석가**가 참여했습니다")
                    if "ml_engineer" in agent_sequence:
                        st.markdown("🧠 **ML 엔지니어**가 참여했습니다")
                
                # 성능 정보 표시
                execution_time = result.get("execution_time")
                if execution_time:
                    st.info(f"⚡ 실행 시간: {execution_time:.2f}초")
                
                # 협력 정보 표시 제거 (간소화)
                
                # 경고/오류 표시
                if result.get("warnings"):
                    for warning in result["warnings"]:
                        st.warning(f"⚠️ {warning}")
                
                if result.get("errors"):
                    for error in result["errors"]:
                        st.error(f"❌ {error}")
                
                st.markdown(response)

                # SQL 결과 표/차트 출력
                sql_results = result.get("sql_results")
                if sql_results:
                    sql_raw = sql_results.get("raw_data") or []
                    for item in sql_raw:
                        text = item.get("result", "")
                        if "Query executed successfully" in text and ":\n" in text:
                            try:
                                json_part = text.split(":\n", 1)[1]
                                if "Total rows:" in json_part:
                                    json_part = json_part.split("\n\nTotal rows:")[0]
                                import json as _json
                                df = pd.DataFrame(_json.loads(json_part))
                                if not df.empty and len(df) <= 200:
                                    st.markdown("#### 📊 조회 결과")
                                    st.dataframe(df)
                                    # 간단한 라인 차트 (시간 컬럼이 있는 경우)
                                    time_col = "time" if "time" in df.columns else ("일시" if "일시" in df.columns else ("t" if "t" in df.columns else ("date" if "date" in df.columns else None)))
                                    ycol = None
                                    for cand in ["power demand(MW)", "ta", "hm"]:
                                        if cand in df.columns:
                                            ycol = cand; break
                                    if time_col and ycol and len(df) > 2:
                                        fig = px.line(df, x=time_col, y=ycol, title=f"{ycol} 추이", markers=True)
                                        fig.update_layout(xaxis_title="시간/날짜", yaxis_title=ycol)
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass

                # ML 결과 코드 표시
                ml_results = result.get("ml_results")
                if ml_results:
                    if ml_results.get("generated_code_path"):
                        st.markdown("#### 💻 생성된 코드")
                        st.info(f"코드 파일: `{ml_results['generated_code_path']}`")
                        
                        # 생성된 코드 내용 표시 (있으면)
                        try:
                            with open(ml_results["generated_code_path"], "r", encoding="utf-8") as f:
                                code_content = f.read()
                            st.code(code_content, language="python")
                        except Exception:
                            st.warning("코드 파일을 읽을 수 없습니다.")
                    
                    if ml_results.get("strategy"):
                        st.markdown("#### 🎯 모델링 전략")
                        st.markdown(safe_str(ml_results["strategy"]))

                st.session_state["messages"].append({"role": "assistant", "content": response})

            except Exception as e:
                err = f"죄송합니다. 오류가 발생했습니다: {e}"
                st.error(safe_str(err))
                st.session_state["messages"].append({"role": "assistant", "content": safe_str(err)})

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>⚡ 전력수요 AI 분석 시스템 | LangGraph Multi-Agent | Powered by zongseung</div>", unsafe_allow_html=True)
