"""
Mermaid를 사용한 LangGraph 워크플로우 시각화 (단순화: Supervisor ↔ SQL/ML ↔ Synthesizer)
"""
import streamlit as st
from typing import List, Dict, Any, Optional
import streamlit.components.v1 as components
from io import BytesIO
import base64

# LangGraph 네이티브 그래프 기능 체크
try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Pyppeteer 통합 제거 (내부 렌더러 미사용)
PYPPETEER_AVAILABLE = False


class MermaidLangGraphVisualizer:
    """Mermaid를 사용한 LangGraph 워크플로우 시각화"""
    
    def __init__(self):
        self.node_colors = {
            "supervisor": "#e1f5fe",
            "sql_analyst": "#e8f5e8", 
            "ml_engineer": "#fff3e0",
            "collaboration_manager": "#f3e5f5",
            "final_synthesizer": "#fce4ec",
            "start": "#e0f2f1",
            "end": "#ffebee"
        }
        
        self.active_color = "#ffcdd2"  # 실행된 노드 색상
    
    def generate_basic_workflow(self) -> str:
        """기본 LangGraph 워크플로우 Mermaid 다이어그램 (Supervisor → SQL/ML → Synthesizer)"""
        return """
        graph TD
            START([🚀 start]) --> SUPERVISOR{👨‍💼 supervisor<br/>지능형 라우팅}
            
            SUPERVISOR -->|researcher| RESEARCHER[📊 researcher<br/>데이터 분석]
            SUPERVISOR -->|coder| CODER[🤖 coder<br/>코드 생성]
            SUPERVISOR -->|FINISH| END([🏁 end])
            
            RESEARCHER --> SUPERVISOR
            CODER --> SUPERVISOR
            
            classDef startEnd fill:#e0f2f1,stroke:#4caf50,stroke-width:3px
            classDef supervisor fill:#e1f5fe,stroke:#2196f3,stroke-width:3px
            classDef agent fill:#fff3e0,stroke:#ff9800,stroke-width:2px
            
            class START,END startEnd
            class SUPERVISOR supervisor
            class RESEARCHER,CODER agent
        """
    
    def generate_execution_path_diagram(self, agent_sequence: List[str], route_decision: str = "") -> str:
        """실행 경로를 강조한 다이어그램"""
        base_diagram = """
        graph TD
            START([사용자 요청]) --> SUPERVISOR{Supervisor<br/>라우팅 판단}
            
            SUPERVISOR -->|SQL| SQL[SQL Analyst<br/>데이터 분석]
            SUPERVISOR -->|ML| ML[ML Engineer<br/>모델 개발]
            
            SQL --> SYNTH[Final Synthesizer<br/>응답 생성]
            ML --> SYNTH
            
            SYNTH --> END([최종 응답])
            
            classDef default fill:#f9f9f9,stroke:#999,stroke-width:1px
            classDef active fill:#ffcdd2,stroke:#d32f2f,stroke-width:3px
            classDef startEnd fill:#e0f2f1,stroke:#4caf50,stroke-width:2px
            classDef supervisor fill:#e1f5fe,stroke:#2196f3,stroke-width:2px
            classDef sqlAgent fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
            classDef mlAgent fill:#fff3e0,stroke:#ff9800,stroke-width:2px
            classDef synthesizer fill:#fce4ec,stroke:#e91e63,stroke-width:2px
        """
        
        # 클래스 할당
        active_nodes = []
        
        # 실행된 노드들을 active로 표시
        node_mapping = {
            "supervisor": "SUPERVISOR",
            "sql_analyst": "SQL", 
            "ml_engineer": "ML",
            "final_synthesizer": "SYNTH"
        }
        
        for agent in agent_sequence:
            if agent in node_mapping:
                active_nodes.append(node_mapping[agent])
        
        # 시작과 끝은 항상 활성화
        active_nodes.extend(["START", "END"])
        
        # 클래스 정의 추가
        class_assignments = []
        
        for node, class_name in [
            ("START,END", "startEnd"),
            ("SUPERVISOR", "supervisor"), 
            ("SQL", "sqlAgent"),
            ("ML", "mlAgent"),
            ("SYNTH", "synthesizer")
        ]:
            if any(n in active_nodes for n in node.split(",")):
                # 활성 노드면 active 클래스도 추가
                if node in ["START,END"]:
                    class_assignments.append(f"class {node} startEnd")
                else:
                    class_assignments.append(f"class {node} active")
            else:
                class_assignments.append(f"class {node} {class_name}")
        
        return base_diagram + "\n" + "\n".join(class_assignments)
    
    def generate_detailed_flow_diagram(self) -> str:
        """상세한 플로우 다이어그램"""
        return """
        graph TD
            A[사용자 입력] --> B{요청 분석}
            B -->|데이터 조회| C[SQL 쿼리 생성]
            B -->|모델 생성| D[ML 코드 생성]
            B -->|복합 작업| E[협력 워크플로우]
            
            C --> F[데이터 실행]
            F --> G[결과 분석]
            G --> H{추가 작업 필요?}
            H -->|ML 필요| D
            H -->|완료| I[응답 생성]
            
            D --> J[코드 생성]
            J --> K[성찰 개선]
            K --> L[최종 코드]
            L --> M{데이터 필요?}
            M -->|SQL 필요| C
            M -->|완료| I
            
            E --> N[작업 순서 결정]
            N --> O{우선순위}
            O -->|SQL 우선| C
            O -->|ML 우선| D
            O -->|병렬 처리| P[동시 실행]
            P --> C
            P --> D
            
            I --> Q[최종 응답]
            
            classDef input fill:#e3f2fd,stroke:#1976d2
            classDef decision fill:#fff3e0,stroke:#f57c00
            classDef process fill:#e8f5e8,stroke:#388e3c
            classDef output fill:#fce4ec,stroke:#c2185b
            
            class A input
            class B,H,M,O decision
            class C,D,E,F,G,J,K,L,N,P process
            class I,Q output
        """
    
    def render_mermaid_diagram(self, mermaid_code: str, height: int = 600, key: Optional[str] = None, 
                             use_pyppeteer: bool = False) -> None:
        """Streamlit에서 Mermaid 다이어그램 렌더링"""
        # 기본 HTML 렌더링 (CDN 사용)
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 10px;
                    background-color: #fafafa;
                }}
                .mermaid {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: {height-20}px;
                }}
                .mermaid svg {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    flowchart: {{
                        useMaxWidth: true,
                        htmlLabels: true,
                        curve: 'basis'
                    }},
                    themeVariables: {{
                        primaryColor: '#e1f5fe',
                        primaryTextColor: '#1565c0',
                        primaryBorderColor: '#2196f3',
                        lineColor: '#757575',
                        secondaryColor: '#f3e5f5',
                        tertiaryColor: '#fff3e0'
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # Streamlit 컴포넌트로 렌더링 (일부 버전에서 key 미지원)
        components.html(html_template, height=height)
    
    def create_workflow_summary_card(self, agent_sequence: List[str], route_decision: str, 
                                   collaboration: bool = False) -> str:
        """워크플로우 실행 요약 카드"""
        # 이모지 맵핑
        emoji_map = {
            "supervisor": "🎯",
            "sql_analyst": "🔍", 
            "ml_engineer": "🧠",
            "collaboration_manager": "🤝",
            "final_synthesizer": "📄"
        }
        
        sequence_str = " → ".join([f"{emoji_map.get(agent, '❓')} {agent}" for agent in agent_sequence])
        
        summary = f"""
        **🔄 실행 경로:** {sequence_str}
        
        **📊 라우팅 결정:** {route_decision.upper()}
        
        **🤝 협력 모드:** {'활성화' if collaboration else '비활성화'}
        
        **📈 단계 수:** {len(agent_sequence)}
        """
        
        return summary


def render_langgraph_workflow(
    agent_sequence: List[str] = None,
    route_decision: str = "",
    collaboration: bool = False,
    show_detailed: bool = False,
    use_pyppeteer: bool = False
) -> None:
    """LangGraph 워크플로우 렌더링 메인 함수"""
    
    visualizer = MermaidLangGraphVisualizer()
    
    # 탭으로 구성
    tab1, tab2, tab3 = st.tabs(["🔄 실행 경로", "📋 기본 워크플로우", "🔍 상세 플로우"])
    
    with tab1:
        if agent_sequence:
            st.markdown("### 🎯 실행된 워크플로우")
            
            # 실행 요약
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 실행 경로 다이어그램
                execution_diagram = visualizer.generate_execution_path_diagram(agent_sequence, route_decision)
                visualizer.render_mermaid_diagram(execution_diagram, height=500, key="execution_path", use_pyppeteer=use_pyppeteer)
            
            with col2:
                # 요약 정보
                st.markdown("#### 📊 실행 요약")
                summary = visualizer.create_workflow_summary_card(agent_sequence, route_decision, collaboration)
                st.markdown(summary)
        else:
            st.info("워크플로우가 실행되면 실행 경로가 여기에 표시됩니다.")
    
    with tab2:
        st.markdown("### 🏗️ 기본 LangGraph 워크플로우")
        basic_diagram = visualizer.generate_basic_workflow()
        visualizer.render_mermaid_diagram(basic_diagram, height=600, key="basic_workflow", use_pyppeteer=use_pyppeteer)
    
    with tab3:
        if show_detailed:
            st.markdown("### 🔍 상세 처리 플로우")
            detailed_diagram = visualizer.generate_detailed_flow_diagram()
            visualizer.render_mermaid_diagram(detailed_diagram, height=700, key="detailed_flow", use_pyppeteer=use_pyppeteer)
        else:
            st.info("상세 플로우는 요청 시에만 표시됩니다.")


def render_langgraph_native_diagram(workflow_app, height: int = 600, key: str = "native_graph") -> bool:
    """LangGraph 내장 기능으로 실제 워크플로우 다이어그램 생성"""
    try:
        # 🔥 실제 컴파일된 LangGraph에서 Mermaid 코드 추출
        graph = workflow_app.app.get_graph()
        mermaid_code = graph.draw_mermaid()
        
        # 스타일 개선 (단순 구조용)
        enhanced_mermaid = f"""
        {mermaid_code}
        
        %% 🎨 향상된 스타일링
        classDef supervisor fill:#e1f5fe,stroke:#01579b,stroke-width:3px,color:#01579b
        classDef sqlAgent fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
        classDef mlAgent fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100
        classDef synthesizer fill:#fce4ec,stroke:#ad1457,stroke-width:2px,color:#880e4f
        classDef startEnd fill:#f1f8e9,stroke:#388e3c,stroke-width:3px,color:#1b5e20
        
        %% 노드별 클래스 적용
        class supervisor supervisor
        class sql_analyst sqlAgent
        class ml_engineer mlAgent  
        class final_synthesizer synthesizer
        class __start__,__end__ startEnd
        """
        
        # 원본 Mermaid 코드 정보도 표시
        st.markdown("#### 📋 실제 LangGraph 구조")
        st.info("✨ 이것은 코드에서 자동 생성된 **실제 워크플로우 구조**입니다!")
        
        # Streamlit에서 렌더링
        visualizer = MermaidLangGraphVisualizer()
        visualizer.render_mermaid_diagram(enhanced_mermaid, height=height, key=key)
        
        # 원본 Mermaid 코드도 표시 (확장 가능)
        with st.expander("🔍 원본 Mermaid 코드 보기"):
            st.code(mermaid_code, language="mermaid")
            st.markdown("**💡 사용법:** 위 코드를 복사해서 [Mermaid Live Editor](https://mermaid.live)에서도 확인할 수 있습니다!")
        
        return True
        
    except Exception as e:
        st.error(f"❌ LangGraph 네이티브 다이어그램 생성 오류: {e}")
        st.code(str(e))
        return False


def render_langgraph_png_diagram(workflow_app, height: int = 600) -> bool:
    """LangGraph 내장 PNG 생성 기능 사용 (간단한 방식)"""
    try:
        # 컴파일된 워크플로우에서 그래프 가져오기
        graph = workflow_app.app.get_graph()
        
        # 🔥 API 없이 직접 PNG 생성 (훨씬 간단!)
        png_bytes = graph.draw_mermaid_png()
        
        # Streamlit에서 이미지 표시
        st.image(png_bytes, caption="🎯 LangGraph 실제 워크플로우 구조", use_column_width=True)
        
        return True
        
    except Exception as e:
        st.error(f"❌ LangGraph PNG 다이어그램 생성 오류: {e}")
        # 상세한 에러 정보 표시
        st.code(str(e))
        return False


def create_mermaid_workflow_images():
    """Mermaid 워크플로우 이미지 생성 (기존 함수와 호환)"""
    try:
        visualizer = MermaidLangGraphVisualizer()
        
        # 기본 워크플로우 렌더링
        st.markdown("### 🔄 LangGraph 멀티에이전트 워크플로우")
        basic_diagram = visualizer.generate_basic_workflow()
        visualizer.render_mermaid_diagram(basic_diagram, height=600, key="workflow_main")
        
        return {"success": True, "message": "Mermaid 워크플로우가 생성되었습니다."}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_langgraph_structure(workflow_app):
    """LangGraph 구조 분석 및 표시"""
    try:
        graph = workflow_app.app.get_graph()
        
        # 노드와 엣지 분석
        nodes = list(graph.nodes.keys())
        edges = []
        
        # 엣지 정보 수집
        for node_id, node_data in graph.nodes.items():
            if hasattr(node_data, 'edges') and node_data.edges:
                for edge in node_data.edges:
                    edges.append((node_id, edge))
        
        # 구조 정보 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏗️ 노드 구성")
            for i, node in enumerate(nodes, 1):
                if node == "__start__":
                    st.write(f"{i}. 🟢 **{node}** (시작점)")
                elif node == "__end__":
                    st.write(f"{i}. 🔴 **{node}** (종료점)")
                elif "supervisor" in node:
                    st.write(f"{i}. 🧠 **{node}** (라우팅)")
                elif "sql" in node:
                    st.write(f"{i}. 🔍 **{node}** (데이터 분석)")
                elif "ml" in node:
                    st.write(f"{i}. 🤖 **{node}** (모델 개발)")
                elif "collaboration" in node:
                    st.write(f"{i}. 🤝 **{node}** (협력 관리)")
                elif "synthesizer" in node:
                    st.write(f"{i}. 📄 **{node}** (결과 종합)")
                else:
                    st.write(f"{i}. ⚙️ **{node}**")
        
        with col2:
            st.markdown("#### 🔄 조건부 엣지")
            st.write("**Supervisor 분기:**")
            st.write("• `sql` → SQL Analyst")
            st.write("• `ml` → ML Engineer")
        
        return True
        
    except Exception as e:
        st.error(f"구조 분석 오류: {e}")
        return False


def create_langgraph_native_workflow(workflow_app):
    """LangGraph 네이티브 워크플로우 생성 및 분석"""
    try:
        st.markdown("### 🎯 실제 LangGraph 워크플로우")
        
        # 워크플로우 정보 표시
        st.success("✨ 코드에서 **자동 생성된 실제 LangGraph 구조**입니다!")
        
        # 탭으로 다양한 방식 제공
        tab1, tab2, tab3 = st.tabs(["🔄 Mermaid 다이어그램", "🖼️ PNG 이미지", "🏗️ 구조 분석"])
        
        with tab1:
            st.markdown("#### 📝 실제 LangGraph → Mermaid 변환")
            success = render_langgraph_native_diagram(workflow_app, key="native_mermaid")
            if not success:
                st.info("네이티브 Mermaid 생성에 실패했습니다. 수동 다이어그램을 표시합니다.")
                visualizer = MermaidLangGraphVisualizer()
                basic_diagram = visualizer.generate_basic_workflow()
                visualizer.render_mermaid_diagram(basic_diagram, height=600, key="fallback_mermaid")
        
        with tab2:
            st.markdown("#### 🔥 `graph.draw_mermaid_png()` 직접 사용")
            success = render_langgraph_png_diagram(workflow_app)
            if not success:
                st.warning("네이티브 PNG 생성에 실패했습니다.")
        
        with tab3:
            st.markdown("#### 🏗️ LangGraph 구조 상세 분석")
            analyze_langgraph_structure(workflow_app)
        
        return {"success": True, "message": "LangGraph 네이티브 워크플로우가 생성되었습니다."}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def demo_simple_langgraph():
    """간단한 LangGraph 데모 (당신 예시 스타일)"""
    if not LANGGRAPH_AVAILABLE:
        st.error("LangGraph가 설치되지 않았습니다.")
        return
    
    try:
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict
        
        # 간단한 상태 정의
        class SimpleState(TypedDict):
            message: str
        
        # 간단한 노드들
        def node_1(state):
            return {"message": "Node 1 processed"}
        
        def node_2(state):
            return {"message": "Node 2 processed"}
        
        def decide_next(state):
            return "node_2"  # 간단한 라우팅
        
        # 그래프 빌더 초기화
        builder = StateGraph(SimpleState)
        
        # 노드 추가
        builder.add_node("node_1", node_1)
        builder.add_node("node_2", node_2)
        
        # 엣지 추가
        builder.add_edge(START, "node_1")
        builder.add_conditional_edges("node_1", decide_next)
        builder.add_edge("node_2", END)
        
        # 그래프 컴파일
        graph = builder.compile()
        
        # 🔥 당신 방식 그대로: graph.get_graph().draw_mermaid_png()
        st.markdown("#### 🎯 Demo: 간단한 LangGraph")
        png_bytes = graph.get_graph().draw_mermaid_png()
        st.image(png_bytes, caption="데모 LangGraph 구조", use_column_width=True)
        
        st.success("✅ 간단한 데모 성공!")
        
    except Exception as e:
        st.error(f"❌ 데모 생성 오류: {e}")
        st.code(str(e))
