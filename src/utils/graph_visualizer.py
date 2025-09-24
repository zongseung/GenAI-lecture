"""
LangGraph 워크플로우 시각화 도구
"""

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any
import os
from pathlib import Path


class LangGraphVisualizer:
    """LangGraph 워크플로우를 이미지로 생성하는 클래스"""
    
    def __init__(self):
        if GRAPHVIZ_AVAILABLE:
            self.graph = graphviz.Digraph(comment='LangGraph Multi-Agent Workflow')
            self.graph.attr(rankdir='TB', size='12,8')
            self.graph.attr('node', shape='box', style='rounded,filled')
        else:
            self.graph = None
    
    def create_matplotlib_workflow(self, save_path: str = "langgraph_workflow_matplotlib.png") -> str:
        """matplotlib을 사용한 워크플로우 다이어그램 생성"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # 노드 정의 (x, y, width, height, label, color)
        nodes = [
            (4, 11, 2, 0.8, "START\nUser Request", '#e1f5fe'),
            (4, 9.5, 2, 0.8, "SUPERVISOR\nRequest Analysis", '#f3e5f5'),
            (1.5, 7.5, 2, 0.8, "SQL ANALYST\nData Analysis", '#e8f5e8'),
            (6.5, 7.5, 2, 0.8, "ML ENGINEER\nModel Development", '#fff3e0'),
            (4, 5.5, 2, 0.8, "SYNTHESIS\nResult Integration", '#fce4ec'),
            (4, 4, 2, 0.8, "END\nUser Response", '#f1f8e9')
        ]
        
        # 노드 그리기
        node_centers = {}
        for i, (x, y, w, h, label, color) in enumerate(nodes):
            # 박스 그리기
            rect = patches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            # 텍스트 추가
            ax.text(x + w/2, y + h/2, label, 
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold',
                   wrap=True)
            
            # 노드 중심점 저장
            node_centers[i] = (x + w/2, y + h/2)
        
        # 화살표 그리기
        arrows = [
            (0, 1, "Request"),  # START -> SUPERVISOR
            (1, 2, "Data\nAnalysis"),  # SUPERVISOR -> SQL
            (1, 3, "ML\nModeling"),  # SUPERVISOR -> ML
            (2, 4, "Analysis\nResult"),  # SQL -> SYNTHESIS
            (3, 4, "Model\nResult"),  # ML -> SYNTHESIS
            (4, 5, "Final\nResponse")   # SYNTHESIS -> END
        ]
        
        for start_idx, end_idx, label in arrows:
            start_x, start_y = node_centers[start_idx]
            end_x, end_y = node_centers[end_idx]
            
            # 화살표 그리기
            ax.annotate('', xy=(end_x, end_y + 0.4), xytext=(start_x, start_y - 0.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
            
            # 라벨 추가
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center',
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8))
        
        # 제목 추가
        ax.text(5, 12.5, 'LangGraph Multi-Agent Workflow', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # 저장
        output_path = Path(save_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(save_path):
            return save_path
        
        return None
        
    def create_agent_workflow_diagram(self, save_path: str = "langgraph_workflow.png") -> str:
        """에이전트 워크플로우 다이어그램 생성"""
        
        # 노드 스타일 정의
        node_styles = {
            'start': {'fillcolor': '#e1f5fe', 'color': '#01579b'},
            'supervisor': {'fillcolor': '#f3e5f5', 'color': '#4a148c'},
            'sql_agent': {'fillcolor': '#e8f5e8', 'color': '#1b5e20'},
            'ml_agent': {'fillcolor': '#fff3e0', 'color': '#e65100'},
            'synthesis': {'fillcolor': '#fce4ec', 'color': '#880e4f'},
            'end': {'fillcolor': '#f1f8e9', 'color': '#33691e'}
        }
        
        # 노드 추가
        self.graph.node('start', 'START\n사용자 요청', **node_styles['start'])
        self.graph.node('supervisor', 'SUPERVISOR\n🤖 요청 분석\n에이전트 선택', **node_styles['supervisor'])
        self.graph.node('sql_agent', 'SQL ANALYST\n📊 데이터 분석\nSQL 쿼리 실행', **node_styles['sql_agent'])
        self.graph.node('ml_agent', 'ML ENGINEER\n🧠 모델 개발\n예측 분석', **node_styles['ml_agent'])
        self.graph.node('synthesis', 'SYNTHESIS\n📝 결과 종합\n최종 리포트', **node_styles['synthesis'])
        self.graph.node('end', 'END\n사용자에게 응답', **node_styles['end'])
        
        # 엣지 추가 (화살표)
        self.graph.edge('start', 'supervisor', label='요청')
        self.graph.edge('supervisor', 'sql_agent', label='데이터 분석\n필요')
        self.graph.edge('supervisor', 'ml_agent', label='ML 모델링\n필요')
        self.graph.edge('sql_agent', 'synthesis', label='분석 결과')
        self.graph.edge('ml_agent', 'synthesis', label='모델 결과')
        self.graph.edge('synthesis', 'end', label='최종 응답')
        
        # 조건부 경로 표시
        self.graph.edge('supervisor', 'synthesis', label='단순 요청', style='dashed')
        self.graph.edge('sql_agent', 'ml_agent', label='추가 분석\n필요', style='dashed')
        self.graph.edge('ml_agent', 'sql_agent', label='데이터 추가\n요청', style='dashed')
        
        # 파일 저장
        output_path = Path(save_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # PNG 형식으로 렌더링
            self.graph.render(str(output_path.with_suffix('')), format='png', cleanup=True)
            final_path = str(output_path.with_suffix('.png'))
            
            if os.path.exists(final_path):
                return final_path
            else:
                # 대체 경로 시도
                alt_path = str(output_path.with_suffix('')) + '.png'
                if os.path.exists(alt_path):
                    return alt_path
                    
        except Exception as e:
            print(f"그래프 생성 오류: {e}")
            # SVG로 대체 시도
            try:
                svg_path = str(output_path.with_suffix('.svg'))
                self.graph.render(str(output_path.with_suffix('')), format='svg', cleanup=True)
                return svg_path
            except:
                pass
        
        return None
    
    def create_detailed_flow_diagram(self, save_path: str = "detailed_workflow.png") -> str:
        """상세한 워크플로우 다이어그램 생성"""
        
        # 새로운 그래프 생성
        detailed_graph = graphviz.Digraph(comment='Detailed LangGraph Workflow')
        detailed_graph.attr(rankdir='TB', size='14,10')
        detailed_graph.attr('node', shape='box', style='rounded,filled')
        
        # 상세 노드들
        nodes = [
            ('user_input', '👤 사용자 입력\n"최근 가격 분석해줘"', '#e1f5fe'),
            ('parse_request', '🔍 요청 파싱\n의도 분석', '#f3e5f5'),
            ('route_decision', '🎯 라우팅 결정\nSQL vs ML vs Both', '#fff3e0'),
            ('sql_query', '💾 SQL 쿼리\n데이터 추출', '#e8f5e8'),
            ('data_analysis', '📊 데이터 분석\n통계/시각화', '#e8f5e8'),
            ('ml_model', '🧠 ML 모델\n예측/분류', '#fff3e0'),
            ('feature_eng', '⚙️ 피처 엔지니어링\n데이터 전처리', '#fff3e0'),
            ('model_train', '🏋️ 모델 훈련\n성능 평가', '#fff3e0'),
            ('result_merge', '🔄 결과 병합\nSQL + ML', '#fce4ec'),
            ('generate_report', '📝 리포트 생성\n최종 답변', '#fce4ec'),
            ('user_output', '📤 사용자 응답\n분석 결과', '#f1f8e9')
        ]
        
        # 노드 추가
        for node_id, label, color in nodes:
            detailed_graph.node(node_id, label, fillcolor=color, color='black')
        
        # 연결 관계
        edges = [
            ('user_input', 'parse_request'),
            ('parse_request', 'route_decision'),
            ('route_decision', 'sql_query', 'SQL 필요'),
            ('route_decision', 'ml_model', 'ML 필요'),
            ('sql_query', 'data_analysis'),
            ('ml_model', 'feature_eng'),
            ('feature_eng', 'model_train'),
            ('data_analysis', 'result_merge'),
            ('model_train', 'result_merge'),
            ('result_merge', 'generate_report'),
            ('generate_report', 'user_output')
        ]
        
        # 엣지 추가
        for edge in edges:
            if len(edge) == 3:
                detailed_graph.edge(edge[0], edge[1], label=edge[2])
            else:
                detailed_graph.edge(edge[0], edge[1])
        
        # 저장
        output_path = Path(save_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            detailed_graph.render(str(output_path.with_suffix('')), format='png', cleanup=True)
            final_path = str(output_path.with_suffix('.png'))
            
            if os.path.exists(final_path):
                return final_path
                
        except Exception as e:
            print(f"상세 그래프 생성 오류: {e}")
        
        return None
    
    def get_workflow_description(self) -> Dict[str, Any]:
        """워크플로우 설명 반환"""
        return {
            "workflow_type": "LangGraph Multi-Agent System",
            "agents": [
                {
                    "name": "Supervisor Agent",
                    "role": "요청 분석 및 에이전트 라우팅",
                    "capabilities": ["의도 파악", "에이전트 선택", "결과 통합"]
                },
                {
                    "name": "SQL Analyst Agent", 
                    "role": "데이터 분석 및 SQL 쿼리",
                    "capabilities": ["SQL 생성", "데이터 조회", "통계 분석"]
                },
                {
                    "name": "ML Engineer Agent",
                    "role": "머신러닝 모델 개발",
                    "capabilities": ["모델 훈련", "예측", "성능 평가"]
                }
            ],
            "flow_steps": [
                "1. 사용자 요청 수신",
                "2. Supervisor가 요청 분석",
                "3. 적절한 에이전트로 라우팅", 
                "4. 각 에이전트가 작업 수행",
                "5. 결과를 Supervisor가 통합",
                "6. 최종 응답 생성"
            ]
        }


def create_workflow_images():
    """워크플로우 이미지들 생성"""
    visualizer = LangGraphVisualizer()
    
    # matplotlib 버전 우선 사용
    basic_path = visualizer.create_matplotlib_workflow("static/langgraph_workflow.png")
    
    # GraphViz 사용 가능하면 상세 워크플로우도 생성
    detailed_path = None
    if GRAPHVIZ_AVAILABLE:
        try:
            detailed_path = visualizer.create_detailed_flow_diagram("static/detailed_workflow.png")
        except Exception:
            pass
    
    return {
        "basic_workflow": basic_path,
        "detailed_workflow": detailed_path,
        "description": visualizer.get_workflow_description(),
        "matplotlib_used": True
    }


if __name__ == "__main__":
    # 테스트 실행
    results = create_workflow_images()
    print("워크플로우 이미지 생성 완료:")
    for key, path in results.items():
        if isinstance(path, str):
            print(f"  {key}: {path}")
