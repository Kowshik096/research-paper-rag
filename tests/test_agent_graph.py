import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.agent_graph import (
    AgentState,
    route_after_grade,
    route_after_guardrail,
    route_after_retrieve,
)


def state(**kwargs) -> AgentState:
    base = {"question": "q", "top_k": 5, "attempt": 0, "current_query": "q", "retrieved_docs": [],
            "graded": [], "selected_docs": [], "grades_trace": [], "attempted_rewrite": False,
            "rewritten_query": None, "initial_usable": None, "final_usable": False,
            "refused": False, "refusal_reason": "", "answer": ""}
    base.update(kwargs)
    return base


def test_guardrail_route():
    assert route_after_guardrail(state(refused=True)) == "refuse"
    assert route_after_guardrail(state(refused=False)) == "retrieve"


def test_retrieve_route():
    assert route_after_retrieve(state(retrieved_docs=[])) == "empty"
    assert route_after_retrieve(state(retrieved_docs=[{"content": "x"}])) == "grade"


def test_grade_route_usable_goes_generate():
    s = state(graded=[{"grade": "correct"}], attempt=0)
    assert route_after_grade(s) == "generate"


def test_grade_route_rewrites_once_then_generates():
    first = state(graded=[{"grade": "incorrect"}], attempt=0, initial_usable=False)
    assert route_after_grade(first) == "rewrite"
    second = state(graded=[{"grade": "ambiguous"}], attempt=1, initial_usable=False)
    assert route_after_grade(second) == "generate"


def test_graph_compiles_and_mermaid_exportable():
    from src.services.agent_graph import build_agent_graph

    app = build_agent_graph()
    mermaid = app.get_graph().draw_mermaid()
    assert "guardrail" in mermaid
    assert "generate" in mermaid
