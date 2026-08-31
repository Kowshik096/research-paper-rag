from typing import TypedDict

MAX_ATTEMPTS = 2


class AgentState(TypedDict):
    question: str
    top_k: int
    attempt: int
    current_query: str
    retrieved_docs: list[dict]
    graded: list[dict]
    selected_docs: list[dict]
    grades_trace: list[dict]
    attempted_rewrite: bool
    rewritten_query: str | None
    initial_usable: bool | None
    final_usable: bool
    refused: bool
    refusal_reason: str
    answer: str


def initial_state(question: str, top_k: int) -> AgentState:
    return AgentState(
        question=question,
        top_k=top_k,
        attempt=0,
        current_query=question,
        retrieved_docs=[],
        graded=[],
        selected_docs=[],
        grades_trace=[],
        attempted_rewrite=False,
        rewritten_query=None,
        initial_usable=None,
        final_usable=False,
        refused=False,
        refusal_reason="",
        answer="",
    )


def route_after_guardrail(state: AgentState) -> str:
    return "refuse" if state.get("refused") else "retrieve"


def route_after_retrieve(state: AgentState) -> str:
    return "empty" if not state["retrieved_docs"] else "grade"


def route_after_grade(state: AgentState) -> str:
    from src.services import crag

    if crag.has_usable(state["graded"]):
        return "generate"
    if state["attempt"] + 1 < MAX_ATTEMPTS and state.get("initial_usable") is False:
        return "rewrite"
    return "generate"
