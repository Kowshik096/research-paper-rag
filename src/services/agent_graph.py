import json
import re
from typing import Any

from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.services import crag
from src.services import embeddings as embeddings_service
from src.services import opensearch_service as oss
from src.services.agent_routing import (
    AgentState,
    generate_node,
    route_after_grade,
    route_after_guardrail,
    route_after_retrieve,
)
from src.services.retrieval import hybrid_search

GUARDRAIL_PROMPT = """You are the domain gatekeeper of a research assistant whose knowledge base contains
arXiv papers on ocean, river, atmosphere, land sciences and adjacent earth-science topics.

QUESTION: {question}

Decide whether this is a genuine research or technical question that could plausibly be answered
from such a literature (including methodology, instruments, models, data analysis).
Casual chat, creative writing, personal advice, sports, celebrity gossip, coding help and
current-events questions are OUT of domain.
Reply with ONLY a JSON object: {{"in_domain": true}} or {{"in_domain": false}}
"""


def guardrail_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    try:
        response = _llm_text(GUARDRAIL_PROMPT.format(question=question))
        match = re.search(r"\{.*\}", response, re.DOTALL)
        in_domain = json.loads(match.group()).get("in_domain", True) if match else True
    except Exception as exc:
        print(f"guardrail failed (fail-open): {str(exc)[:100]}")
        in_domain = True
    if not in_domain:
        return {
            "refused": True,
            "refusal_reason": "Question appears to be outside the research domain of the indexed papers.",
        }
    return {"refused": False}


def retrieve_node(state: AgentState) -> dict[str, Any]:
    question = state["current_query"]
    top_k = state["top_k"]
    try:
        query_vector = embeddings_service.embed_query(question)
    except RuntimeError as exc:
        if "Daily" in str(exc):
            print("embedding quota exhausted -> falling back to BM25-only retrieval")
            docs = oss.bm25_search(question, top_k=reranker_candidates(top_k))
            return {"retrieved_docs": rerank(question, docs, top_k)}
        raise
    docs = hybrid_search(question, query_vector, top_k=reranker_candidates(top_k))
    return {"retrieved_docs": rerank(question, docs, top_k)}


def reranker_candidates(top_k: int) -> int:
    from src.config import get_settings

    return max(top_k, get_settings().rerank_candidates)


def rerank(question: str, docs: list[dict], top_k: int) -> list[dict]:
    from src.services import reranker

    return reranker.rerank(question, docs, top_k)


def grade_node(state: AgentState) -> dict[str, Any]:
    graded = crag.grade_documents(state["current_query"], state["retrieved_docs"])
    grades_trace = list(state.get("grades_trace") or [])
    offset = len(grades_trace)
    grades_trace += [
        {
            "index": offset + i + 1,
            "title": d.get("title", ""),
            "grade": d["grade"],
            "reason": d.get("grade_reason", ""),
        }
        for i, d in enumerate(graded)
    ]
    updates: dict[str, Any] = {"graded": graded, "grades_trace": grades_trace}
    if state.get("initial_usable") is None:
        updates["initial_usable"] = crag.has_usable(graded)
    return updates


def rewrite_node(state: AgentState) -> dict[str, Any]:
    rewritten = crag.rewrite_query(state["current_query"], state["graded"])
    return {
        "attempt": state["attempt"] + 1,
        "current_query": rewritten or state["question"],
        "rewritten_query": rewritten,
        "attempted_rewrite": True,
    }


def refuse_node(state: AgentState) -> dict[str, Any]:
    if state.get("refused"):
        return {"answer": GUARDRAIL_REFUSAL, "selected_docs": []}
    return {"answer": "No indexed papers matched this question.", "selected_docs": []}


GUARDRAIL_REFUSAL = (
    "This question appears to be outside the research domain of the indexed papers. "
    "Try asking about ocean, atmosphere, land or climate-related research."
)


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("empty", refuse_node)

    graph.set_entry_point("guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"refuse": "refuse", "retrieve": "retrieve"})
    graph.add_conditional_edges("retrieve", route_after_retrieve, {"empty": "empty", "grade": "grade"})
    graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)
    graph.add_edge("empty", END)
    return graph.compile()


def _llm_text(prompt: str) -> str:
    settings = get_settings()
    if settings.llm_provider == "zen":
        from src.services.llm import _stream_openai_compatible

        return "".join(_stream_openai_compatible(prompt, "Respond now."))
    from src.services.llm import _stream_gemini

    return "".join(_stream_gemini(prompt, "Respond now."))
