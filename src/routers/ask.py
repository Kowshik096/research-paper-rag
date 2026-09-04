import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.config import get_settings
from src.schemas import AskRequest, AskResponse, Citation, CragTrace, GradeOut, SearchHit
from src.services import cache as cache_service
from src.services.agent_graph import build_agent_graph
from src.services.agent_routing import initial_state
from src.services.auth import verify_api_key
from src.services.telemetry import record_cache, record_tokens, traced

router = APIRouter(tags=["ask"], dependencies=[Depends(verify_api_key)])

_agent = build_agent_graph()


def to_citations(docs: list[dict]) -> list[Citation]:
    return [
        Citation(index=i, title=d.get("title", ""), section=d.get("section", "body"), snippet=d["content"][:300])
        for i, d in enumerate(docs, 1)
    ]


@traced("run_agent")
def run_agent(request: AskRequest) -> dict:
    settings = get_settings()
    result = _agent.invoke(initial_state(request.question, request.top_k or settings.top_k))
    return result


def _make_trace(result: dict) -> CragTrace:
    return CragTrace(
        attempted=result.get("attempted_rewrite", False),
        rewritten_query=result.get("rewritten_query"),
        initial_usable=bool(result.get("initial_usable")) if result.get("initial_usable") is not None else False,
        final_usable=bool(result.get("final_usable")),
        refused=result.get("refused", False),
        grades=[
            GradeOut(index=g["index"], title=g.get("title", ""), grade=g["grade"], reason=g.get("reason", ""))
            for g in result.get("grades_trace", [])
        ],
    )


def build_response(result: dict, cached: bool = False) -> AskResponse:
    return AskResponse(
        answer=result.get("answer", ""),
        citations=to_citations(result.get("selected_docs", [])),
        cached=cached,
        crag=_make_trace(result),
    )


@router.post("/api/v1/ask", response_model=AskResponse)
def ask(request: AskRequest):
    payload = {"question": request.question, "top_k": request.top_k}
    cached = cache_service.get_cached("ask", payload)
    if cached:
        record_cache(True)
        cached["cached"] = True
        return AskResponse(**cached)

    response = build_response(run_agent(request))
    record_cache(False)
    cache_service.set_cached(
        "ask",
        payload,
        {
            "answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "crag": response.crag.model_dump() if response.crag else None,
        },
    )
    return response


@router.post("/api/v1/ask/stream")
def ask_stream(request: AskRequest):
    def event_source() -> Iterator[str]:
        payload_key = {"question": request.question, "top_k": request.top_k}
        cached = cache_service.get_cached("ask", payload_key)
        result = None
        if not cached:
            try:
                result = run_agent(request)
            except Exception:
                yield _sse({"type": "error", "data": "Agent execution failed."})
                return

        hits = [
            SearchHit(title=d.get("title", ""), section=d.get("section", ""), content="", score=0)
            for d in (result.get("selected_docs", []) if result else [])
        ]
        yield _sse({"type": "sources", "data": [h.model_dump() for h in hits]})
        if result:
            yield _sse({"type": "crag", "data": _make_trace(result).model_dump()})
            yield _sse({"type": "guardrail", "data": {"refused": result.get("refused", False)}})

        answer_text = cached["answer"] if cached else (result.get("answer", "") if result else "")
        if not cached:
            record_tokens(len(answer_text), get_settings().gemini_chat_model)
        for token in split_tokens(answer_text):
            yield _sse({"type": "token", "data": token})

        if not cached and result is not None:
            cache_service.set_cached(
                "ask",
                payload_key,
                {
                    "answer": answer_text,
                    "citations": [c.model_dump() for c in to_citations(result.get("selected_docs", []))],
                    "crag": _make_trace(result).model_dump(),
                },
            )
        yield _sse({"type": "done", "data": ""})

    return StreamingResponse(event_source(), media_type="text/event-stream")


def split_tokens(text: str, size: int = 4) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"
