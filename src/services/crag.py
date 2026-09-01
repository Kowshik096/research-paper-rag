import json
import re
from collections.abc import Iterator

from src.config import get_settings
from src.services.llm import _stream_gemini, _stream_openai_compatible

VERDICTS = ("correct", "ambiguous", "incorrect")

GRADE_PROMPT = """You are a retrieval relevance grader for a research-paper RAG system.

QUESTION: {question}

DOCUMENT:
{document}

Decide whether this document contains information needed to answer the question.
Reply with ONLY a JSON object: {{"verdict": "<verdict>", "reason": "<max 10 words>"}}
where <verdict> is one of: correct (directly useful), ambiguous (partially related), incorrect (unrelated).
"""

REWRITE_PROMPT = """The following question retrieved no relevant research-paper excerpts.

QUESTION: {question}

GRADED DOCUMENTS (all judged irrelevant):
{documents}

Rewrite the question to improve keyword and semantic retrieval over an arXiv corpus of
ocean/atmosphere/land-science papers. Expand abbreviations, add likely technical synonyms,
keep it a single sentence. Reply with only the rewritten question.
"""


def _collect(tokens: Iterator[str]) -> str:
    return "".join(tokens)


def _llm_text(prompt: str) -> str:
    if get_settings().llm_provider == "zen":
        return _collect(_stream_openai_compatible(prompt, "Respond now."))
    return _collect(_stream_gemini(prompt, "Respond now."))


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else text


def grade_documents(question: str, docs: list[dict]) -> list[dict]:
    graded = []
    for i, doc in enumerate(docs):
        verdict, reason = "ambiguous", "grader unavailable"
        try:
            response = _llm_text(GRADE_PROMPT.format(question=question, document=doc["content"][:2500]))
            parsed = json.loads(_extract_json(response))
            candidate = str(parsed.get("verdict", "")).lower()
            if candidate in VERDICTS:
                verdict = candidate
                reason = str(parsed.get("reason", ""))[:100]
        except Exception as exc:
            print(f"grader failed on doc {i}: {str(exc)[:100]}")
        graded.append({**doc, "grade": verdict, "grade_reason": reason})
    return graded


def has_usable(graded_docs: list[dict]) -> bool:
    return any(d["grade"] == "correct" for d in graded_docs)


def select_context(graded_docs: list[dict]) -> list[dict]:
    correct = [d for d in graded_docs if d["grade"] == "correct"]
    if correct:
        return correct
    ambiguous = [d for d in graded_docs if d["grade"] == "ambiguous"]
    if ambiguous:
        return ambiguous
    return []


def rewrite_query(question: str, graded_docs: list[dict]) -> str:
    summaries = "\n".join(f"- ({d.get('grade')}) {d['content'][:150]}" for d in graded_docs[:5])
    rewritten = _llm_text(REWRITE_PROMPT.format(question=question, documents=summaries))
    cleaned = rewritten.strip().strip('"').split("\n")[0].strip()
    return cleaned or question
