from collections.abc import Iterator

from google.genai import types

from src.config import get_settings

SYSTEM_PROMPT = """You are a research assistant that answers questions strictly from the provided paper excerpts.

Rules:
1. Answer only using the context below.
2. Cite sources inline as [1], [2] matching the numbered excerpts.
3. If the context does not contain the answer, reply exactly: "I could not find this in the indexed papers."
4. Be concise and technically precise.

Context:
{context}
"""


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = f"[{i}] {doc.get('title', 'Untitled')} ({doc.get('section', 'body')})"
        parts.append(f"{source}\n{doc['content']}")
    return "\n\n".join(parts)


def _stream_gemini(prompt: str, question: str) -> Iterator[str]:
    settings = get_settings()
    client = _gemini_client()
    response = client.models.generate_content_stream(
        model=settings.gemini_chat_model,
        contents=question,
        config=types.GenerateContentConfig(system_instruction=prompt),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def _gemini_client():
    from src.services.ai_client import get_gemini_client

    return get_gemini_client()


def _stream_openai_compatible(prompt: str, question: str) -> Iterator[str]:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.zen_api_key, base_url=settings.zen_base_url)
    candidates = [settings.zen_model] + [
        m.strip() for m in settings.zen_fallback_models.split(",") if m.strip()
    ]
    last_error: Exception | None = None
    for model_id in candidates:
        emitted = False
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                stream=True,
            )
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    emitted = True
                    yield event.choices[0].delta.content
            return
        except Exception as exc:
            last_error = exc
            print(f"model {model_id} failed ({str(exc)[:120]}), trying next...")
            if emitted:
                raise
            continue
    raise RuntimeError(f"All Zen models failed; last error: {last_error}")


def stream_answer(question: str, docs: list[dict]) -> Iterator[str]:
    settings = get_settings()
    prompt = SYSTEM_PROMPT.format(context=build_context(docs))
    if settings.llm_provider == "zen":
        yield from _stream_openai_compatible(prompt, question)
    else:
        yield from _stream_gemini(prompt, question)


def answer(question: str, docs: list[dict]) -> str:
    return "".join(stream_answer(question, docs))
