import threading
from functools import lru_cache

from src.config import get_settings

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(get_settings().rerank_model)
    return _model


@lru_cache(maxsize=1)
def _rerank_available() -> bool:
    try:
        get_model()
        return True
    except ImportError:
        print("sentence-transformers not installed -> reranking disabled")
        return False
    except Exception as exc:
        print(f"reranker load failed ({str(exc)[:100]}) -> keeping fusion order")
        return False


def rerank(query: str, docs: list[dict], top_k: int) -> list[dict]:
    """Re-score retrieved chunks with a local cross-encoder; returns top_k reordered by relevance."""
    settings = get_settings()
    if not settings.rerank_enabled or not _rerank_available() or len(docs) <= 1:
        return docs[:top_k]

    model = get_model()
    pairs = [(query, d["content"][:1500]) for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in ranked][:top_k]
