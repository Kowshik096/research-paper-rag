from src.config import get_settings

_model = None
_import_failed = False


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(get_settings().rerank_model)
    return _model


def rerank(query: str, docs: list[dict], top_k: int) -> list[dict]:
    """Re-score retrieved chunks with a local cross-encoder; returns top_k reordered by relevance."""
    global _import_failed
    settings = get_settings()
    if not settings.rerank_enabled or _import_failed or len(docs) <= 1:
        return docs[:top_k]
    try:
        model = get_model()
    except ImportError:
        _import_failed = True
        print("sentence-transformers not installed -> reranking disabled")
        return docs[:top_k]
    except Exception as exc:
        print(f"reranker load failed ({str(exc)[:100]}) -> keeping fusion order")
        return docs[:top_k]

    pairs = [(query, d["content"][:1500]) for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in ranked][:top_k]
