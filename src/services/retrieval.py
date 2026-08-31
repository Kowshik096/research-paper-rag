def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_k: int = 6) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("_id") or doc.get("arxiv_id", "") + ":" + str(doc.get("chunk_index", "")))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(doc_id, doc)
    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
    fused = []
    for doc_id in ranked_ids:
        doc = dict(docs[doc_id])
        doc["_rrf_score"] = scores[doc_id]
        fused.append(doc)
    return fused


def hybrid_search(
    query: str, query_embedding: list[float], top_k: int, index_name: str | None = None
) -> list[dict]:
    from src.services import opensearch_service as oss

    kwargs = {} if index_name is None else {"index_name": index_name}
    bm25_results = oss.bm25_search(query, top_k=top_k * 2, **kwargs)
    vector_results = oss.vector_search(query_embedding, top_k=top_k * 2, **kwargs)
    return reciprocal_rank_fusion([bm25_results, vector_results], top_k=top_k)
