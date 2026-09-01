from fastapi import APIRouter, Depends, HTTPException

from src.config import get_settings
from src.schemas import SearchHit, SearchRequest
from src.services import embeddings as embeddings_service
from src.services import opensearch_service as oss
from src.services.auth import verify_api_key
from src.services.retrieval import hybrid_search

router = APIRouter(prefix="/api/v1/search", tags=["search"], dependencies=[Depends(verify_api_key)])


@router.post("", response_model=list[SearchHit])
def search(request: SearchRequest):
    top_k = request.top_k or get_settings().top_k
    try:
        if request.mode == "bm25":
            docs = oss.bm25_search(request.query, top_k=top_k)
            scores = {d["_id"]: d["_score"] for d in docs}
        else:
            query_vector = embeddings_service.embed_query(request.query)
            if request.mode == "vector":
                docs = oss.vector_search(query_vector, top_k=top_k)
                scores = {d["_id"]: d["_score"] for d in docs}
            else:
                docs = hybrid_search(request.query, query_vector, top_k=top_k)
                scores = {d["_id"]: d["_rrf_score"] for d in docs}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search backend unavailable: {exc}") from exc

    return [
        SearchHit(title=d.get("title", ""), section=d.get("section", ""), content=d["content"], score=scores[d["_id"]])
        for d in docs
    ]
