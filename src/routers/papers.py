from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.schemas import IngestRequest, PaperOut
from src.services.auth import verify_api_key
from src.services.ingest import ingest_papers
from src.services.postgres import get_session_factory

router = APIRouter(prefix="/api/v1/papers", tags=["papers"], dependencies=[Depends(verify_api_key)])


def get_db():
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@router.post("/ingest")
def ingest(request: IngestRequest):
    results = ingest_papers(query=request.query, max_results=request.max_results)
    if not results:
        raise HTTPException(status_code=409, detail="No new papers ingested (already indexed or fetch failed)")
    return {"ingested": results}


@router.get("", response_model=list[PaperOut])
def list_papers(db: Session = Depends(get_db)):
    from src.services.postgres import Paper

    papers = db.query(Paper).order_by(Paper.published.desc()).all()
    return [
        PaperOut(
            arxiv_id=p.arxiv_id,
            title=p.title,
            authors=p.authors,
            published=p.published,
            chunk_count=len(p.chunks),
        )
        for p in papers
    ]
