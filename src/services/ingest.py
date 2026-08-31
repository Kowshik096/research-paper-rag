import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import get_settings
from src.services import arxiv_client, embeddings, opensearch_service, pdf_parser
from src.services.chunker import split_text
from src.services.postgres import Chunk, Paper, get_session_factory

logger = logging.getLogger(__name__)


def ingest_papers(query: str, max_results: int) -> list[dict]:
    settings = get_settings()
    papers = arxiv_client.fetch_papers(query=query, max_results=max_results)
    ingested = []

    for paper in papers:
        try:
            if _paper_exists(paper["arxiv_id"]):
                logger.info("Skipping %s (already indexed)", paper["arxiv_id"])
                continue

            pdf_path = arxiv_client.download_pdf(paper["arxiv_id"], paper["pdf_url"])
            pages = pdf_parser.parse_pdf(str(pdf_path))
            sections = pdf_parser.pages_to_sections(pages)

            raw_chunks: list[tuple[str, str]] = []
            for section_name, section_text in sections:
                for piece in split_text(section_text, settings.chunk_size, settings.chunk_overlap):
                    raw_chunks.append((section_name, piece))
            if not raw_chunks:
                continue

            vectors = embeddings.embed_texts([c[1] for c in raw_chunks])

            paper_id = _upsert_paper(paper, str(pdf_path), len(raw_chunks))
            session = get_session_factory()()
            try:
                docs = []
                for idx, ((section_name, content), vector) in enumerate(zip(raw_chunks, vectors, strict=True)):
                    session.add(Chunk(paper_id=paper_id, chunk_index=idx, section=section_name, content=content))
                    docs.append(
                        {
                            "paper_id": paper_id,
                            "arxiv_id": paper["arxiv_id"],
                            "title": paper["title"],
                            "section": section_name,
                            "content": content,
                            "embedding": vector,
                        }
                    )
                session.commit()
            finally:
                session.close()

            opensearch_service.index_chunks(docs)
            ingested.append({"arxiv_id": paper["arxiv_id"], "title": paper["title"], "chunks": len(raw_chunks)})
            logger.info("Indexed %s with %d chunks", paper["arxiv_id"], len(raw_chunks))
        except Exception:
            logger.exception("Failed to ingest %s", paper.get("arxiv_id"))
            continue
    return ingested


def _paper_exists(arxiv_id: str) -> bool:
    session = get_session_factory()()
    try:
        return session.query(Paper.id).filter(Paper.arxiv_id == arxiv_id).first() is not None
    finally:
        session.close()


def _upsert_paper(meta: dict, pdf_path: str, chunk_count: int) -> int:
    stmt = (
        pg_insert(Paper.__table__)
        .values(
            arxiv_id=meta["arxiv_id"],
            title=meta["title"],
            authors=meta["authors"],
            abstract=meta["abstract"],
            published=meta["published"],
            pdf_path=pdf_path,
        )
        .on_conflict_do_update(index_elements=["arxiv_id"], set_={"title": meta["title"]})
        .returning(Paper.id)
    )
    session = get_session_factory()()
    try:
        paper_id = session.execute(stmt).scalar_one()
        session.commit()
        return paper_id
    finally:
        session.close()
