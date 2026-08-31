import logging
from dataclasses import dataclass
from pathlib import Path

from src.services import embeddings, opensearch_service, pdf_parser
from src.services.chunker import sliding_window_chunks, split_text

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    name: str
    strategy: str
    size: int
    overlap: int


DEFAULT_MATRIX = [
    ChunkConfig("win_600_o150", "window", 600, 150),
    ChunkConfig("win_1200_o0", "window", 1200, 0),
    ChunkConfig("win_1200_o150", "window", 1200, 150),
    ChunkConfig("para_1200_o0", "paragraph", 1200, 0),
    ChunkConfig("para_1200_o150", "paragraph", 1200, 150),
    ChunkConfig("para_2000_o300", "paragraph", 2000, 300),
]


def chunk_text(text: str, config: ChunkConfig) -> list[str]:
    if config.strategy == "window":
        return sliding_window_chunks(text, config.size, config.overlap)
    return split_text(text, config.size, config.overlap)


def load_pdf_texts() -> dict[str, str]:
    texts = {}
    for pdf_path in sorted(Path("data/pdfs").glob("*.pdf")):
        pages = pdf_parser.parse_pdf(str(pdf_path))
        texts[pdf_path.stem] = "\n\n".join(p["text"] for p in pages)
    return texts


def build_benchmark_index(config: ChunkConfig, force: bool = False, embed_model: str | None = None) -> str:
    index_name = f"bench_{config.name}"
    if not force and opensearch_service.get_client().indices.exists(index=index_name):
        count = opensearch_service.get_client().count(index=index_name)["count"]
        if count > 0:
            logger.info("Reusing existing %s (%d docs)", index_name, count)
            return index_name

    opensearch_service.delete_index(index_name)
    opensearch_service.init_index(index_name)

    docs = []
    for stem, text in load_pdf_texts().items():
        pieces = chunk_text(text, config)
        vectors = embeddings.embed_texts(pieces, model=embed_model)
        for i, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
            docs.append(
                {
                    "paper_id": 0,
                    "arxiv_id": stem,
                    "title": stem,
                    "section": f"chunk_{i}",
                    "content": piece,
                    "embedding": vector,
                }
            )
    opensearch_service.index_chunks(docs, index_name=index_name)
    logger.info("Built %s with %d chunks", index_name, len(docs))
    return index_name
