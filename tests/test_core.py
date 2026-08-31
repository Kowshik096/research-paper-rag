import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.chunker import split_text
from src.services.retrieval import reciprocal_rank_fusion


def test_split_text_respects_size():
    text = "word " * 1000
    chunks = split_text(text, chunk_size=1200, overlap=150)
    assert all(len(c) <= 1300 for c in chunks)
    assert len(chunks) >= 4


def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_rrf_prefers_docs_in_both_lists():
    doc_a = {"_id": "a", "content": "only bm25"}
    doc_b = {"_id": "b", "content": "both lists"}
    doc_c = {"_id": "c", "content": "only vector"}
    fused = reciprocal_rank_fusion([[doc_a, doc_b], [doc_b, doc_c]], top_k=3)
    assert fused[0]["_id"] == "b"


def test_rrf_limit():
    docs = [{"_id": str(i), "content": f"doc {i}"} for i in range(10)]
    fused = reciprocal_rank_fusion([docs], top_k=5)
    assert len(fused) == 5


def test_rrf_score_monotonic():
    doc_a = {"_id": "a"}
    doc_b = {"_id": "b"}
    fused = reciprocal_rank_fusion([[doc_a, doc_b]], top_k=2)
    assert fused[0]["_rrf_score"] > fused[1]["_rrf_score"]
