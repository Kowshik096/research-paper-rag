import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.services.reranker as rr


class FakeModel:
    def predict(self, pairs):
        return [float(len(doc_content) % 7) for _query, doc_content in pairs]


def test_rerank_orders_by_score_desc_and_truncates(monkeypatch):
    monkeypatch.setattr(rr, "_model", FakeModel())
    docs = [
        {"content": "aa"},
        {"content": "bbbbbbb"},
        {"content": "aaa"},
        {"content": "aaaaa"},
    ]
    ranked = rr.rerank("query", docs, top_k=2)
    assert [d["content"] for d in ranked] == ["aaaaa", "aaa"]


def test_rerank_noop_single_doc(monkeypatch):
    monkeypatch.setattr(rr, "_model", None)
    doc = {"content": "only"}
    assert rr.rerank("q", [doc], top_k=5) == [doc]


def test_rerank_truncates_to_top_k(monkeypatch):
    monkeypatch.setattr(rr, "_model", FakeModel())
    docs = [{"content": "x" * n} for n in range(10)]
    assert len(rr.rerank("q", docs, top_k=3)) == 3
