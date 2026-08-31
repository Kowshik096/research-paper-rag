import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.llm import build_context


def test_build_context_numbers_docs():
    docs = [
        {"title": "Paper A", "section": "intro", "content": "alpha"},
        {"title": "Paper B", "section": "body", "content": "beta"},
    ]
    context = build_context(docs)
    assert "[1] Paper A (intro)" in context
    assert "[2] Paper B (body)" in context
    assert "alpha" in context and "beta" in context
