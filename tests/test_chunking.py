import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.chunker import sliding_window_chunks


def test_window_basic():
    text = "a" * 3000
    chunks = sliding_window_chunks(text, chunk_size=1200, overlap=150)
    assert chunks[0] == text[:1200]
    assert all(len(c) <= 1200 for c in chunks)
    assert len(chunks) >= 3
    assert chunks[1] == text[1050:2250]


def test_window_short_text():
    assert sliding_window_chunks("hello world") == ["hello world"]
    assert sliding_window_chunks("") == []


def test_window_overlap_zero():
    text = "x" * 2500
    chunks = sliding_window_chunks(text, chunk_size=1200, overlap=0)
    assert len(chunks) == 3
