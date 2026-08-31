import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.services.embeddings as emb


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "CACHE_PATH", tmp_path / "cache.sqlite3")
    conn = emb._cache_conn()
    try:
        key = emb._cache_key("model", 768, "hello world")
        assert emb._load_cached(conn, [key]) == {}
        conn.execute("INSERT INTO cache (key, vector) VALUES (?, ?)", (key, "[0.1, 0.2]"))
        conn.commit()
        assert emb._load_cached(conn, [key]) == {key: [0.1, 0.2]}
        assert emb._load_cached(conn, ["missing"]) == {}
    finally:
        conn.close()


def test_cache_key_deterministic_and_sensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "CACHE_PATH", tmp_path / "cache.sqlite3")
    k1 = emb._cache_key("m", 768, "text")
    k2 = emb._cache_key("m", 768, "text")
    k3 = emb._cache_key("m", 1536, "text")
    k4 = emb._cache_key("other", 768, "text")
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_embed_texts_empty():
    assert emb.embed_texts([]) == []
