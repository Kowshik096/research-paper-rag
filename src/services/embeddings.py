import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from google import genai
from google.genai import types

from src.config import get_settings

_client = None
CACHE_PATH = Path("data/embedding_cache.sqlite3")
_RETRY_DELAY = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_settings().gemini_api_key)
    return _client


def _cache_conn() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
    return conn


def _cache_key(model: str, dim: int, text: str) -> str:
    return hashlib.sha256(f"{model}:{dim}:{text}".encode()).hexdigest()


def _load_cached(conn: sqlite3.Connection, keys: list[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for start in range(0, len(keys), 400):
        chunk = keys[start : start + 400]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(f"SELECT key, vector FROM cache WHERE key IN ({placeholders})", chunk).fetchall()
        found.update({k: json.loads(v) for k, v in rows})
    return found


_st_model = None


def _embed_local(texts: list[str], model_spec: str | None = None) -> list[list[float]]:
    global _st_model
    settings = get_settings()
    requested = model_spec.split(":", 1)[1] if model_spec and ":" in model_spec else settings.local_embed_model
    if _st_model is None or getattr(_st_model, "model_name", None) != requested:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(requested)
        _st_model.model_name = requested
    return _st_model.encode(texts, normalize_embeddings=True).tolist()


def _embed_batched(texts: list[str], batch_size: int, max_retries: int, model: str | None = None) -> list[list[float]]:
    settings = get_settings()
    model = model or settings.gemini_embedding_model
    client = get_client()
    effective_batch = 1 if "embedding-2" in model else batch_size
    vectors: list[list[float]] = []
    for start in range(0, len(texts), effective_batch):
        batch = texts[start : start + effective_batch]
        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=settings.embed_dim),
                )
                embs = [e.values for e in result.embeddings]
                if len(embs) != len(batch):
                    raise ValueError(f"API returned {len(embs)} embeddings for {len(batch)} inputs")
                vectors.extend(embs)
                break
            except Exception as exc:
                err = str(exc)
                if attempt == max_retries - 1:
                    raise
                if "PerDay" in err:
                    raise RuntimeError(
                        f"Daily embedding quota exhausted for {model}. "
                        "Re-run with a fallback model; cached chunks are reused automatically."
                    ) from exc
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    match = _RETRY_DELAY.search(err)
                    wait = min(float(match.group(1)) + 2, 300) if match else 60
                else:
                    wait = 2**attempt
                print(f"rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
        time.sleep(2)
    return vectors


def embed_texts(
    texts: list[str], batch_size: int = 16, max_retries: int = 8, model: str | None = None
) -> list[list[float]]:
    """Embed texts with a persistent disk cache: every unique (model, text) pair embeds at most once ever."""
    if not texts:
        return []
    settings = get_settings()
    model = model or settings.gemini_embedding_model

    def do_embed(chunk_texts: list[str]) -> list[list[float]]:
        if model.startswith("local:") or settings.embedding_provider == "local":
            return _embed_local(chunk_texts, model)
        return _embed_batched(chunk_texts, batch_size, max_retries, model)

    keys = [_cache_key(model, settings.embed_dim, t) for t in texts]

    conn = _cache_conn()
    try:
        cached = _load_cached(conn, keys)
        missing_texts = list({t for t, k in zip(texts, keys, strict=True) if k not in cached})
        if missing_texts:
            fresh = do_embed(missing_texts)
            fresh_keys = [_cache_key(model, settings.embed_dim, t) for t in missing_texts]
            conn.executemany(
                "INSERT OR REPLACE INTO cache (key, vector) VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in zip(fresh_keys, fresh, strict=True)],
            )
            conn.commit()
            cached = {**cached, **dict(zip(fresh_keys, fresh, strict=True))}
        return [cached[k] for k in keys]
    finally:
        conn.close()


def embed_query(text: str, model: str | None = None) -> list[float]:
    return embed_texts([text], model=model)[0]


def fallback_models() -> list[str]:
    settings = get_settings()
    chain = [settings.gemini_embedding_model]
    chain += [m.strip() for m in settings.embed_fallback_models.split(",") if m.strip()]
    seen: set[str] = set()
    unique = []
    for m in chain:
        if m and m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def cache_size() -> int:
    conn = _cache_conn()
    try:
        return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    finally:
        conn.close()
