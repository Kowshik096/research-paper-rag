import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "gemini"
    zen_api_key: str = ""
    zen_base_url: str = "https://opencode.ai/zen/v1"
    zen_model: str = "big-pickle"
    zen_fallback_models: str = ""

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"
    embed_fallback_models: str = "gemini-embedding-2"
    embedding_provider: str = "gemini"
    local_embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768

    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 12
    hf_home: str = "./hf_cache"

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/ragdb"
    opensearch_url: str = "http://localhost:9200"
    redis_url: str = "redis://localhost:6379/0"

    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 6
    cache_ttl: int = 86400
    api_key: str = ""
    rate_limit_rpm: int = 60
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Resolve project hf_cache (./hf_cache relative to repo root)
    project_root = Path(__file__).resolve().parents[1]
    expected = (project_root / "hf_cache").resolve()
    # Fix stale HF_HOME (e.g. old P:\hf_cache) that no longer exists or is outside project
    current = os.getenv("HF_HOME")
    if current:
        # stale if points to non-existent P:\hf_cache or P:\Resume_Projects\hf_cache after move
        stale_paths = {r"P:\hf_cache", r"P:/hf_cache", r"P:\Resume_Projects\hf_cache", r"P:/Resume_Projects/hf_cache"}
        if current in stale_paths or not Path(current).exists():
            os.environ["HF_HOME"] = str(expected)
            os.environ["HF_HUB_CACHE"] = str(expected / "hub")
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(expected)
            os.environ["TRANSFORMERS_CACHE"] = str(expected / "hub")
            settings.hf_home = str(expected)
    elif settings.hf_home:
        hf_path = (
            (project_root / settings.hf_home).resolve()
            if not Path(settings.hf_home).is_absolute()
            else Path(settings.hf_home)
        )
        os.environ["HF_HOME"] = str(hf_path)
        os.environ.setdefault("HF_HUB_CACHE", str(hf_path / "hub"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_path))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_path / "hub"))
        settings.hf_home = str(hf_path)
    return settings
