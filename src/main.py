import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.routers import ask, papers, search
from src.services.auth import RateLimitMiddleware
from src.services.cache import get_client as get_redis
from src.services.opensearch_service import get_client as get_oss
from src.services.opensearch_service import init_index
from src.services.postgres import get_engine, init_db

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    init_index()
    yield


app = FastAPI(title="Hybrid RAG Research Assistant", version="0.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
settings = get_settings()
if settings.api_key:
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)
app.include_router(papers.router)
app.include_router(search.router)
app.include_router(ask.router)


@app.get("/health")
def health() -> dict:
    checks = {"api": "ok"}
    try:
        get_engine().connect().close()
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"
    try:
        get_oss().ping()
        checks["opensearch"] = "ok"
    except Exception as exc:
        checks["opensearch"] = f"error: {exc}"
    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"
    checks["llm_model"] = get_settings().gemini_chat_model
    return checks
