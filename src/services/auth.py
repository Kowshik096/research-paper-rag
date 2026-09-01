import time
from collections import defaultdict
from functools import wraps

from fastapi import Depends, Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60

        _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if t > window_start]

        if len(_rate_limit_store[client_ip]) >= self.rpm:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 60 requests/minute.")

        _rate_limit_store[client_ip].append(now)
        return await call_next(request)


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    settings = get_settings()
    expected = settings.api_key
    if not expected:
        return "dev"  # No key configured, allow in dev
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def require_api_key(func):
    @wraps(func)
    async def wrapper(*args, api_key: str = Depends(verify_api_key), **kwargs):
        return await func(*args, api_key=api_key, **kwargs)
    return wrapper
