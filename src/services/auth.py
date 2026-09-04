import time
from functools import wraps

from fastapi import Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings

_rate_limit_store: dict[str, list[float]] = {}


def _is_rate_limited_memory(client_ip: str, rpm: int) -> bool:
    now = time.time()
    window_start = now - 60
    timestamps = [t for t in _rate_limit_store.get(client_ip, []) if t > window_start]

    # Bounded cleanup if table grows large
    if len(_rate_limit_store) > 1000:
        stale_keys = [k for k, v in _rate_limit_store.items() if not v or max(v) <= window_start]
        for k in stale_keys:
            _rate_limit_store.pop(k, None)

    if len(timestamps) >= rpm:
        _rate_limit_store[client_ip] = timestamps
        return True
    timestamps.append(now)
    _rate_limit_store[client_ip] = timestamps
    return False


def _check_rate_limit(client_ip: str, rpm: int) -> bool:
    try:
        from src.services.cache import get_client

        r = get_client()
        key = f"ratelimit:{client_ip}"
        now = time.time()
        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, now - 60)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, 65)
        results = pipe.execute()
        count = results[1]
        return count >= rpm
    except Exception:
        return _is_rate_limited_memory(client_ip, rpm)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        if _check_rate_limit(client_ip, self.rpm):
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {self.rpm} requests/minute."},
            )

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
