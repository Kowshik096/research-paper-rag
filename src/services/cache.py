import hashlib
import json

import redis

from src.config import get_settings

_client = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def cache_key(prefix: str, payload: dict) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{prefix}:{digest}"


def get_cached(prefix: str, payload: dict) -> dict | None:
    key = cache_key(prefix, payload)
    value = get_client().get(key)
    return json.loads(value) if value else None


def set_cached(prefix: str, payload: dict, value: dict) -> None:
    ttl = get_settings().cache_ttl
    get_client().set(cache_key(prefix, payload), json.dumps(value), ex=ttl)
