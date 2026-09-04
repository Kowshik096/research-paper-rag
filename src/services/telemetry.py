import logging
import os
from contextlib import contextmanager, suppress
from functools import wraps
from typing import Any, Generator

_OTEL_AVAILABLE = True
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ImportError:
    _OTEL_AVAILABLE = False
    metrics = None  # type: ignore
    trace = None  # type: ignore

_log = logging.getLogger(__name__)

_tracer = None
_meter = None

_request_counter = None
_request_latency = None
_cache_hits = None
_cache_misses = None
_tokens_generated = None
_retrieval_latency = None
_embedding_latency = None

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
except ImportError:
    FastAPIInstrumentor = None
    HTTPXClientInstrumentor = None
    LoggingInstrumentor = None
    PsycopgInstrumentor = None
    RedisInstrumentor = None


def init_telemetry(app=None) -> None:
    global _tracer, _meter
    global _request_counter, _request_latency, _cache_hits, _cache_misses
    global _tokens_generated, _retrieval_latency, _embedding_latency

    if not _OTEL_AVAILABLE or trace is None or metrics is None:
        _log.info("OpenTelemetry packages not available; running with no-op telemetry")
        return

    service_name = "rag-assistant"
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    try:
        resource = Resource.create({"service.name": service_name})

        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")))
        trace.set_tracer_provider(trace_provider)
        _tracer = trace.get_tracer(service_name)

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics"),
            export_interval_millis=30_000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(service_name)

        _request_counter = _meter.create_counter("http_requests_total", description="Total HTTP requests", unit="1")
        _request_latency = _meter.create_histogram(
            "http_request_duration_seconds", description="HTTP request latency", unit="s"
        )
        _cache_hits = _meter.create_counter("cache_hits_total", description="Cache hits", unit="1")
        _cache_misses = _meter.create_counter("cache_misses_total", description="Cache misses", unit="1")
        _tokens_generated = _meter.create_counter(
            "tokens_generated_total", description="LLM tokens generated", unit="1"
        )
        _retrieval_latency = _meter.create_histogram(
            "retrieval_duration_seconds", description="Retrieval latency", unit="s"
        )
        _embedding_latency = _meter.create_histogram(
            "embedding_duration_seconds", description="Embedding generation latency", unit="s"
        )
    except Exception as exc:
        _log.warning("Failed to initialize OTLP providers (%s); continuing with no-op telemetry", exc)
        return

    if app and FastAPIInstrumentor:
        try:
            FastAPIInstrumentor.instrument_app(app)
        except Exception as exc:
            _log.warning("FastAPI instrumentation failed: %s", exc)

    if HTTPXClientInstrumentor:
        with suppress(Exception):
            HTTPXClientInstrumentor().instrument()
    if RedisInstrumentor:
        with suppress(Exception):
            RedisInstrumentor().instrument()
    if PsycopgInstrumentor:
        with suppress(Exception):
            PsycopgInstrumentor().instrument()
    if LoggingInstrumentor:
        with suppress(Exception):
            LoggingInstrumentor().instrument(set_logging_format=True)

    _log.info("OpenTelemetry initialized (endpoint=%s, available=%s)", otlp_endpoint, _OTEL_AVAILABLE)


def get_tracer():
    if _tracer is None and _OTEL_AVAILABLE:
        init_telemetry()
    return _tracer


def get_meter():
    if _meter is None and _OTEL_AVAILABLE:
        init_telemetry()
    return _meter


@contextmanager
def span(name: str, **attrs) -> Generator[Any, None, None]:
    tracer = get_tracer()
    if tracer is not None:
        with tracer.start_as_current_span(name, attributes=attrs) as s:
            yield s
    else:
        yield None


def record_request(method: str, path: str, status: int, duration: float) -> None:
    if _request_counter:
        _request_counter.add(1, {"method": method, "path": path, "status": str(status)})
    if _request_latency:
        _request_latency.record(duration, {"method": method, "path": path})


def record_cache(hit: bool) -> None:
    if hit and _cache_hits:
        _cache_hits.add(1)
    elif not hit and _cache_misses:
        _cache_misses.add(1)


def record_tokens(count: int, model: str) -> None:
    if _tokens_generated:
        _tokens_generated.add(count, {"model": model})


def record_retrieval(duration: float, mode: str) -> None:
    if _retrieval_latency:
        _retrieval_latency.record(duration, {"mode": mode})


def record_embedding(duration: float, model: str) -> None:
    if _embedding_latency:
        _embedding_latency.record(duration, {"model": model})


def traced(name: str | None = None, **attrs):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with span(span_name, **attrs):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with span(span_name, **attrs):
                return await func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
