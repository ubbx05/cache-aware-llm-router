"""Cache-aware router: OpenAI-compatible reverse proxy in front of N vLLM workers.

Design constraint that shapes everything here: Open WebUI must not be able to
tell the router exists. Same endpoints, same response shapes, and -- the part
that is easy to get wrong -- streaming responses forwarded byte-for-byte so
tokens appear in the UI as they are generated rather than in one block at the
end.
"""
from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

import config
from prefix_tracker import PrefixTracker, block_hashes, get_tokenizer
from strategies import NoHealthyWorker, RequestContext, build_strategy
from worker_metrics import MetricsPoller

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("router")
# One scrape line per worker per second drowns everything else during long runs.
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Router's own metrics --------------------------------------------------
# Exported so Prometheus can scrape the router alongside the workers. The
# decision-latency histogram is what backs the "router overhead is negligible"
# claim; without a measurement that sentence is not defensible.
ROUTED = Counter("router_requests_total", "Requests routed", ["worker", "strategy", "reason"])
DECISION_S = Histogram(
    "router_decision_seconds",
    "Time spent choosing a worker",
    buckets=(1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2),
)
CACHE_GAIN = Histogram(
    "router_cache_gain",
    "Router-believed prefix hit fraction of the chosen worker",
    buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)
UPSTREAM_ERRORS = Counter("router_upstream_errors_total", "Upstream failures", ["worker"])
WORKER_LOAD = Gauge("router_worker_load", "Load(w) as seen by the router", ["worker"])
WORKER_HEALTHY = Gauge("router_worker_healthy", "1 if worker is scrapeable", ["worker"])

HOP_BY_HOP = {
    "host", "content-length", "connection", "keep-alive",
    "transfer-encoding", "upgrade", "te", "trailers",
}

app = FastAPI(title="cache-aware-router")

state: dict[str, Any] = {}


@app.on_event("startup")
async def _startup() -> None:
    # Read timeout is None because generation can legitimately run for minutes;
    # a read timeout here would truncate long streams.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=None),
        limits=httpx.Limits(max_connections=config.PROXY_MAX_CONNECTIONS,
                            max_keepalive_connections=config.PROXY_MAX_CONNECTIONS),
    )
    # Deliberately a second client. If the poller shares a pool with proxied
    # traffic, heavy streaming starves the /metrics scrape, it times out, and
    # the router marks a healthy worker dead and starts returning 503s.
    poll_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.POLL_TIMEOUT_S),
        limits=httpx.Limits(max_connections=config.POLL_MAX_CONNECTIONS,
                            max_keepalive_connections=config.POLL_MAX_CONNECTIONS),
    )
    poller = MetricsPoller(config.WORKERS, poll_client)
    names = [w.name for w in config.WORKERS if w.enabled]
    tracker = PrefixTracker(names)
    strategy = build_strategy(config.STRATEGY, tracker)

    state.update(
        client=client,
        poll_client=poll_client,
        poller=poller,
        tracker=tracker,
        strategy=strategy,
        tokenizer=get_tokenizer(),
    )
    await poller.start()
    log.info(
        "router up | strategy=%s workers=%s alpha=%.3f beta=%.3f delta0=%.3f load_ref=%.1f",
        strategy.name, names, config.ALPHA, config.BETA, config.DELTA0, config.LOAD_REF,
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    with contextlib.suppress(Exception):
        await state["poller"].stop()
    with contextlib.suppress(Exception):
        await state["client"].aclose()
    with contextlib.suppress(Exception):
        await state["poll_client"].aclose()


# --- helpers ---------------------------------------------------------------

def _prompt_text(payload: dict) -> str:
    """Flatten a request into the string whose prefix determines cache reuse.

    Chat requests resend the whole conversation each turn, so the natural
    prefix is the serialised message list in order. Role markers are included
    because they are part of what the engine actually tokenises.
    """
    if "messages" in payload:
        parts = []
        for m in payload["messages"]:
            content = m.get("content", "")
            if isinstance(content, list):  # multimodal content blocks
                content = "".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            parts.append(f"<|{m.get('role', 'user')}|>{content}")
        return "".join(parts)
    prompt = payload.get("prompt", "")
    return prompt if isinstance(prompt, str) else "".join(prompt)


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
    }
    if config.UPSTREAM_API_KEY:
        headers["authorization"] = f"Bearer {config.UPSTREAM_API_KEY}"
    return headers


def _choose(payload: dict):
    tokenizer = state["tokenizer"]
    tracker: PrefixTracker = state["tracker"]
    strategy = state["strategy"]
    poller: MetricsPoller = state["poller"]

    text = _prompt_text(payload)
    hashes = block_hashes(tokenizer.encode(text))
    ctx = RequestContext(prompt_text=text, block_hashes=hashes)

    t0 = time.perf_counter()
    decision = strategy.select(ctx, poller.snapshot())
    DECISION_S.observe(time.perf_counter() - t0)

    # Belief about the chosen worker, computed here rather than read off the
    # Decision so it exists for every strategy (round_robin never computes one).
    # Must be read BEFORE record(), otherwise the tracker has already been told
    # about these blocks and would report a perfect hit every time.
    believed_blocks = tracker.matched_blocks(decision.worker.name, hashes)
    believed_tokens = believed_blocks * config.BLOCK_SIZE
    believed_frac = believed_blocks / len(hashes) if hashes else 0.0

    # Record before the response comes back: the worker will hold these blocks
    # from the moment it starts prefill, and concurrent arrivals should already
    # see the affinity.
    tracker.record(decision.worker.name, hashes)
    CACHE_GAIN.observe(decision.cache_gain)
    ROUTED.labels(decision.worker.name, strategy.name, decision.reason).inc()
    return decision, len(hashes), believed_tokens, believed_frac


async def _proxy_stream(worker_url: str, path: str, body: bytes,
                        headers: dict[str, str], worker_name: str) -> AsyncIterator[bytes]:
    client: httpx.AsyncClient = state["client"]
    poller: MetricsPoller = state["poller"]
    try:
        async with client.stream(
            "POST", f"{worker_url}{path}", content=body, headers=headers
        ) as upstream:
            # aiter_raw avoids any decoding/re-encoding of the SSE frames.
            async for chunk in upstream.aiter_raw():
                yield chunk
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE error
        UPSTREAM_ERRORS.labels(worker_name).inc()
        log.warning("stream failed on %s: %s", worker_name, exc)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
    finally:
        poller.mark_complete(worker_name)


async def _handle_completion(request: Request, path: str) -> Response:
    body = await request.body()
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    try:
        decision, n_blocks, believed_tokens, believed_frac = _choose(payload)
    except NoHealthyWorker:
        return JSONResponse({"error": "no healthy worker available"}, status_code=503)

    worker = decision.worker
    headers = _forward_headers(request)
    poller: MetricsPoller = state["poller"]
    poller.mark_dispatch(worker.name)

    # Reported per request so the tracker's belief can be checked against what
    # the engine actually reused (usage.prompt_tokens_details.cached_tokens).
    # The tracker's LRU is only an approximation of the engine's eviction, and
    # every cache_gain-driven routing result rests on that approximation being
    # roughly right -- so it needs measuring, not assuming.
    resp_headers = {
        "x-router-worker": worker.name,
        "x-router-reason": decision.reason,
        "x-router-believed-cached-tokens": str(believed_tokens),
        "x-router-believed-frac": f"{believed_frac:.4f}",
        "x-router-prompt-blocks": str(n_blocks),
        "x-router-block-size": str(config.BLOCK_SIZE),
    }

    if payload.get("stream"):
        return StreamingResponse(
            _proxy_stream(worker.url, path, body, headers, worker.name),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    client: httpx.AsyncClient = state["client"]
    try:
        upstream = await client.post(f"{worker.url}{path}", content=body, headers=headers)
    except Exception as exc:  # noqa: BLE001
        UPSTREAM_ERRORS.labels(worker.name).inc()
        poller.mark_complete(worker.name)
        return JSONResponse({"error": f"upstream failure: {exc}"}, status_code=502)

    poller.mark_complete(worker.name)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers=resp_headers,
    )


# --- OpenAI-compatible surface --------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _handle_completion(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await _handle_completion(request, "/v1/completions")


@app.get("/v1/models")
async def models() -> Response:
    """Answered from any healthy worker: the pool is homogeneous by design."""
    client: httpx.AsyncClient = state["client"]
    poller: MetricsPoller = state["poller"]
    for worker in poller.snapshot().healthy():
        try:
            r = await client.get(f"{worker.url}/v1/models", timeout=5.0)
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
        except Exception:  # noqa: BLE001 - try the next worker
            continue
    return JSONResponse({"error": "no healthy worker available"}, status_code=503)


# --- Observability ---------------------------------------------------------

@app.get("/metrics")
async def metrics() -> Response:
    poller: MetricsPoller = state["poller"]
    for name, s in poller.states.items():
        WORKER_LOAD.labels(name).set(s.load())
        WORKER_HEALTHY.labels(name).set(1 if s.healthy else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> JSONResponse:
    poller: MetricsPoller = state["poller"]
    healthy = poller.snapshot().healthy()
    return JSONResponse(
        {"status": "ok" if healthy else "degraded",
         "healthy_workers": [s.name for s in healthy]},
        status_code=200 if healthy else 503,
    )


@app.get("/router/state")
async def router_state() -> JSONResponse:
    """Human-readable dump for debugging routing decisions."""
    poller: MetricsPoller = state["poller"]
    tracker: PrefixTracker = state["tracker"]
    snap = poller.snapshot()
    return JSONResponse({
        "strategy": state["strategy"].name,
        "alpha": config.ALPHA,
        "beta": config.BETA,
        "delta": config.DELTA0 * (1.0 - snap.mean_kv_usage()),
        "mean_kv_usage": snap.mean_kv_usage(),
        "workers": [
            {
                "name": s.name,
                "url": s.url,
                "healthy": s.healthy,
                "waiting": s.num_requests_waiting,
                "running": s.num_requests_running,
                "pending_since_scrape": s.dispatched_since_scrape,
                "kv_cache_usage_perc": s.kv_cache_usage_perc,
                "load": s.load(),
                "tracked_blocks": tracker.size(s.name),
                "age_s": round(time.time() - s.last_scrape_ts, 2) if s.last_scrape_ts else None,
            }
            for s in poller.states.values()
        ],
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
