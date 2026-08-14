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
from strategies import Decision, NoHealthyWorker, RequestContext, build_strategy
from worker_metrics import MetricsPoller

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("router")
# One scrape line per worker per second drowns everything else during long runs.
logging.getLogger("httpx").setLevel(logging.WARNING)

# MUST match replay.py's CHUNK_SEP exactly, byte for byte. Only replay.py's
# --order per_worker_tree path inserts this sentinel between RAG chunks in
# the prompt; canonical/relevance still join with plain "\n\n" (unchanged,
# byte-identical to every prior run) so this constant existing here has zero
# effect on any already-measured strategy. It is the only way this router can
# tell where one chunk ends and the next begins in an already-flattened
# prompt string, which is what lets it rewrite the chunk order server-side
# without a second request round-trip.
CHUNK_SEP = "\n\n<<<CHUNK>>>\n\n"

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

    # Optional capability, same hasattr convention as decide_order's check in
    # /router/decide_order below. Only semantic_per_worker_tree defines this
    # today -- forces SentenceTransformer's lazy load (~5-6s one-time,
    # measured gun-raporu 2026-08-13) to happen now, blocking startup, rather
    # than silently landing on whichever live request arrives first.
    if hasattr(strategy, "warmup"):
        log.info("warming up %s ...", strategy.name)
        t0 = time.perf_counter()
        strategy.warmup()
        log.info("warmup done in %.1fs", time.perf_counter() - t0)

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

def _reorder_prompt(
    payload: dict, original_chunk_ids: list[str], new_order: list[str]
) -> dict | None:
    """Rewrite the system message's chunk block into `new_order`, using
    CHUNK_SEP to find the original chunk boundaries. Returns None (do
    nothing) on ANY mismatch -- this is the safety valve that keeps the
    other four strategies byte-identical to every run measured so far,
    since they never produce an ordered_chunk_ids and never reach here.

    Fragile-coupling note: this assumes replay.py's exact template
    (`f"{SYSTEM_PROMPT}\\n\\nBağlam:\\n\\n{blocks}"`). If that template
    changes, this function must change with it -- there is no schema
    linking the two files, only this comment.
    """
    if new_order == original_chunk_ids:
        return None  # nothing to rewrite, avoid a pointless re-serialise

    messages = payload.get("messages")
    if not messages:
        return None
    sys_msg = next((m for m in messages if m.get("role") == "system"), None)
    if sys_msg is None or not isinstance(sys_msg.get("content"), str):
        return None

    content = sys_msg["content"]
    marker = "Bağlam:\n\n"
    idx = content.find(marker)
    if idx == -1:
        return None
    prefix, blocks = content[: idx + len(marker)], content[idx + len(marker):]

    pieces = blocks.split(CHUNK_SEP)
    if len(pieces) != len(original_chunk_ids):
        return None  # sentinel absent or count mismatch -- not our request shape
    id_to_text = dict(zip(original_chunk_ids, pieces))
    if set(new_order) != set(original_chunk_ids):
        return None  # reorder must be a permutation, never change the set

    new_blocks = CHUNK_SEP.join(id_to_text[c] for c in new_order)
    sys_msg["content"] = prefix + new_blocks
    return payload


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


def _choose(payload: dict, chunk_ids: list[str] | None = None,
           forced_worker: str | None = None, session_id: str | None = None):
    tokenizer = state["tokenizer"]
    tracker: PrefixTracker = state["tracker"]
    strategy = state["strategy"]
    poller: MetricsPoller = state["poller"]

    text = _prompt_text(payload)
    hashes = block_hashes(tokenizer.encode(text))
    ctx = RequestContext(prompt_text=text, block_hashes=hashes, chunk_hashes=chunk_ids,
                         session_id=session_id)

    t0 = time.perf_counter()
    if forced_worker is not None:
        # /router/decide_order already made this call (worker + ordering);
        # this request is the follow-up completion, so we skip select() and
        # go straight to the named worker. Metrics/tracker recording below
        # still run, so Prometheus counters and prefix_tracker stay
        # consistent with every other strategy's bookkeeping.
        healthy = poller.snapshot().healthy()
        worker = next((s for s in healthy if s.name == forced_worker), None)
        if worker is None:
            raise NoHealthyWorker()
        decision = Decision(worker=worker, reason="forced_by_decide_order", scores={})
    else:
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
    # see the affinity. Only when TRACKER_TIMING="dispatch" (the default,
    # matches every prior measurement in this project) -- "completion" mode
    # records later, in _proxy_stream / the non-streaming path below, only
    # once the response has actually finished.
    if config.TRACKER_TIMING == "dispatch":
        tracker.record(decision.worker.name, hashes)
    CACHE_GAIN.observe(decision.cache_gain)
    ROUTED.labels(decision.worker.name, strategy.name, decision.reason).inc()
    return decision, len(hashes), believed_tokens, believed_frac, hashes


async def _proxy_stream(worker_url: str, path: str, body: bytes,
                        headers: dict[str, str], worker_name: str,
                        tracker: PrefixTracker | None = None,
                        hashes: list[str] | None = None,
                        n_tokens: int = 0) -> AsyncIterator[bytes]:
    client: httpx.AsyncClient = state["client"]
    poller: MetricsPoller = state["poller"]
    try:
        async with client.stream(
            "POST", f"{worker_url}{path}", content=body, headers=headers
        ) as upstream:
            # aiter_raw avoids any decoding/re-encoding of the SSE frames.
            async for chunk in upstream.aiter_raw():
                yield chunk
        # Reached only if the stream finished without raising -- i.e. the
        # response genuinely completed. completion-mode records here, not in
        # the except branch: a request that errored never actually got these
        # blocks served, so it should not be credited as a cache contributor.
        if config.TRACKER_TIMING == "completion" and tracker is not None and hashes is not None:
            tracker.record(worker_name, hashes)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE error
        UPSTREAM_ERRORS.labels(worker_name).inc()
        log.warning("stream failed on %s: %s", worker_name, exc)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
    finally:
        # n_tokens must match what mark_dispatch() was called with for this
        # same request (bkz. _handle_completion) -- it undoes exactly that
        # addition to inflight_tokens, not a freshly recomputed value.
        poller.mark_complete(worker_name, n_tokens)


async def _handle_completion(request: Request, path: str) -> Response:
    body = await request.body()
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    _cid_header = request.headers.get("x-chunk-ids")
    chunk_ids = _cid_header.split(",") if _cid_header else None
    forced_worker = request.headers.get("x-router-force-worker")
    session_id = request.headers.get("x-session-id")

    try:
        decision, n_blocks, believed_tokens, believed_frac, hashes = _choose(
            payload, chunk_ids=chunk_ids, forced_worker=forced_worker, session_id=session_id
        )
    except NoHealthyWorker:
        return JSONResponse({"error": "no healthy worker available"}, status_code=503)

    # If the strategy decided on a chunk order jointly with the worker (only
    # per_worker_tree does this), rewrite the prompt server-side before
    # forwarding. This only activates when: (a) the strategy actually
    # returned an ordering, (b) the client sent x-chunk-ids so we know the
    # ORIGINAL per-chunk boundaries, and (c) CHUNK_SEP is actually present in
    # the system message (replay.py only inserts it for --order
    # per_worker_tree). Any mismatch falls back to forwarding the original,
    # unmodified body -- this must never be allowed to crash or silently
    # corrupt a prompt for the other four strategies.
    if decision.ordered_chunk_ids is not None and chunk_ids:
        rewritten = _reorder_prompt(payload, chunk_ids, decision.ordered_chunk_ids)
        if rewritten is not None:
            payload = rewritten
            body = json.dumps(payload).encode()

    worker = decision.worker
    headers = _forward_headers(request)
    poller: MetricsPoller = state["poller"]
    # n_blocks * BLOCK_SIZE is a real (block-floored) prompt token count --
    # already computed inside _choose() via the router's own tokenizer, not
    # re-tokenized here. Feeds WorkerState.inflight_tokens, which
    # CacheWeaverDualMapStrategy reads in place of the old
    # num_requests_waiting * AVG_PROMPT_TOKENS_ESTIMATE guess.
    prompt_tokens_est = n_blocks * config.BLOCK_SIZE
    poller.mark_dispatch(worker.name, prompt_tokens_est)

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
            _proxy_stream(worker.url, path, body, headers, worker.name,
                         tracker=state["tracker"], hashes=hashes,
                         n_tokens=prompt_tokens_est),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    client: httpx.AsyncClient = state["client"]
    try:
        upstream = await client.post(f"{worker.url}{path}", content=body, headers=headers)
    except Exception as exc:  # noqa: BLE001
        UPSTREAM_ERRORS.labels(worker.name).inc()
        poller.mark_complete(worker.name, prompt_tokens_est)
        return JSONResponse({"error": f"upstream failure: {exc}"}, status_code=502)

    poller.mark_complete(worker.name, prompt_tokens_est)
    if config.TRACKER_TIMING == "completion":
        state["tracker"].record(worker.name, hashes)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers=resp_headers,
    )


@app.post("/router/decide_order")
async def decide_order(request: Request) -> Response:
    """Two-phase entry point for strategies where the best chunk order
    depends on which worker is chosen (currently only per_worker_tree).
    replay.py calls this FIRST with the raw, unordered retrieval-order chunk
    ids, gets back {worker, ordered_chunk_ids}, builds the actual prompt in
    that order, then sends the real completion request with
    x-router-force-worker set so /v1/chat/completions skips routing and goes
    straight to the named worker (bookkeeping still runs normally there).
    """
    strategy = state["strategy"]
    if not hasattr(strategy, "decide_order"):
        return JSONResponse(
            {"error": f"strategy '{strategy.name}' does not support decide_order "
                      f"(only per_worker_tree does)"},
            status_code=400,
        )
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    chunk_ids = body.get("chunk_ids")
    if not chunk_ids:
        return JSONResponse({"error": "chunk_ids (non-empty list) required"}, status_code=400)

    poller: MetricsPoller = state["poller"]
    try:
        decision = strategy.decide_order(chunk_ids, poller.snapshot())
    except NoHealthyWorker:
        return JSONResponse({"error": "no healthy worker available"}, status_code=503)

    worker_url = next((w.url for w in config.WORKERS if w.name == decision.worker_name), None)
    return JSONResponse({
        "worker": decision.worker_name,
        "worker_url": worker_url,
        "ordered_chunk_ids": decision.ordered_chunk_ids,
        "cache_gain": decision.cache_gain,
        "scores": decision.scores,
    })


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
