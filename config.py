"""Router configuration.

Everything an experiment might sweep lives here (or in an env var), so that
switching baselines or tuning alpha/beta never requires touching router code.
That property is what makes the ablation section of the paper cheap to produce.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


@dataclass(frozen=True)
class WorkerConfig:
    name: str           # short label used in metrics: w1, w2, ...
    url: str            # vLLM OpenAI-compatible base URL, no trailing slash
    enabled: bool = True


# --- Worker pool -----------------------------------------------------------
# ADDING THE SECOND WORKER = two edits:
#   1. put the peer's Tailscale IP in W2_URL (or edit the default below)
#   2. W2_ENABLED=true
# Nothing else in the codebase needs to change.
WORKERS: list[WorkerConfig] = [
    WorkerConfig(
        name="w1",
        url=os.getenv("W1_URL", "http://100.89.101.52:8000"),
        enabled=_env_bool("W1_ENABLED", True),
    ),
    WorkerConfig(
        name="w2",
        url=os.getenv("W2_URL", "http://100.64.0.2:8000"),
        enabled=_env_bool("W2_ENABLED", False),
    ),
]

# --- Routing strategy ------------------------------------------------------
# One of: round_robin | least_loaded | cache_aware | adaptive_cache_aware |
#         cacheweaver_dualmap | per_worker_tree | semantic_per_worker_tree
STRATEGY: str = os.getenv("ROUTER_STRATEGY", "round_robin")

# score(w) = ALPHA * cache_gain(w) - BETA * load(w)
ALPHA: float = _env_float("ROUTER_ALPHA", 1.0)
BETA: float = _env_float("ROUTER_BETA", 1.0)

# Adaptive guard band: delta = DELTA0 * (1 - mean(kv_usage))
# When caches are empty there is little to protect, so the router is allowed to
# deviate far from the least-loaded choice. As caches fill, delta shrinks and
# the router behaves more like a load balancer.
DELTA0: float = _env_float("ROUTER_DELTA0", 0.5)

# Relative weights of the load signals.
#
# Measured on this setup: `num_requests_waiting` stays at 0 even when the engine
# is fully busy, because vLLM admits requests into the running batch rather than
# queueing them. Load shows up as *batch size*, not queue depth, so `running` has
# to be the primary term here. `waiting` still gets the higher weight for when it
# does move -- a genuinely queued request is worse than a batched one -- but it
# cannot be the signal the policy relies on.
LOAD_W_WAITING: float = _env_float("ROUTER_LOAD_W_WAITING", 2.0)
LOAD_W_RUNNING: float = _env_float("ROUTER_LOAD_W_RUNNING", 1.0)

# Requests dispatched since the last scrape are not yet visible in the worker's
# own counters, so they are added separately. They must NOT be added on top of
# `waiting`/`running`: once a scrape lands, those same requests are already
# counted there, and adding both double-counts every in-flight request.
LOAD_W_PENDING: float = _env_float("ROUTER_LOAD_W_PENDING", 1.0)

# Reference load for normalisation: load_norm = load / (load + LOAD_REF).
#
# This saturating form is deliberate. The obvious `min(load / ref, 1.0)` clips,
# and once both workers clip at 1.0 every difference between them vanishes:
# least_loaded sees a tie and falls through to list order, and the guard band
# can never fire because `1.0 > 1.0 + delta` is false. That is exactly what
# happened at 64-way concurrency with ref=8 -- the policy silently degenerated
# into pure cache affinity. The saturating form approaches 1 but never reaches
# it, so relative differences survive at any absolute load.
#
# MEASURED, not guessed: bench/sweep_batch.py sweeps one worker at fixed
# concurrency and locates the peak of Kleinrock power (throughput / latency).
# On this setup (Qwen2.5-7B, ~1700-token prompts, 128 output tokens) the peak
# sits at 16 concurrent requests, and it did NOT move between --enforce-eager
# and CUDA-graph mode -- so this number is a property of the model/prompt shape,
# not of the engine's execution mode.
#
# Do not reuse 16 on different hardware. Re-run the sweep and set this to the
# reported power knee; that is the recipe, the number is just this setup's answer.
#
# Note the criterion is power, not "where TTFT starts to degrade" -- TTFT on vLLM
# rises from the very first added request, so that phrasing has no well-defined
# answer and naive readings of it collapse to 1.
#
# Do NOT normalise against the busiest worker either: with two replicas that maps
# any imbalance to (1.0, 0.0).
LOAD_REF: float = _env_float("ROUTER_LOAD_REF", 16.0)

# --- Semantic pre-filter (semantic_per_worker_tree only) -------------------
# How many workers the semantic candidate predictor hands to
# per_worker_tree_router.choose(), instead of every healthy worker. Lower is
# cheaper (n*O(k^2) reorder cost) but risks excluding the actual best-cache
# worker if the centroid signal is noisy/cold.
SEMANTIC_TOP_K: int = _env_int("ROUTER_SEMANTIC_TOP_K", 2)

# EWMA learning rate for each worker's query centroid -- same role as
# adaptive_drift_model.py's OnlineDriftEstimator.lam. Small = slow but
# noise-resistant tracking of "what topics has this worker been serving";
# large = reacts fast but jumpy.
SEMANTIC_CENTROID_LR: float = _env_float("ROUTER_SEMANTIC_CENTROID_LR", 0.15)

# --- Adaptive beta/delta via drift detection (adaptive_cache_aware only) ---
# See adaptive_drift_model.py. The router observes session-adjacent retrieved-
# chunk-set Jaccard overlap live (needs x-session-id from replay.py) and feeds
# it through an EWMA + CUSUM pair to decide when locality has genuinely
# shifted, then scales BETA/DELTA0 accordingly instead of leaving them fixed.
#
# D_TARGET is the "normal" (pre-drift) session-adjacent overlap the detector
# compares live readings against. MEASURED, not guessed -- same discipline as
# LOAD_REF: run bench/overlap_measurement.py on your trace and use its
# session-adjacent mean. On trace_hot.jsonl (zipf-s=1.5, session-len=4) that
# was 0.529 (800 queries, 2026-07-30) -- re-measure for any other trace shape,
# this number is a property of the workload, not the router.
D_TARGET: float = _env_float("ROUTER_D_TARGET", 0.529)

# EWMA learning rate for the live overlap estimate (OnlineDriftEstimator.lam).
DRIFT_LAM: float = _env_float("ROUTER_DRIFT_LAM", 0.1)

# CUSUM sensitivity margin and alarm threshold (CusumDriftDetector.k / .h).
# Defaults are adaptive_drift_model.py's own self-test values, not yet
# calibrated against a real multi-thousand-request live run -- see the open
# item in today's report.
CUSUM_K: float = _env_float("ROUTER_CUSUM_K", 0.03)
CUSUM_H: float = _env_float("ROUTER_CUSUM_H", 0.20)

# --- cacheweaver_dualmap baseline thresholds --------------------------------
# CacheWeaverDualMapRouter's own defaults (20_000 / 30_000) are borrowed from
# DualMap's paper hardware, not measured on this setup -- same class of gap
# as an uncalibrated LOAD_REF. MEASURED, not guessed: mean prompt_tokens on
# this workload (top_k=10, RAG) was 2392 (n=800, runs/ca_r1.jsonl,
# 2026-08-13); threshold = LOAD_REF * mean_prompt_tokens, same recipe as
# LOAD_REF itself. rebalance keeps DualMap's own 30_000/20_000 = 1.5x ratio.
# Re-measure if top_k, corpus, or prompt template changes.
CACHEWEAVER_TTFT_SLO_THRESHOLD_TOKENS: int = _env_int(
    "ROUTER_CACHEWEAVER_TTFT_SLO_THRESHOLD_TOKENS", 38_272
)
CACHEWEAVER_REBALANCE_THRESHOLD_TOKENS: int = _env_int(
    "ROUTER_CACHEWEAVER_REBALANCE_THRESHOLD_TOKENS", 57_408
)

# --- Tracker bookkeeping timing ---------------------------------------------
# "dispatch": tracker.record() runs the moment a worker is chosen, before the
# response comes back -- optimistic (concurrent arrivals see the affinity
# immediately) but can overestimate under heavy concurrency (Faz 1, 2026-07-26:
# correlation 0.415 at speedup=20 vs 0.532 at speedup=1, same trace/capacity).
# "completion": tracker.record() runs only after the response has actually
# finished streaming back successfully -- never overestimates, but a burst of
# concurrent identical requests won't see each other's affinity in time.
# This was previously only observed as a side effect of a concurrency sweep;
# TRACKER_TIMING makes it a first-class, directly comparable toggle.
TRACKER_TIMING: str = os.getenv("ROUTER_TRACKER_TIMING", "dispatch")
if TRACKER_TIMING not in ("dispatch", "completion"):
    raise ValueError(f"ROUTER_TRACKER_TIMING must be 'dispatch' or 'completion', got {TRACKER_TIMING!r}")

# --- Prefix / cache model --------------------------------------------------
# vLLM hashes the KV cache in fixed-size token blocks (16 by default). The
# router mirrors that granularity so its notion of a "cache hit" lines up with
# what the engine actually reuses.
BLOCK_SIZE: int = _env_int("ROUTER_BLOCK_SIZE", 16)

# How many block hashes to remember per worker before evicting (LRU).
# This is the router's approximation of the engine's own cache eviction.
TRACKER_CAPACITY: int = _env_int("ROUTER_TRACKER_CAPACITY", 50_000)

# "hf" uses the real tokenizer (accurate, ~1ms/request). "approx" splits on
# characters (zero dependency, biased). Use hf for anything that goes in the paper.
TOKENIZER_MODE: str = os.getenv("ROUTER_TOKENIZER", "approx")
TOKENIZER_MODEL: str = os.getenv("ROUTER_TOKENIZER_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# --- Metric polling --------------------------------------------------------
POLL_INTERVAL_S: float = _env_float("ROUTER_POLL_INTERVAL", 1.0)
POLL_TIMEOUT_S: float = _env_float("ROUTER_POLL_TIMEOUT", 2.0)
UNHEALTHY_AFTER: int = _env_int("ROUTER_UNHEALTHY_AFTER", 3)

# The poller gets its own connection pool. Sharing one with the proxy means that
# under heavy streaming the /metrics scrape queues behind proxied requests, times
# out, and the router declares a perfectly healthy worker dead -- which surfaced
# as a wave of 503s at 512-way concurrency.
POLL_MAX_CONNECTIONS: int = _env_int("ROUTER_POLL_MAX_CONNECTIONS", 8)

# Upstream pool for proxied traffic. httpx defaults to 100, which silently caps
# concurrency well below what the engine can absorb.
PROXY_MAX_CONNECTIONS: int = _env_int("ROUTER_PROXY_MAX_CONNECTIONS", 1024)

# --- Server ----------------------------------------------------------------
HOST: str = os.getenv("ROUTER_HOST", "0.0.0.0")
PORT: int = _env_int("ROUTER_PORT", 8080)

# Upstream API key, if the vLLM workers were started with --api-key.
UPSTREAM_API_KEY: str | None = os.getenv("UPSTREAM_API_KEY") or None
