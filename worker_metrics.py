"""Background poller for per-worker vLLM metrics.

The router cannot ask a worker "how busy are you?" synchronously on every
request without adding that round-trip to TTFT. So a background task scrapes
each worker's /metrics endpoint on a fixed interval and keeps the latest values
in memory. Routing decisions read that snapshot, which costs nothing.

The staleness this introduces (up to POLL_INTERVAL_S) is a real limitation and
belongs in the paper's limitations section: at high arrival rates the router
can send a burst to a worker whose queue has already grown since the last poll.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

import httpx

import config

# Matches: vllm:num_requests_waiting{engine="0",model_name="..."} 3.0
_SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)$")

# Gauges are summed across engines; usage percentages are averaged.
_SUM_METRICS = {
    "vllm:num_requests_waiting",
    "vllm:num_requests_running",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
}
_AVG_METRICS = {
    "vllm:kv_cache_usage_perc",
}


@dataclass
class WorkerState:
    name: str
    url: str
    healthy: bool = False
    consecutive_failures: int = 0
    last_scrape_ts: float = 0.0

    num_requests_waiting: float = 0.0
    num_requests_running: float = 0.0
    kv_cache_usage_perc: float = 0.0
    prefix_cache_queries_total: float = 0.0
    prefix_cache_hits_total: float = 0.0

    # Requests dispatched since the last successful scrape. Reset on every
    # scrape, because from that point they are visible in waiting/running.
    # Counting them in addition to those gauges -- as an unbounded `inflight`
    # counter would -- double-counts every open request and pushes the load
    # signal into saturation.
    dispatched_since_scrape: int = 0

    # Sum of prompt tokens for every request currently dispatched to this
    # worker and not yet completed. Unlike dispatched_since_scrape, this is
    # NOT reset on scrape -- vLLM's /metrics exposes request COUNTS
    # (num_requests_waiting/running), never per-request token sizes, so there
    # is no scraped signal to hand off to. This counter is the router's only
    # source of truth for "how many tokens are actually outstanding on this
    # worker", and it stays accurate for as long as every mark_dispatch() is
    # paired with exactly one mark_complete() carrying the same token count
    # (see main.py's _handle_completion/_proxy_stream). Used by
    # CacheWeaverDualMapStrategy in place of the old
    # num_requests_waiting * AVG_PROMPT_TOKENS_ESTIMATE guess.
    inflight_tokens: int = 0

    def load(self) -> float:
        """Load(w) from the signals this engine actually moves.

        `waiting` carries the higher weight because a queued request delays the
        next arrival in full, while a running one is already being amortised by
        continuous batching. But on vLLM V1 `waiting` frequently stays at zero
        under real load -- requests go straight into the running batch -- so
        `running` is what the policy ends up steering on.
        """
        return (
            config.LOAD_W_WAITING * self.num_requests_waiting
            + config.LOAD_W_RUNNING * self.num_requests_running
            + config.LOAD_W_PENDING * self.dispatched_since_scrape
        )


@dataclass
class Snapshot:
    """Immutable view of all workers at decision time."""
    states: dict[str, WorkerState] = field(default_factory=dict)

    def healthy(self) -> list[WorkerState]:
        return [s for s in self.states.values() if s.healthy]

    def mean_kv_usage(self) -> float:
        healthy = self.healthy()
        if not healthy:
            return 0.0
        return sum(s.kv_cache_usage_perc for s in healthy) / len(healthy)

    def max_load(self) -> float:
        healthy = self.healthy()
        return max((s.load() for s in healthy), default=0.0)


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Reduce a Prometheus exposition payload to the handful of series we use.

    Counters ending in `_created` are timestamps emitted automatically by
    prometheus_client and are ignored.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in _SUM_METRICS and name not in _AVG_METRICS:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1

    out: dict[str, float] = {}
    for name, total in sums.items():
        out[name] = total / counts[name] if name in _AVG_METRICS else total
    return out


class MetricsPoller:
    def __init__(self, workers: list[config.WorkerConfig], client: httpx.AsyncClient):
        self._client = client
        self._states = {
            w.name: WorkerState(name=w.name, url=w.url)
            for w in workers
            if w.enabled
        }
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def states(self) -> dict[str, WorkerState]:
        return self._states

    def snapshot(self) -> Snapshot:
        return Snapshot(states=self._states)

    def mark_dispatch(self, name: str, n_tokens: int = 0) -> None:
        if name in self._states:
            self._states[name].dispatched_since_scrape += 1
            self._states[name].inflight_tokens += n_tokens

    def mark_complete(self, name: str, n_tokens: int = 0) -> None:
        # dispatched_since_scrape still needs no bookkeeping here (comment
        # above it explains why -- the next scrape absorbs it). inflight_tokens
        # is different: nothing scraped ever reflects it, so it is the only
        # place this request's tokens get removed. max(0, ...) guards against
        # drift (e.g. a mark_complete arriving for a request whose worker was
        # reset/restarted mid-flight) turning into a permanently negative,
        # nonsensical load reading.
        if name in self._states:
            s = self._states[name]
            s.inflight_tokens = max(0, s.inflight_tokens - n_tokens)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.gather(*(self._scrape(s) for s in self._states.values()))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=config.POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                pass

    async def _scrape(self, state: WorkerState) -> None:
        try:
            r = await self._client.get(
                f"{state.url}/metrics", timeout=config.POLL_TIMEOUT_S
            )
            r.raise_for_status()
        except Exception:
            state.consecutive_failures += 1
            if state.consecutive_failures >= config.UNHEALTHY_AFTER:
                state.healthy = False
            return

        parsed = parse_prometheus_text(r.text)
        state.num_requests_waiting = parsed.get("vllm:num_requests_waiting", 0.0)
        state.num_requests_running = parsed.get("vllm:num_requests_running", 0.0)
        state.kv_cache_usage_perc = parsed.get("vllm:kv_cache_usage_perc", 0.0)
        state.prefix_cache_queries_total = parsed.get("vllm:prefix_cache_queries_total", 0.0)
        state.prefix_cache_hits_total = parsed.get("vllm:prefix_cache_hits_total", 0.0)
        state.last_scrape_ts = time.time()
        state.consecutive_failures = 0
        state.healthy = True
        state.dispatched_since_scrape = 0
