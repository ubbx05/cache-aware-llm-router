"""Routing strategies.

Every strategy implements the same `select` signature, so baselines and the
proposed policy are interchangeable via one config value. That is what makes
the ablation table in the paper a matter of re-running with a different env
var rather than editing code.

All three are special cases of the same scoring function:

    score(w) = ALPHA * cache_gain(w) - BETA * load_norm(w)

    round_robin  : ALPHA = 0, BETA = 0  (plus a rotating tie-break)
    least_loaded : ALPHA = 0, BETA > 0
    cache_aware  : ALPHA > 0, BETA > 0, plus the adaptive guard band

Framing the baselines as degenerate parameterisations of the proposal is
stronger than presenting the proposal as an unrelated new heuristic.
"""
from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import config
from prefix_tracker import PrefixTracker
from worker_metrics import Snapshot, WorkerState


def _argmax(candidates: list[WorkerState], scores: dict[str, float]) -> WorkerState:
    """Highest score, ties broken uniformly at random.

    Plain `max()` returns the first maximum, and the worker list is always in
    the same order, so every tie would go to w1 -- a systematic bias that is
    easy to mistake for a policy decision when reading the traffic split.
    """
    best = max(scores[c.name] for c in candidates)
    return random.choice([c for c in candidates if scores[c.name] >= best - 1e-12])


@dataclass
class Decision:
    worker: WorkerState
    reason: str
    scores: dict[str, float]
    cache_gain: float = 0.0


@dataclass
class RequestContext:
    """Everything a strategy is allowed to know about an incoming request."""
    prompt_text: str
    block_hashes: list[str]
    chunk_hashes: list[str] | None = None  # populated in the RAG phase


class Strategy(ABC):
    name: str = "base"

    def __init__(self, tracker: PrefixTracker):
        self.tracker = tracker

    @abstractmethod
    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision: ...

    @staticmethod
    def _normalised_loads(snap: Snapshot) -> dict[str, float]:
        """Map raw loads onto [0, 1) with load / (load + LOAD_REF).

        Saturating rather than clipping. `min(load / ref, 1.0)` looks
        equivalent but destroys the policy once both workers exceed the
        reference: they both read 1.0, least_loaded sees a tie, and the guard
        band's `load > min_load + delta` can never be true. This form keeps a
        usable gradient at any absolute load, which matters because the right
        value of LOAD_REF is not knowable in advance.
        """
        ref = max(config.LOAD_REF, 1e-6)
        return {s.name: s.load() / (s.load() + ref) for s in snap.healthy()}


class RoundRobin(Strategy):
    """Baseline: ignores both load and cache. The floor to beat."""

    name = "round_robin"

    def __init__(self, tracker: PrefixTracker):
        super().__init__(tracker)
        self._cycle = itertools.count()

    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision:
        healthy = snap.healthy()
        if not healthy:
            raise NoHealthyWorker()
        worker = healthy[next(self._cycle) % len(healthy)]
        return Decision(worker=worker, reason="round_robin", scores={})


class LeastLoaded(Strategy):
    """Baseline: pure load balancing, cache-blind."""

    name = "least_loaded"

    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision:
        healthy = snap.healthy()
        if not healthy:
            raise NoHealthyWorker()
        loads = self._normalised_loads(snap)
        scores = {name: -load for name, load in loads.items()}
        best = _argmax(healthy, scores)
        return Decision(worker=best, reason="least_loaded", scores=scores)


class CacheAware(Strategy):
    """The proposed policy: adaptive cache/load trade-off.

    Two mechanisms, deliberately kept separable so the ablation can attribute
    the gain to one or the other:

    1. The weighted score itself, ALPHA * cache_gain - BETA * load.
    2. An adaptive guard band. Pure cache-affinity routing is unstable: the
       worker holding a popular prefix keeps attracting traffic, which grows
       its queue, which makes it the worst choice by latency even though it
       stays the best choice by cache. The guard band caps how far above the
       least-loaded worker the winner is allowed to sit:

           delta = DELTA0 * (1 - mean(kv_usage))

       When caches are cold there is little affinity worth protecting, so the
       band is wide and cache preference is cheap. As caches fill, the band
       narrows and the policy degrades gracefully towards least-loaded.
    """

    name = "cache_aware"

    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision:
        healthy = snap.healthy()
        if not healthy:
            raise NoHealthyWorker()

        loads = self._normalised_loads(snap)
        gains = {
            s.name: self.tracker.prefix_gain(s.name, ctx.block_hashes)
            for s in healthy
        }
        scores = {
            s.name: config.ALPHA * gains[s.name] - config.BETA * loads[s.name]
            for s in healthy
        }

        best = _argmax(healthy, scores)

        delta = config.DELTA0 * (1.0 - snap.mean_kv_usage())
        min_load = min(loads.values())
        if loads[best.name] > min_load + delta:
            # Winner is too far into the overloaded region: fall back.
            fallback = _argmax(healthy, {n: -v for n, v in loads.items()})
            return Decision(
                worker=fallback,
                reason=f"guard_band(delta={delta:.3f})",
                scores=scores,
                cache_gain=gains[fallback.name],
            )

        return Decision(
            worker=best,
            reason="score",
            scores=scores,
            cache_gain=gains[best.name],
        )


class NoHealthyWorker(RuntimeError):
    pass


_REGISTRY: dict[str, type[Strategy]] = {
    RoundRobin.name: RoundRobin,
    LeastLoaded.name: LeastLoaded,
    CacheAware.name: CacheAware,
}


def build_strategy(name: str, tracker: PrefixTracker) -> Strategy:
    try:
        return _REGISTRY[name](tracker)
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; expected one of {sorted(_REGISTRY)}"
        ) from None
