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
from adaptive_drift_model import CusumDriftDetector, OnlineDriftEstimator, adaptive_beta, adaptive_delta
from cacheweaver_dualmap_router import CacheWeaverDualMapRouter
from per_worker_tree_router import PerWorkerDecision, PerWorkerTreeRouter
from prefix_tracker import PrefixTracker
from semantic_worker_router import SemanticWorkerRouter, real_embed
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
    # Set only by strategies that decide ordering jointly with worker choice
    # (per_worker_tree). None for every other strategy -- main.py must treat
    # None as "nothing to rewrite, forward the prompt unchanged".
    ordered_chunk_ids: list[str] | None = None


@dataclass
class RequestContext:
    """Everything a strategy is allowed to know about an incoming request."""
    prompt_text: str
    block_hashes: list[str]
    chunk_hashes: list[str] | None = None  # populated in the RAG phase
    # From the x-session-id header (replay.py). None for any client that
    # doesn't send it -- every existing strategy ignores this field, so its
    # absence changes nothing for them.
    session_id: str | None = None


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


def _jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class AdaptiveCacheAware(CacheAware):
    """CacheAware + live drift-adaptive beta/delta (adaptive_drift_model.py).

    Deliberately a SIBLING of CacheAware, not a modification -- CacheAware's
    own select() is never touched, so every measurement already taken
    against it (calibration, ablations, 2-worker comparison) stays valid and
    comparable. This class overrides select() completely rather than trying
    to parameterise the parent.

    Needs ctx.session_id (x-session-id header, added to replay.py) to compute
    a SESSION-ADJACENT retrieved-chunk-set Jaccard signal. This is not a
    stylistic choice: bench/overlap_measurement.py measured session-adjacent
    overlap at mean=0.529 vs global-adjacent (no session grouping) at
    mean=0.178, median=0.000, nonzero=19.8% (trace_hot.jsonl, 800 queries,
    2026-07-30) -- global-adjacent is mostly reading noise, not signal, and
    would make the CUSUM detector nearly useless. If a client never sends
    x-session-id (or every session is still on its first turn), this
    degrades gracefully to CacheAware's exact fixed BETA/DELTA0 formula
    rather than guessing or crashing.

    Dispatch-time convention: the drift estimator/detector are updated here,
    inside select(), at the same point every other adaptive strategy in this
    file does its bookkeeping -- keeps the ablation timing-comparable.
    """

    name = "adaptive_cache_aware"

    def __init__(self, tracker: PrefixTracker):
        super().__init__(tracker)
        self._estimator = OnlineDriftEstimator(lam=config.DRIFT_LAM)
        self._detector = CusumDriftDetector(
            d_ref=config.D_TARGET, k=config.CUSUM_K, h=config.CUSUM_H
        )
        self._last_chunk_set_by_session: dict[str, set] = {}

    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision:
        healthy = snap.healthy()
        if not healthy:
            raise NoHealthyWorker()

        loads = self._normalised_loads(snap)
        gains = {
            s.name: self.tracker.prefix_gain(s.name, ctx.block_hashes)
            for s in healthy
        }

        if ctx.session_id is not None:
            chunk_ids = ctx.chunk_hashes or ctx.block_hashes
            current_set = set(chunk_ids) if chunk_ids else set()
            prev_set = self._last_chunk_set_by_session.get(ctx.session_id)
            if prev_set is not None:
                jaccard = _jaccard_sets(current_set, prev_set)
                d_t = self._estimator.update(jaccard)
                self._detector.update(jaccard, current_ewma_estimate=d_t)
            self._last_chunk_set_by_session[ctx.session_id] = current_set

        # _d_t (not the .current property) is the only way to distinguish
        # "never observed anything yet" from "observed exactly 0.0 overlap" --
        # adaptive_drift_model.py doesn't expose this as a public flag; reaching
        # into the private field is the same accepted-debt pattern as
        # cacheweaver_dualmap_router.py's `tree._root` (see its own comment).
        warmed_up = self._estimator._d_t is not None
        if warmed_up:
            d_t = self._estimator.current
            effective_beta = adaptive_beta(config.BETA, d_t, config.D_TARGET)
            delta = adaptive_delta(config.DELTA0, snap.mean_kv_usage(), d_t, config.D_TARGET)
        else:
            effective_beta = config.BETA
            delta = config.DELTA0 * (1.0 - snap.mean_kv_usage())

        scores = {
            s.name: config.ALPHA * gains[s.name] - effective_beta * loads[s.name]
            for s in healthy
        }
        best = _argmax(healthy, scores)

        min_load = min(loads.values())
        if loads[best.name] > min_load + delta:
            fallback = _argmax(healthy, {n: -v for n, v in loads.items()})
            return Decision(
                worker=fallback,
                reason=f"guard_band(delta={delta:.3f},beta={effective_beta:.3f})",
                scores=scores,
                cache_gain=gains[fallback.name],
            )

        return Decision(
            worker=best,
            reason=f"score(beta={effective_beta:.3f})",
            scores=scores,
            cache_gain=gains[best.name],
        )


class CacheWeaverDualMapStrategy(Strategy):
    """Baseline: CacheWeaver greedy-reorder + DualMap dual-hash/SLO/hotspot
    hybrid (cacheweaver_dualmap_router.py), wrapped to fit this router's
    Strategy interface.

    Timing convention: on_request_finished() is called here, inside select(),
    right after the decision is made -- the same dispatch-time point where
    CacheAware's tracker.record() happens in main.py's _choose(). This keeps
    the ablation table comparable: any TTFT/hit-rate gap against cache_aware
    is attributable to the algorithm, not to a bookkeeping-timing mismatch.
    If you want to characterise dispatch-vs-completion as its own axis later,
    apply the same change to BOTH strategies, not just this one.

    Units note: DualMap's ReplicaState.num_pending_prefill_tokens wants a
    TOKEN count, but this router only exposes request-level queue depth
    (num_requests_waiting). AVG_PROMPT_TOKENS_ESTIMATE converts one to the
    other; MEASURED (mean prompt_tokens, n=800, runs/ca_r1.jsonl, top_k=10,
    2026-08-13) rather than guessed -- same number used to calibrate
    CACHEWEAVER_TTFT_SLO_THRESHOLD_TOKENS in config.py. Re-measure if top_k
    or the prompt template changes; this is a workload property, not a
    router property.

    (An alternative to this estimate -- WorkerState.inflight_tokens, a real
    per-worker running total of dispatched-but-not-completed prompt tokens,
    fed by main.py's mark_dispatch/mark_complete -- was tried in a parallel
    branch and is still wired through worker_metrics.py/main.py. This class
    uses the calibrated-estimate version instead by team decision; the real
    counter is dormant but harmless, and is available if this estimate ever
    needs replacing again.)
    """

    name = "cacheweaver_dualmap"

    AVG_PROMPT_TOKENS_ESTIMATE = 2392

    def __init__(self, tracker: PrefixTracker):
        super().__init__(tracker)
        # Fixed at construction time so replica indices stay stable across
        # calls even if a worker briefly drops out of snap.healthy().
        self._names = [w.name for w in config.WORKERS if w.enabled]
        self._name_to_idx = {name: i for i, name in enumerate(self._names)}
        self._router = CacheWeaverDualMapRouter(
            num_replicas=len(self._names),
            ttft_slo_threshold_tokens=config.CACHEWEAVER_TTFT_SLO_THRESHOLD_TOKENS,
            rebalance_threshold_tokens=config.CACHEWEAVER_REBALANCE_THRESHOLD_TOKENS,
        )
        self._req_counter = itertools.count()

    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision:
        healthy = snap.healthy()
        if not healthy:
            raise NoHealthyWorker()

        idx_to_worker: dict[int, WorkerState] = {}
        for s in healthy:
            idx = self._name_to_idx.get(s.name)
            if idx is None:
                continue  # worker not part of the fixed index mapping (unexpected)
            idx_to_worker[idx] = s
            pending_tokens = s.num_requests_waiting * self.AVG_PROMPT_TOKENS_ESTIMATE
            self._router.update_replica_load(idx, int(pending_tokens))

        chunk_ids = ctx.chunk_hashes or ctx.block_hashes
        decision = self._router.route_request(
            request_id=str(next(self._req_counter)),
            retrieved_chunk_ids=chunk_ids,
        )

        # Dispatch-time bookkeeping -- see class docstring.
        self._router.on_request_finished(decision)

        worker = idx_to_worker.get(decision.primary_replica)
        if worker is None:
            # Chosen replica isn't currently healthy (e.g. dropped out between
            # calls) -- fail safe rather than crash the request.
            worker = healthy[0]

        reason = "cache_affinity" if decision.used_cache_affinity else "slo_min_ttft"
        if decision.migrated:
            reason += "+migrated"

        return Decision(worker=worker, reason=reason, scores={}, cache_gain=0.0)


def adaptive_weights(
    overlap: float,
    threshold: float = 0.3,
    low_alpha: float = 0.2,
    high_alpha: float = 0.8,
) -> tuple[float, float]:
    """Deliberately simple threshold switch -- NOT an EWMA/CUSUM estimator
    (that mechanism, adaptive_drift_model.py, was scoped OUT of this task).

    overlap: Jaccard(previous request's retrieved chunks, this request's) --
    same definition as bench/overlap_measurement.py's jaccard(), a single
    consecutive-pair comparison, not a rolling window. Below `threshold`,
    cache affinity is a weak signal (the last request barely predicts this
    one), so alpha drops and the load term dominates; at/above threshold,
    cache affinity is trusted more. beta is each alpha's complement -- one
    free parameter per regime, not two.
    """
    alpha = low_alpha if overlap < threshold else high_alpha
    return alpha, 1.0 - alpha


class PerWorkerTreeStrategy(Strategy):
    """Worker-basina agac + iki-aday reorder tasarimi (bkz.
    per_worker_tree_router.py). Her worker kendi CacheWeaverKnowledgeTree'sini
    tutar; en iyi chunk sirasi WORKER SECIMINE bagli olarak degisir -- tek,
    worker'dan bagimsiz bir "dogru sira" yok. cacheweaver_dualmap'ten farki:
    orada n=2'de dual-hash aday secimi kanitlanmis sekilde no-op'tu (her
    zaman {0,1}); burada hash/aday kismi hic yok, direkt karsilastirma var.

    Iki giris noktasi var:
    - decide_order(): ham chunk_ids alir, (worker, sira) doner. main.py'nin
      /router/decide_order endpoint'i BUNU cagirir -- gercek prompt'u
      yeniden sirasiyla kurmak icin gereken tam bilgi burada.
    - select(): diger stratejilerle ayni Strategy arayuzunu saglamak icin
      var (registry, genel smoke-testler). ctx.chunk_hashes main.py'nin
      normal /v1/chat/completions akisinda artik x-chunk-ids header'indan
      dolduruluyor (bkz. main.py _choose), yani chunk kimlikleri select()'e
      de ulasiyor -- ama bu yoldan donen ordered_chunk_ids'in gercekten
      uygulanmasi main.py'nin decision.ordered_chunk_ids'i okuyup prompt'u
      yeniden yazmasina bagli (bkz. main.py _handle_completion, CHUNK_SEP).

    Overlap-adaptive alpha/beta (config.OVERLAP_ADAPTIVE_ENABLED, varsayilan
    KAPALI): acikken, her decide_order() cagrisinda bu isteğin chunk_ids'i
    ile BIR ONCEKI cagrinin chunk_ids'i arasindaki Jaccard hesaplanir (bkz.
    adaptive_weights() yukarida, jaccard() bench/overlap_measurement.py'den
    import edildi -- yeniden yazilmadi) ve config.ALPHA/BETA yerine bu
    ölçüme gore secilen alpha/beta kullanilir. KAPALIYKEN davranis eskisiyle
    BIREBIR ayni -- config.ALPHA/BETA sabitleri degismeden kullanilir, ne
    self._prev_chunk_ids okunur ne de jaccard() cagrilir.
    """

    name = "per_worker_tree"

    def __init__(self, tracker: PrefixTracker):
        super().__init__(tracker)
        self._names = [w.name for w in config.WORKERS if w.enabled]
        self._router = PerWorkerTreeRouter(worker_names=self._names)
        self._prev_chunk_ids: set[str] | None = None  # sadece OVERLAP_ADAPTIVE_ENABLED iken kullanilir

    def decide_order(self, chunk_ids: list[str], snap: Snapshot) -> PerWorkerDecision:
        """Ham (siralanmamis, retrieval sirasindaki) chunk_ids alir, worker +
        o worker'a gore en iyi sira karari doner. Dispatch-time bookkeeping
        burada yapilir (cacheweaver_dualmap ile ayni zamanlama konvansiyonu,
        ablation tablosunun timing farkindan degil algoritma farkindan
        etkilenmesi icin)."""
        healthy_names = [s.name for s in snap.healthy() if s.name in self._names]
        if not healthy_names:
            raise NoHealthyWorker()

        if config.OVERLAP_ADAPTIVE_ENABLED:
            from bench.overlap_measurement import jaccard

            overlap = jaccard(self._prev_chunk_ids or set(), set(chunk_ids))
            alpha, beta = adaptive_weights(
                overlap, config.OVERLAP_THRESHOLD,
                config.LOW_OVERLAP_ALPHA, config.HIGH_OVERLAP_ALPHA,
            )
            self._prev_chunk_ids = set(chunk_ids)
        else:
            alpha, beta = config.ALPHA, config.BETA

        loads = self._normalised_loads(snap)
        decision = self._router.choose(
            candidate_worker_names=healthy_names,
            retrieved_chunk_ids=chunk_ids,
            load_norm=loads,
            alpha=alpha,
            beta=beta,
        )
        self._router.on_request_finished(decision.worker_name, decision.ordered_chunk_ids)
        return decision

    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision:
        healthy = snap.healthy()
        if not healthy:
            raise NoHealthyWorker()

        chunk_ids = ctx.chunk_hashes or ctx.block_hashes
        pw = self.decide_order(chunk_ids, snap)
        worker = next((s for s in healthy if s.name == pw.worker_name), healthy[0])

        return Decision(
            worker=worker,
            reason="per_worker_tree",
            scores=pw.scores,
            cache_gain=pw.cache_gain,
            ordered_chunk_ids=pw.ordered_chunk_ids,
        )


class SemanticPerWorkerTreeStrategy(Strategy):
    """PerWorkerTreeStrategy + retrieval-oncesi semantik aday-on-filtresi
    (bkz. semantic_worker_router.py, GOREV_semantic_router_entegrasyonu.md).

    per_worker_tree'nin maliyeti n aday icin n*O(k^2) -- n=2'de ucuz ama
    buyudukce pahalilasiyor. Bu strateji, PerWorkerTreeRouter.choose()'u TUM
    healthy worker'lar yerine SADECE sorgunun icerigine gore en alakali
    SEMANTIC_TOP_K worker'la cagirir. Aday listesi bos kalirsa (ornegin
    semantik favoriler o an unhealthy ise) tum healthy worker'lara geri
    duser -- bir istegi asla ac birakmaz.

    SINIRLAMA -- durustce isaretlenmeli: semantic_worker_router.py'nin
    kendi docstring'i "retrieval'dan ONCE" diyor, ama bu router'da soru
    metni chunk'lar retrieve edildikten SONRA, zaten kurulmus prompt'un
    icinde geliyor (main.py _prompt_text()). decide_order() bu yuzden
    query_text'i AYRI bir parametre olarak aliyor (chunk_ids'ten degil) --
    su an main.py'nin /router/decide_order endpoint'i bunu göndermiyor
    (sadece chunk_ids var, bkz. main.py:354), yani query_text=None
    gelirse semantik filtre pasif kalir ve davranis duz per_worker_tree'ye
    esdegerdir. Bu strateji UCTAN UCA henuz vLLM/main.py uzerinden
    dogrulanmadi -- bkz. smoke_test_semantic.py (izole, main.py'ye
    dokunmadan dogrulama; ayni PerWorkerTreeStrategy'nin kendi
    docstring'inde itiraf ettigi sinirlama).
    """

    name = "semantic_per_worker_tree"

    def __init__(self, tracker: PrefixTracker):
        super().__init__(tracker)
        self._names = [w.name for w in config.WORKERS if w.enabled]
        self._per_worker_router = PerWorkerTreeRouter(worker_names=self._names)
        self._semantic_router = SemanticWorkerRouter(
            worker_names=self._names, embed_fn=real_embed, lr=config.SEMANTIC_CENTROID_LR,
        )

    def warmup(self) -> None:
        """main.py's startup checks for this (hasattr, same convention as
        decide_order's optional-capability check) so the SentenceTransformer
        load lands here, once, rather than on the first live request."""
        self._semantic_router.warmup()

    def decide_order(self, chunk_ids: list[str], snap: Snapshot,
                     query_text: str | None = None) -> PerWorkerDecision:
        healthy_names = [s.name for s in snap.healthy() if s.name in self._names]
        if not healthy_names:
            raise NoHealthyWorker()

        if query_text:
            pred = self._semantic_router.predict_candidates(
                query_text, top_k=config.SEMANTIC_TOP_K
            )
            candidates = [n for n in pred.ranked_workers if n in healthy_names]
            if not candidates:
                candidates = healthy_names  # semantic favourites are down -- don't starve
        else:
            candidates = healthy_names

        loads = self._normalised_loads(snap)
        decision = self._per_worker_router.choose(
            candidate_worker_names=candidates,
            retrieved_chunk_ids=chunk_ids,
            load_norm=loads,
            alpha=config.ALPHA,
            beta=config.BETA,
        )
        self._per_worker_router.on_request_finished(decision.worker_name, decision.ordered_chunk_ids)
        if query_text:
            self._semantic_router.on_request_finished(decision.worker_name, query_text)
        return decision

    def select(self, ctx: RequestContext, snap: Snapshot) -> Decision:
        healthy = snap.healthy()
        if not healthy:
            raise NoHealthyWorker()

        chunk_ids = ctx.chunk_hashes or ctx.block_hashes
        # ctx.prompt_text is the flattened prompt (already includes the
        # retrieved chunks) rather than a pre-retrieval question -- see the
        # class docstring's SINIRLAMA note. Still the closest thing to query
        # content this Strategy interface exposes, and good enough for the
        # centroid signal (chunk text dominates a hashing/e5 embedding far
        # less than the recurring question phrasing does).
        pw = self.decide_order(chunk_ids, snap, query_text=ctx.prompt_text)
        worker = next((s for s in healthy if s.name == pw.worker_name), healthy[0])

        return Decision(
            worker=worker,
            reason="semantic_per_worker_tree",
            scores=pw.scores,
            cache_gain=pw.cache_gain,
            ordered_chunk_ids=pw.ordered_chunk_ids,
        )


class NoHealthyWorker(RuntimeError):
    pass


_REGISTRY: dict[str, type[Strategy]] = {
    RoundRobin.name: RoundRobin,
    LeastLoaded.name: LeastLoaded,
    CacheAware.name: CacheAware,
    AdaptiveCacheAware.name: AdaptiveCacheAware,
    CacheWeaverDualMapStrategy.name: CacheWeaverDualMapStrategy,
    PerWorkerTreeStrategy.name: PerWorkerTreeStrategy,
    SemanticPerWorkerTreeStrategy.name: SemanticPerWorkerTreeStrategy,
}


def build_strategy(name: str, tracker: PrefixTracker) -> Strategy:
    try:
        return _REGISTRY[name](tracker)
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; expected one of {sorted(_REGISTRY)}"
        ) from None
