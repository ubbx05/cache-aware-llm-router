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

    Units caveat: DualMap's ReplicaState.num_pending_prefill_tokens wants a
    TOKEN count, but this router only exposes request-level queue depth
    (num_requests_waiting). AVG_PROMPT_TOKENS_ESTIMATE below is a placeholder
    conversion -- replace it with a number you actually measured (e.g. mean
    prompt_tokens from a validate_tracker.py run) before trusting any TTFT
    estimate this strategy produces.
    """

    name = "cacheweaver_dualmap"

    # TODO(kalibrasyon): placeholder, gercek RAG promptlarinin ortalama token
    # sayisiyla degistir (validate_tracker.py ciktisindaki "prompt ort" satiri).
    AVG_PROMPT_TOKENS_ESTIMATE = 1700

    def __init__(self, tracker: PrefixTracker):
        super().__init__(tracker)
        # Fixed at construction time so replica indices stay stable across
        # calls even if a worker briefly drops out of snap.healthy().
        self._names = [w.name for w in config.WORKERS if w.enabled]
        self._name_to_idx = {name: i for i, name in enumerate(self._names)}
        self._router = CacheWeaverDualMapRouter(num_replicas=len(self._names))
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
    """

    name = "per_worker_tree"

    def __init__(self, tracker: PrefixTracker):
        super().__init__(tracker)
        self._names = [w.name for w in config.WORKERS if w.enabled]
        self._router = PerWorkerTreeRouter(worker_names=self._names)

    def decide_order(self, chunk_ids: list[str], snap: Snapshot) -> PerWorkerDecision:
        """Ham (siralanmamis, retrieval sirasindaki) chunk_ids alir, worker +
        o worker'a gore en iyi sira karari doner. Dispatch-time bookkeeping
        burada yapilir (cacheweaver_dualmap ile ayni zamanlama konvansiyonu,
        ablation tablosunun timing farkindan degil algoritma farkindan
        etkilenmesi icin)."""
        healthy_names = [s.name for s in snap.healthy() if s.name in self._names]
        if not healthy_names:
            raise NoHealthyWorker()

        loads = self._normalised_loads(snap)
        decision = self._router.choose(
            candidate_worker_names=healthy_names,
            retrieved_chunk_ids=chunk_ids,
            load_norm=loads,
            alpha=config.ALPHA,
            beta=config.BETA,
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
