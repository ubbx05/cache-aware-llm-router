"""Replay a trace against the router and measure what happened.

This is the experiment harness. It owns four jobs:

1. Retrieval -- embed the query, take the top-k chunks by cosine similarity.
2. Chunk ordering -- the ablation lever. `canonical` sorts retrieved chunks by
   chunk_id so that two queries sharing chunks also share a prompt *prefix*;
   `relevance` keeps the retriever's own ranking, which is better for answer
   quality but breaks the prefix chain at the first differing chunk; `greedy`
   is CacheWeaver's Algorithm 1 (bench/cacheweaver_util.py) run against a
   single global knowledge tree, i.e. the published, worker-blind version of
   the reordering idea. It is the third arm precisely because canonical and
   relevance bracket the trade-off from either end -- canonical buys prefix
   reuse by discarding the ranking, relevance keeps the ranking and gets no
   reuse -- and greedy claims to get reuse *without* fixing an order in
   advance. Whether that claim survives a real multi-worker deployment is
   what `per_worker_tree` (the fourth arm) exists to test: greedy here has
   exactly one tree for the whole cluster, so anything it believes is cached
   may in fact be cached on the other replica.
3. Timed dispatch -- fire each request at its scheduled offset regardless of
   whether earlier ones have finished. Waiting for completions would collapse
   the queue to depth one and the Load(w) term would never activate.
4. Measurement -- TTFT, TPOT, worker assignment, and the worker-side prefix
   cache counters across the whole run.

Prompt layout is deliberate:

    system message : fixed instructions + retrieved chunks
    user message   : the question

Constant text first, then chunks (canonically ordered), then the only part
that always varies. That ordering is what gives the prefix cache something to
hold onto.

Usage:
    python replay.py --corpus ./corpus --trace trace.jsonl --out results.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

# Same module the router itself uses for per_worker_tree / cacheweaver_dualmap,
# imported rather than reimplemented so the `greedy` arm and the router's own
# reordering can never drift apart. Only touched when --order greedy.
from cacheweaver_util import CacheWeaverKnowledgeTree

SYSTEM_PROMPT = (
    "Sen Türk-İslam bilim tarihi üzerine sorulara cevap veren bir asistansın. "
    "Sana verilen bağlam parçalarını kullanarak soruyu yanıtla. "
    "Cevabın kısa ve doğrudan olsun. Her zaman sadece Türkçe cevap ver."
)


@dataclass
class Corpus:
    chunk_ids: np.ndarray
    texts: list[str]
    embeddings: np.ndarray  # L2-normalised, so a dot product is cosine similarity

    @classmethod
    def load(cls, d: Path) -> "Corpus":
        ids, texts = [], []
        with (d / "corpus.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ids.append(row["chunk_id"])
                # padded text is what the engine sees; unpadded would misalign blocks
                texts.append(row["text_padded"])
        emb = np.load(d / "embeddings.npy")
        if len(ids) != emb.shape[0]:
            raise SystemExit(f"corpus/embedding mismatch: {len(ids)} vs {emb.shape[0]}")
        return cls(np.array(ids), texts, emb)


@dataclass
class Result:
    request_id: int
    session_id: int
    scheduled_s: float
    sent_s: float
    ttft_s: float | None = None
    total_s: float | None = None
    output_tokens: int = 0
    worker: str | None = None
    reason: str | None = None
    retrieved: list[int] = field(default_factory=list)
    expected: list[int] = field(default_factory=list)
    retrieval_hit: bool = False
    error: str | None = None
    output_text: str = ""
    gold_answer: str | None = None

    # The two sides of the tracker-validation comparison.
    # `believed_*` is the router's own estimate, read off a response header.
    # `actual_cached_tokens` is what the engine reports it truly reused, via
    # usage.prompt_tokens_details.cached_tokens. Everything cache_gain drives
    # rests on these two agreeing, so it gets measured rather than assumed.
    believed_cached_tokens: int | None = None
    believed_frac: float | None = None
    prompt_blocks: int | None = None
    actual_cached_tokens: int | None = None
    prompt_tokens: int | None = None

    # --order greedy only. `greedy_depth` is how many leading chunks of the
    # order we shipped were already on a cached root path, measured against
    # the tree BEFORE this request was inserted into it -- i.e. the reusable
    # prefix depth Algorithm 1 was actually able to find, in chunks. It is
    # the arm's own claim, recorded separately from the engine's
    # actual_cached_tokens so the two can be compared instead of conflated:
    # depth > 0 with actual_frac ~ 0 is the single-global-tree assumption
    # failing out loud, which on a 2-worker cluster is the expected result.
    greedy_depth: int | None = None
    greedy_reordered: bool | None = None

    @property
    def actual_frac(self) -> float | None:
        if self.actual_cached_tokens is None or not self.prompt_tokens:
            return None
        return self.actual_cached_tokens / self.prompt_tokens

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.total_s is None or self.output_tokens < 2:
            return None
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)


# MUST match main.py's CHUNK_SEP exactly, byte for byte -- see the comment
# there. Only used when --order per_worker_tree; canonical/relevance keep
# joining with plain "\n\n", so every previously-measured run (top_k A/B,
# tracker validation, 2-worker comparison) is completely unaffected by this.
CHUNK_SEP = "\n\n<<<CHUNK>>>\n\n"


def retrieve(corpus: Corpus, qvec: np.ndarray, k: int, order: str) -> list[int]:
    """Top-k by cosine similarity, then ordered according to the ablation arm.

    per_worker_tree and greedy both get the same (unsorted-by-id) relevance
    order as the "relevance" arm: their final order is decided later, from
    cache state -- per-worker by /router/decide_order for the former, from the
    harness-side global tree in fire_greedy() for the latter. What we send here
    is only the retrieval-order candidate set they choose from, and it matters
    that it *is* relevance order: greedy_reorder()'s fallback when nothing is
    cached is "keep the input order", and protect_top_k pins the first K of it,
    so both only mean "most relevant first" if this list is ranked.
    """
    sims = corpus.embeddings @ qvec
    top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top = top[np.argsort(-sims[top])]          # relevance order
    if order == "canonical":
        top = top[np.argsort(corpus.chunk_ids[top])]
    return top.tolist()


def build_messages(corpus: Corpus, idxs: list[int], question: str, sep: str = "\n\n") -> list[dict]:
    blocks = sep.join(corpus.texts[i] for i in idxs)
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nBağlam:\n\n{blocks}"},
        {"role": "user", "content": question},
    ]


async def fire(client: httpx.AsyncClient, args, rec: dict, messages: list[dict],
               retrieved_ids: list[int], t0: float) -> Result:
    res = Result(
        request_id=rec["request_id"],
        session_id=rec["session_id"],
        scheduled_s=rec["arrival_offset_s"] / args.speedup,
        sent_s=time.perf_counter() - t0,
        retrieved=retrieved_ids,
        expected=rec.get("expected_chunk_ids", []),
        gold_answer=rec.get("gold_answer"),
    )
    res.retrieval_hit = bool(set(res.retrieved) & set(res.expected))

    payload = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        # Makes vLLM append a final chunk carrying usage, including
        # prompt_tokens_details.cached_tokens -- the engine's own count of how
        # many prompt tokens it served from cache. Without this the run cannot
        # be graded against the router's belief.
        "stream_options": {"include_usage": True},
    }
    # Chunk ids travel alongside the request so a chunk-coverage strategy can use
    # them directly. Under canonical ordering the router does not need them --
    # prefix matching on the ordered prompt already captures chunk overlap.
    headers = {
        "x-chunk-ids": ",".join(str(c) for c in retrieved_ids),
        "x-session-id": str(rec["session_id"]),
    }

    start = time.perf_counter()
    try:
        async with client.stream("POST", f"{args.router}/v1/chat/completions",
                                 json=payload, headers=headers) as r:
            res.worker = r.headers.get("x-router-worker")
            res.reason = r.headers.get("x-router-reason")
            # Absent when talking to a bare vLLM or a third-party router; the
            # run still works, only the tracker comparison is unavailable.
            _bt = r.headers.get("x-router-believed-cached-tokens")
            _bf = r.headers.get("x-router-believed-frac")
            _pb = r.headers.get("x-router-prompt-blocks")
            res.believed_cached_tokens = int(_bt) if _bt else None
            res.believed_frac = float(_bf) if _bf else None
            res.prompt_blocks = int(_pb) if _pb else None
            if r.status_code != 200:
                await r.aread()
                res.error = f"http {r.status_code}"
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue

                # The usage chunk arrives last and carries an EMPTY choices
                # list. Indexing choices[0] first would raise and skip it, which
                # is exactly how this data gets lost silently.
                usage = chunk.get("usage")
                if usage:
                    res.prompt_tokens = usage.get("prompt_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    res.actual_cached_tokens = details.get("cached_tokens")

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if res.ttft_s is None:
                        res.ttft_s = time.perf_counter() - start
                    res.output_tokens += 1
                    res.output_text += delta["content"]
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"

    res.total_s = time.perf_counter() - start
    return res


async def fire_two_phase(client: httpx.AsyncClient, args, rec: dict,
                         id_to_text: dict[str, str], chunk_ids: list[str],
                         question: str, t0: float) -> Result:
    """per_worker_tree only: calls /router/decide_order FIRST with the raw
    (retrieval-order) chunk ids, builds the prompt with the RETURNED order
    (using CHUNK_SEP so main.py could in principle re-split it), then sends
    the real completion with x-router-force-worker so /v1/chat/completions
    skips select() and goes straight to the already-chosen worker. Kept as a
    separate function from fire() on purpose -- zero risk of the extra
    round-trip or CHUNK_SEP changing behaviour for canonical/relevance.
    """
    res = Result(
        request_id=rec["request_id"],
        session_id=rec["session_id"],
        scheduled_s=rec["arrival_offset_s"] / args.speedup,
        sent_s=time.perf_counter() - t0,
        retrieved=chunk_ids,
        expected=rec.get("expected_chunk_ids", []),
        gold_answer=rec.get("gold_answer"),
    )
    res.retrieval_hit = bool(set(res.retrieved) & set(res.expected))

    try:
        dr = await client.post(f"{args.router}/router/decide_order",
                               json={"chunk_ids": chunk_ids})
        dr.raise_for_status()
        decided = dr.json()
    except Exception as exc:  # noqa: BLE001
        res.error = f"decide_order failed: {type(exc).__name__}: {exc}"
        return res

    worker_name = decided.get("worker")
    ordered_ids = decided.get("ordered_chunk_ids") or chunk_ids
    blocks = CHUNK_SEP.join(id_to_text[c] for c in ordered_ids)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nBağlam:\n\n{blocks}"},
        {"role": "user", "content": question},
    ]

    payload = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "x-chunk-ids": ",".join(str(c) for c in ordered_ids),
        "x-router-force-worker": worker_name,
        "x-session-id": str(rec["session_id"]),
    }

    start = time.perf_counter()
    try:
        async with client.stream("POST", f"{args.router}/v1/chat/completions",
                                 json=payload, headers=headers) as r:
            res.worker = r.headers.get("x-router-worker")
            res.reason = r.headers.get("x-router-reason")
            _bt = r.headers.get("x-router-believed-cached-tokens")
            _bf = r.headers.get("x-router-believed-frac")
            _pb = r.headers.get("x-router-prompt-blocks")
            res.believed_cached_tokens = int(_bt) if _bt else None
            res.believed_frac = float(_bf) if _bf else None
            res.prompt_blocks = int(_pb) if _pb else None
            if r.status_code != 200:
                await r.aread()
                res.error = f"http {r.status_code}"
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    res.prompt_tokens = usage.get("prompt_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    res.actual_cached_tokens = details.get("cached_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if res.ttft_s is None:
                        res.ttft_s = time.perf_counter() - start
                    res.output_tokens += 1
                    res.output_text += delta["content"]
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"

    res.total_s = time.perf_counter() - start
    return res


def reusable_prefix_depth(tree: CacheWeaverKnowledgeTree, ordered_ids: list[str]) -> int:
    """How many leading chunks of `ordered_ids` sit on a still-cached root
    path. greedy_reorder() computes exactly this internally but returns only
    the order, so we re-walk the same tree with the same _is_cached() test --
    the same pattern (and the same reason) as
    per_worker_tree_router._cache_hit_tokens. Must be called BEFORE inserting
    this request, or it trivially returns len(ordered_ids).

    Counted in chunks, not tokens, on purpose: this arm's claim is about
    Algorithm 1's reusable-prefix *depth*, which is what CacheWeaver reports.
    The token-level truth is already measured independently, by the engine,
    as actual_cached_tokens.
    """
    node = tree._root
    depth = 0
    for chunk_id in ordered_ids:
        child = node.get_child(chunk_id)
        if child is None or not tree._is_cached(child):
            break
        node = child
        depth += 1
    return depth


async def fire_greedy(client: httpx.AsyncClient, args, rec: dict,
                      tree: CacheWeaverKnowledgeTree, id_to_text: dict[str, str],
                      chunk_ids: list[str], question: str, t0: float) -> Result:
    """--order greedy: CacheWeaver Algorithm 1 against one global tree.

    Ordering happens HERE, at dispatch time, not in the up-front plan loop the
    way canonical/relevance do it -- the whole point of the arm is that the
    order depends on cache state, which only exists once earlier requests have
    run. Everything downstream of the ordering is deliberately identical to
    the canonical/relevance path: same "\n\n" join (NOT CHUNK_SEP), same
    fire(), same single round-trip through /v1/chat/completions. So a
    greedy-vs-canonical delta is attributable to chunk order and nothing else.

    Insertion timing is a real choice, not a detail. CacheWeaver inserts a
    path when the request *finishes*, on the argument that only then are its
    KV blocks certainly computed -- but under the concurrent dispatch this
    harness is built to produce, in-flight requests are then invisible to each
    other and a burst of similar queries all reorder against a stale tree.
    --greedy-insert dispatch is the other end of that trade (visible
    immediately, but claims a path the engine may not have finished caching),
    and is the same dispatch-vs-completion question the router-side
    bookkeeping ablation asks.
    """
    ordered = tree.greedy_reorder(list(chunk_ids), protect_top_k=args.greedy_protect_top_k)
    depth = reusable_prefix_depth(tree, ordered)
    if args.greedy_insert == "dispatch":
        tree.insert(ordered)

    messages = [
        {"role": "system",
         "content": f"{SYSTEM_PROMPT}\n\nBağlam:\n\n" + "\n\n".join(id_to_text[c] for c in ordered)},
        {"role": "user", "content": question},
    ]

    # `ordered` (not chunk_ids) goes to fire(), so the x-chunk-ids header the
    # router's tracker reads matches the order actually present in the prompt.
    res = await fire(client, args, rec, messages, ordered, t0)

    if args.greedy_insert == "completion":
        # Insert even on error: a request that failed mid-stream still had its
        # prompt prefilled, so those blocks are in the engine's cache and the
        # tree would otherwise under-report. Matches _is_cached()'s recency
        # approximation, which never claimed to know about failures either.
        tree.insert(ordered)

    res.greedy_depth = depth
    res.greedy_reordered = ordered != list(chunk_ids)
    return res


async def scrape_counters(client: httpx.AsyncClient, urls: list[str]) -> dict[str, float]:
    """Worker-side prefix cache counters, summed across workers."""
    wanted = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
              "vllm:prompt_tokens_cached_total")
    out = {k: 0.0 for k in wanted}
    for url in urls:
        try:
            r = await client.get(f"{url}/metrics", timeout=5.0)
        except Exception:  # noqa: BLE001
            continue
        for line in r.text.splitlines():
            if line.startswith("#"):
                continue
            for name in wanted:
                if line.startswith(name + "{") or line.startswith(name + " "):
                    try:
                        out[name] += float(line.rsplit(" ", 1)[1])
                    except (ValueError, IndexError):
                        pass
    return out


async def run(args) -> None:
    corpus = Corpus.load(Path(args.corpus))
    trace = [json.loads(l) for l in Path(args.trace).open(encoding="utf-8")]
    if args.limit:
        trace = trace[:args.limit]

    # Queries are embedded up front, in one batch, before the clock starts.
    # Doing it inline would put the load generator's own CPU work on the
    # critical path and jitter the arrival times we are trying to control.
    # Retrieval is identical across strategies, so this removes noise without
    # favouring any arm.
    from sentence_transformers import SentenceTransformer
    print(f"embedding {len(trace)} queries on {args.device} ...")
    model = SentenceTransformer(args.embed_model, device=args.device)
    qvecs = model.encode([f"query: {r['query_text']}" for r in trace],
                         batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True).astype("float32")

    id_to_text = {cid: text for cid, text in zip(corpus.chunk_ids.tolist(), corpus.texts)}

    # One tree for the whole cluster -- CacheWeaver as published. Built here
    # rather than per-run-arm so its recency clock starts with the replay,
    # matching the engine caches, which are cold at this point by protocol.
    greedy_tree = (CacheWeaverKnowledgeTree(cache_ttl_seconds=args.greedy_ttl)
                   if args.order == "greedy" else None)

    plan = []
    for rec, qv in zip(trace, qvecs):
        idxs = retrieve(corpus, qv, args.top_k, args.order)
        chunk_ids = corpus.chunk_ids[idxs].tolist()
        if args.order in ("per_worker_tree", "greedy"):
            # Messages aren't built yet -- the final chunk order depends on
            # cache state that does not exist yet. per_worker_tree resolves it
            # in fire_two_phase() (which worker /router/decide_order picks),
            # greedy in fire_greedy() (the global tree at dispatch time).
            plan.append((rec, None, chunk_ids, rec["query_text"]))
        else:
            plan.append((rec, build_messages(corpus, idxs, rec["query_text"]), chunk_ids, None))

    limits = httpx.Limits(max_connections=args.max_concurrency,
                          max_keepalive_connections=args.max_concurrency)
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        before = await scrape_counters(client, args.worker)

        print(f"replaying {len(plan)} requests | order={args.order} "
              f"top_k={args.top_k} speedup={args.speedup}x")
        t0 = time.perf_counter()
        tasks = []
        for rec, messages, ids, question in plan:
            due = rec["arrival_offset_s"] / args.speedup
            delay = due - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            if args.order == "per_worker_tree":
                tasks.append(asyncio.create_task(
                    fire_two_phase(client, args, rec, id_to_text, ids, question, t0)))
            elif args.order == "greedy":
                tasks.append(asyncio.create_task(
                    fire_greedy(client, args, rec, greedy_tree, id_to_text, ids, question, t0)))
            else:
                tasks.append(asyncio.create_task(fire(client, args, rec, messages, ids, t0)))
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

        after = await scrape_counters(client, args.worker)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            row = r.__dict__ | {"tpot_s": r.tpot_s, "actual_frac": r.actual_frac}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summarise(results, before, after, wall, args, out)


def summarise(results, before, after, wall, args, out: Path) -> None:
    ok = [r for r in results if r.error is None and r.ttft_s is not None]
    failed = len(results) - len(ok)

    def pct(vals, p):
        return statistics.quantiles(sorted(vals), n=100)[p - 1] if len(vals) > 1 else (vals[0] if vals else float("nan"))

    ttfts = [r.ttft_s for r in ok]
    tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
    lags = [r.sent_s - r.scheduled_s for r in results]

    dq = after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"]
    dh = after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"]

    workers: dict[str, int] = {}
    for r in results:
        workers[r.worker or "?"] = workers.get(r.worker or "?", 0) + 1

    print()
    print(f"completed        : {len(ok)}/{len(results)}" + (f"  ({failed} failed)" if failed else ""))
    print(f"wall clock       : {wall:.1f}s")
    print(f"schedule lag     : p50={pct(lags, 50) * 1000:.0f}ms  p99={pct(lags, 99) * 1000:.0f}ms")
    if ttfts:
        print(f"TTFT             : p50={pct(ttfts, 50):.3f}s  p90={pct(ttfts, 90):.3f}s  p99={pct(ttfts, 99):.3f}s")
    if tpots:
        print(f"TPOT             : p50={pct(tpots, 50) * 1000:.1f}ms")
    print(f"prefix cache     : {dh:.0f}/{dq:.0f} tokens = {dh / dq:.1%} hit rate" if dq > 0 else "prefix cache     : no data")
    print(f"retrieval recall : {sum(r.retrieval_hit for r in results) / len(results):.1%}")
    print(f"worker split     : {workers}")

    # The greedy arm's own claim, next to the engine's verdict on it. These
    # two are measured independently -- reorder depth comes from the harness
    # tree, cached frac from usage.prompt_tokens_details -- so a large gap
    # between them is a finding, not a bug: it is the single-global-tree
    # assumption being wrong about which replica holds the blocks.
    if args.order == "greedy":
        depths = [r.greedy_depth for r in results if r.greedy_depth is not None]
        fracs = [r.actual_frac for r in ok if r.actual_frac is not None]
        moved = sum(1 for r in results if r.greedy_reordered)
        if depths:
            print(f"reorder depth    : mean={statistics.mean(depths):.2f} chunks  "
                  f"nonzero={sum(1 for d in depths if d > 0) / len(depths):.1%}  "
                  f"(insert={args.greedy_insert} ttl={args.greedy_ttl:g}s "
                  f"protect_top_k={args.greedy_protect_top_k})")
            print(f"order changed    : {moved / len(results):.1%} of requests")
        if fracs:
            print(f"engine cached    : mean={statistics.mean(fracs):.1%} of prompt tokens")

    print(f"results          : {out}")

    if pct(lags, 99) > 1.0:
        print()
        print("WARNING: the client fell behind its own schedule by >1s at p99.")
        print("  The measured arrival pattern no longer matches the trace, so the")
        print("  load axis of this run is not trustworthy. Lower --speedup.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--trace", default="trace.jsonl")
    p.add_argument("--out", default="results.jsonl")
    p.add_argument("--router", default="http://localhost:8080")
    p.add_argument("--worker", action="append", default=None,
                   help="worker base URL for cache counters; repeatable")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--order", choices=["canonical", "relevance", "greedy", "per_worker_tree"],
                   default="canonical")
    p.add_argument("--greedy-protect-top-k", type=int, default=0,
                   help="--order greedy: pin the first K retrieval-ranked chunks in place, "
                        "reorder only what follows (config.PROTECT_TOP_K's client-side twin). "
                        "0 = pure Algorithm 1")
    p.add_argument("--greedy-ttl", type=float, default=30.0,
                   help="--order greedy: seconds a tree node is assumed still cached. "
                        "30.0 matches per_worker_tree_router / cacheweaver_dualmap_router")
    p.add_argument("--greedy-insert", choices=["completion", "dispatch"], default="completion",
                   help="--order greedy: when a served order enters the tree. completion = "
                        "CacheWeaver's own semantics (default); dispatch = visible to "
                        "concurrent requests immediately")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--speedup", type=float, default=1.0,
                   help="compress the trace timeline; 10 = ten times faster")
    p.add_argument("--limit", type=int, default=0, help="replay only the first N requests")
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--embed-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if not args.worker:
        args.worker = ["http://localhost:8000"]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
