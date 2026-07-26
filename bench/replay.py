"""Replay a trace against the router and measure what happened.

This is the experiment harness. It owns four jobs:

1. Retrieval -- embed the query, take the top-k chunks by cosine similarity.
2. Chunk ordering -- the ablation lever. `canonical` sorts retrieved chunks by
   chunk_id so that two queries sharing chunks also share a prompt *prefix*;
   `relevance` keeps the retriever's own ranking, which is better for answer
   quality but breaks the prefix chain at the first differing chunk.
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

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.total_s is None or self.output_tokens < 2:
            return None
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)


def retrieve(corpus: Corpus, qvec: np.ndarray, k: int, order: str) -> list[int]:
    """Top-k by cosine similarity, then ordered according to the ablation arm."""
    sims = corpus.embeddings @ qvec
    top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top = top[np.argsort(-sims[top])]          # relevance order
    if order == "canonical":
        top = top[np.argsort(corpus.chunk_ids[top])]
    return top.tolist()


def build_messages(corpus: Corpus, idxs: list[int], question: str) -> list[dict]:
    blocks = "\n\n".join(corpus.texts[i] for i in idxs)
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
    }
    # Chunk ids travel alongside the request so a chunk-coverage strategy can use
    # them directly. Under canonical ordering the router does not need them --
    # prefix matching on the ordered prompt already captures chunk overlap.
    headers = {"x-chunk-ids": ",".join(str(c) for c in retrieved_ids)}

    start = time.perf_counter()
    try:
        async with client.stream("POST", f"{args.router}/v1/chat/completions",
                                 json=payload, headers=headers) as r:
            res.worker = r.headers.get("x-router-worker")
            res.reason = r.headers.get("x-router-reason")
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
                    delta = json.loads(body)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("content"):
                    if res.ttft_s is None:
                        res.ttft_s = time.perf_counter() - start
                    res.output_tokens += 1
                    res.output_text += delta["content"]
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"

    res.total_s = time.perf_counter() - start
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

    plan = []
    for rec, qv in zip(trace, qvecs):
        idxs = retrieve(corpus, qv, args.top_k, args.order)
        plan.append((rec, build_messages(corpus, idxs, rec["query_text"]),
                     corpus.chunk_ids[idxs].tolist()))

    limits = httpx.Limits(max_connections=args.max_concurrency,
                          max_keepalive_connections=args.max_concurrency)
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        before = await scrape_counters(client, args.worker)

        print(f"replaying {len(plan)} requests | order={args.order} "
              f"top_k={args.top_k} speedup={args.speedup}x")
        t0 = time.perf_counter()
        tasks = []
        for rec, messages, ids in plan:
            due = rec["arrival_offset_s"] / args.speedup
            delay = due - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(fire(client, args, rec, messages, ids, t0)))
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

        after = await scrape_counters(client, args.worker)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            row = r.__dict__ | {"tpot_s": r.tpot_s}
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
    p.add_argument("--order", choices=["canonical", "relevance"], default="canonical")
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
