"""Router-side model of what each worker has in its KV cache.

vLLM exposes aggregate cache statistics but not "do you currently hold a cache
entry for this specific prefix?". A cache-aware router therefore has to keep
its own approximation, which is what SGLang-style routers do as well.

The approximation here: for each worker, remember the hashes of the
block-aligned prefixes it has been sent, in an LRU of bounded size. The LRU is
a stand-in for the engine's own eviction policy; it will drift from reality
whenever the engine evicts on a different schedule. Quantifying that drift
(router-believed hit rate vs. vllm:prefix_cache_hits_total) is a cheap and
convincing validation experiment for the paper.

Chunk-level affinity for RAG plugs in here: instead of hashing only the
contiguous leading prefix, hash each retrieved chunk independently and score a
worker by how many of the query's chunks it already holds. `chunk_gain` below
is the hook for that; it is deliberately separate from `prefix_gain`.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Iterable, Protocol

import config


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class ApproxTokenizer:
    """Character-bucket stand-in for a real tokenizer.

    Zero dependencies and fast, but the block boundaries it produces do not
    line up with the engine's. Fine for wiring and smoke tests; swap in
    HFTokenizer before collecting any number that goes into the paper.
    """

    CHARS_PER_TOKEN = 3.5

    def encode(self, text: str) -> list[int]:
        step = max(1, int(self.CHARS_PER_TOKEN))
        return [ord(text[i]) for i in range(0, len(text), step)]


class HFTokenizer:
    """Real tokenizer, lazily imported so `transformers` stays optional."""

    def __init__(self, model: str):
        from transformers import AutoTokenizer  # imported lazily on purpose

        self._tok = AutoTokenizer.from_pretrained(model)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)


def get_tokenizer() -> Tokenizer:
    if config.TOKENIZER_MODE == "hf":
        return HFTokenizer(config.TOKENIZER_MODEL)
    return ApproxTokenizer()


def _hash(previous: str, token_ids: Iterable[int]) -> str:
    """Chained hash so that block i's hash depends on blocks 0..i-1.

    This mirrors vLLM's own prefix hashing: two requests share block i only if
    every preceding block matched too.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(previous.encode())
    h.update(",".join(str(t) for t in token_ids).encode())
    return h.hexdigest()


def block_hashes(token_ids: list[int], block_size: int | None = None) -> list[str]:
    block_size = block_size or config.BLOCK_SIZE
    hashes: list[str] = []
    previous = ""
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        previous = _hash(previous, token_ids[start:start + block_size])
        hashes.append(previous)
    return hashes


class PrefixTracker:
    def __init__(self, worker_names: Iterable[str], capacity: int | None = None):
        self.capacity = capacity or config.TRACKER_CAPACITY
        self._seen: dict[str, OrderedDict[str, None]] = {
            name: OrderedDict() for name in worker_names
        }

    def ensure_worker(self, name: str) -> None:
        self._seen.setdefault(name, OrderedDict())

    def prefix_gain(self, worker: str, hashes: list[str]) -> float:
        """Fraction of the request's leading blocks this worker likely holds.

        Returns a value in [0, 1]. Matching stops at the first miss, because a
        prefix cache cannot skip a gap: block i is only reusable if every block
        before it was reused too.
        """
        if not hashes:
            return 0.0
        seen = self._seen.get(worker)
        if not seen:
            return 0.0
        matched = 0
        for h in hashes:
            if h in seen:
                matched += 1
            else:
                break
        return matched / len(hashes)

    def matched_blocks(self, worker: str, hashes: list[str]) -> int:
        """Absolute count of reusable leading blocks (for token-level cost modelling)."""
        return int(round(self.prefix_gain(worker, hashes) * len(hashes)))

    def chunk_gain(self, worker: str, chunk_hashes: list[str]) -> float:
        """Chunk-coverage affinity, for RAG workloads.

        Unlike prefix_gain this does NOT stop at the first miss: retrieved
        chunks sit in the middle of the prompt and are reusable individually
        (given a blending mechanism such as CacheBlend). This is the term that
        separates this router from prefix-only baselines.

        Not wired into the scoring function yet -- it activates in the RAG phase,
        once the retriever forwards its chunk ids to the router.
        """
        if not chunk_hashes:
            return 0.0
        seen = self._seen.get(worker)
        if not seen:
            return 0.0
        return sum(1 for h in chunk_hashes if h in seen) / len(chunk_hashes)

    def record(self, worker: str, hashes: list[str]) -> None:
        """Register that `worker` has now processed these blocks."""
        seen = self._seen.setdefault(worker, OrderedDict())
        for h in hashes:
            if h in seen:
                seen.move_to_end(h)
            else:
                seen[h] = None
        while len(seen) > self.capacity:
            seen.popitem(last=False)  # evict least recently used

    def size(self, worker: str) -> int:
        return len(self._seen.get(worker, ()))
