"""Measure the throughput/latency curve of a single worker, and read LOAD_REF off it.

LOAD_REF started as a guess. `config.py` asks for "the batch size at which TTFT
starts to degrade", because that is where load_norm = 0.5 and the load term has
its steepest gradient -- but nobody had measured where that batch size is. Pick
it too low and every worker reads as saturated, so the load term stops
discriminating; too high and the router keeps piling onto a worker already past
its knee. Either way the alpha/beta sweep explores a mis-scaled axis.

So measure the curve. Closed-loop, one worker, fixed concurrency per level.

The knee is located by **Kleinrock power**, throughput / latency. An earlier
version of this script looked for where TTFT "departs from flat" and that was
simply wrong: on a real engine TTFT rises from the very first added request,
exactly as queueing theory says it must, so "last flat level" collapsed to c=1
and recommended a LOAD_REF that saturates load_norm for every worker at once --
the degenerate case config.py warns about. Power needs no flat region. It climbs
while added concurrency buys more work than it costs in latency, peaks, and
falls once it stops. That peak is the operating point.

Throughput saturation is reported alongside as a second, independent read. When
the two agree the knee is solid; when they don't, the gap between them is the
region where the engine still does more aggregate work while each individual
request pays disproportionately for it -- worth a sentence in the paper.

Prompt mode is the subtle knob. Sending one fixed prompt at every concurrency
level means every request after the first hits the prefix cache, prefill goes
near-free, and the curve you measure is a decode-only curve that will place the
knee far too high. Default is therefore `unique`: each request gets its own
filler prefix, so every request pays full prefill. `--prompt-mode shared` gives
the cached-path curve deliberately, and the gap between the two is a real result
-- it is the range LOAD_REF should sit in for a workload with partial hits.

Comparing two engine configurations is only meaningful under an identical
protocol: restart, burn in, sweep. Measured on this setup, the warm-vs-cold
difference at c=16 was ~7%, the same order as the CUDA-graph effect being
measured -- comparing a warm arm against a cold one inverted the conclusion.
Hence --warmup-requests, and hence restarting for both arms, not just the one
whose flag changed.

Usage:
    # arm 1, on a freshly restarted engine
    python sweep_batch.py --worker http://100.89.101.52:8000 \
        --prompt-tokens 1687 --label eager --out sweep_eager.json

    # restart with the flag changed, keep every other argument identical
    python sweep_batch.py --worker http://100.89.101.52:8000 \
        --prompt-tokens 1687 --label cudagraph --out sweep_graph.json

    # side by side, and does the knee actually move?
    python sweep_batch.py --compare sweep_eager.json sweep_graph.json

    # recompute knees on an old sweep without spending GPU time again
    python sweep_batch.py --rescore sweep_eager.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Filler vocabulary. Real-ish Turkish words rather than random hex: token counts
# per word stay in a sane range, so --prompt-tokens lands near what was asked
# for. The text is nonsense on purpose -- this measures engine mechanics, not
# answer quality, and nonsense keeps the model from producing early EOS.
_WORDS = (
    "bilim tarih kitap yazar alim medrese astronomi matematik cebir hekim "
    "gozlem cizim harita yildiz gunes ay yer olcum deney hesap kagit murekkep "
    "kutuphane tercume eser risale bolum sayfa satir kelime anlam kaynak "
    "donem yuzyil sehir yol ticaret gemi pusula mesafe aci yaricap daire"
).split()

SYSTEM_PROMPT = (
    "Sen bir metin isleme asistanisin. Verilen baglami oku ve kisa bir ozet cikar."
)


@dataclass
class Sample:
    ttft_s: float | None = None
    total_s: float | None = None
    output_tokens: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.total_s is None or self.output_tokens < 2:
            return None
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)


@dataclass
class LevelResult:
    concurrency: int
    n_ok: int = 0
    n_err: int = 0
    wall_s: float = 0.0
    ttft_p50: float = float("nan")
    ttft_p90: float = float("nan")
    ttft_p99: float = float("nan")
    tpot_p50_ms: float = float("nan")
    output_tok_s: float = float("nan")
    prompt_tok_s: float = float("nan")
    req_s: float = float("nan")
    mean_total_s: float = float("nan")
    mean_prompt_tokens: float = float("nan")
    cache_hit_frac: float = float("nan")
    # What the router would actually see for this level. The whole point of the
    # sweep is to set a threshold on `load()`, so the mapping from concurrency to
    # the engine counters that feed load() has to be measured, not assumed equal.
    mean_running: float = float("nan")
    mean_waiting: float = float("nan")
    max_running: float = float("nan")
    errors: list[str] = field(default_factory=list)


def pct(values: list[float], p: int) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(sorted(values), n=100)[p - 1]


def make_prompt(rng: random.Random, target_tokens: int, nonce: str) -> str:
    """Filler of roughly `target_tokens` tokens, drawn from `rng`.

    Length is approximate -- the engine's own prompt_tokens is what gets
    reported, so an inexact target costs nothing.

    The nonce leads the prompt, and it has to. `rng` is seeded deterministically,
    so two sweeps run against the same engine would otherwise generate the very
    same "unique" prompts -- and the second sweep would find them all sitting in
    the prefix cache, measuring the cached path while claiming to measure cold
    prefill. That silently breaks exactly the comparison this script exists for
    (two sweeps, one box, different engine flags). A per-run nonce at the front
    breaks the prefix chain for every block that follows it.
    """
    # Calibrated against Qwen2.5's tokenizer on this Turkish vocabulary: it
    # splits these words into ~2.25 tokens each. An earlier 1.6 overshot the
    # target by 40% (1200 asked, 1715 delivered). The engine's own prompt_tokens
    # is reported either way, so this only affects how close the target lands.
    n_words = max(8, int(target_tokens / 2.25))
    body = " ".join(rng.choice(_WORDS) for _ in range(n_words))
    return f"[{nonce}] {body}"


async def one_request(client: httpx.AsyncClient, args, prompt: str) -> Sample:
    s = Sample()
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": args.max_tokens,
        # Greedy, and no early stop: every request must generate exactly
        # max_tokens, otherwise levels differ in decode work and the
        # tokens/s comparison across concurrency is not like-for-like.
        "temperature": 0.0,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    try:
        async with client.stream("POST", f"{args.worker}/v1/chat/completions",
                                 json=payload) as r:
            if r.status_code != 200:
                await r.aread()
                s.error = f"http {r.status_code}: {r.text[:120]}"
                return s
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

                # The usage chunk carries an empty choices list, so it has to be
                # read before any choices[0] access.
                usage = chunk.get("usage")
                if usage:
                    s.prompt_tokens = usage.get("prompt_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    s.cached_tokens = details.get("cached_tokens")

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if s.ttft_s is None:
                        s.ttft_s = time.perf_counter() - start
                    s.output_tokens += 1
    except Exception as exc:  # noqa: BLE001
        s.error = f"{type(exc).__name__}: {exc}"
    s.total_s = time.perf_counter() - start
    return s


async def scrape(client: httpx.AsyncClient, worker: str) -> dict[str, float]:
    wanted = ("vllm:num_requests_running", "vllm:num_requests_waiting",
              "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")
    out: dict[str, float] = {}
    try:
        r = await client.get(f"{worker}/metrics", timeout=5.0)
    except Exception:  # noqa: BLE001
        return out
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        for name in wanted:
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return out


async def run_level(client: httpx.AsyncClient, args, concurrency: int) -> LevelResult:
    """Closed loop at fixed concurrency: exactly `concurrency` requests in flight."""
    res = LevelResult(concurrency=concurrency)
    n_total = max(args.min_requests, concurrency * args.rounds)
    n_warmup = concurrency  # one full wave, discarded

    # Distinct seed per level so `unique` prompts never repeat across levels
    # either -- otherwise level N would inherit level N-1's cache entries and
    # read as artificially fast.
    rng = random.Random(args.seed * 1000 + concurrency)
    # In shared mode every request must be byte-identical, so the nonce is drawn
    # once per run rather than per request -- still unique across runs, still a
    # guaranteed hit within one.
    shared_prompt = make_prompt(random.Random(args.seed), args.prompt_tokens, args.nonce)

    counter = {"issued": 0}
    samples: list[Sample] = []
    measuring = asyncio.Event()
    gauges: list[tuple[float, float]] = []

    async def sampler() -> None:
        """Track the engine counters that feed the router's load()."""
        while True:
            if measuring.is_set():
                m = await scrape(client, args.worker)
                if m:
                    gauges.append((m.get("vllm:num_requests_running", 0.0),
                                   m.get("vllm:num_requests_waiting", 0.0)))
            await asyncio.sleep(args.sample_interval)

    async def worker_loop() -> None:
        while counter["issued"] < n_total:
            counter["issued"] += 1
            idx = counter["issued"]
            prompt = (shared_prompt if args.prompt_mode == "shared"
                      else make_prompt(rng, args.prompt_tokens, f"{args.nonce}-{concurrency}-{idx}"))
            s = await one_request(client, args, prompt)
            if idx > n_warmup:
                samples.append(s)

    sampler_task = asyncio.create_task(sampler())
    # Warmup runs untimed: the first requests after a level change pay Triton JIT
    # and CUDA-graph capture costs that have nothing to do with batch size.
    t_start = time.perf_counter()
    workers = [asyncio.create_task(worker_loop()) for _ in range(concurrency)]

    # Flip on measurement once the warmup wave should be through.
    async def arm() -> None:
        while counter["issued"] <= n_warmup:
            await asyncio.sleep(0.05)
        measuring.set()

    arm_task = asyncio.create_task(arm())
    before = await scrape(client, args.worker)
    await asyncio.gather(*workers)
    after = await scrape(client, args.worker)
    wall = time.perf_counter() - t_start
    arm_task.cancel()
    sampler_task.cancel()
    for t in (arm_task, sampler_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    ok = [s for s in samples if s.error is None and s.ttft_s is not None]
    errs = [s.error for s in samples if s.error]
    res.n_ok, res.n_err = len(ok), len(errs)
    res.errors = sorted({e for e in errs if e})[:3]
    res.wall_s = wall
    if not ok:
        return res

    ttfts = [s.ttft_s for s in ok]
    tpots = [s.tpot_s for s in ok if s.tpot_s is not None]
    res.ttft_p50, res.ttft_p90, res.ttft_p99 = pct(ttfts, 50), pct(ttfts, 90), pct(ttfts, 99)
    if tpots:
        res.tpot_p50_ms = pct(tpots, 50) * 1000

    # Throughput is measured over the span the sampled requests actually
    # occupied, not the level's full wall clock, which still includes warmup.
    span = sum(s.total_s for s in ok) / max(1, concurrency)
    res.output_tok_s = sum(s.output_tokens for s in ok) / span if span > 0 else float("nan")
    prompts = [s.prompt_tokens for s in ok if s.prompt_tokens]
    if prompts:
        res.mean_prompt_tokens = sum(prompts) / len(prompts)
        res.prompt_tok_s = sum(prompts) / span if span > 0 else float("nan")
    res.req_s = len(ok) / span if span > 0 else float("nan")
    # Stored explicitly because the power metric needs end-to-end latency, and
    # deriving it from req_s via Little's law inherits the span approximation.
    res.mean_total_s = sum(s.total_s for s in ok) / len(ok)

    cached = [s.cached_tokens for s in ok if s.cached_tokens is not None]
    if cached and prompts:
        res.cache_hit_frac = sum(cached) / max(1.0, sum(prompts))

    dq = after.get("vllm:prefix_cache_queries_total", 0.0) - before.get("vllm:prefix_cache_queries_total", 0.0)
    dh = after.get("vllm:prefix_cache_hits_total", 0.0) - before.get("vllm:prefix_cache_hits_total", 0.0)
    if dq > 0 and res.cache_hit_frac != res.cache_hit_frac:  # NaN check
        res.cache_hit_frac = dh / dq

    if gauges:
        res.mean_running = sum(g[0] for g in gauges) / len(gauges)
        res.mean_waiting = sum(g[1] for g in gauges) / len(gauges)
        res.max_running = max(g[0] for g in gauges)
    return res


def latency_of(lvl: LevelResult) -> float:
    """End-to-end latency for one request at this level.

    Prefers the measured mean; falls back to Little's law (L = C / X) so that
    sweeps recorded before mean_total_s existed can still be rescored.
    """
    if lvl.mean_total_s == lvl.mean_total_s and lvl.mean_total_s > 0:
        return lvl.mean_total_s
    if lvl.req_s == lvl.req_s and lvl.req_s > 0:
        return lvl.concurrency / lvl.req_s
    return float("nan")


def find_knees(levels: list[LevelResult], ttft_factor: float, gain_threshold: float) -> dict:
    """Locate the knee, and say plainly when the sweep did not contain one.

    Primary metric is Kleinrock power, throughput / latency. The first version of
    this function looked for the batch size where TTFT "departs from flat", and
    that assumption does not survive contact with a real engine: TTFT on vLLM
    rises from the very first added request, exactly as queueing theory says it
    should, so "last flat level" collapses to c=1 and LOAD_REF gets set to a
    value that saturates load_norm for every worker at once -- the degenerate
    case config.py warns about.

    Power has no flat-region assumption. It rises while added concurrency buys
    more work than it costs in latency, peaks, and falls once it stops. The peak
    is the operating point LOAD_REF wants, and on a well-behaved engine the
    curve is cleanly unimodal.
    """
    usable = [l for l in levels if l.n_ok > 0 and l.ttft_p50 == l.ttft_p50]
    out: dict = {"power_knee": None, "ttft_knee": None,
                 "throughput_knee": None, "notes": []}
    if len(usable) < 2:
        out["notes"].append("not enough successful levels to locate a knee")
        return out

    # --- primary: peak power -------------------------------------------------
    powers: list[tuple[int, float]] = []
    for lvl in usable:
        lat = latency_of(lvl)
        if lat == lat and lat > 0 and lvl.req_s == lvl.req_s:
            powers.append((lvl.concurrency, lvl.req_s / lat))
    if powers:
        peak_c, peak_p = max(powers, key=lambda t: t[1])
        out["power_knee"] = peak_c
        out["power_curve"] = powers
        out["peak_power"] = peak_p
        if peak_c == powers[0][0]:
            out["notes"].append(
                "power peaks at the lowest level swept -- the engine was already "
                "past its knee at minimum concurrency; nothing to calibrate against")
        elif peak_c == powers[-1][0]:
            out["notes"].append(
                f"power still rising at the top of the range (c={peak_c}); "
                "extend --concurrency upward before trusting this knee")

    # --- diagnostic only: where TTFT crosses `ttft_factor` -------------------
    # Kept because "TTFT had already doubled by c=N" is a useful sentence, but
    # deliberately NOT the basis of the recommendation. On an engine with no
    # flat region this lands on the first level and means nothing.
    base = usable[0]
    out["ttft_baseline_s"] = base.ttft_p50
    last_flat = base
    first_degraded = None
    for lvl in usable[1:]:
        if lvl.ttft_p50 >= base.ttft_p50 * ttft_factor:
            first_degraded = lvl
            break
        last_flat = lvl
    if first_degraded is not None:
        out["ttft_knee"] = last_flat.concurrency
        out["ttft_knee_upper"] = first_degraded.concurrency
        out["ttft_at_knee_s"] = last_flat.ttft_p50
        out["ttft_at_upper_s"] = first_degraded.ttft_p50
        out["ttft_factor_x"] = first_degraded.ttft_p50 / base.ttft_p50
        if last_flat is base:
            out["notes"].append(
                f"TTFT was already {first_degraded.ttft_p50 / base.ttft_p50:.1f}x "
                f"baseline by c={first_degraded.concurrency}, the very first step -- "
                "no flat region exists, so the TTFT criterion is uninformative here "
                "and the power knee is what the recommendation uses")
    else:
        out["notes"].append(
            f"TTFT never reached {ttft_factor}x baseline within the swept range "
            f"(max c={usable[-1].concurrency})")

    # Throughput knee: last level that still bought a real gain over its
    # predecessor. Past this, concurrency converts directly into queueing.
    best = usable[0]
    saturated = False
    last_gain = float("nan")
    for prev, cur in zip(usable, usable[1:]):
        if prev.output_tok_s <= 0 or prev.output_tok_s != prev.output_tok_s:
            continue
        gain = (cur.output_tok_s - prev.output_tok_s) / prev.output_tok_s
        last_gain = gain
        if gain >= gain_threshold:
            best = cur
        else:
            saturated = True
            break
    out["throughput_knee"] = best.concurrency
    # Without this the top of the swept range gets reported as a knee even when
    # throughput was still climbing steeply -- "we ran out of range" silently
    # dressed up as "we found saturation".
    if not saturated:
        out["throughput_saturated"] = False
        out["notes"].append(
            f"throughput never saturated: the last step still gained {last_gain:+.0%} "
            f"(threshold {gain_threshold:.0%}), so c={best.concurrency} is the end of "
            "the swept range, not a knee")
    else:
        out["throughput_saturated"] = True
    out["peak_output_tok_s"] = max(
        (l.output_tok_s for l in usable if l.output_tok_s == l.output_tok_s), default=float("nan"))

    # Does concurrency actually equal what the router reads as load? If vLLM caps
    # the running batch below the offered concurrency, LOAD_REF must be set in
    # units of `running`, not in units of client concurrency.
    rec = out["power_knee"] or out["throughput_knee"]
    at_knee = next((l for l in usable if l.concurrency == rec), None)
    if at_knee and at_knee.mean_running == at_knee.mean_running:
        out["mean_running_at_knee"] = at_knee.mean_running
        if at_knee.mean_running < at_knee.concurrency * 0.6:
            out["notes"].append(
                f"engine ran only {at_knee.mean_running:.1f} of {at_knee.concurrency} offered "
                "requests concurrently -- set LOAD_REF from mean_running, not concurrency")
    return out


def show_table(levels: list[LevelResult], peak_c: int | None = None) -> None:
    print(f"{'conc':>5} {'ok':>5} {'TTFT p50':>9} {'p90':>8} {'TPOT':>8} "
          f"{'out tok/s':>10} {'req/s':>7} {'e2e lat':>8} {'power':>7} "
          f"{'running':>8} {'wait':>6} {'cache':>6}")
    print("-" * 100)
    for l in levels:
        lat = latency_of(l)
        power = l.req_s / lat if lat == lat and lat > 0 else float("nan")
        mark = " <-- peak" if peak_c is not None and l.concurrency == peak_c else ""
        print(f"{l.concurrency:>5} {l.n_ok:>5} {l.ttft_p50:>9.3f} {l.ttft_p90:>8.3f} "
              f"{l.tpot_p50_ms:>7.1f}m {l.output_tok_s:>10.1f} {l.req_s:>7.2f} "
              f"{lat:>8.2f} {power:>7.3f} {l.mean_running:>8.1f} "
              f"{l.mean_waiting:>6.1f} {l.cache_hit_frac:>5.0%}{mark}")
        if l.errors:
            print(f"      errors: {l.errors}")


def report(payload: dict) -> None:
    levels = [LevelResult(**{k: v for k, v in l.items()
                             if k in LevelResult.__dataclass_fields__})
              for l in payload["levels"]]
    knees = payload["knees"]
    print()
    show_table(levels, knees.get("power_knee"))
    print()
    print(f"prompt mode      : {payload['args']['prompt_mode']} "
          f"(~{payload['args']['prompt_tokens']} tokens target, "
          f"{levels[0].mean_prompt_tokens:.0f} measured)")
    print(f"max tokens       : {payload['args']['max_tokens']}")
    if knees.get("power_knee"):
        print(f"power knee       : c={knees['power_knee']}  "
              f"(peak {knees['peak_power']:.3f} req/s per second of latency)   <- primary")
    sat = "saturates" if knees.get("throughput_saturated") else "still climbing at"
    print(f"throughput       : {sat} c={knees['throughput_knee']} "
          f"({knees.get('peak_output_tok_s', float('nan')):.1f} out tok/s)")
    if knees["ttft_knee"]:
        print(f"TTFT (diagnostic): {knees['ttft_baseline_s']:.3f}s unloaded -> "
              f"{knees['ttft_factor_x']:.1f}x by c={knees['ttft_knee_upper']}")
    for n in knees["notes"]:
        print(f"NOTE: {n}")

    # A `unique` sweep that hit the prefix cache did not measure what it claims
    # to. Usually means --nonce was pinned, or the engine was already holding
    # these prompts from an earlier run.
    if payload["args"]["prompt_mode"] == "unique":
        hits = [l.cache_hit_frac for l in levels if l.cache_hit_frac == l.cache_hit_frac]
        if hits and max(hits) > 0.2:
            print(f"WARNING: prompt_mode=unique but the engine reported up to "
                  f"{max(hits):.0%} cached tokens.")
            print("  Prefill was not cold, so this curve understates real prefill cost")
            print("  and puts the knee too high. Restart vLLM (or drop --nonce) and re-run.")

    rec = knees.get("power_knee") or knees["throughput_knee"]
    if knees.get("mean_running_at_knee") is not None and \
            knees["mean_running_at_knee"] < (rec or 0) * 0.6:
        rec = round(knees["mean_running_at_knee"])
    print()
    print("  -> reçete: LOAD_REF'i power (throughput/latency) tepesine ayarla.")
    print("     Orada load_norm = 0.5 olur; yük terimi motorun eşzamanlılığı hâlâ")
    print("     kâra çevirdiği bölge ile kuyruğa çevirdiği bölgeyi tam ayırır.")
    print()
    print(f"     ROUTER_LOAD_REF={rec}")
    print()
    if knees.get("power_knee") and knees.get("throughput_knee"):
        if knees["power_knee"] == knees["throughput_knee"]:
            print(f"  YORUM: power tepesi ve throughput doyumu aynı yerde (c={rec}).")
            print("         Dirsek net, LOAD_REF bu değere sabitlenebilir.")
        else:
            print(f"  YORUM: power c={knees['power_knee']}'de tepe yapıyor ama toplam")
            print(f"         throughput c={knees['throughput_knee']}'e kadar artmaya devam ediyor.")
            print("         Aradaki bölge motorun toplam iş çıkardığı ama tek isteğin")
            print("         orantısız yavaşladığı bölge. Power tepesini seçmek gecikme-")
            print("         öncelikli bir tercih; raporda bunu bir cümleyle gerekçelendir.")


async def run(args) -> None:
    limits = httpx.Limits(max_connections=max(args.concurrency) + 8,
                          max_keepalive_connections=max(args.concurrency) + 8)
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
    levels: list[LevelResult] = []

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        try:
            r = await client.get(f"{args.worker}/v1/models", timeout=10.0)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"worker unreachable at {args.worker}: {exc}")

        # Global burn-in, discarded. Per-level warmup covers each batch shape,
        # but it cannot cover what only the FIRST level pays: Triton kernel JIT
        # on a freshly started engine, and a GPU still at idle clocks. Those land
        # entirely on the lowest concurrency level, which is also the baseline
        # every other level is judged against -- so leaving them in tilts the
        # whole curve. Matters most when comparing two sweeps across a restart,
        # which is the only way to change an engine flag.
        if args.warmup_requests:
            print(f"burn-in {args.warmup_requests} requests ... ", end="", flush=True)
            rng = random.Random(args.seed ^ 0x5EED)
            t_burn = time.perf_counter()
            for i in range(args.warmup_requests):
                await one_request(client, args,
                                  make_prompt(rng, args.prompt_tokens, f"{args.nonce}-burn-{i}"))
            print(f"{time.perf_counter() - t_burn:.1f}s")
            await asyncio.sleep(args.cooldown)

        for c in args.concurrency:
            print(f"level c={c} ... ", end="", flush=True)
            lvl = await run_level(client, args, c)
            levels.append(lvl)
            print(f"ttft p50={lvl.ttft_p50:.3f}s  out={lvl.output_tok_s:.1f} tok/s"
                  + (f"  ({lvl.n_err} failed)" if lvl.n_err else ""))
            # Let the engine drain before the next level, so a level never
            # measures the tail of its predecessor.
            await asyncio.sleep(args.cooldown)

    payload = {
        "label": args.label,
        "worker": args.worker,
        "args": vars(args),
        "levels": [l.__dict__ for l in levels],
        "knees": find_knees(levels, args.ttft_factor, args.gain_threshold),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report(payload)
    print(f"\nwritten: {args.out}")


def compare(a_path: str, b_path: str) -> None:
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    la = {l["concurrency"]: l for l in a["levels"]}
    lb = {l["concurrency"]: l for l in b["levels"]}
    name_a = a.get("label") or a_path
    name_b = b.get("label") or b_path

    print(f"A = {name_a}")
    print(f"B = {name_b}")
    print()
    print(f"{'conc':>5} {'TTFT A':>8} {'TTFT B':>8} {'delta':>8} "
          f"{'tok/s A':>9} {'tok/s B':>9} {'delta':>8}")
    print("-" * 62)
    for c in sorted(set(la) & set(lb)):
        x, y = la[c], lb[c]
        dt = (y["ttft_p50"] - x["ttft_p50"]) / x["ttft_p50"] if x["ttft_p50"] else float("nan")
        dv = (y["output_tok_s"] - x["output_tok_s"]) / x["output_tok_s"] if x["output_tok_s"] else float("nan")
        print(f"{c:>5} {x['ttft_p50']:>8.3f} {y['ttft_p50']:>8.3f} {dt:>+7.1%} "
              f"{x['output_tok_s']:>9.1f} {y['output_tok_s']:>9.1f} {dv:>+7.1%}")
    # .get() because sweeps recorded before the power metric existed have no
    # such key; --rescore backfills them.
    ka, kb = a["knees"].get("power_knee"), b["knees"].get("power_knee")
    print()
    print(f"knee A: power c={ka}, throughput c={a['knees']['throughput_knee']}")
    print(f"knee B: power c={kb}, throughput c={b['knees']['throughput_knee']}")
    if ka is None or kb is None:
        print()
        print("  bir tarafta power dirseği yok -- once: sweep_batch.py --rescore <dosya>")
        return
    print()
    if ka == kb:
        print("  YORUM: dirsek yer değiştirmedi -- LOAD_REF bu konfigürasyona duyarsız,")
        print("         yani caveat'ı kaldırabilirsin.")
    else:
        print("  YORUM: dirsek kaydı. LOAD_REF motorun çalıştırma moduna bağlı;")
        print("         raporda hangi modda ölçüldüğünü yaz.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="compare two finished sweeps instead of running one")
    p.add_argument("--rescore", metavar="SWEEP.json",
                   help="recompute the knees from a finished sweep and rewrite it, "
                        "without spending GPU time on another run")
    p.add_argument("--worker", default="http://localhost:8000",
                   help="vLLM base URL; the sweep goes straight to the engine, not the router")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out", default="sweep.json")
    p.add_argument("--label", default="", help="name for this sweep in --compare output")
    p.add_argument("--concurrency", default="1,2,4,8,16,32,64",
                   help="comma-separated levels to sweep")
    p.add_argument("--prompt-mode", choices=["unique", "shared"], default="unique",
                   help="unique = every request pays full prefill (default); "
                        "shared = one prompt for all, measures the cached path")
    p.add_argument("--prompt-tokens", type=int, default=1200,
                   help="approximate prompt size; match your RAG prompts")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--rounds", type=int, default=3,
                   help="requests per level = max(min_requests, rounds * concurrency)")
    p.add_argument("--min-requests", type=int, default=16)
    p.add_argument("--cooldown", type=float, default=5.0, help="idle seconds between levels")
    p.add_argument("--warmup-requests", type=int, default=8,
                   help="discarded requests fired before the sweep starts, to absorb "
                        "kernel JIT and idle GPU clocks on a freshly restarted engine. "
                        "Keep identical across sweeps you intend to compare")
    p.add_argument("--sample-interval", type=float, default=0.5,
                   help="how often to scrape running/waiting during a level")
    p.add_argument("--ttft-factor", type=float, default=1.5,
                   help="a level counts as degraded once p50 TTFT reaches this "
                        "multiple of the unloaded p50; the knee is the last level below it")
    p.add_argument("--gain-threshold", type=float, default=0.15,
                   help="throughput knee = last level that gained at least this fraction")
    p.add_argument("--seed", type=int, default=401)
    p.add_argument("--nonce", default="",
                   help="per-run prompt salt; defaults to a timestamp. Only pin it "
                        "when you deliberately want to reuse a previous run's prompts")
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.rescore:
        path = Path(args.rescore)
        payload = json.loads(path.read_text(encoding="utf-8"))
        levels = [LevelResult(**{k: v for k, v in l.items()
                                 if k in LevelResult.__dataclass_fields__})
                  for l in payload["levels"]]
        payload["knees"] = find_knees(levels, args.ttft_factor, args.gain_threshold)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report(payload)
        print(f"\nrescored: {path}")
        return

    args.nonce = args.nonce or f"run{int(time.time())}"
    args.worker = args.worker.rstrip("/")
    args.concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
