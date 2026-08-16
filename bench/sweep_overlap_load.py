"""3x3 deney matrisi: overlap (dusuk/orta/yuksek) x yuk (dusuk/orta/yuksek).

KATKI_OZETI.md Bolum 7b'deki 3x3 tabloyu doldurmak icin. Her hucrede TUM
STRATEGY_CONFIGS kosuluyor (varsayilan 7 strateji: round_robin,
least_loaded, cacheweaver_dualmap, per_worker_tree'nin uc overlap-adaptive
modu -- off/threshold/ewma_cusum, bkz. config.OVERLAP_ADAPTIVE_MODE -- ve
semantic_per_worker_tree) -- 7x9=63 kosum, coksa --strategies ile
filtrelenebilir (ornek: --strategies round_robin,per_worker_tree).

Neden overlap ekseni icin zipf-s (gen_trace.py'nin var olan parametresi)
kullanildi, yeni bir "session karisma" parametresi eklenmedi: zaten olculdu
(bkz. bench/overlap_measurement.py ile yapilan onceki deney) -- zipf_s
0.0 -> 2.0 arasinda session-adjacent VE global-adjacent Jaccard overlap'i
guclu ve monoton sekilde degistiriyor (global mean 0.023 -> 0.308, 13 kat).
Ayri bir parametre icat etmek yerine zaten dogrulanmis bu kolu kullanmak
daha savunulabilir.

Neden yuk ekseni icin replay.py'nin --speedup'i kullanildi: gun-raporlarinda
zaten --speedup 2/8/20 ile calisilmis, trace'in zaman eksenini sikistirarak
ayni istek sayisini daha kisa surede/dahayogun gonderiyor -- ekstra kod
gerektirmeyen, var olan bir yuk kolu.

Router her strateji icin AYRI bir surec olarak baslatiliyor (uvicorn
main:app), cunku config.STRATEGY/config.OVERLAP_ADAPTIVE_MODE main.py
baslarken BIR KERE okunuyor (bkz. main.py _startup, strategies.build_strategy
cagrisi) -- calisirken degistirilemez. Bu yuzden strateji basina restart
sarti var, sweep_batch.py'nin tek-worker konsantrasyon taramasindan farkli
olarak (o script router'a hic dokunmuyor, dogrudan vLLM'e gidiyor).

BU SCRIPT'I CALISTIRMAK GPU/vLLM GEREKTIRIR. --dry-run ile (GPU'suz) sadece
secili hucrelerin planini yazdirip hicbir subprocess baslatmadan
dogrulayabilirsin -- syntax/import/orkestrasyon mantigini kontrol etmek icin
yeterli.

Usage:
    # plan/orkestrasyon mantigini GPU olmadan dogrula (tum 63 hucre)
    python sweep_overlap_load.py --dry-run

    # sadece bazi stratejileri dogrula/kos (kademeli calistirma icin)
    python sweep_overlap_load.py --dry-run --strategies round_robin,per_worker_tree

    # gercek kosum (GPU + calisan vLLM worker'lari gerektirir)
    python sweep_overlap_load.py --corpus ./corpus \
        --worker http://100.89.101.52:8000 --worker http://100.64.0.2:8000 \
        --out sweep_results.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent

# (etiket, gen_trace.py --zipf-s degeri). Dusukten yuksege: az skew -> cok
# skew, bkz. modul docstring'i -- daha once bench/overlap_measurement.py ile
# bu araligin overlap'i guclu sekilde degistirdigi olculdu.
OVERLAP_LEVELS: list[tuple[str, float]] = [
    ("low", 0.3),
    ("med", 1.0),
    ("high", 2.0),
]

# (etiket, replay.py --speedup degeri). Gun-raporlarinda kullanilan degerler.
LOAD_LEVELS: list[tuple[str, float]] = [
    ("low", 2.0),
    ("med", 8.0),
    ("high", 20.0),
]


@dataclass
class StrategyConfig:
    label: str                     # sweep_results.csv'de gorunen isim
    env: dict[str, str]            # main.py'yi baslatirken set edilecek env
    order: str                     # replay.py --order (canonical | per_worker_tree)


# 7 strateji: 3 cache-blind/tek-mekanizma baseline + per_worker_tree'nin 3
# overlap-adaptive modu (off/threshold/ewma_cusum -- bkz.
# strategies.PerWorkerTreeStrategy, config.OVERLAP_ADAPTIVE_MODE) +
# semantic_per_worker_tree. semantic_per_worker_tree de decide_order()
# uyguladigi icin (per_worker_tree ile ayni iki-asamali akis) order=
# "per_worker_tree" kullaniyor. replay.py iki-asamali istekte soru metnini
# query_text olarak yollar; semantic top-k gercekten aday eleyebilsin diye iki
# worker'li bir deneyde ROUTER_SEMANTIC_TOP_K=1 kullanilmalidir.
STRATEGY_CONFIGS: list[StrategyConfig] = [
    StrategyConfig("round_robin", {"ROUTER_STRATEGY": "round_robin"}, "canonical"),
    StrategyConfig("least_loaded", {"ROUTER_STRATEGY": "least_loaded"}, "canonical"),
    StrategyConfig("cacheweaver_dualmap", {"ROUTER_STRATEGY": "cacheweaver_dualmap"}, "canonical"),
    StrategyConfig(
        "per_worker_tree",
        {"ROUTER_STRATEGY": "per_worker_tree", "ROUTER_OVERLAP_ADAPTIVE_MODE": "off"},
        "per_worker_tree",
    ),
    StrategyConfig(
        "per_worker_tree+threshold",
        {"ROUTER_STRATEGY": "per_worker_tree", "ROUTER_OVERLAP_ADAPTIVE_MODE": "threshold"},
        "per_worker_tree",
    ),
    StrategyConfig(
        "per_worker_tree+ewma_cusum",
        {"ROUTER_STRATEGY": "per_worker_tree", "ROUTER_OVERLAP_ADAPTIVE_MODE": "ewma_cusum"},
        "per_worker_tree",
    ),
    StrategyConfig(
        "semantic_per_worker_tree",
        {"ROUTER_STRATEGY": "semantic_per_worker_tree"},
        "per_worker_tree",
    ),
]


@dataclass
class CellResult:
    overlap_level: str
    load_level: str
    strategy: str
    speedup: float
    zipf_s: float
    n_ok: int = 0
    n_failed: int = 0
    ttft_p50_s: float = float("nan")
    ttft_p95_s: float = float("nan")
    ttft_p99_s: float = float("nan")
    e2e_ttft_p50_s: float = float("nan")
    e2e_ttft_p99_s: float = float("nan")
    throughput_req_s: float = float("nan")
    cache_hit_rate: float = float("nan")
    cache_metric: str = "unavailable"
    cached_tokens_total: int = 0
    prompt_tokens_total: int = 0
    load_cv: float = float("nan")
    error: str = ""


CSV_FIELDS = [f for f in CellResult.__dataclass_fields__]


def pct(values: list[float], p: int) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    # The default "exclusive" method extrapolates outside the observed range
    # for small samples (for example a two-value p99 can exceed max(values)).
    # Inclusive quantiles remain bounded and match replay.py's summary.
    return statistics.quantiles(sorted(values), n=100, method="inclusive")[p - 1]


def gen_trace(args, zipf_s: float, out_path: Path) -> None:
    """gen_trace.py'yi subprocess olarak cagirir -- trace uretme mantigini
    burada yeniden yazmiyoruz."""
    cmd = [
        sys.executable, str(BENCH_DIR / "gen_trace.py"),
        "--corpus", args.corpus,
        "--out", str(out_path),
        "--n", str(args.n),
        "--zipf-s", str(zipf_s),
        "--seed", str(args.seed),
    ]
    subprocess.run(cmd, cwd=BENCH_DIR, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def start_router(args, env_overrides: dict[str, str], port: int,
                 log_path: Path) -> subprocess.Popen:
    """main.py'yi (uvicorn) verilen env override'lariyla ayri bir surec
    olarak baslatir. STRATEGY/OVERLAP_ADAPTIVE main.py'nin startup'inda BIR
    KERE okunuyor (bkz. modul docstring'i), o yuzden her strateji icin
    restart sarti."""
    env = os.environ.copy()
    env.update(env_overrides)
    env["ROUTER_PORT"] = str(port)
    if args.worker:
        env["W1_URL"] = args.worker[0]
        env["W1_ENABLED"] = "true"
        if len(args.worker) > 1:
            env["W2_URL"] = args.worker[1]
            env["W2_ENABLED"] = "true"
    cmd = [sys.executable, "-m", "uvicorn", "main:app",
          "--host", "127.0.0.1", "--port", str(port)]

    # Do not leave uvicorn attached to an unread PIPE.  Its access log can fill
    # a ~64 KiB pipe during a long replay and block the router process itself.
    # Keep the file handle on the Popen object so stop_router can close it.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    try:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env,
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise
    proc._router_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


async def wait_healthy(base_url: str, timeout_s: float) -> None:
    """/health 200 donene kadar (en az bir worker healthy oldugunda) bekler.
    Router surecinin ayakta olmasi yetmez -- poller'in gercekten bir vLLM
    worker'ini scrape edebilmesi gerekir, yoksa strategy.select() her istekte
    NoHealthyWorker firlatir ve olcum anlamsizlasir."""
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as client:
        last_err = "unknown"
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url}/health", timeout=3.0)
                if r.status_code == 200:
                    return
                last_err = f"http {r.status_code}: {r.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)
    raise TimeoutError(f"router at {base_url} never became healthy: {last_err}")


def stop_router(proc: subprocess.Popen, timeout_s: float = 10.0) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout_s)
    finally:
        log_handle = getattr(proc, "_router_log_handle", None)
        if log_handle is not None and not log_handle.closed:
            log_handle.close()


def run_replay(args, trace_path: Path, router_url: str, order: str,
              speedup: float, out_path: Path) -> None:
    cmd = [
        sys.executable, str(BENCH_DIR / "replay.py"),
        "--corpus", args.corpus,
        "--trace", str(trace_path),
        "--out", str(out_path),
        "--router", router_url,
        "--order", order,
        "--top-k", str(args.top_k),
        "--speedup", str(speedup),
        "--model", args.model,
    ]
    for w in args.worker:
        cmd += ["--worker", w]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    subprocess.run(cmd, cwd=BENCH_DIR, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def score_results(results_path: Path,
                  expected_workers: list[str] | None = None) -> CellResult:
    """replay.py'nin yazdigi results.jsonl'i okuyup hucre metriklerini
    cikarir -- replay.py'nin kendi summarise() ciktisini yeniden ayristirmak
    yerine ayni ham satirlari (Result.__dict__) dogrudan kullaniyoruz."""
    rows = [json.loads(l) for l in results_path.open(encoding="utf-8")]
    ok = [r for r in rows if r.get("error") is None and r.get("ttft_s") is not None]

    res = CellResult(overlap_level="", load_level="", strategy="", speedup=0.0, zipf_s=0.0)
    res.n_ok = len(ok)
    res.n_failed = len(rows) - len(ok)
    if not ok:
        return res

    ttfts = [r["ttft_s"] for r in ok]
    res.ttft_p50_s = pct(ttfts, 50)
    res.ttft_p95_s = pct(ttfts, 95)
    res.ttft_p99_s = pct(ttfts, 99)

    e2e_ttfts = [
        float(r["e2e_ttft_s"])
        for r in ok
        if r.get("e2e_ttft_s") is not None
    ]
    if e2e_ttfts:
        res.e2e_ttft_p50_s = pct(e2e_ttfts, 50)
        res.e2e_ttft_p99_s = pct(e2e_ttfts, 99)

    # Throughput: istek basina toplam sure / concurrency yerine, gozlenen
    # araligin (ilk gonderim - son bitis) uzerinden -- sweep_batch.py'nin
    # "span" yaklasimiyla ayni ruhta, burada wall-clock'un kendisi yeterli
    # cunku replay.py istekleri kendi zamanlamasina gore ateşliyor.
    starts = [r["sent_s"] for r in ok]
    ends = [r["sent_s"] + r["total_s"] for r in ok if r.get("total_s") is not None]
    if ends:
        span = max(ends) - min(starts)
        res.throughput_req_s = len(ok) / span if span > 0 else float("nan")

    # Primary metric: token-weighted engine truth.  This is invariant to the
    # distribution of prompt lengths across arms.  Old JSONL artifacts may
    # have only actual_frac, so retain a deliberately labelled macro fallback
    # rather than silently changing the meaning of cache_hit_rate.
    token_pairs = [
        (int(r["actual_cached_tokens"]), int(r["prompt_tokens"]))
        for r in ok
        if r.get("actual_cached_tokens") is not None
        and r.get("prompt_tokens") is not None
        and int(r["prompt_tokens"]) > 0
    ]
    if token_pairs:
        res.cached_tokens_total = sum(cached for cached, _ in token_pairs)
        res.prompt_tokens_total = sum(prompt for _, prompt in token_pairs)
        res.cache_hit_rate = res.cached_tokens_total / res.prompt_tokens_total
        res.cache_metric = "aggregate_cached_tokens/prompt_tokens"
    else:
        fracs = [
            float(r["actual_frac"])
            for r in ok
            if r.get("actual_frac") is not None
        ]
        if fracs:
            res.cache_hit_rate = statistics.fmean(fracs)
            res.cache_metric = "macro_mean_actual_frac_fallback"

    worker_counts: dict[str, int] = {
        worker: 0 for worker in (expected_workers or [])
    }
    for r in ok:
        w = r.get("worker") or "?"
        worker_counts[w] = worker_counts.get(w, 0) + 1
    if worker_counts:
        counts = list(worker_counts.values())
        mean = statistics.fmean(counts)
        res.load_cv = (statistics.pstdev(counts) / mean) if mean > 0 else float("nan")

    return res


def select_strategies(args) -> list[StrategyConfig]:
    """--strategies verilmemisse (varsayilan) TUMU -- 7 strateji x 9 hucre =
    63 kosu, cok sayida. --strategies round_robin,per_worker_tree gibi
    virgulle ayrilmis bir liste ile kademeli calistirilabilir."""
    if not args.strategies:
        return list(STRATEGY_CONFIGS)
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    by_label = {sc.label: sc for sc in STRATEGY_CONFIGS}
    unknown = [w for w in wanted if w not in by_label]
    if unknown:
        raise SystemExit(
            f"bilinmeyen strateji(ler): {unknown} -- gecerli etiketler: "
            f"{sorted(by_label)}"
        )
    return [by_label[w] for w in wanted]


def build_plan(
    args, strategies: list[StrategyConfig]
) -> list[tuple[str, float, str, float, StrategyConfig]]:
    """(overlap_label, zipf_s, load_label, speedup, strategy_config) -- tum
    hucrelerin listesi (--strategies ile filtrelenmis olabilir). --dry-run
    bu fonksiyonu cagirip subprocess hic baslatmadan yazdirir."""
    plan = []
    for overlap_label, zipf_s in OVERLAP_LEVELS:
        for load_label, speedup in LOAD_LEVELS:
            for sc in strategies:
                plan.append((overlap_label, zipf_s, load_label, speedup, sc))
    return plan


def run(args) -> None:
    strategies = select_strategies(args)
    plan = build_plan(args, strategies)

    if args.dry_run:
        print(f"{'overlap':>8} {'zipf_s':>7} {'load':>6} {'speedup':>8}  strategy")
        print("-" * 60)
        for overlap_label, zipf_s, load_label, speedup, sc in plan:
            print(f"{overlap_label:>8} {zipf_s:>7.1f} {load_label:>6} {speedup:>8.1f}  {sc.label}")
        print(f"\n{len(plan)} hucre (dry-run -- hicbir subprocess baslatilmadi)")
        return

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Trace'ler overlap seviyesi basina BIR KEZ uretilir, 3 yuk seviyesi x 5
    # strateji arasinda yeniden kullanilir -- 45 degil 3 gen_trace.py cagrisi.
    trace_paths: dict[str, Path] = {}
    for overlap_label, zipf_s in OVERLAP_LEVELS:
        trace_path = work_dir / f"trace_{overlap_label}.jsonl"
        print(f"trace uretiliyor: overlap={overlap_label} (zipf_s={zipf_s}) -> {trace_path}")
        gen_trace(args, zipf_s, trace_path)
        trace_paths[overlap_label] = trace_path

    results: list[CellResult] = []
    port = args.port
    for overlap_label, zipf_s in OVERLAP_LEVELS:
        for sc in strategies:
            print(f"\n== router baslatiliyor: strategy={sc.label} port={port} ==")
            log_path = work_dir / f"router_{port}_{overlap_label}_{sc.label}.log"
            proc = start_router(args, sc.env, port, log_path)
            try:
                asyncio.run(wait_healthy(f"http://127.0.0.1:{port}", args.startup_timeout))
            except TimeoutError as exc:
                stop_router(proc)
                for load_label, speedup in LOAD_LEVELS:
                    results.append(CellResult(
                        overlap_level=overlap_label, load_level=load_label,
                        strategy=sc.label, speedup=speedup, zipf_s=zipf_s,
                        error=str(exc),
                    ))
                continue

            for load_label, speedup in LOAD_LEVELS:
                print(f"  kosuluyor: overlap={overlap_label} load={load_label} "
                      f"(speedup={speedup}) strategy={sc.label}")
                out_path = work_dir / f"results_{overlap_label}_{load_label}_{sc.label}.jsonl"
                try:
                    run_replay(args, trace_paths[overlap_label],
                              f"http://127.0.0.1:{port}", sc.order, speedup, out_path)
                    expected_workers = [f"w{i + 1}" for i in range(len(args.worker))]
                    res = score_results(out_path, expected_workers)
                except subprocess.CalledProcessError as exc:
                    res = CellResult(overlap_level="", load_level="", strategy="",
                                     speedup=0.0, zipf_s=0.0, error=str(exc))
                res.overlap_level = overlap_label
                res.load_level = load_label
                res.strategy = sc.label
                res.speedup = speedup
                res.zipf_s = zipf_s
                results.append(res)

            stop_router(proc)
            port += 1  # cooldown'suz hemen restart -- eski port TIME_WAIT'te olabilir

    out_csv = Path(args.out)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in CSV_FIELDS})
    print(f"\nyazildi: {out_csv}  ({len(results)} satir)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="./corpus")
    p.add_argument("--worker", action="append", default=[],
                   help="vLLM worker base URL; repeatable (ilk ikisi W1_URL/W2_URL olur)")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--n", type=int, default=1500, help="trace basina istek sayisi")
    p.add_argument("--seed", type=int, default=401)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="replay.py --limit (hizli deneme icin)")
    p.add_argument("--port", type=int, default=8099,
                   help="router icin baslangic portu; her strateji restart'inda 1 artar")
    p.add_argument("--startup-timeout", type=float, default=60.0,
                   help="router /health 200 donene kadar beklenecek maksimum sure")
    p.add_argument("--work-dir", default="./sweep_overlap_load_work",
                   help="ara trace/results dosyalarinin yazilacagi klasor")
    p.add_argument("--out", default="sweep_results.csv")
    p.add_argument("--strategies", default=None,
                   help="virgulle ayrilmis strateji etiketi listesi (bkz. STRATEGY_CONFIGS); "
                        f"varsayilan: TUMU ({len(STRATEGY_CONFIGS)} strateji x 9 hucre = "
                        f"{len(STRATEGY_CONFIGS) * 9} kosu -- coksa kademeli calistirmak icin "
                        "bu flag'i kullan, ornek: --strategies round_robin,per_worker_tree")
    p.add_argument("--dry-run", action="store_true",
                   help="hicbir subprocess baslatma -- sadece secili hucrelerin planini yazdir "
                        "(GPU/vLLM gerektirmeyen dogrulama)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
