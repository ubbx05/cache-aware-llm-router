"""Standalone smoke test: exercises the routing logic with fake workers.

No vLLM, no network -- WorkerState objects are built by hand and fed straight
into the strategies. Good for sanity-checking scoring/guard-band behaviour
before pointing the router at real infrastructure.
"""
from prefix_tracker import PrefixTracker, ApproxTokenizer, block_hashes
from strategies import build_strategy, RequestContext
from worker_metrics import WorkerState, Snapshot

tok = ApproxTokenizer()


def make_snapshot(w1_running, w2_running, kv1=0.2, kv2=0.2):
    w1 = WorkerState(name="w1", url="fake", healthy=True, num_requests_running=w1_running, kv_cache_usage_perc=kv1)
    w2 = WorkerState(name="w2", url="fake", healthy=True, num_requests_running=w2_running, kv_cache_usage_perc=kv2)
    return Snapshot(states={"w1": w1, "w2": w2})


def ctx_for(text):
    hashes = block_hashes(tok.encode(text))
    return RequestContext(prompt_text=text, block_hashes=hashes), hashes


tracker = PrefixTracker(["w1", "w2"])
strategy = build_strategy("cache_aware", tracker)

prompt_a = "sistem: sen bir asistansin. " * 20 + "soru: istanbulun tarihi nedir?"
prompt_a_devam = prompt_a + " ek soru: ozetle."

print("== Test 1: bos cache, esit yuk -> herhangi biri secilebilir ==")
ctx, hashes = ctx_for(prompt_a)
snap = make_snapshot(w1_running=0, w2_running=0)
decision = strategy.select(ctx, snap)
print(f"secilen: {decision.worker.name}, sebep: {decision.reason}")
tracker.record(decision.worker.name, hashes)

print("\n== Test 2: ayni prefix'in devami, w1'de cache var, yukler esit -> w1 kazanmali ==")
ctx2, hashes2 = ctx_for(prompt_a_devam)
snap2 = make_snapshot(w1_running=0, w2_running=0)
decision2 = strategy.select(ctx2, snap2)
print(f"secilen: {decision2.worker.name}, sebep: {decision2.reason}, cache_gain={decision2.cache_gain:.2f}")
assert decision2.worker.name == "w1", "beklenen: cache'i tutan worker kazanmali"

print("\n== Test 3: ayni prefix ama w1 asiri yuklu -> guard band devreye girip w2'ye dusmeli ==")
snap3 = make_snapshot(w1_running=50, w2_running=0)
decision3 = strategy.select(ctx2, snap3)
print(f"secilen: {decision3.worker.name}, sebep: {decision3.reason}")
assert decision3.worker.name == "w2", "beklenen: guard band w1'i asiri yuklu bulup w2'ye dusmeli"

print("\nTum testler basarili.")
