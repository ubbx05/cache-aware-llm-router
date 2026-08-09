"""Standalone smoke test: per_worker_tree'nin overlap-adaptive alpha/beta'si.

smoke_test.py/smoke_test_semantic.py ile ayni desen -- vLLM/network yok,
WorkerState/Snapshot elle kuruluyor.

Iki seyi kanitliyor:
1. ROUTER_OVERLAP_ADAPTIVE=False (varsayilan) iken davranis eskisiyle
   BIREBIR ayni: config.ALPHA/BETA sabit kullanilir, _prev_chunk_ids hic
   dokunulmaz, jaccard() hic cagrilmaz.
2. ROUTER_OVERLAP_ADAPTIVE=True iken, dusuk vs yuksek ardisik-cift Jaccard
   overlap'i GERCEKTEN farkli bir worker sectirebiliyor -- cache_gain=0
   oldugu bir senaryoda alpha farki gorunmez oldugu icin, burada w1'e
   BILEREK hem cache avantaji HEM yuk verilip alpha/beta oraninin skoru
   gercekten degistirdigi gosteriliyor.
"""
import os

os.environ.setdefault("W2_ENABLED", "true")  # config.WORKERS import-time okunuyor

import config
from prefix_tracker import PrefixTracker
from strategies import build_strategy


def make_snapshot():
    from worker_metrics import WorkerState, Snapshot
    # w1: hem cache avantaji hem yuk tasiyacak (asagida seed edilecek);
    # w2: bos cache, bos yuk -- referans nokta.
    w1 = WorkerState(name="w1", url="fake", healthy=True, num_requests_running=8)
    w2 = WorkerState(name="w2", url="fake", healthy=True, num_requests_running=0)
    return Snapshot(states={"w1": w1, "w2": w2})


print("== Test 1: ROUTER_OVERLAP_ADAPTIVE=False -- eski davranisla BIREBIR ayni ==")
config.OVERLAP_ADAPTIVE_ENABLED = False
strat_off = build_strategy("per_worker_tree", PrefixTracker(["w1", "w2"]))
strat_off.decide_order(["A", "B", "C"], make_snapshot())
strat_off.decide_order(["A", "B", "C"], make_snapshot())  # ayni chunk'lar -- flag acik olsa overlap=1.0 olurdu
assert strat_off._prev_chunk_ids is None, (
    "flag KAPALIYKEN _prev_chunk_ids guncellenmemeli -- guncellendiyse "
    "overlap-adaptive kod flag'e bakmadan calisiyor demektir"
)
print("   OK -- _prev_chunk_ids hic dokunulmadi (jaccard() hic cagrilmadi)")

print("\n== Test 2: ROUTER_OVERLAP_ADAPTIVE=True -- overlap'e gore w1/w2 arasinda gercekten farkli secim ==")
config.OVERLAP_ADAPTIVE_ENABLED = True
config.OVERLAP_THRESHOLD = 0.3
config.LOW_OVERLAP_ALPHA = 0.2
config.HIGH_OVERLAP_ALPHA = 0.8

# Iki AYRI strateji ornegi kullaniliyor (tek ornekte iki ardisik decide_order
# cagirmak, ilk cagrinin dispatch-time bookkeeping'inin -- on_request_finished
# -- ikinci cagrinin cache durumunu da degistirmesine yol aciyordu, iki
# senaryoyu birbirine karistiriyordu). Her ikisinde de w1'in cache/yuk durumu
# BIREBIR ayni kurulur; TEK degisken _prev_chunk_ids'in (dolayisiyla
# overlap'in) dusuk/yuksek olmasi.
def seeded_strategy():
    s = build_strategy("per_worker_tree", PrefixTracker(["w1", "w2"]))
    s._router.on_request_finished("w1", ["A", "B", "C"])  # SADECE w1'in cache'i dolu
    return s

strat_low = seeded_strategy()
strat_low._prev_chunk_ids = {"Q", "R", "S"}  # alakasiz -> jaccard=0.0 -> LOW_OVERLAP_ALPHA
d_low = strat_low.decide_order(["A", "B", "C"], make_snapshot())
print(f"   dusuk-overlap (onceki={{Q,R,S}}) -> secilen: {d_low.worker_name}  skorlar: {d_low.scores}")
assert d_low.worker_name == "w2", (
    f"dusuk overlap'te (alpha=0.2, beta=0.8) yuk agir basip w2 secilmeliydi -- w1'in yuku (8) "
    f"cache avantajini gecmeli, gelen: {d_low.worker_name}"
)

strat_high = seeded_strategy()
strat_high._prev_chunk_ids = {"A", "B", "C"}  # ayni set -> jaccard=1.0 -> HIGH_OVERLAP_ALPHA
d_high = strat_high.decide_order(["A", "B", "C"], make_snapshot())
print(f"   yuksek-overlap (onceki={{A,B,C}}) -> secilen: {d_high.worker_name}  skorlar: {d_high.scores}")
assert d_high.worker_name == "w1", (
    f"yuksek overlap'te (alpha=0.8, beta=0.2) cache avantaji agir basip w1 secilmeliydi, "
    f"gelen: {d_high.worker_name}"
)
assert d_low.worker_name != d_high.worker_name, (
    "dusuk ve yuksek overlap AYNI worker'i sectiyse bu senaryo bir fark kanitlamiyor"
)
print("   OK -- overlap dusukken w2 (yuk), yuksekken w1 (cache) kazandi -- adaptif agirlik gercekten calisiyor")

print("\nTum overlap-adaptive testleri basarili.")
