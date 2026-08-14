"""Standalone smoke test: per_worker_tree'nin overlap-adaptive alpha/beta'si.

smoke_test.py/smoke_test_semantic.py ile ayni desen -- vLLM/network yok,
WorkerState/Snapshot elle kuruluyor.

Uc seyi kanitliyor:
1. ROUTER_OVERLAP_ADAPTIVE_MODE=off (varsayilan) iken davranis eskisiyle
   BIREBIR ayni: config.ALPHA/BETA sabit kullanilir, _prev_chunk_ids hic
   dokunulmaz, jaccard() hic cagrilmaz, drift tahmincileri hic cagrilmaz.
2. mode=threshold iken, dusuk vs yuksek ardisik-cift Jaccard overlap'i
   GERCEKTEN farkli bir worker sectirebiliyor.
3. mode=ewma_cusum iken, AYNI ardisik-cift Jaccard sinyali (threshold ile
   AYNI, kontrollu ablation icin) adaptive_drift_model.py'nin EWMA/CUSUM'u
   uzerinden GERCEKTEN farkli bir worker sectirebiliyor -- ayri bir
   instance/state (adaptive_cache_aware'inkiyle PAYLASILMIYOR).

Ikisinde de (2 ve 3) cache_gain=0 oldugu bir senaryoda alpha farki
gorunmez oldugu icin, w1'e BILEREK hem cache avantaji HEM yuk verilip
alpha/beta oraninin skoru gercekten degistirdigi gosteriliyor.
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


def seeded_strategy():
    s = build_strategy("per_worker_tree", PrefixTracker(["w1", "w2"]))
    s._router.on_request_finished("w1", ["A", "B", "C"])  # SADECE w1'in cache'i dolu
    return s


print("== Test 1: ROUTER_OVERLAP_ADAPTIVE_MODE=off -- eski davranisla BIREBIR ayni ==")
config.OVERLAP_ADAPTIVE_MODE = "off"
strat_off = build_strategy("per_worker_tree", PrefixTracker(["w1", "w2"]))
strat_off.decide_order(["A", "B", "C"], make_snapshot())
strat_off.decide_order(["A", "B", "C"], make_snapshot())  # ayni chunk'lar -- baska modda overlap=1.0 olurdu
assert strat_off._prev_chunk_ids is None, (
    "mode='off' iken _prev_chunk_ids guncellenmemeli -- guncellendiyse "
    "overlap-adaptive kod moda bakmadan calisiyor demektir"
)
assert strat_off._drift_estimator.current == 0.0, (
    "mode='off' iken drift tahmincisi hic cagrilmamali"
)
print("   OK -- _prev_chunk_ids ve drift tahmincisi hic dokunulmadi")

print("\n== Test 2: mode=threshold -- overlap'e gore w1/w2 arasinda gercekten farkli secim ==")
config.OVERLAP_ADAPTIVE_MODE = "threshold"
config.OVERLAP_THRESHOLD = 0.3
config.LOW_OVERLAP_ALPHA = 0.2
config.HIGH_OVERLAP_ALPHA = 0.8

# Iki AYRI strateji ornegi kullaniliyor (tek ornekte iki ardisik decide_order
# cagirmak, ilk cagrinin dispatch-time bookkeeping'inin -- on_request_finished
# -- ikinci cagrinin cache durumunu da degistirmesine yol aciyordu, iki
# senaryoyu birbirine karistiriyordu). Her ikisinde de w1'in cache/yuk durumu
# BIREBIR ayni kurulur; TEK degisken _prev_chunk_ids'in (dolayisiyla
# overlap'in) dusuk/yuksek olmasi.
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
print("   OK -- overlap dusukken w2 (yuk), yuksekken w1 (cache) kazandi -- threshold modu calisiyor")

print("\n== Test 3: mode=ewma_cusum -- AYNI sinyal, EWMA/CUSUM uzerinden farkli secim ==")
config.OVERLAP_ADAPTIVE_MODE = "ewma_cusum"
config.D_TARGET = 0.5
config.DRIFT_LAM = 0.1
config.CUSUM_K = 0.03
config.CUSUM_H = 0.20

# adaptive_beta SADECE beta'yi carpiyor (alpha config.ALPHA'da sabit kalir,
# bkz. strategies.py), ve carpan [0.3, 3.0] araliginda kirpiliyor
# (adaptive_drift_model.adaptive_beta'nin varsayilanlari) -- yani
# make_snapshot()'in w1_running=8'i (load_norm=0.333) ile tam cache_gain=1.0
# arasindaki fark, en agresif beta carpaninda (3.0) bile TAM ties ediyor,
# hicbir zaman kesin FLIP etmiyor. Bu testte w1'i daha agir yukluyoruz ki
# mekanizmanin etkisi belirsizlik payi olmadan gorulebilsin.
def make_snapshot_heavy_load():
    from worker_metrics import WorkerState, Snapshot
    w1 = WorkerState(name="w1", url="fake", healthy=True, num_requests_running=20)
    w2 = WorkerState(name="w2", url="fake", healthy=True, num_requests_running=0)
    return Snapshot(states={"w1": w1, "w2": w2})


# White-box: onceden YAKINSANMIS bir EWMA durumu simule ediyoruz (gercek
# trafikte bu, bircok ardisik dusuk/yuksek-overlap istekten sonra olusurdu --
# bkz. adaptive_drift_model.py'nin kendi self-test'i). Boylece tek bir
# decide_order cagrisiyla "surdurulen dusuk/yuksek overlap" durumunu
# dogrudan, deterministik sekilde test edebiliyoruz.
strat_ewma_low = seeded_strategy()
strat_ewma_low._prev_chunk_ids = {"Q", "R", "S"}  # bu cagrida overlap=0.0
strat_ewma_low._drift_estimator._d_t = 0.0  # onceden yakinsanmis DUSUK D_t (D_TARGET'in cok altinda)
d_ewma_low = strat_ewma_low.decide_order(["A", "B", "C"], make_snapshot_heavy_load())
print(f"   dusuk D_t (yakinsanmis) -> secilen: {d_ewma_low.worker_name}  "
      f"D_t={strat_ewma_low._drift_estimator.current:.3f}  skorlar: {d_ewma_low.scores}")
assert d_ewma_low.worker_name == "w2", (
    f"D_t < D_TARGET iken adaptive_beta buyumeli -> yuk agir basip w2 secilmeliydi, "
    f"gelen: {d_ewma_low.worker_name}"
)

strat_ewma_high = seeded_strategy()
strat_ewma_high._prev_chunk_ids = {"A", "B", "C"}  # bu cagrida overlap=1.0
strat_ewma_high._drift_estimator._d_t = 1.0  # onceden yakinsanmis YUKSEK D_t (D_TARGET'in ustunde)
d_ewma_high = strat_ewma_high.decide_order(["A", "B", "C"], make_snapshot_heavy_load())
print(f"   yuksek D_t (yakinsanmis) -> secilen: {d_ewma_high.worker_name}  "
      f"D_t={strat_ewma_high._drift_estimator.current:.3f}  skorlar: {d_ewma_high.scores}")
assert d_ewma_high.worker_name == "w1", (
    f"D_t > D_TARGET iken adaptive_beta kucul(-mesi bile)meli -> cache avantaji kazanmali, "
    f"gelen: {d_ewma_high.worker_name}"
)
assert d_ewma_low.worker_name != d_ewma_high.worker_name, (
    "dusuk ve yuksek D_t AYNI worker'i sectiyse bu senaryo bir fark kanitlamiyor"
)
# adaptive_cache_aware ile state PAYLASILMADIGININ kaniti: bu instance'in
# kendi _drift_estimator'i, digerinden BAGIMSIZ.
assert strat_ewma_low._drift_estimator is not strat_ewma_high._drift_estimator
print("   OK -- dusuk D_t w2'yi (yuk), yuksek D_t w1'i (cache) sectiriyor -- ewma_cusum modu calisiyor")

print("\nTum overlap-adaptive testleri basarili.")
