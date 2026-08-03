"""Standalone smoke test: semantic_per_worker_tree stratejisini dogrular.

smoke_test.py ile ayni desen -- vLLM/network yok, WorkerState/Snapshot elle
kuruluyor. Ekstra olarak: SemanticPerWorkerTreeStrategy.__init__ config.WORKERS
uzerinden worker listesini sabitledigi icin (bkz. strategies.py), W2_ENABLED
env var'i strategies/config import EDILMEDEN once set edilmeli -- config.py
WORKERS listesini import-time'da bir kere olusturuyor.

GOREV_semantic_router_entegrasyonu.md'nin kabul kriterlerinden biri: mevcut
5 stratejinin davranisi degismemeli. Bu dosya SADECE yeni stratejiyi test
ediyor; regresyon kontrolu icin smoke_test.py ayrica calistirilmali (degisik
env gerektirmiyor, o dosyaya dokunulmadi).
"""
import os

os.environ.setdefault("W2_ENABLED", "true")  # config.WORKERS import-time okunuyor

import config
from prefix_tracker import PrefixTracker, ApproxTokenizer, block_hashes
from strategies import build_strategy
from worker_metrics import WorkerState, Snapshot

tok = ApproxTokenizer()


def make_snapshot(w1_healthy=True, w2_healthy=True, w1_running=0, w2_running=0):
    w1 = WorkerState(name="w1", url="fake", healthy=w1_healthy, num_requests_running=w1_running)
    w2 = WorkerState(name="w2", url="fake", healthy=w2_healthy, num_requests_running=w2_running)
    return Snapshot(states={"w1": w1, "w2": w2})


tracker = PrefixTracker(["w1", "w2"])
strategy = build_strategy("semantic_per_worker_tree", tracker)
assert strategy._names == ["w1", "w2"], f"beklenen ['w1','w2'], gelen {strategy._names}"

# Testin cekirdegi: semantik on-filtrenin GERCEKTEN aday havuzunu daralttigini
# kanitlamak icin top_k=1'e indiriyoruz. Boylece "en iyi cache'e sahip worker"
# ile "semantik olarak en alakali worker" celistiginde hangisinin kazandigini
# net gorebiliyoruz.
config.SEMANTIC_TOP_K = 1

osmanli_queries = [
    "Osmanli Imparatorlugu nasil kuruldu",
    "Fatih Sultan Mehmet Istanbul'u nasil fethetti",
    "Osmanli padisahlari kimlerdir",
]

print("== Adim 1: w1'in semantik centroid'i Osmanli konusuna egitiliyor ==")
print("   (chunk_ids kasitli olarak farkli/kucuk -- cache degil, SADECE centroid onemli)")
for i, q in enumerate(osmanli_queries):
    snap = make_snapshot()
    strategy.decide_order([f"osmanli_chunk_{i}"], snap, query_text=q)

print("\n== Adim 2: w2'ye, konuyla ILGISIZ bir sorguyla BUYUK bir cache avantaji veriliyor ==")
print("   query_text=None -- bu cagri semantik katmani hic etkilemiyor (centroid guncellenmiyor)")
snap = make_snapshot()
strategy.decide_order(["X", "Y", "Z"], snap, query_text=None)
# Dogrulama: query_text=None iken semantik filtre devre disi, per_worker_tree
# X/Y/Z icin gercekten w2'yi secmis olmali (ilk cagri, cache bos, ama cache_gain
# esit(0) oldugundan yuk/rastgelelik karar verir -- asil kontrol asagida).
w2_seeded = strategy._per_worker_router._trees["w2"]
w2_seeded.insert(["X", "Y", "Z"])  # w2'nin X,Y,Z icin tam cache'i oldugunu GARANTIYE al

print("\n== Test 1: Osmanli sorusu + [X,Y,Z] retrieve edilmis (w2'de tam cache var) ==")
print("   Cache-kor bir per_worker_tree burada KESINLIKLE w2'yi secerdi (cache_gain=1.0).")
print("   SEMANTIC_TOP_K=1 oldugu icin beklenen: w1 (semantik olarak tek aday), cache'e RAGMEN.")
snap = make_snapshot()
decision = strategy.decide_order(["X", "Y", "Z"], snap, query_text="Osmanli sultanlarinin listesi nedir")
print(f"   secilen worker: {decision.worker_name}  (skorlar: {decision.scores})")
assert decision.worker_name == "w1", (
    f"semantik on-filtre calismiyor -- w2 secildi (cache_gain onceligi almis olmali), "
    f"beklenen w1. scores={decision.scores}"
)
assert list(decision.scores.keys()) == ["w1"], (
    f"aday havuzu w1'e daralmamis, gelen: {list(decision.scores.keys())}"
)

print("\n== Test 2: ayni sorgu ama w1 DUSUK (unhealthy) -- ac birakmama kontrolu ==")
print("   Semantik favori (w1) elenmis olsa da istek w2'ye dusmeli, NoHealthyWorker firlatilmamali.")
snap_w1_down = make_snapshot(w1_healthy=False)
decision2 = strategy.decide_order(["X", "Y", "Z"], snap_w1_down, query_text="Osmanli sultanlarinin listesi nedir")
assert decision2.worker_name == "w2", f"w1 down iken w2'ye dusmeli, gelen: {decision2.worker_name}"
print(f"   secilen worker: {decision2.worker_name}  (dogru -- fallback calisti)")

print("\n== Test 3: query_text=None -- semantik katman devre disi, duz per_worker_tree gibi davranmali ==")
snap = make_snapshot()
decision3 = strategy.decide_order(["X", "Y", "Z"], snap, query_text=None)
print(f"   secilen worker: {decision3.worker_name}  (skorlar: {decision3.scores})")
# Asil kontrol: aday havuzu TUM healthy worker'lari icermeli (semantik filtre
# yok). Kazananin w1 mi w2 mi oldugunu SABIT beklemiyoruz -- Test 1'in
# dispatch-time bookkeeping'i (on_request_finished) w1'in agacina da X,Y,Z'yi
# yazdi (per_worker_tree'nin normal davranisi: kazanan worker'in cache'i
# guncellenir), yani burada ikisi de cache_gain=1.0'a esit olabilir ve
# per_worker_tree_router.choose() esitlikte rastgele secer (kasitli, w1-
# yanliligini onlemek icin, bkz. per_worker_tree_router.py). O yuzden burada
# SADECE aday havuzunun daralmadigini dogruluyoruz.
assert set(decision3.scores.keys()) == {"w1", "w2"}, (
    f"query_text=None iken aday havuzu TUM healthy worker'lari icermeli, gelen: {decision3.scores.keys()}"
)

print("\n== Test 4: select() arayuzu de calisiyor mu (registry/genel smoke-test uyumlulugu) ==")
from strategies import RequestContext

ctx = RequestContext(
    prompt_text="Osmanli padisahlarinin sirasi nedir",
    block_hashes=block_hashes(tok.encode("Osmanli padisahlarinin sirasi nedir")),
    chunk_hashes=["osmanli_chunk_0"],
)
snap = make_snapshot()
decision4 = strategy.select(ctx, snap)
print(f"   secilen worker: {decision4.worker.name}, sebep: {decision4.reason}")
assert decision4.reason == "semantic_per_worker_tree"

print("\nTum semantic_per_worker_tree testleri basarili.")
