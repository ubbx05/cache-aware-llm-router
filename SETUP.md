# Başka bir makinede kurulum

`git clone` sana **sadece kodu** verir — korpus, embedding'ler ve trace'ler
`.gitignore`'da (`corpus/`, `*.jsonl`, `*.json`, `embeddings.npy`), yani repoyla
birlikte gelmezler. Karşı makinede bu dosyalar zaten mevcut olduğu için asıl iş
onları taşımak değil, **doğru olduklarını doğrulamak** (1. bölüm).

Genel ilke: bu kurulumda sessizce yanlış olabilecek iki şey var, ikisi de hata
vermeden makul görünen sayılar üretir.

- **Yanlış veri** (1. bölüm): farklı bir korpus/trace, sorunsuz koşar ama
  buradaki ölçümlerle karşılaştırılamaz.
- **Yanlış kalibrasyon** (3. bölüm): sabitler bu donanıma ait. Paper'ın kendi
  bulgusu (Bölüm VI-E), yanlış kalibre edilmiş tek bir sabitin bir ablation'ın
  sonucunu *ters çevirdiği*.

Eksik dosya ya da yanlış IP kendini hemen belli eder; bu ikisi etmez. O yüzden
1. ve 3. bölümler diğerlerinden önce gelir.

---

## 1. Veri dosyaları — kopyalama değil, DOĞRULAMA problemi

Karşı makinede korpus ve trace'ler **zaten var**. O yüzden buradaki risk
"dosya eksik" değil (o kendini hemen belli eder, `replay.py` açılışta ölür).
Risk, oradaki dosyaların **başka bir veri seti** olması: farklı bir korpus ya
da farklı parametrelerle üretilmiş bir trace, hiçbir hata vermeden koşar,
makul görünen sayılar üretir ve o sayılar buradaki koşularla
karşılaştırılamaz. Sessizce yanlış olan tek şey bu.

Bu makinedeki kanonik veri setinin parmak izi (2026-08-15'te ölçüldü;
`~/bil401/old/cache-aware-llm-router/` ile `~/Downloads/` kopyaları
byte-byte aynı):

```
92fba696def911504a1286e2e3c96e9d  corpus/corpus.jsonl
a0d29eb13fbd72060a3338b2434de039  corpus/embeddings.npy
a7a395db102006c6132de5f886fe04ea  corpus/qa.jsonl
a5874ffece10fcf3aa44eea2d4308679  trace.jsonl
abbe8dec6fa2267e0f919eae0f5f2414  trace_hot.jsonl
```

16 Ağustos'ta kullanılan `trace_hot_drift.jsonl` ve
`trace_low_locality.jsonl` bu eski parmak izi tablosunda yoktur. Uzak'a
yüklemeden önce bu iki dosyanın MD5 değerlerini ve üretim parametrelerini veri
manifestine ekle; isim benzerliğini doğrulama yerine kullanma.

Karşı makinede, veri neredeyse:

```bash
cd <oradaki-veri-dizini>
md5sum corpus/corpus.jsonl corpus/embeddings.npy corpus/qa.jsonl \
       trace.jsonl trace_hot.jsonl
```

**Hepsi tutuyorsa:** aynı veri setindesiniz, buradaki tüm ölçümlerle
(paper'ın 0.476 session-adjacent özeti ve top-k A/B'leri; canlı
top-k=10 detektörü için `D_TARGET=0.322`)
doğrudan karşılaştırılabilir. Devam et.

**`trace*.jsonl` tutmuyor ama `corpus/` tutuyorsa:** başka parametrelerle
üretilmiş bir trace var demektir. Ölümcül değil ama `ROUTER_D_TARGET`
o trace için yeniden ölçülmeli (bkz. 3. bölüm) ve sonuçlar buradaki
koşuların devamı olarak değil, ayrı bir seri olarak raporlanmalı.

**`corpus/` tutmuyorsa:** dur. Farklı korpus = farklı chunk id'leri = farklı
retrieval = hiçbir şey karşılaştırılabilir değil. Bu durumda buradaki
kanonik kopyayı gönder (~13 MB) ve oradakinin üstüne yaz:

```bash
rsync -av ~/bil401/old/cache-aware-llm-router/corpus \
          ~/bil401/old/cache-aware-llm-router/trace.jsonl \
          ~/bil401/old/cache-aware-llm-router/trace_hot.jsonl \
          karsi-pc:<hedef-dizin>/
```

**Korpusu orada yeniden ÜRETME.** `build_corpus.py` deterministik görünüyor
ama embedding'ler CPU'da float32 üretiliyor ve farklı BLAS/donanımda son
bit'leri oynayabiliyor; retrieval top-k'sı eşiğe yakın chunk'larda buna
duyarlı. Yani yeniden üretilmiş bir korpus checksum'ı tutmaz ve pratikte
üçüncü şıktaki duruma düşersin — üstelik ~1 saat CPU harcayarak. Sıfırdan
üretmek sadece korpusu gerçekten değiştirmek istiyorsan anlamlı:

```bash
python3 bench/build_corpus.py --tquad train-v0.1.json --out ./corpus
python3 bench/gen_trace.py --corpus ./corpus --out trace.jsonl \
    --n 3000 --seed 401 --zipf-s 1.0 --session-len 4 --think-time 8 --session-rate 0.5
```

Üretilen her trace'in yanına bir `.manifest.json` yazılıyor ve o manifest'ler
repoda commitli (`bench/*.manifest.json`) — karşı taraftaki trace'in hangi
parametrelerle üretildiğini, checksum tutmasa bile oradan okuyabilirsin.

---

## 2. Bağımlılıklar

`requirements.txt` **sadece router'ın** bağımlılıklarını listeliyor. Bench
harness'ı (`replay.py`, `build_corpus.py`, `overlap_measurement.py`) fazladan üç
paket istiyor ve bunlar orada yorum satırında duruyor:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install numpy sentence-transformers transformers
```

`transformers` opsiyonel değil pratikte: `ROUTER_TOKENIZER=hf` kullanıyorsun
(kullanmalısın da — `approx` mod karakter bazlı ve yanlı, config.py'nin kendi
notu "paper'a girecek hiçbir şeyde kullanma" diyor).

Python 3.11 kullanılıyor. vLLM makinesinde ayrıca vLLM'in kendi kurulumu var,
o ayrı bir ortam.

İlk `replay.py` koşusu `intfloat/multilingual-e5-base`'i HuggingFace'ten indirir
(~1 GB). Ağı olmayan bir makinede önce `HF_HOME` cache'ini kopyala.

---

## 3. Kalibrasyon sabitleri — YENİ DONANIMDA YENİDEN ÖLÇÜLMELİ

Paper'daki Tablo (Bölüm IV-C) "measured, not guessed" diye beş sabit sayıyor.
Bunların hepsi **bu iki makinenin GPU'suna ve bu trace'e** ait. Yeni makinede
körlemesine taşınırsa sayılar çıkar ama anlamsız olur.

| Sabit | Bu donanımdaki değer | Yeni makinede ne yapmalı |
|---|---|---|
| `ROUTER_TRACKER_CAPACITY` | **5840 blok** | vLLM'in açılış log'undaki KV cache boyutunu oku, 16'ya böl (BLOCK_SIZE), **iki makinenin küçüğünü** al |
| `ROUTER_LOAD_REF` | 16 eşzamanlı istek | tek-worker concurrency sweep, throughput/latency (Kleinrock power) tepe noktası |
| `ROUTER_D_TARGET` | 0.322 | Router'ın varış sıralı, aynı-session önceki istek Jaccard akışı, top-k=10 — trace **veya top-k** değişirse yeniden ölçülmeli |
| `ROUTER_CUSUM_H` | 0.30 | `bench/calibrate_cusum.py`, nominal ve drift-0.1 trace ayrımı; alarm telemetrisi için, routing ağırlığını doğrudan değiştirmez |
| DualMap SLO / rebalance | 38272 / 57408 token | `LOAD_REF × ortalama prompt token`; oran 1.5× sabit |

### `TRACKER_CAPACITY` neden bu listenin başında

Kodda uyumluluk için varsayılan **50 000** (`config.py`), fakat rapora girecek
runner'lar canlı koşuda pozitif bir `--tracker-capacity` değeri zorunlu tutar.
16 Ağustos ana strateji, yük/lokalite, ordering-kalibrasyon ve beta koşuları
5840 ile yapıldı. Yalnız tarihsel timing ve ilk ordering sweep'i 50 000'deydi;
raporda ayrıca etiketlenir. Paper Bölüm VI-E, eski beta deseninin kapasite
kalibrasyonundan sonra yeniden üretilmediğini gösteriyor.

`cache_aware` ailesi için bu sabit doğrudan skoru belirliyor. `per_worker_tree`
ailesi kendi knowledge tree'sini kullandığı için ondan daha az etkileniyor —
ama sıralama ablation'ını (`canonical`/`relevance`/`greedy`) `cache_aware` router'ı
altında koşuyorsan tam ortasındasın.

Yeni GPU'da kapasiteyi bulmak için vLLM'i başlat ve açılış log'unda KV cache
satırını ara; `--gpu-memory-utilization` değeri değişirse bu sayı da değişir.

---

## 4. Worker adresleri

`config.py:40,45` içinde W1/W2 için **sabit kodlanmış Tailscale IP'leri** var
(`100.89.101.52`, `100.64.0.2`). Yeni makinede bunlar neredeyse kesin yanlış.
Env ile geç, dosyayı düzenleme:

```bash
W1_URL=http://localhost:8000        # vLLM aynı makinedeyse
W2_URL=http://<peer-tailscale-ip>:8000
W2_ENABLED=true                     # ikinci worker varsa
```

Tek worker'la koşacaksan `W2_ENABLED` set etme — o zaman routing kolları
anlamsızlaşır (her şey w1'e gider) ama **sıralama ablation'ı hâlâ geçerlidir**,
çünkü orada değişken chunk sırası, worker seçimi değil.

---

## 5. Doğrulama sırası

GPU zamanı harcamadan önce, artan maliyet sırasıyla:

```bash
# 1. Router mantığı, GPU'suz (hepsi saniyeler içinde)
python3 smoke_test.py
python3 smoke_test_semantic.py
python3 smoke_test_overlap_adaptive.py
python3 smoke_test_greedy_order.py
python3 bench/cacheweaver_util.py

# 2. Veri parmak izi (anlik) -- 1. bolumdeki md5 tablosuyla karsilastir
md5sum <veri-dizini>/corpus/corpus.jsonl <veri-dizini>/trace.jsonl

# 3. Retrieval gercekten calisiyor mu (GPU'suz, ~30 sn)
python3 bench/overlap_measurement.py --corpus <veri-dizini>/corpus \
    --trace <veri-dizini>/trace.jsonl --limit 400
#    -> session-adjacent mean 0.485, global-adjacent 0.053 çıkmalı.
#       (Bu makinede 2026-08-15'te doğrulandı; tam 3000 istekte 0.476/0.055.)

# 4. Router ayakta mı (vLLM gerekli)
curl -s localhost:8080/health
curl -s localhost:8080/router/state | head

# 5. Kısa canlı koşu (--limit ile, GPU gerekli ama ucuz)
cd bench && python3 replay.py --corpus <veri-dizini>/corpus \
    --trace <veri-dizini>/trace.jsonl --limit 20 --order canonical --out /tmp/x.jsonl
```

2. ve 3. adım ayrı ayrı duruyor çünkü ayrı şeyler kanıtlıyorlar. md5, karşı
taraftaki dosyanın **buradakiyle aynı dosya** olduğunu söyler;
`overlap_measurement.py` ise korpus ile trace'in **birbirine ait** olduğunu ve
retrieval hattının çalıştığını söyler. İkincisi, checksum'lar tutmadığında
(ör. bilerek yeni bir trace ürettiğinde) elindeki tek kontrol. Eşleşmeyen bir
korpus/trace çifti canlı koşuda "düşük recall" olarak görünür ve kolayca
stratejinin suçu sanılır.

---

## 6. Ölçüm protokolü (makine değişse de değişmez)

- **Her koldan önce vLLM'i cold restart et.** Prefix cache kalıcı; önceki kolun
  cache'i bir sonrakine kredi yazar.
- Bir ablation'da **tek bir değişken** oynasın. Sıralama ablation'ında router
  stratejisi dört kolda da aynı kalmalı.
- `replay.py` sonunda "client fell behind its own schedule" uyarısı verirse o
  koşunun yük ekseni geçersizdir — `--speedup` düşür, koşuyu tekrarla.
- Tek koşu bir bulgu değil. `score_quality.py` %3'ün altındaki farkları zaten
  "single-run noise" diye işaretliyor; rapora girecek her kol **3×**.

---

## 7. Ordering ablation'ını çalıştırma

Dört kolun tek değişkeni chunk sırasıdır; router bütün kollarda
`cache_aware` kalır. Runner kolları repeat-major döndürür, router'ı her koşuda
yeniler ve iki vLLM'in soğuk restart edildiğini kullanıcıdan onaylatır.

```bash
python3 bench/sweep_ordering.py \
  --corpus <veri-dizini>/corpus \
  --trace <veri-dizini>/trace_hot.jsonl \
  --worker http://localhost:8000 \
  --worker http://<peer>:8000 \
  --arms canonical,relevance,greedy,pinned_prefix \
  --repeats 3 --top-k 10 --limit 300 --speedup 5 \
  --tokenizer hf --tracker-capacity 5840 \
  --outdir runs/ordering_calibrated
```

`--no-pause` yalnız dry-run/orkestrasyon testi içindir. Canlı rapor koşusunda
kullanılırsa soğuk-cache protokolü doğrulanmış sayılmaz. `greedy` bayraklarını
değiştirmek (`--greedy-protect-top-k`, TTL veya insertion timing) ayrı bir
ablation'dır.

## 8. Kalibre ana strateji sweep'i

Önce GPU kullanmadan planı doğrula:

```bash
python3 bench/sweep_strategy.py --dry-run
```

Beş çekirdek stratejinin 3 × 5 = 15 canlı koşusu:

```bash
python3 bench/sweep_strategy.py \
  --corpus <veri-dizini>/corpus \
  --trace <veri-dizini>/trace_hot.jsonl \
  --worker http://localhost:8000 \
  --worker http://<peer>:8000 \
  --tracker-capacity 5840 \
  --outdir runs/strategy_sweep
```

Varsayılanlar paper protokolüdür: 800 istek, top-k=10, speedup=8, üç tekrar,
`hf` tokenizer. Çekirdek kollar:

- `round_robin`
- `least_loaded`
- `cacheweaver_dualmap`
- `cache_aware`
- `per_worker_tree`

`--all-strategies` ayrıca `adaptive_cache_aware` ve
`semantic_per_worker_tree` kollarını ekler. Yeni semantik sweep varsayılan
`--semantic-top-k 1` ile gerçek aday eleme yapar; rapordaki eski semantik
no-op sonucu `top-k=2=N` koşuluna aittir ve bunun yerine kullanılamaz.

Her koşunun raw JSONL'i, özet JSON/CSV'si ve
`router_<port>_<strategy>_r<repeat>.log` dosyası `--outdir` altına yazılır.
İki-aşamalı kolda eski completion-fazı alanları korunur; `decision_s`,
`e2e_ttft_s` ve `e2e_total_s` ayrıca kaydedilir. Yeni karşılaştırmada uçtan
uca alanı kullan.

## 9. Kalibre beta × yük × lokalite sweep'i

Plan kontrolü:

```bash
python3 bench/sweep_beta.py --dry-run
```

Raporlanan 2 × 2 replikasyonu yeniden üretmek için:

```bash
python3 bench/sweep_beta.py \
  --corpus <veri-dizini>/corpus \
  --high-trace <veri-dizini>/trace_hot.jsonl \
  --low-trace <veri-dizini>/trace_low_locality.jsonl \
  --worker http://localhost:8000 \
  --worker http://<peer>:8000 \
  --loads 30,35 --betas 1,0 --repeats 1 \
  --tracker-capacity 5840 \
  --outdir runs/beta_sweep
```

`--repeats 1` mevcut kalibrasyon replikasyonuyla aynıdır; formal ayrışma
iddiası için en az üç tekrar kullan. Runner `cache_aware` ve canonical prompt
sırasını sabit tutar, hücre sırasını tekrarlar arasında döndürür ve sonucu
nedensel “kazanan” yorumu eklemeden JSON/CSV olarak yazar.

## 10. Runner çıktılarının doğrulanması

- Schedule-lag p99 1 saniyeyi geçerse koşuyu geçersiz say.
- `n_failed` sıfır değilse ilgili router logunu incele.
- Cached-token fraction yeni runner'larda toplam cached / toplam prompt token
  olarak hesaplanır; tarihsel ordering tablosundaki istek-başına ortalamayla
  aynı değildir.
- Load CV iki beklenen worker'ı da içerir; tüm trafik tek worker'a giderse
  iki-worker CV sıfır değil 1'dir.
- TTFT p99 ana özetin parçasıdır; yalnız p50'ye bakma.
- Ham JSONL, summary JSON/CSV, router logu, trace manifesti ve çalışan komut
  birlikte saklanmadan sonuç yeniden üretilebilir sayılmaz.

Uzun ömürlü router süreçleri artık okunmayan `stdout=PIPE` kullanmaz. Uvicorn
çıktısı dosyaya yazıldığı için yüzlerce access-log satırı pipe buffer'ını
doldurup sweep'i sessizce kilitlemez.
