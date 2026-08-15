# Başka bir makinede kurulum

`git clone` sana **sadece kodu** verir. Deneyi çalıştırmak için gereken her
şeyin geri kalanı — korpus, embedding'ler, trace'ler, TQuAD kaynak dosyası —
`.gitignore`'da (`corpus/`, `*.jsonl`, `*.json`, `embeddings.npy`). Yani temiz
bir clone'da `replay.py` ilk satırda ölür. Bu dosya aradaki farkı kapatıyor.

En kritik madde 3. bölümde: **kalibrasyon sabitleri bu donanıma ait**. Paper'ın
kendi bulgusu (Bölüm VI-B), yanlış kalibre edilmiş bir sabitin bir ablation'ın
sonucunu *ters çevirdiği*. Yeni makinede o sabitleri taşımak, sessizce yanlış
bir sonuç üretmenin en kolay yolu.

---

## 1. Veri dosyaları — kopyala, yeniden üretme

Şu an bu dosyalar `/home/os/bil401/old/cache-aware-llm-router/` altında:

| Dosya | Boyut | Ne işe yarıyor |
|---|---|---|
| `corpus/corpus.jsonl` | 3.5 MB | 2619 chunk, BLOCK_SIZE'a padlenmiş |
| `corpus/embeddings.npy` | 8.0 MB | chunk embedding'leri (multilingual-e5-base, L2-normalize) |
| `corpus/embed_meta.json` | 161 B | hangi modelle üretildiği |
| `corpus/qa.jsonl` | 1.8 MB | soru + gold answer + ground-truth chunk |
| `trace.jsonl` | 812 KB | 3000 istek, zipf_s=1.0, seed=401 |
| `trace_hot.jsonl` | 804 KB | zipf_s=1.5, session-len=4 |
| `train-v0.1.json` | 3.0 MB | TQuAD kaynağı (sadece korpusu yeniden üretmek için) |

Toplam ~18 MB. Yeni makineye:

```bash
rsync -av /home/os/bil401/old/cache-aware-llm-router/corpus \
          /home/os/bil401/old/cache-aware-llm-router/trace.jsonl \
          /home/os/bil401/old/cache-aware-llm-router/trace_hot.jsonl \
          yeni-pc:~/bil401/data/
```

**Neden yeniden üretmek yerine kopyalıyoruz:** `build_corpus.py` deterministik
görünüyor ama embedding'ler CPU'da float32 üretiliyor ve farklı BLAS/donanımda
son bit'leri oynayabiliyor. Retrieval top-k'sı eşiğe yakın chunk'larda buna
duyarlı — yani yeniden üretilmiş bir korpus, eski koşularınla **karşılaştırılabilir
olmayan** bir top-k üretebilir. Sıfırdan üretmek ayrıca ~1 saat CPU
(2619 chunk × forward pass).

Gerçekten sıfırdan gerekiyorsa (ör. korpusu büyütmek):

```bash
python3 bench/build_corpus.py --tquad train-v0.1.json --out ./corpus
python3 bench/gen_trace.py --corpus ./corpus --out trace.jsonl \
    --n 3000 --seed 401 --zipf-s 1.0 --session-len 4 --think-time 8 --session-rate 0.5
```

Üretilen her trace'in yanına bir `.manifest.json` yazılıyor ve o manifest'ler
repoda commitli (`bench/*.manifest.json`) — hangi parametrelerle üretildiğini
oradan doğrulayabilirsin.

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

Paper'daki Tablo (Bölüm V-B) "measured, not guessed" diye beş sabit sayıyor.
Bunların hepsi **bu iki makinenin GPU'suna ve bu trace'e** ait. Yeni makinede
körlemesine taşınırsa sayılar çıkar ama anlamsız olur.

| Sabit | Bu donanımdaki değer | Yeni makinede ne yapmalı |
|---|---|---|
| `ROUTER_TRACKER_CAPACITY` | **5840 blok** | vLLM'in açılış log'undaki KV cache boyutunu oku, 16'ya böl (BLOCK_SIZE), **iki makinenin küçüğünü** al |
| `ROUTER_LOAD_REF` | 16 eşzamanlı istek | tek-worker concurrency sweep, throughput/latency (Kleinrock power) tepe noktası |
| `ROUTER_D_TARGET` | 0.529 | `bench/overlap_measurement.py`, session-adjacent mean — trace değişmiyorsa **aynı kalır**, bu bir workload özelliği |
| DualMap SLO / rebalance | 38272 / 57408 token | `LOAD_REF × ortalama prompt token`; oran 1.5× sabit |

### `TRACKER_CAPACITY` neden bu listenin başında

Kodda varsayılanı **50 000** (`config.py:220`) ve şu an çalıştırdığın komutlarda
`ROUTER_TRACKER_CAPACITY` **hiç geçmiyor** — yani router, her iki makinenin
gerçek kapasitesinin ~9 katı bir cache'e sahip olduğuna inanarak koşuyor. Paper
Bölüm VI-B tam olarak bunun sonucunu anlatıyor: kalibrasyondan önce β=0
kazanıyor görünüyordu, 5840'a düzeltilince 30× yükte kazanan β=1'e döndü.

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

# 2. Veri dosyaları yerinde mi + retrieval çalışıyor mu (GPU'suz, ~30 sn)
python3 bench/overlap_measurement.py --corpus ~/bil401/data/corpus \
    --trace ~/bil401/data/trace.jsonl --limit 400
#    -> session-adjacent mean 0.485, global-adjacent 0.053 çıkmalı.
#       (Bu makinede 2026-08-15'te doğrulandı; tam 3000 istekte 0.476/0.055.)
#       Belirgin şekilde farklıysa korpus ile trace birbirine ait değil.

# 3. Router ayakta mı (vLLM gerekli)
curl -s localhost:8080/health
curl -s localhost:8080/router/state | head

# 4. Kısa canlı koşu (--limit ile, GPU gerekli ama ucuz)
cd bench && python3 replay.py --corpus ~/bil401/data/corpus \
    --trace ~/bil401/data/trace.jsonl --limit 20 --order canonical --out /tmp/x.jsonl
```

2. adım en değerlisi: veri dosyalarının birbirine ait olduğunu GPU'ya
dokunmadan kanıtlıyor. Yanlış korpus/trace çifti canlı koşuda "düşük recall"
olarak görünür ve kolayca stratejinin suçu sanılır.

---

## 6. Ölçüm protokolü (makine değişse de değişmez)

- **Her koldan önce vLLM'i cold restart et.** Prefix cache kalıcı; önceki kolun
  cache'i bir sonrakine kredi yazar.
- Bir ablation'da **tek bir değişken** oynasın. Sıralama ablation'ında router
  stratejisi üç kolda da aynı kalmalı.
- `replay.py` sonunda "client fell behind its own schedule" uyarısı verirse o
  koşunun yük ekseni geçersizdir — `--speedup` düşür, koşuyu tekrarla.
- Tek koşu bir bulgu değil. `score_quality.py` %3'ün altındaki farkları zaten
  "single-run noise" diye işaretliyor; rapora girecek her kol **3×**.

---

## 7. Sıralama ablation'ını (faz 2) çalıştırma

Üç kol, router stratejisi sabit, her koldan önce vLLM restart:

```bash
# --- router (tek sefer başlat, strateji üç kolda da AYNI) ---
W1_URL=http://localhost:8000 W2_URL=http://<peer>:8000 W2_ENABLED=true \
  ROUTER_TOKENIZER=hf ROUTER_TRACKER_CAPACITY=<yeni-ölçülen-blok> \
  ROUTER_STRATEGY=cache_aware python3 main.py &

# --- her kol için: vLLM restart, sonra ---
cd bench
python3 replay.py --corpus ~/bil401/data/corpus --trace ~/bil401/data/trace.jsonl \
  --worker http://localhost:8000 --worker http://<peer>:8000 \
  --order canonical --out r_canon.jsonl
#   ... --order relevance --out r_relevance.jsonl
#   ... --order greedy    --out r_greedy.jsonl

python3 score_quality.py r_canon.jsonl r_relevance.jsonl r_greedy.jsonl
```

`greedy` kolunun üç ek bayrağı var (`--greedy-protect-top-k`, `--greedy-ttl`,
`--greedy-insert`); varsayılanlar CacheWeaver'ın yayınlanmış semantiğini verir
ve `--greedy-ttl 30` zaten `per_worker_tree_router` ile aynı. Bunları oynatmak
ayrı bir ablation'dır, ana üç kolla karıştırma.

`--order greedy` çıktısında `reorder depth` (harness'ın kendi ağacının bulduğu
yeniden kullanılabilir önek) ile `engine cached` (motorun gerçekten cache'ten
servis ettiği oran) yan yana basılıyor. **İkisi arasındaki uçurum bug değil,
bulgudur**: tek-global-tree varsayımının blokların hangi replikada olduğu
konusunda yanılması — `per_worker_tree`'nin var oluş sebebi tam olarak bu.
