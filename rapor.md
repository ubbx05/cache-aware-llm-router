# Çalışma Raporu — 15 Ağustos 2026

Tier 1'in kalan altı maddesi (A, B, C, D, N, O) kapatıldı. Bu rapor ne
yapıldığını, hangi sayıların nereden geldiğini ve bulguların paper'ın mevcut
iddialarını nasıl değiştirdiğini kaydeder.

**Özet:** Sıralama ablation'ı dört kola çıkarıldı ve `greedy`'nin canonical'a
göre ortalama istek-başı cache oranını +8.1 yüzde puan artırdığı, ana kalite
metriğinde relevance sırasından ayrışmadığı 3× tekrarla gösterildi. Kendi
önerimiz olan `pinned_prefix` negatif sonuç verdi ve olası mekanizması tartışıldı. Dispatch-vs-completion
bulgusunun yük ile ölçeklendiği, üstelik önerilen mekanizmanın öngördüğü yönde
ölçeklendiği gösterildi. CUSUM parametreleri ilk kez gerçek trafikte kalibre
edildi ve iki sabitin yanlış olduğu bulundu.

---

## 1. Deney ortamı ve protokol

| | |
|---|---|
| Trace | `trace_hot.jsonl` (zipf-s=1.5, session-len=4), 300 istek |
| Korpus | 2619 chunk, TQuAD, `multilingual-e5-base` |
| Model | Qwen2.5-7B-Instruct, 2 worker (w1 lokal, w2 Tailscale) |
| Router stratejisi | `cache_aware` (tüm kollarda **sabit**) |
| `TRACKER_CAPACITY` | 50000 (kalibre **değil**, tüm kollarda sabit — bkz. §7) |
| Tokenizer | `hf` |

**Protokol kuralları (tüm sweep'lerde uygulandı):**

- Her koldan önce **iki vLLM motoru da** yeniden başlatıldı. Prefix cache
  kalıcı; önceki kolun cache'i sonrakine kredi yazar.
- **Router da her kolda yeniden başlatıldı.** Sadece motorları restart etmek
  yetmiyor: router'ın `PrefixTracker`'ı silinmiş blokları hâlâ cache'te sanarak
  yeni kola başlar.
- Kollar **iç içe ve döndürülerek** koşuldu (repeat-major, her tekrarda sıra
  kaydırılıyor). Bir kolu arka arkaya 3 kez koşmak, makinedeki yavaş bir
  kaymanın (ısınma, arka plan yükü) tamamen o kola binmesine ve sıralama etkisi
  gibi okunmasına yol açardı.
- Ayrışma hükmü **sabit eşikle değil**, deneyin kendi varyansıyla veriliyor:
  kollar arası fark, kol içi standart sapmanın 2 katını aşmalı.
- 12 timing koşusunun tamamında schedule lag p99 ≤ 6 ms (eşik 1000 ms), yani
  yük ekseni güvenilir (`bench/check_lag.py`).

---

## 2. Madde C — Sıralama ablation'ına üçüncü kol: `greedy`

`replay.py --order greedy` eklendi: CacheWeaver Algoritma 1'i (router'ın kendi
kullandığı `bench/cacheweaver_util.py`) **tek bir global** knowledge tree'ye
karşı çalıştırıyor. Yani fikrin yayınlanmış, worker-kör hâli.

Sıralamadan sonrası canonical/relevance yoluyla birebir aynı tutuldu: aynı
`"\n\n"` join (CHUNK_SEP *değil*), aynı `fire()`, tek round-trip. Böylece
greedy-vs-canonical farkı yalnızca chunk sırasına atfedilebiliyor.

### 2.1 k=3'te hiçbir kol ayrışmadı

| Kol | ortalama istek-başı cache oranı |
|---|---|
| canonical | 79.1% |
| relevance | 78.7% |
| greedy | 81.1% |

Toplam yayılım 2.4pp, tek koşu. Kalite farkları da `score_quality`'nin %3
gürültü eşiğinin altında. **Bu bir bulgu:** top_k=3'te sıralama kaldıracı
yalnızca 3 chunk'ı permüte edebiliyor, ablation etkiyi görmek için fazla dar.

12 koşu harcayıp bu null sonucu daha yüksek güvenle öğrenmek yerine önce
etkinin var olup olmadığına bakıldı — k=10'a çıkıldı.

### 2.2 k=10'da etki ortaya çıktı (12 koşu, 4 kol × 3 tekrar)

| Kol | hit rate | TTFT p50 | kalite (contains gold) |
|---|---|---|---|
| **greedy** | **73.1% ±0.2%** | **0.088s ±0.006** | 79.6% ±0.4% |
| relevance | 67.7% ±0.1% | 0.111s ±0.025 | **79.9% ±0.4%** |
| canonical | 65.0% ±0.5% | 0.122s ±0.015 | 77.8% ±0.2% |
| pinned_prefix | 62.9% ±0.2% | 0.161s ±0.012 | 79.7% ±0.3% |

Fark 10.2pp, gürültü eşiği 0.4pp → **25 katı**.

**İki bağımsız sweep aynı sonucu verdi.** Ayrı oturumlar, ayrı restart'lar:

| Kol | 9-koşu sweep | 12-koşu sweep |
|---|---|---|
| canonical | 64.5% | 65.0% ±0.5% |
| relevance | 67.5% | 67.7% ±0.1% |
| greedy | 73.0% | 73.1% ±0.2% |

### 2.3 Rapora yazılabilecek cümle

> greedy, canonical'a göre ortalama istek-başı cache oranını +8.1 yüzde puan
> artırıyor ve TTFT p50'yi göreli %27.9 düşürüyor; ana kalite metriğinde
> relevance sırasından ayrışmıyor.

Kalite tarafında greedy (79.6%) ile relevance (79.9%) arasındaki 0.3pp fark
±0.2–0.4 sapmayla **ayırt edilemez**. Yani "greedy kalitede de kazandı" veya
"kalite kaybı kesinlikle yok" denemez; yalnızca bu ölçümde kalite farkının
ayrışmadığı söylenebilir. CacheWeaver'ın
kendi makalesinin ve paper'ın Bölüm III-C.2'sinin açık bıraktığı risk — greedy
reorder'ın alakalı chunk'ı sona itip kaliteyi düşürmesi — bu tek iş yükünde ve
contains-gold metriğinde gözlenmedi. 10 chunk'ın 7.17'si yer değiştirdiği hâlde.

**greedy teşhis verileri (k=10):** reorder depth ortalama 7.17/10, %84.0'ünde
sıfırdan büyük, isteklerin %43.3'ünde sıra değişiyor.

---

## 3. Madde D — `pinned_prefix` (session geçmişi öneki + relevance kuyruğu)

Tasarım: aynı session'da bir önceki turda servis edilen ve bu turda tekrar
retrieve edilen chunk'lar öne, **önceki sıralarıyla**; kalan relevance
sırasında. Gerekçe: projenin kendi merkezi ölçümü, session-adjacent overlap
0.48 iken global-adjacent 0.055. Bu, session affinity korunuyorsa önceki turdan
tekrarlanan chunk'ların cache'te olabileceği hipotezini kuruyor; kol worker
kimliğini sabitlemediği için gerçek cache yerleşimini tek başına kanıtlamıyor.

### 3.1 Mekanizma doğru çalıştı

k=3'te `engaged=57.0%` çıktı. Bu rastgele bir sayı değil: session uzunluğu 4,
yani turların %25'i ilk tur (pin yok); kalan %75 içinde paper'ın ölçtüğü
"ardışık aynı-session çiftlerinin %76.2'si en az bir chunk paylaşıyor" oranı
geçerli. **0.75 × 0.762 = %57.15**, ölçülen %57.0. Kol trace'in gerçek session
yapısını birebir takip ediyor.

### 3.2 Ama sonuç negatif

62.9% ±0.2% cache oranı ve 0.161s TTFT ile dört kolun **sonuncusu**. Relevance
67.7% cache + 79.9% kalite verirken pinned 62.9% cache + 79.7% kalite veriyor;
ana kalite metriği yakın kalırken cache ve TTFT sonuçları kötüleşiyor.

**Olası açıklama:** k=10'da `full=0.0%` — hiçbir istekte tüm chunk'lar
pinlenmiyor, ortalama 3.38/10 pinleniyor. Yani prompt'un başına ~3 chunk'lık
**session'a özgü** bir önek koyup arkasına 6.6 taze chunk diziliyor. Her session
kendine özel bir sıra üretiyor; bu, session'lar arası prefix tutarlılığını
azaltmış olabilir. Sıra isteklerin %65.7'sinde değişiyor, karşılığında kazanç
yok; deney bu nedensel yolu doğrudan ölçmüyor.

### 3.3 Asıl bulgu — dört kolun ortak örüntüsü

Dört kolu yan yana koyunca ortaya çıkan şey, tek tek sonuçlardan daha güçlü:

> **Denenen iki cache-oblivious şema, retriever'ın sırasını olduğu gibi
> bırakmaktan daha kötü.**

- canonical (chunk_id'ye göre diz): 65.0% — relevance'ın altında
- pinned_prefix (session geçmişine göre diz): 62.9% — relevance'ın altında
- relevance (hiç dizme): 67.7%
- greedy (tahminî global cache-history ağacına göre diz): 73.1% — **tek kazanan**

Sonuç, relevance sıralamasının benzer sorgularda benzer prefix'ler üretmiş
olabileceği ve cache-oblivious dayatmaların bu tutarlılığı bozduğu hipoteziyle
uyumlu. Deney prefix paylaşım yolunu doğrudan ölçmediği için bu açıklama kanıt
olarak sunulmuyor.

Bu, faz 2'nin kurgusunu değiştiriyor: soru "hangi sıralama şeması" değil,
**"sıralama cache durumunu biliyor mu"**.

Kalite tarafında canonical 77.8%, relevance 79.9%, greedy 79.6% ve
`pinned_prefix` 79.7% ölçülüyor. Bu farklar neden-sonuç açıklaması veya eşdeğerlik
kanıtı değil; yalnızca prespecified spread kuralıyla yorumlanıyor.

---

## 4. Madde N — Dispatch vs. completion, artık bir yük ekseniyle

Bu, projenin en şaşırtıcı bulgusuydu ve tek koşuya dayanıyordu. İki yük
seviyesinde 3'er tekrar yapıldı.

Metrikler bilerek sıralama sweep'inden farklı: iki kol da aynı prompt'ları aynı
motorlara gönderiyor, değişen tek şey router'ın **seçim yaparken neye
inandığı**. O yüzden ölçüm router inancı ile motorun kendi raporu arasında.

| | speedup=5 | speedup=15 |
|---|---|---|
| **dispatch** korelasyon | 0.972 ±0.010 | 0.980 ±0.003 |
| **dispatch** sıralama uyumu | 97.1% ±0.6% | 98.0% ±0.0% |
| **dispatch** bias | +31.3 ±9.2 tok | +23.7 ±1.0 tok |
| **dispatch** az-tahmin | 0.0% ±0.0% | 0.0% ±0.0% |
| **completion** korelasyon | 0.941 ±0.011 | 0.806 ±0.008 |
| **completion** sıralama uyumu | 92.5% ±0.6% | 79.7% ±0.9% |
| **completion** bias | −5.2 ±20.3 tok | −170.3 ±25.5 tok |
| **completion** az-tahmin | 6.2% ±0.7% | 29.6% ±0.8% |
| **fark / gürültü** | 1.45× | **10.3×** |

### 4.1 Yük etkileşimi mekanizma hipoteziyle uyumlu

Yük 3 katına çıkarken dispatch'in routing-relevant korelasyon ve sıra uyumu
yaklaşık sabit kaldı, completion ise bozuldu. Önerilen mekanizma şudur:
completion-time kayıt uçuştaki kardeş istekleri henüz göremez; dispatch-time
kayıt onları hemen görür, fakat başarısız bir prefill'i fazla tahmin edebilir.
Yükle birlikte yalnız completion tarafının bozulması bu açıklamayla uyumludur,
ancak tek nedeni izole etmez.

### 4.2 Önceki tek koşu doğrudan karşılaştırılabilir değil

| | paper (n=1) | speedup=5 | speedup=15 |
|---|---|---|---|
| dispatch korelasyon | 0.932 | 0.972 | 0.980 |
| completion korelasyon | 0.749 | 0.941 | 0.806 |
| **fark** | **0.183** | 0.032 | **0.175** |

Paper'ın önceki tek koşusunun yük koşulu kaydedilmemiş ve eski dispatch-bias
işareti yeni protokolde yeniden üretilmemiştir. Bu nedenle eski satır yeni
sweep'le eşdeğer bir hücre sayılmaz. Yeni speedup=5 koşusunda korelasyon farkı
0.031, speedup=15'te ise 0.174'tür.

### 4.3 Teoriyi güçlendiren detay

Paper dispatch bias'ını −13.6 diyor, yani dispatch *az* tahmin ediyor — ki bu
kendi açıklamasıyla çelişiyordu. Dispatch-time kayıt tanımı gereği **iyimser**:
prefill olmadan bloğu cache'te sayıyorsun, *fazla* tahmin etmeli. Yeni ölçüm
her iki yükte de pozitif (+31.3 ve +23.7). Ayrıca dispatch'in az-tahmin oranı
altı koşunun tamamında %0.0; bu, gözlenen koşuların sonucudur ve yapısal bir
imkânsızlık iddiası değildir.

---

## 5. Madde O — CUSUM parametrelerinin gerçek trafikte kalibrasyonu

`CUSUM_K`, `CUSUM_H`, `DRIFT_LAM` şimdiye kadar yalnızca
`adaptive_drift_model.__main__`'daki sentetik sıçrama senaryosuyla
doğrulanmıştı. O test mekanizmanın tetiklendiğini gösteriyor; bu değerlerin
**bu iş yüküne uygun olduğunu** göstermiyor.

GPU gerekmedi: detektör overlap dizisinin saf bir fonksiyonu.

**Kritik nokta:** router'ın gördüğü dizi, her istek için **varış sırasında**,
aynı session'ın bir önceki isteğiyle Jaccard. Bu,
`overlap_measurement.session_adjacent_pairs`'in döndürdüğü şey **değil** — o
session'a göre gruplayıp veriyor, ki özet istatistik için doğru, davranışı
tamamen sıralı olan bir detektör için yanlış. Gruplu sıra üzerinde kalibre etmek
makul görünen ama hiçbir yere transfer olmayan sayılar üretirdi.

Veri: `trace.jsonl` (stable, 2170 gözlem) ve `--drift 0.1` ile üretilmiş
`trace_drift01.jsonl` (2139 gözlem). İki trace arasındaki tek fark drift.

### 5.1 İlk koşu yanlış top_k'daydı ve sonucu tersine çeviriyordu

`--top-k` varsayılanı 3, deneyler k=10'da. k=3'te detektör bazı ayarlarda
**kararlı trafikte drift'ten daha çok** alarm veriyor (0.20/0.08/0.30: stable
68.2, drift 28.5 → 0.42×). Sinyal ters. k=10'da yön düzeliyor.

### 5.2 Kalibrasyon öncesi varsayılanlar bozuktu

top_k=10, `lam=0.1`, `k=0.03` sabit:

| `CUSUM_H` | nominal-trace alarmı /1000 | ayrım |
|---|---|---|
| **0.20** (önceki) | 108.76 (isteklerin %10.9'u) | 1.52× |
| **0.30** | 26.73 (%2.7) | **2.20×** |

`h=0.30` **iki eksende birden** daha iyi: nominal trace üzerinde 4 kat az alarm
ve daha iyi ayrım. Ancak nominal trace'te etiketli değişim noktaları olmadığı
için bunlar kesin "yanlış pozitif" değil, yalnızca muhafazakâr bir vekil
metriktir. Dahası, mevcut kodda CUSUM alarmı routing beta/delta değerlerini
değiştirmez; bu kalibrasyon routing kazancını değil, tanısal telemetrinin alarm
davranışını iyileştirir.

### 5.3 `D_TARGET` de yanlış — ve top_k'ya bağlı

| | ölçülen `d_ref` |
|---|---|
| kalibrasyon öncesi `config.py` değeri | 0.529 |
| trace.jsonl, top_k=3 | 0.478 |
| trace.jsonl, **top_k=10** | **0.322** |

Detektör daha trafiği görmeden yanlış referanstan başlıyor.

### 5.4 Kalibre sonrası bile detektör zayıf

Takas eğrisi (en iyi ayrım, her yanlış-alarm bütçesinde):

| bütçe /1000 | en iyi ayrım | lam | k | h |
|---|---|---|---|---|
| 5 | 1.35× | 0.20 | 0.08 | 0.30 |
| 10 | 1.52× | 0.20 | 0.03 | 0.30 |
| 25 | 1.71× | 0.05 | 0.08 | 0.30 |
| 50 | **2.20×** | 0.10 | 0.03 | 0.30 |

Nominal-trace alarm bütçesini 1/1000'in altına çekersen ayrım 1.35×'e düşüyor — 2170 gözlemde
3 alarma karşı 4 alarm, yani gürültü.

**Denenip reddedilen bir düzeltme:** gürültünün kaynağı olarak CUSUM'a ham
Jaccard yerine yumuşatılmış EWMA beslemek denendi. İşe yaramadı — detektörü
keskinleştirmek yerine tamamen susturuyor (2000 gözlemde 1-4 alarm, iki trace'te
de, ayrım yok). `--feed-ewma` bayrağıyla tekrarlanabilir bırakıldı.

### 5.5 Önerilen değerler

```
ROUTER_CUSUM_H=0.30        # 0.20'den
ROUTER_D_TARGET=0.322      # 0.529'dan (top_k=10'da ölçüldü)
ROUTER_CUSUM_K=0.03        # değişmiyor
ROUTER_DRIFT_LAM=0.1       # değişmiyor
```

Dürüst sınır: kalibrasyondan sonra bile ayrım gücü 2.20× ve bu nominal trace'te
%2.7 alarm karşılığında geliyor. Bu yalnızca detektör telemetrisini değerlendirir;
alarm mevcut routing kararını kontrol etmediği için drift-adaptif routing kazancı
gösterilmiş değildir.

---

## 6. Paper'ın mevcut iddialarını değiştiren üç şey

1. **Faz 2'nin çerçevesi.** `score_quality.py`'nin docstring'i ve ablation'ın
   kurgusu "canonical ordering cache hit rate'i yükseltti" varsayımına dayanıyor.
   k=10'da canonical, relevance sırasının hem cache hem TTFT sonucundan geri;
   `pinned_prefix` ise bu iki ölçütte daha da kötü (§3.3). Soru "hangi şema"
   değil, "şema cache geçmişini kullanıyor mu".

2. **Bölüm VI-E'nin yük koşulu.** Güncel paper iki yük seviyesini de veriyor:
   speedup=5'te korelasyon farkı 0.031, speedup=15'te 0.174 (§4.2).

3. **Bölüm IV-C'nin kalibrasyon tablosu.** Güncel paper `D_TARGET=0.322` ve
   `CUSUM_H=0.30` değerlerini, ölçüm kaynağı ve kapsam sınırlarıyla veriyor.

---

## 7. Bilinen sınırlar

- **`TRACKER_CAPACITY` kalibre değil.** Yeni ordering/timing koşuları 50000
  varsayılanında sabit tutuldu. Kollar arasında doğrudan bir ayar değişikliği
  yoktur; ancak kapasite-politika etkileşimi dışlanamaz ve mutlak sonuçlar
  kalibre deployment iddiası taşımaz. Paper bunu açıkça belirtiyor; Bölüm VI-B
  bu sabitin başka bir ablation'ı ters çevirebildiğini gösteriyor.
- Ordering ve timing sweep'leri tek trace şekli (`trace_hot`), tek korpus,
  300 istek ve 2 worker ile sınırlı.
- Sıralama sonuçları yalnızca **k=10** için geçerli; k=3'te hiçbir kol
  ayrışmıyor. İddia "k=10'da" diye nitelendirilmeli.
- `pinned_prefix` tek bir tasarım varyantı (session geçmişi). Global hot-set
  veya canonical-baş/relevance-kuyruk varyantları denenmedi.
- CUSUM kalibrasyonu tek bir drift seviyesinde (`--drift 0.1`). İkinci bir
  seviye, alarm davranışının drift şiddetiyle nasıl değiştiğini sınamak için
  gereklidir.

---

## 8. Yazılan kod

| Dosya | Ne |
|---|---|
| `bench/replay.py` | `--order greedy` ve `--order pinned_prefix` kolları, greedy bayrakları, teşhis metrikleri |
| `bench/score_quality.py` | 2 dosya yerine N dosya (3. kol sessizce düşüyordu) |
| `bench/sweep_ordering.py` | Sıralama ablation'ı × N tekrar, restart koreografisi, ayrışma hükmü |
| `bench/sweep_timing.py` | dispatch/completion × N tekrar, iki yükte etkileşim testi |
| `bench/calibrate_cusum.py` | CUSUM grid taraması, ayrım oranı, takas tablosu |
| `bench/check_lag.py` | Koşuların programını tutup tutmadığı |
| `bench/validate_tracker.py` | `compute()`/`report()` ayrımı (sweep'in toplayabilmesi için) |
| `smoke_test_greedy_order.py` | greedy sıralamasının GPU'suz doğrulaması |
| `smoke_test_pinned_prefix.py` | pinned_prefix'in GPU'suz doğrulaması |
| `SETUP.md` | Başka makinede kurulum, veri parmak izleri, kalibrasyon uyarıları |

### 8.1 Yol boyunca bulunup düzeltilen hatalar

- `sweep_ordering`: alt süreç `cwd=bench/` ile başlatıldığı için proje kökünden
  verilen görece yollar bulunamıyordu; ayrıca çocuğun hata çıktısı PIPE'a
  yutuluyordu, hata çıplak bir `CalledProcessError` olarak görünüyordu.
- Her iki sweep: `replay.py`'nin schedule-lag uyarısı yutuluyordu — tam da
  uyarının tetiklendiği yüklerde. N'in tüm iddiası yük eksenine dayandığı için
  kritikti.
- `calibrate_cusum`: etiket doğrulaması dakikalarca süren embedding'den **sonra**
  yapılıyordu; ayrıca cache varken trace dosyasının varlığı zorunlu tutuluyordu,
  ki cache'in amacı tam olarak buydu.
- `calibrate_cusum`'un ilk tavsiye kuralı, iki trace'te de sessiz kalan bir ayarı
  öneriyordu — sessizlik tespit sanılıyordu. Ayrım oranı eklendi.

---

## 9. Sırada ne var

**Tier 1 bitti** (A, B, C, D, N, O). Sonuçlar `paper/paper.tex`'e işlendi;
`config.py` varsayılanları `CUSUM_H=0.30` ve `D_TARGET=0.322` olarak güncellendi.

Yapılmadı:

- Merkezi sistem karşılaştırması: `per_worker_tree` ile global worker-kör
  `greedy`, eşlenmiş prompt serileştirmesi ve uçtan uca zamanlayıcıyla henüz
  karşılaştırılmadı.
- Tier 3: J (Load(w) formel modelleme), K (DecodeTime belirsizliği),
  L (k=10 bulgusu — artık §2.2'de ölçülmüş hâli var), M (mean_kv_usage EWMA).
