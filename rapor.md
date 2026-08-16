# Çalışma Raporu — 16 Ağustos 2026

Bu rapor, 15 Ağustos raporundaki tek-koşu ve kalibrasyonsuz sonuçları yeni
tekrarlı deneylerle günceller. Ana sonuç artık `per_worker_tree` stratejisinin
kalibre edilmiş iki-worker karşılaştırmasıdır. Aynı zamanda dört önceki iddia
geri çekilmiş veya daraltılmıştır.

## 1. Sonuçların kısa özeti

- Beş kollu ana strateji deneyi 800 istek, top-k=10, speedup=8, üç tekrar ve
  `TRACKER_CAPACITY=5840` ile tamamlandı.
- `per_worker_tree`, `cache_aware`'a göre cached-token fraction'da +9.7 yüzde
  puan kazandı (76.1% vs 66.4%).
- Bu fark tekrarlı 12x yükte +12.0 puana çıktı; düşük sentetik lokalitede
  +4.2 puana indi. Kazanç lokaliteye bağlıdır.
- `cache_aware` ile `cacheweaver_dualmap` ve adaptif ile sabit politika,
  önceden belirlenen yayılım kuralıyla ayrışmadı.
- Kalibre ordering tekrarı eski sıralamayı korudu (her kolda değişim en fazla
  0.2 puan); kalibre beta matrisi ise eski çapraz deseni yeniden üretmedi.
- Uvicorn çıktısını okunmayan pipe'a yazan sweep altyapısı router'ı
  kilitliyordu. Çıktı artık koşu-bazlı log dosyasına yönlendiriliyor.

“Ayrıştı” ifadesi formal anlamlılık testi değildir. Kullanılan sezgisel kural:

> |iki kolun ortalama farkı| > 2 × en büyük koşular-arası standart sapma.

Üç tekrar düşük örneklem sayısıdır; kuralı geçmeyen sonuçlar “eşit” değil,
“bu tekrar düzeyinde ayrışmadı” diye yorumlanır.

## 2. Ortak protokol

| Özellik | Değer |
|---|---|
| Donanım | 2 × RTX 4090; biri lokal, biri Tailscale |
| Model | Qwen2.5-7B-Instruct, gpu-memory-utilization 0.85 |
| Tokenizer | Hugging Face (`hf`) |
| Kalibre kapasite | 5,840 blok = min(93,440, 93,840) / 16 |
| Soğuk başlangıç | Her kol öncesi iki vLLM ve router yeniden başlatıldı |
| Koşu sırası | Repeat-major ve döndürülmüş kol sırası |
| Ana sweep | 800 istek, top-k=10, speedup=8, her kol 3 tekrar |
| Yük/lokalite | 4x/8x/12x ve yüksek/düşük sentetik lokalite |
| Zamanlama kontrolü | Ana 15 koşuda schedule-lag p99 ≤ 2 ms |
| Tracker doğrulaması | Pearson 1.000, MAE 0.6 token, az-tahmin 0.0% |

Ana sweep'in speedup=8 noktası düşük ve aşırı yük uçları kontrol edilerek
seçildi. Top-k=3, speedup=5'te kollar 0.6 puanda toplanırken speedup=20'de
TTFT p50 8.8 saniyeye ulaşıyor ve bütün stratejiler kuyrukta eşitleniyor.

## 3. Kalibre ana strateji karşılaştırması

Ana tablodaki cached-token fraction, her koşuda toplam cached token / toplam
prompt token olarak hesaplanır. ± değerleri üç koşunun örnek standart
sapmasıdır.

| Strateji | Cached-token fraction | TTFT p50 | TTFT p99 | Load CV | Contains gold |
|---|---:|---:|---:|---:|---:|
| least_loaded | 55.5% ±0.8 | 0.785 s ±0.126 | 3.368 s ±0.739 | 0.008 | 77.8% ±0.3 |
| round_robin | 56.2% ±0.2 | 0.742 s ±0.063 | 2.564 s ±0.082 | 0.000 | 77.7% ±0.3 |
| cacheweaver_dualmap | 66.1% ±0.6 | 0.484 s ±0.303 | 2.633 s ±0.679 | 0.358 | 77.7% ±0.1 |
| cache_aware | 66.4% ±0.4 | 0.278 s ±0.028 | 2.320 s ±0.645 | 0.260 | 77.5% ±0.1 |
| **per_worker_tree** | **76.1% ±0.06** | **0.147 s ±0.001*** | **1.184 s ±0.190*** | 0.196 | 77.6% ±0.15 |

\* `per_worker_tree` TTFT'si yalnız completion fazını ölçüyor. İlk
`/router/decide_order` çağrısı, prompt kurma süresi ve ilk HTTP round-trip dahil
değildir; tek-aşamalı kollarla uçtan uca gecikme kıyası sayılamaz.

Savunulabilir ayrışmalar:

- `cache_aware` − `least_loaded`: +10.9 puan; eşik 1.6 puan.
- `per_worker_tree` − `cache_aware`: +9.7 puan; eşik 0.9 puan.
- `cache_aware` ile `cacheweaver_dualmap` hit, p50 ve p99'da ayrışmıyor.
- `per_worker_tree` ile `cache_aware` p99 farkı kuralı az farkla geçmiyor.
- Beş contains-gold sonucu 77.5–77.8% bandında; kalite ayrışmıyor.

İkinci karşılaştırılabilirlik sınırı prompt şablonudur. Raporlanan iki-aşamalı
koşuda chunk ayıracı düz kola göre 5 ek token taşır. Dokuz ayıraç × 5 token ×
800 istek yaklaşık 36,000 token eder; gözlenen payda farkı 35,997 tokendir
(%1.9). Yeni harness aynı ayıracı kullanıyor fakat yukarıdaki sayılar eski
şablonla ölçüldüğü için temiz bir tekrar gereklidir.

## 4. per_worker_tree: yük ve lokalite

### 4.1 Yük ekseni

| Yük | Tekrar | CA cached | PWT cached | Fark | CA p50 | PWT p50* | Oran |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4x | 1 | 66.0% | 76.2% | +10.2 puan | 0.094 s | 0.072 s | 1.3x |
| 8x | 3 | 66.4% ±0.45 | 76.1% ±0.06 | +9.7 puan | 0.278 s | 0.147 s | 1.9x |
| 12x | 3 | 64.6% ±0.50 | 76.6% ±0.31 | +12.0 puan | 2.411 s ±0.520 | 0.226 s ±0.014 | 10.7x |

\* Completion-fazı TTFT; karar çağrısı hariçtir.

8x ve 12x tekrarlı noktalarında cache farkı 9.7–12.0 puan bandındadır.
4x yalnız tek koşudur; “yükten bağımsız” kanıtı olarak değil, yönü destekleyen
keşif noktası olarak tutulur. 10.7x oran, uçtan uca hızlanma iddiası değildir.
Gözlem daha az prefill işinin kuyruk büyümesini azalttığı hipoteziyle uyumludur,
ancak nedensel mekanizma ayrı ölçülmedi.

### 4.2 Lokalite ekseni

| Lokalite | CA cached | PWT cached | Fark | Contains gold (CA → PWT) |
|---|---:|---:|---:|---:|
| Yüksek (Zipf 1.5, session 4) | 64.6% ±0.5 | 76.6% ±0.3 | +12.0 puan | 77.8% → 77.8% |
| Düşük (Zipf 0.7, session 2) | 53.1% ±0.1 | 57.3% ±0.3 | +4.2 puan | 64.0% ±0.2 → 65.2% ±0.1 |

Düşük lokalitede avantaj yaklaşık üçte bire iner ve TTFT p99 ayrışmaz.
+1.2 puanlık contains-gold farkı üç koşuda aynı yöndedir fakat soru-bazlı
eşleştirilmiş test yapılmadan kalite artışı diye sunulmaz. Her iki trace de
TQuAD üzerinde sentetik üretilmiştir.

## 5. Tekrarlı negatif ve kontrol sonuçları

| Karşılaştırma | Trace | Sol kol | Sağ kol | Hüküm |
|---|---|---:|---:|---|
| adaptive / fixed cache-aware | hot | 66.5% ±0.5 | 66.4% ±0.5 | ayrışmıyor |
| adaptive / fixed cache-aware | drift 0.1 | 43.8% ±0.7 | 44.0% ±0.5 | ayrışmıyor |
| semantic-labelled / PWT | hot | 76.1% ±0.40 | 76.1% ±0.06 | protokol no-op'u |

Adaptif kolda beta gerçekten hareket eder: 405 farklı değer, 0.65–1.28 aralığı.
Buna rağmen sonuç değişmez. CUSUM alarmı routing tarafından kullanılmadığı
için bu null sonucun nedeni olarak gösterilemez; deney nedeni izole etmiyor.

Semantik sonuç da semantik sinyalin faydasızlığını kanıtlamaz.
`SEMANTIC_TOP_K=2`, iki-worker havuzunun tamamına eşittir; aday elenmez. Yeni
kod query text'i iki-aşamalı sözleşmeden geçirir, fakat gerçek pruning için
N > semantic top-k veya ayrı top-k=1 deneyi gerekir.

## 6. Kalibrasyon replikasyonları

### 6.1 Ordering sonucu kalibrasyondan sağ çıktı

Bu tabloda metrik tarihsel scorer ile istek-başına cached fraction
ortalamasıdır; ana strateji tablosunun run-level agregatıyla aynı metrik
değildir.

| Kol | 50,000 blok | 5,840 blok | Değişim |
|---|---:|---:|---:|
| greedy | 73.1% ±0.2 | 73.2% ±0.3 | +0.1 puan |
| relevance | 67.7% ±0.1 | 67.6% ±0.4 | −0.1 puan |
| canonical | 65.0% ±0.5 | 64.9% ±0.2 | −0.1 puan |
| pinned_prefix | 62.9% ±0.2 | 62.7% ±0.4 | −0.2 puan |

Sıralama değişmedi ve her ortalama en fazla 0.2 puan oynadı. Kalibre k=10
sonucunda greedy, relevance'ın 5.6; canonical'ın 8.3 puan üzerindedir.
k=3 tek koşusunda yayılım yalnız 2.4 puandır. Bu nedenle kaldıraç top-k ile
açılır. `pinned_prefix`'in başarısızlığının “cross-session paylaşım bozuldu”
açıklaması hipotezdir; worker binding ölçülmemiştir.

### 6.2 Kalibre beta matrisi eski deseni yeniden üretmedi

| Lokalite | Yük | beta=1 p50 | beta=0 p50 | Gözlenen yön |
|---|---:|---:|---:|---|
| Yüksek | 30x | 0.175 s | 0.484 s | beta=1 |
| Yüksek | 35x | 1.323 s | 2.734 s | beta=1 |
| Düşük | 30x | 0.151 s | 0.147 s | ayrışmıyor |
| Düşük | 35x | 0.442 s | 0.940 s | beta=1 |

Beta=1 yük-duyarlı cache routing, beta=0 saf cache routing'dir. Eski
kalibrasyonsuz matriste kazanan çapraz köşelerde değişiyordu; yeni gözlemde
beta=1 üç hücrede daha düşük, dördüncü fark 4 ms'dir. Hücrelerin tekrar
yayılımları eklerde bulunmadığından “üç anlamlı galibiyet” denmez. Hit oranları
beta'dan neredeyse etkilenmez; gözlenen fark kuyruk davranışındadır.

## 7. Korunan önceki bulgular

### Dispatch ve completion bookkeeping

Üç tekrar ve iki yükte dispatch korelasyonu yaklaşık 0.98'de kalırken
completion korelasyonu 0.941'den 0.806'ya, sıra uyumu 92.5%'ten 79.7%'ye
iner; az-tahmin 6.2%'den 29.6%'ya çıkar. Bu sonuç 50,000 blokta iki kol için
aynı kapasiteyle ölçülmüştür. Yük etkileşimini destekler, kalibre mutlak hata
iddiası değildir.

### CUSUM

H=0.20 → 0.30 değişimi nominal alarm oranını 108.76'dan 26.73/1000'e indirir
ve drift/nominal ayrımını 1.52x'ten 2.20x'e çıkarır. Nominal trace etiketli
değişim noktası içermediğinden bunlar ground-truth false positive değildir.
CUSUM routing kontrolü değil, telemetridir.

## 8. Kod ve sweep altyapısı

- `bench/sweep_strategy.py`: beş çekirdek stratejiyi varsayılan olarak,
  deneysel kolları isteğe bağlı çalıştırır; üç tekrar, döndürülmüş sıra,
  p50/p99, load CV, kalite, config özeti ve koşu-bazlı log üretir.
- `bench/sweep_beta.py`: yüksek/düşük trace × 30/35 yük × beta 1/0 matrisini
  üretir; kapasiteyi açıkça ister ve nötr sonuç tablosu yazar.
- `bench/sweep_overlap_load.start_router` ve onu kullanan runner'lar artık
  uzun ömürlü, okunmayan `PIPE` kullanmaz; Uvicorn çıktısı dosyaya gider.
- İki-aşamalı replay artık query text'i taşır, düz kolla aynı chunk ayıracını
  kullanır ve completion-fazı alanlarını korurken karar süresi ile uçtan uca
  TTFT/total alanlarını ayrıca kaydeder.
- Ortak scorer p99 raporlar, cached token'ı toplam token üzerinden agregatlar
  ve kullanılmayan worker'ı sıfır sayımla load CV'ye dahil eder.

Pipe hatası nedeniyle yazılan eski “iki-aşamalı protokol event loop'u
doyuruyor” teşhisi geri çekilmiştir. Kilitlenme router stratejisinden değil,
Uvicorn access log buffer'ından kaynaklanmıştır.

## 9. Sınırlılıklar ve sonraki işler

1. 4x yük noktasını üç kez tekrarla; ana karşılaştırmayı eşit prompt şablonu ve
   yeni uçtan uca timer ile yeniden ölç.
2. PWT reorder depth, order-changed oranı, worker-bazlı ağaç doluluğu ve karar
   overhead'ini kaydet; “neden kazanıyor?” sorusunu ölç.
3. ShareGPT/LMSYS gibi dışsal bir trace üzerinde lokalite sonucunu doğrula.
4. Semantik pruning ve DualMap-benzeri tasarımı N>2 worker ile test et.
5. Yeni ham JSONL/summary dosyalarını, ortam fingerprint'ini ve soru-bazlı
   kalite sonuçlarını artifact'e ekle.

Yeni tabloların ham JSONL dosyaları mevcut repo snapshot'ında değildir ve
`tum_kosular.csv` bu yeni koşuları içermeyen eski 50-koşuluk dosyadır. Bu
nedenle sayıların bağımsız yeniden hesabı için ham çıktılar ayrıca eklenmeli
veya iki-worker koşuları yeniden çalıştırılmalıdır.
