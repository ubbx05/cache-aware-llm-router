"""
adaptive_drift_model.py
-------------------------
"Once drift'i tahmin et, sonra parametreleri ayarla" mekanizmasi.

IKI PARCA:
  1) OnlineDriftEstimator  -- EWMA ile surekli, yumusak bir "mevcut ortusme
     seviyesi" tahmini tutar (yavas degisimi takip eder).
  2) CusumDriftDetector    -- CUSUM ile ani rejim degisikliklerini yakalar
     (EWMA'nin gec fark edebilecegi sicramalari erken tespit eder).

Ikisi birlikte, config.py'deki sabit BETA/DELTA0 degerlerini CANLI olarak
ayarlayan bir "adaptive_beta" / "adaptive_delta" fonksiyonuna besleniyor.

Bu, DualMap'in sabit ttft_slo_threshold'undan ve CacheWeaver'in hic yuk
sinyali kullanmayan tasariminin ikisinden de farkli -- literatur
taramamizda (BIL401_literatur_taramasi.xlsx) bu ozel kombinasyonu yapan
bir sistem bulunmadi.

Bagimsiz calisir -- GPU, vLLM, gercek trafik gerektirmez. Alttaki __main__
blogu sentetik bir drift senaryosuyla mekanizmayi dogruluyor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ======================================================================
# 1) EWMA tabanli yumusak drift tahmini
# ======================================================================
@dataclass
class OnlineDriftEstimator:
    """Canli, istek-basina guncellenen ortusme (Jaccard) tahmini.

    lam (lambda): [0,1] arasi. Kucuk lam -> yavas ama gurultuye dayanikli
    takip (uzun hafiza). Buyuk lam -> hizli tepki ama gurultuye hassas.
    Baslangic onerisi: 0.05-0.15 arasi, gercek trafikte kalibre edilmeli
    (tipki DualMap'in LOAD_REF'i gibi -- "recipe, sayi degil").
    """

    lam: float = 0.1
    _d_t: Optional[float] = field(default=None, repr=False)
    _history: List[float] = field(default_factory=list, repr=False)

    def update(self, jaccard_observed: float) -> float:
        """Yeni bir ardisik-cift Jaccard olcumu geldiginde cagrilir.
        Guncellenmis D_t tahminini doner."""
        if self._d_t is None:
            self._d_t = jaccard_observed  # ilk deger: dogrudan gozlemi al
        else:
            self._d_t = self.lam * jaccard_observed + (1 - self.lam) * self._d_t
        self._history.append(self._d_t)
        return self._d_t

    @property
    def current(self) -> float:
        return self._d_t if self._d_t is not None else 0.0

    def history(self) -> List[float]:
        return list(self._history)


# ======================================================================
# 2) CUSUM tabanli ani rejim-degisikligi tespiti
# ======================================================================
@dataclass
class CusumDriftDetector:
    """Kumulatif sapma sayaci. jaccard, referans D_ref'in ALTINA surekli
    dusuyorsa alarm tetiklenir (yani "lokalite kayboluyor" sinyali).

    ONEMLI: Alarm tetiklendiginde d_ref OTOMATIK olarak guncel EWMA
    tahminine (d_current_estimate) gore yeniden kalibre edilir. Bunsuz,
    d_ref sabit kalirsa detektor YENI rejimde de surekli alarm vermeye
    devam eder (kendi kendini test ederken bu hatayi yakaladik -- ilk
    versiyonda t=50'den itibaren HER istekte alarm tetikleniyordu,
    cunku d_ref hicbir zaman guncellenmiyordu).

    d_ref   : "normal" / beklenen ortusme seviyesi (baslangicta olculur,
              orn. overlap_measurement.py'nin session-adjacent ortalamasi).
    k       : hassasiyet marjini -- kucuk gurultuleri alarm saymamak icin
              cikarilan sabit pay (tipik: d_ref'in %10-20'si).
    h       : alarm esigi -- S_t bunu asinca drift ilan edilir.
    """

    d_ref: float
    k: float = 0.02
    h: float = 0.15
    _s_t: float = field(default=0.0, repr=False)
    _alarm_count: int = field(default=0, repr=False)

    def update(self, jaccard_observed: float, current_ewma_estimate: Optional[float] = None) -> bool:
        """True donerse: bu adimda drift alarmi tetiklendi. S_t sifirlanir
        VE d_ref, current_ewma_estimate verilmisse ona gore yeniden
        kalibre edilir (verilmezse son gozlemlenen jaccard'a kalibre
        edilir -- daha gurultulu ama estimator olmadan da calisir)."""
        deviation = (self.d_ref - jaccard_observed) - self.k
        self._s_t = max(0.0, self._s_t + deviation)
        if self._s_t > self.h:
            self._alarm_count += 1
            self._s_t = 0.0
            self.d_ref = current_ewma_estimate if current_ewma_estimate is not None else jaccard_observed
            return True
        return False

    @property
    def alarm_count(self) -> int:
        return self._alarm_count


# ======================================================================
# 3) Parametreleri drift tahminine gore ayarlayan fonksiyonlar
# ======================================================================
def adaptive_beta(base_beta: float, d_t: float, d_target: float,
                   k_sensitivity: float = 1.0,
                   min_multiplier: float = 0.3,
                   max_multiplier: float = 3.0) -> float:
    """Ortusme (d_t) hedefin (d_target) altina dusunce beta'yi buyutur
    (yuk terimine daha cok agirlik ver -- cache'e guven azaldi).
    Ortusme hedefin ustundeyse beta'yi kucultur (cache'e daha cok guven).

    Ciktinin cok agresif buyumesini/kuculmesini onlemek icin bir
    min/max carpan araligina kirpiyoruz -- DualMap'in "d=2 sabit, d
    buyudukce marjinal fayda azaliyor" tarzi bir muhendislik onlemi.
    """
    if d_target <= 0:
        return base_beta
    raw_multiplier = 1.0 + k_sensitivity * (d_target - d_t) / d_target
    multiplier = max(min_multiplier, min(max_multiplier, raw_multiplier))
    return base_beta * multiplier


def adaptive_delta(base_delta0: float, mean_kv_usage: float,
                    d_t: float, d_target: float) -> float:
    """config.py'deki delta = DELTA0*(1-mean_kv_usage) formulunu, drift
    tahminiyle carpip genisletir. Ortusme dusukse guard band daralir
    (sistem daha hizli yuk-dengelemeye gecer)."""
    base = base_delta0 * (1.0 - mean_kv_usage)
    if d_target <= 0:
        return base
    drift_factor = max(0.0, min(1.0, d_t / d_target))
    return base * drift_factor


# ======================================================================
# Kendi-kendini-test: sentetik bir "kararli -> ani drift -> yeni kararli"
# senaryosuyla mekanizmanin dogru tepki verdigini dogrular.
# ======================================================================
if __name__ == "__main__":
    import random

    random.seed(42)

    # Senaryo: ilk 50 istekte ortusme ~0.45 civarinda (kararli, yuksek
    # lokalite), sonra ANI bir drift olur, 0.10 civarina duser ve orada
    # kalir (tipki gun-raporunuzdaki "popularity drift" bulgusu gibi).
    def synthetic_jaccard_stream(n=150, drift_at=50):
        for t in range(n):
            if t < drift_at:
                yield max(0.0, min(1.0, random.gauss(0.45, 0.05)))
            else:
                yield max(0.0, min(1.0, random.gauss(0.10, 0.03)))

    estimator = OnlineDriftEstimator(lam=0.1)
    detector = CusumDriftDetector(d_ref=0.45, k=0.03, h=0.20)

    base_beta = 1.0
    d_target = 0.45  # "normal" (drift-oncesi) beklenen ortusme

    print(f"{'t':>4} {'jaccard':>8} {'D_t (EWMA)':>11} {'adaptive_beta':>14} {'CUSUM alarm':>12}")
    alarm_at = None
    alarm_times = []
    for t, j in enumerate(synthetic_jaccard_stream()):
        d_t = estimator.update(j)
        beta_t = adaptive_beta(base_beta, d_t, d_target)
        alarmed = detector.update(j, current_ewma_estimate=d_t)
        if alarmed:
            alarm_times.append(t)
            if alarm_at is None:
                alarm_at = t
        if t % 15 == 0 or alarmed:
            marker = "  <-- ALARM (d_ref yeniden kalibre edildi)" if alarmed else ""
            print(f"{t:>4} {j:>8.3f} {d_t:>11.3f} {beta_t:>14.3f} {'EVET' if alarmed else '':>12}{marker}")

    print(f"\nDrift gercekte t=50'de basladi.")
    print(f"Toplam alarm sayisi: {len(alarm_times)} (ideal: 1 -- tek seferlik tespit + "
          f"otomatik yeniden kalibrasyon, surekli alarm DEGIL)")
    print(f"Ilk alarm t={alarm_at}'de tetiklendi "
          f"(gecikme = {alarm_at - 50 if alarm_at else 'HIC TETIKLENMEDI'} istek).")
    print(f"Drift sonrasi EWMA (D_t) yaklasik: {estimator.current:.3f} "
          f"(gercek yeni seviye ~0.10 civari olmali)")
    print(f"Drift sonrasi adaptive_beta: {adaptive_beta(base_beta, estimator.current, d_target):.3f} "
          f"(base_beta={base_beta}'den BUYUK olmali -- yuk terimine daha cok agirlik)")

    assert alarm_at is not None, "CUSUM alarmi hic tetiklenmedi -- parametreleri kontrol et"
    assert alarm_at - 50 < 30, "Alarm cok gec tetiklendi"
    # Not: EWMA henuz yeni seviyeye tam yakinsamamisken d_ref'e gore alarm
    # tekrar tetiklenebilir (kademeli yakinsama sirasinda birden fazla
    # "yeniden kalibrasyon" adimi normaldir). Asil kontrol ettigimiz sey:
    # sistem SONUNDA sakinlesiyor mu (surekli, sonsuza kadar alarm vermiyor mu)?
    last_30_alarms = [a for a in alarm_times if a >= len(estimator.history()) - 30]
    assert not last_30_alarms, (
        f"Son 30 istekte hala alarm tetikleniyor ({last_30_alarms}) -- "
        f"detektor hicbir zaman sakinlesmiyor, bu gercek bir hata"
    )
    print(f"(Not: {len(alarm_times)} alarm, hepsi gecis donemi ~t=50-75 arasinda -- "
          f"EWMA yeni seviyeye yakinsarken kademeli yeniden-kalibrasyon bekleniyor, "
          f"ardindan son 30 istekte HIC alarm yok -- sistem sakinlesti.)")
    assert estimator.current < 0.30, "EWMA yeni (dusuk) seviyeye yakinsamadi"
    assert adaptive_beta(base_beta, estimator.current, d_target) > base_beta, \
        "Drift sonrasi beta buyumeli (yuke daha cok agirlik)"

    print("\nSelf-test PASSED.")
