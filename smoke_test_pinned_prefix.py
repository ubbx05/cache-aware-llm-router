"""Offline checks for replay.py's `--order pinned_prefix` arm (GPU/router-suz).

pinned_prefix'in tek iddiasi su: onceki turda servis edilen ve bu turda tekrar
gelen chunk'lari, ONCEKI SIRALARIYLA one al; kalani relevance sirasinda birak.
Bu dosyanin test ettigi sey, o cumlenin her parcasinin gercekten dogru olmasi.

Canli kosuda sessizce yanlis olabilecekler:

1. Pinlenen kismin ONCEKI sirayi degil, bu turun relevance sirasini kullanmasi
   -- ciktilar makul gorunur ama motor eslesecek bir onek bulamaz, yani arm
   hicbir sey kazanmadan relevance gibi davranir.
2. Kuyrugun relevance sirasini kaybetmesi -- kalite gerekcesi cokmus olur.
3. Chunk kumesinin degismesi (dusme/tekrar) -- modelin gordugu icerik degisir,
   o zaman bu bir SIRALAMA ablation'i olmaktan cikar.
4. Session'lar arasi sizinti -- A'nin gecmisi B'yi siralarsa olculen sey
   session locality olmaz.

Kosum:
    python3 smoke_test_pinned_prefix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bench"))

from replay import pinned_prefix_order  # noqa: E402


def main() -> None:
    # --- 1. gecmis yoksa: relevance'a BIREBIR esit ----------------------
    rel = ["c7", "c2", "c9", "c4"]
    ordered, n = pinned_prefix_order([], rel)
    assert ordered == rel, f"gecmis yokken sira degismemeli: {ordered}"
    assert n == 0
    print(f"1. gecmis yok    : {ordered}  pinned=0  (== relevance)  OK")

    # --- 2. kismi ortusme: pinlenen kisim ONCEKI sirasiyla --------------
    # Onceki tur [c2, c9, c5] sirasiyla servis edildi. Bu tur {c7,c2,c9,c4}
    # geldi. Kesisim {c2, c9} -- ve onceki sirada c2, c9 seklinde.
    prev = ["c2", "c9", "c5"]
    ordered, n = pinned_prefix_order(prev, rel)
    assert ordered == ["c2", "c9", "c7", "c4"], f"beklenmedik sira: {ordered}"
    assert n == 2
    print(f"2. kismi ortusme : {ordered}  pinned=2  OK")

    # Kuyruk relevance sirasini korumali: c7, c4 (c7 daha alakali)
    assert ordered[2:] == ["c7", "c4"], "kuyruk relevance sirasini kaybetti"
    # Kume degismemeli
    assert set(ordered) == set(rel), "chunk kumesi degisti"
    assert len(ordered) == len(rel), "chunk tekrarlandi ya da dustu"

    # --- 3. ONCEKI sira gercekten kullaniliyor mu? ----------------------
    # Kritik test: onceki sira relevance sirasinin TERSI olsun. Kod yanlislikla
    # relevance sirasini kullaniyorsa bu test onu yakalar, 2. test yakalamaz.
    prev_rev = ["c9", "c2"]           # relevance'ta c2 once geliyordu
    ordered_rev, n_rev = pinned_prefix_order(prev_rev, rel)
    assert ordered_rev == ["c9", "c2", "c7", "c4"], f"onceki sira kullanilmadi: {ordered_rev}"
    assert ordered_rev != ordered, "onceki sira degisince cikti da degismeliydi"
    print(f"3. ters gecmis   : prev={prev_rev} -> {ordered_rev}  OK")

    # --- 4. tam ortusme: sira tamamen onceki tura esit ------------------
    ordered_full, n_full = pinned_prefix_order(["c4", "c9", "c2", "c7"], rel)
    assert ordered_full == ["c4", "c9", "c2", "c7"], f"tam pin yanlis: {ordered_full}"
    assert n_full == 4
    print(f"4. tam ortusme   : {ordered_full}  pinned=4  (prefix birebir korunur)  OK")

    # --- 5. gecmiste olup bu tur GELMEYEN chunk siralamaya girmemeli ----
    # prev'de c5 var ama bu turun retrieval'inda yok -> cikti'da olmamali.
    assert "c5" not in ordered, "gecmisten gelen ama retrieve EDILMEYEN chunk eklendi"
    print("5. hayalet chunk : prev'deki c5 cikti'ya sizmadi  OK")

    # --- 6. session izolasyonu -----------------------------------------
    # Bu, fonksiyonun degil cagiranin sorumlulugu; burada sozlesmeyi
    # dogruluyoruz: fonksiyon SADECE kendisine verilen prev'i kullanir,
    # global bir durum tutmaz. Ayni girdiyle iki cagri ayni ciktiyi verir.
    a1, _ = pinned_prefix_order(["c2"], rel)
    b1, _ = pinned_prefix_order(["c4"], rel)
    a2, _ = pinned_prefix_order(["c2"], rel)
    assert a1 == a2, "fonksiyon durum tutuyor -- session'lar birbirine sizabilir"
    assert a1 != b1, "farkli gecmisler ayni ciktiyi verdi"
    print(f"6. durumsuzluk   : prev=[c2]->{a1[0]}  prev=[c4]->{b1[0]}  tekrar kararli  OK")

    # --- 7. hicbir ortusme yoksa: yine relevance --------------------------
    ordered_none, n_none = pinned_prefix_order(["c99", "c98"], rel)
    assert ordered_none == rel and n_none == 0, f"ortusme yokken relevance olmali: {ordered_none}"
    print(f"7. ortusme yok   : {ordered_none}  pinned=0  (== relevance)  OK")

    print("\nsmoke_test_pinned_prefix PASSED.")


if __name__ == "__main__":
    main()
