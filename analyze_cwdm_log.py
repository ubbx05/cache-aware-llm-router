"""
Kullanim:
    python3 analyze_cwdm_log.py runs/cwdm_tie_log.jsonl

cacheweaver_dualmap_router.py'nin ROUTER_CWDM_TIE_LOG=... ile uretilen
jsonl'unu okuyup asagidakileri raporlar:

  1. genuine_tie orani VE agirlikli tie-kutlesi -- ham tie SAYISI yaniltici
     olabilir: her tekil hash_key'in ILK gorulusu bir tie'dir (agaclar bos),
     ve o tek yazi-tura sonrasi key'in TUM tekrarlari ayni sonucu miras alir.
     Onemli olan tie sayisi degil, o ilk-gorulus yazi-turalarinin etkiledigi
     TOPLAM TRAFIK AGIRLIGI -- bir avuc nadir tie, Zipf skew altinda trafigin
     buyuk kismini belirleyebilir.
  2. final_primary'nin kumulatif payi (50/100/200/... istekte) -- erken mi
     kilitleniyor (feedback-lock-in kaniti) yoksa surekli mi kayiyor.
  3. hash_key yogunlasmasi (tekil key sayisi, en sik %10 key'in payi) --
     Zipf skew'in title seviyesinden chunk-set seviyesine tasinip tasinmadigi.
  4. en sik key'lerin r1 (ham hash, cache-affinity ONCESI) dagilimi --
     yapisal/deterministik bir bias var mi, gercek TQuAD chunk_id'leriyle.
"""
import json
import sys
from collections import Counter


def main(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    if n == 0:
        print("Log bos.")
        return

    print(f"toplam kayit: {n}")

    # 1) tie orani -- HAM SAYI YANILTICI: her tekil key'in ILK gorulusu bir
    # tie'dir (agaclar bos), ve o tek yazi-tura sonrasi key'in TUM tekrarlari
    # ayni sonucu miras alir. Onemli olan tie SAYISI degil, o ilk-gorulus
    # yazi-turalarinin TOPLAM TRAFIK AGIRLIGI.
    n_ties = sum(1 for r in rows if r.get("genuine_tie"))
    tied_keys = {r["hash_key"] for r in rows if r.get("genuine_tie")}
    key_freq = Counter(r["hash_key"] for r in rows)
    weighted_tie_mass = sum(key_freq[k] for k in tied_keys) / n
    print(f"\n== genuine_tie orani ==")
    print(f"  ham tie sayisi: {n_ties}/{n} = {n_ties/n:.1%}  (kayitlarin yuzdesi)")
    print(f"  ETKILENEN TRAFIK PAYI (bu tie'lerin ait oldugu key'lerin TUM "
          f"tekrarlari): {weighted_tie_mass:.1%}")
    if weighted_tie_mass > 0.5:
        print("  -> YUKSEK: trafiğin coğunluğunun kaderi bir avuc rastgele "
              "yazi-turayla (ilk-gorulus tie'leri) belirleniyor -- feedback-"
              "lock-in mekanizmasi baskin. Ayni trace'i farkli zamanlarda / "
              "farkli request-completion sirasiyla kosarsan sonuc BUYUK "
              "olcude degisebilir.")
    else:
        print("  -> DUSUK: trafiğin cogu hic tie olmadan (agaclar zaten "
              "doluyken) karar veriliyor, sonuc byuk olcude tie-lotarya "
              "disinda belirleniyor.")

    # 2) kumulatif primary payi
    print(f"\n== kumulatif final_primary payi (surukleme var mi?) ==")
    history = [r["final_primary"] for r in rows]
    checkpoints = [c for c in [50, 100, 200, 400, n] if c <= n]
    for cp in checkpoints:
        window = history[:cp]
        c0 = window.count(0)
        print(f"  ilk {cp:4d} istek: worker0 payi = {c0/cp:.1%}")

    # 3) hash_key yogunlasmasi
    print(f"\n== hash_key yogunlasmasi ==")
    freq = Counter(r["hash_key"] for r in rows)
    n_distinct = len(freq)
    top10_share = sum(c for _, c in freq.most_common(max(1, n_distinct // 10))) / n
    print(f"  tekil hash_key: {n_distinct} ({n_distinct/n:.1%} tekillik)")
    print(f"  en sik key payi: {freq.most_common(1)[0][1]/n:.1%}")
    print(f"  en sik %10 key payi: {top10_share:.1%}")

    # 4) en sik key'lerin ham r1 dagilimi
    print(f"\n== en sik 20 key'in ham r1 (cache-affinity oncesi) dagilimi ==")
    key_to_r1 = {}
    for r in rows:
        key_to_r1.setdefault(r["hash_key"], r["r1"])
    top20 = freq.most_common(20)
    node_counts = Counter()
    for key, count in top20:
        node_counts[key_to_r1[key]] += count
    total = sum(node_counts.values())
    for node in sorted(node_counts):
        print(f"  node {node}: {node_counts[node]} ({node_counts[node]/total:.1%})")

    print(f"\n== migration orani ==")
    n_migrated = sum(1 for r in rows if r.get("migrated"))
    print(f"  {n_migrated}/{n} = {n_migrated/n:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("kullanim: python3 analyze_cwdm_log.py <log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
