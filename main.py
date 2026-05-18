"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Ana Pipeline
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Çalıştırma:
    python main.py --dosya tüketim.xlsx --n_trials 30
"""

import argparse, pickle, time, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from veri_hazirlama import veri_yukle, talep_analizi
from model_egitim  import global_ml_egit, batch_tahmin
from evaluate      import metrik_tablosu


def pipeline(dosya_yolu: str,
             n_trials: int = 30,
             cache: str = "enm412_cache.pkl") -> dict:

    t0 = time.time()
    print("=" * 60)
    print("  ENM412 – STOK YÖNETİMİ OPTİMİZASYON PİPELİNE")
    print("=" * 60)

    # ── 1. Veri Hazırlama ────────────────────────────────────────
    print("\n[1/4] Veri yükleniyor ve segmentleniyor...")
    veri = veri_yukle(dosya_yolu)

    # ── 2. Talep Analizi ─────────────────────────────────────────
    print("[2/4] Talep analizi yapılıyor...")
    analiz_df = talep_analizi(veri["ml_df"])

    print(f"\n  Sıfır Talep Oranı Dağılımı:")
    bins   = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    labels = ["<%10", "10-30%", "30-50%", "50-70%", ">70%"]
    analiz_df["sifir_grubu"] = pd.cut(
        analiz_df["Sifir_Oran"], bins=bins, labels=labels, include_lowest=True)
    for g, c in analiz_df["sifir_grubu"].value_counts().sort_index().items():
        print(f"  {g:>8}: {c:>4} parça")

    # ── 3. ML Eğitimi ────────────────────────────────────────────
    print(f"\n[3/4] ML modelleri eğitiliyor ({n_trials} trial/model)...")
    print(f"  RF · XGBoost · LightGBM · CatBoost + Optuna")
    global_ml = global_ml_egit(veri["train_df"], n_trials=n_trials)

    # ── 4. Batch Tahmin ──────────────────────────────────────────
    print(f"\n[4/4] Tüm parçalar için 7 yöntem karşılaştırılıyor...")
    print(f"  ML: RF · XGBoost · LightGBM · CatBoost")
    print(f"  Klasik: Hareketli Ort. · Üstel Düzeltme · Croston-SBA")
    print(f"  Metrik: MAE · RMSE · WAPE · sMAPE → Voting şampiyon")
    batch_df = batch_tahmin(veri["ml_df"], global_ml, veri["parcalar"])

    # ── Özet ─────────────────────────────────────────────────────
    print("\n  Şampiyon Dağılımı:")
    for s, c in batch_df["Sampiyon"].value_counts().items():
        tip = "ML" if s in {"RF","XGBoost","LightGBM","CatBoost"} else "Klasik"
        print(f"  [{tip:6}] {s}: {c:,} parça")

    ml_c  = (batch_df["Sampiyon_Tip"] == "ML").sum()
    gel_c = (batch_df["Sampiyon_Tip"] == "Klasik").sum()
    print(f"\n  Toplam → ML: {ml_c:,} | Klasik: {gel_c:,}")

    print(f"\n  Ortalama Metrikler (şampiyon modeller):")
    for met in ["MAE","RMSE","WAPE","sMAPE"]:
        if met in batch_df.columns:
            v = batch_df[met].dropna().median()
            print(f"  {met:8}: {v:,.1f}")

    # ── Cache ────────────────────────────────────────────────────
    sonuc = {
        "veri":       veri,
        "global_ml":  global_ml,
        "batch_df":   batch_df,
        "analiz_df":  analiz_df,
    }
    with open(cache, "wb") as f:
        import pickle
        pickle.dump(sonuc, f, protocol=4)

    print(f"\n✅ Tamamlandı! Süre: {(time.time()-t0)/60:.1f} dk")
    print(f"   Cache: {cache}")
    print("=" * 60)
    return sonuc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ENM412 Pipeline")
    p.add_argument("--dosya",    default="tüketim.xlsx")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--cache",    default="enm412_cache.pkl")
    args = p.parse_args()
    pipeline(args.dosya, args.n_trials, args.cache)
