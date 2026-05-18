"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi
Modül 1: Veri Hazırlama ve Ürün Segmentasyonu
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Segment Mantığı (veri odaklı):
    DUZENSIZ  : is_intermittent=1  → Croston-SBA
    DUSUK     : is_low_volume=1    → Min-Max stok politikası
    DUZENLI   : diğerleri          → ML modelleri yarışır
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Sabitler ────────────────────────────────────────────────────
PARCA    = "Parça_Kodu"
TARIH    = "Tarih"
SPLIT    = "Split"
TALEP    = "Talep"       # Winsorize edilmiş (ML hedefi)
TALEP_HAM= "Talep_ham"  # Ham talep (hata ölçümü için)

SEGMENT_DUZENLI  = "DUZENLI"
SEGMENT_DUZENSIZ = "DUZENSIZ"
SEGMENT_DUSUK    = "DUSUK"

FEATURE_COLS = [
    "ABC_enc", "XYZ_enc",
    "is_synthetic", "lag12_is_real",
    "is_top15", "is_intermittent", "is_low_volume",
    "is_yil_basi", "is_yil_sonu", "is_ceyrek_sonu",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_std_3", "roll_std_6", "roll_max_3",
    "Ay", "Yil", "Ceyrek", "sin_ay", "cos_ay",
    "sifir_oran_gecmis", "talep_trend",
    "yil1_mean", "yil1_cv", "yil1_zeros",
    "MPS_Toplam_Arac", "MPS_lag_1",
    "MPS_LC12m", "MPS_LC18m", "MPS_Coach", "MPS_Coach2", "MPS_Skyliner",
]


def veri_yukle(dosya_yolu: str) -> dict:
    """
    Excel dosyasını okur, ürünleri segmentler, train/test ayırır.

    Segment Mantığı:
        is_intermittent = 1  →  DUZENSIZ  (Croston-SBA)
        is_low_volume   = 1  →  DUSUK     (Min-Max politika)
        diğer           →  DUZENLI   (ML yarışması)
    """
    xl = pd.ExcelFile(dosya_yolu)

    # ── ML hazır veri ────────────────────────────────────────────
    ml = pd.read_excel(xl, sheet_name="ML_Hazir_Veri", header=0)
    ml[PARCA] = ml[PARCA].astype(str).str.strip()
    ml[TARIH] = ml[TARIH].astype(str).str.strip()

    for c in FEATURE_COLS:
        if c in ml.columns:
            ml[c] = pd.to_numeric(ml[c], errors="coerce").fillna(0)
    ml[TALEP]     = pd.to_numeric(ml[TALEP],     errors="coerce").fillna(0)
    ml[TALEP_HAM] = pd.to_numeric(ml[TALEP_HAM], errors="coerce").fillna(0)

    # ── Optimizasyon parametreleri ───────────────────────────────
    opt = pd.read_excel(xl, sheet_name="Optimizasyon_Parametreleri", header=0)
    opt[PARCA] = opt[PARCA].astype(str).str.strip()
    opt = opt.rename(columns={
        "Tedarik Süresi (gün)":    "LT_gun",
        "Birim Maliyet (TL)":      "Birim_Maliyet",
        "Sipariş Maliyeti (TL)":   "S",
        "Elde Tutma (TL/adet/ay)": "h",
        "Stoksuz Maliyet (TL)":    "p",
        "Başlangıç Stok":          "Baslangic_Stok",
    })
    opt["LT_ay"] = opt["LT_gun"] / 30

    # ── ABC/XYZ ──────────────────────────────────────────────────
    abc = pd.read_excel(xl, sheet_name="ABC_XYZ_Segmentasyon", header=0)
    abc[PARCA] = abc[PARCA].astype(str).str.strip()

    # ── Merge ────────────────────────────────────────────────────
    ml = ml.merge(
        opt[[PARCA, "LT_gun", "LT_ay", "Birim_Maliyet", "S", "h", "p", "Baslangic_Stok"]],
        on=PARCA, how="left"
    )
    ml = ml.merge(
        abc[[PARCA, "Ort_Aylik_Talep", "Std_Sapma", "CV"]],
        on=PARCA, how="left", suffixes=("", "_abc")
    )

    # ── Veri odaklı segment ataması ──────────────────────────────
    def _segment_ata(row):
        if row.get("is_low_volume", 0) == 1:
            return SEGMENT_DUSUK
        elif row.get("is_intermittent", 0) == 1:
            return SEGMENT_DUZENSIZ
        else:
            return SEGMENT_DUZENLI

    ml["talep_segment"] = ml.apply(_segment_ata, axis=1)

    # ── Train / Test ─────────────────────────────────────────────
    train = ml[ml[SPLIT] == "Train"].copy().reset_index(drop=True)
    test  = ml[ml[SPLIT] == "Test"].copy().reset_index(drop=True)
    parcalar = sorted(ml[PARCA].unique().tolist())

    # ── Segment bazlı parça listeleri ────────────────────────────
    parca_segment = ml.groupby(PARCA)["talep_segment"].first().to_dict()
    seg_parcalar  = {
        SEGMENT_DUZENLI:  [p for p,s in parca_segment.items() if s == SEGMENT_DUZENLI],
        SEGMENT_DUZENSIZ: [p for p,s in parca_segment.items() if s == SEGMENT_DUZENSIZ],
        SEGMENT_DUSUK:    [p for p,s in parca_segment.items() if s == SEGMENT_DUSUK],
    }

    # ── Özet ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  ENM412 – VERİ HAZIRLAMA ÖZET")
    print("=" * 60)
    print(f"  Toplam parça    : {len(parcalar):,}")
    print(f"  Train satır     : {len(train):,}  (Ay 1-30)")
    print(f"  Test satır      : {len(test):,}   (Ay 31-36)")
    print(f"\n  Segment Dağılımı:")
    print(f"  {'DUZENLI':<12}: {len(seg_parcalar[SEGMENT_DUZENLI]):>4} parça → ML yarışması")
    print(f"  {'DUZENSIZ':<12}: {len(seg_parcalar[SEGMENT_DUZENSIZ]):>4} parça → Croston-SBA")
    print(f"  {'DUSUK':<12}: {len(seg_parcalar[SEGMENT_DUSUK]):>4} parça → Min-Max politikası")
    print()

    return {
        "ml_df":         ml,
        "abc_df":        abc,
        "opt_df":        opt,
        "train_df":      train,
        "test_df":       test,
        "parcalar":      parcalar,
        "parca_segment": parca_segment,
        "seg_parcalar":  seg_parcalar,
    }


def talep_analizi(ml_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parça bazlı talep analizi — veri keşfi için.
    """
    train = ml_df[ml_df[SPLIT] == "Train"]
    sonuc = []
    for pid, grp in train.groupby(PARCA):
        ts = grp[TALEP_HAM].values
        n  = len(ts)
        sifir_oran = float((ts == 0).mean())
        sonuc.append({
            "Parça_Kodu":    pid,
            "Segment":       grp["talep_segment"].iloc[0],
            "ABC":           grp["ABC"].iloc[0] if "ABC" in grp.columns else "?",
            "XYZ":           grp["XYZ"].iloc[0] if "XYZ" in grp.columns else "?",
            "Ort_Talep":     float(np.mean(ts)),
            "Std_Talep":     float(np.std(ts)),
            "CV":            float(np.std(ts) / np.mean(ts)) if np.mean(ts) > 0 else 0,
            "Sifir_Oran":    sifir_oran,
            "Min":           float(np.min(ts)),
            "Max":           float(np.max(ts)),
            "is_intermittent": int(grp["is_intermittent"].iloc[0]),
            "is_low_volume":   int(grp["is_low_volume"].iloc[0]),
        })
    return pd.DataFrame(sonuc)


def parca_verisi(ml_df: pd.DataFrame, parca_kodu: str) -> dict:
    """Tek parça için train/test zaman serisi döner."""
    pdf   = ml_df[ml_df[PARCA] == parca_kodu].sort_values(TARIH)
    train = pdf[pdf[SPLIT] == "Train"]
    test  = pdf[pdf[SPLIT] == "Test"]
    return {
        "pdf":           pdf,
        "train":         train,
        "test":          test,
        "ts_train":      train[TALEP].values,
        "ts_test":       test[TALEP].values,
        "ts_train_ham":  train[TALEP_HAM].values,
        "ts_test_ham":   test[TALEP_HAM].values,
        "tarihler":      pdf[TARIH].tolist(),
        "abc":           str(pdf["ABC"].iloc[0])          if "ABC"          in pdf.columns else "?",
        "xyz":           str(pdf["XYZ"].iloc[0])          if "XYZ"          in pdf.columns else "?",
        "segment":       str(pdf["talep_segment"].iloc[0]) if "talep_segment" in pdf.columns else "?",
        "abc_xyz":       str(pdf["Segment"].iloc[0])      if "Segment"      in pdf.columns else "?",
    }
