"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi
Modül 2: Model Değerlendirme Metrikleri
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

MAPE yerine WAPE ve sMAPE kullanılır:
    - WAPE: Sıfır taleplerde sonsuz vermez
    - sMAPE: Simetrik, ölçeğe göre normalize edilmiş
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def mae(y_true, y_pred):
    """Ortalama Mutlak Hata."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    """Hata Kareleri Ortalamasının Karekökü."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))


def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error.
    MAPE'nin aksine sıfır taleplerde çalışır.
    WAPE = Σ|gerçek - tahmin| / Σ|gerçek|
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    toplam_gercek = np.sum(np.abs(y_true))
    if toplam_gercek == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / toplam_gercek * 100)


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    Sıfır taleplerde daha az bozulur.
    sMAPE = 2 * |gerçek - tahmin| / (|gerçek| + |tahmin|)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom  = np.abs(y_true) + np.abs(y_pred)
    mask   = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def tum_metrikler(y_true, y_pred, model_adi=""):
    """Tüm metrikleri hesapla."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return {
        "model":  model_adi,
        "MAE":    mae(y_true, y_pred),
        "RMSE":   rmse(y_true, y_pred),
        "WAPE":   wape(y_true, y_pred),
        "sMAPE":  smape(y_true, y_pred),
    }


def voting_sampiyon(metrik_dict: dict) -> tuple:
    """
    Her metrikteki (MAE, RMSE, WAPE, sMAPE) en iyi modele 1 oy.
    En fazla oy → Şampiyon. Beraberlik → WAPE ile boz.

    Returns: (sampiyon_adi, oy_dict)
    """
    oylar = {k: 0 for k in metrik_dict}

    for met in ["MAE", "RMSE", "WAPE", "sMAPE"]:
        degerler = {}
        for model, m in metrik_dict.items():
            v = m.get(met, float("nan"))
            if not np.isnan(v):
                degerler[model] = v
        if not degerler:
            continue
        kazanan = min(degerler, key=lambda k: degerler[k])
        oylar[kazanan] += 1

    maks_oy = max(oylar.values()) if oylar else 0
    adaylar = [k for k, v in oylar.items() if v == maks_oy]

    if len(adaylar) == 1:
        return adaylar[0], oylar

    # Beraberlik → WAPE ile boz
    sampiyon = min(
        adaylar,
        key=lambda k: metrik_dict[k].get("WAPE", float("nan"))
    )
    return sampiyon, oylar


def metrik_tablosu(metrik_dict: dict, sampiyon: str = "") -> pd.DataFrame:
    """Metrikleri tablo olarak döner."""
    rows = []
    for model, m in sorted(metrik_dict.items(),
                            key=lambda x: x[1].get("WAPE", float("nan"))):
        rows.append({
            "Model":   model,
            "MAE":     round(m.get("MAE",   float("nan")), 2),
            "RMSE":    round(m.get("RMSE",  float("nan")), 2),
            "WAPE (%)":round(m.get("WAPE",  float("nan")), 2),
            "sMAPE (%)":round(m.get("sMAPE",float("nan")), 2),
            "Şampiyon": "⭐" if model == sampiyon else "",
        })
    return pd.DataFrame(rows)
