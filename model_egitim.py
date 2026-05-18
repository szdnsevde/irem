"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi
Modül 3: Model Eğitimi – 7 Yöntem
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Yöntemler:
    Klasik : Hareketli Ortalama · Üstel Düzeltme · Croston-SBA
    ML     : Random Forest · XGBoost · LightGBM · CatBoost

Segment → Yöntem Mantığı:
    DUZENSIZ → Önce Croston-SBA, ama tüm yöntemler test edilir
    DUSUK    → Min-Max politika parametresi hesaplanır
    DUZENLI  → Tüm 7 yöntem, voting ile şampiyon seçilir
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from veri_hazirlama import FEATURE_COLS, TALEP, PARCA, SPLIT
from evaluate import tum_metrikler, voting_sampiyon

try:
    import xgboost as xgb;          XGB_OK = True
except ImportError:                  XGB_OK = False
try:
    import lightgbm as lgb;         LGB_OK = True
except ImportError:                  LGB_OK = False
try:
    from catboost import CatBoostRegressor; CAT_OK = True
except ImportError:                  CAT_OK = False

N_AY = 6
ML_ISIMLER  = {"RF", "XGBoost", "LightGBM", "CatBoost"}
GEL_ISIMLER = {"Hareketli Ort.", "Üstel Düzeltme", "Croston-SBA"}


# ══════════════════════════════════════════════════════════════════
# KLASİK YÖNTEMLER
# ══════════════════════════════════════════════════════════════════

def hareketli_ort(ts, n=6, pencere=6):
    k = len(ts)
    v = float(np.mean(ts[-min(pencere, k):])) if k > 0 else 0.0
    return [v] * n


def ustel_duzeltme(ts, n=6, alpha=0.3):
    k = len(ts)
    if k == 0: return [0.0] * n
    v = float(ts[0])
    for x in ts:
        v = alpha * float(x) + (1 - alpha) * v
    return [v] * n


def croston_sba(ts, n=6, alpha=0.1):
    """
    Croston-SBA (Syntetos-Boylan Approximation).
    Aralıklı talep için: büyüklük ve aralığı ayrı modeller.
    """
    ts  = np.array(ts, dtype=float)
    k   = len(ts)
    if k == 0: return [0.0] * n
    nz  = np.where(ts > 0)[0]
    if len(nz) == 0: return [0.0] * n
    z = float(ts[nz[0]])
    p = float(nz[0] + 1)
    q = 1
    for i in range(1, k):
        if ts[i] > 0:
            z = alpha * ts[i] + (1 - alpha) * z
            p = alpha * q    + (1 - alpha) * p
            q = 1
        else:
            q += 1
    val = max((1 - alpha / 2) * (z / p), 0.0)
    return [float(val)] * n


def klasik_tahmin(ts_train, n=6):
    """3 klasik yöntemi çalıştırır."""
    return {
        "Hareketli Ort.": hareketli_ort(ts_train, n),
        "Üstel Düzeltme": ustel_duzeltme(ts_train, n),
        "Croston-SBA":    croston_sba(ts_train, n),
    }


# ══════════════════════════════════════════════════════════════════
# ML YARDIMCILAr
# ══════════════════════════════════════════════════════════════════

def _X(df, feat):
    X = np.zeros((len(df), len(feat)))
    for i, c in enumerate(feat):
        if c in df.columns:
            X[:, i] = df[c].fillna(0).values
    return X


def _cv_rmse(base, X, y, n_splits=4):
    n = len(X); fold = max(n // (n_splits + 1), 1); sc = []
    for i in range(1, n_splits + 1):
        te = i * fold; ve = min(te + fold, n)
        if ve <= te: continue
        try:
            m = type(base)(**base.get_params())
            m.fit(X[:te], y[:te])
            p = m.predict(X[te:ve])
            sc.append(float(np.sqrt(mean_squared_error(
                y[te:ve].flatten(), p.flatten()))))
        except: pass
    return float(np.mean(sc)) if sc else 1e9


def _multi_hedef(df, n_ay=N_AY):
    """t+1 … t+n_ay hedef sütunları ekler."""
    parts = []
    for _, grp in df.groupby(PARCA):
        grp = grp.sort_values("Tarih").copy()
        ts  = grp[TALEP].values
        for i in range(1, n_ay + 1):
            sh = np.roll(ts, -i).astype(float); sh[-i:] = np.nan
            grp[f"h_{i}"] = sh
        parts.append(grp)
    res  = pd.concat(parts)
    hcol = [f"h_{i}" for i in range(1, n_ay + 1)]
    return res.dropna(subset=hcol), hcol


# ══════════════════════════════════════════════════════════════════
# GLOBAL ML EĞİTİMİ
# ══════════════════════════════════════════════════════════════════

def global_ml_egit(train_df: pd.DataFrame, n_trials: int = 30) -> dict:
    """
    4 ML modelini TÜM train verisiyle global eğitir.
    Multi-output: 6 aylık tahmini tek seferde üretir.
    """
    train_mo, hcol = _multi_hedef(train_df)
    feat = [c for c in FEATURE_COLS if c in train_mo.columns]
    X_tr = _X(train_mo, feat)
    y_tr = train_mo[hcol].fillna(0).values

    print(f"  Global ML eğitimi: {len(X_tr):,} satır × {len(feat)} özellik")
    modeller = {}

    # ── Random Forest ───────────────────────────────────────────
    print("  [ML] Random Forest + Optuna...")
    def rf_obj(t):
        p = {"n_estimators": t.suggest_int("n", 50, 200),
             "max_depth":    t.suggest_int("d", 3, 12),
             "max_features": t.suggest_float("f", 0.3, 1.0),
             "min_samples_leaf": t.suggest_int("l", 1, 10),
             "random_state": 42, "n_jobs": -1}
        return _cv_rmse(RandomForestRegressor(**p), X_tr, y_tr[:, 0])
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(rf_obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params
    m  = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=bp["n"], max_depth=bp["d"],
        max_features=bp["f"], min_samples_leaf=bp["l"],
        random_state=42, n_jobs=-1))
    m.fit(X_tr, y_tr)
    modeller["RF"] = m
    print(f"     RF hazır.")

    # ── XGBoost ─────────────────────────────────────────────────
    print("  [ML] XGBoost + Optuna...")
    def xgb_obj(t):
        p = {"n_estimators":  t.suggest_int("n", 50, 300),
             "max_depth":     t.suggest_int("d", 2, 8),
             "learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True),
             "subsample":     t.suggest_float("ss", 0.5, 1.0),
             "random_state":  42}
        base = xgb.XGBRegressor(**p, verbosity=0, n_jobs=-1) if XGB_OK \
               else GradientBoostingRegressor(
                   n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                   learning_rate=p["learning_rate"], random_state=42)
        return _cv_rmse(base, X_tr, y_tr[:, 0])
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params
    base = xgb.XGBRegressor(n_estimators=bp["n"], max_depth=bp["d"],
           learning_rate=bp["lr"], subsample=bp["ss"],
           random_state=42, verbosity=0, n_jobs=-1) if XGB_OK \
           else GradientBoostingRegressor(n_estimators=bp["n"],
           max_depth=bp["d"], learning_rate=bp["lr"], random_state=42)
    m = MultiOutputRegressor(base)
    m.fit(X_tr, y_tr)
    modeller["XGBoost"] = m
    print(f"     XGBoost hazır.")

    # ── LightGBM ────────────────────────────────────────────────
    print("  [ML] LightGBM + Optuna...")
    def lgb_obj(t):
        p = {"n_estimators":  t.suggest_int("n", 50, 300),
             "max_depth":     t.suggest_int("d", 2, 10),
             "learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True),
             "num_leaves":    t.suggest_int("nl", 15, 63),
             "random_state":  42}
        base = lgb.LGBMRegressor(**p, verbose=-1, n_jobs=-1) if LGB_OK \
               else GradientBoostingRegressor(n_estimators=p["n_estimators"],
               max_depth=p["max_depth"], learning_rate=p["learning_rate"],
               random_state=42)
        return _cv_rmse(base, X_tr, y_tr[:, 0])
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(lgb_obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params
    base = lgb.LGBMRegressor(n_estimators=bp["n"], max_depth=bp["d"],
           learning_rate=bp["lr"], num_leaves=bp["nl"],
           random_state=42, verbose=-1, n_jobs=-1) if LGB_OK \
           else GradientBoostingRegressor(n_estimators=bp["n"],
           max_depth=bp["d"], learning_rate=bp["lr"], random_state=42)
    m = MultiOutputRegressor(base)
    m.fit(X_tr, y_tr)
    modeller["LightGBM"] = m
    print(f"     LightGBM hazır.")

    # ── CatBoost ────────────────────────────────────────────────
    print("  [ML] CatBoost + Optuna...")
    def cat_obj(t):
        p = {"iterations":    t.suggest_int("it", 50, 300),
             "depth":         t.suggest_int("d", 2, 8),
             "learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True),
             "random_seed":   42}
        base = CatBoostRegressor(**p, verbose=0) if CAT_OK \
               else GradientBoostingRegressor(n_estimators=p["iterations"],
               max_depth=p["depth"], learning_rate=p["learning_rate"],
               random_state=42)
        return _cv_rmse(base, X_tr, y_tr[:, 0])
    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(cat_obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params
    base = CatBoostRegressor(iterations=bp["it"], depth=bp["d"],
           learning_rate=bp["lr"], random_seed=42, verbose=0) if CAT_OK \
           else GradientBoostingRegressor(n_estimators=bp["it"],
           max_depth=bp["d"], learning_rate=bp["lr"], random_state=42)
    m = MultiOutputRegressor(base)
    m.fit(X_tr, y_tr)
    modeller["CatBoost"] = m
    print(f"     CatBoost hazır.")

    return {"modeller": modeller, "feat": feat, "hcol": hcol}


# ══════════════════════════════════════════════════════════════════
# PARÇA BAZLI TAHMİN
# ══════════════════════════════════════════════════════════════════

def parca_tahmin(parca_kodu: str,
                 ml_df: pd.DataFrame,
                 global_ml: dict,
                 n_ay: int = N_AY) -> dict:
    """
    Her parça için tüm 7 yöntem çalıştırılır.
    Test setinde MAE · RMSE · WAPE · sMAPE hesaplanır.
    Voting ile şampiyon seçilir.
    Şampiyon ile gelecek n_ay tahmini üretilir.
    """
    from veri_hazirlama import parca_verisi
    pv       = parca_verisi(ml_df, parca_kodu)
    ts_train = pv["ts_train"]
    ts_test  = pv["ts_test_ham"]   # Ham talep ile hata ölç
    ts_pred_base = pv["ts_test"]   # Winsorize ile tahmin yap
    test_df  = pv["test"]
    segment  = pv["segment"]

    feat = global_ml["feat"]
    hcol = global_ml["hcol"]

    tum_met     = {}
    tum_tahmin  = {}

    # ── ML modelleri (parça bazlı test tahmini) ─────────────────
    X_te = _X(test_df, feat)
    for m_adi, m_obj in global_ml["modeller"].items():
        try:
            pr = np.maximum(m_obj.predict(X_te), 0)
            p1 = pr[:, 0] if pr.ndim > 1 else pr
            tum_met[m_adi]    = tum_metrikler(ts_test, p1, m_adi)
            tum_tahmin[m_adi] = p1.tolist()
        except Exception as e:
            print(f"  [!] {m_adi} ({parca_kodu}): {e}")

    # ── Klasik yöntemler (parça bazlı) ──────────────────────────
    gel = klasik_tahmin(ts_train, n=max(len(ts_test), n_ay))
    for g_adi, g_pred in gel.items():
        g_te = g_pred[:len(ts_test)]
        tum_met[g_adi]    = tum_metrikler(ts_test, g_te, g_adi)
        tum_tahmin[g_adi] = g_te

    # ── Voting şampiyon ─────────────────────────────────────────
    sampiyon, oylar = voting_sampiyon(tum_met) if tum_met \
                      else ("Hareketli Ort.", {})
    tip = "ML" if sampiyon in ML_ISIMLER else "Klasik"

    # ── Gelecek n_ay tahmini ────────────────────────────────────
    if tip == "ML":
        try:
            X_son = _X(test_df.iloc[-1:], feat)
            pr    = np.maximum(
                global_ml["modeller"][sampiyon].predict(X_son), 0)
            tahminler = pr[0, :n_ay].tolist() \
                        if pr.ndim > 1 and pr.shape[1] >= n_ay \
                        else gel["Hareketli Ort."][:n_ay]
        except:
            tahminler = gel["Hareketli Ort."][:n_ay]
    else:
        tahminler = gel[sampiyon][:n_ay]

    return {
        "tahminler":     tahminler,
        "y_test_ham":    ts_test.tolist(),
        "y_pred_test":   tum_tahmin.get(sampiyon, []),
        "ts_train":      ts_train.tolist(),
        "sampiyon":      sampiyon,
        "sampiyon_tip":  tip,
        "oylar":         oylar,
        "segment":       segment,
        "abc":           pv["abc"],
        "xyz":           pv["xyz"],
        "tum_met":       tum_met,
        "tum_ml_pred":   {k: v for k, v in tum_tahmin.items() if k in ML_ISIMLER},
        "tum_gel_pred":  {k: v for k, v in tum_tahmin.items() if k in GEL_ISIMLER},
    }


def batch_tahmin(ml_df: pd.DataFrame,
                 global_ml: dict,
                 parcalar: list,
                 n_ay: int = N_AY) -> pd.DataFrame:
    """Tüm parçalar için toplu tahmin."""
    rows    = []
    toplam  = len(parcalar)
    for i, pid in enumerate(parcalar, 1):
        if i % 200 == 0:
            print(f"  {i}/{toplam} parça...")
        try:
            res = parca_tahmin(pid, ml_df, global_ml, n_ay)
            rec = {
                "Parça_Kodu":   pid,
                "Segment":      res["segment"],
                "ABC":          res["abc"],
                "XYZ":          res["xyz"],
                "Sampiyon":     res["sampiyon"],
                "Sampiyon_Tip": res["sampiyon_tip"],
            }
            for j, t in enumerate(res["tahminler"], 1):
                rec[f"Tahmin_Ay_{j}"] = round(t, 1)
            sm = res["tum_met"].get(res["sampiyon"], {})
            rec["MAE"]    = round(sm.get("MAE",   float("nan")), 2)
            rec["RMSE"]   = round(sm.get("RMSE",  float("nan")), 2)
            rec["WAPE"]   = round(sm.get("WAPE",  float("nan")), 2)
            rec["sMAPE"]  = round(sm.get("sMAPE", float("nan")), 2)
            rows.append(rec)
        except Exception as e:
            print(f"  [!] {pid}: {e}")
    return pd.DataFrame(rows)
