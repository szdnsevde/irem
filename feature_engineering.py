"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi
Modül 4: Stok Optimizasyonu (Grid Search + Optuna + SimPy)
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

DUSUK segment → Min-Max politika parametresi
DUZENSIZ/DUZENLI → Grid Search + Optuna + SimPy
"""

import numpy as np
import pandas as pd
import optuna
import simpy
from scipy.stats import norm
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from veri_hazirlama import SEGMENT_DUSUK


# ══════════════════════════════════════════════════════════════════
# ANALİTİK YARDIMCILAR
# ══════════════════════════════════════════════════════════════════

def lt_param(mu, sigma, LT_ay):
    mu_L    = mu * LT_ay
    sigma_L = np.sqrt(sigma**2 * LT_ay + (mu * LT_ay * 0.20)**2)
    return float(mu_L), float(max(sigma_L, 1e-6))


def tc_analitik(Q, r, mu, mu_L, sigma_L, h, p, S):
    """Analitik toplam maliyet (TL/ay). h, p, S sabittir."""
    Q    = max(float(Q), 1.0)
    z    = (r - mu_L) / sigma_L if sigma_L > 0 else 0.0
    E_so = max(sigma_L * (norm.pdf(z) - z * (1 - norm.cdf(z))), 0.0)
    SS   = max(0.0, r - mu_L)
    et   = h * (Q / 2.0 + SS)
    si   = S * (mu / Q)
    sk   = p * (mu / Q) * E_so
    return {"ET": et, "SI": si, "SK": sk, "TC": et + si + sk}


# ══════════════════════════════════════════════════════════════════
# DÜŞÜK HACİM → MİN-MAX POLİTİKA
# ══════════════════════════════════════════════════════════════════

def minmax_politika(mu, sigma, LT_ay, h, p, S, z_hizmet=1.65):
    """
    Düşük hacimli parçalar için basit Min-Max stok politikası.
    Min = Emniyet Stoğu + Teslim Süresindeki Beklenen Talep
    Max = Min + Ekonomik Sipariş Miktarı
    """
    mu_L, sigma_L = lt_param(mu, sigma, LT_ay)
    SS  = max(0, int(round(z_hizmet * sigma_L)))
    r   = int(round(mu_L + SS))
    Q   = max(1, int(round(np.sqrt(2 * S * max(mu, 1) / max(h, 1e-9)))))
    maks = r + Q
    komp = tc_analitik(Q, r, mu, mu_L, sigma_L, h, p, S)
    return {
        "optimal_Q":  Q,
        "optimal_r":  r,
        "optimal_SS": SS,
        "stok_min":   r,
        "stok_max":   maks,
        "sim_TC":     komp["TC"],
        "sim_ET":     komp["ET"],
        "sim_SI":     komp["SI"],
        "sim_SK":     komp["SK"],
        "sim_HZ":     float(norm.cdf((r - mu_L) / sigma_L)) if sigma_L > 0 else 1.0,
        "yontem":     "Min-Max Politika",
    }


# ══════════════════════════════════════════════════════════════════
# GRID SEARCH + OPTUNA
# ══════════════════════════════════════════════════════════════════

def grid_search(mu, sigma, LT_ay, h, p, S, Q_ref, adim=15):
    mu_L, sigma_L = lt_param(mu, sigma, LT_ay)
    Q_ref = max(Q_ref, 1.0)
    Q_grid = np.unique(np.linspace(max(1., Q_ref * 0.3), Q_ref * 4, adim).astype(int))
    r_grid = np.unique(np.linspace(max(0., mu_L - sigma_L), mu_L + 3 * sigma_L, adim).astype(int))
    best_tc, best_Q, best_r = np.inf, Q_ref, mu_L + 1.65 * sigma_L
    for Qv in Q_grid:
        for rv in r_grid:
            val = tc_analitik(Qv, rv, mu, mu_L, sigma_L, h, p, S)["TC"]
            if val < best_tc:
                best_tc, best_Q, best_r = val, float(Qv), float(rv)
    return best_Q, best_r, mu_L, sigma_L


def optuna_optimize(mu, sigma, LT_ay, h, p, S, best_Qg, best_rg,
                    n_trials=50, MOQ=1):
    mu_L, sigma_L = lt_param(mu, sigma, LT_ay)
    Q_lb = max(float(MOQ), best_Qg * 0.4)
    Q_ub = max(best_Qg * 2.5, Q_lb + 1.)
    r_lb = max(0., best_rg - sigma_L)
    r_ub = best_rg + sigma_L

    def obj(t):
        Q = t.suggest_float("Q", Q_lb, Q_ub)
        r = t.suggest_float("r", r_lb, r_ub)
        return tc_analitik(Q, r, mu, mu_L, sigma_L, h, p, S)["TC"]

    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    opt_Q  = max(int(round(s.best_params["Q"])), MOQ)
    opt_r  = max(0, int(round(s.best_params["r"])))
    opt_SS = max(0, int(round(1.65 * sigma_L)))
    hiz    = float(norm.cdf((opt_r - mu_L) / sigma_L)) if sigma_L > 0 else 1.0
    komp   = tc_analitik(opt_Q, opt_r, mu, mu_L, sigma_L, h, p, S)
    return {"Q": opt_Q, "r": opt_r, "SS": opt_SS,
            "hiz": hiz, "TC": komp["TC"],
            "ET": komp["ET"], "SI": komp["SI"], "SK": komp["SK"]}


# ══════════════════════════════════════════════════════════════════
# SİMPY DOĞRULAMA
# ══════════════════════════════════════════════════════════════════

class _Stok:
    def __init__(self, env, Q, r, LT_ay, h, p, S, talep, MOQ=1):
        self.env=env; self.Q=max(int(Q),MOQ); self.r=int(r)
        self.LT=LT_ay; self.h=h; self.p=p; self.S=S; self.talep=talep
        self.stok=self.r+self.Q; self.yolda=0
        self.ET=self.SI=self.SK=0.; self.kar=self.tot=0; self.ay=0
        env.process(self._talep()); env.process(self._kontrol())

    def _talep(self):
        for t in self.talep:
            yield self.env.timeout(1)
            t=max(0,round(t)); self.tot+=t
            if self.stok>=t: self.stok-=t; self.kar+=t
            else:
                self.kar+=self.stok; self.SK+=(t-self.stok)*self.p; self.stok=0
            self.ET+=self.stok*self.h; self.ay+=1

    def _kontrol(self):
        while True:
            yield self.env.timeout(0.01)
            if (self.stok+self.yolda)<=self.r and self.yolda==0:
                self.env.process(self._siparis())

    def _siparis(self):
        self.SI+=self.S; self.yolda+=self.Q
        lt=max(0.1, np.random.normal(self.LT, self.LT*0.20))
        yield self.env.timeout(lt)
        self.stok+=self.Q; self.yolda-=self.Q


def simpy_dogrula(Q, r, mu, LT_ay, h, p, S, talep_serisi, n_rep=20, seed=42):
    rng = np.random.default_rng(seed)
    ts  = list(talep_serisi) if talep_serisi else [mu]*12
    while len(ts) < 24: ts = ts + ts
    ts  = ts[:36]
    TC_l,HZ_l,ET_l,SI_l,SK_l = [],[],[],[],[]
    for _ in range(n_rep):
        g   = rng.normal(1.0, 0.05, len(ts))
        t_r = [max(0., t*gi) for t,gi in zip(ts,g)]
        env = simpy.Environment()
        sim = _Stok(env, Q=Q, r=r, LT_ay=LT_ay, h=h, p=p, S=S, talep=t_r)
        env.run(until=len(t_r)+1)
        ay  = max(sim.ay, 1)
        TC_l.append((sim.ET+sim.SI+sim.SK)/ay)
        HZ_l.append(sim.kar/max(sim.tot,1))
        ET_l.append(sim.ET/ay); SI_l.append(sim.SI/ay); SK_l.append(sim.SK/ay)
    return {
        "TC": float(np.mean(TC_l)), "HZ": float(np.mean(HZ_l)),
        "ET": float(np.mean(ET_l)), "SI": float(np.mean(SI_l)),
        "SK": float(np.mean(SK_l)), "std": float(np.std(TC_l)),
    }


# ══════════════════════════════════════════════════════════════════
# ANA OPTİMİZASYON FONKSİYONU
# ══════════════════════════════════════════════════════════════════

def parca_optimize(parca_kodu: str,
                   opt_df: pd.DataFrame,
                   abc_df: pd.DataFrame,
                   tahmin_listesi: list,
                   segment: str = "DUZENLI",
                   grid_adim: int = 12,
                   n_trials: int = 50,
                   n_rep: int = 20) -> dict:
    """
    DUSUK    → Min-Max politika parametresi
    Diğer    → Grid Search + Optuna + SimPy
    Karşılaştırma: EOQ klasik vs ML+SimPy (aynı h, p, S)
    """
    opt_row = opt_df[opt_df["Parça_Kodu"] == parca_kodu]
    abc_row = abc_df[abc_df["Parça_Kodu"] == parca_kodu]
    if opt_row.empty:
        raise ValueError(f"{parca_kodu} opt_df'de bulunamadı")

    o  = opt_row.iloc[0]
    lt = float(o.get("LT_ay", o.get("LT_gun", 20) / 30))
    h  = float(o["h"])
    p  = float(o["p"])
    S  = float(o["S"])

    # Tarihsel μ ve σ
    mu_hist  = float(abc_row.iloc[0]["Ort_Aylik_Talep"]) if not abc_row.empty else 1.
    sig_hist = float(abc_row.iloc[0]["Std_Sapma"])        if not abc_row.empty else 1.

    # ML tahmin μ
    mu_tahmin = float(np.mean([t for t in tahmin_listesi if t > 0])) \
                if tahmin_listesi else mu_hist

    # ── DUSUK segment → Min-Max ──────────────────────────────────
    if segment == SEGMENT_DUSUK:
        sonu = minmax_politika(mu_hist, sig_hist, lt, h, p, S)
        mu_L, sigma_L = lt_param(mu_hist, sig_hist, lt)
        Q_eoq  = max(int(round(np.sqrt(2*S*max(mu_hist,1)/max(h,1e-9)))), 1)
        r_eoq  = int(round(mu_L + 1.65*sigma_L))
        SS_eoq = int(round(1.65*sigma_L))
        eoq    = tc_analitik(Q_eoq, r_eoq, mu_hist, mu_L, sigma_L, h, p, S)
        tasarruf = eoq["TC"] - sonu["sim_TC"]
        return {**sonu,
                "Q_eoq": Q_eoq, "r_eoq": r_eoq, "SS_eoq": SS_eoq,
                "eoq_TC": eoq["TC"], "eoq_ET": eoq["ET"],
                "eoq_SI": eoq["SI"], "eoq_SK": eoq["SK"],
                "tasarruf_tl": tasarruf,
                "tasarruf_oran": (tasarruf/eoq["TC"]*100) if eoq["TC"]>0 else 0.,
                "mu_hist": mu_hist, "mu_tahmin": mu_tahmin,
                "sigma_L": sigma_L, "mu_L": mu_L}

    # ── Diğer segmentler → Grid Search + Optuna + SimPy ─────────
    mu_L_h, sigma_L_h = lt_param(mu_hist, sig_hist, lt)
    Q_eoq  = max(int(round(np.sqrt(2*S*max(mu_hist,1)/max(h,1e-9)))), 1)
    r_eoq  = int(round(mu_L_h + 1.65*sigma_L_h))
    SS_eoq = int(round(1.65*sigma_L_h))
    eoq    = tc_analitik(Q_eoq, r_eoq, mu_hist, mu_L_h, sigma_L_h, h, p, S)

    # ML tahminiyle optimize et
    Q_ref = max(np.sqrt(2*S*max(mu_tahmin,1)/max(h,1e-9)), 1.)
    best_Qg, best_rg, mu_L_t, sigma_L_t = grid_search(
        mu_tahmin, sig_hist, lt, h, p, S, Q_ref, grid_adim)
    opt = optuna_optimize(mu_tahmin, sig_hist, lt, h, p, S,
                          best_Qg, best_rg, n_trials=n_trials)

    # SimPy doğrulama (tarihsel μ ile adil karşılaştırma)
    sim = simpy_dogrula(opt["Q"], opt["r"], mu_hist, lt, h, p, S,
                        tahmin_listesi, n_rep=n_rep)

    tasarruf_tl   = eoq["TC"] - sim["TC"]
    tasarruf_oran = (tasarruf_tl / eoq["TC"] * 100) if eoq["TC"] > 0 else 0.

    return {
        "optimal_Q":  opt["Q"],  "optimal_r":  opt["r"],
        "optimal_SS": opt["SS"],
        "sim_TC":     sim["TC"], "sim_HZ":     sim["HZ"],
        "sim_ET":     sim["ET"], "sim_SI":     sim["SI"], "sim_SK": sim["SK"],
        "sim_std":    sim["std"],
        "Q_eoq":  Q_eoq,    "r_eoq":  r_eoq, "SS_eoq": SS_eoq,
        "eoq_TC": eoq["TC"], "eoq_ET": eoq["ET"],
        "eoq_SI": eoq["SI"], "eoq_SK": eoq["SK"],
        "tasarruf_tl":   tasarruf_tl,
        "tasarruf_oran": tasarruf_oran,
        "mu_hist":   mu_hist,  "mu_tahmin": mu_tahmin,
        "sigma_L":   sigma_L_t, "mu_L":     mu_L_t,
        "h": h, "p": p, "S": S,
        "yontem": "Grid+Optuna+SimPy",
    }


def aksiyon_uyarisi(opt_res: dict, tahminler: list) -> dict:
    Q   = opt_res["optimal_Q"]
    r   = opt_res["optimal_r"]
    SS  = opt_res["optimal_SS"]
    hiz = opt_res.get("sim_HZ", 0.95)
    trend = ((tahminler[-1]-tahminler[0])/max(tahminler[0],1)*100
             if len(tahminler) >= 2 else 0.)
    if hiz < 0.85:
        return {"renk":"kirmizi","trend":trend,
                "mesaj":f"🔴 KRİTİK: Hizmet düzeyi düşük (%{hiz*100:.0f}). Q={Q:,} adet acil sipariş verin."}
    elif trend > 20:
        return {"renk":"sari","trend":trend,
                "mesaj":f"🟡 UYARI: Talep %{trend:.0f} artış bekleniyor. Q={Q:,} sipariş, r={r:,} izleyin."}
    elif trend < -20:
        return {"renk":"mavi","trend":trend,
                "mesaj":f"🔵 BİLGİ: Talep %{abs(trend):.0f} azalış bekleniyor. Sipariş öncesi stoku kontrol edin."}
    else:
        return {"renk":"yesil","trend":trend,
                "mesaj":f"🟢 NORMAL: Q={Q:,} adet sipariş. r={r:,} eşiğinde tetikleyin. SS={SS:,} bulundurun."}
