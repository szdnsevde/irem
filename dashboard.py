"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Streamlit Dashboard
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN
Çalıştırma: streamlit run dashboard.py
"""
import sys, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="MAN Türkiye – Stok Optimizasyon",
    page_icon="🏭", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main{background:#F8F9FC}
.kart{background:white;border-radius:12px;padding:18px 14px;
      box-shadow:0 2px 10px rgba(0,0,0,.07);text-align:center;border-top:4px solid}
.kart-Q  {border-color:#1E4D8C} .kart-r  {border-color:#C8102E}
.kart-SS {border-color:#F39200} .kart-HZ {border-color:#00843D}
.kart-GEL{border-color:#888}
.kart-baslik{font-size:11px;color:#666;font-weight:700;text-transform:uppercase;margin-bottom:5px}
.kart-deger {font-size:26px;font-weight:800;color:#1A1A2E;line-height:1.1}
.kart-alt   {font-size:11px;color:#999;margin-top:3px}
.bolum{font-size:16px;font-weight:700;color:#1A1A2E;
       border-left:4px solid #1E4D8C;padding-left:10px;margin:20px 0 12px}
.aks{border-radius:10px;padding:14px 22px;color:white;font-size:14px;font-weight:600;margin:12px 0}
.aks-yesil  {background:linear-gradient(135deg,#00843D,#00A84F)}
.aks-sari   {background:linear-gradient(135deg,#F39200,#F5A623)}
.aks-kirmizi{background:linear-gradient(135deg,#C8102E,#E01535)}
.aks-mavi   {background:linear-gradient(135deg,#1E4D8C,#2D6FAE)}
</style>
""", unsafe_allow_html=True)

ETIKET_TRAIN = ["Oca-22","Şub-22","Mar-22","Nis-22","May-22","Haz-22",
                 "Tem-22","Ağu-22","Eyl-22","Eki-22","Kas-22","Ara-22",
                 "Oca-23","Şub-23","Mar-23","Nis-23","May-23","Haz-23",
                 "Tem-23","Ağu-23","Eyl-23","Eki-23","Kas-23","Ara-23",
                 "Oca-24","Şub-24","Mar-24","Nis-24","May-24","Haz-24"]
ETIKET_TEST  = ["Tem-24","Ağu-24","Eyl-24","Eki-24","Kas-24","Ara-24"]
ETIKET_GEL   = ["Oca-25","Şub-25","Mar-25","Nis-25","May-25","Haz-25"]

RENK_ML  = {"RF":"#1E4D8C","XGBoost":"#C8102E","LightGBM":"#00843D","CatBoost":"#F39200"}
RENK_GEL = {"Hareketli Ort.":"#888","Üstel Düzeltme":"#AAA","Croston-SBA":"#555"}
RENK     = {**RENK_ML, **RENK_GEL}

KART = ('<div class="kart kart-{c}"><div class="kart-baslik">{b}</div>'
        '<div class="kart-deger">{d}</div><div class="kart-alt">{a}</div></div>')


@st.cache_resource(show_spinner="Model yükleniyor…")
def yukle(cache_yolu, veri_yolu):
    if Path(cache_yolu).exists():
        with open(cache_yolu, "rb") as f:
            return pickle.load(f)
    from main import pipeline
    return pipeline(veri_yolu, n_trials=20, cache=cache_yolu)


@st.cache_data(show_spinner=False)
def analiz(_ml_df, _global_ml, _opt_df, _abc_df, pid, gadim, nt, nr, seg):
    from model_egitim       import parca_tahmin
    from feature_engineering import parca_optimize, aksiyon_uyarisi
    t = parca_tahmin(pid, _ml_df, _global_ml, n_ay=6)
    o = parca_optimize(pid, _opt_df, _abc_df, t["tahminler"],
                       segment=seg, grid_adim=gadim, n_trials=nt, n_rep=nr)
    u = aksiyon_uyarisi(o, t["tahminler"])
    return t, o, u


# ── SİDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 MAN Türkiye A.Ş.")
    st.caption("ENM412 – Stok Yönetimi Dashboard")
    st.divider()

    st.subheader("⚙️ Sistem Ayarları")
    veri_yolu  = st.text_input("Veri Dosyası", value="tüketim.xlsx")
    cache_yolu = st.text_input("Model Cache",  value="enm412_cache.pkl")

    try:
        sistem    = yukle(cache_yolu, veri_yolu)
        veri      = sistem["veri"]
        global_ml = sistem["global_ml"]
        batch     = sistem["batch_df"]
        analiz_df = sistem.get("analiz_df", pd.DataFrame())
        st.success(f"✅ {len(veri['parcalar']):,} parça yüklendi")
        if not batch.empty and "Sampiyon_Tip" in batch.columns:
            ml_c  = int((batch["Sampiyon_Tip"] == "ML").sum())
            gel_c = int((batch["Sampiyon_Tip"] == "Klasik").sum())
            st.caption(f"ML: {ml_c:,} | Klasik: {gel_c:,} parça")
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()

    st.divider()
    st.subheader("🔍 Ürün Seçimi")
    pid = st.selectbox("Product ID", veri["parcalar"], index=0)

    st.divider()
    st.subheader("🎛️ Optimizasyon")
    gadim = st.slider("Grid Çözünürlüğü",   6, 20, 12)
    nt    = st.slider("Optuna Trial Sayısı", 20, 100, 40)
    nr    = st.slider("SimPy Replikasyon",   10, 30, 15)
    btn   = st.button("🚀 Analizi Çalıştır", use_container_width=True, type="primary")

    st.divider()
    st.caption("Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN")
    st.caption("ENM412 – Endüstri Mühendisliğinde Tasarım II")


# ── HEADER ───────────────────────────────────────────────────────
st.markdown("<h1 style='color:#1A1A2E;font-size:26px;font-weight:800;'>"
            "🏭 Stok Yönetimi Optimizasyon Paneli</h1>", unsafe_allow_html=True)
st.caption("MAN Türkiye A.Ş. | ENM412 | "
           "RF · XGBoost · LightGBM · CatBoost · "
           "Hareketli Ort. · Üstel Düzeltme · Croston-SBA | "
           "Voting: MAE · RMSE · WAPE · sMAPE")
st.divider()

ml_df  = veri["ml_df"]
opt_df = veri["opt_df"]
abc_df = veri["abc_df"]

pr   = abc_df[abc_df["Parça_Kodu"] == pid]
seg_ = pr["Segment"].values[0]            if not pr.empty else "—"
abc_ = pr["ABC"].values[0]               if not pr.empty else "—"
xyz_ = pr["XYZ"].values[0]               if not pr.empty else "—"
mu_  = float(pr["Ort_Aylik_Talep"].values[0]) if not pr.empty else 0.
std_ = float(pr["Std_Sapma"].values[0])       if not pr.empty else 0.

br    = batch[batch["Parça_Kodu"] == pid] if not batch.empty else pd.DataFrame()
samp_ = br["Sampiyon"].values[0]     if not br.empty else "—"
tip_  = br["Sampiyon_Tip"].values[0] if not br.empty and "Sampiyon_Tip" in br.columns else "—"

# Talep segmenti
parca_seg = veri["parca_segment"].get(pid, "DUZENLI")

chip = lambda t, bg, fg: (
    f'<span style="background:{bg};color:{fg};padding:3px 10px;'
    f'border-radius:16px;font-size:12px;font-weight:700;">{t}</span>')

c1,c2,c3,c4,c5,c6 = st.columns([2.5,1,1,1,1,2])
with c1: st.markdown(f"### 📦 {pid}")
with c2: st.markdown(chip(f"ABC:{abc_}", "#E8F0FB","#1E4D8C"), unsafe_allow_html=True)
with c3: st.markdown(chip(f"XYZ:{xyz_}", "#FFF3E0","#F39200"), unsafe_allow_html=True)
with c4: st.markdown(chip(seg_, "#F3E8FB","#7B2D8B"), unsafe_allow_html=True)
with c5:
    seg_renk  = {"DUZENLI":"#00843D","DUZENSIZ":"#C8102E","DUSUK":"#888"}
    seg_icon  = {"DUZENLI":"📈 Düzenli","DUZENSIZ":"📉 Düzensiz","DUSUK":"📦 Düşük"}
    st.markdown(chip(seg_icon.get(parca_seg, parca_seg),
                     "#F0FFF0", seg_renk.get(parca_seg,"#333")),
                unsafe_allow_html=True)
with c6: st.markdown(f"μ={mu_:,.0f} | σ={std_:,.0f} adet/ay")


# ── ANALİZ ───────────────────────────────────────────────────────
if btn or "analiz" not in st.session_state or st.session_state.get("son_pid") != pid:
    with st.spinner(f"🔄 {pid} analiz ediliyor…"):
        try:
            t_res, o_res, uyari = analiz(
                ml_df, global_ml, opt_df, abc_df,
                pid, gadim, nt, nr, parca_seg)
            st.session_state["analiz"]  = (t_res, o_res, uyari)
            st.session_state["son_pid"] = pid
        except Exception as e:
            st.error(f"Analiz hatası: {e}")
            st.stop()
else:
    t_res, o_res, uyari = st.session_state["analiz"]


# ── MODEL KARŞILAŞTIRMA ───────────────────────────────────────────
st.markdown('<div class="bolum">📊 7 Yöntem Karşılaştırması – Test Seti (Tem-24 → Ara-24)'
            ' | Voting: MAE · RMSE · WAPE · sMAPE</div>', unsafe_allow_html=True)

tum_met = t_res.get("tum_met", {})
oylar   = t_res.get("oylar", {})
samp    = t_res.get("sampiyon", "")

if tum_met:
    rows = []
    for m_adi, m_val in sorted(tum_met.items(), key=lambda x: x[1].get("WAPE", 999)):
        tip = "🤖 ML" if m_adi in RENK_ML else "📐 Klasik"
        oy  = oylar.get(m_adi, 0)
        rows.append({
            "Model":    m_adi,
            "Tür":      tip,
            "MAE":      round(m_val.get("MAE",   float("nan")), 1),
            "RMSE":     round(m_val.get("RMSE",  float("nan")), 1),
            "WAPE (%)": round(m_val.get("WAPE",  float("nan")), 1),
            "sMAPE (%)":round(m_val.get("sMAPE", float("nan")), 1),
            "Oy":       "⭐" * oy if oy > 0 else "",
            "Şampiyon": "✅ Şampiyon" if m_adi == samp else "",
        })
    met_df = pd.DataFrame(rows)

    fig_met = go.Figure()
    for _, row in met_df.iterrows():
        fig_met.add_trace(go.Bar(
            name=row["Model"], x=[row["Model"]], y=[row["WAPE (%)"]],
            marker_color=RENK.get(row["Model"], "#999"),
            opacity=1.0 if row["Model"] == samp else 0.5,
            text=f"{row['WAPE (%)']:.1f}%", textposition="outside",
        ))
    if samp in tum_met:
        fig_met.add_annotation(
            x=samp, y=tum_met[samp].get("WAPE",0) * 1.2,
            text="⭐", showarrow=False, font=dict(size=20))
    fig_met.update_layout(
        barmode="group", height=300,
        title="WAPE Karşılaştırması (düşük = iyi) | Voting ile Şampiyon Seçimi",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="WAPE (%)", gridcolor="#F0F0F0"),
        xaxis=dict(showgrid=False), showlegend=False,
        margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_met, use_container_width=True)

    st.info(f"**Voting Sonucu:** **{samp}** şampiyon seçildi "
            f"({t_res.get('sampiyon_tip','')}) | "
            f"Oylar: {', '.join([f'{k}={v}' for k,v in sorted(oylar.items(), key=lambda x:-x[1]) if v>0])}")
    st.dataframe(met_df, hide_index=True, use_container_width=True)


# ── TAHMİN GRAFİĞİ ───────────────────────────────────────────────
st.markdown('<div class="bolum">📈 Tüketim Geçmişi · Test · Gelecek 6 Ay</div>',
            unsafe_allow_html=True)

ts_train  = t_res.get("ts_train",    [])
y_test    = t_res.get("y_test_ham",  [])
tahminler = t_res.get("tahminler",   [])
ml_pred   = t_res.get("tum_ml_pred", {})
gel_pred  = t_res.get("tum_gel_pred",{})
samp_m    = t_res.get("sampiyon", "")

fig = go.Figure()
if ts_train:
    fig.add_trace(go.Scatter(
        x=ETIKET_TRAIN[:len(ts_train)], y=ts_train,
        name="Gerçek (Eğitim)", mode="lines",
        line=dict(color="#1E4D8C", width=2.5),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>"))
if y_test:
    fig.add_trace(go.Scatter(
        x=ETIKET_TEST[:len(y_test)], y=y_test,
        name="Gerçek (Test)", mode="lines+markers",
        line=dict(color="#1E4D8C", width=2.5, dash="dot"),
        marker=dict(size=7),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>"))
for m_adi, m_pred in ml_pred.items():
    is_s = m_adi == samp_m
    fig.add_trace(go.Scatter(
        x=ETIKET_TEST[:len(m_pred)], y=m_pred,
        name=m_adi, mode="lines",
        opacity=1.0 if is_s else 0.35,
        line=dict(color=RENK_ML.get(m_adi,"#888"),
                  width=3 if is_s else 1.5,
                  dash="solid" if is_s else "dot"),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>"))
for g_adi, g_pred in gel_pred.items():
    is_s = g_adi == samp_m
    fig.add_trace(go.Scatter(
        x=ETIKET_TEST[:len(g_pred)], y=g_pred,
        name=g_adi, mode="lines",
        opacity=1.0 if is_s else 0.35,
        line=dict(color=RENK_GEL.get(g_adi,"#888"),
                  width=3 if is_s else 1.5,
                  dash="solid" if is_s else "dashdot"),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>"))
if tahminler:
    fig.add_trace(go.Scatter(
        x=ETIKET_GEL[:len(tahminler)], y=tahminler,
        name=f"Tahmin ({samp_m}) ⭐",
        mode="lines+markers",
        line=dict(color=RENK.get(samp_m,"#C8102E"), width=3.5, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>"))
for xv, lbl in [("Tem-24","← Eğitim | Test →"),("Oca-25","← Test | Tahmin →")]:
    fig.add_vline(x=xv, line_dash="dot", line_color="#F39200", line_width=1.5)
    fig.add_annotation(x=xv, text=lbl, showarrow=False,
                       font=dict(size=9, color="#F39200"),
                       yref="paper", yanchor="bottom", y=0.02)
fig.update_layout(
    height=420, margin=dict(l=20,r=20,t=20,b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="right", x=1, font=dict(size=10)),
    xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=9)),
    yaxis=dict(showgrid=True, gridcolor="#F0F0F0",
               title="Tüketim (adet)", tickformat=","))
st.plotly_chart(fig, use_container_width=True)


# ── AKSİYON UYARISI ──────────────────────────────────────────────
renk  = uyari.get("renk",  "yesil")
mesaj = uyari.get("mesaj", "")
st.markdown(f'<div class="aks aks-{renk}">{mesaj}</div>', unsafe_allow_html=True)


# ── STOK POLİTİKASI KARTLARI ─────────────────────────────────────
st.markdown('<div class="bolum">📦 Stok Politikası: EOQ Klasik vs ML + Grid Search + Optuna + SimPy</div>',
            unsafe_allow_html=True)

opt_Q  = o_res["optimal_Q"];  opt_r  = o_res["optimal_r"]
opt_SS = o_res["optimal_SS"]; Q_eoq  = o_res["Q_eoq"]
r_eoq  = o_res["r_eoq"];      SS_eoq = o_res["SS_eoq"]
hiz    = o_res.get("sim_HZ", 0.95) * 100
mu_h   = o_res.get("mu_hist", 0)
mu_t   = o_res.get("mu_tahmin", 0)
yontem = o_res.get("yontem", "Grid+Optuna+SimPy")

st.markdown(
    f"<p style='font-size:11px;font-weight:700;color:#F39200;margin:4px 0;'>"
    f"📐 EOQ KLASİK — Tarihsel μ = {mu_h:,.0f} adet/ay</p>",
    unsafe_allow_html=True)
e1,e2,e3,e4 = st.columns(4)
with e1: st.markdown(KART.format(c="GEL",b="EOQ Q", d=f"{Q_eoq:,}", a="√(2SD/h)"),         unsafe_allow_html=True)
with e2: st.markdown(KART.format(c="GEL",b="EOQ r", d=f"{r_eoq:,}", a="μ_L+1.65·σ_L"),     unsafe_allow_html=True)
with e3: st.markdown(KART.format(c="GEL",b="EOQ SS",d=f"{SS_eoq:,}",a="1.65·σ_L"),         unsafe_allow_html=True)
with e4: st.markdown(KART.format(c="GEL",b="EOQ Maliyet",d=f"{o_res['eoq_TC']:,.0f}",a="TL/ay"), unsafe_allow_html=True)

st.markdown(
    f"<p style='font-size:11px;font-weight:700;color:#1E4D8C;margin:12px 0 4px;'>"
    f"🤖 {yontem} — Tahmin μ = {mu_t:,.0f} adet/ay</p>",
    unsafe_allow_html=True)
k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown(KART.format(c="Q",  b="Önerilen Q*",  d=f"{opt_Q:,}",  a="adet/sipariş"), unsafe_allow_html=True)
with k2: st.markdown(KART.format(c="r",  b="Önerilen r*",  d=f"{opt_r:,}",  a="adet (ROP)"),   unsafe_allow_html=True)
with k3: st.markdown(KART.format(c="SS", b="Önerilen SS*", d=f"{opt_SS:,}", a="adet (z=1.65)"),unsafe_allow_html=True)
with k4: st.markdown(KART.format(c="HZ", b="SimPy Hizmet", d=f"{hiz:.1f}%",
                                  a=f"{o_res['sim_TC']:,.0f} TL/ay"), unsafe_allow_html=True)


# ── TASARRUF BANNER ──────────────────────────────────────────────
tl   = o_res["tasarruf_tl"]
oran = o_res["tasarruf_oran"]
gren = "linear-gradient(135deg,#00843D,#00A84F)" if tl >= 0 \
       else "linear-gradient(135deg,#C8102E,#E01535)"
ikon = "✅" if tl >= 0 else "⚠️"
lbl  = "EOQ'ya Göre Tasarruf" if tl >= 0 else "EOQ'ya Göre Maliyet Artışı"

st.markdown(f"""
<div style="background:{gren};border-radius:12px;padding:18px 26px;color:white;
     display:flex;justify-content:space-between;align-items:center;
     box-shadow:0 4px 16px rgba(0,0,0,.15);margin:14px 0;">
  <div>
    <div style="font-size:12px;opacity:.85;">{ikon} {lbl}</div>
    <div style="font-size:30px;font-weight:900;">{abs(tl):,.2f} TL/ay</div>
    <div style="font-size:11px;opacity:.75;">Yıllık: <b>{abs(tl)*12:,.0f} TL</b></div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:12px;opacity:.85;">Tasarruf Oranı</div>
    <div style="font-size:40px;font-weight:900;">{abs(oran):.1f}%</div>
    <div style="font-size:11px;opacity:.75;">
      EOQ:{o_res['eoq_TC']:,.0f} → ML:{o_res['sim_TC']:,.0f} TL/ay
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── MALİYET TABLOSU ──────────────────────────────────────────────
st.markdown('<div class="bolum">💰 Maliyet Karşılaştırması: EOQ vs ML + Optuna + SimPy</div>',
            unsafe_allow_html=True)

kar = pd.DataFrame({
    "Maliyet Kalemi":  ["Elde Tutma","Sipariş","Stoksuz Kalma","TOPLAM"],
    "EOQ Klasik":      [o_res["eoq_ET"],o_res["eoq_SI"],o_res["eoq_SK"],o_res["eoq_TC"]],
    "ML+Optuna+SimPy": [o_res["sim_ET"],o_res["sim_SI"],o_res["sim_SK"],o_res["sim_TC"]],
})
kar["Tasarruf (TL/ay)"] = (kar["EOQ Klasik"] - kar["ML+Optuna+SimPy"]).round(2)
kar["Tasarruf (%)"]     = (kar["Tasarruf (TL/ay)"] /
                            kar["EOQ Klasik"].replace(0, float("nan")) * 100).round(1)

tab1, tab2 = st.tabs(["📋 Tablo", "📊 Grafik"])
with tab1:
    d = kar.copy()
    for c in ["EOQ Klasik","ML+Optuna+SimPy","Tasarruf (TL/ay)"]:
        d[c] = d[c].apply(lambda x: f"{x:,.2f} TL")
    d["Tasarruf (%)"] = d["Tasarruf (%)"].apply(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    st.dataframe(d, hide_index=True, use_container_width=True)
    st.caption("h, p, S maliyet katsayıları sabittir — her iki sistemde aynı değerler kullanılır.")

with tab2:
    kategoriler = ["Elde Tutma","Sipariş","Stoksuz Kalma"]
    eoq_v = kar.iloc[:3]["EOQ Klasik"].tolist()
    ml_v  = kar.iloc[:3]["ML+Optuna+SimPy"].tolist()
    fig2  = go.Figure()
    for i, kat in enumerate(kategoriler):
        fig2.add_trace(go.Bar(name=f"{kat} (EOQ)", x=["EOQ"], y=[eoq_v[i]],
                               marker_color=["#F39200","#F5A623","#F7C46A"][i],
                               text=f"{eoq_v[i]:,.0f}", textposition="inside"))
        fig2.add_trace(go.Bar(name=f"{kat} (ML)",  x=["ML+SimPy"], y=[ml_v[i]],
                               marker_color=["#1E4D8C","#2D6FAE","#5B93C5"][i],
                               text=f"{ml_v[i]:,.0f}", textposition="inside"))
    fig2.update_layout(barmode="stack", height=320,
                       plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(title="TL/ay",gridcolor="#F0F0F0"),
                       xaxis=dict(showgrid=False),
                       legend=dict(orientation="h",yanchor="bottom",y=1.01),
                       margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ── PORTFÖY ──────────────────────────────────────────────────────
st.markdown('<div class="bolum">📊 Portföy Genel Bakış</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)

with p1:
    if not batch.empty and "Sampiyon" in batch.columns:
        sc = batch["Sampiyon"].value_counts().reset_index()
        sc.columns = ["Model","Sayı"]
        fig_s = px.pie(sc, values="Sayı", names="Model",
                       title="Şampiyon Dağılımı (Voting)",
                       color="Model", color_discrete_map=RENK, hole=0.4)
        fig_s.update_layout(height=280, margin=dict(l=5,r=5,t=40,b=5),
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_s, use_container_width=True)

with p2:
    if not batch.empty and "Segment" in batch.columns:
        sc2 = batch["Segment"].value_counts().reset_index()
        sc2.columns = ["Segment","Sayı"]
        renk_seg = {"DUZENLI":"#00843D","DUZENSIZ":"#C8102E","DUSUK":"#888"}
        fig_seg = px.bar(sc2, x="Segment", y="Sayı",
                         title="Talep Segmenti Dağılımı",
                         color="Segment", color_discrete_map=renk_seg)
        fig_seg.update_layout(height=280, margin=dict(l=5,r=5,t=40,b=5),
                               plot_bgcolor="white", paper_bgcolor="white",
                               showlegend=False, xaxis=dict(showgrid=False),
                               yaxis=dict(gridcolor="#F0F0F0"))
        st.plotly_chart(fig_seg, use_container_width=True)

with p3:
    if not batch.empty and "WAPE" in batch.columns:
        wape_d = batch["WAPE"].dropna()
        fig_w = px.histogram(wape_d, nbins=30,
                             title="WAPE Dağılımı (Tüm Parçalar)",
                             color_discrete_sequence=["#1E4D8C"])
        fig_w.update_layout(height=280, margin=dict(l=5,r=5,t=40,b=5),
                             plot_bgcolor="white", paper_bgcolor="white",
                             xaxis_title="WAPE (%)",
                             yaxis=dict(gridcolor="#F0F0F0"), showlegend=False)
        median_wape = wape_d.median()
        fig_w.add_vline(x=median_wape, line_dash="dash", line_color="#C8102E")
        fig_w.add_annotation(x=median_wape, y=0, text=f"Medyan: {median_wape:.1f}%",
                              showarrow=False, yref="paper", y=0.95,
                              font=dict(color="#C8102E", size=10))
        st.plotly_chart(fig_w, use_container_width=True)

if not batch.empty and "Sampiyon_Tip" in batch.columns:
    ml_c  = int((batch["Sampiyon_Tip"] == "ML").sum())
    gel_c = int((batch["Sampiyon_Tip"] == "Klasik").sum())
    st.info(f"🤖 **{ml_c:,}** parçada ML şampiyon  ·  "
            f"📐 **{gel_c:,}** parçada Klasik şampiyon  "
            f"(Voting: MAE · RMSE · WAPE · sMAPE)")

st.divider()
st.markdown("<div style='text-align:center;color:#999;font-size:11px;'>"
            "ENM412 | MAN Türkiye A.Ş. | "
            "Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN | 2024-2025</div>",
            unsafe_allow_html=True)
