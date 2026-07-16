"""
=============================================================
STREAMLIT APP - Prediksi Risiko Penyakit Jantung
Nama : Laily Muthia N | NIM: A11.2024.15618
Tugas: Capstone UAS - Machine Learning (CRISP-DM Deployment)
=============================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import pathlib

# -------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Risiko Jantung | Laily Muthia N",
    page_icon="🫀",
    layout="wide",
)

# Palet Warna: Mengikuti Gambar "Soft Rose Workspace"
BG_CANVAS = "#FAF6F0"       # Cream/Beige lembut untuk latar utama
BG_SIDEBAR = "#FADCD5"      # Soft Rose Pink untuk sidebar
INK_DARK = "#4A3531"        # Cokelat tua arang untuk teks dominan
ACCENT_ROSE = "#E39688"     # Coral Rose untuk tombol/aktif
ACCENT_ACTIVE = "#F3B5A9"   # Highlight navigasi aktif
MUTED = "#8C7671"           # Teks sekunder
SAFE = "#9DBA99"            # Hijau Sage untuk Normal/Rendah
DANGER = "#D98880"          # Merah Rose pudar untuk Risiko Tinggi

# CSS Custom untuk merombak total UI Streamlit agar identik dengan contoh gambar
st.markdown(f"""
<style>
    /* Mengubah font global dan warna teks utama */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: {INK_DARK} !important;
    }}

    /* Background Utama (Canvas) */
    .stApp {{
        background-color: {BG_CANVAS} !important;
        background-attachment: fixed !important;
    }}

    /* Menghilangkan padding atas bawaan Streamlit agar header tidak terlalu ke bawah */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }}

    [data-testid="stHeader"] {{
        background: rgba(255,255,255,0) !important;
        height: 0px !important;
    }}

    /* Styling Sidebar ala Soft Rose Workspace */
    [data-testid="stSidebar"] {{
        background-color: {BG_SIDEBAR} !important;
        border-right: 1px solid #EBC4BC !important;
    }}

    /* Memperbaiki warna teks di dalam sidebar */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
        color: {INK_DARK} !important;
    }}

    /* Mengubah gaya pilihan Menu di Sidebar (Radio Button bergaya Tab Vertikal) */
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] {{
        gap: 8px !important;
    }}

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label {{
        background-color: rgba(255,255,255,0.4) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        transition: all 0.2s ease;
        cursor: pointer !important;
    }}

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label:hover {{
        background-color: rgba(255,255,255,0.7) !important;
    }}

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label[data-checked="true"] {{
        background-color: {ACCENT_ACTIVE} !important;
        border: 1px solid {ACCENT_ROSE} !important;
        box-shadow: 0 4px 10px rgba(227, 150, 136, 0.2) !important;
    }}

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label[data-checked="true"] p {{
        font-weight: 700 !important;
    }}

    /* Judul dan Subjudul */
    .main h1 {{ font-weight: 800 !important; color: {INK_DARK} !important; letter-spacing: -0.5px; }}
    .main h2 {{ font-weight: 700 !important; color: {INK_DARK} !important; margin-top: 10px !important; }}
    .main h3, .main h4 {{ font-weight: 600 !important; color: {INK_DARK} !important; }}

    /* Wadah Kartu Putih Minimalis dengan Border Rose Gold Tipis */
    .card {{
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #EFEAE2;
        box-shadow: 0 4px 12px rgba(74, 53, 49, 0.03);
        margin-bottom: 20px;
    }}

    .guide-box {{
        background: #FDF9F5;
        border-left: 4px solid {ACCENT_ROSE};
        padding: 14px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 14.5px;
        color: {INK_DARK};
        border: 1px solid #F2ECE2;
    }}
    .guide-box strong {{ color: {INK_DARK}; }}

    /* Tombol Utama (Submit) Bergaya Soft Rose */
    button[data-testid="baseButton-primary"] {{
        background-color: {ACCENT_ROSE} !important;
        border: 1px solid #D68779 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        width: 100% !important;
        box-shadow: 0 3px 8px rgba(227, 150, 136, 0.2) !important;
        transition: background 0.2s;
    }}
    button[data-testid="baseButton-primary"]:hover {{
        background-color: #D68779 !important;
    }}
    button[data-testid="baseButton-primary"] p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }}

    /* Tombol Sekunder (Outline) untuk aksi kedua seperti "Selengkapnya" */
    button[data-testid="baseButton-secondary"] {{
        background-color: #FFFFFF !important;
        border: 1.5px solid {ACCENT_ROSE} !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        width: 100% !important;
        transition: background 0.2s;
    }}
    button[data-testid="baseButton-secondary"]:hover {{
        background-color: #FDF1EE !important;
    }}
    button[data-testid="baseButton-secondary"] p {{
        color: {ACCENT_ROSE} !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }}

    /* Metric Cards Atas ala Planfix Workspace */
    .metric-card {{
        background: #FFFFFF;
        border-radius: 10px;
        padding: 20px 14px;
        border: 1px solid #EFEAE2;
        box-shadow: 0 3px 8px rgba(0,0,0,0.02);
        text-align: center;
    }}
    .metric-card h4 {{ color: {MUTED} !important; margin: 0; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; }}
    .metric-card h2 {{ color: {INK_DARK} !important; margin: 4px 0 0 0; font-size: 28px; font-weight: 800; }}

    /* Badges Hasil Prediksi Pasien */
    .badge-high {{
        background-color: {DANGER} !important;
        color: #FFFFFF !important;
        padding: 14px; border-radius: 8px; font-weight: 700; font-size: 15px;
        text-align: center; display: block;
        box-shadow: 0 3px 8px rgba(217, 136, 128, 0.2);
    }}
    .badge-low {{
        background-color: {SAFE} !important;
        color: #FFFFFF !important;
        padding: 14px; border-radius: 8px; font-weight: 700; font-size: 15px;
        text-align: center; display: block;
        box-shadow: 0 3px 8px rgba(157, 186, 153, 0.2);
    }}

    /* Kartu Fitur di Beranda */
    .feature-card {{
        background: #FFFFFF;
        border-radius: 12px;
        padding: 22px 18px;
        border: 1px solid #EFEAE2;
        box-shadow: 0 4px 12px rgba(74, 53, 49, 0.03);
        height: 100%;
        transition: transform 0.15s ease;
    }}
    .feature-card:hover {{
        transform: translateY(-3px);
    }}
    .feature-card .icon {{ font-size: 26px; }}
    .feature-card h4 {{ margin: 8px 0 4px 0 !important; font-size: 15px !important; }}
    .feature-card p {{ color: {MUTED}; font-size: 13px; margin: 0; line-height: 1.5; }}

    /* Hero Section Beranda */
    .hero-box {{
        background: linear-gradient(135deg, #FFFFFF 0%, #FDF1EE 100%);
        border-radius: 16px;
        padding: 36px 32px;
        border: 1px solid #F2DAD3;
        box-shadow: 0 6px 18px rgba(227, 150, 136, 0.08);
        margin-bottom: 24px;
    }}

    hr {{ border-color: #EFEAE2 !important; }}

    /* Sidebar navigasi "roll" - collapsed hanya menampilkan halaman aktif */
    [data-testid="stSidebar"] details {{
        background-color: rgba(255,255,255,0.35) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        padding: 2px 4px !important;
    }}
    [data-testid="stSidebar"] details summary {{
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 8px 6px !important;
        color: {INK_DARK} !important;
        list-style: none !important;
    }}
    [data-testid="stSidebar"] details summary::-webkit-details-marker {{
        display: none !important;
    }}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_assets():
    best_model = joblib.load(os.path.join(BASE, "models/best_model.pkl"))
    with open(os.path.join(BASE, "models/best_model_name.txt")) as f:
        best_name = f.read().strip()
    metrics_df = pd.read_csv(os.path.join(BASE, "outputs/model_comparison_metrics.csv"))
    df_raw = pd.read_csv(os.path.join(BASE, "data/raw/heart.csv"))
    return best_model, best_name, metrics_df, df_raw

best_model, best_name, metrics_df, df_raw = load_assets()

# -------------------------------------------------------------
# 2. STATE NAVIGASI (agar tombol CTA di Beranda bisa pindah halaman)
# -------------------------------------------------------------
MENU_OPTIONS = [
    "🏠 Beranda",
    "📊 Dashboard EDA",
    "🩺 Model Demo",
    "📈 Evaluasi Model",
    "🔍 Interpretasi Hasil",
    "📄 Dokumentasi",
]

if "menu_choice" not in st.session_state:
    st.session_state.menu_choice = MENU_OPTIONS[0]

def goto(menu_target: str):
    """Helper untuk pindah halaman dari tombol mana pun (di luar widget radio)."""
    st.session_state.menu_choice = menu_target
    st.rerun()

def _sync_menu_from_radio():
    """Dipanggil saat user klik radio di sidebar -> sinkronkan ke sumber kebenaran menu_choice."""
    st.session_state.menu_choice = st.session_state.nav_radio

# -------------------------------------------------------------
# 3. SIDEBAR NAVIGATION (bentuk "roll": collapsed = nama halaman aktif saja,
#    dipencet -> memanjang ke bawah menampilkan semua menu)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<div style='margin-bottom: 10px;'><h2 style='margin:0; font-size:24px;'>Heart Risk App</h2>"
        f"<span style='font-style: italic; color:{MUTED}; font-size:13px;'>PREDIKSI RISIKO PENYAKIT JANTUNG</span></div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr style='margin-top:0; margin-bottom:20px;'>", unsafe_allow_html=True)

    # Expander sebagai navigasi "roll": label selalu menampilkan halaman yang
    # sedang dibuka. Saat tidak dipencet -> hanya tulisan nama halaman aktif.
    # Saat dipencet -> memanjang ke bawah menampilkan seluruh pilihan menu.
    with st.expander(f"📍 {st.session_state.menu_choice}", expanded=False):
        st.radio(
            "NAVIGATION MENU",
            MENU_OPTIONS,
            index=MENU_OPTIONS.index(st.session_state.menu_choice),
            label_visibility="collapsed",
            key="nav_radio",
            on_change=_sync_menu_from_radio,
        )
    menu_pilihan = st.session_state.menu_choice

    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background: rgba(255,255,255,0.5); padding: 12px; border-radius:8px; border:1px solid rgba(0,0,0,0.05);'>"
        f"<span style='font-size:10px; text-transform:uppercase; font-weight:700; color:{MUTED}; display:block;'>Workspace Plan</span>"
        f"<strong style='font-size:14px; color:{INK_DARK}; font-style:italic;'>Professional</strong>"
        f"<div style='margin-top:5px; font-size:11px; color:{INK_DARK};'>🤖 Model Active: <strong>{best_name}</strong></div>"
        f"</div>",
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# 4. HEADER UTAMA
# -------------------------------------------------------------
st.markdown("<h1 style='font-size:32px; margin-bottom:2px; padding-top:0px;'>🫀 Heart Risk App</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='font-size:14px; color:{MUTED}; margin-bottom:2px;'>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='font-size:12px; color:{MUTED}; margin-bottom:12px;'>📍 Anda sedang berada di: <strong>{menu_pilihan}</strong></p>",
    unsafe_allow_html=True,
)
st.markdown(f"<hr style='border: 0; height: 1.5px; background: linear-gradient(to right, {ACCENT_ROSE}, rgba(239,234,226,0.1)); margin-top:0px; margin-bottom:24px;'>", unsafe_allow_html=True)

# -------------------------------------------------------------
# KOORDINASI KONTEN BERDASARKAN PILIHAN MENU SIDEBAR
# -------------------------------------------------------------

# --- MENU 0: BERANDA (landing page, hanya pengenalan aplikasi) ---
if menu_pilihan == "🏠 Beranda":
    st.markdown(f"""
    <div class="hero-box">
        <h2 style="margin-top:0;">Selamat Datang di Heart Risk App 👋</h2>
        <p style="font-size:15px; color:{INK_DARK}; line-height:1.7; max-width:760px;">
            Aplikasi ini adalah sistem bantu skrining awal <strong>risiko penyakit jantung koroner</strong>
            berbasis <em>Machine Learning</em>, dikembangkan sebagai proyek Capstone UAS dengan
            pendekatan metodologi <strong>CRISP-DM</strong>. Model dilatih menggunakan data rekam medis
            918 pasien dengan 11 parameter klinis, lalu dievaluasi dan diinterpretasikan secara transparan
            sebelum dideploy dalam bentuk aplikasi web interaktif ini.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h4>Total Data Pasien</h4><h2>{len(df_raw)}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h4>Fitur Klinis</h4><h2>11</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h4>Model Terbaik</h4><h2 style="font-size:20px;">{best_name}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h4>Metodologi</h4><h2 style="font-size:20px;">CRISP-DM</h2></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### ✨ Apa yang Bisa Anda Lakukan di Sini?")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(f"""<div class="feature-card"><div class="icon">📊</div>
        <h4>Dashboard EDA</h4><p>Eksplorasi statistik & visualisasi data pasien secara interaktif.</p></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown(f"""<div class="feature-card"><div class="icon">🩺</div>
        <h4>Model Demo</h4><p>Input data klinis pasien baru, dapatkan prediksi risiko secara real-time.</p></div>""", unsafe_allow_html=True)
    with f3:
        st.markdown(f"""<div class="feature-card"><div class="icon">📈</div>
        <h4>Evaluasi Model</h4><p>Bandingkan performa 3 algoritma: akurasi, precision, recall, ROC-AUC.</p></div>""", unsafe_allow_html=True)
    with f4:
        st.markdown(f"""<div class="feature-card"><div class="icon">🔍</div>
        <h4>Interpretasi</h4><p>Pahami fitur klinis paling berpengaruh lewat Feature Importance & SHAP.</p></div>""", unsafe_allow_html=True)

    st.write("")
    st.write("")

    cta1, cta2 = st.columns(2)
    with cta1:
        if st.button("🚀 Jelajahi Aplikasi (Dashboard & Demo)", type="primary", use_container_width=True):
            goto("📊 Dashboard EDA")
    with cta2:
        if st.button("📚 Selengkapnya (Evaluasi, Interpretasi & Dokumentasi)", type="secondary", use_container_width=True):
            goto("📈 Evaluasi Model")

    st.write("")
    st.caption("⚠️ Aplikasi ini mengahsilkan data sesuai pada database yang saya gunakan.")

# --- MENU 1: DASHBOARD EDA ---
elif menu_pilihan == "📊 Dashboard EDA":
    st.markdown('<div class="guide-box"><strong>📌 Dashboard EDA</strong><br>Rangkuman statistik dari 918 data rekam medis pasien yang digunakan sebagai data latih model.</div>', unsafe_allow_html=True)

    st.button("🩺 Lanjut ke Model Demo →", type="primary", use_container_width=True, on_click=goto, args=("🩺 Model Demo",))
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h4>Total Pasien</h4><h2>{len(df_raw)}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h4>Rasio Sakit Jantung</h4><h2>{df_raw.HeartDisease.mean()*100:.1f}%</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h4>Rata-rata Usia</h4><h2>{df_raw.Age.mean():.0f} th</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h4>Jumlah Fitur</h4><h2>11</h2></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Sebaran Distribusi Data Pasien")
    st.caption("📌 Grafik interaktif — arahkan kursor ke bar/titik untuk lihat detail angka, bisa di-zoom.")
    col1, col2 = st.columns(2)
    with col1:
        counts = df_raw["HeartDisease"].value_counts().sort_index()
        plot_df = pd.DataFrame({
            "Kelas": ["Sehat", "Sakit Jantung"],
            "Jumlah": counts.values
        })
        fig_bar = px.bar(
            plot_df, x="Kelas", y="Jumlah", color="Kelas",
            color_discrete_map={"Sehat": SAFE, "Sakit Jantung": DANGER},
            text="Jumlah", title="Perbandingan Jumlah Kelas Target",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color=INK_DARK, showlegend=False, height=360,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        num_col = st.selectbox("Pilih Parameter Klinis:", ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"])
        plot_data = df_raw.copy()
        plot_data["Status"] = plot_data["HeartDisease"].map({0: "Sehat", 1: "Sakit Jantung"})
        fig_hist = px.histogram(
            plot_data, x=num_col, color="Status",
            color_discrete_map={"Sehat": SAFE, "Sakit Jantung": DANGER},
            marginal="box", opacity=0.75, barmode="overlay",
            title=f"Distribusi Fitur {num_col}",
        )
        fig_hist.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color=INK_DARK, height=360,
            margin=dict(t=50, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Heatmap Korelasi Fitur Numerik")
    st.caption("📌 Arahkan kursor ke tiap sel untuk lihat nilai korelasi persis.")
    numeric_df = df_raw.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)
    fig_heat = px.imshow(
        corr, text_auto=True, color_continuous_scale="Reds",
        aspect="auto", zmin=-1, zmax=1,
    )
    fig_heat.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=INK_DARK, height=480,
        margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- MENU 2: MODEL DEMO ---
elif menu_pilihan == "🩺 Model Demo":
    st.markdown('<div class="guide-box"><strong>📌 Model Demo</strong><br>Masukkan data klinis pasien baru. Model AI akan memproses prediksi secara real-time.</div>', unsafe_allow_html=True)

    st.button("📚 Selengkapnya (Evaluasi, Interpretasi & Dokumentasi) →", type="secondary", use_container_width=True, on_click=goto, args=("📈 Evaluasi Model",))
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🩺 Profil & Tensi**")
        age = st.slider("Usia Pasien", 20, 90, 50)
        sex = st.selectbox("Jenis Kelamin", ["M", "F"], format_func=lambda x: "Laki-laki" if x == "M" else "Perempuan")
        resting_bp = st.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 120)
        chest_pain = st.selectbox("Tipe Nyeri Dada", ["ASY", "NAP", "ATA", "TA"], format_func=lambda x: {
            "ASY": "Asymptomatic", "NAP": "Non-Anginal Pain", "ATA": "Atypical Angina", "TA": "Typical Angina"
        }[x])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🧪 Lemak & Gula Darah**")
        cholesterol = st.slider("Kolesterol Serum (mg/dl)", 100, 600, 200)
        fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        resting_ecg = st.selectbox("Hasil EKG Istirahat", ["Normal", "ST", "LVH"], format_func=lambda x: {
            "Normal": "Normal", "ST": "Kelainan ST-T", "LVH": "Hipertrofi Ventrikel Kiri"
        }[x])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**⚡ Kinerja & Ritme Jantung**")
        max_hr = st.slider("Detak Jantung Maksimal", 60, 210, 150)
        exercise_angina = st.selectbox("Angina saat Olahraga?", ["N", "Y"], format_func=lambda x: "Ya" if x == "Y" else "Tidak")
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.5, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], format_func=lambda x: {
            "Up": "Naik", "Flat": "Datar", "Down": "Menurun"
        }[x])
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    submitted = st.button("🔥 Prediksi Risiko Pasien", use_container_width=True, type="primary")

    if submitted:
        input_data = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
            "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak, "ST_Slope": st_slope
        }
        input_df = pd.DataFrame([input_data])

        with st.spinner("Menghitung probabilitas risiko..."):
            pred = best_model.predict(input_df)[0]
            proba = best_model.predict_proba(input_df)[0][1]
        status_prediksi = "Berisiko Jantung" if pred == 1 else "Normal / Risiko Rendah"

        # --- Simpan otomatis ke riwayat_prediksi.csv ---
        csv_path = os.path.join(BASE, "data", "riwayat_prediksi.csv")
        save_data = input_data.copy()
        save_data["Waktu_Pengecekan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_data["Hasil_Prediksi"] = status_prediksi
        save_data["Probabilitas"] = f"{proba*100:.1f}%"
        save_df = pd.DataFrame([save_data])

        saved_successfully = True
        try:
            pathlib.Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(csv_path):
                save_df.to_csv(csv_path, index=False)
            else:
                save_df.to_csv(csv_path, mode="a", header=False, index=False)
        except PermissionError:
            saved_successfully = False

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Hasil Prediksi")
        colA, colB = st.columns([1, 2])
        with colA:
            if pred == 1:
                st.markdown('<span class="badge-high">⚠️ Berisiko Tinggi</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-low">✅ Risiko Rendah / Normal</span>', unsafe_allow_html=True)
            st.write("")
            st.markdown("**Probabilitas Heart Disease:**")
            st.markdown(f"<h2 style='margin-top:0px;'>{proba*100:.1f}%</h2>", unsafe_allow_html=True)
        with colB:
            gauge_color = DANGER if pred == 1 else SAFE
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 32, "color": INK_DARK}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": INK_DARK},
                    "bar": {"color": gauge_color},
                    "bgcolor": "#F5EFE6",
                    "steps": [
                        {"range": [0, 50], "color": "#EDE3D8"},
                        {"range": [50, 100], "color": "#F7DCD7"},
                    ],
                    "threshold": {"line": {"color": INK_DARK, "width": 2}, "value": 50},
                },
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(t=20, b=10, l=30, r=30),
                paper_bgcolor="#FFFFFF", font_color=INK_DARK,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        if saved_successfully:
            st.success("💾 Data berhasil disimpan ke riwayat: `data/riwayat_prediksi.csv`")
        else:
            st.warning("⚠️ Prediksi selesai, namun gagal menyimpan riwayat. Tutup dulu file `riwayat_prediksi.csv` kalau sedang terbuka di Excel.")

        st.caption("⚠️ Hasil prediksi sesuai database yang digunakan.")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Fitur tambahan: unduh riwayat & lihat riwayat tanpa harus buka file ---
        if os.path.exists(csv_path):
            colD1, colD2 = st.columns(2)
            with colD1:
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "⬇️ Unduh Riwayat Prediksi (CSV)",
                        data=f,
                        file_name="riwayat_prediksi.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with colD2:
                with st.expander("📋 Lihat Riwayat Prediksi"):
                    st.dataframe(pd.read_csv(csv_path).tail(10), use_container_width=True)

# --- MENU 3: EVALUASI MODEL ---
elif menu_pilihan == "📈 Evaluasi Model":
    st.markdown('<div class="guide-box"><strong>📌 Evaluasi Model</strong><br>Perbandingan performa 3 model algoritma pada data uji.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Tabel Perbandingan Metrik")
    st.dataframe(metrics_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#F7DCD7"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Grafik Perbandingan Metrik (Interaktif)")
    st.caption("📌 Klik nama model di legenda untuk sembunyikan/tampilkan, arahkan kursor untuk lihat angka persis.")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    metrics_long = metrics_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metrik", value_name="Skor")
    fig_metrics = px.bar(
        metrics_long, x="Metrik", y="Skor", color="Model", barmode="group",
        color_discrete_sequence=[ACCENT_ROSE, SAFE, MUTED],
        text_auto=".3f",
    )
    fig_metrics.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=INK_DARK, height=420, yaxis_range=[0, 1.05],
        margin=dict(t=20, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Confusion Matrix")
        img_path = os.path.join(BASE, "outputs/08_confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(img_path)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ROC Curve")
        img_path = os.path.join(BASE, "outputs/09_roc_curve.png")
        if os.path.exists(img_path):
            st.image(img_path)
        st.markdown('</div>', unsafe_allow_html=True)

    st.button("🔍 Lanjut ke Interpretasi Hasil →", type="primary", on_click=goto, args=("🔍 Interpretasi Hasil",))

# --- MENU 4: INTERPRETASI HASIL ---
elif menu_pilihan == "🔍 Interpretasi Hasil":
    st.markdown('<div class="guide-box"><strong>📌 Interpretasi Hasil</strong><br>Fitur klinis yang paling berpengaruh terhadap keputusan model.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Feature Importance")
        img_path = os.path.join(BASE, "outputs/10_feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### SHAP Summary Plot")
        img_path = os.path.join(BASE, "outputs/11_shap_summary.png")
        if os.path.exists(img_path):
            st.image(img_path)
        st.markdown('</div>', unsafe_allow_html=True)

    st.button("📄 Lanjut ke Dokumentasi →", type="primary", on_click=goto, args=("📄 Dokumentasi",))

# --- MENU 5: DOKUMENTASI ---
elif menu_pilihan == "📄 Dokumentasi":
    st.markdown('<div class="guide-box"><strong>📌 Dokumentasi</strong><br>Informasi dataset, metodologi, dan cara penggunaan aplikasi.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
### 📖 Tentang Dataset
Aplikasi ini dilatih menggunakan **Heart Failure Prediction Dataset** (Kaggle - fedesoriano), berisi 918 data pasien dengan 11 fitur klinis.

### ⚙️ Metodologi (CRISP-DM)
1. **Business Understanding** — alat bantu skrining awal penyakit kardiovaskular.
2. **Data Understanding** — eksplorasi sebaran fitur rekam medis pasien.
3. **Data Preparation** — imputasi nilai klinis tidak wajar, One-Hot Encoding, scaling.
4. **Modelling** — Random Forest, XGBoost, SVM dengan GridSearchCV.
5. **Evaluation** — model **XGBoost** terpilih (F1-Score tertinggi, ±0.91).
6. **Deployment** — aplikasi web interaktif berbasis Streamlit ini.

### 💡 Cara Penggunaan
1. Mulai dari **🏠 Beranda** untuk memahami gambaran umum aplikasi.
2. Klik **🚀 Jelajahi Aplikasi** untuk masuk ke Dashboard EDA lalu Model Demo.
3. Untuk cek pasien baru, buka menu **🩺 Model Demo**, isi data parameter klinis pasien, klik tombol **Prediksi Risiko Pasien**.
4. Hasil analisis dan grafik probabilitas risiko akan langsung muncul secara seketika, lengkap dengan opsi unduh riwayat.
5. Klik **📚 Selengkapnya** di Beranda (atau menu sidebar) untuk melihat Evaluasi Model & Interpretasi Hasil.

""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.button("🏠 Kembali ke Beranda", type="secondary", on_click=goto, args=("🏠 Beranda",))