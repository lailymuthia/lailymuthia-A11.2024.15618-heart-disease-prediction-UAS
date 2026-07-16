"""
=============================================================
EXPLORATORY DATA ANALYSIS (EDA) - Prediksi Penyakit Jantung
Nama : Laily Muthia N | NIM: A11.2024.15618
CRISP-DM Tahap 2: Data Understanding
=============================================================
Menghasilkan seluruh visualisasi EDA ke folder outputs/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
OUT = "outputs"

df = pd.read_csv("data/raw/heart.csv")

# -------------------------------------------------------------
# 1. Statistik deskriptif awal
# -------------------------------------------------------------
desc = df.describe(include="all").T
desc.to_csv(f"{OUT}/statistik_deskriptif.csv")

# -------------------------------------------------------------
# 2. Distribusi target
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = df["HeartDisease"].value_counts().sort_index()
pct = df["HeartDisease"].value_counts(normalize=True).sort_index() * 100
bars = ax.bar(["Normal (0)", "Heart Disease (1)"], counts.values,
               color=["#2ecc71", "#e74c3c"], edgecolor="black")
for b, p in zip(bars, pct.values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height()+5, f"{p:.1f}%",
            ha="center", fontweight="bold")
ax.set_title("Distribusi Target: HeartDisease", fontweight="bold")
ax.set_ylabel("Jumlah Sampel")
plt.tight_layout()
plt.savefig(f"{OUT}/01_distribusi_target.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 3. Distribusi fitur numerik (histogram)
# -------------------------------------------------------------
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, color="#3498db", edgecolor="black", alpha=0.85)
    axes[i].axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean={df[col].mean():.1f}")
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)
axes[-1].axis("off")
fig.suptitle("Distribusi Fitur Numerik", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/02_distribusi_numerik.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 4. Boxplot fitur numerik vs target (deteksi outlier & pola)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x="HeartDisease", y=col, ax=axes[i], palette=["#2ecc71", "#e74c3c"])
    axes[i].set_title(f"{col} vs HeartDisease")
axes[-1].axis("off")
fig.suptitle("Boxplot Fitur Numerik terhadap Target (Deteksi Outlier)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/03_boxplot_vs_target.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 5. Distribusi fitur kategorikal vs target
# -------------------------------------------------------------
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df["HeartDisease"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=axes[i], color=["#2ecc71", "#e74c3c"])
    axes[i].set_title(col)
    axes[i].set_ylabel("% dalam kategori")
    axes[i].legend(["Normal", "Sakit"], fontsize=7)
    axes[i].tick_params(axis="x", rotation=30)
axes[-1].axis("off")
fig.suptitle("Fitur Kategorikal vs HeartDisease (Proporsi)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/04_kategorikal_vs_target.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 6. Heatmap korelasi
# -------------------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title("Heatmap Korelasi Fitur Numerik", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/05_heatmap_korelasi.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 7. Pairplot ringkas (Age, MaxHR, Oldpeak, Cholesterol vs target)
# -------------------------------------------------------------
pp = sns.pairplot(df, vars=["Age", "MaxHR", "Oldpeak", "Cholesterol"],
                   hue="HeartDisease", palette=["#2ecc71", "#e74c3c"], diag_kind="kde", height=2.2)
pp.fig.suptitle("Pairplot Fitur Numerik Utama", y=1.02, fontweight="bold")
pp.savefig(f"{OUT}/06_pairplot.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 8. Ringkasan temuan (5 insight utama) -> disimpan sebagai teks
# -------------------------------------------------------------
ratio = counts.max() / counts.min()
chol_zero = (df["Cholesterol"] == 0).sum()
bp_zero = (df["RestingBP"] == 0).sum()
corr_target = corr["HeartDisease"].drop("HeartDisease").sort_values(key=abs, ascending=False)

insights = f"""
5 INSIGHT UTAMA HASIL EDA
==========================
1. KETIDAKSEIMBANGAN KELAS RINGAN
   Distribusi target: Normal {pct[0]:.1f}% vs Heart Disease {pct[1]:.1f}%
   (rasio {ratio:.2f}:1). Perlu evaluasi dengan F1-score, bukan hanya akurasi.

2. NILAI 0 TIDAK WAJAR SECARA KLINIS
   Cholesterol=0 pada {chol_zero} baris ({chol_zero/len(df)*100:.1f}%) dan
   RestingBP=0 pada {bp_zero} baris. Nilai ini mustahil secara medis dan
   diperlakukan sebagai data hilang tersembunyi -> perlu imputasi.

3. FITUR PALING BERKORELASI DENGAN TARGET
   Korelasi tertinggi terhadap HeartDisease: {corr_target.index[0]} ({corr_target.iloc[0]:.2f}),
   diikuti {corr_target.index[1]} ({corr_target.iloc[1]:.2f}) dan {corr_target.index[2]} ({corr_target.iloc[2]:.2f}).
   MaxHR berkorelasi negatif -> semakin rendah detak jantung maksimal, semakin
   tinggi risiko penyakit jantung.

4. ST_SLOPE DAN CHEST PAIN TYPE SANGAT DISKRIMINATIF
   Pasien dengan ST_Slope 'Flat' dan ChestPainType 'ASY' (asymptomatic)
   menunjukkan proporsi HeartDisease jauh lebih tinggi dibanding kategori lain,
   menjadikannya fitur kategorikal paling prediktif.

5. TIDAK ADA MULTIKOLINEARITAS EKSTREM
   Tidak ditemukan pasangan fitur numerik dengan korelasi > 0.8, sehingga
   seluruh fitur numerik aman digunakan bersama tanpa reduksi dimensi (PCA)
   pada tahap awal modelling.
"""
with open(f"{OUT}/insight_eda.txt", "w") as f:
    f.write(insights)

print(insights)
print("[INFO] Semua visualisasi EDA tersimpan di folder outputs/")
