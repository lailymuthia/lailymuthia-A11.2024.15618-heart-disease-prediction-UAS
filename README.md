# Prediksi Risiko Penyakit Jantung — Capstone Project Machine Learning

**Nama**: Laily Muthia N
**NIM**: A11.2024.15618
**Program Studi**: Teknik Informatika

## Deskripsi Proyek
Proyek Capstone UAS Mata Kuliah Pembelajaran Mesin yang mengimplementasikan pipeline
Machine Learning end-to-end - dari akuisisi data, EDA, preprocessing, modelling,
evaluasi, hingga deployment - menggunakan metodologi CRISP-DM, untuk memprediksi
risiko penyakit jantung (heart disease) berdasarkan data klinis pasien.

## Dataset
Heart Failure Prediction Dataset (Kaggle - fedesoriano)
918 baris, 11 fitur klinis + 1 target biner (`HeartDisease`).
Link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## Metodologi: CRISP-DM
| Tahap | Implementasi |
|---|---|
| 1. Business Understanding | Problem statement: klasifikasi risiko penyakit jantung untuk skrining dini |
| 2. Data Understanding | `notebooks/01_eda.ipynb`, `src/eda.py` |
| 3. Data Preparation | `src/data_preprocessing.py` |
| 4. Modelling | `notebooks/02_modeling.ipynb`, `src/train_model.py` |
| 5. Evaluation | `notebooks/03_interpretation.ipynb`, `src/evaluate_model.py` |
| 6. Deployment | `app/app.py` (Streamlit) |

## Model & Hasil
3 algoritma dibandingkan dengan hyperparameter tuning (GridSearchCV):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (terbaik) | 0.913 | 0.930 | 0.912 | 0.921 | 0.947 |
| Random Forest | 0.859 | 0.858 | 0.892 | 0.875 | 0.932 |
| SVM | 0.853 | 0.857 | 0.882 | 0.870 | 0.932 |

Model terbaik: XGBoost (F1-Score tertinggi pada data test).

## Struktur Repository
```
capstone-heart-disease/
├── data/
│   ├── raw/heart.csv              # data mentah
│   └── processed/                 # data hasil cleaning & split
├── notebooks/
│   ├── 01_eda.ipynb                # EDA & preprocessing
│   ├── 02_modeling.ipynb           # training & tuning model
│   └── 03_interpretation.ipynb     # evaluasi & interpretasi (SHAP)
├── src/
│   ├── data_preprocessing.py       # pipeline cleaning, FE, split
│   ├── eda.py                      # generator visualisasi EDA
│   ├── train_model.py              # training 3 model + GridSearchCV
│   └── evaluate_model.py           # evaluasi & interpretasi model
├── models/
│   ├── best_model.pkl               # model terbaik (XGBoost)
│   ├── preprocessing.pkl
│   └── RandomForest.pkl / XGBoost.pkl / SVM.pkl
├── app/
│   └── app.py                       # aplikasi Streamlit
├── outputs/                         # seluruh visualisasi & tabel hasil
├── reports/
│   └── Laporan_Teknis_UAS.docx
├── requirements.txt
└── README.md
```

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan pipeline (opsional, model sudah tersedia di /models)
```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
```

### 3. Jalankan aplikasi Streamlit
```bash
streamlit run app/app.py
```

## Fitur Aplikasi Streamlit
- Dashboard EDA — eksplorasi data interaktif
- Model Demo — prediksi risiko pasien baru secara real-time
- Evaluasi Model — perbandingan performa 3 model
- Interpretasi Hasil — feature importance & SHAP
- Dokumentasi — penjelasan dataset, metodologi, dan cara pakai

## Disclaimer
Aplikasi ini dibuat untuk tujuan akademik. Hasil prediksi bukan diagnosis medis
dan tidak menggantikan konsultasi dengan tenaga medis profesional.