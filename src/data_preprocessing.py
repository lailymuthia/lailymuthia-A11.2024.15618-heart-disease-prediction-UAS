"""
=============================================================
DATA PREPROCESSING - Prediksi Penyakit Jantung (Heart Failure)
Nama : Laily Muthia N | NIM: A11.2024.15618
CRISP-DM Tahap 3: Data Preparation
=============================================================
Modul ini berisi seluruh fungsi untuk membersihkan, mengolah,
dan menyiapkan data sebelum masuk ke tahap modelling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

RAW_PATH = "data/raw/heart.csv"
PROCESSED_DIR = "data/processed"

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET = "HeartDisease"


def load_data(path=RAW_PATH):
    """Load dataset mentah dari CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Membersihkan data:
    - Cholesterol = 0 dan RestingBP = 0 secara klinis tidak mungkin
      (bukan missing value eksplisit, tapi representasi data hilang),
      sehingga diimputasi dengan median per kelompok (berdasarkan HeartDisease)
      agar tidak menghilangkan informasi antar kelas.
    - Hapus baris duplikat jika ada.
    """
    df = df.copy()

    # Cek & hapus duplikat
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df = df.drop_duplicates()

    # Tangani Cholesterol = 0 (172 baris di dataset ini secara klinis tidak valid)
    for col in ["Cholesterol", "RestingBP"]:
        df[col] = df[col].astype(float)
        zero_mask = df[col] == 0
        if zero_mask.sum() > 0:
            # imputasi median per kelas target supaya distribusi kelas tetap terjaga
            median_by_class = df[df[col] != 0].groupby(TARGET)[col].median()
            for cls, med_val in median_by_class.items():
                fix_mask = zero_mask & (df[TARGET] == cls)
                df.loc[fix_mask, col] = med_val
        df[col] = df[col].round(0).astype(int)

    return df


def feature_engineering(df):
    """
    Menambahkan fitur turunan sederhana yang relevan secara klinis:
    - AgeGroup: kelompok usia (kategorikal, untuk EDA)
    - HR_Reserve: MaxHR dikurangi estimasi HR maksimal (220-Age), indikator kebugaran jantung
    """
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 40, 50, 60, 100],
        labels=["<=40", "41-50", "51-60", ">60"]
    )
    df["HR_Reserve"] = (220 - df["Age"]) - df["MaxHR"]
    return df


def get_preprocessor():
    """
    Membuat ColumnTransformer: scaling untuk numerik, one-hot encoding untuk kategorikal.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="if_binary")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Membagi dataset menjadi train / validation / test dengan stratifikasi
    pada target (karena kelas sedikit tidak seimbang: 55.3% vs 44.7%).
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_relative,
        stratify=y_train_full, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline(save=True):
    """Menjalankan seluruh pipeline preprocessing end-to-end."""
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_fe = feature_engineering(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    if save:
        joblib.dump(preprocessor, "models/preprocessing.pkl")
        df_fe.to_csv(f"{PROCESSED_DIR}/heart_clean.csv", index=False)
        joblib.dump(
            (X_train, X_val, X_test, y_train, y_val, y_test),
            f"{PROCESSED_DIR}/train_test_split.pkl"
        )

    return {
        "df_raw": df_raw, "df_clean": df_clean, "df_fe": df_fe,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_proc": X_train_proc, "X_val_proc": X_val_proc, "X_test_proc": X_test_proc,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("[INFO] Preprocessing selesai.")
    print(f"       Train : {result['X_train'].shape}")
    print(f"       Val   : {result['X_val'].shape}")
    print(f"       Test  : {result['X_test'].shape}")
