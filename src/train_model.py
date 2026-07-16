"""
=============================================================
TRAINING & HYPERPARAMETER TUNING - Prediksi Penyakit Jantung
Nama : Laily Muthia N | NIM: A11.2024.15618
CRISP-DM Tahap 4: Modelling
=============================================================
Melatih 3 model (Random Forest, XGBoost, SVM) dengan GridSearchCV,
lalu memilih model terbaik berdasarkan F1-score pada data validasi.
"""
import sys
sys.path.append("src")
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

from data_preprocessing import run_pipeline, get_preprocessor

RANDOM_STATE = 42


def build_pipelines():
    """Membangun pipeline (preprocessing + model) untuk tiap algoritma."""
    pipelines = {
        "RandomForest": Pipeline([
            ("prep", get_preprocessor()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
        ]),
        "XGBoost": Pipeline([
            ("prep", get_preprocessor()),
            ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"))
        ]),
        "SVM": Pipeline([
            ("prep", get_preprocessor()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
    }
    return pipelines


def param_grids():
    """Grid hyperparameter untuk tiap model (GridSearchCV)."""
    return {
        "RandomForest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10, 15],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
        },
        "XGBoost": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
        },
        "SVM": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
    }


def train_all(X_train, y_train, X_val, y_val, cv=5):
    pipelines = build_pipelines()
    grids = param_grids()
    results = {}
    fitted_models = {}

    for name, pipe in pipelines.items():
        print(f"\n[TRAINING] {name} ...")
        gs = GridSearchCV(pipe, grids[name], cv=cv, scoring="f1", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        val_pred = best_model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred)

        results[name] = {
            "best_params": gs.best_params_,
            "cv_best_f1": gs.best_score_,
            "val_f1": val_f1,
            "val_accuracy": val_acc,
        }
        fitted_models[name] = best_model
        print(f"  Best CV F1   : {gs.best_score_:.4f}")
        print(f"  Val F1       : {val_f1:.4f}")
        print(f"  Val Accuracy : {val_acc:.4f}")
        print(f"  Best Params  : {gs.best_params_}")

    return fitted_models, results


if __name__ == "__main__":
    data = run_pipeline(save=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    fitted_models, results = train_all(X_train, y_train, X_val, y_val)

    # Pilih model terbaik berdasarkan F1-score validasi
    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best_model = fitted_models[best_name]

    print(f"\n{'='*60}")
    print(f"MODEL TERBAIK: {best_name} (Val F1 = {results[best_name]['val_f1']:.4f})")
    print(f"{'='*60}")

    # Simpan semua model & hasil
    for name, model in fitted_models.items():
        joblib.dump(model, f"models/{name}.pkl")
    joblib.dump(best_model, "models/best_model.pkl")
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    with open("outputs/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    results_df = pd.DataFrame(results).T
    results_df.to_csv("outputs/training_results.csv")
    print("\n[INFO] Semua model & hasil tuning tersimpan di folder models/ dan outputs/")
