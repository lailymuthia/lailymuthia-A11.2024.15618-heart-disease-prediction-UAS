"""
Evaluasi & Interpretasi Model - Prediksi Penyakit Jantung
Nama : Laily Muthia N | NIM: A11.2024.15618
"""
import sys
sys.path.append("src")
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import shap

from data_preprocessing import run_pipeline

OUT = "outputs"

data = run_pipeline(save=False)
X_test, y_test = data["X_test"], data["y_test"]
X_train = data["X_train"]

models = {
    "RandomForest": joblib.load("models/RandomForest.pkl"),
    "XGBoost": joblib.load("models/XGBoost.pkl"),
    "SVM": joblib.load("models/SVM.pkl"),
}
with open("models/best_model_name.txt") as f:
    best_name = f.read().strip()
best_model = joblib.load("models/best_model.pkl")

# 1. Tabel perbandingan metrik semua model (test set)
rows = []
roc_data = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    rows.append(row)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, row["ROC-AUC"])

metrics_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
metrics_df.to_csv(f"{OUT}/model_comparison_metrics.csv", index=False)
print(metrics_df.to_string(index=False))

# 2. Tabel perbandingan (gambar)
fig, ax = plt.subplots(figsize=(9, 2.2))
ax.axis("off")
tbl = ax.table(cellText=metrics_df.round(4).values, colLabels=metrics_df.columns,
                cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
for j in range(len(metrics_df.columns)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")
plt.title("Tabel Perbandingan Performa Model (Test Set)", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUT}/07_tabel_perbandingan_model.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Confusion matrix semua model
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Sakit"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
plt.suptitle("Confusion Matrix - Perbandingan Model", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/08_confusion_matrix.png", dpi=150)
plt.close()

# 4. ROC Curve semua model
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"RandomForest": "#3498db", "XGBoost": "#e74c3c", "SVM": "#2ecc71"}
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name], linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Perbandingan Model", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/09_roc_curve.png", dpi=150)
plt.close()

# 5. Feature importance (model terbaik, jika tree-based)
prep = best_model.named_steps["prep"]
feature_names = prep.get_feature_names_out()
clf = best_model.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#8e44ad")
    ax.set_title(f"Feature Importance - {best_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT}/10_feature_importance.png", dpi=150)
    plt.close()

# 6. SHAP - interpretasi model terbaik
X_test_transformed = prep.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

try:
    if best_name in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        X_train_transformed = prep.transform(X_train)
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
        background = pd.DataFrame(X_train_transformed, columns=feature_names).sample(50, random_state=42)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_test_df.sample(50, random_state=42))[1]
        X_test_df = X_test_df.sample(50, random_state=42)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title(f"SHAP Summary Plot - {best_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/11_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] SHAP summary plot berhasil dibuat.")
except Exception as e:
    print(f"[WARNING] SHAP gagal dibuat: {e}")

# 7. Classification report model terbaik (teks)
y_pred_best = best_model.predict(X_test)
report = classification_report(y_test, y_pred_best, target_names=["Normal", "Sakit"])
with open(f"{OUT}/classification_report_best_model.txt", "w") as f:
    f.write(f"MODEL TERBAIK: {best_name}\n\n")
    f.write(report)
print(f"\nClassification Report ({best_name}):\n{report}")

print("\n[INFO] Evaluasi selesai. Semua hasil tersimpan di folder outputs/")