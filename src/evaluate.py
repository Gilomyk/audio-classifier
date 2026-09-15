# src/evaluate.py
"""
Ewaluacja wytrenowanego modelu: accuracy, macierz pomyłek,
raport klasyfikacji na zbiorze testowym.

Wejście:  models/random_forest.pkl, models/scaler.pkl,
          data/processed/test_features.csv
Wyjście:  reports/confusion_matrix.png, reports/feature_importance.png

Uruchomienie (z głównego katalogu projektu):
    python src/evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

TEST_FEATURES_PATH = Path("data/processed/test_features.csv")
MODEL_PATH = Path("models/random_forest.pkl")
SCALER_PATH = Path("models/scaler.pkl")
REPORTS_DIR = Path("reports")


def load_test_data():
    df = pd.read_csv(TEST_FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ("label", "source_file")]
    X_test = df[feature_cols]
    y_test = df["label"]
    return X_test, y_test, feature_cols


def plot_confusion_matrix(y_test, y_pred, labels, output_path: Path):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Zapisano macierz pomyłek do {output_path}")


def plot_feature_importance(model, feature_cols, output_path: Path, top_n=15):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_cols[i] for i in sorted_idx][::-1],
        importances[sorted_idx][::-1],
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Most Important Features")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Zapisano wykres ważności cech do {output_path}")


def main():
    X_test, y_test, feature_cols = load_test_data()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = sorted(y_test.unique())
    plot_confusion_matrix(y_test, y_pred, labels, REPORTS_DIR / "confusion_matrix.png")
    plot_feature_importance(
        model, feature_cols, REPORTS_DIR / "feature_importance.png"
    )


if __name__ == "__main__":
    main()