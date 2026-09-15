# src/train.py
"""
Trening klasyfikatora (Random Forest) na cechach wyekstrahowanych
z podzbioru MUSAN. Zapisuje wytrenowany model do models/.

Wejście:  data/processed/features.csv
Wyjście:  models/random_forest.pkl, models/scaler.pkl,
          data/processed/test_features.csv (zbiór testowy do evaluate.py)

Uruchomienie (z głównego katalogu projektu):
    python src/train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# --- Konfiguracja ---
SEED = 42
TEST_SIZE = 0.2

FEATURES_PATH = Path("data/processed/features.csv")
TEST_OUTPUT_PATH = Path("data/processed/test_features.csv")
MODEL_PATH = Path("models/random_forest.pkl")
SCALER_PATH = Path("models/scaler.pkl")


def load_data():
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [c for c in df.columns if c not in ("label", "source_file")]
    X = df[feature_cols]
    y = df["label"]
    groups = df["source_file"]

    return df, X, y, groups, feature_cols


def split_by_file(df, X, y, groups):
    """Dzieli dane tak, żeby segmenty z tego samego pliku nie trafiły
    jednocześnie do treningu i testu (unikamy data leakage)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    return train_df, test_df


def main():
    df, X, y, groups, feature_cols = load_data()
    print(f"Wczytano {len(df)} segmentów, {len(feature_cols)} cech")
    print(f"Liczba unikalnych plików źródłowych: {groups.nunique()}")

    train_df, test_df = split_by_file(df, X, y, groups)
    print(f"\nTrain: {len(train_df)} segmentów")
    print(train_df["label"].value_counts())
    print(f"\nTest: {len(test_df)} segmentów")
    print(test_df["label"].value_counts())

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]

    # Skalowanie - dopasowujemy TYLKO na treningu, żeby nie "podglądać" testu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"\nTrain accuracy: {train_accuracy:.3f}")

    # Zapis artefaktów
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"\nZapisano model do {MODEL_PATH}")
    print(f"Zapisano scaler do {SCALER_PATH}")
    print(f"Zapisano zbiór testowy do {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()