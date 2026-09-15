# src/features.py
"""
Ekstrakcja cech audio (MFCC, spectral centroid, ZCR, RMS) do klasyfikacji
speech / music / noise.

Wejście:  data/raw/subset/{music,speech,noise}/*.wav
Wyjście:  data/processed/features.csv

Uruchomienie (z głównego katalogu projektu):
    python src/features.py
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Konfiguracja ---
SAMPLE_RATE = 22050
SEGMENT_DURATION = 4.0  # sekundy
MIN_SEGMENT_DURATION = 2.0  # krótsze końcówki odrzucamy
N_MFCC = 13

SUBSET_DIR = Path("data/raw/subset")
OUTPUT_PATH = Path("data/processed/features.csv")

CLASSES = ["music", "speech", "noise"]


def segment_audio(y: np.ndarray, sr: int) -> list[np.ndarray]:
    """Dzieli sygnał audio na segmenty o stałej długości (bez nakładania)."""
    segment_len_samples = int(SEGMENT_DURATION * sr)
    min_len_samples = int(MIN_SEGMENT_DURATION * sr)

    segments = []
    for start in range(0, len(y), segment_len_samples):
        end = start + segment_len_samples
        chunk = y[start:end]
        if len(chunk) >= min_len_samples:
            segments.append(chunk)
    return segments


def extract_features(segment: np.ndarray, sr: int) -> dict:
    """Liczy wektor cech (mean + std każdej cechy) dla pojedynczego segmentu."""
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=segment)
    rms = librosa.feature.rms(y=segment)

    features = {}

    for i in range(N_MFCC):
        features[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc{i+1}_std"] = np.std(mfcc[i])

    features["centroid_mean"] = np.mean(centroid)
    features["centroid_std"] = np.std(centroid)

    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    return features


def process_file(file_path: Path, label: str) -> list[dict]:
    """Ładuje plik, dzieli na segmenty, liczy cechy dla każdego segmentu."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    segments = segment_audio(y, sr)

    rows = []
    for segment in segments:
        feats = extract_features(segment, sr)
        feats["label"] = label
        feats["source_file"] = file_path.name
        rows.append(feats)

    return rows


def main():
    if not SUBSET_DIR.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {SUBSET_DIR}. Uruchom najpierw scripts/prepare_subset.py"
        )

    all_rows = []

    for class_name in CLASSES:
        class_dir = SUBSET_DIR / class_name
        wav_files = list(class_dir.glob("*.wav"))
        print(f"\n== Klasa: {class_name} ({len(wav_files)} plików) ==")

        for file_path in tqdm(wav_files, desc=f"Przetwarzanie ({class_name})"):
            try:
                rows = process_file(file_path, class_name)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  Pominięto {file_path.name} (błąd: {e})")

    df = pd.DataFrame(all_rows)
    print(f"\nŁącznie {len(df)} segmentów, {df.shape[1]} kolumn")
    print(df["label"].value_counts())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nZapisano do {OUTPUT_PATH}")


if __name__ == "__main__":
    main()