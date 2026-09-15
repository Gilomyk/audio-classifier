# scripts/prepare_subset.py
"""
Losuje podzbiór korpusu MUSAN (~40-45 min audio na klasę) i kopiuje
wybrane pliki .wav do data/raw/subset/{music,speech,noise}/.

Uruchomienie (z głównego katalogu projektu):
    python scripts/prepare_subset.py
"""

import random
import shutil
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# --- Konfiguracja ---
SEED = 42
TARGET_MINUTES_PER_CLASS = 42  # środek zakresu 40-45
TARGET_SECONDS_PER_CLASS = TARGET_MINUTES_PER_CLASS * 60

RAW_MUSAN_DIR = Path("data/raw/musan")
OUTPUT_DIR = Path("data/raw/subset")

CLASSES = ["music", "speech", "noise"]


def list_wav_files(class_dir: Path) -> list[Path]:
    """Zwraca wszystkie pliki .wav w danej klasie (przeszukuje podfoldery źródeł)."""
    return list(class_dir.rglob("*.wav"))


def get_duration_seconds(path: Path) -> float:
    """Szybki odczyt długości pliku audio bez dekodowania całej zawartości."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


def select_subset(files: list[Path], target_seconds: float) -> list[Path]:
    """Losowo dobiera pliki aż do osiągnięcia docelowej sumy sekund."""
    shuffled = files.copy()
    random.shuffle(shuffled)

    selected = []
    total_seconds = 0.0

    for f in shuffled:
        if total_seconds >= target_seconds:
            break
        try:
            duration = get_duration_seconds(f)
        except Exception as e:
            print(f"  Pominięto {f} (błąd odczytu: {e})")
            continue

        selected.append(f)
        total_seconds += duration

    return selected, total_seconds


def main():
    random.seed(SEED)

    if not RAW_MUSAN_DIR.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {RAW_MUSAN_DIR}. Upewnij się, że MUSAN jest "
            f"wypakowany w data/raw/musan/"
        )

    for class_name in CLASSES:
        class_dir = RAW_MUSAN_DIR / class_name
        print(f"\n== Klasa: {class_name} ==")

        all_files = list_wav_files(class_dir)
        print(f"Znaleziono {len(all_files)} plików .wav")

        selected_files, total_seconds = select_subset(
            all_files, TARGET_SECONDS_PER_CLASS
        )
        print(
            f"Wybrano {len(selected_files)} plików, "
            f"łącznie {total_seconds / 60:.1f} min"
        )

        out_dir = OUTPUT_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in tqdm(selected_files, desc=f"Kopiowanie ({class_name})"):
            # Spłaszczamy strukturę - dodajemy prefiks źródła do nazwy,
            # żeby uniknąć kolizji nazw plików z różnych podfolderów.
            source_tag = f.parent.name
            dest_name = f"{source_tag}__{f.name}"
            shutil.copy2(f, out_dir / dest_name)

    print("\nGotowe. Podzbiór zapisany w data/raw/subset/")


if __name__ == "__main__":
    main()