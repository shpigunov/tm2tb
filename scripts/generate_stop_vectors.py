"""
Generate averaged stopword embedding vectors for a given language.

Usage:
    python scripts/generate_stop_vectors.py --lang uk

The script reads stopwords from a language-specific file under data/,
encodes them with the LaBSE sentence-transformer model, averages the
vectors, and saves the result as a (1, 768) float32 .npy file under
stops_vectors/768/.

Ukrainian stopwords sourced from:
    skupriienko/Ukrainian-Stopwords
    https://github.com/skupriienko/Ukrainian-Stopwords
"""

import argparse
import os
import sys

import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_stopwords_file(lang):
    """Locate the stopwords file for `lang` under data/."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    candidates = [
        os.path.join(data_dir, f"stopwords_{lang}.txt"),
        os.path.join(data_dir, f"{lang}.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No stopwords file found for '{lang}' in {data_dir}. Tried: {candidates}"
    )


def generate(lang):
    stopwords_path = find_stopwords_file(lang)
    print(f"Reading stopwords from: {stopwords_path}")

    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(stopwords)} stopwords for '{lang}'")

    from tm2tb import trf_model

    print("Encoding stopwords...")
    vectors = trf_model.encode(stopwords)  # shape: (N, 768)

    avg_vector = vectors.mean(axis=0, keepdims=True).astype(
        np.float32
    )  # shape: (1, 768)
    print(f"Averaged vector shape: {avg_vector.shape}")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "stops_vectors",
        "768",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang}.npy")
    np.save(out_path, avg_vector)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stopword embedding for a language."
    )
    parser.add_argument(
        "--lang", required=True, help="Two-character language code (e.g. 'uk')"
    )
    args = parser.parse_args()
    generate(args.lang)
