#!/usr/bin/env python3
"""
Extract SBERT vectors using E5-Large model for micro datasets.

This script extracts embeddings using the E5-Large model (intfloat/e5-large-v2)
as an alternative to all-MiniLM-L6-v2 for comparison testing.

Output files are saved as sbert_e5_vectors.csv to distinguish from
the original sbert_vectors.csv (which uses all-MiniLM-L6-v2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils.file_utils import read_text as read_utf8_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
LEVELS = ("LV1", "LV2", "LV3")
MAX_DOC_CHARS = 8_000
MODEL_NAME = "intfloat/e5-large-v2"  # E5-Large model
OUTPUT_FILENAME = "sbert_e5_vectors.csv"  # Different filename to avoid overwriting


def load_text(rel_path: str) -> str:
    """Read UTF-8 text from project-relative path with truncation."""
    abs_path = PROJECT_ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"Missing text file: {abs_path}")
    text = read_utf8_text(str(abs_path))
    return text[:MAX_DOC_CHARS]


def collect_entries(domain: str) -> List[Dict]:
    """Collect all samples for a domain (human + each provider/level)."""
    entries: List[Dict] = []

    def _load_csv(csv_path: Path, label: str, provider: str, level: str) -> None:
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel_path = row.get("path")
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue

            try:
                text = load_text(rel_path)
            except FileNotFoundError:
                continue

            metadata = {
                "filename": row.get("filename"),
                "path": rel_path,
                "label": label,
                "domain": row.get("domain", domain),
                "field": row.get("field"),
                "author_id": row.get("author_id"),
                "provider": provider,
                "level": level,
                "model": row.get("model"),
            }

            out_path = (
                DATA_ROOT / "human" / domain / OUTPUT_FILENAME
                if label == "human"
                else DATA_ROOT / "LLM" / provider / level / domain / OUTPUT_FILENAME
            )

            metadata["output_path"] = str(out_path)
            entries.append(
                {
                    "text": text,
                    "metadata": metadata,
                }
            )

    # Human split
    human_csv = DATA_ROOT / "human" / domain / "combined_merged.csv"
    _load_csv(human_csv, label="human", provider="human", level="LV0")

    # LLM splits - ONLY LV3
    for provider in PROVIDERS:
        level = "LV3"  # Only extract LV3 data
        csv_path = DATA_ROOT / "LLM" / provider / level / domain / "combined_merged.csv"
        _load_csv(csv_path, label="llm", provider=provider, level=level)

    return entries


def write_outputs(entries: List[Dict], vectors) -> None:
    """Write SBERT vectors grouped by output path.
    
    Uses 'sbert_e5_' prefix for column names to distinguish from
    the original all-MiniLM-L6-v2 embeddings (which use 'sbert_' prefix).
    """
    vectors_arr = np.asarray(vectors)
    if vectors_arr.ndim != 2:
        raise ValueError("Expected 2D array of SBERT vectors.")

    dim = vectors_arr.shape[1]
    vector_columns = [f"sbert_e5_{i+1}" for i in range(dim)]  # Use sbert_e5_ prefix
    vectors_df = pd.DataFrame(vectors_arr, columns=vector_columns)
    meta_df = pd.DataFrame([entry["metadata"] for entry in entries])
    combined_df = pd.concat([meta_df, vectors_df], axis=1)

    for output_path, group in combined_df.groupby("output_path"):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["output_path"]).to_csv(output_file, index=False)
        print(f"✅ {output_file}: {len(group)} samples")


def process_domain(domain: str) -> None:
    """Process a single domain to extract E5-Large embeddings."""
    entries = collect_entries(domain)
    if not entries:
        print(f"⚠ No entries found for domain '{domain}'. Skipping.")
        return

    texts = [entry["text"] for entry in entries]
    
    print(f"Loading E5-Large model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    write_outputs(entries, embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SBERT vectors using E5-Large model for micro datasets."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 80)
    print("E5-Large SBERT Vector Extraction (LV3 only)")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Output filename: {OUTPUT_FILENAME}")
    print(f"Column prefix: sbert_e5_")
    print(f"Level: LV3 only (for LLM)")
    print("=" * 80)
    
    for domain in args.domains:
        print(f"\n=== Processing domain: {domain} ===")
        process_domain(domain)
    
    print("\n" + "=" * 80)
    print("✅ E5-Large embedding extraction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

