#!/usr/bin/env python3
"""
Aggregate TF-IDF vectors into per-author per-year representations.

For each domain (academic/blogs/news) and each split (human, per-provider LV),
this script reads the previously generated `tfidf_vectors.csv`, parses years
from filenames, averages multiple samples within the same year, and writes
`tfidf_author_yearly_vectors.csv` beside the source file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from utils.parse_dataset_filename import parse_filename

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")
INPUT_FILENAME = "tfidf_vectors.csv"
OUTPUT_FILENAME = "tfidf_author_yearly_vectors.csv"


def add_year_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Parse filename to extract year (handles human vs. llm formats)."""
    years: List[str] = []

    for _, row in df.iterrows():
        filename = row.get("filename")
        if not isinstance(filename, str):
            years.append(None)
            continue

        meta = parse_filename(filename, is_llm=(label == "llm"))
        years.append(meta["year"] if meta else None)

    df = df.copy()
    df["year"] = years
    df = df.dropna(subset=["year"])
    return df


def aggregate_vectors(df: pd.DataFrame, vector_prefix: str) -> pd.DataFrame:
    """Average vectors per (author, year) while retaining metadata."""
    vector_cols = [col for col in df.columns if col.startswith(vector_prefix)]
    if not vector_cols:
        raise ValueError(f"No columns found with prefix '{vector_prefix}'.")

    group_cols = [
        "label",
        "domain",
        "field",
        "author_id",
        "provider",
        "level",
        "model",
        "year",
    ]
    agg_df = (
        df.groupby(group_cols, dropna=False)[vector_cols]
        .mean()
        .reset_index()
    )
    return agg_df


def process_split(csv_path: Path) -> None:
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    label = df["label"].iloc[0]
    df = add_year_column(df, label=label)
    if df.empty:
        return

    aggregated = aggregate_vectors(df, vector_prefix="tfidf_")
    output_path = csv_path.with_name(OUTPUT_FILENAME)
    aggregated.to_csv(output_path, index=False)
    print(f"✅ {output_path}: {len(aggregated)} rows")


def process_domain(domain: str) -> None:
    # Human split
    human_csv = DATA_ROOT / "human" / domain / INPUT_FILENAME
    process_split(human_csv)

    # LLM splits
    for provider in PROVIDERS:
        for level in LEVELS:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / INPUT_FILENAME
            process_split(csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate TF-IDF vectors into per-author per-year averages."
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
    for domain in args.domains:
        print(f"\n=== Aggregating TF-IDF yearly vectors for {domain} ===")
        process_domain(domain)


if __name__ == "__main__":
    main()


