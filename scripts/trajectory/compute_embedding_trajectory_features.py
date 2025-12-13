#!/usr/bin/env python3
"""
Compute trajectory features (local drift + global geometry) from
per-author yearly embedding vectors (TF-IDF or SBERT).

Usage:
    python scripts/micro/trajectory/compute_embedding_trajectory_features.py \
        --space tfidf
    python scripts/micro/trajectory/compute_embedding_trajectory_features.py \
        --space sbert --domains blogs news
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")


def compute_yearly_metrics(vectors: np.ndarray) -> dict:
    """Given ordered yearly vectors, compute drift + geometry."""
    if len(vectors) < 2:
        return {
            "mean_distance": math.nan,
            "std_distance": math.nan,
            "net_displacement": math.nan,
            "path_length": math.nan,
            "tortuosity": math.nan,
            "direction_consistency": math.nan,
            "n_years": len(vectors),
        }

    diffs = vectors[1:] - vectors[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    mean_distance = float(np.mean(dists))
    std_distance = float(np.std(dists))

    net_displacement = float(np.linalg.norm(vectors[-1] - vectors[0]))
    path_length = float(np.sum(dists))
    tortuosity = float(path_length / net_displacement) if net_displacement > 0 else 1.0

    if len(diffs) >= 2:
        v1 = diffs[:-1]
        v2 = diffs[1:]
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        valid = denom > 0
        cos_vals = np.empty(len(denom))
        cos_vals[:] = np.nan
        cos_vals[valid] = np.sum(v1[valid] * v2[valid], axis=1) / denom[valid]
        direction_consistency = float(np.nanmean(cos_vals))
    else:
        direction_consistency = math.nan

    return {
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "net_displacement": net_displacement,
        "path_length": path_length,
        "tortuosity": tortuosity,
        "direction_consistency": direction_consistency,
        "n_years": len(vectors),
    }


def process_split(csv_path: Path, output_name: str, vector_prefix: str) -> None:
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    vector_cols = [col for col in df.columns if col.startswith(vector_prefix)]
    if not vector_cols:
        raise ValueError(f"No columns with prefix '{vector_prefix}' in {csv_path}")

    rows: List[dict] = []
    for (author_id, provider, level, model, label, domain, field), group in df.groupby(
        ["author_id", "provider", "level", "model", "label", "domain", "field"],
        dropna=False,
    ):
        ordered = group.sort_values("year")
        vectors = ordered[vector_cols].to_numpy(dtype=float)
        metrics = compute_yearly_metrics(vectors)
        rows.append(
            {
                "label": label,
                "domain": domain,
                "field": field,
                "author_id": author_id,
                "provider": provider,
                "level": level,
                "model": model,
                **metrics,
            }
        )

    output_df = pd.DataFrame(rows)
    output_path = csv_path.with_name(output_name)
    output_df.to_csv(output_path, index=False)
    print(f"✅ {output_path}: {len(output_df)} authors")


def process_domain(domain: str, space: str) -> None:
    if space == "tfidf":
        input_name = "tfidf_author_yearly_vectors.csv"
        output_name = "tfidf_trajectory_features.csv"
        prefix = "tfidf_"
    elif space == "sbert":
        input_name = "sbert_author_yearly_vectors.csv"
        output_name = "sbert_trajectory_features.csv"
        prefix = "sbert_"
    else:
        raise ValueError("space must be 'tfidf' or 'sbert'")

    # Human
    human_csv = DATA_ROOT / "human" / domain / input_name
    process_split(human_csv, output_name, prefix)

    for provider in PROVIDERS:
        for level in LEVELS:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / input_name
            process_split(csv_path, output_name, prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute embedding trajectory features.")
    parser.add_argument(
        "--space",
        choices=("tfidf", "sbert"),
        required=True,
        help="Which embedding space to process.",
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
        print(f"\n=== Computing {args.space.upper()} trajectory features for {domain} ===")
        process_domain(domain, args.space)


if __name__ == "__main__":
    main()


