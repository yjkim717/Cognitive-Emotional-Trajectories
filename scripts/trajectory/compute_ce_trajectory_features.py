#!/usr/bin/env python3
"""
Compute trajectory geometry (D / L / τ / direction cosines) for the CE feature space.

Reads `combined_merged.csv` for each domain + split, aggregates yearly means of the
handcrafted features per author, then applies the same trajectory metrics used for
TF-IDF / SBERT.
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

from utils.parse_dataset_filename import parse_filename

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")
INPUT_NAME = "combined_merged.csv"
OUTPUT_NAME = "ce_trajectory_features.csv"
META_COLS = {
    "filename",
    "path",
    "label",
    "domain",
    "field",
    "author_id",
    "model",
    "level",
    "provider",
    "sample_count",
}


def compute_yearly_metrics(vectors: np.ndarray) -> dict:
    """Same geometry metrics as embedding trajectory."""
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


def extract_year(row: pd.Series) -> str | None:
    filename = row.get("filename")
    if not isinstance(filename, str):
        return None
    meta = parse_filename(filename, is_llm=row.get("label") == "llm")
    return meta["year"] if meta else None


def determine_features(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in META_COLS]


def process_split(csv_path: Path, label_override: str | None = None, provider_override: str | None = None, level_override: str | None = None) -> None:
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    df["label"] = df["label"].fillna(label_override or "human")
    df["provider"] = df.get("provider", pd.Series(dtype=object)).fillna(provider_override or df["label"])
    df["level"] = df.get("level", pd.Series(dtype=object)).fillna(level_override or ("LV0" if label_override == "human" else None))

    df["year"] = df.apply(extract_year, axis=1)
    df = df.dropna(subset=["year", "author_id"])
    feature_cols = determine_features(df)
    if not feature_cols:
        return

    yearly = df.groupby(
        ["author_id", "provider", "level", "model", "label", "domain", "field", "year"],
        dropna=False,
    )[feature_cols].mean().reset_index()

    rows: List[dict] = []
    for (author_id, provider, level, model, label, domain, field), group in yearly.groupby(
        ["author_id", "provider", "level", "model", "label", "domain", "field"],
        dropna=False,
    ):
        ordered = group.sort_values("year")
        vectors = ordered[feature_cols].to_numpy(dtype=float)
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
    output_path = csv_path.with_name(OUTPUT_NAME)
    output_df.to_csv(output_path, index=False)
    print(f"✅ {output_path}: {len(output_df)} authors")


def process_domain(domain: str) -> None:
    human_csv = DATA_ROOT / "human" / domain / INPUT_NAME
    process_split(human_csv, label_override="human", provider_override="human", level_override="LV0")

    for provider in PROVIDERS:
        for level in LEVELS:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / INPUT_NAME
            process_split(csv_path, label_override="llm", provider_override=provider, level_override=level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CE trajectory geometry from combined_merged.csv")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for domain in args.domains:
        print(f"\n=== Computing CE trajectory features for {domain} ===")
        process_domain(domain)


if __name__ == "__main__":
    main()

