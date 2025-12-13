#!/usr/bin/env python3
"""
Assemble the unified trajectory feature table (~72 dims per author) by joining
CE variability stats with CE / TF-IDF / SBERT geometry metrics.

Outputs `trajectory_features_combined.csv` per domain/provider/level split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")
KEY_COLS = ["label", "domain", "field", "author_id", "provider", "level", "model"]


def ensure_key_columns(df: pd.DataFrame, defaults: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
        else:
            df[col] = df[col].fillna(value)
    return df


def prefix_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in KEY_COLS]
    renamed = df[KEY_COLS + metric_cols].rename(
        columns={c: f"{prefix}{c}" for c in metric_cols}
    )
    return renamed


def load_base_stats(path: Path, defaults: Dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["domain"] = defaults["domain"]
    df["label"] = defaults["label"]
    df["provider"] = defaults["provider"]
    df["level"] = defaults["level"]
    df["model"] = defaults["model"]
    return df


def join_features(domain: str, provider: str, level: str, label: str) -> None:
    base_path = (
        DATA_ROOT / "human" / domain / "author_timeseries_stats_merged.csv"
        if label == "human"
        else DATA_ROOT / "LLM" / provider / level / domain / "author_timeseries_stats_merged.csv"
    )
    base_df = load_base_stats(
        base_path,
        defaults={
            "domain": domain,
            "label": label,
            "provider": provider,
            "level": level,
            "model": provider,
        },
    )
    if base_df.empty:
        return

    ce_geom_path = base_path.with_name("ce_trajectory_features.csv")
    tfidf_path = base_path.with_name("tfidf_trajectory_features.csv")
    sbert_path = base_path.with_name("sbert_trajectory_features.csv")

    merges: List[pd.DataFrame] = []
    for path, prefix in [
        (ce_geom_path, "ce_"),
        (tfidf_path, "tfidf_"),
        (sbert_path, "sbert_"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = ensure_key_columns(
            df,
            {
                "label": label,
                "domain": domain,
                "provider": provider,
                "level": level,
                "model": provider,
            },
        )
        merges.append(prefix_metrics(df, prefix))

    result = ensure_key_columns(
        base_df,
        {
            "label": label,
            "domain": domain,
            "provider": provider,
            "level": level,
            "model": provider,
        },
    )
    for feat_df in merges:
        result = result.merge(feat_df, on=KEY_COLS, how="left")

    output_path = base_path.with_name("trajectory_features_combined.csv")
    result.to_csv(output_path, index=False)
    print(f"✅ {output_path}: {len(result)} authors")


def process_domain(domain: str) -> None:
    join_features(domain, provider="human", level="LV0", label="human")
    for provider in PROVIDERS:
        for level in LEVELS:
            join_features(domain, provider=provider, level=level, label="llm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified trajectory feature tables.")
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
        print(f"\n=== Building combined trajectory features for {domain} ===")
        process_domain(domain)


if __name__ == "__main__":
    main()


