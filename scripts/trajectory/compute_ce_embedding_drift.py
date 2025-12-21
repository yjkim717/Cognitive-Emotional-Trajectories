#!/usr/bin/env python3
"""
Compute CE embedding drift using L2 norm (squared) as shown in the image.

Methodology:
1. Measure the drift of human embedding with L2 norm:
   drifth,i+1 = ||embh,i+1 – embh,i||2^2, i = 2020, ..., 2023
2. Measure the drift of AI embedding with L2 norm:
   driftAI,i+1 = ||embAI,i+1 – embAI,i||2^2, i = 2020, ..., 2023
3. Check whether the drift is consistently smaller for AI:
   drifth,i+1 > driftAI,i+1, i = 2020, ..., 2023

The 20 CE features are treated as a 20D vector:
- Big Five (5): Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- Sentiment (2): polarity, subjectivity
- VADER (4): vader_compound, vader_pos, vader_neu, vader_neg
- Readability/Length (6): word_diversity, flesch_reading_ease, gunning_fog, 
                          average_word_length, num_words, avg_sentence_length
- Word ratios (3): verb_ratio, function_word_ratio, content_word_ratio
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVEL = "LV3"

# 20 CE features (in order)
CE_FEATURES = [
    # Big Five (5)
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    # Sentiment (2)
    "polarity",
    "subjectivity",
    # VADER (4)
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
    # Readability/Length (6)
    "word_diversity",
    "flesch_reading_ease",
    "gunning_fog",
    "average_word_length",
    "num_words",
    "avg_sentence_length",
    # Word ratios (3)
    "verb_ratio",
    "function_word_ratio",
    "content_word_ratio",
]


def parse_year_from_filename(filename: str) -> int | None:
    """Parse year from filename like: Academic_CS_09_2020_01.txt -> 2020"""
    if not filename or not isinstance(filename, str):
        return None
    match = re.search(r"(\d{4})_\d{2}", filename)
    if match:
        return int(match.group(1))
    return None


def compute_l2_squared_diff(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute squared L2 norm of difference: ||vec1 - vec2||2^2"""
    diff = vec1 - vec2
    return float(np.sum(diff ** 2))


def load_combined_data(domains: List[str]) -> pd.DataFrame:
    """Load combined_with_embeddings data only."""
    frames: List[pd.DataFrame] = []
    
    for domain in domains:
        # Human: only use combined_with_embeddings
        human_path = DATA_ROOT / "human" / domain / "combined_with_embeddings.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["label"] = "human"
            df_h["domain"] = domain
            df_h["provider"] = "human"
            df_h["level"] = "LV0"
            df_h["model"] = "human"
            frames.append(df_h)
            print(f"  Human {domain}: using combined_with_embeddings.csv")
        else:
            print(f"  ⚠️  Human {domain}: combined_with_embeddings.csv not found, skipping")
        
        # LLM: LV3 only, only use combined_with_embeddings
        for provider in PROVIDERS:
            llm_path = DATA_ROOT / "LLM" / provider / "LV3" / domain / "combined_with_embeddings.csv"
            if llm_path.exists():
                df_l = pd.read_csv(llm_path)
                df_l["label"] = "llm"
                df_l["domain"] = domain
                df_l["provider"] = provider
                df_l["level"] = "LV3"
                df_l["model"] = provider
                frames.append(df_l)
                print(f"  LLM {provider}/LV3/{domain}: using combined_with_embeddings.csv")
            else:
                print(f"  ⚠️  LLM {provider}/LV3/{domain}: combined_with_embeddings.csv not found, skipping")
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def compute_drift_per_author(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drift for each author's trajectory using year-centroid method.
    Before computing drift, z-score normalize the 20 CE features across all author×year samples.
    Returns dataframe with columns: author_id, domain, field, label, provider, level, model, 
                                     year_from, year_to, drift
    """
    # Parse year from filename
    if "filename" not in df.columns:
        raise ValueError("No 'filename' column found")
    
    df = df.copy()
    df["year"] = df["filename"].apply(parse_year_from_filename)
    # Fix: reset_index after dropna to ensure proper numpy indexing
    df = df.dropna(subset=["year", "author_id"]).reset_index(drop=True)
    
    # Ensure all CE features exist
    missing_features = [f for f in CE_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing CE features: {missing_features}")
    
    # Fix: Compute year-centroid first (author × year mean)
    yearly_group_cols = [
        "author_id", "field", "domain", "label", "provider", "level", "model", "year"
    ]
    yearly = (
        df.groupby(yearly_group_cols, dropna=False)[CE_FEATURES]
        .mean()
        .reset_index()
    )
    
    # Z-score normalization: normalize each dimension (feature) across all author×year centroids
    # Shape: (n_yearly_centroids, 20)
    ce_data = yearly[CE_FEATURES].fillna(0.0).to_numpy()
    ce_mean = np.mean(ce_data, axis=0)  # Mean for each of 20 features
    ce_std = np.std(ce_data, axis=0)   # Std for each of 20 features
    # Avoid division by zero
    ce_std[ce_std == 0] = 1.0
    # Normalize: (x - mean) / std
    ce_data_normalized = (ce_data - ce_mean) / ce_std
    
    # Fix: Write normalized features back to dataframe to avoid index issues
    yearly_norm = yearly.copy()
    yearly_norm[CE_FEATURES] = ce_data_normalized
    
    # Group by author trajectory (without year)
    traj_group_cols = ["author_id", "field", "domain", "label", "provider", "level", "model"]
    results: List[Dict] = []
    
    for key, group in yearly_norm.groupby(traj_group_cols, dropna=False):
        (author_id, field, domain, label, provider, level, model) = key
        
        # Sort by year
        group_sorted = group.sort_values("year")
        group_years = group_sorted["year"].values
        
        # Get normalized CE vectors directly from dataframe (no index issues)
        group_ce = group_sorted[CE_FEATURES].to_numpy()
        
        # Compute drift between consecutive years
        for i in range(len(group_years) - 1):
            year_from = int(group_years[i])
            year_to = int(group_years[i + 1])
            vec_from = group_ce[i]
            vec_to = group_ce[i + 1]
            
            drift = compute_l2_squared_diff(vec_to, vec_from)
            
            results.append({
                "author_id": author_id,
                "field": field,
                "domain": domain,
                "label": label,
                "provider": provider,
                "level": level,
                "model": model,
                "year_from": year_from,
                "year_to": year_to,
                "drift": drift,
            })
    
    return pd.DataFrame(results)


def compare_human_vs_llm_drift_paired(df_drift: pd.DataFrame) -> pd.DataFrame:
    """
    Compare human vs LLM drift using paired comparison with merge.
    For each human author, compare with each of their 4 LLM shadows (DS, G4B, G12B, LMK) separately.
    Returns paired dataframe with one row per (author, field, domain, year_from, year_to, provider).
    """
    # Extract human drifts
    H = df_drift[df_drift["label"] == "human"][
        ["author_id", "field", "domain", "year_from", "year_to", "drift"]
    ].rename(columns={"drift": "drift_h"})
    
    # Extract LLM drifts (with provider)
    L = df_drift[df_drift["label"] == "llm"][
        ["author_id", "field", "domain", "year_from", "year_to", "provider", "drift"]
    ].rename(columns={"drift": "drift_l"})
    
    # Merge on matching keys (one-to-one pairing per provider)
    paired = H.merge(
        L,
        on=["author_id", "field", "domain", "year_from", "year_to"],
        how="inner",
    )
    
    # Compute win: human drift > LLM drift
    paired["win"] = (paired["drift_h"] > paired["drift_l"]).astype(int)
    
    return paired


def summarize_paired_comparison(df_paired: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize paired comparison results by provider and year transition.
    """
    # Summary by provider (overall)
    summary_provider = (
        df_paired.groupby("provider", dropna=False)["win"]
        .agg(n_pairs="count", wins="sum", win_rate="mean")
        .reset_index()
    )
    
    # Summary by provider + year transition
    summary_provider_year = (
        df_paired.groupby(["provider", "year_from", "year_to"], dropna=False)["win"]
        .agg(n_pairs="count", wins="sum", win_rate="mean")
        .reset_index()
        .sort_values(["provider", "year_from", "year_to"])
    )
    
    # Also add drift statistics
    summary_provider_year = summary_provider_year.merge(
        df_paired.groupby(["provider", "year_from", "year_to"], dropna=False).agg(
            human_drift_mean=("drift_h", "mean"),
            human_drift_std=("drift_h", "std"),
            llm_drift_mean=("drift_l", "mean"),
            llm_drift_std=("drift_l", "std"),
        ).reset_index(),
        on=["provider", "year_from", "year_to"],
    )
    
    return summary_provider, summary_provider_year


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CE embedding drift using L2 norm (squared)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to include (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT / "ce_embedding_drift",
        help="Output directory for results.",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("CE Embedding Drift Analysis (LV3 only)")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"LLM Level: LV3 only")
    print(f"CE Features (20D): {', '.join(CE_FEATURES)}")
    print(f"Note: CE features are z-score normalized before computing drift")
    print()
    
    # Load data
    print("Loading combined data...")
    df_all = load_combined_data(args.domains)
    if df_all.empty:
        print("⚠ No data found.")
        return
    
    print(f"Loaded {len(df_all)} rows")
    print(f"  Human: {(df_all['label'] == 'human').sum()}")
    print(f"  LLM: {(df_all['label'] == 'llm').sum()}")
    print()
    
    # Compute drift per author
    print("Computing drift per author (year-centroid method)...")
    df_drift = compute_drift_per_author(df_all)
    print(f"Computed {len(df_drift)} drift measurements")
    
    # Filter: only keep consecutive years (year_to == year_from + 1)
    df_drift_consecutive = df_drift[df_drift["year_to"] == df_drift["year_from"] + 1].copy()
    print(f"  Consecutive years only: {len(df_drift_consecutive)} measurements")
    print()
    
    # Compare human vs LLM using paired comparison (each author vs their 4 LLM shadows)
    print("Comparing human vs LLM drift (paired: each author vs each LLM provider separately)...")
    df_paired = compare_human_vs_llm_drift_paired(df_drift_consecutive)
    summary_provider, summary_provider_year = summarize_paired_comparison(df_paired)
    print()
    
    # Print summary by provider
    print("=" * 80)
    print("Summary: Human vs LLM Drift Comparison by Provider (Paired)")
    print("=" * 80)
    print(summary_provider.to_string(index=False))
    print()
    
    # Print summary by provider + year
    print("=" * 80)
    print("Summary: Human vs LLM Drift Comparison by Provider and Year Transition")
    print("=" * 80)
    print(summary_provider_year.to_string(index=False))
    print()
    
    # Overall statistics
    total_pairs = len(df_paired)
    total_wins = df_paired["win"].sum()
    overall_win_rate = total_wins / total_pairs if total_pairs > 0 else 0.0
    
    print("=" * 80)
    print("Overall Statistics (Paired Comparison)")
    print("=" * 80)
    print(f"Total pairs: {total_pairs} (author × year transitions × providers)")
    print(f"Total wins: {total_wins} / {total_pairs} (human drift > LLM drift)")
    print(f"Overall win rate: {overall_win_rate:.4f} ({100 * overall_win_rate:.2f}%)")
    print()
    print("Win rate by provider:")
    for _, row in summary_provider.iterrows():
        print(f"  {row['provider']}: {row['win_rate']:.4f} ({100 * row['win_rate']:.2f}%) "
              f"[{row['wins']}/{row['n_pairs']} pairs]")
    print()
    print(f"Human drift:")
    print(f"  Mean: {df_paired['drift_h'].mean():.6f}")
    print(f"  Std:  {df_paired['drift_h'].std():.6f}")
    print()
    print(f"LLM drift (by provider):")
    for provider in ["DS", "G4B", "G12B", "LMK"]:
        provider_data = df_paired[df_paired["provider"] == provider]
        if len(provider_data) > 0:
            print(f"  {provider}: Mean={provider_data['drift_l'].mean():.6f}, "
                  f"Std={provider_data['drift_l'].std():.6f}")
    print()
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    drift_path = args.output_dir / "ce_embedding_drift_per_author.csv"
    df_drift_consecutive.to_csv(drift_path, index=False)
    print(f"✅ Saved per-author drift to: {drift_path}")
    
    paired_path = args.output_dir / "ce_embedding_drift_paired.csv"
    df_paired.to_csv(paired_path, index=False)
    print(f"✅ Saved paired comparison to: {paired_path}")
    
    summary_provider_path = args.output_dir / "ce_embedding_drift_summary_by_provider.csv"
    summary_provider.to_csv(summary_provider_path, index=False)
    print(f"✅ Saved summary by provider to: {summary_provider_path}")
    
    summary_provider_year_path = args.output_dir / "ce_embedding_drift_summary_by_provider_year.csv"
    summary_provider_year.to_csv(summary_provider_year_path, index=False)
    print(f"✅ Saved summary by provider and year to: {summary_provider_year_path}")


if __name__ == "__main__":
    main()

