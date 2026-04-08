#!/usr/bin/env python3
"""
Compute drift for each individual CE feature separately using raw values from combined_merged.csv.

For each of the 20 CE features:
1. Extract year from filename
2. If multiple samples per year, compute yearly centroids (mean)
3. Z-score normalize per author (all years together) for this feature
4. Compute drift = |value_y+1 - value_y| for consecutive years (absolute difference)

Note: Uses combined_merged.csv which already has outliers removed, so we use raw values directly.
No shadow version needed - outliers are already handled in the input data.

Output: CSV file (ce_feature_drift.csv) with columns:
- author_id, domain, field, label, model, level, feature, year_from, year_to, drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.parse_dataset_filename import parse_filename

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")

# 20 CE features (in order)
CE_FEATURES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "polarity",
    "subjectivity",
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
    "word_diversity",
    "flesch_reading_ease",
    "gunning_fog",
    "average_word_length",
    "num_words",
    "avg_sentence_length",
    "verb_ratio",
    "function_word_ratio",
    "content_word_ratio",
]


def extract_year_from_filename(row: pd.Series) -> str | None:
    """Extract year from filename using parse_filename utility."""
    filename = row.get("filename")
    if not isinstance(filename, str):
        return None
    
    label = row.get("label", "human")
    is_llm = label == "llm"
    
    meta = parse_filename(filename, is_llm=is_llm)
    return meta["year"] if meta else None


def compute_abs_diff(value1: float, value2: float) -> float:
    """Compute absolute difference: |value1 - value2|"""
    return float(np.abs(value1 - value2))


def zscore_normalize_per_author_single_feature(values: np.ndarray) -> np.ndarray:
    """
    Z-score normalize values per author (across all years) for a single feature.
    
    Args:
        values: shape (n_years,) - all yearly values for one author and one feature
    
    Returns:
        Normalized values with same shape
    """
    if len(values) == 0:
        return values
    
    # Compute mean and std across all years for this author
    mean = np.nanmean(values)
    std = np.nanstd(values)
    
    # Avoid division by zero
    if std == 0 or np.isnan(std):
        std = 1.0
    
    # Normalize: (x - mean) / std
    normalized = (values - mean) / std
    
    return normalized


def compute_drift_for_single_feature(
    df: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    """
    Compute drift for a single CE feature.
    
    Args:
        df: DataFrame with data
        feature_name: Name of the CE feature
    
    Returns:
        DataFrame with drift measurements for this feature
    """
    # Extract year
    df = df.copy()
    df["year"] = df.apply(extract_year_from_filename, axis=1)
    df = df.dropna(subset=["year", "author_id"]).reset_index(drop=True)
    
    # Convert year to int for sorting
    df["year"] = df["year"].astype(int)
    
    # Check if feature exists
    if feature_name not in df.columns:
        print(f"  ⚠️  Feature {feature_name} not found in data")
        return pd.DataFrame()
    
    # Group by author × year to compute yearly centroids
    yearly_group_cols = [
        "author_id", "field", "domain", "label", "provider", "level", "model", "year"
    ]
    
    # Ensure all group columns exist
    for col in yearly_group_cols:
        if col not in df.columns:
            if col == "provider":
                df[col] = df.get("label", "unknown")
            elif col == "level":
                df[col] = df.get("level", "LV0")
            elif col == "model":
                df[col] = df.get("provider", df.get("label", "unknown"))
            else:
                df[col] = "unknown"
    
    # Compute yearly centroids (mean() automatically ignores NaN values)
    yearly = (
        df.groupby(yearly_group_cols, dropna=False)[feature_name]
        .mean()
        .reset_index()
        .rename(columns={feature_name: "value"})
    )
    
    # Group by author trajectory (without year)
    traj_group_cols = ["author_id", "field", "domain", "label", "provider", "level", "model"]
    results: List[Dict] = []
    
    for key, group in yearly.groupby(traj_group_cols, dropna=False):
        (author_id, field, domain, label, provider, level, model) = key
        
        # Sort by year
        group_sorted = group.sort_values("year")
        group_years = group_sorted["year"].values
        group_values = group_sorted["value"].values
        
        if len(group_years) < 2:
            # Need at least 2 years to compute drift
            continue
        
        # Check which years have valid data (no NaN)
        valid_years_mask = ~np.isnan(group_values)
        
        if valid_years_mask.sum() < 2:
            # Need at least 2 valid years to compute any drift
            continue
        
        # Z-score normalize per author using only valid years
        valid_values = group_values[valid_years_mask]
        normalized_valid = zscore_normalize_per_author_single_feature(valid_values)
        
        # Create normalized array for all years (NaN for invalid years)
        group_values_normalized = np.full_like(group_values, np.nan)
        valid_indices = np.where(valid_years_mask)[0]
        for norm_idx, orig_idx in enumerate(valid_indices):
            group_values_normalized[orig_idx] = normalized_valid[norm_idx]
        
        # Compute drift between consecutive years (only if both consecutive years are valid)
        for i in range(len(group_years) - 1):
            if not (valid_years_mask[i] and valid_years_mask[i + 1]):
                # Skip this drift if either consecutive year has NaN
                continue
            
            year_from = int(group_years[i])
            year_to = int(group_years[i + 1])
            value_from = group_values_normalized[i]
            value_to = group_values_normalized[i + 1]
            
            drift = compute_abs_diff(value_to, value_from)
            
            results.append({
                "author_id": author_id,
                "domain": domain,
                "field": field,
                "label": label,
                "model": model,
                "level": level,
                "feature": feature_name,
                "year_from": year_from,
                "year_to": year_to,
                "drift": drift,
            })
    
    return pd.DataFrame(results)


def process_single_split(
    csv_path: Path,
    output_dir: Path,
    domain: str,
    label: str,
    provider: str | None = None,
    level: str | None = None,
) -> None:
    """
    Process a single combined_merged.csv file to compute CE feature drifts.
    
    Args:
        csv_path: Path to combined_merged.csv (outliers already removed)
        output_dir: Directory to save output CSV files
        domain: Domain name (academic, blogs, news)
        label: Label (human or llm)
        provider: Provider name (for LLM) or None (for human)
        level: Level (LV1, LV2, LV3) or None (for human)
    """
    if not csv_path.exists():
        return
    
    print(f"Processing: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  ⚠️  Empty dataframe, skipping")
        return
    
    # Set metadata columns
    df["label"] = label
    df["domain"] = domain
    if provider:
        df["provider"] = provider
        df["model"] = provider
    else:
        df["provider"] = "human"
        df["model"] = "human"
    
    if level:
        df["level"] = level
    else:
        df["level"] = "LV0"
    
    # Compute drift for each CE feature
    all_feature_drifts = []
    
    for feature in CE_FEATURES:
        feature_drift = compute_drift_for_single_feature(df, feature)
        if not feature_drift.empty:
            all_feature_drifts.append(feature_drift)
    
    if not all_feature_drifts:
        print(f"  ⚠️  No drift data computed")
        return
    
    # Combine all features
    combined_drift = pd.concat(all_feature_drifts, ignore_index=True)
    
    # Save to CSV
    output_path = output_dir / "ce_feature_drift.csv"
    combined_drift.to_csv(output_path, index=False)
    
    print(f"  ✅ CE feature drift: {len(combined_drift)} measurements ({len(all_feature_drifts)} features)")
    print(f"  Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute drift for each individual CE feature separately"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="LLM providers to process (default: all)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=LEVELS,
        default=list(LEVELS),
        help="LLM levels to process (default: all)",
    )
    args = parser.parse_args()
    
    input_file = "combined_merged.csv"
    
    print("=" * 80)
    print("Compute CE Feature Drift (Individual Features)")
    print("=" * 80)
    print(f"Input: {input_file} (outliers already removed)")
    print(f"Domains: {args.domains}")
    print(f"LLM Providers: {args.providers}")
    print(f"LLM Levels: {args.levels}")
    print(f"CE Features: {len(CE_FEATURES)} features")
    print(f"Drift method: |value_i+1 - value_i| (absolute difference after z-score normalization per author)")
    print(f"Note: Using raw values from combined_merged.csv (outliers already removed)")
    print()
    
    # Process human data
    for domain in args.domains:
        human_path = DATA_ROOT / "human" / domain / input_file
        
        process_single_split(
            csv_path=human_path,
            output_dir=human_path.parent,
            domain=domain,
            label="human",
        )
    
    # Process LLM data
    for provider in args.providers:
        for level in args.levels:
            for domain in args.domains:
                llm_path = DATA_ROOT / "LLM" / provider / level / domain / input_file
                
                process_single_split(
                    csv_path=llm_path,
                    output_dir=llm_path.parent,
                    domain=domain,
                    label="llm",
                    provider=provider,
                    level=level,
                )
    
    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Output files: ce_feature_drift.csv")


if __name__ == "__main__":
    main()

