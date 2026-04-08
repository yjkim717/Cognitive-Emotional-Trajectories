#!/usr/bin/env python3
"""
Compute E5-Large SBERT embedding drift using L2 norm difference between consecutive years.

This script computes drift for E5-Large SBERT embeddings (1024D) similar to
compute_embedding_drift.py but specifically for E5 embeddings.

For E5-Large SBERT:
1. Extract year from filename
2. If multiple samples per year, compute yearly centroids (mean)
3. Compute drift = ||emb_y+1 - emb_y||₂ for consecutive years (NO z-score normalization)

Note:
- E5 SBERT features: Loaded from combined_with_embeddings_e5.csv (columns: sbert_e5_1 to sbert_e5_1024)
- Only processes LV3 data for LLM

Output: sbert_e5_drift.csv
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
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
E5_SBERT_DIM = 1024  # E5-Large embedding dimension


def extract_year_from_filename(row: pd.Series) -> str | None:
    """Extract year from filename using parse_filename utility."""
    filename = row.get("filename")
    if not isinstance(filename, str):
        return None
    
    label = row.get("label", "human")
    is_llm = label == "llm"
    
    meta = parse_filename(filename, is_llm=is_llm)
    return meta["year"] if meta else None


def compute_l2_norm_diff(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L2 norm of difference: ||vec1 - vec2||₂"""
    diff = vec1 - vec2
    return float(np.linalg.norm(diff, ord=2))


def compute_drift_for_e5_sbert(
    df: pd.DataFrame,
    embedding_cols: List[str],
) -> pd.DataFrame:
    """
    Compute drift for E5-Large SBERT embeddings.
    
    Args:
        df: DataFrame with data
        embedding_cols: List of column names for E5 SBERT embeddings (sbert_e5_1 to sbert_e5_1024)
    
    Returns:
        DataFrame with drift measurements
    """
    # Extract year
    df = df.copy()
    df["year"] = df.apply(extract_year_from_filename, axis=1)
    df = df.dropna(subset=["year", "author_id"]).reset_index(drop=True)
    
    # Convert year to int for sorting
    df["year"] = df["year"].astype(int)
    
    # Check if all embedding columns exist
    missing_cols = [col for col in embedding_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Missing E5 SBERT columns: {len(missing_cols)}/{len(embedding_cols)}")
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
        df.groupby(yearly_group_cols, dropna=False)[embedding_cols]
        .mean()
        .reset_index()
    )
    
    # Group by author trajectory (without year)
    traj_group_cols = ["author_id", "field", "domain", "label", "provider", "level", "model"]
    results: List[Dict] = []
    
    for key, group in yearly.groupby(traj_group_cols, dropna=False):
        (author_id, field, domain, label, provider, level, model) = key
        
        # Sort by year
        group_sorted = group.sort_values("year")
        group_years = group_sorted["year"].values
        
        if len(group_years) < 2:
            # Need at least 2 years to compute drift
            continue
        
        # Get embedding vectors
        group_emb = group_sorted[embedding_cols].to_numpy(dtype=float)
        
        # Check which years have valid data (no NaN in any feature)
        valid_years_mask = ~np.isnan(group_emb).any(axis=1)
        
        if valid_years_mask.sum() < 2:
            # Need at least 2 valid years to compute any drift
            continue
        
        # Compute drift between consecutive years (only if both consecutive years are valid)
        # NO z-score normalization - use raw values directly
        for i in range(len(group_years) - 1):
            if not (valid_years_mask[i] and valid_years_mask[i + 1]):
                # Skip this drift if either consecutive year has NaN
                continue
            
            year_from = int(group_years[i])
            year_to = int(group_years[i + 1])
            vec_from = group_emb[i]
            vec_to = group_emb[i + 1]
            
            drift = compute_l2_norm_diff(vec_to, vec_from)
            
            results.append({
                "author_id": author_id,
                "domain": domain,
                "field": field,
                "label": label,
                "model": model,
                "level": level,
                "rep_space": "sbert_e5",
                "year_from": year_from,
                "year_to": year_to,
                "drift": drift,
            })
    
    return pd.DataFrame(results)


def process_single_split(
    embeddings_csv_path: Path,
    output_dir: Path,
    domain: str,
    label: str,
    provider: str | None = None,
    level: str | None = None,
) -> None:
    """
    Process data files to compute E5-Large SBERT embedding drift.
    
    Args:
        embeddings_csv_path: Path to combined_with_embeddings_e5.csv (contains E5 SBERT features)
        output_dir: Directory to save output CSV files
        domain: Domain name (academic, blogs, news)
        label: Label (human or llm)
        provider: Provider name (for LLM) or None (for human)
        level: Level (LV3 only for LLM) or None (for human, use LV0)
    """
    # Load E5 SBERT features from combined_with_embeddings_e5.csv
    if not embeddings_csv_path.exists():
        print(f"  ⚠️  E5 embeddings file not found: {embeddings_csv_path}")
        return
    
    df_emb = pd.read_csv(embeddings_csv_path)
    if df_emb.empty:
        print(f"  ⚠️  Empty dataframe: {embeddings_csv_path}")
        return
    
    # Set metadata columns
    df_emb["label"] = label
    df_emb["domain"] = domain
    if provider:
        df_emb["provider"] = provider
        df_emb["model"] = provider
    else:
        df_emb["provider"] = "human"
        df_emb["model"] = "human"
    if level:
        df_emb["level"] = level
    else:
        df_emb["level"] = "LV0"
    
    # E5 SBERT features (1024D) - columns: sbert_e5_1 to sbert_e5_1024
    e5_sbert_cols = [col for col in df_emb.columns if col.startswith("sbert_e5_") and col.replace("sbert_e5_", "").isdigit()]
    e5_sbert_cols = sorted(e5_sbert_cols, key=lambda x: int(x.replace("sbert_e5_", "")))
    
    if len(e5_sbert_cols) == E5_SBERT_DIM:
        e5_drift = compute_drift_for_e5_sbert(df_emb, e5_sbert_cols)
        print(f"  E5 SBERT drift: {len(e5_drift)} measurements")
        
        # Save output file
        if not e5_drift.empty:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "sbert_e5_drift.csv"
            e5_drift.to_csv(output_path, index=False)
            print(f"  ✅ Saved E5 SBERT drift to {output_path}")
    else:
        print(f"  ⚠️  Missing E5 SBERT features (found {len(e5_sbert_cols)}/{E5_SBERT_DIM})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute E5-Large SBERT embedding drift using L2 norm difference"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("E5-Large SBERT Embedding Drift Computation")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"Embedding space: E5-Large SBERT (1024D)")
    print(f"Normalization: NONE (raw values used directly)")
    print(f"E5 SBERT features: loaded from combined_with_embeddings_e5.csv")
    print(f"LLM Level: LV3 only")
    print()
    
    # Process human data
    print("Processing Human data...")
    for domain in args.domains:
        emb_path = DATA_ROOT / "human" / domain / "combined_with_embeddings_e5.csv"
        
        process_single_split(
            embeddings_csv_path=emb_path,
            output_dir=emb_path.parent,
            domain=domain,
            label="human",
            provider=None,
            level=None,
        )
    
    # Process LLM data (LV3 only)
    print()
    print("Processing LLM data (LV3 only)...")
    for domain in args.domains:
        for provider in PROVIDERS:
            level = "LV3"
            emb_path = DATA_ROOT / "LLM" / provider / level / domain / "combined_with_embeddings_e5.csv"
            
            process_single_split(
                embeddings_csv_path=emb_path,
                output_dir=emb_path.parent,
                domain=domain,
                label="llm",
                provider=provider,
                level=level,
            )
    
    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()


