#!/usr/bin/env python3
"""
Compute embedding drift using L2 norm difference between consecutive years.

For each embedding space (CE, TFIDF, SBERT):
1. Extract year from filename
2. If multiple samples per year, compute yearly centroids (mean)
3. Compute drift = ||emb_y+1 - emb_y||₂ for consecutive years (NO z-score normalization)

Note:
- CE features: Loaded from combined_merged.csv (raw values, outliers not removed)
- TFIDF/SBERT features: Loaded from combined_with_embeddings.csv

Output: Three CSV files (ce_drift.csv, tfidf_drift.csv, sbert_drift.csv)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.parse_dataset_filename import parse_filename

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
LEVELS = ("LV1", "LV2", "LV3")
LLM_WITH_HISTORY_MODELS = ("DS", "CL35", "G4OM")
LLM_WITH_HISTORY_LEVEL = "LV3"
LLM_WITH_HISTORY_DOMAIN = "news"

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


def compute_l2_norm_diff(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L2 norm of difference: ||vec1 - vec2||₂"""
    diff = vec1 - vec2
    return float(np.linalg.norm(diff, ord=2))


def compute_drift_for_embedding(
    df: pd.DataFrame,
    embedding_cols: List[str],
    rep_space: str,
) -> pd.DataFrame:
    """
    Compute drift for a specific embedding space.
    
    Args:
        df: DataFrame with data
        embedding_cols: List of column names for the embedding
        rep_space: Name of representation space (ce, tfidf, sbert)
    
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
        print(f"  ⚠️  Missing columns for {rep_space}: {missing_cols}")
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
    # If a year has multiple samples, even if some have NaN, mean() will use non-NaN values
    # If ALL samples in a year have NaN for a feature, the centroid will have NaN for that feature
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
                "rep_space": rep_space,
                "year_from": year_from,
                "year_to": year_to,
                "drift": drift,
            })
    
    return pd.DataFrame(results)


def process_single_split(
    ce_csv_path: Path,
    embeddings_csv_path: Path,
    output_dir: Path,
    domain: str,
    label: str,
    provider: str | None = None,
    level: str | None = None,
) -> None:
    """
    Process data files to compute embedding drift.
    
    Args:
        ce_csv_path: Path to combined_merged.csv (for CE features)
        embeddings_csv_path: Path to combined_with_embeddings.csv (for TFIDF/SBERT features)
        output_dir: Directory to save output CSV files
        domain: Domain name (academic, blogs, news)
        label: Label (human or llm)
        provider: Provider name (for LLM) or None (for human)
        level: Level (LV1, LV2, LV3) or None (for human, use LV0)
    """
    # Load CE features from combined_merged.csv
    if not ce_csv_path.exists():
        print(f"  ⚠️  CE file not found: {ce_csv_path}")
        df_ce = pd.DataFrame()
    else:
        df_ce = pd.read_csv(ce_csv_path)
        if not df_ce.empty:
            # Set metadata columns
            df_ce["label"] = label
            df_ce["domain"] = domain
            if provider:
                df_ce["provider"] = provider
                df_ce["model"] = provider
            else:
                df_ce["provider"] = "human"
                df_ce["model"] = "human"
            if level:
                df_ce["level"] = level
            else:
                df_ce["level"] = "LV0"
    
    # Load TFIDF/SBERT features from combined_with_embeddings.csv
    if not embeddings_csv_path.exists():
        print(f"  ⚠️  Embeddings file not found: {embeddings_csv_path}")
        df_emb = pd.DataFrame()
    else:
        df_emb = pd.read_csv(embeddings_csv_path)
        if not df_emb.empty:
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
    
    # Compute drift for each embedding space
    all_results: Dict[str, pd.DataFrame] = {
        "ce": pd.DataFrame(),
        "tfidf": pd.DataFrame(),
        "sbert": pd.DataFrame(),
    }
    
    # CE features (20D) - from combined_merged.csv
    if not df_ce.empty:
        ce_cols = [col for col in CE_FEATURES if col in df_ce.columns]
        if len(ce_cols) == 20:
            ce_drift = compute_drift_for_embedding(df_ce, ce_cols, "ce")
            all_results["ce"] = ce_drift
            print(f"  CE drift: {len(ce_drift)} measurements (from combined_merged.csv)")
        else:
            print(f"  ⚠️  Missing CE features (found {len(ce_cols)}/20)")
    else:
        print(f"  ⚠️  No CE data available")
    
    # TFIDF features (10D) - from combined_with_embeddings.csv
    if not df_emb.empty:
        tfidf_cols = [col for col in df_emb.columns if col.startswith("tfidf_") and col.replace("tfidf_", "").isdigit()]
        tfidf_cols = sorted(tfidf_cols, key=lambda x: int(x.replace("tfidf_", "")))
        if len(tfidf_cols) == 10:
            tfidf_drift = compute_drift_for_embedding(df_emb, tfidf_cols, "tfidf")
            all_results["tfidf"] = tfidf_drift
            print(f"  TFIDF drift: {len(tfidf_drift)} measurements")
        else:
            print(f"  ⚠️  Missing TFIDF features (found {len(tfidf_cols)}/10)")
    else:
        print(f"  ⚠️  No TFIDF data available")
    
    # SBERT features (384D) - from combined_with_embeddings.csv
    if not df_emb.empty:
        sbert_cols = [col for col in df_emb.columns if col.startswith("sbert_") and col.replace("sbert_", "").isdigit()]
        sbert_cols = sorted(sbert_cols, key=lambda x: int(x.replace("sbert_", "")))
        if len(sbert_cols) == 384:
            sbert_drift = compute_drift_for_embedding(df_emb, sbert_cols, "sbert")
            all_results["sbert"] = sbert_drift
            print(f"  SBERT drift: {len(sbert_drift)} measurements")
        else:
            print(f"  ⚠️  Missing SBERT features (found {len(sbert_cols)}/384)")
    else:
        print(f"  ⚠️  No SBERT data available")
    
    # Save output files (one per embedding space)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for rep_space, drift_df in all_results.items():
        if drift_df.empty:
            continue
        
        output_path = output_dir / f"{rep_space}_drift.csv"
        drift_df.to_csv(output_path, index=False)
        print(f"  ✅ Saved {rep_space} drift to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute embedding drift using L2 norm difference"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all)",
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Process only LLM_with_history (news, DS/CL35/G4OM, LV3).",
    )
    args = parser.parse_args()
    if args.llm_with_history:
        print("=" * 80)
        print("Embedding Drift: LLM_with_history")
        print("=" * 80)
        print(f"Domains: {args.domains}")
        for provider in LLM_WITH_HISTORY_MODELS:
            for domain in args.domains:
                base = DATA_ROOT / "LLM_with_history" / provider / LLM_WITH_HISTORY_LEVEL / domain
                ce_path = base / "combined_merged.csv"
                emb_path = base / "combined_with_embeddings.csv"
                if not ce_path.exists() or not emb_path.exists():
                    print(f"⚠️  Skip {provider}/{domain}: missing combined_merged or combined_with_embeddings")
                    continue
                process_single_split(
                    ce_csv_path=ce_path,
                    embeddings_csv_path=emb_path,
                    output_dir=base,
                    domain=domain,
                    label="llm",
                    provider=provider,
                    level=LLM_WITH_HISTORY_LEVEL,
                )
        print("\nDone (LLM_with_history).")
        return
    print("=" * 80)
    print("Embedding Drift Computation")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"Embedding spaces: CE (20D), TFIDF (10D), SBERT (384D)")
    print(f"Normalization: NONE (raw values used directly)")
    print(f"CE features: loaded from combined_merged.csv (raw, outliers not removed)")
    print(f"TFIDF/SBERT features: loaded from combined_with_embeddings.csv")
    print()
    
    # Process human data
    print("Processing Human data...")
    for domain in args.domains:
        ce_path = DATA_ROOT / "human" / domain / "combined_merged.csv"
        emb_path = DATA_ROOT / "human" / domain / "combined_with_embeddings.csv"
        
        process_single_split(
            ce_csv_path=ce_path,
            embeddings_csv_path=emb_path,
            output_dir=ce_path.parent,
            domain=domain,
            label="human",
            provider=None,
            level=None,
        )
    
    # Process LLM data (all providers, all levels)
    print()
    print("Processing LLM data...")
    for domain in args.domains:
        for provider in PROVIDERS:
            for level in LEVELS:
                ce_path = DATA_ROOT / "LLM" / provider / level / domain / "combined_merged.csv"
                emb_path = DATA_ROOT / "LLM" / provider / level / domain / "combined_with_embeddings.csv"
                
                process_single_split(
                    ce_csv_path=ce_path,
                    embeddings_csv_path=emb_path,
                    output_dir=ce_path.parent,
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
