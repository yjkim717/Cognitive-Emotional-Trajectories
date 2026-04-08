#!/usr/bin/env python3
"""
Merge TF-IDF and SBERT features with combined.csv for micro datasets.

This script merges:
- TF-IDF vectors (10 dimensions)
- SBERT vectors (384 dimensions)
- With existing combined_merged.csv (for human) or combined_merged_outliers_removed.csv (for LLM)

Output: combined_with_embeddings.csv

Human: combined_merged.csv + tfidf_vectors.csv + sbert_vectors.csv
LLM: combined_merged_outliers_removed.csv + tfidf_vectors.csv + sbert_vectors.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
LEVELS = ("LV1", "LV2", "LV3")
LLM_WITH_HISTORY_MODELS = ("DS", "CL35", "G4OM")
LLM_WITH_HISTORY_LEVEL = "LV3"
LLM_WITH_HISTORY_DOMAIN = "news"
TFIDF_FILENAME = "tfidf_vectors.csv"
SBERT_FILENAME = "sbert_vectors.csv"
HUMAN_COMBINED_FILENAME = "combined_merged.csv"  # Human uses combined_merged (not outliers_removed)
LLM_COMBINED_FILENAME = "combined_merged_outliers_removed.csv"  # LLM uses outliers_removed
OUTPUT_FILENAME = "combined_with_embeddings.csv"

# Merge keys (columns to match on)
MERGE_KEYS = ["filename", "path", "label"]


def merge_features_for_file(
    combined_path: Path, 
    tfidf_path: Path, 
    sbert_path: Path, 
    output_path: Path
) -> None:
    """Merge combined.csv with TF-IDF and SBERT features."""
    # Load files
    if not combined_path.exists():
        print(f"⚠️  Combined file not found: {combined_path}")
        return
    
    df_combined = pd.read_csv(combined_path)
    file_type = combined_path.name
    print(f"Loaded {file_type}: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    
    df_tfidf = None
    df_sbert = None
    
    if tfidf_path.exists():
        df_tfidf = pd.read_csv(tfidf_path)
        print(f"Loaded TF-IDF: {len(df_tfidf)} rows, {len(df_tfidf.columns)} columns")
    else:
        print(f"⚠️  TF-IDF file not found: {tfidf_path}")
    
    if sbert_path.exists():
        df_sbert = pd.read_csv(sbert_path)
        print(f"Loaded SBERT: {len(df_sbert)} rows, {len(df_sbert.columns)} columns")
    else:
        print(f"⚠️  SBERT file not found: {sbert_path}")
    
    # Merge TF-IDF
    if df_tfidf is not None:
        # Find common merge keys
        common_keys = [k for k in MERGE_KEYS if k in df_combined.columns and k in df_tfidf.columns]
        if not common_keys:
            print(f"⚠️  No common merge keys found between combined and TF-IDF")
            print(f"  Combined columns: {list(df_combined.columns[:10])}")
            print(f"  TF-IDF columns: {list(df_tfidf.columns[:10])}")
        else:
            df_combined = pd.merge(
                df_combined, 
                df_tfidf, 
                on=common_keys, 
                how="inner", 
                suffixes=("", "_tfidf")
            )
            print(f"Merged TF-IDF: {len(df_combined)} rows after merge")
    
    # Merge SBERT
    if df_sbert is not None:
        # Find common merge keys
        common_keys = [k for k in MERGE_KEYS if k in df_combined.columns and k in df_sbert.columns]
        if not common_keys:
            print(f"⚠️  No common merge keys found between combined and SBERT")
            print(f"  Combined columns: {list(df_combined.columns[:10])}")
            print(f"  SBERT columns: {list(df_sbert.columns[:10])}")
        else:
            df_combined = pd.merge(
                df_combined, 
                df_sbert, 
                on=common_keys, 
                how="inner", 
                suffixes=("", "_sbert")
            )
            print(f"Merged SBERT: {len(df_combined)} rows after merge")
    
    # Remove duplicate columns (from merge suffixes)
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    
    # Remove duplicate columns (keep only original, drop suffixed versions)
    # Handle label, model, level, domain, field, author_id, provider
    duplicate_patterns = ['label', 'model', 'level', 'domain', 'field', 'author_id', 'provider']
    
    for pattern in duplicate_patterns:
        pattern_cols = [col for col in df_combined.columns 
                       if col.lower() == pattern.lower() or col.lower().startswith(f'{pattern.lower()}_')]
        if len(pattern_cols) > 1:
            # Keep only the original column (exact match, no suffix)
            original_col = next((col for col in pattern_cols if col.lower() == pattern.lower()), None)
            if original_col:
                cols_to_drop = [col for col in pattern_cols if col != original_col]
                df_combined = df_combined.drop(columns=cols_to_drop)
                if cols_to_drop:
                    print(f"Removed duplicate {pattern} columns: {cols_to_drop}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"✅ Saved merged features to {output_path}")
    print(f"   Final shape: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    
    # Print feature summary
    tfidf_cols = [c for c in df_combined.columns if c.startswith("tfidf_")]
    sbert_cols = [c for c in df_combined.columns if c.startswith("sbert_")]
    ce_cols = [c for c in df_combined.columns if c in [
        "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism",
        "polarity", "subjectivity", "vader_compound", "vader_pos", "vader_neu", "vader_neg",
        "word_diversity", "flesch_reading_ease", "gunning_fog", "average_word_length",
        "num_words", "avg_sentence_length", "verb_ratio", "function_word_ratio", "content_word_ratio"
    ]]
    print(f"   Features: {len(ce_cols)} CE, {len(tfidf_cols)} TF-IDF, {len(sbert_cols)} SBERT")
    print(f"   Total features: {len(ce_cols) + len(tfidf_cols) + len(sbert_cols)}")


def process_human_domain(domain: str) -> None:
    """Process human files for a domain."""
    print(f"\n=== Merging features for Human {domain} ===")
    
    human_dir = DATA_ROOT / "human" / domain
    combined_path = human_dir / HUMAN_COMBINED_FILENAME
    tfidf_path = human_dir / TFIDF_FILENAME
    sbert_path = human_dir / SBERT_FILENAME
    output_path = human_dir / OUTPUT_FILENAME
    
    if combined_path.exists():
        print(f"\n--- Human {domain} ---")
        merge_features_for_file(combined_path, tfidf_path, sbert_path, output_path)
    else:
        print(f"⚠️  Human combined file not found: {combined_path}")


def process_llm_domain(provider: str, level: str, domain: str) -> None:
    """Process LLM files for a provider/level/domain combination."""
    llm_dir = DATA_ROOT / "LLM" / provider / level / domain
    combined_path = llm_dir / LLM_COMBINED_FILENAME
    tfidf_path = llm_dir / TFIDF_FILENAME
    sbert_path = llm_dir / SBERT_FILENAME
    output_path = llm_dir / OUTPUT_FILENAME
    
    if combined_path.exists():
        print(f"\n--- LLM {provider} {level} {domain} ---")
        merge_features_for_file(combined_path, tfidf_path, sbert_path, output_path)
    else:
        print(f"⚠️  LLM combined file not found: {combined_path}")


def process_llm_with_history(domains=None) -> None:
    """Merge TFIDF + SBERT with combined_merged for LLM_with_history across all specified domains."""
    if domains is None:
        domains = list(DOMAINS)
    for provider in LLM_WITH_HISTORY_MODELS:
        for domain in domains:
            llm_dir = DATA_ROOT / "LLM_with_history" / provider / LLM_WITH_HISTORY_LEVEL / domain
            combined_path = llm_dir / "combined_merged.csv"  # no outliers_removed for this branch
            tfidf_path = llm_dir / TFIDF_FILENAME
            sbert_path = llm_dir / SBERT_FILENAME
            output_path = llm_dir / OUTPUT_FILENAME
            if combined_path.exists():
                print(f"\n--- LLM_with_history {provider} LV3 {domain} ---")
                merge_features_for_file(combined_path, tfidf_path, sbert_path, output_path)
            else:
                print(f"⚠️  LLM_with_history combined not found: {combined_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge TF-IDF and SBERT features with combined.csv for micro datasets."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="LLM providers to process (default: all).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=LEVELS,
        default=list(LEVELS),
        help="Levels to process (default: all).",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="Only process human data.",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only process LLM data.",
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Only process LLM_with_history (news, DS/CL35/G4OM, LV3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_with_history:
        print("\n" + "=" * 80)
        print("PROCESSING LLM_with_history")
        print("=" * 80)
        process_llm_with_history(domains=args.domains)
        print("\n✅ LLM_with_history features merged!")
        return
    # Process human data
    if not args.llm_only:
        print("\n" + "="*80)
        print("PROCESSING HUMAN DATA")
        print("="*80)
        for domain in args.domains:
            process_human_domain(domain)
    
    # Process LLM data
    if not args.human_only:
        print("\n" + "="*80)
        print("PROCESSING LLM DATA")
        print("="*80)
        for provider in args.providers:
            for level in args.levels:
                for domain in args.domains:
                    process_llm_domain(provider, level, domain)
    
    print("\n✅ All features merged!")


if __name__ == "__main__":
    main()






