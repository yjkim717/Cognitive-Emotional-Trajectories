#!/usr/bin/env python3
"""
Merge E5-Large SBERT features with combined_with_embeddings.csv for micro datasets.

This script merges E5-Large SBERT vectors (sbert_e5_vectors.csv) with 
combined_with_embeddings.csv to create combined_with_embeddings_e5.csv.

This allows separate comparison between all-MiniLM-L6-v2 (384D) and E5-Large (1024D).
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
SBERT_E5_FILENAME = "sbert_e5_vectors.csv"
INPUT_FILENAME = "combined_with_embeddings.csv"
OUTPUT_FILENAME = "combined_with_embeddings_e5.csv"

# Merge keys (columns to match on)
MERGE_KEYS = ["filename", "path", "label"]


def merge_e5_features_for_file(
    input_path: Path,
    e5_path: Path,
    output_path: Path
) -> None:
    """Merge E5-Large SBERT features with combined_with_embeddings.csv."""
    if not input_path.exists():
        print(f"⚠️  Input file not found: {input_path}")
        return
    
    df_input = pd.read_csv(input_path)
    print(f"Loaded input: {len(df_input)} rows, {len(df_input.columns)} columns")
    
    if not e5_path.exists():
        print(f"⚠️  E5 vectors file not found: {e5_path}")
        return
    
    df_e5 = pd.read_csv(e5_path)
    print(f"Loaded E5 vectors: {len(df_e5)} rows, {len(df_e5.columns)} columns")
    
    # Find common merge keys
    common_keys = [k for k in MERGE_KEYS if k in df_input.columns and k in df_e5.columns]
    if not common_keys:
        print(f"⚠️  No common merge keys found")
        print(f"  Input columns: {list(df_input.columns[:10])}")
        print(f"  E5 columns: {list(df_e5.columns[:10])}")
        return
    
    # Merge E5 features
    df_merged = pd.merge(
        df_input,
        df_e5,
        on=common_keys,
        how="inner",
        suffixes=("", "_e5")
    )
    print(f"Merged E5 features: {len(df_merged)} rows after merge")
    
    # Remove duplicate columns (keep only original, drop suffixed versions)
    duplicate_patterns = ['label', 'model', 'level', 'domain', 'field', 'author_id', 'provider']
    
    for pattern in duplicate_patterns:
        pattern_cols = [col for col in df_merged.columns 
                       if col.lower() == pattern.lower() or col.lower().startswith(f'{pattern.lower()}_')]
        if len(pattern_cols) > 1:
            original_col = next((col for col in pattern_cols if col.lower() == pattern.lower()), None)
            if original_col:
                cols_to_drop = [col for col in pattern_cols if col != original_col]
                df_merged = df_merged.drop(columns=cols_to_drop)
                if cols_to_drop:
                    print(f"Removed duplicate {pattern} columns: {cols_to_drop}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"✅ Saved merged file to {output_path}")
    print(f"   Final shape: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    
    # Print feature summary
    e5_cols = [c for c in df_merged.columns if c.startswith("sbert_e5_")]
    print(f"   E5 SBERT features: {len(e5_cols)}")


def process_human_domain(domain: str) -> None:
    """Process human files for a domain."""
    print(f"\n=== Merging E5 features for Human {domain} ===")
    
    human_dir = DATA_ROOT / "human" / domain
    input_path = human_dir / INPUT_FILENAME
    e5_path = human_dir / SBERT_E5_FILENAME
    output_path = human_dir / OUTPUT_FILENAME
    
    if input_path.exists():
        merge_e5_features_for_file(input_path, e5_path, output_path)
    else:
        print(f"⚠️  Input file not found: {input_path}")


def process_llm_domain(provider: str, level: str, domain: str) -> None:
    """Process LLM files for a provider/level/domain combination."""
    llm_dir = DATA_ROOT / "LLM" / provider / level / domain
    input_path = llm_dir / INPUT_FILENAME
    e5_path = llm_dir / SBERT_E5_FILENAME
    output_path = llm_dir / OUTPUT_FILENAME
    
    if input_path.exists():
        print(f"\n--- LLM {provider} {level} {domain} ---")
        merge_e5_features_for_file(input_path, e5_path, output_path)
    else:
        print(f"⚠️  Input file not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge E5-Large SBERT features with combined_with_embeddings.csv."
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
        default=["LV3"],
        help="Levels to process (default: LV3 only).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
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
    
    print("\n✅ All E5 features merged!")


if __name__ == "__main__":
    main()


