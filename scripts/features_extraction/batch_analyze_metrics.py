#!/usr/bin/env python3
"""
Batch analyze metrics (Big Five + merged NELA) for all human and LLM datasets.

This script:
1. Extracts merged NELA + Big Five features for all human domains
2. Extracts merged NELA + Big Five features for all LLM models, levels, and domains
3. Optionally merges the two tables into combined_merged.csv files

Usage:
    python batch_analyze_metrics.py [--skip-combine]
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from utils.metric_big5 import extract_big5_features
from utils.metric_nela_merged import extract_nela_features_merged


DATA_ROOT = Path("dataset")
HUMAN_DIR = DATA_ROOT / "human"
LLM_DIR = DATA_ROOT / "llm"
OUTPUT_ROOT = DATA_ROOT / "process"
MERGE_KEYS = ["filename", "path", "label"]

# LLM with history: incremental context (summaries of previous outputs), raw txt now lives under dataset/llm/llm_with_history/
LLM_WITH_HISTORY_DIR = LLM_DIR / "llm_with_history"
LLM_WITH_HISTORY_MODELS = ["DS", "CL35", "G4OM"]
LLM_WITH_HISTORY_LEVEL = "LV3"
LLM_WITH_HISTORY_DOMAIN = "news"

# Default models and levels
DEFAULT_MODELS = ["DS", "G4B", "G12B", "LMK"]
DEFAULT_LEVELS = ["LV1", "LV2", "LV3"]
DEFAULT_DOMAINS = ["academic", "news", "blogs"]


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns after merge."""
    rename_map = {}
    drop_columns = []

    for col in df.columns:
        if col.endswith("_x"):
            base = col[:-2]
            other = f"{base}_y"
            if other in df.columns:
                drop_columns.append(other)
            rename_map[col] = base
        elif col.endswith("_y"):
            base = col[:-2]
            if base not in df.columns and base not in rename_map.values():
                rename_map[col] = base

    df = df.rename(columns=rename_map)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    return df


def process_human_domains(domains=None, skip_big5=False, skip_combine=False):
    """Process all human domains."""
    if domains is None:
        domains = DEFAULT_DOMAINS
    
    print("="*70)
    print("Processing Human Domains")
    print("="*70)
    
    for domain in domains:
        print(f"\n{'#'*70}")
        print(f"Processing: human/{domain}")
        print(f"{'#'*70}")
        
        output_dir = OUTPUT_ROOT / "human" / domain
        output_dir.mkdir(parents=True, exist_ok=True)
        
        big5_path = output_dir / "big5.csv"
        nela_path = output_dir / "nela_merged.csv"  # Merged NELA features (15 features)
        combined_path = output_dir / "combined_merged.csv"  # Combined Big5 + Merged NELA
        
        df_big5 = None
        df_nela = None
        
        # Load or extract Big Five
        if not skip_big5:
            print(f"\nExtracting Big Five features for human/{domain}...")
            df_big5 = extract_big5_features(
                str(HUMAN_DIR),
                "human",
                str(big5_path),
                domain=domain,
                model_name=None,
                level=None,
            )
        else:
            if big5_path.exists():
                print(f"Loading existing Big Five features from {big5_path}")
                df_big5 = pd.read_csv(big5_path)
            else:
                print(f"⚠️  Warning: Big Five features not found at {big5_path}")
                print(f"   Please run without --skip-big5 to generate them, or skip combining with --skip-combine")
                df_big5 = None
        
        # Extract merged NELA
        print(f"\nExtracting merged NELA features for human/{domain}...")
        df_nela = extract_nela_features_merged(
            str(HUMAN_DIR),
            "human",
            str(nela_path),
            domain=domain,
            model_name=None,
            level=None,
        )
        
        # Combine
        if not skip_combine and df_big5 is not None and df_nela is not None:
            print(f"\nCombining Big Five and merged NELA features for human/{domain}...")
            
            # Remove duplicated metadata columns from NELA results
            duplicate_meta_cols = [
                col for col in [
                    "domain", "field", "author_id", "year", "item_index", "model", "level"
                ] if col in df_nela.columns
            ]
            if duplicate_meta_cols:
                df_nela = df_nela.drop(columns=duplicate_meta_cols)
            
            df_combined = pd.merge(df_big5, df_nela, on=MERGE_KEYS, how="inner")
            df_combined = _deduplicate_columns(df_combined)
            
            drop_meta_cols = [
                col for col in ["year", "item_index", "year_x", "year_y", "item_index_x", "item_index_y"]
                if col in df_combined.columns
            ]
            if drop_meta_cols:
                df_combined = df_combined.drop(columns=drop_meta_cols)
            
            df_combined.to_csv(combined_path, index=False)
            print(f"✅ Combined features saved to {combined_path}")
            print(f"   Total samples: {len(df_combined)}")
        
        print(f"\n✅ Completed: human/{domain}")


def process_llm_models(
    models=None,
    levels=None,
    domains=None,
    skip_big5=False,
    skip_combine=False,
):
    """Process all LLM models, levels, and domains."""
    if models is None:
        models = DEFAULT_MODELS
    if levels is None:
        levels = DEFAULT_LEVELS
    if domains is None:
        domains = DEFAULT_DOMAINS
    
    print("\n" + "="*70)
    print("Processing LLM Models")
    print("="*70)
    
    total_tasks = len(models) * len(levels) * len(domains)
    current_task = 0
    
    for model in models:
        for level in levels:
            for domain in domains:
                current_task += 1
                print(f"\n{'#'*70}")
                print(f"Processing: LLM/{model}/{level}/{domain} ({current_task}/{total_tasks})")
                print(f"{'#'*70}")
                
                output_dir = OUTPUT_ROOT / "LLM" / model.upper() / level.upper() / domain
                output_dir.mkdir(parents=True, exist_ok=True)
                
                big5_path = output_dir / "big5.csv"
                nela_path = output_dir / "nela_merged.csv"  # Merged NELA features (15 features)
                combined_path = output_dir / "combined_merged.csv"  # Combined Big5 + Merged NELA
                
                df_big5 = None
                df_nela = None
                
                # Load or extract Big Five
                if not skip_big5:
                    print(f"\nExtracting Big Five features for LLM/{model}/{level}/{domain}...")
                    df_big5 = extract_big5_features(
                        str(LLM_DIR),
                        "llm",
                        str(big5_path),
                        domain=domain,
                        model_name=model,
                        level=level,
                    )
                else:
                    if big5_path.exists():
                        print(f"Loading existing Big Five features from {big5_path}")
                        df_big5 = pd.read_csv(big5_path)
                    else:
                        print(f"⚠️  Warning: Big Five features not found at {big5_path}")
                        print(f"   Please run without --skip-big5 to generate them, or skip combining with --skip-combine")
                        df_big5 = None
                
                # Extract merged NELA
                print(f"\nExtracting merged NELA features for LLM/{model}/{level}/{domain}...")
                df_nela = extract_nela_features_merged(
                    str(LLM_DIR),
                    "llm",
                    str(nela_path),
                    domain=domain,
                    model_name=model,
                    level=level,
                )
                
                # Combine
                if not skip_combine and df_big5 is not None and df_nela is not None:
                    print(f"\nCombining Big Five and merged NELA features for LLM/{model}/{level}/{domain}...")
                    
                    # Remove duplicated metadata columns from NELA results
                    duplicate_meta_cols = [
                        col for col in [
                            "domain", "field", "author_id", "year", "item_index", "model", "level"
                        ] if col in df_nela.columns
                    ]
                    if duplicate_meta_cols:
                        df_nela = df_nela.drop(columns=duplicate_meta_cols)
                    
                    df_combined = pd.merge(df_big5, df_nela, on=MERGE_KEYS, how="inner")
                    df_combined = _deduplicate_columns(df_combined)
                    
                    drop_meta_cols = [
                        col for col in ["year", "item_index", "year_x", "year_y", "item_index_x", "item_index_y"]
                        if col in df_combined.columns
                    ]
                    if drop_meta_cols:
                        df_combined = df_combined.drop(columns=drop_meta_cols)
                    
                    df_combined.to_csv(combined_path, index=False)
                    print(f"✅ Combined features saved to {combined_path}")
                    print(f"   Total samples: {len(df_combined)}")
                
                print(f"\n✅ Completed: LLM/{model}/{level}/{domain}")


def process_llm_with_history(skip_big5=False, skip_combine=False, domains=None):
    """
    Process LLM-with-history data (dataset/llm/llm_with_history/{domain}/).
    Output: dataset/process/LLM_with_history/{model}/LV3/{domain}/
    Same 20 CE features (Big Five + NELA) per sample; supports academic, blogs, news.
    """
    if domains is None:
        domains = DEFAULT_DOMAINS
    if not LLM_WITH_HISTORY_DIR.exists():
        print(f"⚠️  LLM_with_history directory not found: {LLM_WITH_HISTORY_DIR}")
        return
    dataset_dir = str(LLM_WITH_HISTORY_DIR)
    print("\n" + "=" * 70)
    print("Processing LLM with History (incremental context)")
    print("=" * 70)
    print(f"Input: {LLM_WITH_HISTORY_DIR}")
    print(f"Models: {LLM_WITH_HISTORY_MODELS}, level: {LLM_WITH_HISTORY_LEVEL}, domains: {domains}")
    for model in LLM_WITH_HISTORY_MODELS:
        for domain in domains:
            print(f"\n{'#'*70}")
            print(f"Processing: LLM_with_history/{model}/LV3/{domain}")
            print(f"{'#'*70}")
            output_dir = OUTPUT_ROOT / "LLM_with_history" / model / LLM_WITH_HISTORY_LEVEL / domain
            output_dir.mkdir(parents=True, exist_ok=True)
            big5_path = output_dir / "big5.csv"
            nela_path = output_dir / "nela_merged.csv"
            combined_path = output_dir / "combined_merged.csv"
            df_big5 = None
            df_nela = None
            if not skip_big5:
                print(f"\nExtracting Big Five for LLM_with_history/{model}/LV3/{domain}...")
                df_big5 = extract_big5_features(
                    dataset_dir,
                    "llm",
                    str(big5_path),
                    domain=domain,
                    model_name=model,
                    level=LLM_WITH_HISTORY_LEVEL,
                )
            elif big5_path.exists():
                df_big5 = pd.read_csv(big5_path)
            else:
                print(f"⚠️  Big Five not found at {big5_path}")
            print(f"\nExtracting NELA for LLM_with_history/{model}/LV3/{domain}...")
            df_nela = extract_nela_features_merged(
                dataset_dir,
                "llm",
                str(nela_path),
                domain=domain,
                model_name=model,
                level=LLM_WITH_HISTORY_LEVEL,
            )
            if not skip_combine and df_big5 is not None and df_nela is not None:
                duplicate_meta_cols = [
                    c for c in [
                        "domain", "field", "author_id", "year", "item_index", "model", "level"
                    ] if c in df_nela.columns
                ]
                if duplicate_meta_cols:
                    df_nela = df_nela.drop(columns=duplicate_meta_cols)
                df_combined = pd.merge(df_big5, df_nela, on=MERGE_KEYS, how="inner")
                df_combined = _deduplicate_columns(df_combined)
                drop_meta_cols = [
                    c for c in ["year", "item_index", "year_x", "year_y", "item_index_x", "item_index_y"]
                    if c in df_combined.columns
                ]
                if drop_meta_cols:
                    df_combined = df_combined.drop(columns=drop_meta_cols)
                df_combined.to_csv(combined_path, index=False)
                print(f"✅ Combined saved to {combined_path} ({len(df_combined)} samples)")
            print(f"✅ Completed: LLM_with_history/{model}/LV3/{domain}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract merged NELA features for all datasets."
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="Only process human domains."
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only process LLM models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"LLM models to process (default: {' '.join(DEFAULT_MODELS)})."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=DEFAULT_LEVELS,
        help=f"LLM levels to process (default: {' '.join(DEFAULT_LEVELS)})."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        help=f"Domains to process (default: {' '.join(DEFAULT_DOMAINS)})."
    )
    parser.add_argument(
        "--skip-big5",
        action="store_true",
        help="Skip Big Five extraction (use existing big5.csv files if available)."
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip combining Big Five and NELA (only extract features separately)."
    )
    parser.add_argument(
        "--nela-only",
        action="store_true",
        help="Only extract NELA features (skip Big Five and combine)."
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Process LLM-with-history data from dataset/llm/llm_with_history/ (DS/CL35/G4OM, LV3)."
    )
    
    args = parser.parse_args()
    
    # Default: extract both Big5 and NELA
    # Only skip Big5 if --skip-big5 or --nela-only is set
    skip_big5 = args.skip_big5 or args.nela_only
    
    if args.nela_only:
        args.skip_combine = True
    
    # Process LLM with history (reviewer condition: incremental context)
    if args.llm_with_history:
        process_llm_with_history(skip_big5=skip_big5, skip_combine=args.skip_combine, domains=args.domains)
        print("\n" + "=" * 70)
        print("✅ LLM_with_history extraction completed!")
        print("=" * 70)
        return
    
    # Process human domains
    if not args.llm_only:
        process_human_domains(
            domains=args.domains,
            skip_big5=skip_big5,
            skip_combine=args.skip_combine,
        )
    
    # Process LLM models
    if not args.human_only:
        process_llm_models(
            models=args.models,
            levels=args.levels,
            domains=args.domains,
            skip_big5=skip_big5,
            skip_combine=args.skip_combine,
        )
    
    print("\n" + "="*70)
    print("✅ Batch extraction completed!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_ROOT}")
    print(f"Human NELA files: {OUTPUT_ROOT}/human/*/nela_merged.csv")
    print(f"LLM NELA files: {OUTPUT_ROOT}/LLM/*/*/*/nela_merged.csv")
    if not args.skip_combine:
        print(f"Combined files (Big5 + Merged NELA): {OUTPUT_ROOT}/*/combined_merged.csv")


if __name__ == "__main__":
    main()
