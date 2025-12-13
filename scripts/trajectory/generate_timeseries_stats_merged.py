#!/usr/bin/env python3
"""
Generate time series statistics (variance, CV, RMSSD, MASD) for merged features.

This script processes all combined_merged.csv files and generates 
author_timeseries_stats_merged.csv files containing variance, CV, RMSSD, and MASD 
for each author based on merged NELA features (15 features).

Output file: author_timeseries_stats_merged.csv
Location: Same directory as combined_merged.csv
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

DATA_ROOT = "dataset"
PROCESS_ROOT = os.path.join(DATA_ROOT, "process")


def parse_year_and_index_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse year and item_index from filename like News_WORLD_2024_05_DS_LV1.txt."""
    import re

    if not filename or not isinstance(filename, str):
        return (None, None)

    match = re.search(r"(\d{4})_(\d{2})", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (None, None)


def compute_timeseries_stats(series: pd.Series) -> Dict[str, float]:
    """Compute variance, CV, RMSSD, MASD, and normalized RMSSD/MASD for a series."""
    if series is None or len(series) == 0:
        return {
            "variance": np.nan,
            "cv": np.nan,
            "rmssd": np.nan,
            "masd": np.nan,
            "rmssd_norm": np.nan,
            "masd_norm": np.nan,
        }

    if len(series) < 2:
        mean_val = series.mean()
        return {
            "variance": np.nan,
            "cv": np.nan if mean_val == 0 else np.nan,
            "rmssd": np.nan,
            "masd": np.nan,
            "rmssd_norm": np.nan,
            "masd_norm": np.nan,
        }

    variance = series.var(ddof=1)
    mean_val = series.mean()
    mean_abs = abs(mean_val)
    cv = np.sqrt(variance) / mean_abs if mean_abs > 0 else np.nan

    successive_diffs = series.diff().dropna()
    if len(successive_diffs) > 0:
        rmssd = np.sqrt(np.mean(successive_diffs**2))
        masd = np.mean(np.abs(successive_diffs))
    else:
        rmssd = np.nan
        masd = np.nan

    rmssd_norm = rmssd / mean_abs if (not np.isnan(rmssd) and mean_abs > 0) else np.nan
    masd_norm = masd / mean_abs if (not np.isnan(masd) and mean_abs > 0) else np.nan

    return {
        "variance": variance,
        "cv": cv,
        "rmssd": rmssd,
        "masd": masd,
        "rmssd_norm": rmssd_norm,
        "masd_norm": masd_norm,
    }


def resolve_column(df: pd.DataFrame, target: str, required: bool = True) -> Optional[str]:
    """Best-effort lookup for a column name with flexible aliases."""
    aliases = {
        "author_id": ["author", "authorId", "authorID", "batch"],
        "field": ["subfield"],
        "domain": ["genre"],
    }
    candidates = [target, target.lower(), target.upper(), target.capitalize()]
    candidates.extend(aliases.get(target, []))

    for name in candidates:
        if name in df.columns:
            return name

    if required:
        raise KeyError(f"Column '{target}' not found in dataframe columns: {list(df.columns)}")
    return None

DEFAULT_MODELS = ["DS", "G4B", "G12B", "LMK"]
DEFAULT_LEVELS = ["LV1", "LV2", "LV3"]
DEFAULT_DOMAINS = ["academic", "news", "blogs"]


def process_single_dataset(
    combined_csv_path: Path,
    output_csv_path: Path,
    target: str = "human",
    domain: str = None,
    model: str = None,
    level: str = None,
) -> bool:
    """
    Process a single combined_merged.csv file and generate author_timeseries_stats_merged.csv.
    
    Args:
        combined_csv_path: Path to combined_merged.csv input file
        output_csv_path: Path to output author_timeseries_stats_merged.csv file
        target: "human" or "llm"
        domain: Domain name (for logging)
        model: Model name (for logging)
        level: Level name (for logging)
    
    Returns:
        True if successful, False otherwise
    """
    if not combined_csv_path.exists():
        print(f"[Skip] Combined merged CSV not found: {combined_csv_path}")
        return False
    
    try:
        print(f"\nProcessing: {combined_csv_path}")
        df = pd.read_csv(combined_csv_path)
        
        if df.empty:
            print(f"[Skip] Empty combined merged CSV: {combined_csv_path}")
            return False
        
        # Resolve column names
        author_col = resolve_column(df, "author_id")
        field_col = resolve_column(df, "field")
        domain_col = resolve_column(df, "domain", required=False)
        
        if target == "llm":
            model_col = resolve_column(df, "model", required=False)
            level_col = resolve_column(df, "level", required=False)
        else:
            model_col = level_col = None
        
        # Domain filtering
        if domain and domain_col:
            domain_mask = df[domain_col].str.lower() == domain.lower()
            before = len(df)
            df = df[domain_mask]
            if before != len(df):
                print(f"  Filtered domain='{domain}': {len(df)} / {before} rows retained")
        
        if df.empty:
            print(f"[Skip] No records after filtering")
            return False
        
        # Get numeric feature columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove metadata columns
        metadata_cols_to_remove = [author_col, field_col, domain_col, model_col, level_col, "year", "item_index"]
        if "model" in df.columns:
            metadata_cols_to_remove.append("model")
        if "level" in df.columns:
            metadata_cols_to_remove.append("level")
        
        for col in [c for c in metadata_cols_to_remove if c]:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        if not numeric_cols:
            print(f"[Skip] No numeric feature columns found")
            return False
        
        # Extract year and item_index from filename for temporal ordering
        filename_col = "filename" if "filename" in df.columns else None
        if filename_col is None:
            print(f"[Warning] No 'filename' column found. Cannot sort by time order.")
            df["year"] = None
            df["item_index"] = None
        else:
            year_index = df[filename_col].apply(parse_year_and_index_from_filename)
            df["year"] = year_index.apply(lambda x: x[0])
            df["item_index"] = year_index.apply(lambda x: x[1])
            parsed = df["year"].notna().sum()
            if parsed > 0:
                print(f"  Extracted temporal info from {parsed} / {len(df)} files")
        
        # Compute time series statistics for each author-field group
        group_cols = [field_col, author_col]
        results_list = []
        
        for (field_val, author_val), group in df.groupby(group_cols, dropna=False):
            sample_count = len(group)
            
            # Sort by year and item_index if available
            if "year" in group.columns and group["year"].notna().any():
                group_sorted = group.sort_values(["year", "item_index"], na_position="last")
            else:
                group_sorted = group
            
            # Compute statistics for each numeric feature
            stats_dict = {
                "field": field_val,
                "author_id": author_val,
                "sample_count": sample_count,
            }
            
            for feature_col in numeric_cols:
                feature_series = group_sorted[feature_col].dropna()
                
                if len(feature_series) == 0:
                    # No valid data for this feature
                    for stat_name in ["variance", "cv", "rmssd", "masd"]:
                        stats_dict[f"{feature_col}_{stat_name}"] = np.nan
                else:
                    # Compute all time series statistics (original)
                    ts_stats = compute_timeseries_stats(feature_series, normalize=False)
                    for stat_name, stat_value in ts_stats.items():
                        # Only save main stats, skip normalized_variance for now (will add later)
                        if stat_name in ["variance", "cv", "rmssd", "masd"]:
                            stats_dict[f"{feature_col}_{stat_name}"] = stat_value
            
            results_list.append(stats_dict)
        
        result = pd.DataFrame(results_list)
        
        # Reorder columns: metadata first, then features with consistent ordering
        metadata_cols = ["field", "author_id", "sample_count"]
        feature_stat_cols = []
        stat_names = ["variance", "cv", "rmssd", "masd"]
        
        for feature_col in numeric_cols:
            for stat_name in stat_names:
                col_name = f"{feature_col}_{stat_name}"
                if col_name in result.columns:
                    feature_stat_cols.append(col_name)
        
        result = result[metadata_cols + feature_stat_cols]
        
        # Sort for readability: by field then author
        result = result.sort_values(["field", "author_id"])
        
        # Save to output path
        os.makedirs(output_csv_path.parent, exist_ok=True)
        result.to_csv(output_csv_path, index=False)
        
        print(f"  ✅ Saved: {output_csv_path}")
        print(f"     Processed {len(result)} field-author pairs across {len(df)} samples")
        print(f"     Features: {len(numeric_cols)} numeric columns")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to process {combined_csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_timeseries_stats_merged(
    target: Optional[str] = None,
    domain: Optional[str] = None,
    model: Optional[str] = None,
    level: Optional[str] = None,
    models: Optional[list[str]] = None,
    levels: Optional[list[str]] = None,
    domains: Optional[list[str]] = None,
    output_filename: str = "author_timeseries_stats_merged.csv",
):
    """
    Generate author_timeseries_stats_merged.csv for all or specified datasets.
    
    Args:
        target: "human", "llm", or None (process both)
        domain: Specific domain (academic, news, blogs) or None (all domains)
        model: Specific model (DS, G4B, G12B, LMK) or None (all models)
        level: Specific level (LV1, LV2, LV3) or None (all levels)
        models: List of models to process (overrides model parameter)
        levels: List of levels to process (overrides level parameter)
        domains: List of domains to process (overrides domain parameter)
        output_filename: Output CSV filename (default: "author_timeseries_stats_merged.csv")
    """
    process_root = Path(PROCESS_ROOT)
    processed = 0
    skipped = 0
    errors = 0
    
    print("="*70)
    print("GENERATING AUTHOR TIMESERIES STATISTICS (MERGED VERSION)")
    print("="*70)
    print(f"Input files: combined_merged.csv")
    print(f"Output filename: {output_filename}")
    print(f"Statistics included: variance, CV, RMSSD, MASD")
    print("="*70)
    
    # Process Human data
    if target is None or target == "human":
        print("\n📁 Processing Human data...")
        human_root = process_root / "human"
        
        domains_to_process = domains if domains else ([domain] if domain else DEFAULT_DOMAINS)
        
        for domain_name in domains_to_process:
            combined_path = human_root / domain_name / "combined_merged.csv"
            output_path = human_root / domain_name / output_filename
            
            if process_single_dataset(
                combined_path, output_path,
                target="human", domain=domain_name
            ):
                processed += 1
            else:
                skipped += 1
    
    # Process LLM data
    if target is None or target == "llm":
        print("\n📁 Processing LLM data...")
        llm_root = process_root / "LLM"
        
        if not llm_root.exists():
            print(f"[Skip] LLM directory not found: {llm_root}")
        else:
            models_to_process = models if models else ([model] if model else DEFAULT_MODELS)
            levels_to_process = levels if levels else ([level] if level else DEFAULT_LEVELS)
            domains_to_process = domains if domains else ([domain] if domain else DEFAULT_DOMAINS)
            
            total_tasks = len(models_to_process) * len(levels_to_process) * len(domains_to_process)
            current_task = 0
            
            for model_name in models_to_process:
                model_dir = llm_root / model_name.upper()
                if not model_dir.exists():
                    print(f"[Skip] Model directory not found: {model_dir}")
                    continue
                
                for level_name in levels_to_process:
                    level_dir = model_dir / level_name.upper()
                    if not level_dir.exists():
                        print(f"[Skip] Level directory not found: {level_dir}")
                        continue
                    
                    for domain_name in domains_to_process:
                        current_task += 1
                        combined_path = level_dir / domain_name / "combined_merged.csv"
                        output_path = level_dir / domain_name / output_filename
                        
                        if process_single_dataset(
                            combined_path, output_path,
                            target="llm", domain=domain_name,
                            model=model_name, level=level_name
                        ):
                            processed += 1
                        else:
                            skipped += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Successfully processed: {processed} datasets")
    print(f"⏭️  Skipped: {skipped} datasets")
    print(f"❌ Errors: {errors} datasets")
    print(f"\nOutput files saved as: {output_filename}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate author_timeseries_stats_merged.csv files (variance, CV, RMSSD, MASD) from combined_merged.csv for all or specified datasets.",
    )
    parser.add_argument(
        "--target",
        choices=["human", "llm"],
        help="Target dataset type (human or llm). If not specified, process both.",
    )
    parser.add_argument(
        "--domain",
        help="Specific domain (academic, news, blogs). If not specified, process all domains.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=f"LLM models to process (default: {' '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        help=f"LLM levels to process (default: {' '.join(DEFAULT_LEVELS)}).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help=f"Domains to process (default: {' '.join(DEFAULT_DOMAINS)}).",
    )
    parser.add_argument(
        "--model",
        help="Specific LLM model (DS, G4B, G12B, LMK). Only used when --target=llm.",
    )
    parser.add_argument(
        "--level",
        help="Specific LLM level (LV1, LV2, LV3). Only used when --target=llm.",
    )
    parser.add_argument(
        "--output-filename",
        default="author_timeseries_stats_merged.csv",
        help="Output CSV filename (default: author_timeseries_stats_merged.csv)",
    )
    
    args = parser.parse_args()
    
    generate_all_timeseries_stats_merged(
        target=args.target,
        domain=args.domain,
        model=args.model,
        level=args.level,
        models=args.models,
        levels=args.levels,
        domains=args.domains,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()

