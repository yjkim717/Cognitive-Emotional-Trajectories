#!/usr/bin/env python3
"""
Generate time series statistics (variance, CV, RMSSD, MASD, and normalized versions) from combined_merged_outliers_removed.csv files.
Only processes LV1 data.

This script processes combined_merged_outliers_removed.csv files (where outliers are already set to NaN)
and generates author_timeseries_stats_merged.csv files containing:
- variance (original, length sensitive)
- cv (normalized, coefficient of variation)
- rmssd (original, length sensitive)
- masd (original, length sensitive)
- rmssd_norm (normalized, RMSSD / mean)
- masd_norm (normalized, MASD / mean)

NaN values are automatically excluded during calculation.
Normalization uses the mean of the same time series to remove length sensitivity.

Output file: author_timeseries_stats_merged.csv (same location as input)
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

DATA_ROOT = "dataset"
PROCESS_ROOT = os.path.join(DATA_ROOT, "process")

# Import functions from compute_author_variance.py
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


def parse_year_and_index_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse year and item_index from filename.
    
    Expected format: {Domain}_{Field}_{AuthorID}_{Year}_{ItemIndex}_{Model}_LV{Level}.txt
    or similar variations.
    
    Returns:
        (year, item_index) tuple, or (None, None) if parsing fails
    """
    import re
    if not filename or not isinstance(filename, str):
        return (None, None)
    
    # Try to extract year (4 digits) and item_index (2 digits after year)
    pattern = r'(\d{4})_(\d{2})'
    match = re.search(pattern, filename)
    if match:
        year = int(match.group(1))
        item_index = int(match.group(2))
        return (year, item_index)
    
    return (None, None)


def compute_timeseries_stats(series: pd.Series, normalize: bool = False, robust: bool = False) -> Dict[str, float]:
    """Compute time series statistics: variance, CV, RMSSD, MASD, and normalized versions.
    
    Args:
        series: Time series data (already sorted by time)
        normalize: Whether to normalize (not used in this version)
        robust: Whether to use robust statistics (not used in this version)
    
    Returns:
        Dictionary with keys: variance, cv, rmssd, masd, rmssd_norm, masd_norm
    """
    if len(series) == 0:
        return {
            "variance": np.nan, "cv": np.nan, 
            "rmssd": np.nan, "masd": np.nan,
            "rmssd_norm": np.nan, "masd_norm": np.nan
        }
    
    if len(series) < 2:
        # Need at least 2 values for variance and time series stats
        mean_val = series.mean()
        return {
            "variance": np.nan,
            "cv": np.nan if mean_val == 0 else np.nan,
            "rmssd": np.nan,
            "masd": np.nan,
            "rmssd_norm": np.nan,
            "masd_norm": np.nan,
        }
    
    # Variance
    variance = series.var(ddof=1)  # Sample variance
    
    # CV (Coefficient of Variation) - normalized version of variance
    mean_val = series.mean()
    if mean_val == 0:
        cv = np.nan
        mean_abs = abs(mean_val)
    else:
        cv = np.sqrt(variance) / abs(mean_val)
        mean_abs = abs(mean_val)
    
    # RMSSD (Root Mean Square of Successive Differences) - original (length sensitive)
    if len(series) >= 2:
        successive_diffs = series.diff().dropna()
        if len(successive_diffs) > 0:
            rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        else:
            rmssd = np.nan
    else:
        rmssd = np.nan
    
    # MASD (Mean Absolute Successive Differences) - original (length sensitive)
    if len(series) >= 2:
        successive_diffs = series.diff().dropna()
        if len(successive_diffs) > 0:
            masd = np.mean(np.abs(successive_diffs))
        else:
            masd = np.nan
    else:
        masd = np.nan
    
    # Normalized RMSSD (RMSSD / mean) - removes length sensitivity
    if not np.isnan(rmssd) and mean_abs > 0:
        rmssd_norm = rmssd / mean_abs
    else:
        rmssd_norm = np.nan
    
    # Normalized MASD (MASD / mean) - removes length sensitivity
    if not np.isnan(masd) and mean_abs > 0:
        masd_norm = masd / mean_abs
    else:
        masd_norm = np.nan
    
    return {
        "variance": variance,
        "cv": cv,
        "rmssd": rmssd,
        "masd": masd,
        "rmssd_norm": rmssd_norm,
        "masd_norm": masd_norm,
    }

DEFAULT_MODELS = ["DS", "G4B", "G12B", "LMK"]
DEFAULT_DOMAINS = ["academic", "news", "blogs"]


def process_single_dataset(
    input_csv_path: Path,
    output_csv_path: Path,
    target: str = "llm",
    domain: str = None,
    model: str = None,
) -> bool:
    """
    Process a single combined_merged_outliers_removed.csv file and generate author_timeseries_stats_merged.csv.
    
    Args:
        input_csv_path: Path to combined_merged_outliers_removed.csv input file
        output_csv_path: Path to output author_timeseries_stats_merged.csv file
        target: "human" or "llm"
        domain: Domain name (for logging)
        model: Model name (for logging)
    
    Returns:
        True if successful, False otherwise
    """
    if not input_csv_path.exists():
        print(f"[Skip] Input CSV not found: {input_csv_path}")
        return False
    
    try:
        print(f"\nProcessing: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        if df.empty:
            print(f"[Skip] Empty CSV: {input_csv_path}")
            return False
        
        # Note: Level is determined by the input file path, not by filtering the dataframe
        # The dataframe may contain a 'level' column but we process all rows
        
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
        
        print(f"  Processing {len(numeric_cols)} feature columns")
        
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
        total_nan_excluded = 0
        total_values = 0
        
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
                # Use dropna() to automatically exclude NaN values (outliers)
                feature_series = group_sorted[feature_col].dropna()
                
                total_values += len(group_sorted[feature_col])  # Original count including NaN
                total_nan_excluded += group_sorted[feature_col].isna().sum()
                
                if len(feature_series) < 2:
                    # Need at least 2 values for variance and time series stats
                    for stat_name in ["variance", "cv", "rmssd", "masd", "rmssd_norm", "masd_norm"]:
                        stats_dict[f"{feature_col}_{stat_name}"] = np.nan
                    continue
                
                # Compute time series statistics (NaN values already excluded by dropna())
                ts_stats = compute_timeseries_stats(feature_series, normalize=False, robust=False)
                
                # Store all statistics: original and normalized versions
                for stat_name in ["variance", "cv", "rmssd", "masd", "rmssd_norm", "masd_norm"]:
                    stats_dict[f"{feature_col}_{stat_name}"] = ts_stats.get(stat_name, np.nan)
            
            results_list.append(stats_dict)
        
        if total_values > 0:
            nan_percentage = (total_nan_excluded / total_values * 100) if total_values > 0 else 0
            print(f"  Excluded {total_nan_excluded:,} / {total_values:,} NaN values ({nan_percentage:.2f}%)")
        
        if not results_list:
            print(f"[Skip] No valid author-field groups found")
            return False
        
        result = pd.DataFrame(results_list)
        
        # Reorder columns: field, author_id, sample_count, then feature stats
        metadata_cols = ["field", "author_id", "sample_count"]
        feature_stats_cols = [c for c in result.columns if c not in metadata_cols]
        # Sort feature stats columns by feature name, then by stat type
        feature_stats_cols.sort(key=lambda x: (x.rsplit('_', 1)[0], x.rsplit('_', 1)[1]))
        
        result = result[metadata_cols + feature_stats_cols]
        
        # Sort for readability
        result = result.sort_values(["field", "author_id"])
        
        # Save output
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv_path, index=False)
        print(f"  ✅ Saved: {output_csv_path} ({len(result)} author-field pairs)")
        
        return True
        
    except Exception as e:
        print(f"[Error] Failed to process {input_csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_levels_timeseries_stats(
    process_root: Path = None,
    models: list = None,
    domains: list = None,
    target: str = "llm",
    levels: list = None,
) -> dict:
    """
    Generate time series statistics for all levels (LV1, LV2, LV3) combined_merged_outliers_removed.csv files.
    
    Args:
        process_root: Root directory for processed data (default: dataset/process)
        models: List of models to process (default: DEFAULT_MODELS)
        domains: List of domains to process (default: DEFAULT_DOMAINS)
        target: "human" or "llm"
        levels: List of levels to process (default: ["LV1", "LV2", "LV3"])
    
    Returns:
        Dictionary with processing statistics
    """
    if process_root is None:
        process_root = Path(PROCESS_ROOT)
    if models is None:
        models = DEFAULT_MODELS
    if domains is None:
        domains = DEFAULT_DOMAINS
    if levels is None:
        levels = ["LV1", "LV2", "LV3"]
    
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }
    
    print("="*80)
    print("Generate Time Series Statistics from Outliers-Removed Data (All Levels)")
    print("="*80)
    print(f"Target: {target.upper()}")
    print(f"Models: {models}")
    print(f"Domains: {domains}")
    print("="*80)
    
    if target == "human":
        # Process human data
        for domain in domains:
            input_path = process_root / "human" / domain / "combined_merged_outliers_removed.csv"
            output_path = process_root / "human" / domain / "author_timeseries_stats_merged.csv"
            
            if input_path.exists():
                success = process_single_dataset(
                    input_csv_path=input_path,
                    output_csv_path=output_path,
                    target="human",
                    domain=domain,
                )
                if success:
                    stats["processed"] += 1
                else:
                    stats["skipped"] += 1
            else:
                print(f"[Skip] Input file not found: {input_path}")
                stats["skipped"] += 1
    
    else:  # target == "llm"
        # Process LLM data for all levels
        for model in models:
            for level in levels:
                for domain in domains:
                    input_path = process_root / "LLM" / model / level / domain / "combined_merged_outliers_removed.csv"
                    output_path = process_root / "LLM" / model / level / domain / "author_timeseries_stats_merged.csv"
                    
                    if input_path.exists():
                        success = process_single_dataset(
                            input_csv_path=input_path,
                            output_csv_path=output_path,
                            target="llm",
                            domain=domain,
                            model=model,
                        )
                        if success:
                            stats["processed"] += 1
                        else:
                            stats["skipped"] += 1
                    else:
                        print(f"[Skip] Input file not found: {input_path}")
                        stats["skipped"] += 1
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print("="*80)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate time series statistics from combined_merged_outliers_removed.csv files (all levels)."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["LV1", "LV2", "LV3"],
        help="Levels to process (default: LV1 LV2 LV3)"
    )
    parser.add_argument(
        "--target",
        choices=["human", "llm"],
        default="llm",
        help="Target dataset: human or llm (default: llm)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to process (default: {DEFAULT_MODELS})"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        help=f"Domains to process (default: {DEFAULT_DOMAINS})"
    )
    parser.add_argument(
        "--process-root",
        type=Path,
        default=Path(PROCESS_ROOT),
        help=f"Root directory for processed data (default: {PROCESS_ROOT})"
    )
    
    args = parser.parse_args()
    
    generate_all_levels_timeseries_stats(
        process_root=args.process_root,
        models=args.models,
        domains=args.domains,
        target=args.target,
        levels=args.levels,
    )


if __name__ == "__main__":
    main()

