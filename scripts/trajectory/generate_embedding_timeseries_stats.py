#!/usr/bin/env python3
"""
Generate time series statistics (variance, CV, RMSSD, MASD) for TFIDF and SBERT embeddings.

This script processes combined_with_embeddings.csv files and generates 
author_timeseries_stats_embeddings.csv files containing variance, CV, RMSSD, and MASD 
for each author based on TFIDF (10D) and SBERT (384D) embedding dimensions.

For each embedding dimension (tfidf_1 to tfidf_10, sbert_1 to sbert_384), computes:
- variance (sample variance)
- cv (coefficient of variation = std / |mean|)
- rmssd (root mean square of successive differences)
- masd (mean absolute successive differences)
- rmssd_norm (normalized RMSSD = rmssd / |mean|)
- masd_norm (normalized MASD = masd / |mean|)

Output file: author_timeseries_stats_embeddings.csv
Location: Same directory as combined_with_embeddings.csv
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


def compute_timeseries_stats(series: pd.Series, normalize: bool = False, robust: bool = False) -> Dict[str, float]:
    """Compute variance, CV, RMSSD, MASD, and normalized RMSSD/MASD for a series.
    
    Args:
        series: Time series data (already sorted by time)
        normalize: Whether to normalize (not used in this version, kept for API compatibility)
        robust: Whether to use robust statistics (not used in this version, kept for API compatibility)
    
    Returns:
        Dictionary with keys: variance, cv, rmssd, masd, rmssd_norm, masd_norm
    """
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


DEFAULT_DOMAINS = ["academic", "news", "blogs"]


def process_single_dataset(
    combined_csv_path: Path,
    output_csv_path: Path,
    target: str = "llm",
    domain: str = None,
    model: str = None,
    level: str = None,
) -> bool:
    """
    Process a single combined_with_embeddings.csv file and generate author_timeseries_stats_embeddings.csv.
    
    Args:
        combined_csv_path: Path to combined_with_embeddings.csv input file
        output_csv_path: Path to output author_timeseries_stats_embeddings.csv file
        target: "human" or "llm"
        domain: Domain name (for logging)
        model: Model name (for logging)
        level: Level name (for logging)
    """
    if not combined_csv_path.exists():
        print(f"[Skip] File not found: {combined_csv_path}")
        return False
    
    try:
        print(f"\nProcessing: {combined_csv_path}")
        df = pd.read_csv(combined_csv_path)
        
        if df.empty:
            print(f"[Skip] Empty combined_with_embeddings CSV: {combined_csv_path}")
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
        
        # Get TFIDF and SBERT vector columns
        tfidf_cols = [col for col in df.columns if col.startswith("tfidf_") and col.replace("tfidf_", "").isdigit()]
        sbert_cols = [col for col in df.columns if col.startswith("sbert_") and col.replace("sbert_", "").isdigit()]
        
        # Sort columns numerically (tfidf_1, tfidf_2, ..., tfidf_10, sbert_1, sbert_2, ..., sbert_384)
        tfidf_cols.sort(key=lambda x: int(x.replace("tfidf_", "")))
        sbert_cols.sort(key=lambda x: int(x.replace("sbert_", "")))
        
        numeric_cols = tfidf_cols + sbert_cols
        
        if not numeric_cols:
            print(f"[Skip] No TFIDF or SBERT vector columns found")
            return False
        
        print(f"  Found {len(tfidf_cols)} TFIDF dimensions and {len(sbert_cols)} SBERT dimensions")
        
        # Remove metadata columns from numeric_cols if they were accidentally included
        metadata_cols_to_remove = [author_col, field_col, domain_col, model_col, level_col, "year", "item_index"]
        for col in [c for c in metadata_cols_to_remove if c]:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
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
            
            # Compute statistics for each embedding dimension
            stats_dict = {
                "field": field_val,
                "author_id": author_val,
                "sample_count": sample_count,
            }
            
            for feature_col in numeric_cols:
                feature_series = group_sorted[feature_col].dropna()
                
                if len(feature_series) == 0:
                    # No valid data for this feature
                    for stat_name in ["variance", "cv", "rmssd", "masd", "rmssd_norm", "masd_norm"]:
                        stats_dict[f"{feature_col}_{stat_name}"] = np.nan
                else:
                    # Compute all time series statistics (original and normalized)
                    ts_stats = compute_timeseries_stats(feature_series, normalize=False)
                    for stat_name, stat_value in ts_stats.items():
                        # Save all stats including normalized versions
                        if stat_name in ["variance", "cv", "rmssd", "masd", "rmssd_norm", "masd_norm"]:
                            stats_dict[f"{feature_col}_{stat_name}"] = stat_value
            
            results_list.append(stats_dict)
        
        result = pd.DataFrame(results_list)
        
        # Reorder columns: metadata first, then features with consistent ordering
        metadata_cols = ["field", "author_id", "sample_count"]
        feature_stat_cols = [c for c in result.columns if c not in metadata_cols]
        
        # Sort feature stats columns: first by prefix (tfidf vs sbert), then by dimension number, then by stat type
        def sort_key(col_name):
            if '_' not in col_name:
                return ('', 0, col_name)
            parts = col_name.rsplit('_', 1)
            feature_part = parts[0]  # e.g., tfidf_1, sbert_1
            stat_part = parts[1]  # e.g., cv, variance
            
            # Extract prefix and number
            if '_' in feature_part:
                prefix, num_str = feature_part.rsplit('_', 1)
                try:
                    num = int(num_str)
                    return (prefix, num, stat_part)
                except ValueError:
                    return (prefix, 0, stat_part)
            return (feature_part, 0, stat_part)
        
        feature_stat_cols.sort(key=sort_key)
        
        result = result[metadata_cols + feature_stat_cols]
        
        # Sort for readability: by field then author
        result = result.sort_values(["field", "author_id"])
        
        # Save to output path
        os.makedirs(output_csv_path.parent, exist_ok=True)
        result.to_csv(output_csv_path, index=False)
        
        print(f"  ✅ Saved: {output_csv_path}")
        print(f"     Processed {len(result)} field-author pairs across {len(df)} samples")
        print(f"     Features: {len(tfidf_cols)} TFIDF + {len(sbert_cols)} SBERT dimensions")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to process {combined_csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_embedding_timeseries_stats(
    target: Optional[str] = None,
    domain: Optional[str] = None,
    model: Optional[str] = None,
    level: Optional[str] = None,
    models: Optional[list[str]] = None,
    levels: Optional[list[str]] = None,
    domains: Optional[list[str]] = None,
    output_filename: str = "author_timeseries_stats_embeddings.csv",
):
    """
    Generate author_timeseries_stats_embeddings.csv for all or specified datasets.
    
    Args:
        target: "human", "llm", or None (process both)
        domain: Specific domain (academic, news, blogs) or None (all domains)
        model: Specific model (DS, G4B, G12B, LMK, CL35, G4OM) or None (all models)
        level: Specific level (LV1, LV2, LV3) or None (all levels)
        models: List of models to process (overrides model parameter)
        levels: List of levels to process (overrides level parameter)
        domains: List of domains to process (overrides domain parameter)
        output_filename: Output CSV filename (default: "author_timeseries_stats_embeddings.csv")
    """
    process_root = Path(PROCESS_ROOT)
    processed = 0
    skipped = 0
    errors = 0
    
    print("="*70)
    print("GENERATING AUTHOR TIMESERIES STATISTICS FOR EMBEDDINGS (TFIDF + SBERT)")
    print("="*70)
    print(f"Input files: combined_with_embeddings.csv")
    print(f"Output filename: {output_filename}")
    print(f"Statistics included: variance, CV, RMSSD, MASD (and normalized versions)")
    print("="*70)
    
    # Process Human data
    if target is None or target == "human":
        human_root = process_root / "human"
        if human_root.exists():
            domains_to_process = domains if domains else ([domain] if domain else DEFAULT_DOMAINS)
            
            for domain_name in domains_to_process:
                combined_path = human_root / domain_name / "combined_with_embeddings.csv"
                output_path = human_root / domain_name / output_filename
                
                if process_single_dataset(
                    combined_path, output_path, target="human", domain=domain_name
                ):
                    processed += 1
                else:
                    skipped += 1
        else:
            print("[Skip] Human data root not found")
    
    # Process LLM_with_history (DS/CL35/G4OM, LV3, all specified domains)
    if target == "llm_with_history":
        print("\n📁 Processing LLM_with_history data...")
        history_models = ["DS", "CL35", "G4OM"]
        domains_to_process = domains if domains else DEFAULT_DOMAINS
        for model_name in history_models:
            for domain_name in domains_to_process:
                combined_path = process_root / "LLM_with_history" / model_name / "LV3" / domain_name / "combined_with_embeddings.csv"
                output_path = process_root / "LLM_with_history" / model_name / "LV3" / domain_name / output_filename
                if combined_path.exists():
                    if process_single_dataset(
                        combined_path, output_path,
                        target="llm", domain=domain_name,
                        model=model_name, level="LV3"
                    ):
                        processed += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
    
    # Process LLM data
    elif target is None or target == "llm":
        llm_root = process_root / "LLM"
        if llm_root.exists():
            models_to_process = models if models else ([model] if model else None)
            levels_to_process = levels if levels else ([level] if level else None)
            domains_to_process = domains if domains else ([domain] if domain else DEFAULT_DOMAINS)
            
            if models_to_process is None:
                # Auto-detect models
                models_to_process = [d.name for d in llm_root.iterdir() if d.is_dir()]
            
            if levels_to_process is None:
                # Auto-detect levels
                for model_name in models_to_process:
                    model_dir = llm_root / model_name
                    if model_dir.exists():
                        levels_to_process = [d.name for d in model_dir.iterdir() if d.is_dir()]
                        break
            
            total_tasks = len(models_to_process) * len(levels_to_process) * len(domains_to_process)
            current_task = 0
            
            for model_name in models_to_process:
                model_dir = llm_root / model_name
                if not model_dir.exists():
                    print(f"[Skip] Model directory not found: {model_dir}")
                    continue
                
                for level_name in levels_to_process:
                    level_dir = model_dir / level_name
                    if not level_dir.exists():
                        print(f"[Skip] Level directory not found: {level_dir}")
                        continue
                    
                    for domain_name in domains_to_process:
                        current_task += 1
                        combined_path = level_dir / domain_name / "combined_with_embeddings.csv"
                        output_path = level_dir / domain_name / output_filename
                        
                        if process_single_dataset(
                            combined_path, output_path, 
                            target="llm", domain=domain_name, model=model_name, level=level_name
                        ):
                            processed += 1
                        else:
                            skipped += 1
        else:
            print("[Skip] LLM data root not found")
    
    print("\n" + "="*70)
    print(f"SUMMARY: {processed} processed, {skipped} skipped, {errors} errors")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate author_timeseries_stats_embeddings.csv files (variance, CV, RMSSD, MASD) from combined_with_embeddings.csv for TFIDF and SBERT embeddings.",
    )
    parser.add_argument(
        "--target",
        choices=["human", "llm", "llm_with_history"],
        default=None,
        help="Target: 'human', 'llm', or 'llm_with_history' (default: both human and llm)",
    )
    parser.add_argument(
        "--domain",
        choices=["academic", "blogs", "news"],
        default=None,
        help="Domain to process (default: all)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to process (default: all)",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Level to process (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to process (overrides --model)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="List of levels to process (overrides --level)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["academic", "blogs", "news"],
        default=None,
        help="List of domains to process (overrides --domain)",
    )
    parser.add_argument(
        "--output-filename",
        default="author_timeseries_stats_embeddings.csv",
        help="Output filename (default: author_timeseries_stats_embeddings.csv)",
    )
    
    args = parser.parse_args()
    
    generate_all_embedding_timeseries_stats(
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

