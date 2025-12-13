#!/usr/bin/env python3
"""
Remove outliers from combined_merged.csv files.
For each model-feature combination, detect outliers using IQR method and set them to NaN.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

def remove_outliers_iqr(series: pd.Series, iqr_factor: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method. Returns boolean mask (True = keep, False = outlier)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        # No variability, keep all values
        return pd.Series([True] * len(series), index=series.index)
    
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    
    # Return mask: True for non-outliers, False for outliers
    return (series >= lower_bound) & (series <= upper_bound)


def remove_outliers_from_combined_merged(
    input_file: Path,
    output_file: Path,
    iqr_factor: float = 1.5,
    feature_columns: List[str] = None,
    models_filter: List[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove outliers from combined_merged.csv file.
    
    Args:
        input_file: Path to input combined_merged.csv
        output_file: Path to output file
        iqr_factor: IQR factor for outlier detection (default: 1.5)
        feature_columns: List of feature columns to process (None = auto-detect)
        models_filter: List of model names to process (None = process all models)
    
    Returns:
        Tuple of (processed DataFrame, statistics dictionary)
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Identify metadata columns
    metadata_cols = ['filename', 'path', 'label', 'domain', 'field', 'author_id', 'model', 'level']
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    
    # Auto-detect feature columns if not provided
    if feature_columns is None:
        feature_columns = [col for col in df.columns 
                          if col not in metadata_cols 
                          and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"  Processing {len(feature_columns)} feature columns: {feature_columns[:5]}...")
    
    # Identify LLM models
    LLM_MODELS = ['DS', 'G4B', 'G12B', 'LMK']
    
    # Check if 'model' column exists
    if 'model' not in df.columns:
        print("  Warning: 'model' column not found. Treating all data as single model.")
        df['model'] = 'ALL'
    else:
        # Keep NaN for human samples, don't fill with 'ALL'
        df['model'] = df['model'].copy()
    
    # Identify which samples are LLM vs Human
    # LLM samples: label='llm' OR model is in LLM_MODELS
    if 'label' in df.columns:
        is_llm = (df['label'] == 'llm') | (df['model'].isin(LLM_MODELS))
    else:
        is_llm = df['model'].isin(LLM_MODELS)
    
    # Determine which models to process
    if models_filter is not None:
        # Only process specified LLM models
        is_llm_to_process = is_llm & df['model'].isin(models_filter)
        print(f"  Processing only LLM models: {models_filter}")
        print(f"  Human samples: {len(df[~is_llm])} rows (will be kept unchanged)")
        print(f"  LLM samples to process: {is_llm_to_process.sum()} rows")
    else:
        # Process all LLM models
        is_llm_to_process = is_llm
        print(f"  Human samples: {len(df[~is_llm])} rows (will be kept unchanged)")
        print(f"  LLM samples to process: {is_llm_to_process.sum()} rows")
    
    if is_llm_to_process.sum() == 0:
        print("  ⚠️  Warning: No LLM samples to process. Returning original data.")
        df.to_csv(output_file, index=False)
        return df, {
            'total_rows': len(df), 
            'human_rows': len(df[~is_llm]),
            'llm_rows_processed': 0,
            'total_values': 0, 
            'total_outliers': 0, 
            'by_model_feature': {}
        }
    
    # Create outlier group identifier for LLM samples only
    # For human samples, we'll skip processing
    df['_outlier_group'] = None
    df.loc[is_llm_to_process, '_outlier_group'] = (
        df.loc[is_llm_to_process, 'model'].astype(str) + "_" + 
        df.loc[is_llm_to_process, 'domain'].fillna("UNKNOWN").astype(str)
    )
    
    # Statistics
    stats = {
        'total_rows': len(df),
        'human_rows': len(df[~is_llm]),
        'llm_rows_processed': is_llm_to_process.sum(),
        'total_values': 0,
        'total_outliers': 0,
        'by_model_feature': {}
    }
    
    # Create a copy for processing (keep all data, including human samples)
    df_processed = df.copy()
    
    # Process each model-feature combination (only for LLM samples)
    groups = df.loc[is_llm_to_process, '_outlier_group'].dropna().unique()
    print(f"\nProcessing {len(groups)} LLM model-domain group(s): {list(groups)}")
    
    for group_key in groups:
        # Only process LLM samples in this group
        group_mask = (df['_outlier_group'] == group_key) & is_llm_to_process
        group_data = df[group_mask]
        print(f"\n  Group: {group_key} ({len(group_data)} rows)")
        
        for feature_col in feature_columns:
            if feature_col not in df.columns:
                continue
            
            # Get all non-null values for this model-feature combination
            feature_series = group_data[feature_col].dropna()
            
            if len(feature_series) == 0:
                continue
            
            # Detect outliers using IQR
            mask = remove_outliers_iqr(feature_series, iqr_factor=iqr_factor)
            outliers_mask = ~mask  # True for outliers
            
            # Count outliers
            n_outliers = outliers_mask.sum()
            n_total = len(feature_series)
            
            if n_outliers > 0:
                # Set outliers to NaN in the processed DataFrame
                model_indices = group_data.index
                feature_indices = feature_series.index
                outlier_indices = feature_indices[outliers_mask]
                
                df_processed.loc[outlier_indices, feature_col] = np.nan
                
                stats['by_model_feature'][f"{group_key}_{feature_col}"] = {
                    'total': n_total,
                    'outliers': n_outliers,
                    'percentage': (n_outliers / n_total * 100) if n_total > 0 else 0
                }
                
                stats['total_values'] += n_total
                stats['total_outliers'] += n_outliers
            else:
                # Count total values even if no outliers found
                stats['total_values'] += n_total
    
    # Save processed DataFrame
    print(f"\nSaving processed data to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if '_outlier_group' in df_processed.columns:
        df_processed = df_processed.drop(columns=['_outlier_group'])
    df_processed.to_csv(output_file, index=False)
    print(f"  Saved {len(df_processed)} rows to {output_file}")
    
    return df_processed, stats


def main():
    parser = argparse.ArgumentParser(
        description="Remove outliers from combined_merged.csv files using IQR method per model-feature combination."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input combined_merged.csv file (required if not using --batch-llm)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output file (default: input file with '_outliers_removed' suffix)"
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR factor for outlier detection (default: 1.5)"
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="List of feature columns to process (default: auto-detect all numeric columns)"
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only process LLM models (DS, G4B, G12B, LMK). Human samples will be kept unchanged."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of specific models to process (e.g., DS G4B). If not specified, process all models."
    )
    parser.add_argument(
        "--batch-llm",
        action="store_true",
        help="Batch process all LLM combined_merged.csv files in dataset/process/LLM/"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/process"),
        help="Root directory for data (default: dataset/process)"
    )
    
    args = parser.parse_args()
    
    # Batch processing mode
    if args.batch_llm:
        llm_dir = args.data_root / "LLM"
        if not llm_dir.exists():
            print(f"❌ Error: LLM directory not found: {llm_dir}")
            return
        
        # Find all combined_merged.csv files in LLM directory
        csv_files = list(llm_dir.rglob("combined_merged.csv"))
        if not csv_files:
            print(f"❌ Error: No combined_merged.csv files found in {llm_dir}")
            return
        
        print("="*80)
        print("Batch Processing LLM Files")
        print("="*80)
        print(f"Found {len(csv_files)} files to process")
        print(f"IQR Factor: {args.iqr_factor}")
        print("="*80)
        
        total_stats = {
            'total_files': len(csv_files),
            'processed_files': 0,
            'total_rows': 0,
            'total_values': 0,
            'total_outliers': 0
        }
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'#'*80}")
            print(f"Processing file {i}/{len(csv_files)}: {csv_file}")
            print(f"{'#'*80}")
            
            output_file = csv_file.parent / f"{csv_file.stem}_outliers_removed{csv_file.suffix}"
            
            try:
                df_processed, stats = remove_outliers_from_combined_merged(
                    input_file=csv_file,
                    output_file=output_file,
                    iqr_factor=args.iqr_factor,
                    feature_columns=args.feature_columns,
                    models_filter=None  # Process all LLM models in the file
                )
                
                total_stats['processed_files'] += 1
                total_stats['total_rows'] += stats['total_rows']
                total_stats['total_values'] += stats['total_values']
                total_stats['total_outliers'] += stats['total_outliers']
                
                print(f"✅ Completed: {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")
                continue
        
        # Print summary
        print("\n" + "="*80)
        print("Batch Processing Summary")
        print("="*80)
        print(f"Total files found: {total_stats['total_files']}")
        print(f"Successfully processed: {total_stats['processed_files']}")
        print(f"Total rows processed: {total_stats['total_rows']}")
        print(f"Total values processed: {total_stats['total_values']}")
        print(f"Total outliers removed: {total_stats['total_outliers']}")
        if total_stats['total_values'] > 0:
            print(f"Overall outlier percentage: {total_stats['total_outliers'] / total_stats['total_values'] * 100:.2f}%")
        print("="*80)
        print("✅ Batch processing complete!")
        print("="*80)
        return
    
    # Single file processing mode
    if args.input is None:
        parser.error("--input is required when not using --batch-llm")
    
    # Determine which models to process
    models_filter = None
    if args.llm_only:
        models_filter = ['DS', 'G4B', 'G12B', 'LMK']
        print("  Mode: LLM-only (processing: DS, G4B, G12B, LMK)")
    elif args.models:
        models_filter = args.models
        print(f"  Mode: Custom models (processing: {models_filter})")
    
    # Set output path if not provided
    if args.output is None:
        input_stem = args.input.stem
        input_suffix = args.input.suffix
        output_dir = args.input.parent
        args.output = output_dir / f"{input_stem}_outliers_removed{input_suffix}"
    
    print("="*80)
    print("Remove Outliers from Combined Merged CSV")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"IQR Factor: {args.iqr_factor}")
    print("="*80)
    
    # Process file
    df_processed, stats = remove_outliers_from_combined_merged(
        input_file=args.input,
        output_file=args.output,
        iqr_factor=args.iqr_factor,
        feature_columns=args.feature_columns,
        models_filter=models_filter
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("Statistics")
    print("="*80)
    print(f"Total rows: {stats['total_rows']}")
    print(f"  - Human samples (kept unchanged): {stats.get('human_rows', 0)}")
    print(f"  - LLM samples processed: {stats.get('llm_rows_processed', 0)}")
    print(f"Total values processed (LLM only): {stats['total_values']}")
    print(f"Total outliers removed (LLM only): {stats['total_outliers']}")
    if stats['total_values'] > 0:
        print(f"Outlier percentage: {stats['total_outliers'] / stats['total_values'] * 100:.2f}%")
    
    print(f"\nOutliers by model-feature combination:")
    print("-" * 80)
    for key, value in sorted(stats['by_model_feature'].items()):
        if value['outliers'] > 0:
            print(f"  {key}: {value['outliers']}/{value['total']} ({value['percentage']:.2f}%)")
    
    print("\n" + "="*80)
    print("✅ Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()


