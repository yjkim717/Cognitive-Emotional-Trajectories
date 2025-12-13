#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on Trajectory Features
==============================================================

Compare Human vs LLM for trajectory features from trajectory_features_combined.csv:
1. CE-VAR: CE variability features (_cv, _rmssd_norm, _masd_norm)
2. CE-GEO: CE geometry features (ce_mean_distance, ce_net_displacement, etc.)
3. TF-IDF-GEO: TF-IDF geometry features (tfidf_mean_distance, tfidf_net_displacement, etc.)
4. SBERT-GEO: SBERT geometry features (sbert_mean_distance, sbert_net_displacement, etc.)

Each comparison yields a binary outcome (Human_metric > LLM_metric).
Binomial tests evaluate whether Human wins occur significantly more than the 50% random baseline.
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import binomtest
from pathlib import Path
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

# Feature groups
CE_VAR_SUFFIXES = ['_cv', '_rmssd_norm', '_masd_norm']
CE_GEO_FEATURES = ['ce_mean_distance', 'ce_std_distance', 'ce_net_displacement', 
                   'ce_path_length', 'ce_tortuosity']
TFIDF_GEO_FEATURES = ['tfidf_mean_distance', 'tfidf_std_distance', 'tfidf_net_displacement',
                      'tfidf_path_length', 'tfidf_tortuosity']
SBERT_GEO_FEATURES = ['sbert_mean_distance', 'sbert_std_distance', 'sbert_net_displacement',
                      'sbert_path_length', 'sbert_tortuosity']

# Exclude direction_consistency features
EXCLUDED_FEATURES = ['ce_direction_consistency', 'tfidf_direction_consistency', 'sbert_direction_consistency']

# Metadata columns to exclude
METADATA_COLS = {'field', 'author_id', 'sample_count', 'domain', 'label', 'provider', 
                 'level', 'model', 'ce_n_years', 'tfidf_n_years', 'sbert_n_years',
                 'ce_direction_consistency', 'tfidf_direction_consistency', 'sbert_direction_consistency'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binomial test for Human vs LLM comparison on trajectory features"
    )
    parser.add_argument(
        '--human-data',
        type=str,
        required=True,
        help='Path to human trajectory_features_combined.csv file directory'
    )
    parser.add_argument(
        '--llm-data-dir',
        type=str,
        required=True,
        help='Directory containing LLM trajectory_features_combined.csv files'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['DS', 'G4B', 'G12B', 'LMK'],
        help='LLM models to include (default: DS G4B G12B LMK)'
    )
    parser.add_argument(
        '--level',
        type=str,
        default='LV3',
        help='LLM level to use (default: LV3)'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=['academic', 'news', 'blogs'],
        help='Domains to analyze (default: academic news blogs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='micro_results/binomial/binomial_test_trajectory_features.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    return parser.parse_args()


def load_human_data(human_path: Path, domain: str) -> pd.DataFrame:
    """Load Human trajectory features data."""
    domain_file = human_path / domain / 'trajectory_features_combined.csv'
    if not domain_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(domain_file)
    df['domain'] = domain
    return df


def load_llm_data(llm_data_dir: Path, model: str, level: str, domain: str) -> pd.DataFrame:
    """Load LLM trajectory features data for a specific model."""
    model_file = llm_data_dir / model / level / domain / 'trajectory_features_combined.csv'
    if not model_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(model_file)
    df['domain'] = domain
    df['model'] = model
    df['level'] = level
    return df


def classify_feature(feature_name: str) -> str:
    """Classify a feature into one of the four groups."""
    # CE-VAR: ends with variability suffixes
    for suffix in CE_VAR_SUFFIXES:
        if feature_name.endswith(suffix):
            return 'CE-VAR'
    
    # CE-GEO
    if feature_name in CE_GEO_FEATURES:
        return 'CE-GEO'
    
    # TF-IDF-GEO
    if feature_name in TFIDF_GEO_FEATURES:
        return 'TF-IDF-GEO'
    
    # SBERT-GEO
    if feature_name in SBERT_GEO_FEATURES:
        return 'SBERT-GEO'
    
    return 'OTHER'


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get features grouped by type."""
    groups = {
        'CE-VAR': [],
        'CE-GEO': [],
        'TF-IDF-GEO': [],
        'SBERT-GEO': []
    }
    
    for col in df.columns:
        if col in METADATA_COLS:
            continue
        if col in EXCLUDED_FEATURES:
            continue  # Exclude direction_consistency features
        feature_type = classify_feature(col)
        if feature_type in groups:
            groups[feature_type].append(col)
    
    return groups


def compare_human_vs_llm_feature(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature: str,
    model: str
) -> pd.DataFrame:
    """
    Compare Human vs LLM for each author × feature combination.
    Returns a DataFrame with comparison results.
    """
    results = []
    
    # Check if feature exists
    if feature not in human_df.columns or feature not in llm_df.columns:
        return pd.DataFrame()
    
    # Get unique author-field combinations from Human data
    human_groups = human_df.groupby(['domain', 'field', 'author_id'])
    
    for (domain, field, author_id), human_group in human_groups:
        # Get Human value for this author-field-feature
        human_value = human_group[feature].dropna()
        
        if len(human_value) == 0:
            continue
        
        human_val = human_value.iloc[0]  # Should be unique per author-field
        
        # Get LLM value for the same author-field-feature-model
        llm_group = llm_df[
            (llm_df['domain'] == domain) &
            (llm_df['field'] == field) &
            (llm_df['author_id'] == author_id) &
            (llm_df['model'] == model)
        ]
        
        if len(llm_group) == 0:
            continue
        
        llm_value = llm_group[feature].dropna()
        
        if len(llm_value) == 0:
            continue
        
        llm_val = llm_value.iloc[0]  # Should be unique per author-field-model
        
        # Skip if either value is NaN
        if pd.isna(human_val) or pd.isna(llm_val):
            continue
        
        # Binary outcome: Human > LLM
        human_wins = 1 if human_val > llm_val else 0
        
        feature_type = classify_feature(feature)
        
        results.append({
            'domain': domain,
            'field': field,
            'author_id': author_id,
            'feature': feature,
            'feature_type': feature_type,
            'model': model,
            'human_value': human_val,
            'llm_value': llm_val,
            'human_wins': human_wins
        })
    
    return pd.DataFrame(results)


def perform_binomial_test(comparisons_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform binomial test for each feature group (and overall).
    H0: p = 0.5 (Human wins 50% of the time by chance)
    
    For most features: H1: p > 0.5 (Human wins significantly more than 50%)
    For TF-IDF-GEO: Use two-sided test or auto-detect direction (since LLM values may be higher)
    """
    results = []
    
    # Test for each feature group
    feature_groups = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    
    for group in feature_groups:
        group_comparisons = comparisons_df[comparisons_df['feature_type'] == group]
        
        if len(group_comparisons) == 0:
            continue
        
        n = len(group_comparisons)
        k = group_comparisons['human_wins'].sum()
        
        # Proportion of Human wins
        p_observed = k / n
        
        # For TF-IDF-GEO, use two-sided test or auto-detect direction
        # For other features, use one-sided (greater) as originally designed
        if group == 'TF-IDF-GEO':
            # Two-sided test: test if p != 0.5 (either direction)
            test_result = binomtest(k, n, p=0.5, alternative='two-sided')
            direction = 'two-sided'
            # Also test both directions for interpretation
            test_greater = binomtest(k, n, p=0.5, alternative='greater')
            test_less = binomtest(k, n, p=0.5, alternative='less')
            
            # Determine which direction is significant
            if p_observed > 0.5:
                direction_note = 'Human > LLM' if test_greater.pvalue < alpha else 'not significant'
            elif p_observed < 0.5:
                direction_note = 'LLM > Human' if test_less.pvalue < alpha else 'not significant'
            else:
                direction_note = 'equal'
        else:
            # One-sided test: p > 0.5 (Human wins more)
            test_result = binomtest(k, n, p=0.5, alternative='greater')
            direction = 'greater'
            direction_note = 'Human > LLM' if test_result.pvalue < alpha else 'not significant'
        
        # Significance
        is_significant = test_result.pvalue < alpha
        
        results.append({
            'feature_group': group,
            'n_comparisons': n,
            'human_wins': k,
            'human_win_rate': p_observed,
            'pvalue': test_result.pvalue,
            'test_direction': direction,
            'direction_note': direction_note if group == 'TF-IDF-GEO' else None,
            'significant': is_significant,
            'significant_star': '***' if test_result.pvalue < 0.001 else
                               '**' if test_result.pvalue < 0.01 else
                               '*' if test_result.pvalue < 0.05 else ''
        })
    
    # Overall test (all groups combined)
    n_total = len(comparisons_df)
    k_total = comparisons_df['human_wins'].sum()
    
    if n_total > 0:
        p_observed_total = k_total / n_total
        test_result_total = binomtest(k_total, n_total, p=0.5, alternative='greater')
        is_significant_total = test_result_total.pvalue < alpha
        
        results.append({
            'feature_group': 'OVERALL',
            'n_comparisons': n_total,
            'human_wins': k_total,
            'human_win_rate': p_observed_total,
            'pvalue': test_result_total.pvalue,
            'significant': is_significant_total,
            'significant_star': '***' if test_result_total.pvalue < 0.001 else
                               '**' if test_result_total.pvalue < 0.01 else
                               '*' if test_result_total.pvalue < 0.05 else ''
        })
    
    return pd.DataFrame(results)


def main():
    args = parse_args()
    
    human_path = Path(args.human_data)
    llm_data_dir = Path(args.llm_data_dir)
    
    if not human_path.exists():
        raise FileNotFoundError(f"Human data directory not found: {human_path}")
    if not llm_data_dir.exists():
        raise FileNotFoundError(f"LLM data directory not found: {llm_data_dir}")
    
    print("="*80)
    print("BINOMIAL TEST: Human vs LLM Trajectory Features")
    print("="*80)
    print(f"Domains: {args.domains}")
    print(f"Models: {args.models}")
    print(f"Level: {args.level}")
    print(f"Feature Groups: CE-VAR, CE-GEO, TF-IDF-GEO, SBERT-GEO")
    
    # Load and combine data from all domains
    all_comparisons = []
    
    for domain in args.domains:
        print(f"\nProcessing domain: {domain}")
        
        # Load Human data
        human_df = load_human_data(human_path, domain)
        if len(human_df) == 0:
            print(f"  Warning: No Human data found for {domain}")
            continue
        print(f"  Human samples: {len(human_df)}")
        
        # Get feature groups
        feature_groups = get_feature_groups(human_df)
        print(f"  Feature counts:")
        for group, features in feature_groups.items():
            print(f"    {group}: {len(features)} features")
        
        # Load LLM data for each model
        for model in args.models:
            llm_df = load_llm_data(llm_data_dir, model, args.level, domain)
            if len(llm_df) == 0:
                print(f"  Warning: No LLM data found for {model}/{args.level}/{domain}")
                continue
            
            print(f"  {model} samples: {len(llm_df)}")
            
            # Compare all features
            all_features = []
            for group_features in feature_groups.values():
                all_features.extend(group_features)
            
            for feature in all_features:
                comparisons = compare_human_vs_llm_feature(
                    human_df, llm_df, feature, model
                )
                
                if len(comparisons) > 0:
                    all_comparisons.append(comparisons)
    
    if len(all_comparisons) == 0:
        print("\nNo comparisons found!")
        return
    
    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    print(f"\nTotal comparisons: {len(comparisons_df):,}")
    print(f"  - Unique authors: {comparisons_df.groupby(['domain', 'field', 'author_id']).ngroups}")
    print(f"  - Unique features: {comparisons_df['feature'].nunique()}")
    print(f"  - Unique models: {comparisons_df['model'].nunique()}")
    print(f"  - Feature groups: {', '.join(comparisons_df['feature_type'].unique())}")
    
    # Perform binomial tests
    print("\nPerforming binomial tests...")
    test_results = perform_binomial_test(comparisons_df, args.alpha)
    
    # Sort by feature group order
    group_order = {'CE-VAR': 0, 'CE-GEO': 1, 'TF-IDF-GEO': 2, 'SBERT-GEO': 3, 'OVERALL': 4}
    test_results['sort_order'] = test_results['feature_group'].map(group_order)
    test_results = test_results.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparisons
    comparisons_output = output_path.parent / f"{output_path.stem}_detailed.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"\nDetailed comparisons saved to: {comparisons_output}")
    
    # Save test results
    test_results.to_csv(output_path, index=False)
    print(f"Test results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BINOMIAL TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Significance level (alpha): {args.alpha}")
    print(f"\nH₀: Human and LLM have equal values (p = 0.5)")
    print(f"H₁: Human values are greater (p > 0.5)")
    print("\nResults by Feature Group:")
    print("-"*80)
    
    for _, row in test_results.iterrows():
        sig_marker = row['significant_star'] if row['significant'] else ''
           direction_info = ""
           if row['feature_group'] == 'TF-IDF-GEO' and row.get('direction_note'):
               direction_info = f" [{row['direction_note']}]"
           print(f"{row['feature_group']:15s} | Win Rate: {row['human_win_rate']:.1%} | "
                     f"Wins: {row['human_wins']:6,}/{row['n_comparisons']:6,} | "
                     f"p-value: {row['pvalue']:.6f} {sig_marker}{direction_info}")
    
    # Overall summary
    overall = test_results[test_results['feature_group'] == 'OVERALL'].iloc[0]
    print("\n" + "="*80)
    print("OVERALL RESULT:")
    print(f"Human wins: {overall['human_win_rate']:.1%} ({overall['human_wins']:,}/{overall['n_comparisons']:,})")
    print(f"p-value: {overall['pvalue']:.6f} {overall['significant_star']}")
    if overall['significant']:
        print("→ Human trajectory features are systematically > LLM")
    print("="*80)


if __name__ == '__main__':
    main()







