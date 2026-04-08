#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on 20 CE CV Features
============================================================

Compare Human vs LLM CV (Coefficient of Variation) for 20 CE features.
Input source: trajectory_features_combined.csv

Tests:
- 20 CE features (each feature separately)

H0: P(Human CV > LLM CV) = 0.5
H1: P(Human CV > LLM CV) > 0.5

Uses paired comparisons: each human author matched with LLM author (same author_id, field, domain).
Uses FDR (Benjamini-Hochberg) correction for multiple testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
LEVELS = ("LV1", "LV2", "LV3")

# 20 CE CV features
CV20_FEATURES = [
    "Agreeableness_cv",
    "Conscientiousness_cv",
    "Extraversion_cv",
    "Neuroticism_cv",
    "Openness_cv",
    "average_word_length_cv",
    "avg_sentence_length_cv",
    "content_word_ratio_cv",
    "flesch_reading_ease_cv",
    "function_word_ratio_cv",
    "gunning_fog_cv",
    "num_words_cv",
    "polarity_cv",
    "subjectivity_cv",
    "vader_compound_cv",
    "vader_neg_cv",
    "vader_neu_cv",
    "vader_pos_cv",
    "verb_ratio_cv",
    "word_diversity_cv",
]


def load_trajectory_features(domain: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load trajectory_features_combined.csv for a specific split."""
    if label == "human":
        data_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
    else:
        data_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
    
    if not data_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    return df


def compare_human_vs_llm_cv(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature: str,
    model: str,
) -> pd.DataFrame:
    """
    Compare Human vs LLM CV for a specific feature using paired authors.
    
    Args:
        human_df: DataFrame with human data (with feature column)
        llm_df: DataFrame with LLM data (with feature column)
        feature: Feature name (e.g., "Agreeableness_cv")
        model: Model name (DS, G4B, etc.)
    
    Returns:
        DataFrame with comparison results (one row per author)
    """
    results = []
    
    # Match on author_id, field, domain
    merge_cols = ["author_id", "field", "domain"]
    
    # Merge on common authors
    merged = pd.merge(
        human_df[merge_cols + [feature]],
        llm_df[merge_cols + [feature]],
        on=merge_cols,
        suffixes=("_human", "_llm"),
        how="inner"
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Drop rows where either human or LLM has NaN for this feature
    merged_clean = merged.dropna(subset=[f"{feature}_human", f"{feature}_llm"])
    
    if merged_clean.empty:
        return pd.DataFrame()
    
    # Binary outcome: Human CV > LLM CV
    human_wins = (merged_clean[f"{feature}_human"] > merged_clean[f"{feature}_llm"]).astype(int)
    
    results_df = pd.DataFrame({
        "domain": merged_clean["domain"],
        "field": merged_clean["field"],
        "author_id": merged_clean["author_id"],
        "feature": feature,
        "model": model,
        "human_cv": merged_clean[f"{feature}_human"],
        "llm_cv": merged_clean[f"{feature}_llm"],
        "human_wins": human_wins,
    })
    
    return results_df


def perform_binomial_test(
    comparisons_df: pd.DataFrame,
    feature: str,
    alpha: float = 0.05,
) -> Dict:
    """
    Perform binomial test for a specific feature.
    
    H0: p = 0.5 (Human wins 50% of the time by chance)
    H1: p > 0.5 (Human wins significantly more than 50%)
    
    Args:
        comparisons_df: DataFrame with comparison results for this feature
        feature: Feature name
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    if comparisons_df.empty:
        return None
    
    n = len(comparisons_df)
    k = comparisons_df["human_wins"].sum()
    
    # Proportion of Human wins
    p_observed = k / n if n > 0 else 0.0
    
    # One-sided binomial test: H1: p > 0.5
    test_result = binomtest(k, n, p=0.5, alternative="greater")
    
    return {
        "feature": feature,
        "n_comparisons": n,
        "human_wins": k,
        "human_win_rate": p_observed,
        "pvalue": test_result.pvalue,
        "significant": test_result.pvalue < alpha,
    }


def apply_fdr_correction(test_results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Args:
        test_results: DataFrame with pvalue column
        alpha: FDR threshold
    
    Returns:
        DataFrame with added pvalue_adjusted column
    """
    if test_results.empty:
        return test_results
    
    pvalues = test_results["pvalue"].values
    n = len(pvalues)
    
    # Benjamini-Hochberg procedure
    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    
    # Calculate adjusted p-values
    adjusted_pvalues = np.zeros(n)
    adjusted_pvalues[sorted_indices[-1]] = sorted_pvalues[-1]
    
    for i in range(n - 2, -1, -1):
        adjusted_pvalues[sorted_indices[i]] = min(
            sorted_pvalues[i] * n / (i + 1),
            adjusted_pvalues[sorted_indices[i + 1]]
        )
    
    # Determine which hypotheses are rejected
    rejected = adjusted_pvalues <= alpha
    
    test_results = test_results.copy()
    test_results["pvalue_adjusted"] = adjusted_pvalues
    test_results["significant_adjusted"] = rejected
    
    return test_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binomial test for Human vs LLM CE CV comparison"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to analyze (default: all)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="LLM providers to analyze (default: all)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=LEVELS,
        default=list(LEVELS),
        help="LLM levels to analyze (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "ce_cv" / "binomial_test",
        help="Output directory for results",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for FDR correction (default: 0.05)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Binomial Test: Human vs LLM CE CV Comparison")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"LLM Providers: {args.providers}")
    print(f"LLM Levels: {args.levels}")
    print(f"Features: {len(CV20_FEATURES)} CE CV features")
    print(f"Input: trajectory_features_combined.csv")
    print(f"H0: P(Human CV > LLM CV) = 0.5")
    print(f"H1: P(Human CV > LLM CV) > 0.5")
    print(f"FDR correction: Benjamini-Hochberg (alpha={args.alpha})")
    print()
    
    # Collect all comparisons
    all_comparisons = []
    
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Load human data
        human_df = load_trajectory_features(domain, "human")
        if human_df.empty:
            print(f"  ⚠️  Human: No data found")
            continue
        
        # Check which CV features are available
        available_features = [f for f in CV20_FEATURES if f in human_df.columns]
        missing_features = [f for f in CV20_FEATURES if f not in human_df.columns]
        if missing_features:
            print(f"  ⚠️  Missing features: {missing_features}")
        print(f"  Human: {len(human_df)} authors, {len(available_features)}/{len(CV20_FEATURES)} features available")
        
        # Load LLM data and compare
        for provider in args.providers:
            for level in args.levels:
                print(f"  Processing {provider}/{level}...")
                
                llm_df = load_trajectory_features(domain, "llm", provider, level)
                if llm_df.empty:
                    print(f"    ⚠️  No LLM data found")
                    continue
                
                # Compare each feature for binomial test
                for feature in available_features:
                    if feature not in llm_df.columns:
                        continue
                    
                    comparisons = compare_human_vs_llm_cv(
                        human_df,
                        llm_df,
                        feature,
                        provider,
                    )
                    
                    if not comparisons.empty:
                        comparisons["level"] = level
                        all_comparisons.append(comparisons)
    
    if not all_comparisons:
        print("\n⚠️  No comparisons found!")
        return
    
    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    print(f"\nTotal comparisons: {len(comparisons_df):,}")
    print(f"  - Unique authors: {comparisons_df.groupby(['domain', 'field', 'author_id']).ngroups}")
    print(f"  - Features: {comparisons_df['feature'].nunique()}")
    print(f"  - Models: {comparisons_df['model'].nunique()}")
    print(f"  - Levels: {comparisons_df['level'].nunique()}")
    
    # Perform binomial tests (per feature, per model, per level)
    print("\nPerforming binomial tests...")
    all_test_results = []
    
    for feature in CV20_FEATURES:
        for provider in args.providers:
            for level in args.levels:
                subset = comparisons_df[
                    (comparisons_df["feature"] == feature) &
                    (comparisons_df["model"] == provider) &
                    (comparisons_df["level"] == level)
                ]
                
                if subset.empty:
                    continue
                
                # Perform test
                test_result = perform_binomial_test(subset, feature, args.alpha)
                if test_result is not None:
                    test_result["model"] = provider
                    test_result["level"] = level
                    all_test_results.append(test_result)
    
    if not all_test_results:
        print("⚠️  No test results!")
        return
    
    test_results_df = pd.DataFrame(all_test_results)
    
    # Apply FDR correction separately for each model/level combination
    print("\nApplying FDR correction (Benjamini-Hochberg)...")
    corrected_results = []
    
    for provider in args.providers:
        for level in args.levels:
            subset_results = test_results_df[
                (test_results_df["model"] == provider) &
                (test_results_df["level"] == level)
            ].copy()
            
            if not subset_results.empty:
                subset_results = apply_fdr_correction(subset_results, args.alpha)
                corrected_results.append(subset_results)
    
    final_results = pd.concat(corrected_results, ignore_index=True)
    
    # Sort by feature, model, level
    feature_order = {feat: i for i, feat in enumerate(CV20_FEATURES)}
    final_results["sort_order"] = final_results["feature"].map(feature_order)
    final_results = final_results.sort_values(["model", "level", "sort_order"]).drop("sort_order", axis=1)
    
    # Print CV comparison statistics for all 20 features
    print("\n" + "=" * 80)
    print("CV COMPARISON STATISTICS (All 20 CE Features)")
    print("=" * 80)
    
    for provider in args.providers:
        for level in args.levels:
            subset_comparisons = comparisons_df[
                (comparisons_df["model"] == provider) &
                (comparisons_df["level"] == level)
            ]
            
            if subset_comparisons.empty:
                continue
            
            print(f"\n{provider}/{level}:")
            print(f"{'Feature':<30s} {'N':>5s} {'Human CV Mean':>15s} {'LLM CV Mean':>15s} {'Human CV Std':>15s} {'LLM CV Std':>15s} {'Win Rate':>10s}")
            print("-" * 110)
            
            for feature in CV20_FEATURES:
                feature_comparisons = subset_comparisons[subset_comparisons["feature"] == feature]
                if feature_comparisons.empty:
                    continue
                
                human_cv_mean = feature_comparisons["human_cv"].mean()
                llm_cv_mean = feature_comparisons["llm_cv"].mean()
                human_cv_std = feature_comparisons["human_cv"].std()
                llm_cv_std = feature_comparisons["llm_cv"].std()
                win_rate = feature_comparisons["human_wins"].mean()
                n = len(feature_comparisons)
                
                print(f"{feature:<30s} {n:>5d} {human_cv_mean:>15.6f} {llm_cv_mean:>15.6f} {human_cv_std:>15.6f} {llm_cv_std:>15.6f} {win_rate:>10.3f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BINOMIAL TEST RESULTS")
    print("=" * 80)
    print(f"\n{final_results.to_string(index=False)}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for provider in args.providers:
        for level in args.levels:
            subset_results = final_results[
                (final_results["model"] == provider) &
                (final_results["level"] == level)
            ]
            
            if subset_results.empty:
                continue
            
            n_significant = subset_results["significant_adjusted"].sum()
            n_total = len(subset_results)
            
            print(f"\n{provider}/{level}:")
            print(f"  Significant tests (FDR-adjusted): {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)")
            print(f"  Mean win rate: {subset_results['human_win_rate'].mean():.3f}")
            print(f"  Mean p-value (adjusted): {subset_results['pvalue_adjusted'].mean():.4f}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparisons
    comparisons_output = args.output_dir / "ce_cv_comparisons_detailed.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"\n✅ Saved detailed comparisons to: {comparisons_output}")
    
    # Save test results
    results_output = args.output_dir / "binomial_test_ce_cv_results.csv"
    final_results.to_csv(results_output, index=False)
    print(f"✅ Saved test results to: {results_output}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
