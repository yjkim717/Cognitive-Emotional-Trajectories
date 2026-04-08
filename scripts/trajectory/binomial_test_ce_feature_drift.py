#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on Individual CE Feature Drift
=======================================================================

Compare Human vs LLM drift for each of the 20 CE features individually.
Each feature's drift is computed as the absolute difference: |value_(i+1) - value_i|
after z-score normalization per author.

Tests:
- Each of 20 CE features separately

H0: P(Human drift > LLM drift) = 0.5
H1: P(Human drift > LLM drift) > 0.5

Uses paired comparisons: each human author matched with LLM author (same author_id, field, domain).
Only common year pairs are used for each model (if LLM has fewer year pairs due to NaN, only common ones are used).
Uses total drift (sum of all year-to-year drifts per author) for comparison.
No FDR correction applied - uses raw p-values for significance testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")

# 20 CE features
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


def load_drift_data(domain: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load CE feature drift data for a specific split.
    
    Args:
        domain: Domain name (academic, blogs, news)
        label: Label (human or llm)
        provider: LLM provider (for llm label)
        level: LLM level (for llm label)
    """
    if label == "human":
        drift_path = DATA_ROOT / "human" / domain / "ce_feature_drift.csv"
    else:
        drift_path = DATA_ROOT / "LLM" / provider / level / domain / "ce_feature_drift.csv"
    
    if not drift_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(drift_path)
    return df


def compare_human_vs_llm_drift(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature: str,
    model: str,
) -> pd.DataFrame:
    """
    Compare Human vs LLM total drift for paired authors for a specific feature.
    
    IMPORTANT: Only uses common year pairs (year_from, year_to) for each author.
    If Human has 4 drifts but LLM has only 3 (due to NaN), only the 3 common
    year pairs are used to compute total drift for comparison.
    
    Args:
        human_df: DataFrame with human drift (with year_from, year_to, drift, feature columns)
        llm_df: DataFrame with LLM drift (with year_from, year_to, drift, feature columns)
        feature: Feature name to filter
        model: Model name (DS, G4B, etc.)
    
    Returns:
        DataFrame with comparison results (one row per author)
    """
    # Filter by feature
    human_feature = human_df[human_df["feature"] == feature].copy()
    llm_feature = llm_df[llm_df["feature"] == feature].copy()
    
    if human_feature.empty or llm_feature.empty:
        return pd.DataFrame()
    
    results = []
    
    # Match on author_id, field, domain
    merge_cols = ["author_id", "field", "domain"]
    
    # For each paired author, find common year pairs and compute total drift only on common pairs
    for (author_id, field, domain), human_group in human_feature.groupby(merge_cols):
        llm_group = llm_feature[
            (llm_feature["author_id"] == author_id) &
            (llm_feature["field"] == field) &
            (llm_feature["domain"] == domain)
        ]
        
        if len(llm_group) == 0:
            continue
        
        # Find common year pairs
        human_years = set(zip(human_group["year_from"], human_group["year_to"]))
        llm_years = set(zip(llm_group["year_from"], llm_group["year_to"]))
        common_years = human_years & llm_years
        
        if len(common_years) == 0:
            # No common year pairs, skip this author
            continue
        
        # Compute total drift only for common year pairs
        human_common = human_group[human_group.apply(lambda row: (row["year_from"], row["year_to"]) in common_years, axis=1)]
        llm_common = llm_group[llm_group.apply(lambda row: (row["year_from"], row["year_to"]) in common_years, axis=1)]
        
        human_total_drift = human_common["drift"].sum()
        llm_total_drift = llm_common["drift"].sum()
        
        results.append({
            "author_id": author_id,
            "field": field,
            "domain": domain,
            "feature": feature,
            "model": model,
            "human_total_drift": human_total_drift,
            "llm_total_drift": llm_total_drift,
            "human_wins": 1 if human_total_drift > llm_total_drift else 0,
            "n_common_pairs": len(common_years),
        })
    
    return pd.DataFrame(results)


def perform_binomial_test(comparisons_df: pd.DataFrame, alpha: float = 0.05) -> Dict | None:
    """
    Perform one-sided binomial test on comparisons.
    
    H0: P(Human wins) = 0.5
    H1: P(Human wins) > 0.5
    
    Args:
        comparisons_df: DataFrame with 'human_wins' column (0 or 1)
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    if comparisons_df.empty:
        return None
    
    n = len(comparisons_df)
    k = comparisons_df["human_wins"].sum()
    p_hat = k / n if n > 0 else 0.0
    
    # One-sided binomial test: P(X >= k) where X ~ Bin(n, 0.5)
    result = binomtest(k, n, p=0.5, alternative='greater')
    p_value = result.pvalue
    
    return {
        "n_comparisons": n,
        "human_wins": k,
        "human_win_rate": p_hat,
        "pvalue": p_value,
        "significant": p_value < alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binomial test for Human vs LLM CE feature drift comparison"
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
        default=None,
        help="Output directory for results (default: binomial_test_ce_feature or binomial_test_ce_feature_shadow based on --use-shadow)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "results" / "drift" / "binomial_test_ce_feature"
    
    print("=" * 80)
    print("Binomial Test: Human vs LLM CE Feature Drift")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"LLM Providers: {args.providers}")
    print(f"LLM Levels: {args.levels}")
    print(f"Features: {len(CE_FEATURES)} CE features (tested individually)")
    print(f"Using: ce_feature_drift.csv files (from combined_merged.csv, outliers already removed)")
    print(f"Drift method: |value_(i+1) - value_i| (absolute difference after z-score normalization per author)")
    print(f"H0: P(Human drift > LLM drift) = 0.5")
    print(f"H1: P(Human drift > LLM drift) > 0.5")
    print(f"Significance level: alpha={args.alpha} (no FDR correction)")
    print()
    
    # Collect all comparisons
    all_comparisons = []
    
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Load human CE feature drift data
        human_df = load_drift_data(domain, "human")
        if human_df.empty:
            print(f"  ⚠️  Human: No data found")
            continue
        print(f"  Human: {len(human_df)} drift measurements")
        
        # Load LLM data and compare for each feature
        for provider in args.providers:
            for level in args.levels:
                print(f"  Processing {provider}/{level}...")
                
                llm_df = load_drift_data(domain, "llm", provider, level)
                if llm_df.empty:
                    print(f"    ⚠️  No LLM data found")
                    continue
                
                # Compare each feature separately
                for feature in CE_FEATURES:
                    comparisons = compare_human_vs_llm_drift(
                        human_df,
                        llm_df,
                        feature,
                        provider,
                    )
                    
                    if not comparisons.empty:
                        comparisons["level"] = level
                        all_comparisons.append(comparisons)
                        print(f"    {feature}: {len(comparisons)} paired authors")
    
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
    
    for provider in args.providers:
        for level in args.levels:
            for feature in CE_FEATURES:
                subset = comparisons_df[
                    (comparisons_df["model"] == provider) &
                    (comparisons_df["level"] == level) &
                    (comparisons_df["feature"] == feature)
                ]
                
                if subset.empty:
                    continue
                
                # Perform test
                test_result = perform_binomial_test(subset, args.alpha)
                if test_result is not None:
                    test_result["model"] = provider
                    test_result["level"] = level
                    test_result["feature"] = feature
                    all_test_results.append(test_result)
    
    if not all_test_results:
        print("⚠️  No test results!")
        return
    
    test_results_df = pd.DataFrame(all_test_results)
    
    # Sort by feature, model, level
    test_results_df = test_results_df.sort_values(["feature", "model", "level"])
    
    # Print summary
    print("\n" + "=" * 80)
    print("BINOMIAL TEST RESULTS (CE FEATURE DRIFT)")
    print("=" * 80)
    
    # Print per-feature summary for LV3 only
    print("\nResults by Feature (LV3 only):")
    print("-" * 80)
    lv3_results = test_results_df[test_results_df["level"] == "LV3"].copy()
    
    for feature in CE_FEATURES:
        feature_results = lv3_results[lv3_results["feature"] == feature]
        if feature_results.empty:
            continue
        
        n_sig = feature_results["significant"].sum()
        n_tot = len(feature_results)
        mean_win_rate = feature_results["human_win_rate"].mean()
        mean_pvalue = feature_results["pvalue"].mean()
        
        print(f"\n{feature}:")
        print(f"  Significant: {n_sig}/{n_tot} ({100*n_sig/n_tot:.1f}%)")
        print(f"  Mean win rate: {mean_win_rate:.3f}")
        print(f"  Mean p-value: {mean_pvalue:.4f}")
        print(f"  Per-model results:")
        for _, row in feature_results.iterrows():
            sig_mark = "***" if row["significant"] else "   "
            print(f"    {sig_mark} {row['model']}/LV3: win_rate={row['human_win_rate']:.3f}, p={row['pvalue']:.4f}, n={row['n_comparisons']}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    n_significant = test_results_df["significant"].sum()
    n_total = len(test_results_df)
    
    print(f"\nOverall:")
    print(f"  Significant tests: {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)")
    print(f"  Mean win rate: {test_results_df['human_win_rate'].mean():.3f}")
    print(f"  Mean p-value: {test_results_df['pvalue'].mean():.4f}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparisons
    comparisons_output = args.output_dir / "ce_feature_drift_comparisons_detailed.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"\n✅ Saved detailed comparisons to: {comparisons_output}")
    
    # Save test results
    results_output = args.output_dir / "binomial_test_ce_feature_drift_results.csv"
    test_results_df.to_csv(results_output, index=False)
    print(f"✅ Saved test results to: {results_output}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

