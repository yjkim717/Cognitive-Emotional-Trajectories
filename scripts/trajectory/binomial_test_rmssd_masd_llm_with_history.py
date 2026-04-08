#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM_with_history Comparison on RMSSD_norm and MASD_norm
===============================================================================

Compare Human vs LLM_with_history RMSSD_norm and MASD_norm for:
- CE features: 20 features (from author_timeseries_stats_merged.csv)

Tests:
- Each CE feature separately for RMSSD_norm (20 tests)
- Each CE feature separately for MASD_norm (20 tests)

H0: P(Human > LLM_with_history) = 0.5
H1: P(Human > LLM_with_history) > 0.5

Uses paired comparisons: each human author matched with LLM_with_history author (same author_id, field, domain).
Uses FDR (Benjamini-Hochberg) correction per model per metric (aligned with binomial_test_ce_rmssd.py / binomial_test_ce_masd.py).
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
PROVIDERS = ("DS", "CL35", "G4OM")  # LLM_with_history models
LEVELS = ("LV1", "LV2", "LV3")

# 20 CE RMSSD_norm features
RMSSD_NORM_FEATURES = [
    "Agreeableness_rmssd_norm",
    "Conscientiousness_rmssd_norm",
    "Extraversion_rmssd_norm",
    "Neuroticism_rmssd_norm",
    "Openness_rmssd_norm",
    "average_word_length_rmssd_norm",
    "avg_sentence_length_rmssd_norm",
    "content_word_ratio_rmssd_norm",
    "flesch_reading_ease_rmssd_norm",
    "function_word_ratio_rmssd_norm",
    "gunning_fog_rmssd_norm",
    "num_words_rmssd_norm",
    "polarity_rmssd_norm",
    "subjectivity_rmssd_norm",
    "vader_compound_rmssd_norm",
    "vader_neg_rmssd_norm",
    "vader_neu_rmssd_norm",
    "vader_pos_rmssd_norm",
    "verb_ratio_rmssd_norm",
    "word_diversity_rmssd_norm",
]

# 20 CE MASD_norm features
MASD_NORM_FEATURES = [
    "Agreeableness_masd_norm",
    "Conscientiousness_masd_norm",
    "Extraversion_masd_norm",
    "Neuroticism_masd_norm",
    "Openness_masd_norm",
    "average_word_length_masd_norm",
    "avg_sentence_length_masd_norm",
    "content_word_ratio_masd_norm",
    "flesch_reading_ease_masd_norm",
    "function_word_ratio_masd_norm",
    "gunning_fog_masd_norm",
    "num_words_masd_norm",
    "polarity_masd_norm",
    "subjectivity_masd_norm",
    "vader_compound_masd_norm",
    "vader_neg_masd_norm",
    "vader_neu_masd_norm",
    "vader_pos_masd_norm",
    "verb_ratio_masd_norm",
    "word_diversity_masd_norm",
]


def load_stats_data(domain: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load CE stats data from author_timeseries_stats_merged.csv."""
    if label == "human":
        data_path = DATA_ROOT / "human" / domain / "author_timeseries_stats_merged.csv"
    elif label == "llm_with_history":
        data_path = DATA_ROOT / "LLM_with_history" / provider / level / domain / "author_timeseries_stats_merged.csv"
    else:
        data_path = DATA_ROOT / "LLM" / provider / level / domain / "author_timeseries_stats_merged.csv"
    
    if not data_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    # Add domain column if missing
    if "domain" not in df.columns:
        df["domain"] = domain
    return df


def compare_human_vs_llm_metric(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature_col: str,
    model: str,
    metric_name: str,
) -> pd.DataFrame:
    """
    Compare Human vs LLM_with_history metric for paired authors for a specific feature.
    
    Args:
        human_df: Human data
        llm_df: LLM_with_history data
        feature_col: Column name for the metric feature
        model: Model name
        metric_name: Name of the metric (rmssd_norm or masd_norm)
    
    Returns:
        DataFrame with paired comparisons
    """
    # Merge on field, author_id, and domain
    merge_cols = ["field", "author_id", "domain"]
    
    # Check which merge columns exist
    available_cols = [col for col in merge_cols if col in human_df.columns and col in llm_df.columns]
    
    merged = pd.merge(
        human_df[available_cols + [feature_col]],
        llm_df[available_cols + [feature_col]],
        on=available_cols,
        how="inner",
        suffixes=("_human", "_llm"),
    )
    
    if len(merged) == 0:
        return pd.DataFrame()
    
    # Remove rows where either human or LLM has NaN for this feature
    human_col = f"{feature_col}_human"
    llm_col = f"{feature_col}_llm"
    merged_clean = merged.dropna(subset=[human_col, llm_col])
    
    if len(merged_clean) == 0:
        return pd.DataFrame()
    
    # Compare: Human > LLM_with_history
    human_wins = (merged_clean[human_col] > merged_clean[llm_col]).astype(int)
    
    results = pd.DataFrame({
        "field": merged_clean["field"],
        "author_id": merged_clean["author_id"],
        "model": model,
        "feature": feature_col,
        "metric": metric_name,
        f"human_{metric_name}": merged_clean[human_col],
        f"llm_with_history_{metric_name}": merged_clean[llm_col],
        "human_wins": human_wins,
    })
    
    if "domain" in merged_clean.columns:
        results["domain"] = merged_clean["domain"]
    
    return results


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
        comparisons_df: DataFrame with comparison results
        feature: Feature name
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    feature_comparisons = comparisons_df[comparisons_df["feature"] == feature]
    
    if len(feature_comparisons) == 0:
        return None
    
    n = len(feature_comparisons)
    k = feature_comparisons["human_wins"].sum()
    
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
    Apply Benjamini-Hochberg FDR correction to p-values (aligned with binomial_test_ce_rmssd/ce_masd).

    Args:
        test_results: DataFrame with pvalue column
        alpha: FDR threshold

    Returns:
        DataFrame with added pvalue_adjusted and significant_adjusted columns
    """
    if test_results.empty:
        return test_results

    pvalues = test_results["pvalue"].values
    n = len(pvalues)

    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]

    adjusted_pvalues = np.zeros(n)
    adjusted_pvalues[sorted_indices[-1]] = sorted_pvalues[-1]

    for i in range(n - 2, -1, -1):
        adjusted_pvalues[sorted_indices[i]] = min(
            sorted_pvalues[i] * n / (i + 1),
            adjusted_pvalues[sorted_indices[i + 1]],
        )

    rejected = adjusted_pvalues <= alpha

    test_results = test_results.copy()
    test_results["pvalue_adjusted"] = adjusted_pvalues
    test_results["significant_adjusted"] = rejected

    return test_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binomial test for Human vs LLM_with_history RMSSD_norm and MASD_norm comparison"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=["news"],  # Default to news since that's what we have for LLM_with_history
        help="Domains to analyze (default: news)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="LLM_with_history providers to analyze (default: all)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=LEVELS,
        default=["LV3"],  # Default to LV3 since that's what we have for LLM_with_history
        help="LLM_with_history levels to analyze (default: LV3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "rmssd_masd" / "binomial_test_llm_with_history",
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
    print("Binomial Test: Human vs LLM_with_history RMSSD_norm and MASD_norm Comparison")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"LLM_with_history Providers: {args.providers}")
    print(f"LLM_with_history Levels: {args.levels}")
    print(f"H0: P(Human > LLM_with_history) = 0.5")
    print(f"H1: P(Human > LLM_with_history) > 0.5")
    print(f"FDR correction: Benjamini-Hochberg (alpha={args.alpha})")
    print()
    
    # Collect all comparisons
    all_rmssd_comparisons = []
    all_masd_comparisons = []
    
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Load human data
        human_df = load_stats_data(domain, "human")
        if human_df.empty:
            print(f"  ⚠️  Human: No data found")
            continue
        
        # Check which features are available
        available_rmssd = [f for f in RMSSD_NORM_FEATURES if f in human_df.columns]
        available_masd = [f for f in MASD_NORM_FEATURES if f in human_df.columns]
        print(f"  Human: {len(human_df)} authors, {len(available_rmssd)} RMSSD_norm + {len(available_masd)} MASD_norm features available")
        
        # Process LLM_with_history data
        for provider in args.providers:
            for level in args.levels:
                print(f"  Processing LLM_with_history {provider}/{level}...")
                
                llm_df = load_stats_data(domain, "llm_with_history", provider, level)
                if llm_df.empty:
                    print(f"    ⚠️  No LLM_with_history data found")
                    continue
                
                # Process RMSSD_norm features
                for feature in available_rmssd:
                    if feature not in llm_df.columns:
                        continue
                    
                    comparisons = compare_human_vs_llm_metric(
                        human_df,
                        llm_df,
                        feature,
                        provider,
                        "rmssd_norm",
                    )
                    
                    if not comparisons.empty:
                        comparisons["level"] = level
                        all_rmssd_comparisons.append(comparisons)
                
                # Process MASD_norm features
                for feature in available_masd:
                    if feature not in llm_df.columns:
                        continue
                    
                    comparisons = compare_human_vs_llm_metric(
                        human_df,
                        llm_df,
                        feature,
                        provider,
                        "masd_norm",
                    )
                    
                    if not comparisons.empty:
                        comparisons["level"] = level
                        all_masd_comparisons.append(comparisons)
    
    # Combine all comparisons
    if not all_rmssd_comparisons and not all_masd_comparisons:
        print("\n⚠️  No comparisons found!")
        return
    
    if all_rmssd_comparisons:
        rmssd_comparisons_df = pd.concat(all_rmssd_comparisons, ignore_index=True)
        print(f"\nRMSSD_norm comparisons: {len(rmssd_comparisons_df):,}")
    else:
        rmssd_comparisons_df = pd.DataFrame()
    
    if all_masd_comparisons:
        masd_comparisons_df = pd.concat(all_masd_comparisons, ignore_index=True)
        print(f"MASD_norm comparisons: {len(masd_comparisons_df):,}")
    else:
        masd_comparisons_df = pd.DataFrame()
    
    # Perform binomial tests
    print("\nPerforming binomial tests...")
    all_test_results = []
    
    # Test RMSSD_norm features
    if not rmssd_comparisons_df.empty:
        for provider in args.providers:
            for level in args.levels:
                subset = rmssd_comparisons_df[
                    (rmssd_comparisons_df["model"] == provider) &
                    (rmssd_comparisons_df["level"] == level)
                ]
                
                if subset.empty:
                    continue
                
                for feature in RMSSD_NORM_FEATURES:
                    if feature not in subset["feature"].values:
                        continue
                    
                    test_result = perform_binomial_test(subset, feature, args.alpha)
                    if test_result is not None:
                        test_result["model"] = provider
                        test_result["level"] = level
                        test_result["metric"] = "rmssd_norm"
                        all_test_results.append(test_result)
    
    # Test MASD_norm features
    if not masd_comparisons_df.empty:
        for provider in args.providers:
            for level in args.levels:
                subset = masd_comparisons_df[
                    (masd_comparisons_df["model"] == provider) &
                    (masd_comparisons_df["level"] == level)
                ]
                
                if subset.empty:
                    continue
                
                for feature in MASD_NORM_FEATURES:
                    if feature not in subset["feature"].values:
                        continue
                    
                    test_result = perform_binomial_test(subset, feature, args.alpha)
                    if test_result is not None:
                        test_result["model"] = provider
                        test_result["level"] = level
                        test_result["metric"] = "masd_norm"
                        all_test_results.append(test_result)
    
    if not all_test_results:
        print("⚠️  No test results!")
        return

    test_results_df = pd.DataFrame(all_test_results)

    # Apply FDR correction separately for each (model, level, metric) - 20 tests each (aligned with ce_rmssd/ce_masd)
    print("\nApplying FDR correction (Benjamini-Hochberg) per model per level per metric...")
    corrected_parts = []
    for provider in args.providers:
        for level in args.levels:
            for metric_name in ["rmssd_norm", "masd_norm"]:
                subset = test_results_df[
                    (test_results_df["model"] == provider)
                    & (test_results_df["level"] == level)
                    & (test_results_df["metric"] == metric_name)
                ].copy()
                if not subset.empty:
                    subset = apply_fdr_correction(subset, args.alpha)
                    corrected_parts.append(subset)
    final_results = pd.concat(corrected_parts, ignore_index=True)

    # Aggregate results by metric, model
    print("\n" + "=" * 80)
    print("BINOMIAL TEST RESULTS SUMMARY")
    print("=" * 80)

    # Summary by metric (FDR-adjusted)
    for metric_name in ["rmssd_norm", "masd_norm"]:
        metric_results = final_results[final_results["metric"] == metric_name]
        if metric_results.empty:
            continue

        print(f"\n{metric_name.upper()}:")
        for provider in args.providers:
            for level in args.levels:
                provider_results = metric_results[
                    (metric_results["model"] == provider) & (metric_results["level"] == level)
                ]
                if provider_results.empty:
                    continue

                n_significant = provider_results["significant_adjusted"].sum()
                n_total = len(provider_results)
                mean_win_rate = provider_results["human_win_rate"].mean()
                mean_pvalue = provider_results["pvalue_adjusted"].mean()

                print(f"  {provider}/{level}: {n_significant}/{n_total} significant (FDR-adj) ({100*n_significant/n_total:.1f}%), "
                      f"mean win rate: {mean_win_rate:.3f}, mean p-value (adj): {mean_pvalue:.4f}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed comparisons
    if not rmssd_comparisons_df.empty:
        rmssd_output = args.output_dir / "rmssd_norm_comparisons_detailed.csv"
        rmssd_comparisons_df.to_csv(rmssd_output, index=False)
        print(f"\n✅ Saved RMSSD_norm comparisons to: {rmssd_output}")

    if not masd_comparisons_df.empty:
        masd_output = args.output_dir / "masd_norm_comparisons_detailed.csv"
        masd_comparisons_df.to_csv(masd_output, index=False)
        print(f"✅ Saved MASD_norm comparisons to: {masd_output}")

    # Save test results (with pvalue_adjusted, significant_adjusted)
    results_output = args.output_dir / "binomial_test_rmssd_masd_llm_with_history_results.csv"
    final_results.to_csv(results_output, index=False)
    print(f"✅ Saved test results to: {results_output}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
