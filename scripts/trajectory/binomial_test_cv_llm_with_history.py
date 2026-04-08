#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM_with_history Comparison on CV (Coefficient of Variation)
=====================================================================================

Compare Human vs LLM_with_history CV for:
- CE features: 20 features (from author_timeseries_stats_merged.csv)
- TFIDF embeddings: 10 dimensions (from author_timeseries_stats_embeddings.csv)
- SBERT embeddings: 384 dimensions (from author_timeseries_stats_embeddings.csv)

Tests:
- Each CE feature separately (20 tests)
- Each TFIDF dimension separately (10 tests)
- Each SBERT dimension separately (384 tests)

H0: P(Human CV > LLM_with_history CV) = 0.5
H1: P(Human CV > LLM_with_history CV) > 0.5

Uses paired comparisons: each human author matched with LLM_with_history author (same author_id, field, domain).
CE: FDR (Benjamini-Hochberg) per model/level (aligned with binomial_test_ce_cv.py). TFIDF/SBERT: raw p-values (aligned with binomial_test_embedding_cv.py).
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

# 20 CE CV features
CE_CV_FEATURES = [
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

# Number of dimensions
TFIDF_DIMS = 10
SBERT_DIMS = 384


def load_ce_cv_data(domain: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load CE CV data from author_timeseries_stats_merged.csv."""
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


def load_embedding_cv_data(domain: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load embedding CV data from author_timeseries_stats_embeddings.csv."""
    if label == "human":
        data_path = DATA_ROOT / "human" / domain / "author_timeseries_stats_embeddings.csv"
    elif label == "llm_with_history":
        data_path = DATA_ROOT / "LLM_with_history" / provider / level / domain / "author_timeseries_stats_embeddings.csv"
    else:
        data_path = DATA_ROOT / "LLM" / provider / level / domain / "author_timeseries_stats_embeddings.csv"
    
    if not data_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    # Add domain column if missing
    if "domain" not in df.columns:
        df["domain"] = domain
    return df


def get_tfidf_cv_columns(df: pd.DataFrame) -> List[str]:
    """Extract TFIDF CV column names from dataframe."""
    return [col for col in df.columns if col.startswith("tfidf_") and col.endswith("_cv")]


def get_sbert_cv_columns(df: pd.DataFrame) -> List[str]:
    """Extract SBERT CV column names from dataframe."""
    return [col for col in df.columns if col.startswith("sbert_") and col.endswith("_cv")]


def compare_human_vs_llm_cv(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature_col: str,
    model: str,
) -> pd.DataFrame:
    """
    Compare Human vs LLM_with_history CV for paired authors for a specific feature.
    
    Args:
        human_df: Human CV data
        llm_df: LLM_with_history CV data
        feature_col: Column name for the CV feature
        model: Model name
    
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
        "human_cv": merged_clean[human_col],
        "llm_with_history_cv": merged_clean[llm_col],
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
    Apply Benjamini-Hochberg FDR correction to p-values (aligned with binomial_test_ce_cv.py).

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
        description="Binomial test for Human vs LLM_with_history CV comparison"
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
        default=PROJECT_ROOT / "results" / "cv" / "binomial_test_llm_with_history",
        help="Output directory for results",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for FDR (CE) / raw (TFIDF/SBERT) (default: 0.05)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Binomial Test: Human vs LLM_with_history CV Comparison")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"LLM_with_history Providers: {args.providers}")
    print(f"LLM_with_history Levels: {args.levels}")
    print(f"H0: P(Human CV > LLM_with_history CV) = 0.5")
    print(f"H1: P(Human CV > LLM_with_history CV) > 0.5")
    print(f"CE: FDR (Benjamini-Hochberg) alpha={args.alpha}; TFIDF/SBERT: raw p-values")
    print()
    
    # Collect all comparisons
    all_ce_comparisons = []
    all_embedding_comparisons = []
    
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Process CE CV features
        print(f"  Processing CE CV features...")
        human_ce_df = load_ce_cv_data(domain, "human")
        if human_ce_df.empty:
            print(f"    ⚠️  Human CE: No data found")
        else:
            available_ce_features = [f for f in CE_CV_FEATURES if f in human_ce_df.columns]
            print(f"    Human CE: {len(human_ce_df)} authors, {len(available_ce_features)}/{len(CE_CV_FEATURES)} features available")
            
            for provider in args.providers:
                for level in args.levels:
                    llm_ce_df = load_ce_cv_data(domain, "llm_with_history", provider, level)
                    if llm_ce_df.empty:
                        print(f"    ⚠️  {provider}/{level}: No LLM_with_history CE data found")
                        continue
                    
                    for feature in available_ce_features:
                        if feature not in llm_ce_df.columns:
                            continue
                        
                        comparisons = compare_human_vs_llm_cv(
                            human_ce_df,
                            llm_ce_df,
                            feature,
                            provider,
                        )
                        
                        if not comparisons.empty:
                            comparisons["level"] = level
                            comparisons["rep_space"] = "ce"
                            all_ce_comparisons.append(comparisons)
        
        # Process Embedding CV features (TFIDF + SBERT)
        print(f"  Processing Embedding CV features (TFIDF + SBERT)...")
        human_emb_df = load_embedding_cv_data(domain, "human")
        if human_emb_df.empty:
            print(f"    ⚠️  Human Embeddings: No data found")
        else:
            tfidf_cols = get_tfidf_cv_columns(human_emb_df)
            sbert_cols = get_sbert_cv_columns(human_emb_df)
            print(f"    Human Embeddings: {len(human_emb_df)} authors, {len(tfidf_cols)} TFIDF + {len(sbert_cols)} SBERT dimensions")
            
            for provider in args.providers:
                for level in args.levels:
                    llm_emb_df = load_embedding_cv_data(domain, "llm_with_history", provider, level)
                    if llm_emb_df.empty:
                        print(f"    ⚠️  {provider}/{level}: No LLM_with_history Embedding data found")
                        continue
                    
                    # Process TFIDF dimensions
                    for feature in tfidf_cols:
                        if feature not in llm_emb_df.columns:
                            continue
                        
                        comparisons = compare_human_vs_llm_cv(
                            human_emb_df,
                            llm_emb_df,
                            feature,
                            provider,
                        )
                        
                        if not comparisons.empty:
                            comparisons["level"] = level
                            comparisons["rep_space"] = "tfidf"
                            all_embedding_comparisons.append(comparisons)
                    
                    # Process SBERT dimensions
                    for feature in sbert_cols:
                        if feature not in llm_emb_df.columns:
                            continue
                        
                        comparisons = compare_human_vs_llm_cv(
                            human_emb_df,
                            llm_emb_df,
                            feature,
                            provider,
                        )
                        
                        if not comparisons.empty:
                            comparisons["level"] = level
                            comparisons["rep_space"] = "sbert"
                            all_embedding_comparisons.append(comparisons)
    
    # Combine all comparisons
    if not all_ce_comparisons and not all_embedding_comparisons:
        print("\n⚠️  No comparisons found!")
        return
    
    if all_ce_comparisons:
        ce_comparisons_df = pd.concat(all_ce_comparisons, ignore_index=True)
        print(f"\nCE CV comparisons: {len(ce_comparisons_df):,}")
    else:
        ce_comparisons_df = pd.DataFrame()
    
    if all_embedding_comparisons:
        embedding_comparisons_df = pd.concat(all_embedding_comparisons, ignore_index=True)
        print(f"Embedding CV comparisons: {len(embedding_comparisons_df):,}")
    else:
        embedding_comparisons_df = pd.DataFrame()
    
    # Perform binomial tests
    print("\nPerforming binomial tests...")
    all_test_results = []
    
    # Test CE features
    if not ce_comparisons_df.empty:
        for provider in args.providers:
            for level in args.levels:
                subset = ce_comparisons_df[
                    (ce_comparisons_df["model"] == provider) &
                    (ce_comparisons_df["level"] == level)
                ]
                
                if subset.empty:
                    continue
                
                for feature in CE_CV_FEATURES:
                    if feature not in subset["feature"].values:
                        continue
                    
                    test_result = perform_binomial_test(subset, feature, args.alpha)
                    if test_result is not None:
                        test_result["model"] = provider
                        test_result["level"] = level
                        test_result["rep_space"] = "ce"
                        all_test_results.append(test_result)
    
    # Test Embedding features
    if not embedding_comparisons_df.empty:
        for provider in args.providers:
            for level in args.levels:
                for rep_space in ["tfidf", "sbert"]:
                    subset = embedding_comparisons_df[
                        (embedding_comparisons_df["model"] == provider) &
                        (embedding_comparisons_df["level"] == level) &
                        (embedding_comparisons_df["rep_space"] == rep_space)
                    ]
                    
                    if subset.empty:
                        continue
                    
                    # Aggregate across all dimensions for this rep_space
                    unique_features = subset["feature"].unique()
                    for feature in unique_features:
                        test_result = perform_binomial_test(subset, feature, args.alpha)
                        if test_result is not None:
                            test_result["model"] = provider
                            test_result["level"] = level
                            test_result["rep_space"] = rep_space
                            all_test_results.append(test_result)
    
    if not all_test_results:
        print("⚠️  No test results!")
        return

    test_results_df = pd.DataFrame(all_test_results)

    # Apply FDR for CE per (model, level); for TFIDF/SBERT use raw (pvalue_adjusted = pvalue)
    print("\nApplying FDR correction for CE (Benjamini-Hochberg)...")
    corrected_parts = []
    for rep_space in ["ce", "tfidf", "sbert"]:
        subset = test_results_df[test_results_df["rep_space"] == rep_space]
        if subset.empty:
            continue
        if rep_space == "ce":
            for provider in args.providers:
                for level in args.levels:
                    sub = subset[(subset["model"] == provider) & (subset["level"] == level)].copy()
                    if not sub.empty:
                        sub = apply_fdr_correction(sub, args.alpha)
                        corrected_parts.append(sub)
        else:
            subset = subset.copy()
            subset["pvalue_adjusted"] = subset["pvalue"]
            subset["significant_adjusted"] = subset["significant"]
            corrected_parts.append(subset)
    final_results = pd.concat(corrected_parts, ignore_index=True)

    # Sort by rep_space, model, level, feature
    final_results = final_results.sort_values(["rep_space", "model", "level", "feature"])

    # Aggregate results by rep_space and model
    print("\n" + "=" * 80)
    print("BINOMIAL TEST RESULTS SUMMARY")
    print("=" * 80)

    # Summary by rep_space (CE: FDR-adjusted; TFIDF/SBERT: raw)
    for rep_space in ["ce", "tfidf", "sbert"]:
        space_results = final_results[final_results["rep_space"] == rep_space]
        if space_results.empty:
            continue

        label = "FDR-adjusted" if rep_space == "ce" else "raw"
        print(f"\n{rep_space.upper()} CV ({label}):")
        for provider in args.providers:
            provider_results = space_results[space_results["model"] == provider]
            if provider_results.empty:
                continue

            n_significant = provider_results["significant_adjusted"].sum()
            n_total = len(provider_results)
            mean_win_rate = provider_results["human_win_rate"].mean()
            mean_pvalue = provider_results["pvalue_adjusted"].mean()

            print(f"  {provider}: {n_significant}/{n_total} significant ({100*n_significant/n_total:.1f}%), "
                  f"mean win rate: {mean_win_rate:.3f}, mean p-value (adj): {mean_pvalue:.4f}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed comparisons
    if not ce_comparisons_df.empty:
        ce_output = args.output_dir / "ce_cv_comparisons_detailed.csv"
        ce_comparisons_df.to_csv(ce_output, index=False)
        print(f"\n✅ Saved CE CV comparisons to: {ce_output}")

    if not embedding_comparisons_df.empty:
        emb_output = args.output_dir / "embedding_cv_comparisons_detailed.csv"
        embedding_comparisons_df.to_csv(emb_output, index=False)
        print(f"✅ Saved Embedding CV comparisons to: {emb_output}")

    # Save test results (with pvalue_adjusted, significant_adjusted)
    results_output = args.output_dir / "binomial_test_cv_llm_with_history_results.csv"
    final_results.to_csv(results_output, index=False)
    print(f"✅ Saved test results to: {results_output}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
