#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on TFIDF and SBERT CV
============================================================

Compare Human vs LLM CV (Coefficient of Variation) for:
- TFIDF embeddings: 10 dimensions (tfidf_1_cv to tfidf_10_cv)
- SBERT embeddings: 384 dimensions (sbert_1_cv to sbert_384_cv)

Tests:
- Each TFIDF dimension separately (10 tests)
- Each SBERT dimension separately (384 tests)

H0: P(Human CV > LLM CV) = 0.5
H1: P(Human CV > LLM CV) > 0.5

Uses paired comparisons: each human author matched with LLM author (same author_id, field, domain).
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
PROVIDERS = ("DS", "CL35", "G4OM")
LEVEL = "LV3"

# Number of dimensions
TFIDF_DIMS = 10
SBERT_DIMS = 384


def load_cv_data(domain: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load embedding CV data from author_timeseries_stats_embeddings.csv.
    
    Args:
        domain: Domain name (academic, blogs, news)
        label: Label (human or llm)
        provider: LLM provider (for llm label)
        level: LLM level (for llm label)
    """
    if label == "human":
        cv_path = DATA_ROOT / "human" / domain / "author_timeseries_stats_embeddings.csv"
    else:
        cv_path = DATA_ROOT / "LLM" / provider / level / domain / "author_timeseries_stats_embeddings.csv"
    
    if not cv_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(cv_path)
    return df


def get_tfidf_cv_columns(df: pd.DataFrame) -> List[str]:
    """Extract TFIDF CV column names from dataframe."""
    tfidf_cv_cols = [col for col in df.columns if col.startswith("tfidf_") and col.endswith("_cv")]
    # Sort numerically: tfidf_1_cv, tfidf_2_cv, ..., tfidf_10_cv
    tfidf_cv_cols.sort(key=lambda x: int(x.replace("tfidf_", "").replace("_cv", "")))
    return tfidf_cv_cols


def get_sbert_cv_columns(df: pd.DataFrame) -> List[str]:
    """Extract SBERT CV column names from dataframe."""
    sbert_cv_cols = [col for col in df.columns if col.startswith("sbert_") and col.endswith("_cv")]
    # Sort numerically: sbert_1_cv, sbert_2_cv, ..., sbert_384_cv
    sbert_cv_cols.sort(key=lambda x: int(x.replace("sbert_", "").replace("_cv", "")))
    return sbert_cv_cols


def compare_human_vs_llm_cv(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature_col: str,
    model: str,
) -> pd.DataFrame:
    """
    Compare Human vs LLM CV for paired authors for a specific embedding dimension.
    
    Args:
        human_df: Human CV data
        llm_df: LLM CV data
        feature_col: Column name for the CV feature (e.g., 'tfidf_1_cv', 'sbert_1_cv')
        model: Model name
    
    Returns:
        DataFrame with paired comparisons
    """
    # Merge on field and author_id
    merged = human_df[["field", "author_id", feature_col]].merge(
        llm_df[["field", "author_id", feature_col]],
        on=["field", "author_id"],
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
    
    # Compare: Human > LLM
    human_wins = (merged_clean[human_col] > merged_clean[llm_col]).astype(int)
    
    results = pd.DataFrame({
        "field": merged_clean["field"],
        "author_id": merged_clean["author_id"],
        "model": model,
        "feature": feature_col,
        "human_cv": merged_clean[human_col],
        "llm_cv": merged_clean[llm_col],
        "human_wins": human_wins,
    })
    
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
    """
    feature_comparisons = comparisons_df[comparisons_df["feature"] == feature]
    
    if len(feature_comparisons) == 0:
        return {
            "feature": feature,
            "n_comparisons": 0,
            "human_wins": 0,
            "human_win_rate": np.nan,
            "pvalue": np.nan,
            "significant": False,
        }
    
    n = len(feature_comparisons)
    k = feature_comparisons["human_wins"].sum()
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


def main():
    parser = argparse.ArgumentParser(
        description="Binomial test for TFIDF and SBERT CV (Human vs LLM)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=list(DOMAINS),
        default=list(DOMAINS),
        help="Domains to process (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(PROVIDERS),
        default=list(PROVIDERS),
        help="Models to process (default: all)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/embedding_cv/binomial_test",
        help="Output directory for results (default: results/embedding_cv/binomial_test)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BINOMIAL TEST: Human vs LLM TFIDF and SBERT CV")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"Models: {args.models}")
    print(f"Level: {LEVEL}")
    print(f"Alpha: {args.alpha}")
    print("=" * 80)
    
    all_comparisons = []
    
    # Process each domain
    for domain in args.domains:
        print(f"\nProcessing domain: {domain}")
        
        # Load Human data
        human_df = load_cv_data(domain, "human")
        if len(human_df) == 0:
            print(f"  Warning: No Human data found for {domain}")
            continue
        
        # Get feature columns
        tfidf_cv_cols = get_tfidf_cv_columns(human_df)
        sbert_cv_cols = get_sbert_cv_columns(human_df)
        
        print(f"  Human samples: {len(human_df)}")
        print(f"  TFIDF CV columns: {len(tfidf_cv_cols)}")
        print(f"  SBERT CV columns: {len(sbert_cv_cols)}")
        
        # Process each model
        for model in args.models:
            # Load LLM data
            llm_df = load_cv_data(domain, "llm", provider=model, level=LEVEL)
            if len(llm_df) == 0:
                print(f"  Warning: No LLM data found for {model}/{LEVEL}/{domain}")
                continue
            
            print(f"  {model} samples: {len(llm_df)}")
            
            # Compare TFIDF dimensions
            for tfidf_col in tfidf_cv_cols:
                comparisons = compare_human_vs_llm_cv(human_df, llm_df, tfidf_col, model)
                if len(comparisons) > 0:
                    comparisons["domain"] = domain
                    comparisons["embedding_type"] = "TFIDF"
                    all_comparisons.append(comparisons)
            
            # Compare SBERT dimensions
            for sbert_col in sbert_cv_cols:
                comparisons = compare_human_vs_llm_cv(human_df, llm_df, sbert_col, model)
                if len(comparisons) > 0:
                    comparisons["domain"] = domain
                    comparisons["embedding_type"] = "SBERT"
                    all_comparisons.append(comparisons)
    
    if len(all_comparisons) == 0:
        print("\nNo comparisons found!")
        return
    
    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    
    print(f"\nTotal comparisons: {len(comparisons_df):,}")
    print(f"  - Unique authors: {comparisons_df.groupby(['domain', 'field', 'author_id']).ngroups}")
    print(f"  - Unique features: {comparisons_df['feature'].nunique()}")
    print(f"  - Unique models: {comparisons_df['model'].nunique()}")
    print(f"  - TFIDF features: {len(comparisons_df[comparisons_df['embedding_type'] == 'TFIDF']['feature'].unique())}")
    print(f"  - SBERT features: {len(comparisons_df[comparisons_df['embedding_type'] == 'SBERT']['feature'].unique())}")
    
    # Perform binomial tests per model and per feature
    print("\nPerforming binomial tests...")
    test_results = []
    
    for model in args.models:
        model_comparisons = comparisons_df[comparisons_df["model"] == model]
        
        if len(model_comparisons) == 0:
            continue
        
        # Get unique features for this model
        unique_features = model_comparisons["feature"].unique()
        
        for feature in sorted(unique_features):
            feature_comparisons = model_comparisons[model_comparisons["feature"] == feature]
            result = perform_binomial_test(feature_comparisons, feature, args.alpha)
            result["model"] = model
            result["embedding_type"] = feature_comparisons["embedding_type"].iloc[0] if len(feature_comparisons) > 0 else "Unknown"
            test_results.append(result)
    
    test_results_df = pd.DataFrame(test_results)
    
    # Sort by model, embedding_type, then feature
    test_results_df["feature_num"] = test_results_df["feature"].str.extract(r"(\d+)").astype(int)
    test_results_df = test_results_df.sort_values(["model", "embedding_type", "feature_num"])
    test_results_df = test_results_df.drop("feature_num", axis=1)
    
    # Save detailed comparisons
    comparisons_output = output_dir / "embedding_cv_detailed_comparisons.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"\nSaved detailed comparisons: {comparisons_output}")
    
    # Save test results
    results_output = output_dir / "embedding_cv_binomial_test_results.csv"
    test_results_df.to_csv(results_output, index=False)
    print(f"Saved test results: {results_output}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for model in args.models:
        model_results = test_results_df[test_results_df["model"] == model]
        if len(model_results) == 0:
            continue
        
        print(f"\n{model}:")
        
        for emb_type in ["TFIDF", "SBERT"]:
            emb_results = model_results[model_results["embedding_type"] == emb_type]
            if len(emb_results) == 0:
                continue
            
            n_total = len(emb_results)
            n_significant = emb_results["significant"].sum()
            mean_win_rate = emb_results["human_win_rate"].mean()
            
            print(f"  {emb_type}:")
            print(f"    Total features: {n_total}")
            print(f"    Significant (p < {args.alpha}): {n_significant} ({n_significant/n_total*100:.1f}%)")
            print(f"    Mean human win rate: {mean_win_rate:.3f}")
            print(f"    Median human win rate: {emb_results['human_win_rate'].median():.3f}")
            
            # Top 10 features by win rate
            top_features = emb_results.nlargest(10, "human_win_rate")[
                ["feature", "human_win_rate", "pvalue", "significant"]
            ]
            print(f"    Top 10 features by win rate:")
            for _, row in top_features.iterrows():
                sig_marker = "***" if row["significant"] else ""
                print(f"      {row['feature']}: {row['human_win_rate']:.3f} (p={row['pvalue']:.4f}) {sig_marker}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()


