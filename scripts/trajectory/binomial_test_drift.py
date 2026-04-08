#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on Embedding Drift
==========================================================

Compare Human vs LLM drift using total drift (sum of all year-to-year drifts per author).

Tests:
1. CE drift
2. TFIDF drift  
3. SBERT drift

H0: P(Human drift > LLM drift) = 0.5
H1: P(Human drift > LLM drift) > 0.5

Uses paired comparisons: each human author matched with LLM author (same author_id, field, domain).
Uses FDR (Benjamini-Hochberg) correction for multiple testing.
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
REP_SPACES = ("ce", "tfidf", "sbert")


def load_drift_data(domain: str, rep_space: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load drift data for a specific split."""
    if label == "human":
        drift_path = DATA_ROOT / "human" / domain / f"{rep_space}_drift.csv"
    else:
        drift_path = DATA_ROOT / "LLM" / provider / level / domain / f"{rep_space}_drift.csv"
    
    if not drift_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(drift_path)
    return df


def compute_total_drift_per_author(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total drift per author by summing all year-to-year drifts.
    
    Args:
        df: DataFrame with columns: author_id, field, domain, drift, etc.
    
    Returns:
        DataFrame with one row per author (author_id, field, domain) with total_drift
    """
    if df.empty:
        return pd.DataFrame()
    
    # Group by author (use only columns that exist)
    base_cols = ["author_id", "field", "domain"]
    optional_cols = ["label", "provider", "level", "model"]
    
    author_cols = base_cols + [col for col in optional_cols if col in df.columns]
    
    total_drift = (
        df.groupby(author_cols, dropna=False)["drift"]
        .sum()
        .reset_index()
        .rename(columns={"drift": "total_drift"})
    )
    
    return total_drift


def compare_human_vs_llm_drift(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    rep_space: str,
    model: str,
) -> pd.DataFrame:
    """
    Compare Human vs LLM total drift for paired authors.
    
    IMPORTANT: Only uses common year pairs (year_from, year_to) for each author.
    If Human has 4 drifts but LLM has only 3 (due to NaN), only the 3 common
    year pairs are used to compute total drift for comparison.
    
    Args:
        human_df: DataFrame with human drift (with year_from, year_to, drift columns)
        llm_df: DataFrame with LLM drift (with year_from, year_to, drift columns)
        rep_space: Representation space name (ce, tfidf, sbert)
        model: Model name (DS, G4B, etc.)
    
    Returns:
        DataFrame with comparison results (one row per author)
    """
    results = []
    
    # Match on author_id, field, domain
    merge_cols = ["author_id", "field", "domain"]
    
    # For each paired author, find common year pairs and compute total drift only on common pairs
    for (author_id, field, domain), human_group in human_df.groupby(merge_cols):
        llm_group = llm_df[
            (llm_df["author_id"] == author_id) &
            (llm_df["field"] == field) &
            (llm_df["domain"] == domain)
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
        
        # Compute total drift only on common year pairs
        human_common = human_group[
            human_group[["year_from", "year_to"]].apply(tuple, axis=1).isin(common_years)
        ]["drift"].sum()
        
        llm_common = llm_group[
            llm_group[["year_from", "year_to"]].apply(tuple, axis=1).isin(common_years)
        ]["drift"].sum()
        
        # Binary outcome: Human > LLM
        human_wins = 1 if human_common > llm_common else 0
        
        results.append({
            "domain": domain,
            "field": field,
            "author_id": author_id,
            "rep_space": rep_space,
            "model": model,
            "human_total_drift": human_common,
            "llm_total_drift": llm_common,
            "human_wins": human_wins,
            "n_common_years": len(common_years),
        })
    
    return pd.DataFrame(results)


def perform_binomial_test(
    comparisons_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform binomial test for each representation space.
    
    H0: p = 0.5 (Human wins 50% of the time by chance)
    H1: p > 0.5 (Human wins significantly more than 50%)
    
    Args:
        comparisons_df: DataFrame with comparison results
        alpha: Significance level
    
    Returns:
        DataFrame with test results
    """
    results = []
    
    # Test for each representation space
    for rep_space in REP_SPACES:
        space_comparisons = comparisons_df[comparisons_df["rep_space"] == rep_space]
        
        if len(space_comparisons) == 0:
            continue
        
        n = len(space_comparisons)
        k = space_comparisons["human_wins"].sum()
        
        # Proportion of Human wins
        p_observed = k / n if n > 0 else 0.0
        
        # One-sided binomial test: H1: p > 0.5
        test_result = binomtest(k, n, p=0.5, alternative="greater")
        
        results.append({
            "rep_space": rep_space,
            "n_comparisons": n,
            "human_wins": k,
            "human_win_rate": p_observed,
            "pvalue": test_result.pvalue,
            "significant": test_result.pvalue < alpha,
        })
    
    return pd.DataFrame(results)


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
        description="Binomial test for Human vs LLM drift comparison"
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
        default=PROJECT_ROOT / "results" / "drift" / "binomial_test",
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
    print("Binomial Test: Human vs LLM Embedding Drift")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"LLM Providers: {args.providers}")
    print(f"LLM Levels: {args.levels}")
    print(f"Representation spaces: {', '.join(REP_SPACES)}")
    print(f"H0: P(Human drift > LLM drift) = 0.5")
    print(f"H1: P(Human drift > LLM drift) > 0.5")
    print(f"FDR correction: Benjamini-Hochberg (alpha={args.alpha})")
    print()
    
    # Collect all comparisons
    all_comparisons = []
    
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Load human drift data for each rep_space (keep year_from, year_to)
        human_drifts = {}
        for rep_space in REP_SPACES:
            human_df = load_drift_data(domain, rep_space, "human")
            if not human_df.empty:
                human_drifts[rep_space] = human_df
                print(f"  Human {rep_space}: {len(human_df)} drift measurements")
            else:
                print(f"  ⚠️  Human {rep_space}: No data found")
        
        # Load LLM data and compare
        for provider in args.providers:
            for level in args.levels:
                print(f"  Processing {provider}/{level}...")
                
                llm_drifts = {}
                for rep_space in REP_SPACES:
                    llm_df = load_drift_data(domain, rep_space, "llm", provider, level)
                    if not llm_df.empty:
                        llm_drifts[rep_space] = llm_df
                    else:
                        print(f"    ⚠️  {rep_space}: No LLM data found")
                
                # Compare each rep_space (using common year pairs only)
                for rep_space in REP_SPACES:
                    if rep_space not in human_drifts or rep_space not in llm_drifts:
                        continue
                    
                    comparisons = compare_human_vs_llm_drift(
                        human_drifts[rep_space],
                        llm_drifts[rep_space],
                        rep_space,
                        provider,
                    )
                    
                    if not comparisons.empty:
                        comparisons["level"] = level
                        all_comparisons.append(comparisons)
                        print(f"    {rep_space}: {len(comparisons)} paired authors")
    
    if not all_comparisons:
        print("\n⚠️  No comparisons found!")
        return
    
    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    print(f"\nTotal comparisons: {len(comparisons_df):,}")
    print(f"  - Unique authors: {comparisons_df.groupby(['domain', 'field', 'author_id']).ngroups}")
    print(f"  - Rep spaces: {comparisons_df['rep_space'].nunique()}")
    print(f"  - Models: {comparisons_df['model'].nunique()}")
    print(f"  - Levels: {comparisons_df['level'].nunique()}")
    
    # Perform binomial tests (per rep_space, per model, per level)
    print("\nPerforming binomial tests...")
    all_test_results = []
    
    for rep_space in REP_SPACES:
        for provider in args.providers:
            for level in args.levels:
                subset = comparisons_df[
                    (comparisons_df["rep_space"] == rep_space) &
                    (comparisons_df["model"] == provider) &
                    (comparisons_df["level"] == level)
                ]
                
                if subset.empty:
                    continue
                
                # Perform test
                test_results = perform_binomial_test(subset, args.alpha)
                if not test_results.empty:
                    test_results["model"] = provider
                    test_results["level"] = level
                    all_test_results.append(test_results)
    
    if not all_test_results:
        print("⚠️  No test results!")
        return
    
    test_results_df = pd.concat(all_test_results, ignore_index=True)
    
    # Apply FDR correction separately for each rep_space
    print("\nApplying FDR correction (Benjamini-Hochberg)...")
    corrected_results = []
    
    for rep_space in REP_SPACES:
        space_results = test_results_df[test_results_df["rep_space"] == rep_space].copy()
        if not space_results.empty:
            space_results = apply_fdr_correction(space_results, args.alpha)
            corrected_results.append(space_results)
    
    final_results = pd.concat(corrected_results, ignore_index=True)
    
    # Sort by rep_space, model, level
    rep_space_order = {rep: i for i, rep in enumerate(REP_SPACES)}
    final_results["sort_order"] = final_results["rep_space"].map(rep_space_order)
    final_results = final_results.sort_values(["sort_order", "model", "level"]).drop("sort_order", axis=1)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BINOMIAL TEST RESULTS")
    print("=" * 80)
    print(f"\n{final_results.to_string(index=False)}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for rep_space in REP_SPACES:
        space_results = final_results[final_results["rep_space"] == rep_space]
        if space_results.empty:
            continue
        
        n_significant = space_results["significant_adjusted"].sum()
        n_total = len(space_results)
        
        print(f"\n{rep_space.upper()} drift:")
        print(f"  Significant tests (FDR-adjusted): {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)")
        print(f"  Mean win rate: {space_results['human_win_rate'].mean():.3f}")
        print(f"  Mean p-value (adjusted): {space_results['pvalue_adjusted'].mean():.4f}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparisons
    comparisons_output = args.output_dir / "drift_comparisons_detailed.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"\n✅ Saved detailed comparisons to: {comparisons_output}")
    
    # Save test results
    results_output = args.output_dir / "binomial_test_drift_results.csv"
    final_results.to_csv(results_output, index=False)
    print(f"✅ Saved test results to: {results_output}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
