#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on E5-Large SBERT Embedding Drift
==========================================================================

Compare Human vs LLM drift using total drift (sum of all year-to-year drifts per author).
This script focuses ONLY on E5-Large SBERT drift.

Tests:
- E5-Large SBERT drift only (1024D)

H0: P(Human drift > LLM drift) = 0.5
H1: P(Human drift > LLM drift) > 0.5

Uses paired comparisons: each human author matched with LLM author (same author_id, field, domain).
No FDR correction applied - uses raw p-values for significance testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
LEVELS = ("LV1", "LV2", "LV3")
REP_SPACE = "sbert_e5"  # E5-Large SBERT


def load_drift_data(domain: str, label: str, provider: str | None = None, level: str | None = None) -> pd.DataFrame:
    """Load E5-Large SBERT drift data for a specific split."""
    if label == "human":
        drift_path = DATA_ROOT / "human" / domain / f"{REP_SPACE}_drift.csv"
    else:
        drift_path = DATA_ROOT / "LLM" / provider / level / domain / f"{REP_SPACE}_drift.csv"
    
    if not drift_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(drift_path)
    return df


def compare_human_vs_llm_drift(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    """
    Compare Human vs LLM total drift for paired authors (E5-Large SBERT only).
    
    IMPORTANT: Only uses common year pairs (year_from, year_to) for each author.
    If Human has 4 drifts but LLM has only 3 (due to NaN), only the 3 common
    year pairs are used to compute total drift for comparison.
    
    Args:
        human_df: DataFrame with human drift (with year_from, year_to, drift columns)
        llm_df: DataFrame with LLM drift (with year_from, year_to, drift columns)
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
) -> Dict:
    """
    Perform binomial test for E5-Large SBERT drift.
    
    H0: p = 0.5 (Human wins 50% of the time by chance)
    H1: p > 0.5 (Human wins significantly more than 50%)
    
    Args:
        comparisons_df: DataFrame with comparison results
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
        "n_comparisons": n,
        "human_wins": k,
        "human_win_rate": p_observed,
        "pvalue": test_result.pvalue,
        "significant": test_result.pvalue < alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binomial test for Human vs LLM E5-Large SBERT drift comparison"
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
        default=["LV3"],
        help="LLM levels to analyze (default: LV3 only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "drift" / "binomial_test_sbert_e5",
        help="Output directory for results",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Binomial Test: Human vs LLM E5-Large SBERT Embedding Drift")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"LLM Providers: {args.providers}")
    print(f"LLM Levels: {args.levels}")
    print(f"Representation space: {REP_SPACE.upper()} (1024D)")
    print(f"H0: P(Human drift > LLM drift) = 0.5")
    print(f"H1: P(Human drift > LLM drift) > 0.5")
    print(f"Significance level: alpha={args.alpha} (no FDR correction)")
    print()
    
    # Collect all comparisons
    all_comparisons = []
    
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Load human E5-Large SBERT drift data
        human_df = load_drift_data(domain, "human")
        if human_df.empty:
            print(f"  ⚠️  Human {REP_SPACE}: No data found")
            continue
        print(f"  Human {REP_SPACE}: {len(human_df)} drift measurements")
        
        # Load LLM data and compare
        for provider in args.providers:
            for level in args.levels:
                print(f"  Processing {provider}/{level}...")
                
                llm_df = load_drift_data(domain, "llm", provider, level)
                if llm_df.empty:
                    print(f"    ⚠️  No LLM data found")
                    continue
                
                # Compare E5-Large SBERT drift
                comparisons = compare_human_vs_llm_drift(
                    human_df,
                    llm_df,
                    provider,
                )
                
                if not comparisons.empty:
                    comparisons["level"] = level
                    all_comparisons.append(comparisons)
                    print(f"    {REP_SPACE}: {len(comparisons)} paired authors")
    
    if not all_comparisons:
        print("\n⚠️  No comparisons found!")
        return
    
    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    print(f"\nTotal comparisons: {len(comparisons_df):,}")
    print(f"  - Unique authors: {comparisons_df.groupby(['domain', 'field', 'author_id']).ngroups}")
    print(f"  - Models: {comparisons_df['model'].nunique()}")
    print(f"  - Levels: {comparisons_df['level'].nunique()}")
    
    # Perform binomial tests (per model, per level)
    print("\nPerforming binomial tests...")
    all_test_results = []
    
    for provider in args.providers:
        for level in args.levels:
            subset = comparisons_df[
                (comparisons_df["model"] == provider) &
                (comparisons_df["level"] == level)
            ]
            
            if subset.empty:
                continue
            
            # Perform test
            test_result = perform_binomial_test(subset, args.alpha)
            if test_result is not None:
                test_result["model"] = provider
                test_result["level"] = level
                all_test_results.append(test_result)
    
    if not all_test_results:
        print("⚠️  No test results!")
        return
    
    test_results_df = pd.DataFrame(all_test_results)
    
    # Sort by model, level
    test_results_df = test_results_df.sort_values(["model", "level"])
    
    # Print summary
    print("\n" + "=" * 80)
    print("BINOMIAL TEST RESULTS (E5-LARGE SBERT DRIFT)")
    print("=" * 80)
    print(f"\n{test_results_df.to_string(index=False)}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    n_significant = test_results_df["significant"].sum()
    n_total = len(test_results_df)
    
    print(f"\nE5-Large SBERT drift:")
    print(f"  Significant tests: {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)")
    print(f"  Mean win rate: {test_results_df['human_win_rate'].mean():.3f}")
    print(f"  Mean p-value: {test_results_df['pvalue'].mean():.4f}")
    
    # Per-model summary
    for provider in args.providers:
        provider_results = test_results_df[test_results_df["model"] == provider]
        if provider_results.empty:
            continue
        n_sig = provider_results["significant"].sum()
        n_tot = len(provider_results)
        print(f"\n{provider}:")
        print(f"  Significant: {n_sig}/{n_tot} ({100*n_sig/n_tot:.1f}%)")
        print(f"  Mean win rate: {provider_results['human_win_rate'].mean():.3f}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparisons
    comparisons_output = args.output_dir / "sbert_e5_drift_comparisons_detailed.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"\n✅ Saved detailed comparisons to: {comparisons_output}")
    
    # Save test results
    results_output = args.output_dir / "binomial_test_sbert_e5_drift_results.csv"
    test_results_df.to_csv(results_output, index=False)
    print(f"✅ Saved test results to: {results_output}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()


