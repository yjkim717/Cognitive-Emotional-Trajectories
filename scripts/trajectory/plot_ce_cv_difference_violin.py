#!/usr/bin/env python3
"""
Plot violin plots showing Human - LLM differences for CE CV features.

For each LLM model, compute differences between human and LLM samples,
then create violin plots showing the distribution of differences.
Each plot shows 20 CE CV features in a 1x20 layout (one row, 20 columns).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
LLM_WITH_HISTORY_PROVIDERS = ("DS", "CL35", "G4OM")
LEVEL = "LV3"

# CE CV features to plot (20 features)
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

# Feature display names (full names)
FEATURE_DISPLAY_NAMES = {
    "Agreeableness_cv": "Agreeableness",
    "Conscientiousness_cv": "Conscientiousness",
    "Extraversion_cv": "Extraversion",
    "Neuroticism_cv": "Neuroticism",
    "Openness_cv": "Openness",
    "average_word_length_cv": "Average Word Length",
    "avg_sentence_length_cv": "Average Sentence Length",
    "content_word_ratio_cv": "Content Word Ratio",
    "flesch_reading_ease_cv": "Flesch Reading Ease",
    "function_word_ratio_cv": "Function Word Ratio",
    "gunning_fog_cv": "Gunning Fog",
    "num_words_cv": "Number of Words",
    "polarity_cv": "Polarity",
    "subjectivity_cv": "Subjectivity",
    "vader_compound_cv": "VADER Compound",
    "vader_neg_cv": "VADER Negative",
    "vader_neu_cv": "VADER Neutral",
    "vader_pos_cv": "VADER Positive",
    "verb_ratio_cv": "Verb Ratio",
    "word_diversity_cv": "Word Diversity",
}


def load_human_data(domains: List[str], llm_with_history: bool = False) -> pd.DataFrame:
    """Load and combine human data from all domains."""
    frames = []
    for domain in domains:
        filename = "author_timeseries_stats_merged.csv" if llm_with_history else "trajectory_features_combined.csv"
        csv_path = DATA_ROOT / "human" / domain / filename
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["domain"] = domain
            frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_llm_data(provider: str, level: str, domains: List[str], llm_with_history: bool = False) -> pd.DataFrame:
    """Load and combine LLM or LLM_with_history data from all domains for a specific provider/level."""
    frames = []
    root_name = "LLM_with_history" if llm_with_history else "LLM"
    filename = "author_timeseries_stats_merged.csv" if llm_with_history else "trajectory_features_combined.csv"
    for domain in domains:
        csv_path = DATA_ROOT / root_name / provider / level / domain / filename
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["domain"] = domain
            frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    return combined


def compute_differences(human_df: pd.DataFrame, llm_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Compute Human - LLM differences for matching (field, author_id) pairs.
    
    Returns a DataFrame with differences for each feature.
    """
    # Merge on field, author_id, and domain to avoid cross-domain mismatches
    merge_keys = ["field", "author_id", "domain"]

    human_df = human_df.copy()
    llm_df = llm_df.copy()
    for key in merge_keys:
        if key in human_df.columns:
            human_df[key] = human_df[key].astype(str).str.strip()
        if key in llm_df.columns:
            llm_df[key] = llm_df[key].astype(str).str.strip()
    
    # Ensure both dataframes have the merge keys
    if not all(key in human_df.columns for key in merge_keys):
        raise ValueError(f"Human data missing merge keys: {merge_keys}")
    if not all(key in llm_df.columns for key in merge_keys):
        raise ValueError(f"LLM data missing merge keys: {merge_keys}")
    
    # Merge to align samples
    merged = pd.merge(
        human_df[merge_keys + features],
        llm_df[merge_keys + features],
        on=merge_keys,
        how="inner",
        suffixes=("_human", "_llm")
    )
    
    # Compute differences
    diff_data = {"field": merged["field"], "author_id": merged["author_id"]}
    
    for feature in features:
        human_col = f"{feature}_human"
        llm_col = f"{feature}_llm"
        
        if human_col in merged.columns and llm_col in merged.columns:
            diff_data[feature] = merged[human_col] - merged[llm_col]
        else:
            print(f"Warning: Feature {feature} not found in merged data")
            diff_data[feature] = np.nan
    
    diff_df = pd.DataFrame(diff_data)
    
    # Count how many values are > 0 (human > llm)
    print(f"\nMatching samples: {len(diff_df)}")
    for feature in features:
        if feature in diff_df.columns:
            above_zero = (diff_df[feature] > 0).sum()
            total = diff_df[feature].notna().sum()
            pct = (above_zero / total * 100) if total > 0 else 0
            print(f"  {feature}: {above_zero}/{total} ({pct:.1f}%) values > 0 (human > llm)")
    
    return diff_df


def remove_outliers(values: np.ndarray, method: str = "iqr") -> np.ndarray:
    """
    Remove outliers from values using IQR method.
    
    Args:
        values: Array of values
        method: Method to use ('iqr' for Interquartile Range)
    
    Returns:
        Array with outliers removed
    """
    if len(values) == 0:
        return values
    
    if method == "iqr":
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (values >= lower_bound) & (values <= upper_bound)
        return values[mask]
    else:
        return values


def plot_feature_subplot(ax, diff_df: pd.DataFrame, feature: str, feature_display: str):
    """Create a violin plot subplot for one feature."""
    if feature not in diff_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    values = diff_df[feature].dropna().values
    
    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    # Remove outliers to prevent scale issues
    values_clean = remove_outliers(values, method="iqr")
    n_outliers = len(values) - len(values_clean)
    
    if len(values_clean) == 0:
        ax.text(0.5, 0.5, "No data\n(after outlier removal)", ha="center", va="center", transform=ax.transAxes)
        return
    
    # Create a single violin plot (no x-axis grouping needed since it's one model)
    plot_data = pd.DataFrame({"Difference": values_clean, "Model": ["Model"] * len(values_clean)})
    
    # Create violin plot with blue color
    sns.violinplot(
        data=plot_data,
        x="Model",
        y="Difference",
        ax=ax,
        inner="box",  # Show box plot inside violin (quartiles, median)
        color="#4A90E2",  # Blue color
        density_norm="width",
        width=0.6  # Make it narrower
    )
    
    # Overlay individual points with jitter (blue shades)
    x_jitter = np.random.normal(0, 0.05, size=len(values_clean))
    ax.scatter(x_jitter, values_clean, alpha=0.5, s=8, color='#2E5C8A', zorder=3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Calculate and display statistics (using original values for counts)
    above_zero = (values > 0).sum()
    total = len(values)
    pct = (above_zero / total * 100) if total > 0 else 0
    
    # Get y-axis limits after plotting
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Add text annotation above the violin with better spacing
    # Use separate lines to avoid crowding
    stat_text = f"{above_zero}/{total}\n({pct:.0f}%)"
    if n_outliers > 0:
        stat_text += f"\n({n_outliers} outliers removed)"
    
    ax.text(0, y_max + y_range * 0.12, stat_text, 
           ha="center", va="bottom", fontsize=6, 
           bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.6),
           linespacing=1.2)
    
    # Set title with feature name
    ax.set_title(feature_display, fontsize=9, fontweight="bold")
    ax.set_ylabel("Difference\n(Human - LLM)", fontsize=7)
    ax.set_xlabel("")  # No x-axis label needed
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='y', labelsize=6)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust y-axis limits to accommodate statistics text
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.2)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Human - LLM differences for CE CV features."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all).",
    )
    parser.add_argument(
        "--level",
        default=LEVEL,
        help=f"LLM level to compare (default: {LEVEL}).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=None,
        help="LLM providers to plot (default: all regular providers or DS/CL35/G4OM for --llm-with-history).",
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Use LLM_with_history author-level CV tables instead of regular LLM trajectory tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT / "ce_cv_difference_plots",
        help="Output directory for plots.",
    )
    
    args = parser.parse_args()
    
    providers = (
        args.providers
        if args.providers is not None
        else list(LLM_WITH_HISTORY_PROVIDERS if args.llm_with_history else PROVIDERS)
    )

    # Load human data (all domains combined)
    print("Loading human data...")
    human_df = load_human_data(args.domains, llm_with_history=args.llm_with_history)
    if human_df.empty:
        print("Error: No human data found")
        return
    
    print(f"  Loaded {len(human_df)} human samples from {args.domains}")
    
    # Process each LLM model separately
    for provider in providers:
        print(f"\n=== Processing {provider} {args.level} ===")
        
        # Load LLM data
        llm_df = load_llm_data(provider, args.level, args.domains, llm_with_history=args.llm_with_history)
        if llm_df.empty:
            print(f"  Warning: No data found for {provider} {args.level}")
            continue
        
        print(f"  Loaded {len(llm_df)} samples for {provider} {args.level}")
        
        # Compute differences
        diff_df = compute_differences(human_df, llm_df, CE_CV_FEATURES)
        if diff_df.empty:
            print(f"  Warning: No matching samples for {provider} {args.level}")
            continue
        
        # Create figure with 20 subplots (1 row, 20 columns)
        # Increased height to prevent label overlap
        fig, axes = plt.subplots(1, 20, figsize=(40, 4.5))
        fig.suptitle(
            f"Human vs {provider} ({args.level}) CE CV Feature Differences"
            f"{' [LLM_with_history]' if args.llm_with_history else ''}\n"
            f"Domains: {', '.join(args.domains)} | Total samples: {len(human_df)}",
            fontsize=12,
            fontweight="bold"
        )
        
        # Plot each feature (each subplot will have independent y-axis scale)
        for idx, feature in enumerate(CE_CV_FEATURES):
            feature_display = FEATURE_DISPLAY_NAMES.get(feature, feature)
            plot_feature_subplot(axes[idx], diff_df, feature, feature_display)
        
        plt.tight_layout()
        
        # Save figure
        args.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_llm_with_history" if args.llm_with_history else ""
        output_path = args.output_dir / f"ce_cv_difference_violin_{provider}_{args.level}{suffix}_{'_'.join(args.domains)}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved plot to: {output_path}")
        
        # Also save as PDF
        output_path_pdf = output_path.with_suffix(".pdf")
        plt.savefig(output_path_pdf, bbox_inches="tight")
        print(f"✅ Saved plot to: {output_path_pdf}")
        
        plt.close()


if __name__ == "__main__":
    main()
