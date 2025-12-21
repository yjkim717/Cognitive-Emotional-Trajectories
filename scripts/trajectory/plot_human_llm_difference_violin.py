#!/usr/bin/env python3
"""
Plot violin plots showing Human - LLM differences for TF-IDF trajectory features.

For each LLM model (LV3), compute differences between human and LLM samples,
then create violin plots showing the distribution of differences.
Each plot shows 5 TF-IDF features: mean_distance, std_distance, net_displacement, path_length, tortuosity.
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
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVEL = "LV3"

# TF-IDF features to plot (5 features)
TFIDF_FEATURES = [
    "tfidf_mean_distance",
    "tfidf_std_distance",
    "tfidf_net_displacement",
    "tfidf_path_length",
    "tfidf_tortuosity",
]

# Feature display names
FEATURE_DISPLAY_NAMES = {
    "tfidf_mean_distance": "Mean Distance",
    "tfidf_std_distance": "Std Distance",
    "tfidf_net_displacement": "Net Displacement",
    "tfidf_path_length": "Path Length",
    "tfidf_tortuosity": "Tortuosity",
}


def load_human_data(domains: List[str]) -> pd.DataFrame:
    """Load and combine human data from all domains."""
    frames = []
    for domain in domains:
        csv_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["domain"] = domain
            frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_llm_data(provider: str, level: str, domains: List[str]) -> pd.DataFrame:
    """Load and combine LLM data from all domains for a specific provider/level."""
    frames = []
    for domain in domains:
        csv_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
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
    # Merge on field and author_id to match samples
    merge_keys = ["field", "author_id"]
    
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


def plot_feature_subplot(ax, all_model_diffs: Dict[str, pd.DataFrame], feature: str, feature_display: str):
    """Create a violin plot subplot for one feature showing all models."""
    # Prepare data for plotting - combine all models for this feature
    plot_data = []
    for model_name, diff_df in sorted(all_model_diffs.items()):
        if feature in diff_df.columns:
            values = diff_df[feature].dropna()
            for val in values:
                plot_data.append({
                    "Model": model_name,
                    "Difference": val
                })
    
    if not plot_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create violin plot with box plot inside to show distribution clearly
    # Use gray color for all violins to keep them uniform
    sns.violinplot(
        data=plot_df,
        x="Model",
        y="Difference",
        ax=ax,
        inner="box",  # Show box plot inside violin (quartiles, median)
        palette=["#CCCCCC"] * len(all_model_diffs),  # All gray
        density_norm="width"
    )
    
    # Overlay individual points with jitter, using different colors for each model
    model_colors = {
        "DS": "#66c2a5",    # Teal
        "G4B": "#fc8d62",   # Orange
        "G12B": "#8da0cb",  # Blue
        "LMK": "#e78ac3"    # Pink
    }
    
    for i, model_name in enumerate(sorted(all_model_diffs.keys())):
        if feature in all_model_diffs[model_name].columns:
            values = all_model_diffs[model_name][feature].dropna().values
            # Add jitter to x-axis to avoid overlap
            x_jitter = np.random.normal(i, 0.05, size=len(values))
            # Use model-specific color for points without edge (or with matching edge)
            point_color = model_colors.get(model_name, '#888888')
            ax.scatter(x_jitter, values, alpha=0.8, s=9, color=point_color, zorder=3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero (Human=LLM)')
    
    # Calculate and display statistics for each model
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    for i, (model_name, diff_df) in enumerate(sorted(all_model_diffs.items())):
        if feature in diff_df.columns:
            values = diff_df[feature].dropna()
            if len(values) > 0:
                above_zero = (values > 0).sum()
                total = len(values)
                pct = (above_zero / total * 100)
                
                # Add text annotation above the violin
                ax.text(i, y_max + y_range * 0.08, f"{above_zero}/{total}\n({pct:.0f}%)", 
                       ha="center", va="bottom", fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6))
    
    # Set title with feature name
    ax.set_title(feature_display, fontsize=12, fontweight="bold")
    ax.set_xlabel("LLM Model (LV3)", fontsize=10)
    ax.set_ylabel("Difference (Human - LLM)", fontsize=9)
    
    # Ensure x-axis labels are visible
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.tick_params(axis='y', labelsize=8)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=7)
    
    # Adjust y-axis limits to accommodate statistics text
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.15)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Human - LLM differences for TF-IDF trajectory features."
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
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT / "difference_plots",
        help="Output directory for plots.",
    )
    
    args = parser.parse_args()
    
    # Load human data (all domains combined)
    print("Loading human data...")
    human_df = load_human_data(args.domains)
    if human_df.empty:
        print("Error: No human data found")
        return
    
    print(f"  Loaded {len(human_df)} human samples from {args.domains}")
    
    # Load LLM data for each model (LV3)
    print(f"\nLoading LLM data (level: {args.level})...")
    model_diffs = {}
    
    for provider in PROVIDERS:
        llm_df = load_llm_data(provider, args.level, args.domains)
        if llm_df.empty:
            print(f"  Warning: No data found for {provider} {args.level}")
            continue
        
        print(f"  Loaded {len(llm_df)} samples for {provider} {args.level}")
        
        # Compute differences
        diff_df = compute_differences(human_df, llm_df, TFIDF_FEATURES)
        if not diff_df.empty:
            model_diffs[provider] = diff_df
    
    if not model_diffs:
        print("Error: No LLM data found for comparison")
        return
    
    # Create figure with 5 subplots (one for each feature)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f"Human vs LLM (LV3) TF-IDF Trajectory Feature Differences\n"
        f"Domains: {', '.join(args.domains)} | Total samples: {len(human_df)}",
        fontsize=14,
        fontweight="bold"
    )
    
    # Plot each feature (each subplot will have independent y-axis scale)
    for idx, feature in enumerate(TFIDF_FEATURES):
        feature_display = FEATURE_DISPLAY_NAMES.get(feature, feature)
        plot_feature_subplot(axes[idx], model_diffs, feature, feature_display)
    
    plt.tight_layout()
    
    # Save figure
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"human_llm_difference_violin_{args.level}_{'_'.join(args.domains)}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved plot to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = output_path.with_suffix(".pdf")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"✅ Saved plot to: {output_path_pdf}")


if __name__ == "__main__":
    main()

