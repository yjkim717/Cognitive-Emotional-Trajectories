#!/usr/bin/env python3
"""
Plot violin plots showing Human - LLM differences for CE CV features,
organized by cognitive/emotional/stylistic feature groups.

Generates three separate plots, one for each feature group, for specified models.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("G4B", "CL35", "G4OM")  # Supported models
LLM_WITH_HISTORY_PROVIDERS = ("DS", "CL35", "G4OM")
# Model mapping: provider code -> full name
PROVIDER_NAMES = {
    "G4B": "Gemma-2B",
    "CL35": "Claude 3.5 Haiku",
    "G4OM": "GPT-4o mini"
}
LEVEL = "LV3"

# Feature group organization
COGNITIVE_FEATURES = [
    "Openness_cv",
    "Conscientiousness_cv",
    "Extraversion_cv",
    "Agreeableness_cv",
    "Neuroticism_cv",
]

EMOTIONAL_FEATURES = [
    "polarity_cv",
    "subjectivity_cv",
    "vader_compound_cv",
    "vader_pos_cv",
    "vader_neu_cv",
    "vader_neg_cv",
]

STYLISTIC_FEATURES = [
    "word_diversity_cv",
    "flesch_reading_ease_cv",
    "gunning_fog_cv",
    "average_word_length_cv",
    "num_words_cv",
    "avg_sentence_length_cv",
    "verb_ratio_cv",
    "function_word_ratio_cv",
    "content_word_ratio_cv",
]

# Split stylistic features into two groups for better visualization
STYLISTIC_FEATURES_GROUP1 = [
    "word_diversity_cv",
    "flesch_reading_ease_cv",
    "gunning_fog_cv",
    "average_word_length_cv",
]

STYLISTIC_FEATURES_GROUP2 = [
    "num_words_cv",
    "avg_sentence_length_cv",
    "verb_ratio_cv",
    "function_word_ratio_cv",
    "content_word_ratio_cv",
]

# Feature display names
FEATURE_DISPLAY_NAMES = {
    "Openness_cv": "Openness",
    "Conscientiousness_cv": "Conscientiousness",
    "Extraversion_cv": "Extraversion",
    "Agreeableness_cv": "Agreeableness",
    "Neuroticism_cv": "Neuroticism",
    "polarity_cv": "Polarity",
    "subjectivity_cv": "Subjectivity",
    "vader_compound_cv": "VADER Compound",
    "vader_pos_cv": "VADER Positive",
    "vader_neu_cv": "VADER Neutral",
    "vader_neg_cv": "VADER Negative",
    "word_diversity_cv": "Word Diversity",
    "flesch_reading_ease_cv": "Flesch Reading Ease",
    "gunning_fog_cv": "Gunning Fog",
    "average_word_length_cv": "Average Word Length",
    "num_words_cv": "Number of Words",
    "avg_sentence_length_cv": "Average Sentence Length",
    "verb_ratio_cv": "Verb Ratio",
    "function_word_ratio_cv": "Function Word Ratio",
    "content_word_ratio_cv": "Content Word Ratio",
}

# Feature group information
FEATURE_GROUP_INFO = {
    "Cognitive": {
        "features": COGNITIVE_FEATURES,
        "title": "Cognitive Features (Personality Traits)",
        "description": "Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism"
    },
    "Emotional": {
        "features": EMOTIONAL_FEATURES,
        "title": "Emotional Features (Sentiment)",
        "description": "Polarity, Subjectivity, VADER scores"
    },
    "Stylistic": {
        "features": STYLISTIC_FEATURES,
        "title": "Stylistic Features (Linguistic Patterns)",
        "description": "Lexical diversity, readability, word/sentence patterns, POS ratios"
    },
}

CONDITION_ORDER = ["Instance-wise", "With History"]
CONDITION_FILL_COLORS = {
    "Instance-wise": "#87CEEB",
    "With History": "#F4A261",
}
CONDITION_DOT_COLORS = {
    "Instance-wise": "#4169E1",
    "With History": "#D55E00",
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
    Compute Human - LLM differences for matching (field, author_id, domain) pairs.
    
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


def plot_feature_subplot(ax, diff_df: pd.DataFrame, feature: str, feature_display: str, show_ylabel: bool = True):
    """Create a violin plot subplot for one feature.
    
    Args:
        ax: Matplotlib axis
        diff_df: DataFrame with differences
        feature: Feature name
        feature_display: Display name for feature
        show_ylabel: Whether to show y-axis label (default: True)
    """
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
    
    # Create violin plot with light blue color (same as drift plot)
    sns.violinplot(
        data=plot_data,
        x="Model",
        y="Difference",
        ax=ax,
        inner="box",  # Show box plot inside violin (quartiles, median)
        color="#87CEEB",  # Light blue (SkyBlue) - same as drift plot
        density_norm="width",
        width=0.6,  # Make it narrower
        alpha=0.6  # Semi-transparent
    )
    
    # Overlay individual points with royal blue (same as drift plot)
    x_jitter = np.random.normal(0, 0.05, size=len(values_clean))
    ax.scatter(x_jitter, values_clean, alpha=0.7, s=9, color='#4169E1', zorder=3)  # Royal blue
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Calculate and display statistics (using original values for counts)
    above_zero = (values > 0).sum()
    total = len(values)
    pct = (above_zero / total * 100) if total > 0 else 0
    
    # Get y-axis limits after plotting
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Extend y-axis upward to make room for statistics labels (increase top by 15%)
    ax.set_ylim(y_min, y_max + y_range * 0.15)
    
    # Get updated y_max after extending
    _, y_max_extended = ax.get_ylim()
    
    # Add text annotation at the top of the plot box, right below the extended top edge
    stat_text = f"{above_zero}/{total} ({pct:.0f}%)"
    ax.text(0, y_max_extended - y_range * 0.02, stat_text, 
           ha="center", va="top", fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9))
    
    # Remove title - feature name will be shown as xlabel at bottom
    # Set feature name as xlabel at bottom, bold
    ax.set_xlabel(feature_display, fontsize=10, fontweight="bold")
    
    if show_ylabel:
        ax.set_ylabel("Difference (Human - LLM)", fontsize=10, fontweight="bold")
    else:
        ax.set_ylabel("")  # No y-axis label
    
    # Remove x-axis ticks but keep the label (feature name)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_feature_subplot_compare(
    ax,
    diff_map: Dict[str, pd.DataFrame],
    feature: str,
    feature_display: str,
    show_ylabel: bool = True,
):
    """Create side-by-side violin plots for instance-wise vs with-history."""
    plot_frames = []
    original_values = {}

    for condition in CONDITION_ORDER:
        diff_df = diff_map.get(condition)
        if diff_df is None or feature not in diff_df.columns:
            continue

        values = diff_df[feature].dropna().values
        if len(values) == 0:
            continue

        values_clean = remove_outliers(values, method="iqr")
        if len(values_clean) == 0:
            continue

        original_values[condition] = values
        plot_frames.append(
            pd.DataFrame(
                {
                    "Condition": [condition] * len(values_clean),
                    "Difference": values_clean,
                }
            )
        )

    if not plot_frames:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    plot_data = pd.concat(plot_frames, ignore_index=True)

    sns.violinplot(
        data=plot_data,
        x="Condition",
        y="Difference",
        ax=ax,
        inner="box",
        order=CONDITION_ORDER,
        palette=CONDITION_FILL_COLORS,
        density_norm="width",
        width=0.7,
        alpha=0.6,
    )

    for idx, condition in enumerate(CONDITION_ORDER):
        values = original_values.get(condition)
        if values is None:
            continue
        values_clean = remove_outliers(values, method="iqr")
        x_jitter = np.random.normal(idx, 0.05, size=len(values_clean))
        ax.scatter(
            x_jitter,
            values_clean,
            alpha=0.7,
            s=9,
            color=CONDITION_DOT_COLORS[condition],
            zorder=3,
        )

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.18)
    _, y_max_extended = ax.get_ylim()

    for idx, condition in enumerate(CONDITION_ORDER):
        values = original_values.get(condition)
        if values is None:
            continue
        above_zero = (values > 0).sum()
        total = len(values)
        pct = (above_zero / total * 100) if total > 0 else 0
        stat_text = f"{above_zero}/{total}\n({pct:.0f}%)"
        ax.text(
            idx,
            y_max_extended - y_range * 0.02,
            stat_text,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.9),
        )

    ax.set_xlabel(feature_display, fontsize=10, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Difference (Human - LLM)", fontsize=10, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_feature_group(provider: str, level: str, domains: List[str], group_name_key: str,
               group_info: Dict, human_df: pd.DataFrame, llm_df: pd.DataFrame,
               output_dir: Path, llm_with_history: bool = False):
    """Plot one feature group with all its features.
    
    For Stylistic features, split into two plots (4 features + 5 features).
    """
    title = group_info["title"]
    description = group_info["description"]
    
    # For Stylistic features, split into two groups
    if group_name_key == "Stylistic":
        feature_groups = [
            ("Group 1", STYLISTIC_FEATURES_GROUP1),
            ("Group 2", STYLISTIC_FEATURES_GROUP2),
        ]
    else:
        # For other feature groups, use all features as one group
        feature_groups = [("All", group_info["features"])]
    
    provider_name = PROVIDER_NAMES.get(provider, provider)
    
    # Plot each group separately for Stylistic features
    for group_name, features in feature_groups:
        print(f"\n=== Plotting {group_name_key} Features {group_name} ===")
        print(f"Features: {features}")
        
        # Compute differences for this group's features
        diff_df = compute_differences(human_df, llm_df, features)
        if diff_df.empty:
            print(f"  Warning: No matching samples for {provider} {level}")
            continue
        
        # Create figure with subplots (1 row, N columns where N = number of features)
        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=(n_features * 2.5, 4.5))
        
        # Handle case when there's only one feature (axes is not an array)
        if n_features == 1:
            axes = [axes]
        
        # Remove main title as requested
        
        # Plot each feature
        # Only show ylabel on the first subplot (leftmost)
        for idx, feature in enumerate(features):
            feature_display = FEATURE_DISPLAY_NAMES.get(feature, feature)
            show_ylabel = (idx == 0)  # Only first subplot shows ylabel
            plot_feature_subplot(axes[idx], diff_df, feature, feature_display, show_ylabel=show_ylabel)
        
        plt.tight_layout()
        
        # Save figure
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_llm_with_history" if llm_with_history else ""
        if group_name_key == "Stylistic":
            output_path = output_dir / f"ce_cv_difference_violin_{provider}_{level}{suffix}_{group_name_key.lower()}_{group_name.lower().replace(' ', '_')}_{'_'.join(domains)}.png"
        else:
            output_path = output_dir / f"ce_cv_difference_violin_{provider}_{level}{suffix}_{group_name_key.lower()}_{'_'.join(domains)}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved plot to: {output_path}")
        
        plt.close()


def plot_feature_group_compare(
    provider: str,
    level: str,
    domains: List[str],
    group_name_key: str,
    group_info: Dict,
    human_df: pd.DataFrame,
    llm_original_df: pd.DataFrame,
    llm_history_df: pd.DataFrame,
    output_dir: Path,
):
    """Plot one feature group with instance-wise and with-history side by side."""
    if group_name_key == "Stylistic":
        feature_groups = [
            ("Group 1", STYLISTIC_FEATURES_GROUP1),
            ("Group 2", STYLISTIC_FEATURES_GROUP2),
        ]
    else:
        feature_groups = [("All", group_info["features"])]

    for group_name, features in feature_groups:
        print(f"\n=== Comparing {group_name_key} Features {group_name} ===")
        diff_original = compute_differences(human_df, llm_original_df, features)
        diff_history = compute_differences(human_df, llm_history_df, features)

        if diff_original.empty and diff_history.empty:
            print(f"  Warning: No matching samples for comparison: {provider} {level}")
            continue

        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=(n_features * 3.0, 4.8))
        if n_features == 1:
            axes = [axes]

        for idx, feature in enumerate(features):
            feature_display = FEATURE_DISPLAY_NAMES.get(feature, feature)
            plot_feature_subplot_compare(
                axes[idx],
                {
                    "Instance-wise": diff_original,
                    "With History": diff_history,
                },
                feature,
                feature_display,
                show_ylabel=(idx == 0),
            )

        legend_handles = [
            Patch(facecolor=CONDITION_FILL_COLORS[c], edgecolor="black", label=c)
            for c in CONDITION_ORDER
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, fontsize=10)
        plt.tight_layout(rect=(0, 0, 1, 0.92))

        output_dir.mkdir(parents=True, exist_ok=True)
        if group_name_key == "Stylistic":
            output_path = output_dir / (
                f"ce_cv_difference_compare_{provider}_{level}_"
                f"{group_name_key.lower()}_{group_name.lower().replace(' ', '_')}_{'_'.join(domains)}.png"
            )
        else:
            output_path = output_dir / (
                f"ce_cv_difference_compare_{provider}_{level}_{group_name_key.lower()}_{'_'.join(domains)}.png"
            )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Saved comparison plot to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Human - LLM differences for CE CV features by feature group."
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
        "--models",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help=f"Models to process (default: all). Options: {', '.join(PROVIDERS)}.",
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Use LLM_with_history author-level CV tables instead of regular LLM trajectory tables.",
    )
    parser.add_argument(
        "--compare-with-history",
        action="store_true",
        help="Plot instance-wise and LLM_with_history together in the same figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT / "ce_cv_difference_plots",
        help="Output directory for plots.",
    )
    
    args = parser.parse_args()
    
    if args.llm_with_history:
        if args.models == list(PROVIDERS):
            args.models = list(LLM_WITH_HISTORY_PROVIDERS)
        invalid = sorted(set(args.models) - set(LLM_WITH_HISTORY_PROVIDERS))
        if invalid:
            raise ValueError(f"Unsupported LLM_with_history models: {invalid}")
    if args.compare_with_history:
        invalid = sorted(set(args.models) - set(LLM_WITH_HISTORY_PROVIDERS))
        if invalid:
            raise ValueError(f"Comparison mode only supports LLM_with_history models: {invalid}")

    # Load human data (all domains combined)
    print("Loading human data...")
    human_df = load_human_data(args.domains, llm_with_history=args.llm_with_history or args.compare_with_history)
    if human_df.empty:
        print("Error: No human data found")
        return
    
    print(f"  Loaded {len(human_df)} human samples from {args.domains}")
    
    # Process each model
    for provider in args.models:
        print(f"\n=== Processing {provider} {args.level} ===")
        if args.compare_with_history:
            llm_original_df = load_llm_data(provider, args.level, args.domains, llm_with_history=False)
            llm_history_df = load_llm_data(provider, args.level, args.domains, llm_with_history=True)
            if llm_original_df.empty or llm_history_df.empty:
                print(f"  Warning: Missing comparison data for {provider} {args.level}")
                continue
            print(f"  Loaded {len(llm_original_df)} instance-wise samples for {provider} {args.level}")
            print(f"  Loaded {len(llm_history_df)} with-history samples for {provider} {args.level}")
            for group_name_key, group_info in FEATURE_GROUP_INFO.items():
                plot_feature_group_compare(
                    provider,
                    args.level,
                    args.domains,
                    group_name_key,
                    group_info,
                    human_df,
                    llm_original_df,
                    llm_history_df,
                    args.output_dir,
                )
        else:
            llm_df = load_llm_data(provider, args.level, args.domains, llm_with_history=args.llm_with_history)
            if llm_df.empty:
                print(f"  Warning: No data found for {provider} {args.level}")
                continue

            print(f"  Loaded {len(llm_df)} samples for {provider} {args.level}")

            for group_name_key, group_info in FEATURE_GROUP_INFO.items():
                plot_feature_group(provider, args.level, args.domains, group_name_key, group_info,
                           human_df, llm_df, args.output_dir, llm_with_history=args.llm_with_history)
    
    print("\n✅ All plots generated!")


if __name__ == "__main__":
    main()
