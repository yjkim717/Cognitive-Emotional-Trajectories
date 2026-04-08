#!/usr/bin/env python3
"""
Plot violin plots showing Human - LLM differences for SBERT and TFIDF total drift.

For each LLM model, compute total drift per author (sum of all year-to-year drifts),
then compute differences between human and LLM samples,
and create violin plots showing the distribution of differences.

Creates one figure with two subplots:
- Top subplot: SBERT total drift differences (4 models)
- Bottom subplot: TFIDF total drift differences (4 models)
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
# Model mapping: provider code -> full name
PROVIDERS = ("DS", "CL35", "G4OM")  # Only three models
PROVIDER_NAMES = {
    "DS": "DeepSeek-V1",
    "CL35": "Claude 3.5 Haiku",
    "G4OM": "GPT-4o mini"
}
LEVEL = "LV3"
CONDITION_FILL_COLORS = {
    "Instance-wise": "#87CEEB",
    "With History": "#F4A261",
}
CONDITION_DOT_COLORS = {
    "Instance-wise": "#4169E1",
    "With History": "#D55E00",
}


def load_drift_data(
    domain: str,
    label: str,
    rep_space: str,
    provider: str | None = None,
    level: str | None = None,
    llm_with_history: bool = False,
) -> pd.DataFrame:
    """Load drift data for a specific split."""
    if label == "human":
        drift_path = DATA_ROOT / "human" / domain / f"{rep_space}_drift.csv"
    elif llm_with_history:
        drift_path = DATA_ROOT / "LLM_with_history" / provider / level / domain / f"{rep_space}_drift.csv"
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
        df: DataFrame with drift data (columns: author_id, field, domain, year_from, year_to, drift)
    
    Returns:
        DataFrame with one row per author (author_id, field, domain, total_drift)
    """
    # Group by author and sum all drifts
    author_cols = ["author_id", "field", "domain"]
    total_drift = df.groupby(author_cols, dropna=False)["drift"].sum().reset_index()
    total_drift = total_drift.rename(columns={"drift": "total_drift"})
    
    return total_drift


def compute_differences(
    human_df: pd.DataFrame, 
    llm_df: pd.DataFrame,
    rep_space: str
) -> pd.DataFrame:
    """
    Compute Human - LLM differences for total drift for matching (field, author_id, domain) pairs.
    
    Returns a DataFrame with differences for each model.
    """
    # Compute total drift per author for human
    human_total = compute_total_drift_per_author(human_df)
    
    # Compute total drift per author for LLM
    llm_total = compute_total_drift_per_author(llm_df)
    
    # Merge on field, author_id, domain to match samples
    merge_keys = ["field", "author_id", "domain"]
    
    # Merge to align samples
    merged = pd.merge(
        human_total,
        llm_total,
        on=merge_keys,
        how="inner",
        suffixes=("_human", "_llm")
    )
    
    # Compute difference: Human - LLM
    merged["difference"] = merged["total_drift_human"] - merged["total_drift_llm"]
    
    return merged


def load_combined_drift_data(domains: List[str], rep_space: str, label: str) -> pd.DataFrame:
    """Load and combine drift data from all domains."""
    frames = []
    for domain in domains:
        if label == "human":
            df = load_drift_data(domain, label, rep_space)
        else:
            # For LLM, we'll load separately per provider/level
            continue
        
        if not df.empty:
            df["domain"] = domain
            frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def plot_drift_difference_violin(
    ax,
    all_differences: pd.DataFrame,
    providers: List[str],
    rep_space: str,
    title: str,
    show_ylabel: bool = True
):
    """
    Create violin plots for drift differences across 3 models.
    
    Args:
        ax: Matplotlib axis
        all_differences: DataFrame with columns: model, difference
        rep_space: Representation space name (for title)
        title: Plot title
    """
    if all_differences.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    # Remove outliers using IQR method
    def remove_outliers(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        if IQR == 0:
            return values
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (values >= lower_bound) & (values <= upper_bound)
        return values[mask]
    
    # Prepare data for plotting (remove outliers per model for violin plot)
    plot_data_list = []
    model_data_dict = {}  # Store original data (with outliers) for scatter points
    for model_code in providers:
        model_name = PROVIDER_NAMES[model_code]
        model_diff = all_differences[all_differences["model"] == model_code]["difference"].dropna().values
        if len(model_diff) > 0:
            # Store original data for scatter points (use model_code as key)
            model_data_dict[model_code] = model_diff
            # Clean data for violin plot (use full name for display)
            model_diff_clean = remove_outliers(model_diff)
            plot_data_list.append(pd.DataFrame({
                "Model": [model_name] * len(model_diff_clean),
                "Difference": model_diff_clean
            }))
    
    if not plot_data_list:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    plot_data = pd.concat(plot_data_list, ignore_index=True)
    
    # Use full model names in order
    model_order = [PROVIDER_NAMES[m] for m in PROVIDERS]
    
    # Create violin plot with light blue color for all violins (semi-transparent)
    sns.violinplot(
        data=plot_data,
        x="Model",
        y="Difference",
        hue="Model",
        ax=ax,
        inner="box",
        palette=["#87CEEB"] * len(providers),  # Light blue (SkyBlue) for all
        order=model_order,
        width=0.7,
        legend=False,
        density_norm="width",
        alpha=0.6  # Semi-transparent
    )
    
    # Use darker blue for scatter points (more visible)
    scatter_color = "#4169E1"  # Royal blue (darker than light blue for better visibility)
    
    # Overlay individual points with jitter using original data (with outliers removed per model)
    for i, model_code in enumerate(providers):
        if model_code in model_data_dict:
            values = model_data_dict[model_code]
            # Remove outliers for scatter points too
            values_clean = remove_outliers(values)
            # Add jitter to x-axis to avoid overlap
            x_jitter = np.random.normal(i, 0.05, size=len(values_clean))
            ax.scatter(x_jitter, values_clean, alpha=0.7, s=9, color=scatter_color, zorder=3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero (Human=LLM)', zorder=0)
    
    # Get initial y-axis limits after plotting
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Extend y-axis upward to make room for statistics labels (increase top by 15%)
    ax.set_ylim(y_min, y_max + y_range * 0.15)
    
    # Get updated y_max after extending
    _, y_max_extended = ax.get_ylim()
    
    # Calculate and display statistics for each model (using original data)
    for idx, model_code in enumerate(providers):
        if model_code in model_data_dict:
            model_diff = model_data_dict[model_code]
            above_zero = (model_diff > 0).sum()
            total = len(model_diff)
            pct = (above_zero / total * 100) if total > 0 else 0
            
            # Add text annotation at the top of the plot box, right below the extended top edge
            stat_text = f"{above_zero}/{total}\n({pct:.0f}%)"
            ax.text(idx, y_max_extended - y_range * 0.02, stat_text,
                   ha="center", va="top", fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9))
    
    # Set labels and title (bold labels)
    ax.set_title(title, fontsize=16, fontweight="bold")  # Increased title font size
    if show_ylabel:
        ax.set_ylabel("Difference (Human - LLM)", fontsize=12, fontweight="bold")  # Simplified ylabel, increased font size
    else:
        ax.set_ylabel("")  # No ylabel for right subplot
    # Removed xlabel as requested - explicitly set to empty string
    ax.set_xlabel("")  # Explicitly remove x-axis label
    ax.tick_params(axis='x', labelsize=14)  # Increased x-axis tick label size
    ax.tick_params(axis='y', labelsize=9)
    
    # Set x-axis labels to bold and larger
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add legend only for the zero line (y=0 red dashed line) - larger font size
    ax.legend(loc='best', fontsize=12)
    
    # No need to extend y-axis limits since text is now inside the plot area


def plot_drift_difference_violin_compare(
    ax,
    all_differences: pd.DataFrame,
    providers: List[str],
    title: str,
    show_ylabel: bool = True,
):
    """Create grouped violin plots for instance-wise vs with-history comparisons."""
    if all_differences.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    def remove_outliers(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr == 0:
            return values
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return values[(values >= lower) & (values <= upper)]

    order = []
    plot_frames = []
    raw_values = {}
    for provider in providers:
        provider_name = PROVIDER_NAMES[provider]
        for condition in ["Instance-wise", "With History"]:
            label = f"{provider_name}\n{condition}"
            order.append(label)
            vals = all_differences[
                (all_differences["model"] == provider) & (all_differences["condition"] == condition)
            ]["difference"].dropna().values
            if len(vals) == 0:
                continue
            raw_values[label] = vals
            vals_clean = remove_outliers(vals)
            plot_frames.append(pd.DataFrame({"Condition": [label] * len(vals_clean), "Difference": vals_clean}))

    if not plot_frames:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    plot_data = pd.concat(plot_frames, ignore_index=True)
    palette = []
    for label in order:
        condition = "With History" if label.endswith("With History") else "Instance-wise"
        palette.append(CONDITION_FILL_COLORS[condition])

    sns.violinplot(
        data=plot_data,
        x="Condition",
        y="Difference",
        order=order,
        palette=palette,
        ax=ax,
        inner="box",
        width=0.55,
        density_norm="width",
    )

    for idx, label in enumerate(order):
        vals = raw_values.get(label)
        if vals is None:
            continue
        vals_clean = remove_outliers(vals)
        condition = "With History" if label.endswith("With History") else "Instance-wise"
        x_jitter = np.random.normal(idx, 0.05, size=len(vals_clean))
        ax.scatter(x_jitter, vals_clean, alpha=0.7, s=9, color=CONDITION_DOT_COLORS[condition], zorder=3)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, zorder=0)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.18)
    _, y_max_extended = ax.get_ylim()

    for idx, label in enumerate(order):
        vals = raw_values.get(label)
        if vals is None:
            continue
        above_zero = (vals > 0).sum()
        total = len(vals)
        pct = (above_zero / total * 100) if total > 0 else 0
        ax.text(
            idx,
            y_max_extended - y_range * 0.02,
            f"{above_zero}/{total}\n({pct:.0f}%)",
            ha="center",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.9),
        )

    ax.set_title(title, fontsize=18, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Difference (Human - LLM)", fontsize=14, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    # Tighten horizontal margins so paired violins sit closer together.
    ax.margins(x=0.02)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    legend_handles = [
        Patch(facecolor=CONDITION_FILL_COLORS[c], edgecolor="black", label=c)
        for c in ["Instance-wise", "With History"]
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=12)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Human - LLM differences for SBERT and TFIDF total drift."
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
        default=list(PROVIDERS),
        help="LLM providers to plot (default: DS CL35 G4OM).",
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Use LLM_with_history drift CSVs instead of regular LLM drift CSVs.",
    )
    parser.add_argument(
        "--compare-with-history",
        action="store_true",
        help="Plot instance-wise and LLM_with_history together for the same providers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT / "drift_difference_plots",
        help="Output directory for plots.",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Plot SBERT and TFIDF Total Drift Differences")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"Providers: {args.providers}")
    if args.compare_with_history:
        print("Mode: Comparison (Instance-wise vs LLM_with_history)")
    else:
        print(f"Mode: {'LLM_with_history' if args.llm_with_history else 'LLM'}")
    print()
    
    # Load human drift data for both SBERT and TFIDF
    print("Loading human drift data...")
    human_sbert = load_combined_drift_data(args.domains, "sbert", "human")
    human_tfidf = load_combined_drift_data(args.domains, "tfidf", "human")
    
    if human_sbert.empty or human_tfidf.empty:
        print("Error: No human drift data found")
        return
    
    print(f"  SBERT: {len(human_sbert)} drift measurements")
    print(f"  TFIDF: {len(human_tfidf)} drift measurements")
    
    # Collect differences for all models
    all_sbert_differences = []
    all_tfidf_differences = []

    # Process each LLM model
    for provider_code in args.providers:
        provider_name = PROVIDER_NAMES[provider_code]
        print(f"\nProcessing {provider_name} ({provider_code})...")
        conditions = [("With History", True), ("Instance-wise", False)] if args.compare_with_history else [
            ("With History" if args.llm_with_history else "Instance-wise", args.llm_with_history)
        ]

        for condition_name, is_history in conditions:
            llm_sbert_frames = []
            llm_tfidf_frames = []
            for domain in args.domains:
                sbert_df = load_drift_data(domain, "llm", "sbert", provider_code, args.level, llm_with_history=is_history)
                tfidf_df = load_drift_data(domain, "llm", "tfidf", provider_code, args.level, llm_with_history=is_history)
                if not sbert_df.empty:
                    sbert_df["domain"] = domain
                    llm_sbert_frames.append(sbert_df)
                if not tfidf_df.empty:
                    tfidf_df["domain"] = domain
                    llm_tfidf_frames.append(tfidf_df)

            if not llm_sbert_frames or not llm_tfidf_frames:
                print(f"  Warning: Missing {condition_name} data for {provider_name} ({provider_code})")
                continue

            llm_sbert = pd.concat(llm_sbert_frames, ignore_index=True)
            llm_tfidf = pd.concat(llm_tfidf_frames, ignore_index=True)
            sbert_diff = compute_differences(human_sbert, llm_sbert, "sbert")
            tfidf_diff = compute_differences(human_tfidf, llm_tfidf, "tfidf")
            sbert_diff["model"] = provider_code
            tfidf_diff["model"] = provider_code
            sbert_diff["condition"] = condition_name
            tfidf_diff["condition"] = condition_name
            all_sbert_differences.append(sbert_diff)
            all_tfidf_differences.append(tfidf_diff)
            print(f"  {condition_name} SBERT: {len(sbert_diff)} matched authors")
            print(f"  {condition_name} TFIDF: {len(tfidf_diff)} matched authors")
    
    if not all_sbert_differences or not all_tfidf_differences:
        print("\nError: No differences computed")
        return
    
    # Combine all differences
    combined_sbert = pd.concat(all_sbert_differences, ignore_index=True)
    combined_tfidf = pd.concat(all_tfidf_differences, ignore_index=True)
    
    # Create figure with 2 subplots (horizontally stacked)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    # Removed fig.suptitle as requested
    
    # Plot SBERT differences (left subplot) - show ylabel
    if args.compare_with_history:
        plot_drift_difference_violin_compare(
            ax1,
            combined_sbert,
            args.providers,
            "SBERT Total Drift Differences",
            show_ylabel=True,
        )
        plot_drift_difference_violin_compare(
            ax2,
            combined_tfidf,
            args.providers,
            "TFIDF Total Drift Differences",
            show_ylabel=False,
        )
    else:
        plot_drift_difference_violin(
            ax1,
            combined_sbert,
            args.providers,
            "sbert",
            f"SBERT Total Drift Differences{' [LLM_with_history]' if args.llm_with_history else ''}",
            show_ylabel=True
        )
        plot_drift_difference_violin(
            ax2,
            combined_tfidf,
            args.providers,
            "tfidf",
            f"TFIDF Total Drift Differences{' [LLM_with_history]' if args.llm_with_history else ''}",
            show_ylabel=False
        )
    
    plt.tight_layout()
    
    # Save figure
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domains_str = "_".join(args.domains)
    providers_str = "_".join(args.providers)
    if args.compare_with_history:
        suffix = "_compare_with_history"
    else:
        suffix = "_llm_with_history" if args.llm_with_history else ""
    output_path = args.output_dir / f"drift_difference_violin_{providers_str}{suffix}_{domains_str}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved plot to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = output_path.with_suffix(".pdf")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"✅ Saved plot to: {output_path_pdf}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
