#!/usr/bin/env python3
"""
Analyze per-feature binomial test for the unified 75 trajectory features (RQ1, LV3).

Input: micro_results/binomial/binomial_test_trajectory_features_lv3_detailed.csv
- CE-VAR       (60 features)
- CE-GEO       (5 features)
- TF-IDF-GEO   (5 features)
- SBERT-GEO    (5 features)

Output:
- CSV with per-feature stats
- Bar plot of 75 features' human win rates, colored by feature_group
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

COLORS = {
    "CE-VAR": "#2E86AB",      # blue
    "CE-GEO": "#A23B72",      # purple
    "TF-IDF-GEO": "#F18F01",  # orange
    "SBERT-GEO": "#C73E1D",   # red
}


def compute_per_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binomial stats per feature (Human > LLM)."""
    rows: List[Dict] = []

    for feature, sub in df.groupby("feature"):
        feature_type = sub["feature_type"].iloc[0]
        n = len(sub)
        k = int(sub["human_wins"].sum())
        p_obs = k / n if n > 0 else np.nan

        # one-sided test: H1: p > 0.5 (Human > LLM)
        test = binomtest(k, n, p=0.5, alternative="greater")
        pval = test.pvalue
        significant = pval < 0.05
        if pval < 0.001:
            star = "***"
        elif pval < 0.01:
            star = "**"
        elif pval < 0.05:
            star = "*"
        else:
            star = ""

        rows.append(
            {
                "feature": feature,
                "feature_group": feature_type,
                "n_comparisons": n,
                "human_wins": k,
                "human_win_rate": p_obs,
                "pvalue": pval,
                "significant": significant,
                "significant_star": star,
            }
        )

    stats = pd.DataFrame(rows)
    # sort by feature_group then win rate descending
    group_order = {"CE-VAR": 0, "CE-GEO": 1, "TF-IDF-GEO": 2, "SBERT-GEO": 3}
    stats["group_order"] = stats["feature_group"].map(group_order)
    stats = stats.sort_values(["group_order", "human_win_rate"], ascending=[True, False]).drop(
        columns=["group_order"]
    )
    return stats


def plot_unified_features(stats: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of all 75 features' human win rates, colored by feature group."""
    # Prepare order & colors
    stats = stats.copy().reset_index(drop=True)
    x = np.arange(len(stats))
    colors = [COLORS.get(g, "#888888") for g in stats["feature_group"]]

    fig, ax = plt.subplots(figsize=(18, 8))

    bars = ax.bar(x, stats["human_win_rate"] * 100, color=colors, edgecolor="black", linewidth=0.8)

    # baseline
    ax.axhline(50, color="#BBBBBB", linestyle="--", linewidth=1.5, label="50% baseline")

    # labels
    for i, row in stats.iterrows():
        y = row["human_win_rate"] * 100
        ax.text(i, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontsize=7)
        if row["significant_star"]:
            ax.text(i, y + 5, row["significant_star"], ha="center", va="bottom", fontsize=9, fontweight="bold")

    # x tick labels: shorter names
    short_names = []
    for f, g in zip(stats["feature"], stats["feature_group"]):
        if g == "CE-VAR":
            # e.g., Agreeableness_cv -> Agr_cv
            base = f.split("_")[0]
            suffix = "_" + f.split("_")[-1]
            short = base[:3] + suffix
        else:
            # geometry features: keep as is but shorter
            short = f.replace("ce_", "ce_").replace("tfidf_", "tf_").replace("sbert_", "sb_")
        short_names.append(short)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=75, ha="right", fontsize=7)

    ax.set_ylabel("Human win rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "RQ1 LV3: Per-feature Binomial Test (Unified 75 Trajectory Features)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # legend
    handles = []
    labels = []
    for g, c in COLORS.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=c))
        labels.append(g)
    ax.legend(handles, labels, title="Feature group", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "unified_75_features_binomial.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-feature binomial test for unified 75 features (RQ1 LV3)")
    parser.add_argument(
        "--detailed",
        type=str,
        default="micro_results/binomial/binomial_test_trajectory_features_lv3_detailed.csv",
        help="Path to detailed comparisons CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/binomial/unified_75_features",
        help="Output directory for plots and CSV",
    )
    args = parser.parse_args()

    detailed_path = Path(args.detailed)
    if not detailed_path.exists():
        raise FileNotFoundError(f"Detailed CSV not found: {detailed_path}")

    df = pd.read_csv(detailed_path)

    # sanity: we expect 4 feature groups
    print("Feature groups in detailed data:", df["feature_type"].unique())

    stats = compute_per_feature_stats(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save CSV
    csv_path = out_dir / "unified_75_features_binomial.csv"
    stats.to_csv(csv_path, index=False)
    print(f"Saved per-feature stats to: {csv_path}")

    # plot
    plot_unified_features(stats, out_dir)


if __name__ == "__main__":
    main()







