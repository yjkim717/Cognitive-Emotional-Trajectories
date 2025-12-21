#!/usr/bin/env python3
"""
Analyze feature importance for CE-VAR classification.

Extracts top important features from RandomForest classifier
for the variability (CE-VAR) feature set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")

METADATA_COLS = {
    "field",
    "author_id",
    "sample_count",
    "domain",
    "label",
    "provider",
    "level",
    "model",
}

VAR_SUFFIXES = ("_cv", "_rmssd_norm", "_masd_norm")


def load_samples(domains: List[str], models: List[str], level: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for domain in domains:
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            df_h["provider"] = "human"
            df_h["level"] = "LV0"
            frames.append(df_h)
        for provider in models:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["domain"] = domain
            df["label"] = "llm"
            df["provider"] = provider
            df["level"] = level
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_ce_var_features(df: pd.DataFrame) -> List[str]:
    """Extract CE-VAR features (CV, RMSSD_norm, MASD_norm)."""
    variability_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLS and any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]
    return sorted(variability_cols)


def analyze_feature_importance(df: pd.DataFrame, feature_cols: List[str], top_k: int = 20) -> pd.DataFrame:
    """Train RandomForest and extract feature importance."""
    X = df[feature_cols].fillna(0.0).to_numpy()
    y = (df["label"] == "human").astype(int).values
    
    # Use 5-fold CV with GroupKFold to prevent group leakage
    # Group by author_id to ensure all samples from the same author stay in the same fold
    groups = df["author_id"].values
    gkf = GroupKFold(n_splits=5)
    all_importances = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        all_importances.append(clf.feature_importances_)
    
    # Average importance across folds
    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": mean_importance,
        "importance_std": std_importance,
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values("importance_mean", ascending=False)
    
    # Add rank
    importance_df["rank"] = range(1, len(importance_df) + 1)
    
    # Reorder columns
    importance_df = importance_df[["rank", "feature", "importance_mean", "importance_std"]]
    
    return importance_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze feature importance for CE-VAR features.")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to include (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="LLM models to include (default: all providers).",
    )
    parser.add_argument(
        "--level",
        default="LV3",
        help="LLM level to include (default: LV3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features to display (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT / "feature_importance",
        help="Output directory for results.",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data for domains: {args.domains}, models: {args.models}, level: {args.level}")
    df = load_samples(args.domains, args.models, args.level)
    if df.empty:
        print("⚠ No data available.")
        return
    
    print(f"Loaded {len(df)} samples")
    
    # Extract CE-VAR features
    feature_cols = get_ce_var_features(df)
    print(f"\nFound {len(feature_cols)} CE-VAR features")
    
    # Analyze importance
    print("\nAnalyzing feature importance...")
    importance_df = analyze_feature_importance(df, feature_cols, args.top_k)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"ce_var_importance_{'_'.join(args.domains)}_{args.level}.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved full results to: {output_path}")
    
    # Display top features
    print(f"\n{'='*80}")
    print(f"Top {args.top_k} CE-VAR Features by Importance")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':<15} {'Std':<10}")
    print(f"{'-'*80}")
    
    for _, row in importance_df.head(args.top_k).iterrows():
        print(f"{row['rank']:<6} {row['feature']:<40} {row['importance_mean']:<15.6f} {row['importance_std']:<10.6f}")
    
    print(f"\n{'='*80}")
    print(f"Total features analyzed: {len(importance_df)}")
    print(f"Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance_mean']:.6f})")


if __name__ == "__main__":
    main()

