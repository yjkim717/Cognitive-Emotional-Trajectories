#!/usr/bin/env python3
"""
Human vs. LLM (LV3 only) classification on 20 CE RMSSD_norm features.

Configuration:
- Features: ONLY 20 RMSSD_norm features (one per CE/text feature)
- Model: RandomForestClassifier
- CV: 5-fold GroupKFold (group by author_id)

Outputs:
- CSV: plots/trajectory/combined/classification_rmssd20_<domains>_<level>.csv
- Feature importance CSV: plots/trajectory/combined/classification_rmssd20_importance_<domains>_<level>.csv
- Feature importance barplot: plots/trajectory/combined/classification_rmssd20_importance_<domains>_<level>.png
- Metrics barplot (Accuracy / ROC-AUC / F1):
  plots/trajectory/combined/classification_rmssd20_metrics_<domains>_<level>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
# Default models for RMSSD/MASD: only DS, G4B, G12B, LMK (LV1-3)
DEFAULT_MODELS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")

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

# 20 RMSSD_norm features
RMSSD20_FEATURES = [
    "Agreeableness_rmssd_norm",
    "Conscientiousness_rmssd_norm",
    "Extraversion_rmssd_norm",
    "Neuroticism_rmssd_norm",
    "Openness_rmssd_norm",
    "average_word_length_rmssd_norm",
    "avg_sentence_length_rmssd_norm",
    "content_word_ratio_rmssd_norm",
    "flesch_reading_ease_rmssd_norm",
    "function_word_ratio_rmssd_norm",
    "gunning_fog_rmssd_norm",
    "num_words_rmssd_norm",
    "polarity_rmssd_norm",
    "subjectivity_rmssd_norm",
    "vader_compound_rmssd_norm",
    "vader_neg_rmssd_norm",
    "vader_neu_rmssd_norm",
    "vader_pos_rmssd_norm",
    "verb_ratio_rmssd_norm",
    "word_diversity_rmssd_norm",
]


def load_samples(domains: List[str], models: List[str], level: str) -> pd.DataFrame:
    """Load Human + LLM trajectory_features_combined for given domains/models/level."""
    frames: List[pd.DataFrame] = []
    for domain in domains:
        # Human
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            df_h["provider"] = "human"
            df_h["level"] = "LV0"
            frames.append(df_h)

        # LLMs
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

    combined = pd.concat(frames, ignore_index=True)
    return combined


def select_rmssd20_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 20 RMSSD_norm features, ensuring they exist in the DataFrame."""
    available = [col for col in RMSSD20_FEATURES if col in df.columns]
    missing = [col for col in RMSSD20_FEATURES if col not in df.columns]
    if missing:
        print(f"⚠ Warning: missing RMSSD_norm features (will be skipped): {missing}")
    if not available:
        raise ValueError("No RMSSD20 features found in DataFrame.")
    return df[available].fillna(0.0)


def evaluate_rmssd20(df: pd.DataFrame, X_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run RandomForest + GroupKFold on 20 RMSSD_norm features.

    Returns:
        - results_df: one row with metrics for rmssd20
        - importance_df: feature importance (mean across folds)
    """
    y = (df["label"] == "human").astype(int).values
    groups = df["author_id"].values
    gkf = GroupKFold(n_splits=5)

    X = X_df.to_numpy()
    feature_names = list(X_df.columns)

    accs: List[float] = []
    rocs: List[float] = []
    f1s: List[float] = []
    recalls: List[float] = []  # Human class recall
    importances: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42 + fold_idx,  # small variation across folds
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]

        accs.append(accuracy_score(y[test_idx], y_pred))
        rocs.append(roc_auc_score(y[test_idx], proba))
        f1s.append(f1_score(y[test_idx], y_pred))
        # Human class recall (y=1 is human)
        recalls.append(recall_score(y[test_idx], y_pred, pos_label=1))
        importances.append(clf.feature_importances_)

    # Metrics summary
    results_row = {
        "feature_set": "rmssd20",
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "roc_auc_mean": float(np.mean(rocs)),
        "roc_auc_std": float(np.std(rocs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "human_recall_mean": float(np.mean(recalls)),
        "human_recall_std": float(np.std(recalls)),
        "n_samples": int(len(df)),
    }
    results_df = pd.DataFrame([results_row])

    # Feature importance (average across folds)
    importances_arr = np.vstack(importances)  # shape: (n_folds, n_features)
    mean_importance = importances_arr.mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": mean_importance,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return results_df, importance_df


def plot_importance_bar(importance_df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart of feature importance for the 20 RMSSD_norm features (percentage)."""
    plt.figure(figsize=(10, 5))
    imp = importance_df.copy()
    imp = imp.sort_values("importance_mean", ascending=False)

    # Convert to percentage of total importance
    total = imp["importance_mean"].sum()
    if total <= 0:
        # Fallback: just plot raw values
        imp["importance_pct"] = imp["importance_mean"]
        ylabel = "Feature importance (raw)"
    else:
        imp["importance_pct"] = imp["importance_mean"] / total * 100.0
        ylabel = "Feature importance (%)"

    x = np.arange(len(imp))
    plt.bar(x, imp["importance_pct"], color="#4A90E2", edgecolor="black", linewidth=0.8)

    plt.xticks(x, imp["feature"], rotation=75, ha="right", fontsize=8)
    plt.ylabel(ylabel, fontsize=11, fontweight="bold")
    plt.title("RandomForest feature importance (20 RMSSD_norm features)", fontsize=13, fontweight="bold", pad=10)

    # Add percentage labels above bars
    for i, v in enumerate(imp["importance_pct"]):
        plt.text(i, v + (max(imp["importance_pct"]) * 0.01), f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved importance plot → {out_path}")

def plot_metrics_bar(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart of accuracy / ROC-AUC / F1 for the rmssd20 model."""
    row = results_df.iloc[0]

    metrics = ["accuracy_mean", "roc_auc_mean", "f1_mean"]
    values = [row[m] for m in metrics]
    labels = ["Accuracy", "ROC-AUC", "F1"]

    plt.figure(figsize=(4, 4))
    x = np.arange(len(metrics))
    plt.bar(
        x,
        values,
        color=["#4A90E2", "#50E3C2", "#F5A623"],
        edgecolor="black",
        linewidth=0.8,
    )

    plt.xticks(x, labels, rotation=0, fontsize=10)
    plt.ylim(0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Human vs LLM (20 RMSSD_norm features)", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics plot → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Human vs LLM (LV3) classification on 20 RMSSD_norm features.")
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
        default=list(DEFAULT_MODELS),
        help=f"LLM models to include (default: {', '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--level",
        nargs="+",
        choices=LEVELS,
        default=list(LEVELS),
        help="LLM level(s) to include (default: all levels).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Process each level separately
    all_results = []
    all_importance = []
    
    for level in args.level:
        df = load_samples(args.domains, args.models, level)
        if df.empty:
            print(f"⚠ No data available for level {level}.")
            continue

        print(f"\n{'=' * 80}")
        print(f"=== Classifying Human vs LLM ({level}) on 20 RMSSD_norm features ===")
        print(f"{'=' * 80}")
        print(f"Domains: {', '.join(args.domains).upper()}")
        print(f"Models:  {', '.join(args.models)}")

        # Select 20 RMSSD_norm features
        X_df = select_rmssd20_features(df)
        print(f"Using {X_df.shape[1]} RMSSD_norm features: {list(X_df.columns)}")

        # Evaluate + importance
        results_df, importance_df = evaluate_rmssd20(df, X_df)
        results_df['level'] = level
        importance_df['level'] = level

        # Print results
        print("\n" + "=" * 80)
        print(f"CLASSIFICATION RESULTS ({level})")
        print("=" * 80)
        row = results_df.iloc[0]
        print(f"Accuracy:       {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
        print(f"AUC:            {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}")
        print(f"F1 Score:       {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
        print(f"Human Recall:   {row['human_recall_mean']:.4f} ± {row['human_recall_std']:.4f}")
        print(f"N samples:      {row['n_samples']}")
        
        print("\n" + "=" * 80)
        print(f"TOP 10 FEATURE IMPORTANCE ({level})")
        print("=" * 80)
        top_10 = importance_df.head(10)
        total_importance = importance_df['importance_mean'].sum()
        for rank, (_, row) in enumerate(top_10.iterrows(), start=1):
            importance_pct = (row['importance_mean'] / total_importance) * 100.0
            print(f"{rank:2d}. {row['feature']:35s} {row['importance_mean']:.6f} ({importance_pct:5.2f}%)")
        
        all_results.append(results_df)
        all_importance.append(importance_df)

        out_dir = PLOTS_ROOT / "combined"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{'_'.join(args.domains)}_{level}"

        # Save metrics
        out_path = out_dir / f"classification_rmssd20_{suffix}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nResults → {out_path}")

        # Save metrics bar plot (Accuracy / ROC-AUC / F1)
        metrics_png_path = out_dir / f"classification_rmssd20_metrics_{suffix}.png"
        plot_metrics_bar(results_df, metrics_png_path)

        # Save importance CSV
        imp_csv_path = out_dir / f"classification_rmssd20_importance_{suffix}.csv"
        importance_df.to_csv(imp_csv_path, index=False)
        print(f"Feature importance CSV → {imp_csv_path}")

        # Save importance bar plot
        imp_png_path = out_dir / f"classification_rmssd20_importance_{suffix}.png"
        plot_importance_bar(importance_df, imp_png_path)
    
    # Save combined results if multiple levels
    if len(all_results) > 1:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_importance = pd.concat(all_importance, ignore_index=True)
        
        out_dir = PLOTS_ROOT / "combined"
        suffix = f"{'_'.join(args.domains)}_{'_'.join(args.level)}"
        
        combined_results_path = out_dir / f"classification_rmssd20_{suffix}.csv"
        combined_results.to_csv(combined_results_path, index=False)
        print(f"\nCombined results → {combined_results_path}")
        
        combined_importance_path = out_dir / f"classification_rmssd20_importance_{suffix}.csv"
        combined_importance.to_csv(combined_importance_path, index=False)
        print(f"Combined importance → {combined_importance_path}")


if __name__ == "__main__":
    main()

