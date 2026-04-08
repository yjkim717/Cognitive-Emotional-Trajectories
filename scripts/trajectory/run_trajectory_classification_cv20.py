#!/usr/bin/env python3
"""
Human vs. LLM (LV3 only) classification on 20 CE CV features.

Configuration:
- Features: ONLY 20 CV features (one per CE/text feature)
- Model: RandomForestClassifier
- CV: 5-fold GroupKFold (group by author_id)

Outputs:
- CSV: plots/trajectory/combined/classification_cv20_<domains>_<level>.csv
- Feature importance CSV: plots/trajectory/combined/classification_cv20_importance_<domains>_<level>.csv
- Feature importance barplot: plots/trajectory/combined/classification_cv20_importance_<domains>_<level>.png
- Metrics barplot (Accuracy / ROC-AUC / F1):
  plots/trajectory/combined/classification_cv20_metrics_<domains>_<level>.png
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

# 20 CV features
CV20_FEATURES = [
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


def load_samples(domains: List[str], models: List[str], level: str, llm_with_history: bool = False) -> pd.DataFrame:
    """Load Human + LLM (or LLM_with_history) data for given domains/models/level.
    
    For regular LLM: uses trajectory_features_combined.csv (per-sample)
    For LLM_with_history: uses author_timeseries_stats_merged.csv (per-author, CV features)
    """
    frames: List[pd.DataFrame] = []
    llm_prefix = "LLM_with_history" if llm_with_history else "LLM"
    
    for domain in domains:
        # Human
        if llm_with_history:
            # For LLM_with_history, use author-level stats (same as LLM_with_history)
            human_path = DATA_ROOT / "human" / domain / "author_timeseries_stats_merged.csv"
        else:
            # For regular LLM, use trajectory_features_combined.csv
            human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
            
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            df_h["provider"] = "human"
            df_h["level"] = "LV0"
            frames.append(df_h)

        # LLMs or LLM_with_history
        for provider in models:
            if llm_with_history:
                # For LLM_with_history, use author-level stats
                csv_path = DATA_ROOT / llm_prefix / provider / level / domain / "author_timeseries_stats_merged.csv"
            else:
                # For regular LLM, use trajectory_features_combined.csv
                csv_path = DATA_ROOT / llm_prefix / provider / level / domain / "trajectory_features_combined.csv"
                
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["domain"] = domain
            df["label"] = "llm_with_history" if llm_with_history else "llm"
            df["provider"] = provider
            df["level"] = level
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def select_cv20_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 20 CV features, ensuring they exist in the DataFrame."""
    available = [col for col in CV20_FEATURES if col in df.columns]
    missing = [col for col in CV20_FEATURES if col not in df.columns]
    if missing:
        print(f"⚠ Warning: missing CV features (will be skipped): {missing}")
    if not available:
        raise ValueError("No CV20 features found in DataFrame.")
    return df[available].fillna(0.0)


def evaluate_cv20(df: pd.DataFrame, X_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run RandomForest + GroupKFold on 20 CV features.

    Returns:
        - results_df: one row with metrics for cv20
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
        "feature_set": "cv20",
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
    """Plot bar chart of feature importance for the 20 CV features (percentage)."""
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
    plt.title("RandomForest feature importance (20 CV features)", fontsize=13, fontweight="bold", pad=10)

    # Add percentage labels above bars
    for i, v in enumerate(imp["importance_pct"]):
        plt.text(i, v + (max(imp["importance_pct"]) * 0.01), f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved importance plot → {out_path}")

def plot_metrics_bar(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart of accuracy / ROC-AUC / F1 for the cv20 model."""
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
    title = "Human vs LLM_with_history (20 CV features)" if "llm_with_history" in str(out_path) else "Human vs LLM (20 CV features)"
    plt.title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics plot → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Human vs LLM (LV3) classification on 20 CV features.")
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
        choices=LEVELS,
        help="LLM level to include (default: LV3).",
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Use LLM_with_history data instead of regular LLM data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_samples(args.domains, args.models, args.level, llm_with_history=args.llm_with_history)
    if df.empty:
        print("⚠ No data available for the requested configuration.")
        return

    data_type = "LLM_with_history" if args.llm_with_history else "LLM"
    print(f"=== Classifying Human vs {data_type} ({args.level}) on 20 CV features ===")
    print(f"Domains: {', '.join(args.domains).upper()}")
    print(f"Models:  {', '.join(args.models)}")

    # Select 20 CV features
    X_df = select_cv20_features(df)
    print(f"Using {X_df.shape[1]} CV features: {list(X_df.columns)}")

    # Evaluate + importance
    results_df, importance_df = evaluate_cv20(df, X_df)

    # Print results
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)
    row = results_df.iloc[0]
    print(f"Accuracy:       {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
    print(f"AUC:            {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}")
    print(f"F1 Score:       {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
    print(f"Human Recall:   {row['human_recall_mean']:.4f} ± {row['human_recall_std']:.4f}")
    print(f"N samples:      {row['n_samples']}")
    
    print("\n" + "=" * 80)
    print("TOP 10 FEATURE IMPORTANCE")
    print("=" * 80)
    top_10 = importance_df.head(10)
    total_importance = importance_df['importance_mean'].sum()
    for rank, (_, row) in enumerate(top_10.iterrows(), start=1):
        importance_pct = (row['importance_mean'] / total_importance) * 100.0
        print(f"{rank:2d}. {row['feature']:30s} {row['importance_mean']:.6f} ({importance_pct:5.2f}%)")
    
    print("\n" + "=" * 80)

    out_dir = PLOTS_ROOT / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_suffix = "llm_with_history" if args.llm_with_history else ""
    suffix = f"{'_'.join(args.domains)}_{args.level}"
    if data_suffix:
        suffix = f"{data_suffix}_{suffix}"

    # Save metrics
    out_path = out_dir / f"classification_cv20_{suffix}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Results → {out_path}")

    # Save metrics bar plot (Accuracy / ROC-AUC / F1)
    metrics_png_path = out_dir / f"classification_cv20_metrics_{suffix}.png"
    plot_metrics_bar(results_df, metrics_png_path)

    # Save importance CSV
    imp_csv_path = out_dir / f"classification_cv20_importance_{suffix}.csv"
    importance_df.to_csv(imp_csv_path, index=False)
    print(f"Feature importance CSV → {imp_csv_path}")

    # Save importance bar plot
    imp_png_path = out_dir / f"classification_cv20_importance_{suffix}.png"
    plot_importance_bar(importance_df, imp_png_path)


if __name__ == "__main__":
    main()