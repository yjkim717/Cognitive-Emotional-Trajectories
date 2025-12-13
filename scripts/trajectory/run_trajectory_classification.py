#!/usr/bin/env python3
"""
Human vs. LLM (LV3 only) classification on trajectory feature sets.

Three configurations per domain:
1. Variability only (CE CV/MASD/RMSSD columns)
2. Geometry only (CE/TFIDF/SBERT mean/std/D/L/tau/C)
3. Unified (entire trajectory_features_combined table, excluding metadata)

Outputs: plots/trajectory/<domain>/classification_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
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

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# CE-VAR: ONLY normalized variability metrics for CE features
#         - coefficient of variation (_cv)
#         - normalized RMSSD (_rmssd_norm)
#         - normalized MASD (_masd_norm)
VAR_SUFFIXES = ("_cv", "_rmssd_norm", "_masd_norm")
GEOMETRY_PREFIXES = ("ce_", "tfidf_", "sbert_")
GEOMETRY_METRICS = (
    "mean_distance",
    "std_distance",
    "net_displacement",
    "path_length",
    "tortuosity",
    # "direction_consistency",  # Excluded for this experiment
)


def load_samples(domains: List[str], models: List[str], level: str, include_geometry_spaces: List[str]) -> pd.DataFrame:
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
    combined = pd.concat(frames, ignore_index=True)
    # Filter geometry columns to requested spaces only
    geom_cols = geometry_columns(combined, include_geometry_spaces + ["all"])
    drop_cols = []
    for space in ["ce", "tfidf", "sbert"]:
        if space not in include_geometry_spaces:
            drop_cols.extend(geom_cols.get(space, []))
    if drop_cols:
        combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])
    return combined


def geometry_columns(df: pd.DataFrame, spaces: List[str]) -> Dict[str, List[str]]:
    cols: Dict[str, List[str]] = {}
    for space in spaces:
        prefix = f"{space}_"
        space_cols = []
        for metric in GEOMETRY_METRICS:
            col = f"{prefix}{metric}"
            if col in df.columns:
                space_cols.append(col)
        cols[space] = space_cols
    cols["all"] = sum(cols.values(), [])
    return cols


def select_feature_sets(df: pd.DataFrame, geometry_spaces: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Build feature sets for:
      1) CE-VAR       : normalized CE variability metrics (CV, RMSSD_norm, MASD_norm)
      2) geometry_all : CE/TFIDF/SBERT geometry metrics (5 per space, no direction_consistency)
      3) unified      : CE-VAR + all geometry metrics (no n_years columns)
      4) ce_geometry  : CE geometry only
      5) tfidf_geometry / sbert_geometry : per-space geometry
    """
    # CE variability: ONLY CV / RMSSD_norm / MASD_norm columns
    variability_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLS and any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]

    # Geometry: CE/TF-IDF/SBERT mean/std + D/L/τ (excludes n_years & direction_consistency)
    geom_cols = geometry_columns(df, geometry_spaces + ["all"])

    # Unified = CE-VAR + all geometry metrics (no n_years, no direction_consistency)
    unified_cols = sorted(set(variability_cols + geom_cols["all"]))

    feature_sets: Dict[str, pd.DataFrame] = {
        "variability": df[variability_cols].fillna(0.0),
        "geometry_all": df[geom_cols["all"]].fillna(0.0),
        "unified": df[unified_cols].fillna(0.0),
    }
    for space in geometry_spaces:
        if geom_cols.get(space):
            feature_sets[f"{space}_geometry"] = df[geom_cols[space]].fillna(0.0)
    return feature_sets


def evaluate(df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    y = (df["label"] == "human").astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: List[Dict] = []
    for name, X_df in feature_sets.items():
        X = X_df.to_numpy()
        accs: List[float] = []
        rocs: List[float] = []
        f1s: List[float] = []
        for train_idx, test_idx in skf.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            accs.append(accuracy_score(y[test_idx], y_pred))
            rocs.append(roc_auc_score(y[test_idx], proba))
            f1s.append(f1_score(y[test_idx], y_pred))
        rows.append(
            {
                "feature_set": name,
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "roc_auc_mean": np.mean(rocs),
                "roc_auc_std": np.std(rocs),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
                "n_samples": len(df),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Human vs LLM (LV3) trajectory classification.")
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
        "--geometry-spaces",
        nargs="+",
        choices=["ce", "tfidf", "sbert"],
        default=["ce", "tfidf", "sbert"],
        help="Which geometry spaces to report separately (default: all three).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_samples(args.domains, args.models, args.level, args.geometry_spaces)
    if df.empty:
        print("⚠ No data available for the requested configuration.")
        return

    print(f"=== Classifying Human vs LLM ({args.level}) across domains: {', '.join(args.domains).upper()} ===")
    feature_sets = select_feature_sets(df, args.geometry_spaces)
    results = evaluate(df, feature_sets)

    out_dir = PLOTS_ROOT / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"classification_results_{'_'.join(args.domains)}_{args.level}.csv"
    results.to_csv(out_path, index=False)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()

