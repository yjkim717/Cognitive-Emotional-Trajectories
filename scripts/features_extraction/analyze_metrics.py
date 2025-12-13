import argparse
import os

import pandas as pd

from utils.metrics_big5 import extract_big5_features
from utils.metric_nela_merged import extract_nela_features_merged


DATA_ROOT = "dataset"
HUMAN_DIR = os.path.join(DATA_ROOT, "human")
LLM_DIR = os.path.join(DATA_ROOT, "llm")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "process")
MERGE_KEYS = ["filename", "path", "label"]


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    drop_columns = []

    for col in df.columns:
        if col.endswith("_x"):
            base = col[:-2]
            other = f"{base}_y"
            if other in df.columns:
                drop_columns.append(other)
            rename_map[col] = base
        elif col.endswith("_y"):
            base = col[:-2]
            if base not in df.columns and base not in rename_map.values():
                rename_map[col] = base

    df = df.rename(columns=rename_map)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Big Five and enhanced NELA features for selected subsets."
    )
    parser.add_argument(
        "--target",
        choices=["human", "llm"],
        required=True,
        help="Choose which dataset to process.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name to process (e.g., academic, news, blogs).",
    )
    parser.add_argument(
        "--model",
        help="LLM model name (required when --target=llm).",
    )
    parser.add_argument(
        "--level",
        help="LLM level (e.g., LV1, LV2; required when --target=llm).",
    )
    parser.add_argument(
        "--skip-big5",
        action="store_true",
        help="Skip Big Five extraction stage.",
    )
    parser.add_argument(
        "--skip-nela",
        action="store_true",
        help="Skip enhanced NELA extraction stage.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip combining Big Five and NELA outputs.",
    )

    args = parser.parse_args()

    if args.target == "llm":
        if not args.model or not args.level:
            parser.error("For LLM analysis, please provide both --model and --level.")
    else:
        if args.model or args.level:
            parser.error("--model and --level are only valid when --target=llm.")

    return args


def main() -> None:
    args = parse_args()

    domain = args.domain.lower()
    run_big5 = not args.skip_big5
    run_nela = not args.skip_nela
    run_combine = not args.skip_combine

    if args.target == "human":
        dataset_dir = HUMAN_DIR
        label = "human"
        model_name = None
        level_name = None
        output_dir = os.path.join(OUTPUT_ROOT, "human", domain)
    else:
        dataset_dir = LLM_DIR
        label = "llm"
        model_name = args.model.upper()
        level_name = args.level.upper()
        output_dir = os.path.join(OUTPUT_ROOT, "LLM", model_name, level_name, domain)

    os.makedirs(output_dir, exist_ok=True)

    big5_path = os.path.join(output_dir, "big5.csv")
    nela_path = os.path.join(output_dir, "nela_merged.csv")
    combined_path = os.path.join(output_dir, "combined.csv")

    df_big5 = None
    df_nela = None

    if run_big5:
        print("Extracting Big Five features...")
        df_big5 = extract_big5_features(
            dataset_dir,
            label,
            big5_path,
            domain=domain,
            model_name=model_name,
            level=level_name,
        )
    else:
        if os.path.exists(big5_path):
            print(f"Loading existing Big Five features from {big5_path}")
            df_big5 = pd.read_csv(big5_path)
        else:
            raise FileNotFoundError(
                f"Big Five features not found at {big5_path}. Remove --skip-big5 to generate them."
            )

    if run_nela:
        print("Extracting merged NELA features...")
        df_nela = extract_nela_features_merged(
            dataset_dir,
            label,
            nela_path,
            domain=domain,
            model_name=model_name,
            level=level_name,
        )
    else:
        if os.path.exists(nela_path):
            print(f"Loading existing merged NELA features from {nela_path}")
            df_nela = pd.read_csv(nela_path)
        else:
            raise FileNotFoundError(
                f"NELA features not found at {nela_path}. Remove --skip-nela to generate them."
            )

    # Remove duplicated metadata columns from NELA results; keep the Big Five versions.
    if df_nela is not None:
        duplicate_meta_cols = [
            col
            for col in [
                "domain",
                "field",
                "author_id",
                "year",
                "item_index",
                "model",
                "level",
            ]
            if col in df_nela.columns
        ]
        if duplicate_meta_cols:
            df_nela = df_nela.drop(columns=duplicate_meta_cols)

    if run_combine:
        if df_big5 is None or df_nela is None:
            raise RuntimeError("Cannot combine features without both Big Five and NELA data.")

        print("Combining Big Five and merged NELA results...")
        df_combined = pd.merge(df_big5, df_nela, on=MERGE_KEYS, how="inner")
        df_combined = _deduplicate_columns(df_combined)
        drop_meta_cols = [col for col in ["year", "item_index", "year_x", "year_y", "item_index_x", "item_index_y"] if col in df_combined.columns]
        if drop_meta_cols:
            df_combined = df_combined.drop(columns=drop_meta_cols)
        df_combined.to_csv(combined_path, index=False)
        print(f"Combined features saved to {combined_path}")
        if not df_combined.empty:
            print("\nSample preview:")
            print(df_combined.head(3))
    else:
        print("Skipping combination stage.")


if __name__ == "__main__":
    main()
