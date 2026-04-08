
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.file_utils import read_text as read_utf8_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK", "CL35", "G4OM")
LEVELS = ("LV1", "LV2", "LV3")
LLM_WITH_HISTORY_MODELS = ("DS", "CL35", "G4OM")
LLM_WITH_HISTORY_LEVEL = "LV3"
LLM_WITH_HISTORY_DOMAIN = "news"
MAX_DOC_CHARS = 8_000
TFIDF_MAX_FEATURES = 20_000
SVD_COMPONENTS = 10
OUTPUT_FILENAME = "tfidf_vectors.csv"


def load_text(rel_path: str) -> str:
    """Read UTF-8 text from project-relative or absolute path with truncation."""
    p = Path(rel_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Missing text file: {p}")
    text = read_utf8_text(str(p))
    return text[:MAX_DOC_CHARS]


def collect_entries(domain: str) -> List[Dict]:
    """Collect all samples for a domain (human + each provider/level)."""
    entries: List[Dict] = []

    def _load_csv(csv_path: Path, label: str, provider: str, level: str) -> None:
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel_path = row.get("path")
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue

            try:
                text = load_text(rel_path)
            except FileNotFoundError:
                continue

            metadata = {
                "filename": row.get("filename"),
                "path": rel_path,
                "label": label,
                "domain": row.get("domain", domain),
                "field": row.get("field"),
                "author_id": row.get("author_id"),
                "provider": provider,
                "level": level,
                "model": row.get("model"),
            }

            out_path = (
                DATA_ROOT / "human" / domain / OUTPUT_FILENAME
                if label == "human"
                else DATA_ROOT / "LLM" / provider / level / domain / OUTPUT_FILENAME
            )

            metadata["output_path"] = str(out_path)
            entries.append(
                {
                    "text": text,
                    "metadata": metadata,
                }
            )

    # Human split
    human_csv = DATA_ROOT / "human" / domain / "combined_merged.csv"
    _load_csv(human_csv, label="human", provider="human", level="LV0")

    # LLM splits
    for provider in PROVIDERS:
        for level in LEVELS:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / "combined_merged.csv"
            _load_csv(csv_path, label="llm", provider=provider, level=level)

    return entries


def collect_entries_llm_with_history(domain: str) -> List[Dict]:
    """Collect samples from LLM_with_history for a given domain (DS/CL35/G4OM, LV3)."""
    entries: List[Dict] = []

    def _load_csv(csv_path: Path, label: str, provider: str, level: str) -> None:
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel_path = row.get("path")
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue
            try:
                text = load_text(rel_path)
            except FileNotFoundError:
                continue
            metadata = {
                "filename": row.get("filename"),
                "path": rel_path,
                "label": label,
                "domain": row.get("domain", domain),
                "field": row.get("field"),
                "author_id": row.get("author_id"),
                "provider": provider,
                "level": level,
                "model": row.get("model"),
            }
            metadata["output_path"] = str(DATA_ROOT / "LLM_with_history" / provider / level / domain / OUTPUT_FILENAME)
            entries.append({"text": text, "metadata": metadata})

    for provider in LLM_WITH_HISTORY_MODELS:
        csv_path = DATA_ROOT / "LLM_with_history" / provider / LLM_WITH_HISTORY_LEVEL / domain / "combined_merged.csv"
        _load_csv(csv_path, label="llm", provider=provider, level=LLM_WITH_HISTORY_LEVEL)
    return entries


def write_outputs(entries: List[Dict], vectors) -> None:
    """Write TF-IDF vectors grouped by output path."""
    vector_columns = [f"tfidf_{i+1}" for i in range(SVD_COMPONENTS)]
    vectors_df = pd.DataFrame(vectors, columns=vector_columns)
    meta_df = pd.DataFrame([entry["metadata"] for entry in entries])
    combined_df = pd.concat([meta_df, vectors_df], axis=1)

    for output_path, group in combined_df.groupby("output_path"):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["output_path"]).to_csv(output_file, index=False)
        print(f"✅ {output_file}: {len(group)} samples")


def process_domain(domain: str) -> None:
    entries = collect_entries(domain)
    if not entries:
        print(f"⚠ No entries found for domain '{domain}'. Skipping.")
        return

    texts = [entry["text"] for entry in entries]
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)

    write_outputs(entries, reduced)


def process_llm_with_history(domains=None) -> None:
    """Extract TF-IDF for LLM_with_history across all specified domains."""
    if domains is None:
        domains = list(DOMAINS)
    for domain in domains:
        print(f"\n=== TF-IDF vectors for LLM_with_history/{domain} ===")
        entries = collect_entries_llm_with_history(domain)
        if not entries:
            print(f"⚠ No LLM_with_history entries found for domain '{domain}'. Run batch_analyze_metrics.py --llm-with-history first.")
            continue
        texts = [e["text"] for e in entries]
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)
        write_outputs(entries, reduced)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract TF-IDF vectors for micro datasets.")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all).",
    )
    parser.add_argument(
        "--llm-with-history",
        action="store_true",
        help="Process only LLM_with_history (news, DS/CL35/G4OM, LV3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_with_history:
        process_llm_with_history(domains=args.domains)
        return
    for domain in args.domains:
        print(f"\n=== TF-IDF vectors for {domain} ===")
        process_domain(domain)


if __name__ == "__main__":
    main()

