#!/usr/bin/env python3
"""
Download and extract the dataset from HuggingFace.

Usage:
    python download_dataset.py

Downloads dataset.zip (~1.2 GB) from HuggingFace and extracts it into
the dataset/ folder at the project root. If a dataset/ folder already
exists it will be replaced.
"""

import shutil
import sys
import zipfile
from pathlib import Path

REPO_ID = "zhanweicao/cognitive-emotional-trajectories"
FILENAME = "dataset.zip"
PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       Run:  pip install huggingface_hub")
        sys.exit(1)

    zip_path = PROJECT_ROOT / FILENAME

    # ── 1. Download ──────────────────────────────────────────────────────────
    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    try:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT),
        )
    except Exception as e:
        print(f"ERROR: Download failed — {e}")
        sys.exit(1)

    downloaded = Path(downloaded)
    print(f"Saved to {downloaded}")

    # ── 2. Validate zip before touching anything ──────────────────────────────
    print("Validating zip file ...")
    if not zipfile.is_zipfile(downloaded):
        print("ERROR: Downloaded file is not a valid zip archive.")
        sys.exit(1)

    # ── 3. Remove old dataset/ only after a confirmed good download ───────────
    dataset_dir = PROJECT_ROOT / "dataset"
    if dataset_dir.exists():
        print("Removing existing dataset/ folder ...")
        shutil.rmtree(dataset_dir)

    # ── 4. Extract ────────────────────────────────────────────────────────────
    print("Extracting dataset.zip ...")
    with zipfile.ZipFile(downloaded, "r") as zf:
        members = zf.infolist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, PROJECT_ROOT)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} files extracted", end="\r")
    print()

    print("Done. dataset/ folder is ready.")
    print("You can now run any script in scripts/ without further setup.")


if __name__ == "__main__":
    main()
