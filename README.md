# Cognitive-Emotional-Trajectories

This repository contains the data-processing, analysis, and visualization pipeline used to compare longitudinal human writing trajectories with LLM-generated writing trajectories.

The project focuses on temporal structure in writing across multiple representational spaces:

- cognitive-emotional features (CE)
- lexical features (TFIDF)
- semantic embeddings (SBERT and E5)

It also includes an `LLM_with_history` branch for experiments in which generation is conditioned on incremental historical context.

## Dataset

The processed dataset (~1.2 GB) is hosted on HuggingFace:
[https://huggingface.co/datasets/zhanweicao/cognitive-emotional-trajectories](https://huggingface.co/datasets/zhanweicao/cognitive-emotional-trajectories)

### Setup (one-time)

**1. Install the HuggingFace Hub client** (skip if already installed):

```bash
pip install huggingface_hub
```

**2. Download and extract the dataset:**

```bash
python download_dataset.py
```

The script will:
- Download `dataset.zip` (~1.2 GB) from HuggingFace into the project root
- Validate the archive
- Extract it, creating (or replacing) the `dataset/` folder

After this step all scripts in `scripts/` can be run without any further setup.

## Repository Structure

- `dataset/`
  Raw and processed data tables.

- `scripts/`
  Feature extraction, trajectory construction, hypothesis tests, classification, and plotting.

- `results/`
  Generated summaries, comparison reports, and exported result tables.

- `plots/`
  Figures and visualization outputs.

- `docs/`
  Pipeline notes and supporting documentation.

- `utils/`
  Shared helper functions used by the scripts.

## Main Analysis Components

### 1. Sample-Level Feature Extraction

Raw text files are converted into:

- Big Five proxy features
- merged NELA features
- TFIDF vectors
- SBERT vectors
- optional E5 semantic vectors

The main CE sample-level table is:

- `combined_merged.csv`

The merged CE + embedding table is:

- `combined_with_embeddings.csv`

### 2. Author-Level Temporal Features

The pipeline builds author-level temporal representations including:

- CE variability statistics:
  - `CV`
  - `RMSSD`
  - `MASD`
  - normalized variants
- drift features in CE, TFIDF, and SBERT spaces

The key author-level CE table is:

- `author_timeseries_stats_merged.csv`

The key drift inputs are:

- `ce_drift.csv`
- `tfidf_drift.csv`
- `sbert_drift.csv`

### 3. Statistical Testing and Classification

The repository includes scripts for:

- matched-pair binomial tests
- CE-based human-vs-LLM classification
- robustness checks across metrics and encoders
- instance-wise vs. `LLM_with_history` comparisons

## `LLM_with_history`

The raw text for the incremental-context condition is stored under:

- `dataset/llm/llm_with_history/`

Processed outputs for this branch are written under:

- `dataset/process/LLM_with_history/`

## Documentation

For a categorized overview of the available scripts and the most common command chains, see:

- `scripts/README.md`

For pipeline-level notes and implementation details, see:

- `PIPELINE_NOTES.md`
- `docs/LLM_with_history_pipeline.md`

## Notes

- CE variability analyses typically use outlier-handled CE tables.
- Drift analyses use the raw CE and embedding trajectory representations.
- The repository contains both the original instance-wise generation setup and the `LLM_with_history` comparison branch.
