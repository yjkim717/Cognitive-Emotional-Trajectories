# Scripts Guide

This document provides a structured overview of the scripts in this repository and explains how they fit into the main experimental workflows.

The scripts are organized into two directories:

- `scripts/features_extraction/`
- `scripts/trajectory/`

## 1. Feature Extraction Scripts

These scripts operate on raw text or sample-level tables and produce the inputs used by downstream trajectory analyses.

### CE Feature Extraction

- `features_extraction/batch_analyze_metrics.py`
  Extracts Big Five proxy features and merged NELA features from raw text and writes:
  - `big5.csv`
  - `nela_merged.csv`
  - `combined_merged.csv`

- `features_extraction/remove_outliers_from_combined_merged.py`
  Applies IQR-based outlier handling to CE feature columns and writes:
  - `combined_merged_outliers_removed.csv`

### Embedding Extraction

- `features_extraction/extract_tfidf_vectors.py`
  Extracts TFIDF vectors and writes:
  - `tfidf_vectors.csv`

- `features_extraction/extract_sbert_vectors.py`
  Extracts SBERT vectors and writes:
  - `sbert_vectors.csv`

- `features_extraction/merge_features.py`
  Merges CE, TFIDF, and SBERT sample-level features and writes:
  - `combined_with_embeddings.csv`

### E5 Semantic Branch

- `features_extraction/extract_sbert_e5_vectors.py`
  Extracts E5 embeddings.

- `features_extraction/merge_e5_features.py`
  Merges E5 embeddings into the feature tables.

## 2. Trajectory Construction Scripts

These scripts build author-level temporal features from the sample-level inputs.

### CE Variability

- `trajectory/generate_timeseries_stats_from_outliers_removed.py`
  Builds author-level CE variability statistics from `combined_merged_outliers_removed.csv`.
  This is the preferred path for CE mainline analyses involving LLM data.

- `trajectory/generate_timeseries_stats_merged.py`
  Builds author-level CE variability statistics directly from `combined_merged.csv`.
  Also supports `--target llm_with_history`.

### Embedding Variability

- `trajectory/generate_embedding_timeseries_stats.py`
  Builds author-level TFIDF and SBERT variability statistics and writes:
  - `author_timeseries_stats_embeddings.csv`

### Drift

- `trajectory/compute_embedding_drift.py`
  Computes author-level drift in CE, TFIDF, and SBERT spaces and writes:
  - `ce_drift.csv`
  - `tfidf_drift.csv`
  - `sbert_drift.csv`

### Additional Feature-Level Drift

- `trajectory/compute_ce_feature_drift.py`
  Computes drift for individual CE features.

## 3. Statistical Test Scripts

These scripts produce the main comparison results.

### Mainline Tests

- `trajectory/binomial_test_drift.py`
  Unified drift binomial test entry point.

- `trajectory/binomial_test_ce_cv.py`
  CE-CV binomial test.

- `trajectory/binomial_test_ce_rmssd.py`
  CE `RMSSD_norm` binomial test.

- `trajectory/binomial_test_ce_masd.py`
  CE `MASD_norm` binomial test.

### `LLM_with_history` Tests

- `trajectory/binomial_test_drift_llm_with_history.py`
  Human vs. `LLM_with_history` drift comparison.

- `trajectory/binomial_test_cv_llm_with_history.py`
  Human vs. `LLM_with_history` CE-CV and embedding-CV comparison.

- `trajectory/binomial_test_rmssd_masd_llm_with_history.py`
  Human vs. `LLM_with_history` `RMSSD_norm` and `MASD_norm` comparison.

### Additional Tests

- `trajectory/binomial_test_embedding_cv.py`
  TFIDF and SBERT CV binomial tests.

- `trajectory/binomial_test_ce_feature_drift.py`
  CE feature-level drift tests.

- `trajectory/binomial_test_drift_sbert_e5.py`
  E5-based semantic drift comparison.

## 4. Classification Scripts

- `trajectory/run_trajectory_classification_cv20.py`
  Human vs. LLM classification using 20 CE-CV features.
  Also supports `--llm-with-history`.

- `trajectory/run_trajectory_classification_rmssd20.py`
  Human vs. LLM classification using 20 `RMSSD_norm` features.

- `trajectory/run_trajectory_classification_masd20.py`
  Human vs. LLM classification using 20 `MASD_norm` features.

## 5. Visualization Scripts

### CE-CV Difference Plots

- `trajectory/plot_ce_cv_difference_violin.py`
  Produces a single wide CE-CV difference figure across all CE features.

- `trajectory/plot_ce_cv_difference_violin_by_layer.py`
  Produces grouped CE-CV figures by feature family:
  - cognitive
  - emotional
  - stylistic group 1
  - stylistic group 2

  This script supports:
  - regular LLM plots
  - `LLM_with_history` plots
  - direct instance-wise vs. `LLM_with_history` comparison plots

### Drift Difference Plots

- `trajectory/plot_drift_difference_violin.py`
  Produces drift difference visualizations and supports:
  - regular LLM plots
  - `LLM_with_history` plots
  - direct instance-wise vs. `LLM_with_history` comparison plots

### Other Plots

- `trajectory/plot_human_llm_difference_violin.py`
  Produces an additional human-vs-LLM trajectory difference visualization.

## 6. Analysis Utilities

- `trajectory/analyze_feature_importance.py`
  Summarizes classifier feature importance outputs.

## 7. Common Workflows

### Main Experiment

```bash
# 1) CE features
python scripts/features_extraction/batch_analyze_metrics.py
python scripts/features_extraction/remove_outliers_from_combined_merged.py --batch-llm

# 2) CE author-level statistics
python scripts/trajectory/generate_timeseries_stats_from_outliers_removed.py --target llm

# 3) TFIDF / SBERT
python scripts/features_extraction/extract_tfidf_vectors.py
python scripts/features_extraction/extract_sbert_vectors.py
python scripts/features_extraction/merge_features.py

# 4) Drift
python scripts/trajectory/compute_embedding_drift.py
python scripts/trajectory/binomial_test_drift.py --levels LV3

# 5) CE tests and classification
python scripts/trajectory/binomial_test_ce_cv.py --levels LV3
python scripts/trajectory/run_trajectory_classification_cv20.py --level LV3
```

### `LLM_with_history`

```bash
# 1) CE features
python scripts/features_extraction/batch_analyze_metrics.py \
  --llm-with-history \
  --domains academic blogs news

# 2) Outlier handling
# Generate combined_merged_outliers_removed.csv for each provider and domain

# 3) CE author-level statistics
python scripts/trajectory/generate_timeseries_stats_from_outliers_removed.py \
  --target llm_with_history \
  --models DS CL35 G4OM \
  --levels LV3 \
  --domains academic blogs news

# 4) TFIDF / SBERT
python scripts/features_extraction/extract_tfidf_vectors.py \
  --llm-with-history \
  --domains academic blogs news

python scripts/features_extraction/extract_sbert_vectors.py \
  --llm-with-history \
  --domains academic blogs news

python scripts/features_extraction/merge_features.py \
  --llm-with-history \
  --domains academic blogs news

# 5) Drift
python scripts/trajectory/compute_embedding_drift.py \
  --llm-with-history \
  --domains academic blogs news

# 6) Tests
python scripts/trajectory/binomial_test_drift_llm_with_history.py \
  --domains academic blogs news \
  --providers DS CL35 G4OM \
  --levels LV3

python scripts/trajectory/binomial_test_cv_llm_with_history.py \
  --domains academic blogs news \
  --providers DS CL35 G4OM \
  --levels LV3

python scripts/trajectory/binomial_test_rmssd_masd_llm_with_history.py \
  --domains academic blogs news \
  --providers DS CL35 G4OM \
  --levels LV3

python scripts/trajectory/run_trajectory_classification_cv20.py \
  --domains academic blogs news \
  --models DS CL35 G4OM \
  --level LV3 \
  --llm-with-history
```

## 8. Key Intermediate Files

- `combined_merged.csv`
  Raw sample-level CE feature table.

- `combined_merged_outliers_removed.csv`
  Outlier-handled CE table for LLM or `LLM_with_history`.

- `author_timeseries_stats_merged.csv`
  Author-level CE temporal statistics.

- `combined_with_embeddings.csv`
  Sample-level CE + TFIDF + SBERT table.

- `ce_drift.csv`, `tfidf_drift.csv`, `sbert_drift.csv`
  Drift inputs.

- `author_timeseries_stats_embeddings.csv`
  Author-level TFIDF and SBERT variability statistics.
