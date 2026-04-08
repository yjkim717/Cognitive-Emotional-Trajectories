# Dataset Summary: Cognitive-Emotional Trajectories

## Overview

| Split | Condition | Levels | Models | Docs | Author-Field Pairs |
|-------|-----------|--------|--------|------|--------------------|
| Human | Ground truth | — | — | 6,086 | 412 |
| IW (Instance-Wise) | LLM w/o history | LV1–LV3 | DS, G4B, G12B, LMK, CL35, G4OM | 85,202 | 412 per model/level |
| LH (LLM_with_history) | LLM w/ history | LV3 only | DS, CL35, G4OM | 18,257 | 412 per model |
| **Grand Total** | | | | **109,545** | |

---

## Human Corpus

### Per-Domain Breakdown

| Domain | Authors | Fields | Author-Field Pairs | Total Docs | Docs/Pair (min/mean/max) | Year Range |
|--------|---------|--------|--------------------|------------|--------------------------|------------|
| Academic | 20 | 5 | 100 | 500 | 5 / 5.0 / 5 | 2020–2024 |
| Blogs | 50 | 4 | 195 | 1,901 | 6 / 9.7 / 10 | 2020–2024 |
| News | 36 | 7 | 117 | 3,685 | 10 / 31.5 / 55 | 2012–2022 |
| **Total** | **106** | — | **412** | **6,086** | | |

### Field Labels

| Domain | Fields |
|--------|--------|
| Academic | BIOLOGY, CHEMISTRY, CS, MEDICINE, PHYSICS |
| Blogs | LIFESTYLE, SOCIAL, SPORTS, TECHNOLOGY |
| News | 5years, 6years, 7years, 8years, 9years, 10years, 11years *(HuffPost tenure buckets)* |

---

## LLM Instance-Wise (IW) Corpus

Each LLM document was generated from the same human document independently, without any cross-document context.

### Models and Prompt Levels

| Model | LV1 | LV2 | LV3 | Notes |
|-------|-----|-----|-----|-------|
| DS (DeepSeek) | ✓ | ✓ | ✓ | All levels |
| G4B (Gemma 4B) | ✓ | ✓ | ✓ | All levels |
| G12B (Gemma 12B) | ✓ | ✓ | ✓ | All levels |
| LMK (LLaMA-3-8B) | ✓ | ✓ | ✓ | All levels |
| CL35 (Claude 3.5) | — | — | ✓ | LV3 only |
| G4OM (GPT-4o-mini) | — | — | ✓ | LV3 only |

### Document Counts by Level

| Level | Docs | Description |
|-------|------|-------------|
| LV1 | 24,343 | 4 models × (500 academic + 1,901 blogs + ~3,685 news) |
| LV2 | 24,343 | 4 models × (500 academic + 1,901 blogs + ~3,685 news) |
| LV3 | 36,516 | 6 models × (500 academic + 1,901 blogs + ~3,685 news) |
| **Total** | **85,202** | |

### Document Counts by Domain (all levels, all models)

| Domain | Docs |
|--------|------|
| Academic | 7,000 |
| Blogs | 26,614 |
| News | 51,588 |
| **Total** | **85,202** |

---

## LLM with History (LH) Corpus

Each LLM document was generated with access to the author's prior writing history as context, simulating longitudinal consistency. LH uses only LV3 prompting and three models.

### Models

| Model | Level | Domains |
|-------|-------|---------|
| DS (DeepSeek) | LV3 | academic, blogs, news |
| CL35 (Claude 3.5) | LV3 | academic, blogs, news |
| G4OM (GPT-4o-mini) | LV3 | academic, blogs, news |

### Document Counts

| Model | Academic | Blogs | News | Subtotal |
|-------|----------|-------|------|----------|
| DS | 500 | 1,901 | 3,684 | 6,085 |
| CL35 | 500 | 1,901 | 3,685 | 6,086 |
| G4OM | 500 | 1,901 | 3,685 | 6,086 |
| **Total** | **1,500** | **5,703** | **11,054** | **18,257** |

> Note: DS/news has 3,684 (one fewer than human's 3,685) due to a single extraction failure at generation time (~0.03% dropout).

---

## Experimental Setup

### Author-Level Analysis (Paired Comparisons)

For both CE variability (RQ2) and embedding drift (RQ1), comparisons are made at the **author-field pair** level:

| Experiment | Pairs | Method |
|------------|-------|--------|
| Drift binomial test | 412 | Human total drift vs LLM total drift per author, common year pairs only |
| CE CV binomial test | 412 | Human CE CV vs LLM CE CV per author |
| CE RMSSD/MASD test | 412 | Normalized robustness metrics per author |
| CE ML classification | 1,648 rows (412 × 4 splits) | GroupKFold(20), features = author-level CE stats |

### CE Feature Processing

| Step | Input | Output | Notes |
|------|-------|--------|-------|
| Sample extraction | raw `.txt` | `combined_merged.csv` | 20 CE features per document |
| Outlier removal | `combined_merged.csv` | `combined_merged_outliers_removed.csv` | IQR-based, replaces outliers with NaN (no rows dropped) |
| CE variability | `combined_merged_outliers_removed.csv` | `author_timeseries_stats_merged.csv` | CV, RMSSD, MASD per author |
| CE drift | `combined_merged.csv` (raw) | `ce_drift.csv` | Raw values, no outlier removal, L2 per consecutive year |

### Embedding Feature Processing

| Feature | Dimensions | Source | Used For |
|---------|-----------|--------|----------|
| CE (Big5 + sentiment + style) | 20D | `combined_merged.csv` | Both CE variability and CE drift |
| TF-IDF | 10D | `combined_with_embeddings.csv` | Drift only |
| SBERT | 384D | `combined_with_embeddings.csv` | Drift only |

---

## Sample Counts: Known Discrepancy

The paper reports **86,413 LLM-generated documents** for the IW corpus; the files on disk total **85,202**. The difference (1,211, ~1.4%) reflects documents generated successfully by the LLM but lost during downstream extraction (parsing failures, encoding issues). Outlier removal does **not** reduce row counts — only feature values are replaced with NaN.

The paper also contains a typographic error: blogs are listed as "2,901 posts" but the correct count is **1,901** (confirmed: 500 + 1,901 + 3,685 = 6,086 human documents).
