# LLM with History Pipeline

Reviewer condition: LLMs given incremental context (summaries of previous outputs).  
Data: `dataset/llm/llm_with_history/news/` — 117 authors, news only, DS/CL35/G4OM, LV3.

## Data flow

1. **Per sample (3685 per model)**  
   Each row has: **20 CE** + **10 TFIDF** + **384 SBERT** (after merge).

2. **Per author (117)**  
   Each author has:
   - **Drift**: `ce_drift.csv`, `tfidf_drift.csv`, `sbert_drift.csv` (year-to-year L2 drift, total drift per author).
   - **Variance (CE)**: `author_timeseries_stats_merged.csv` — variance, CV, RMSSD, MASD, etc. per author.

## Commands (run from project root)

```bash
# 1. CE (20 features): Big Five + NELA from txt
python scripts/features_extraction/batch_analyze_metrics.py --llm-with-history

# 2. TFIDF (10D) and SBERT (384D)
python scripts/features_extraction/extract_tfidf_vectors.py --llm-with-history
python scripts/features_extraction/extract_sbert_vectors.py --llm-with-history

# 3. Merge → combined_with_embeddings.csv (each sample: CE + TFIDF + SBERT)
python scripts/features_extraction/merge_features.py --llm-with-history

# 4. Drift (per author, per space)
python scripts/trajectory/compute_embedding_drift.py --llm-with-history

# 5. Variance for CE (per author: CV, RMSSD, MASD, etc.)
python scripts/trajectory/generate_timeseries_stats_merged.py --target llm_with_history
```

## Output layout

```
dataset/process/LLM_with_history/
├── DS/LV3/news/
│   ├── combined_merged.csv          # 20 CE per sample
│   ├── combined_with_embeddings.csv # + tfidf_1..10, sbert_1..384
│   ├── ce_drift.csv                 # drift per year-pair
│   ├── tfidf_drift.csv
│   ├── sbert_drift.csv
│   └── author_timeseries_stats_merged.csv  # CE variance per author
├── CL35/LV3/news/  (same)
└── G4OM/LV3/news/  (same)
```

## Optional: SBERT/TFIDF author-level variance

If you need variance metrics in embedding space (e.g. CV over years in SBERT/TFIDF), use:

- `scripts/trajectory/generate_embedding_timeseries_stats.py`  
  (add `--llm-with-history` or point it at `LLM_with_history` if needed.)
