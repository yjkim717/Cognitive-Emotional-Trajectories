# Pipeline Notes

This note documents the full CE (Cognitive-Emotional) pipeline in this project, from raw `txt` files to:

- `trajectory_features_combined.csv`
- CE binomial tests
- CE ML validation / classifiers

The goal is to make the following points explicit:

- What the raw inputs are
- Which script is used at each step
- What each step produces
- What the upstream and downstream dependencies are
- Where outlier removal happens
- Which results are used in the main paper vs. the appendix

---

## 0. One-line overview

The CE pipeline can be summarized as:

`raw txt`
-> `big5.csv` + `nela_merged.csv`
-> `combined_merged.csv`
-> `combined_merged_outliers_removed.csv` (for LLM)
-> `author_timeseries_stats_merged.csv`
-> `ce_trajectory_features.csv`
-> `trajectory_features_combined.csv`
-> `binomial test` / `ML validation`

In practice:

- Human data generally flows from `combined_merged.csv`
- LLM mainline data goes through `combined_merged_outliers_removed.csv` before downstream CE statistics
- this outlier-removed file should be understood as the LLM sample-level CE table after robustifying extreme feature values

---

## 1. Raw input data

### 1.1 Human raw text

Directories:

- `dataset/human/academic/`
- `dataset/human/blogs/`
- `dataset/human/news/`

Each sample is an original human-written text that will later be converted into CE features.

### 1.2 LLM raw text

Directories:

- `dataset/llm/academic/`
- `dataset/llm/blogs/`
- `dataset/llm/news/`

The filenames encode temporal information, model identity, prompt level, etc. Later scripts parse the filenames to reconstruct time order.

---

## 2. Step 1: Extract sample-level CE features from raw text

### 2.1 Main scripts

- [`scripts/features_extraction/batch_analyze_metrics.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/features_extraction/batch_analyze_metrics.py)

Single-run / older entry point:

- [`scripts/features_extraction/analyze_metrics.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/features_extraction/analyze_metrics.py)

### 2.2 What this step does

This step extracts two CE-related feature groups for each text sample:

1. Big Five proxy features
2. Merged NELA features

These are computed via:

- `utils.metric_big5.extract_big5_features`
- `utils.metric_nela_merged.extract_nela_features_merged`

### 2.3 Outputs

For each split, this step produces:

- `big5.csv`
- `nela_merged.csv`

These two tables are then merged into:

- `combined_merged.csv`

### 2.4 File locations

Human:

- `dataset/process/human/{domain}/big5.csv`
- `dataset/process/human/{domain}/nela_merged.csv`
- `dataset/process/human/{domain}/combined_merged.csv`

LLM:

- `dataset/process/LLM/{provider}/{level}/{domain}/big5.csv`
- `dataset/process/LLM/{provider}/{level}/{domain}/nela_merged.csv`
- `dataset/process/LLM/{provider}/{level}/{domain}/combined_merged.csv`

### 2.5 Why this step matters

`combined_merged.csv` is the main sample-level CE table.

It contains:

- metadata such as `filename`, `path`, `label`, `domain`, `field`, `author_id`, `model`, `level`
- CE numeric features such as Big Five, sentiment, VADER, readability, and POS-ratio features

---

## 3. Step 2: Apply outlier removal to LLM CE features

### 3.1 Script

- [`scripts/features_extraction/remove_outliers_from_combined_merged.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/features_extraction/remove_outliers_from_combined_merged.py)

### 3.2 What this step does

This step is specifically for LLM samples:

- input: `combined_merged.csv`
- group by `model + domain`
- detect outliers feature-wise using the `IQR` rule
- replace outlier values with `NaN`

Interpretation:

- this is a sample-level cleanup step for LLM CE features
- the purpose is to prevent a small number of extreme LLM generations from disproportionately affecting downstream author-level temporal statistics
- the script does **not** drop the whole sample row; it masks only the feature entries flagged as outliers

### 3.3 Output

- `combined_merged_outliers_removed.csv`

### 3.4 File locations

- `dataset/process/LLM/{provider}/{level}/{domain}/combined_merged_outliers_removed.csv`

### 3.5 Key conclusion

For the LLM splits that actually exist in this repository, outlier removal was applied.

So the LLM CE pipeline is best understood as:

`combined_merged.csv`
-> `combined_merged_outliers_removed.csv`
-> downstream author-level CE statistics

Conceptually:

- `combined_merged.csv` = raw sample-level CE feature table
- `combined_merged_outliers_removed.csv` = LLM sample-level CE feature table after outlier handling
- downstream CE statistics such as `CV`, `RMSSD`, and `MASD` should be understood as operating on this robustified LLM feature table in the mainline interpretation

### 3.6 What about Human data

In the current mainline pipeline:

- outlier removal is mainly applied to LLM data
- Human data generally continues from `combined_merged.csv`

---

## 4. Step 3: Generate author-level CE time-series statistics

### 4.1 Mainline statistics script

- [`scripts/trajectory/generate_timeseries_stats_merged.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/generate_timeseries_stats_merged.py)

### 4.2 Outlier-removed branch script

- [`scripts/trajectory/generate_timeseries_stats_from_outliers_removed.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/generate_timeseries_stats_from_outliers_removed.py)

### 4.3 What this step does

This step:

1. parses `year` and `item_index` from filenames
2. groups samples by `author_id + field`
3. computes the following time-series statistics for each CE feature:
   - `variance`
   - `cv`
   - `rmssd`
   - `masd`
   - `rmssd_norm`
   - `masd_norm`

### 4.4 Output

- `author_timeseries_stats_merged.csv`

### 4.5 File locations

Human:

- `dataset/process/human/{domain}/author_timeseries_stats_merged.csv`

LLM:

- `dataset/process/LLM/{provider}/{level}/{domain}/author_timeseries_stats_merged.csv`

### 4.6 Upstream / downstream relationship

Inputs:

- Human: `combined_merged.csv`
- LLM: mainline should be understood as `combined_merged_outliers_removed.csv`

Output:

- `author_timeseries_stats_merged.csv`

### 4.7 Important columns in this table

For each CE feature, this table includes:

- `*_variance`
- `*_cv`
- `*_rmssd`
- `*_masd`
- `*_rmssd_norm`
- `*_masd_norm`

This is the author-level CE temporal variability table.

---

## 5. Step 4: Compute CE trajectory geometry

### 5.1 Main script

- [`scripts/trajectory/compute_ce_trajectory_features.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/compute_ce_trajectory_features.py)

### 5.2 No-zscore branch

- [`scripts/trajectory/compute_ce_geometry_features_no_zscore.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/compute_ce_geometry_features_no_zscore.py)

### 5.3 What this step does

This step:

1. reads the 20 CE features from the sample-level CE table
2. computes yearly centroids for each `author + year`
3. forms a CE trajectory over years
4. computes geometry features:
   - `mean_distance`
   - `std_distance`
   - `net_displacement`
   - `path_length`
   - `tortuosity`
   - `direction_consistency`
   - `n_years`

### 5.4 Main output

- `ce_trajectory_features.csv`

### 5.5 No-zscore appendix output

- `ce_geometry_features_no_zscore.csv`

### 5.6 Upstream / downstream relationship

Input:

- `combined_merged.csv` or the corresponding CE sample-level table

Outputs:

- `ce_trajectory_features.csv`
- `ce_geometry_features_no_zscore.csv`

---

## 6. Step 5: Build the unified author-level feature table `trajectory_features_combined.csv`

### 6.1 Script

- [`scripts/trajectory/build_combined_trajectory_features.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/build_combined_trajectory_features.py)

### 6.2 What this step does

This step merges multiple author-level tables by key.

Base table:

- `author_timeseries_stats_merged.csv`

Then it merges in:

- `ce_trajectory_features.csv`
- `tfidf_trajectory_features.csv`
- `sbert_trajectory_features.csv`

### 6.3 Output

- `trajectory_features_combined.csv`

### 6.4 File locations

Human:

- `dataset/process/human/{domain}/trajectory_features_combined.csv`

LLM:

- `dataset/process/LLM/{provider}/{level}/{domain}/trajectory_features_combined.csv`

### 6.5 What this table contains for CE

For CE, this table includes two main blocks:

1. CE temporal variability
   - `*_cv`
   - `*_rmssd`
   - `*_masd`
   - `*_variance`
   - `*_rmssd_norm`
   - `*_masd_norm`

2. CE geometry
   - `ce_mean_distance`
   - `ce_std_distance`
   - `ce_net_displacement`
   - `ce_path_length`
   - `ce_tortuosity`
   - `ce_direction_consistency`
   - `ce_n_years`

### 6.6 Why this table matters

This is the most important unified input table for the CE analysis.

The CE:

- binomial tests
- ML validation / classifiers

mainly read their inputs from this table.

---

## 7. CE binomial tests

### 7.1 CE-CV binomial test

Script:

- [`scripts/trajectory/binomial_test_ce_cv.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/binomial_test_ce_cv.py)

Input:

- `trajectory_features_combined.csv`

Columns used:

- the 20 `*_cv` columns

Output:

- CE-CV comparison result CSVs
- corresponding markdown summaries / result files

Paper mapping:

- Main paper Table 2
- Main paper RQ2 CE results

### 7.2 CE-RMSSD binomial test

Script:

- [`scripts/trajectory/binomial_test_ce_rmssd.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/binomial_test_ce_rmssd.py)

Input:

- `trajectory_features_combined.csv`

Columns used:

- `*_rmssd` and/or normalized variants

Paper mapping:

- Appendix H robustness / validity analyses

### 7.3 CE-MASD binomial test

Script:

- [`scripts/trajectory/binomial_test_ce_masd.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/binomial_test_ce_masd.py)

Input:

- `trajectory_features_combined.csv`

Columns used:

- `*_masd` and/or normalized variants

Paper mapping:

- Appendix H robustness / validity analyses

### 7.4 Summary

There is not just one CE binomial test. There are three operator families:

- `CV`
- `RMSSD`
- `MASD`

Main paper:

- emphasizes `CV`

Appendix:

- adds `RMSSD`
- adds `MASD`
- especially normalized robustness variants

---

## 8. CE ML validation / classifiers

### 8.1 CE-CV classifier

Script:

- [`scripts/trajectory/run_trajectory_classification_cv20.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/run_trajectory_classification_cv20.py)

Input:

- `trajectory_features_combined.csv`

Columns used:

- the 20 `*_cv` columns

Task:

- classify Human vs. LLM trajectories using CE temporal variability patterns

Paper mapping:

- Main paper Table 3
- Main paper Table 4

### 8.2 CE-RMSSD classifier

Script:

- [`scripts/trajectory/run_trajectory_classification_rmssd20.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/run_trajectory_classification_rmssd20.py)

Input:

- `trajectory_features_combined.csv`

Columns used:

- `*_rmssd` and/or normalized variants

Paper mapping:

- Appendix H.3 / classifier robustness

### 8.3 CE-MASD classifier

Script:

- [`scripts/trajectory/run_trajectory_classification_masd20.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/run_trajectory_classification_masd20.py)

Input:

- `trajectory_features_combined.csv`

Columns used:

- `*_masd` and/or normalized variants

Paper mapping:

- Appendix H.3 / classifier robustness

### 8.4 Summary

The CE ML validation also has three parallel operator families:

- `CV` classifier
- `RMSSD` classifier
- `MASD` classifier

Main paper:

- emphasizes `CV`

Appendix:

- includes `RMSSD`
- includes `MASD`

---

## 9. What appears in the main paper vs. the appendix

### 9.1 Main paper

The CE results emphasized in the main paper are:

- CE-CV binomial test
- CE-CV classifier / ML validation

Conceptually:

- same 20-dimensional CE feature set
- `CV` used as the main temporal variability operator

### 9.2 Appendix / robustness

The appendix adds:

- CE-RMSSD binomial test
- CE-MASD binomial test
- CE-RMSSD classifier
- CE-MASD classifier
- normalized robustness variants such as `rmssd_norm` and `masd_norm`

One-line memory aid:

**The main paper foregrounds CV; the appendix uses RMSSD and MASD to show robustness.**

---

## 10. Difference between `author_timeseries_stats_merged.csv` and `trajectory_features_combined.csv`

### `author_timeseries_stats_merged.csv`

This is:

- the author-level table containing CE temporal statistics only

Main contents:

- `*_cv`
- `*_rmssd`
- `*_masd`
- `*_variance`
- `*_rmssd_norm`
- `*_masd_norm`

### `trajectory_features_combined.csv`

This is:

- built on top of `author_timeseries_stats_merged.csv`
- further merged with CE / TFIDF / SBERT geometry features
- the unified author-level feature table

So:

- if you only care about CE temporal variability, `author_timeseries_stats_merged.csv` is already sufficient in principle
- but the project’s mainline tests and classifiers typically read from `trajectory_features_combined.csv`

---

## 11. Full CE upstream/downstream dependency map

### 11.1 Sample-level CE features

Inputs:

- `dataset/human/.../*.txt`
- `dataset/llm/.../*.txt`

Script:

- `batch_analyze_metrics.py`

Outputs:

- `big5.csv`
- `nela_merged.csv`
- `combined_merged.csv`

### 11.2 LLM outlier removal

Input:

- `combined_merged.csv`

Script:

- `remove_outliers_from_combined_merged.py`

Output:

- `combined_merged_outliers_removed.csv`

### 11.3 Author-level CE variability

Inputs:

- `combined_merged.csv`
- or `combined_merged_outliers_removed.csv`

Scripts:

- `generate_timeseries_stats_merged.py`
- `generate_timeseries_stats_from_outliers_removed.py`

Output:

- `author_timeseries_stats_merged.csv`

### 11.4 Author-level CE geometry

Input:

- `combined_merged.csv`

Scripts:

- `compute_ce_trajectory_features.py`
- `compute_ce_geometry_features_no_zscore.py`

Outputs:

- `ce_trajectory_features.csv`
- `ce_geometry_features_no_zscore.csv`

### 11.5 Unified author-level feature table

Inputs:

- `author_timeseries_stats_merged.csv`
- `ce_trajectory_features.csv`
- `tfidf_trajectory_features.csv`
- `sbert_trajectory_features.csv`

Script:

- `build_combined_trajectory_features.py`

Output:

- `trajectory_features_combined.csv`

### 11.6 Final validation

Input:

- `trajectory_features_combined.csv`

Scripts:

- `binomial_test_ce_cv.py`
- `binomial_test_ce_rmssd.py`
- `binomial_test_ce_masd.py`
- `run_trajectory_classification_cv20.py`
- `run_trajectory_classification_rmssd20.py`
- `run_trajectory_classification_masd20.py`

Outputs:

- CE binomial test results
- CE classifier / ML validation results

---

## 12. Final takeaway

If you only want one sentence to remember the CE pipeline:

**We first extract Big Five and merged NELA features from raw text to build `combined_merged.csv`; for LLM data we then apply sample-level IQR-based outlier handling to mask extreme CE feature values and produce `combined_merged_outliers_removed.csv`; next we compute author-level CE temporal statistics and CE trajectory geometry, merge them into `trajectory_features_combined.csv`, and finally run binomial tests and ML validation using CV / RMSSD / MASD, with CV emphasized in the main paper and RMSSD/MASD used mainly for appendix robustness analyses.**

---

# Drift Structure Notes

This section documents the drift pipeline and how it maps to the paper's RQ1 results.

Compared with the CE variability pipeline, the drift pipeline is much simpler:

- build a representation space
- aggregate yearly author vectors
- compute adjacent-year Euclidean distance
- sum drift over common year pairs
- run paired binomial tests

---

## 13. One-line overview of the drift pipeline

The drift pipeline can be summarized as:

`raw txt`
-> CE / TFIDF / SBERT representations
-> yearly author centroids
-> adjacent-year L2 / Euclidean drift
-> `ce_drift.csv` / `tfidf_drift.csv` / `sbert_drift.csv`
-> paired Human-vs-LLM comparison
-> binomial test
-> paper Table 1 / Figure 2 / RQ1 text

---

## 14. Representation spaces used for drift

The paper computes drift in three spaces:

1. `CE` (cognitive-emotional feature space)
2. `TFIDF` (lexical space)
3. `SBERT` (semantic space)

These correspond directly to the paper's RQ1 setup:

- CE drift
- TF-IDF drift
- SBERT drift

Conceptually:

- CE drift captures movement in interpretable cognitive-emotional features
- TFIDF drift captures lexical movement
- SBERT drift captures semantic movement

---

## 15. Step 1: Build the upstream feature tables for drift

### 15.1 CE upstream table

Script:

- [`scripts/features_extraction/batch_analyze_metrics.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/features_extraction/batch_analyze_metrics.py)

Main output:

- `combined_merged.csv`

This is the CE sample-level table and serves as the drift input for CE space.

### 15.2 TFIDF and SBERT upstream tables

Scripts:

- `scripts/features_extraction/extract_tfidf_vectors.py`
- `scripts/features_extraction/extract_sbert_vectors.py`
- [`scripts/features_extraction/merge_features.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/features_extraction/merge_features.py)

Main outputs:

- `tfidf_vectors.csv`
- `sbert_vectors.csv`
- `combined_with_embeddings.csv`

This gives the sample-level table for the embedding-based drift spaces.

### 15.3 Important setup note

For drift, the upstream inputs are not all the same:

- CE drift uses `combined_merged.csv`
- TFIDF / SBERT drift use `combined_with_embeddings.csv`

This matters when checking paper results against the data.

---

## 16. Step 2: Build yearly author trajectories

Before drift is computed, the pipeline does not compare documents directly.
Instead, it first builds author-year centroids.

For each author:

1. group all documents in the same year
2. average them within the year
3. obtain one vector per year
4. order these yearly vectors chronologically

This produces an author trajectory over time.

This step corresponds directly to the paper's trajectory construction in Section 4.1.2.

---

## 17. Step 3: Compute drift from adjacent-year Euclidean distance

### 17.1 Main script

- [`scripts/trajectory/compute_embedding_drift.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/compute_embedding_drift.py)

### 17.2 What the script does

For each author trajectory and each representation space:

- sort yearly centroids by time
- compute Euclidean distance between adjacent years

That is:

`drift_t = ||x_(t+1) - x_t||_2`

Then all year-pair drifts are stored explicitly.

### 17.2 Normalization note

For the main drift pipeline:

- no per-author z-score normalization is applied
- no second normalization step is applied during drift computation
- drift is computed directly from the yearly vectors using raw L2 distance

The only caveat is that SBERT embeddings may already be L2-normalized upstream at extraction time, but the drift step itself does not apply any additional normalization.

### 17.3 Outputs

- `ce_drift.csv`
- `tfidf_drift.csv`
- `sbert_drift.csv`

### 17.4 File locations

Human:

- `dataset/process/human/{domain}/ce_drift.csv`
- `dataset/process/human/{domain}/tfidf_drift.csv`
- `dataset/process/human/{domain}/sbert_drift.csv`

LLM:

- `dataset/process/LLM/{provider}/{level}/{domain}/ce_drift.csv`
- `dataset/process/LLM/{provider}/{level}/{domain}/tfidf_drift.csv`
- `dataset/process/LLM/{provider}/{level}/{domain}/sbert_drift.csv`

### 17.5 Relation to the paper

This step corresponds directly to Section 4.2.1 in the paper:

- global evolution is measured with an L2 drift operator
- total drift is the sum of per-year drift values

---

## 18. What is stored in the drift CSV files

The drift CSVs are year-pair level tables.

Typical columns include:

- `author_id`
- `domain`
- `field`
- `label`
- `model`
- `level`
- `rep_space`
- `year_from`
- `year_to`
- `drift`

So these files are not final test outputs yet.
They are the direct inputs to the drift binomial tests.

---

## 19. Step 4: Run paired Human-vs-LLM drift comparisons

### 19.1 Main scripts

- [`scripts/trajectory/binomial_test_drift_ce.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/binomial_test_drift_ce.py)
- [`scripts/trajectory/binomial_test_drift_tfidf.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/binomial_test_drift_tfidf.py)
- [`scripts/trajectory/binomial_test_drift_sbert.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/trajectory/binomial_test_drift_sbert.py)

### 19.2 What these scripts do

For each Human author and matched LLM author:

1. match on `author_id + field + domain`
2. find common year pairs `(year_from, year_to)`
3. keep only those common transitions
4. sum the drift values over those common transitions
5. compare:
   - `Human total drift`
   - `LLM total drift`

This yields a binary outcome per author:

- `Human total drift > LLM total drift`
- or not

### 19.3 Important setup note

This common-year-pair restriction is crucial.

The scripts do not simply sum every available drift value.
They only compare drift on year transitions that both Human and matched LLM have in common.

This is one of the most important reasons paper numbers can differ if someone reimplements the test incorrectly.

---

## 20. Step 5: Run the binomial test

After the paired author-level comparisons are built, each script performs a one-sided binomial test:

- `H0: P(Human > LLM) = 0.5`
- `H1: P(Human > LLM) > 0.5`

The outputs include:

- number of Human wins
- win rate
- p-value

These are exactly the quantities reported in the paper for drift.

---

## 21. How the drift results map to the paper

The paper's main drift results are directly tied to these CSVs and scripts.

### 21.1 Paper Table 1

Mapped inputs:

- `sbert_drift.csv` -> SBERT / semantic drift rows
- `tfidf_drift.csv` -> TF-IDF / lexical drift rows
- `ce_drift.csv` -> Cog-Emo drift rows

Mapped scripts:

- `binomial_test_drift_sbert.py`
- `binomial_test_drift_tfidf.py`
- `binomial_test_drift_ce.py`

### 21.2 Paper Figure 2

Figure 2 visualizes the Human-minus-LLM drift differences.
The underlying values come from the same drift comparison pipeline used for the binomial tests.

### 21.3 Paper Section 5.1

The main findings reported there are exactly the drift comparison conclusions:

- TFIDF: LLMs show greater lexical drift
- SBERT: Humans show greater semantic drift
- CE: Humans show greater cognitive-emotional drift

---

## 22. Result files that summarize the drift pipeline

The most useful summary files in the repository are:

- [`results/drift/RQ1_Drift_Result.md`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/results/drift/RQ1_Drift_Result.md)
- [`results/Combined_Results.md`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/results/Combined_Results.md)

These files summarize the same numbers that are reported in the paper.

---

## 23. Important setup notes for drift

### 23.1 Drift is trajectory-level, not document-level

The pipeline does not compute drift by comparing one document directly to the next.
It first aggregates documents into yearly author centroids.

### 23.2 The distance is Euclidean / L2

The drift operator is Euclidean distance between adjacent yearly vectors.

### 23.3 The three spaces are not numerically comparable by raw scale

- CE is a handcrafted 20D feature space
- TFIDF is a reduced lexical embedding space
- SBERT is a semantic embedding space

So the main comparison is not absolute drift magnitude across spaces, but:

- Human vs LLM within the same space
- win rates and significance patterns

### 23.4 The final test input is always the drift CSV

For drift, the most important final inputs are simply:

- `ce_drift.csv`
- `tfidf_drift.csv`
- `sbert_drift.csv`

Unlike the CE variability pipeline, drift does not require a unified `trajectory_features_combined.csv` table for its main paper result.

### 23.5 Robustness and ablation note for TFIDF / SBERT extraction

There are two different kinds of robustness checks in the repository, and they should not be confused:

1. prompt-level robustness
2. embedding-model / encoder robustness

For prompt-level robustness:

- both SBERT drift and TFIDF drift were tested across `LV1`, `LV2`, and `LV3`

For embedding-model robustness:

- SBERT has an additional alternative encoder branch
- the main SBERT extractor is:
  - [`scripts/features_extraction/extract_sbert_vectors.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/features_extraction/extract_sbert_vectors.py)
  - model: `all-MiniLM-L6-v2`
- the alternative SBERT extractor is:
  - [`scripts/features_extraction/extract_sbert_e5_vectors.py`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/scripts/features_extraction/extract_sbert_e5_vectors.py)
  - model: `intfloat/e5-large-v2`

This SBERT-E5 branch produces:

- `sbert_e5_vectors.csv`
- `combined_with_embeddings_e5.csv`
- `sbert_e5_drift.csv`

and is used for encoder-level robustness checks in the drift analysis.

By contrast, there is no parallel alternative TFIDF extractor branch of the same kind in the repository.
So:

- SBERT has both prompt-level robustness and encoder-level robustness
- TFIDF has prompt-level robustness, but not a comparable alternative-extractor ablation branch

Relevant result summaries include:

- [`results/drift/Drift_Robustness_Test_Result.md`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/results/drift/Drift_Robustness_Test_Result.md)
- [`results/Combined_Results.md`](/Users/zhanweicao/Desktop/Cognitive-Emotional-Trajectories/results/Combined_Results.md)

---

## 24. Full drift upstream/downstream dependency map

### 24.1 Upstream representation building

Inputs:

- `dataset/human/.../*.txt`
- `dataset/llm/.../*.txt`

Scripts:

- `batch_analyze_metrics.py`
- `extract_tfidf_vectors.py`
- `extract_sbert_vectors.py`
- `merge_features.py`

Outputs:

- `combined_merged.csv`
- `combined_with_embeddings.csv`

### 24.2 Drift computation

Inputs:

- CE space: `combined_merged.csv`
- TFIDF / SBERT spaces: `combined_with_embeddings.csv`

Script:

- `compute_embedding_drift.py`

Outputs:

- `ce_drift.csv`
- `tfidf_drift.csv`
- `sbert_drift.csv`

### 24.3 Statistical testing

Inputs:

- `ce_drift.csv`
- `tfidf_drift.csv`
- `sbert_drift.csv`

Scripts:

- `binomial_test_drift_ce.py`
- `binomial_test_drift_tfidf.py`
- `binomial_test_drift_sbert.py`

Outputs:

- detailed comparison CSVs
- binomial test result CSVs
- markdown summaries used to interpret the paper results

---

## 25. Final takeaway for drift

If you only want one sentence to remember the drift pipeline:

**We first build CE / TFIDF / SBERT representations from raw text, aggregate them into yearly author centroids, compute adjacent-year Euclidean drift, store the resulting year-pair values in `ce_drift.csv`, `tfidf_drift.csv`, and `sbert_drift.csv`, and then run matched Human-vs-LLM binomial tests on total drift over common year pairs, which directly produce the paper's RQ1 drift results.**
