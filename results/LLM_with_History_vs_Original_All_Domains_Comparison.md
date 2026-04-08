# Instance-wise LLM vs LLM_with_history — Full Results Comparison

**Date:** 2026-04-07  
**Models:** DS, CL35, G4OM — Level LV3  
**Domains:** academic + blogs + news (pooled)  
**N authors:** 412 paired (100 academic + 195 blogs + 117 news)  

Both conditions use identical methodology, same 412 author-field pairs, same human baseline.

---

## Part A: Instance-wise LLM (Original)

Human authors compared against LLMs generating each text independently, with no memory of prior outputs.

### A1. CE CV Binomial Test

**H₁:** P(Human CV > LLM CV) > 0.5 | FDR (BH), α = 0.05 | N = 412 per feature per model

| Model | Sig. / Total | % Sig. | Mean Win Rate |
|-------|-------------|--------|---------------|
| DS    | 14 / 20     | 70.0%  | 0.662         |
| CL35  | 16 / 20     | 80.0%  | 0.679         |
| G4OM  | 15 / 20     | 75.0%  | 0.707         |

| Feature | DS wr | DS | CL35 wr | CL35 | G4OM wr | G4OM |
|---------|-------|----|---------|------|---------|------|
| Agreeableness_cv          | 0.840 | ✅ | 0.827 | ✅ | 0.849 | ✅ |
| Conscientiousness_cv      | 0.750 | ✅ | 0.638 | ✅ | 0.872 | ✅ |
| Extraversion_cv           | 0.538 | ❌ | 0.622 | ✅ | 0.731 | ✅ |
| Neuroticism_cv            | 0.920 | ✅ | 0.901 | ✅ | 0.862 | ✅ |
| Openness_cv               | 0.696 | ✅ | 0.734 | ✅ | 0.782 | ✅ |
| average_word_length_cv    | 0.747 | ✅ | 0.490 | ❌ | 0.663 | ✅ |
| avg_sentence_length_cv    | 0.920 | ✅ | 0.946 | ✅ | 0.978 | ✅ |
| content_word_ratio_cv     | 0.804 | ✅ | 0.779 | ✅ | 0.821 | ✅ |
| flesch_reading_ease_cv    | 0.292 | ❌ | 0.022 | ❌ | 0.192 | ❌ |
| function_word_ratio_cv    | 0.487 | ❌ | 0.234 | ❌ | 0.583 | ✅ |
| gunning_fog_cv            | 0.885 | ✅ | 0.904 | ✅ | 0.929 | ✅ |
| num_words_cv              | 0.625 | ✅ | 0.962 | ✅ | 1.000 | ✅ |
| polarity_cv               | 0.577 | ✅ | 0.679 | ✅ | 0.635 | ✅ |
| subjectivity_cv           | 0.625 | ✅ | 0.577 | ✅ | 0.532 | ❌ |
| vader_compound_cv         | 0.426 | ❌ | 0.974 | ✅ | 0.692 | ✅ |
| vader_neg_cv              | 0.545 | ❌ | 0.455 | ❌ | 0.378 | ❌ |
| vader_neu_cv              | 0.622 | ✅ | 0.564 | ✅ | 0.484 | ❌ |
| vader_pos_cv              | 0.663 | ✅ | 0.760 | ✅ | 0.657 | ✅ |
| verb_ratio_cv             | 0.513 | ❌ | 0.603 | ✅ | 0.510 | ❌ |
| word_diversity_cv         | 0.769 | ✅ | 0.904 | ✅ | 0.997 | ✅ |

---

### A2. CE RMSSD_norm Binomial Test (Robustness)

| Model | Sig. / Total | % Sig. | Mean Win Rate |
|-------|-------------|--------|---------------|
| DS    | 16 / 20     | 80.0%  | 0.635         |
| CL35  | 16 / 20     | 80.0%  | 0.660         |
| G4OM  | 14 / 20     | 70.0%  | 0.690         |

| Feature | DS wr | DS | CL35 wr | CL35 | G4OM wr | G4OM |
|---------|-------|----|---------|------|---------|------|
| Agreeableness_rmssd_norm          | 0.779 | ✅ | 0.779 | ✅ | 0.827 | ✅ |
| Conscientiousness_rmssd_norm      | 0.689 | ✅ | 0.599 | ✅ | 0.837 | ✅ |
| Extraversion_rmssd_norm           | 0.551 | ✅ | 0.596 | ✅ | 0.699 | ✅ |
| Neuroticism_rmssd_norm            | 0.875 | ✅ | 0.856 | ✅ | 0.811 | ✅ |
| Openness_rmssd_norm               | 0.673 | ✅ | 0.712 | ✅ | 0.737 | ✅ |
| average_word_length_rmssd_norm    | 0.660 | ✅ | 0.497 | ❌ | 0.638 | ✅ |
| avg_sentence_length_rmssd_norm    | 0.885 | ✅ | 0.923 | ✅ | 0.958 | ✅ |
| content_word_ratio_rmssd_norm     | 0.766 | ✅ | 0.740 | ✅ | 0.776 | ✅ |
| flesch_reading_ease_rmssd_norm    | 0.276 | ❌ | 0.016 | ❌ | 0.212 | ❌ |
| function_word_ratio_rmssd_norm    | 0.510 | ❌ | 0.240 | ❌ | 0.548 | ❌ |
| gunning_fog_rmssd_norm            | 0.840 | ✅ | 0.862 | ✅ | 0.888 | ✅ |
| num_words_rmssd_norm              | 0.606 | ✅ | 0.936 | ✅ | 1.000 | ✅ |
| polarity_rmssd_norm               | 0.567 | ✅ | 0.663 | ✅ | 0.647 | ✅ |
| subjectivity_rmssd_norm           | 0.567 | ✅ | 0.574 | ✅ | 0.503 | ❌ |
| vader_compound_rmssd_norm         | 0.423 | ❌ | 0.978 | ✅ | 0.702 | ✅ |
| vader_neg_rmssd_norm              | 0.551 | ✅ | 0.452 | ❌ | 0.391 | ❌ |
| vader_neu_rmssd_norm              | 0.583 | ✅ | 0.561 | ✅ | 0.500 | ❌ |
| vader_pos_rmssd_norm              | 0.641 | ✅ | 0.779 | ✅ | 0.654 | ✅ |
| verb_ratio_rmssd_norm             | 0.506 | ❌ | 0.564 | ✅ | 0.487 | ❌ |
| word_diversity_rmssd_norm         | 0.744 | ✅ | 0.875 | ✅ | 0.994 | ✅ |

---

### A3. CE MASD_norm Binomial Test (Robustness)

| Model | Sig. / Total | % Sig. | Mean Win Rate |
|-------|-------------|--------|---------------|
| DS    | 14 / 20     | 70.0%  | 0.619         |
| CL35  | 16 / 20     | 80.0%  | 0.646         |
| G4OM  | 14 / 20     | 70.0%  | 0.683         |

| Feature | DS wr | DS | CL35 wr | CL35 | G4OM wr | G4OM |
|---------|-------|----|---------|------|---------|------|
| Agreeableness_masd_norm          | 0.737 | ✅ | 0.747 | ✅ | 0.792 | ✅ |
| Conscientiousness_masd_norm      | 0.657 | ✅ | 0.580 | ✅ | 0.811 | ✅ |
| Extraversion_masd_norm           | 0.529 | ❌ | 0.583 | ✅ | 0.705 | ✅ |
| Neuroticism_masd_norm            | 0.853 | ✅ | 0.827 | ✅ | 0.811 | ✅ |
| Openness_masd_norm               | 0.660 | ✅ | 0.683 | ✅ | 0.724 | ✅ |
| average_word_length_masd_norm    | 0.647 | ✅ | 0.474 | ❌ | 0.619 | ✅ |
| avg_sentence_length_masd_norm    | 0.872 | ✅ | 0.904 | ✅ | 0.955 | ✅ |
| content_word_ratio_masd_norm     | 0.756 | ✅ | 0.737 | ✅ | 0.769 | ✅ |
| flesch_reading_ease_masd_norm    | 0.260 | ❌ | 0.016 | ❌ | 0.215 | ❌ |
| function_word_ratio_masd_norm    | 0.519 | ❌ | 0.234 | ❌ | 0.545 | ❌ |
| gunning_fog_masd_norm            | 0.846 | ✅ | 0.872 | ✅ | 0.885 | ✅ |
| num_words_masd_norm              | 0.571 | ✅ | 0.917 | ✅ | 1.000 | ✅ |
| polarity_masd_norm               | 0.564 | ✅ | 0.673 | ✅ | 0.631 | ✅ |
| subjectivity_masd_norm           | 0.554 | ✅ | 0.551 | ✅ | 0.490 | ❌ |
| vader_compound_masd_norm         | 0.413 | ❌ | 0.971 | ✅ | 0.676 | ✅ |
| vader_neg_masd_norm              | 0.519 | ❌ | 0.417 | ❌ | 0.404 | ❌ |
| vader_neu_masd_norm              | 0.583 | ✅ | 0.561 | ✅ | 0.487 | ❌ |
| vader_pos_masd_norm              | 0.587 | ✅ | 0.769 | ✅ | 0.657 | ✅ |
| verb_ratio_masd_norm             | 0.516 | ❌ | 0.580 | ✅ | 0.484 | ❌ |
| word_diversity_masd_norm         | 0.734 | ✅ | 0.830 | ✅ | 0.997 | ✅ |

---

### A4. ML Classification: Human vs Instance-wise LLM

**Classifier:** RandomForestClassifier | **CV:** 5-fold GroupKFold (by author_id)  
**Features:** 20 CE CV features

#### A4a. Combined (Imbalanced, DS+CL35+G4OM)

**N:** 1,648 (412 human + 412×3 LLM)

| Metric       | Mean    | Std     |
|--------------|---------|---------|
| Accuracy     | 0.9363  | ±0.0117 |
| AUC          | 0.9770  | ±0.0085 |
| F1 Score     | 0.8631  | ±0.0283 |
| Human Recall | 0.8082  | ±0.0473 |

**Top 5 feature importances (combined):**

| Rank | Feature                 | Importance |
|------|-------------------------|------------|
| 1    | avg_sentence_length_cv  | 18.72%     |
| 2    | Agreeableness_cv        | 15.87%     |
| 3    | Neuroticism_cv          | 9.31%      |
| 4    | word_diversity_cv       | 7.00%      |
| 5    | gunning_fog_cv          | 6.35%      |

#### A4b. Balanced Per-Model (412 Human vs 412 LLM each)

| Model | Accuracy | AUC | F1 | Human Recall | N |
|-------|----------|-----|----|--------------|---|
| DS    | 0.9064 ± 0.0326 | 0.9632 ± 0.0117 | 0.9058 ± 0.0325 | 0.8980 ± 0.0297 | 824 |
| CL35  | 0.9733 ± 0.0090 | 0.9961 ± 0.0017 | 0.9732 ± 0.0088 | 0.9684 ± 0.0124 | 824 |
| G4OM  | 0.9988 ± 0.0024 | 0.9996 ± 0.0009 | 0.9988 ± 0.0025 | 0.9976 ± 0.0049 | 824 |

**Top 5 feature importances (per model):**

| Rank | DS | CL35 | G4OM |
|------|----|------|------|
| 1 | avg_sentence_length_cv (17.05%) | vader_compound_cv (23.83%) | num_words_cv (35.35%) |
| 2 | Agreeableness_cv (16.97%) | avg_sentence_length_cv (14.99%) | word_diversity_cv (24.70%) |
| 3 | Neuroticism_cv (10.05%) | Agreeableness_cv (8.68%) | avg_sentence_length_cv (10.37%) |
| 4 | gunning_fog_cv (7.44%) | flesch_reading_ease_cv (8.34%) | Agreeableness_cv (7.23%) |
| 5 | average_word_length_cv (4.94%) | Neuroticism_cv (7.84%) | vader_compound_cv (4.27%) |

---

### A5. Drift Binomial Test

**H₁:** P(Human total drift > LLM total drift) > 0.5 | Raw p-values, α = 0.05 | N = 412

| Space | Model | Human Wins | Win Rate | p-value     | Sig |
|-------|-------|-----------|---------|-------------|-----|
| CE    | DS    | 312 / 412 | 0.757   | 9.20e-27    | ✅  |
| CE    | CL35  | 340 / 412 | 0.825   | 4.99e-43    | ✅  |
| CE    | G4OM  | 401 / 412 | 0.973   | 1.24e-103   | ✅  |
| SBERT | DS    | 342 / 412 | 0.830   | 2.17e-44    | ✅  |
| SBERT | CL35  | 351 / 412 | 0.852   | 6.76e-51    | ✅  |
| SBERT | G4OM  | 308 / 412 | 0.748   | 7.95e-25    | ✅  |
| TFIDF | DS    | 135 / 412 | 0.328   | 1.000       | ❌  |
| TFIDF | CL35  | 83 / 412  | 0.201   | 1.000       | ❌  |
| TFIDF | G4OM  | 126 / 412 | 0.306   | 1.000       | ❌  |

**Mean drift per author (CE space):** Human = 686, DS = 498, CL35 = 332, G4OM = 51  
**Mean drift per author (SBERT space):** Human = 3.87, DS = 3.71, CL35 = 3.62, G4OM = 3.63  
**Mean drift per author (TFIDF space):** Human = 0.46, DS = 0.49, CL35 = 0.58, G4OM = 0.51

---

---

## Part B: LLM_with_history

Human authors compared against LLMs that received an incrementally growing summary of their prior outputs before generating each new text (same DS/CL35/G4OM, LV3).

### B1. CE CV Binomial Test

**H₁:** P(Human CV > LLM_with_history CV) > 0.5 | FDR (BH), α = 0.05 | N = 412 per feature per model

| Model | Sig. / Total | % Sig. | Mean Win Rate |
|-------|-------------|--------|---------------|
| DS    | 14 / 20     | 70.0%  | 0.665         |
| CL35  | 16 / 20     | 80.0%  | 0.684         |
| G4OM  | 15 / 20     | 75.0%  | 0.711         |

| Feature | DS wr | DS | CL35 wr | CL35 | G4OM wr | G4OM |
|---------|-------|----|---------|------|---------|------|
| Agreeableness_cv          | 0.856 | ✅ | 0.840 | ✅ | 0.881 | ✅ |
| Conscientiousness_cv      | 0.756 | ✅ | 0.615 | ✅ | 0.853 | ✅ |
| Extraversion_cv           | 0.513 | ❌ | 0.638 | ✅ | 0.728 | ✅ |
| Neuroticism_cv            | 0.862 | ✅ | 0.897 | ✅ | 0.865 | ✅ |
| Openness_cv               | 0.715 | ✅ | 0.779 | ✅ | 0.814 | ✅ |
| average_word_length_cv    | 0.779 | ✅ | 0.484 | ❌ | 0.699 | ✅ |
| avg_sentence_length_cv    | 0.894 | ✅ | 0.939 | ✅ | 0.984 | ✅ |
| content_word_ratio_cv     | 0.792 | ✅ | 0.779 | ✅ | 0.808 | ✅ |
| flesch_reading_ease_cv    | 0.282 | ❌ | 0.019 | ❌ | 0.247 | ❌ |
| function_word_ratio_cv    | 0.545 | ❌ | 0.276 | ❌ | 0.574 | ✅ |
| gunning_fog_cv            | 0.862 | ✅ | 0.901 | ✅ | 0.929 | ✅ |
| num_words_cv              | 0.782 | ✅ | 0.952 | ✅ | 1.000 | ✅ |
| polarity_cv               | 0.603 | ✅ | 0.683 | ✅ | 0.628 | ✅ |
| subjectivity_cv           | 0.577 | ✅ | 0.606 | ✅ | 0.484 | ❌ |
| vader_compound_cv         | 0.426 | ❌ | 0.962 | ✅ | 0.708 | ✅ |
| vader_neg_cv              | 0.468 | ❌ | 0.474 | ❌ | 0.375 | ❌ |
| vader_neu_cv              | 0.590 | ✅ | 0.599 | ✅ | 0.484 | ❌ |
| vader_pos_cv              | 0.615 | ✅ | 0.753 | ✅ | 0.686 | ✅ |
| verb_ratio_cv             | 0.510 | ❌ | 0.603 | ✅ | 0.474 | ❌ |
| word_diversity_cv         | 0.878 | ✅ | 0.891 | ✅ | 1.000 | ✅ |

---

### B2. CE RMSSD_norm Binomial Test (Robustness)

| Model | Sig. / Total | % Sig. | Mean Win Rate |
|-------|-------------|--------|---------------|
| DS    | 15 / 20     | 75.0%  | 0.645         |
| CL35  | 16 / 20     | 80.0%  | 0.671         |
| G4OM  | 15 / 20     | 75.0%  | 0.692         |

| Feature | DS wr | DS | CL35 wr | CL35 | G4OM wr | G4OM |
|---------|-------|----|---------|------|---------|------|
| Agreeableness_rmssd_norm          | 0.827 | ✅ | 0.808 | ✅ | 0.869 | ✅ |
| Conscientiousness_rmssd_norm      | 0.718 | ✅ | 0.583 | ✅ | 0.811 | ✅ |
| Extraversion_rmssd_norm           | 0.522 | ❌ | 0.622 | ✅ | 0.699 | ✅ |
| Neuroticism_rmssd_norm            | 0.827 | ✅ | 0.888 | ✅ | 0.843 | ✅ |
| Openness_rmssd_norm               | 0.660 | ✅ | 0.779 | ✅ | 0.808 | ✅ |
| average_word_length_rmssd_norm    | 0.737 | ✅ | 0.471 | ❌ | 0.657 | ✅ |
| avg_sentence_length_rmssd_norm    | 0.846 | ✅ | 0.920 | ✅ | 0.952 | ✅ |
| content_word_ratio_rmssd_norm     | 0.798 | ✅ | 0.734 | ✅ | 0.776 | ✅ |
| flesch_reading_ease_rmssd_norm    | 0.272 | ❌ | 0.019 | ❌ | 0.253 | ❌ |
| function_word_ratio_rmssd_norm    | 0.583 | ✅ | 0.253 | ❌ | 0.574 | ✅ |
| gunning_fog_rmssd_norm            | 0.808 | ✅ | 0.875 | ✅ | 0.897 | ✅ |
| num_words_rmssd_norm              | 0.718 | ✅ | 0.946 | ✅ | 1.000 | ✅ |
| polarity_rmssd_norm               | 0.583 | ✅ | 0.676 | ✅ | 0.606 | ✅ |
| subjectivity_rmssd_norm           | 0.587 | ✅ | 0.561 | ✅ | 0.474 | ❌ |
| vader_compound_rmssd_norm         | 0.410 | ❌ | 0.955 | ✅ | 0.702 | ✅ |
| vader_neg_rmssd_norm              | 0.465 | ❌ | 0.513 | ❌ | 0.365 | ❌ |
| vader_neu_rmssd_norm              | 0.606 | ✅ | 0.590 | ✅ | 0.484 | ❌ |
| vader_pos_rmssd_norm              | 0.593 | ✅ | 0.763 | ✅ | 0.641 | ✅ |
| verb_ratio_rmssd_norm             | 0.510 | ❌ | 0.577 | ✅ | 0.436 | ❌ |
| word_diversity_rmssd_norm         | 0.833 | ✅ | 0.878 | ✅ | 0.997 | ✅ |

---

### B3. CE MASD_norm Binomial Test (Robustness)

| Model | Sig. / Total | % Sig. | Mean Win Rate |
|-------|-------------|--------|---------------|
| DS    | 14 / 20     | 70.0%  | 0.628         |
| CL35  | 16 / 20     | 80.0%  | 0.654         |
| G4OM  | 15 / 20     | 75.0%  | 0.687         |

| Feature | DS wr | DS | CL35 wr | CL35 | G4OM wr | G4OM |
|---------|-------|----|---------|------|---------|------|
| Agreeableness_masd_norm          | 0.740 | ✅ | 0.750 | ✅ | 0.827 | ✅ |
| Conscientiousness_masd_norm      | 0.699 | ✅ | 0.583 | ✅ | 0.808 | ✅ |
| Extraversion_masd_norm           | 0.532 | ❌ | 0.606 | ✅ | 0.686 | ✅ |
| Neuroticism_masd_norm            | 0.821 | ✅ | 0.888 | ✅ | 0.837 | ✅ |
| Openness_masd_norm               | 0.635 | ✅ | 0.760 | ✅ | 0.817 | ✅ |
| average_word_length_masd_norm    | 0.699 | ✅ | 0.442 | ❌ | 0.641 | ✅ |
| avg_sentence_length_masd_norm    | 0.821 | ✅ | 0.891 | ✅ | 0.933 | ✅ |
| content_word_ratio_masd_norm     | 0.804 | ✅ | 0.721 | ✅ | 0.766 | ✅ |
| flesch_reading_ease_masd_norm    | 0.266 | ❌ | 0.019 | ❌ | 0.253 | ❌ |
| function_word_ratio_masd_norm    | 0.551 | ❌ | 0.237 | ❌ | 0.587 | ✅ |
| gunning_fog_masd_norm            | 0.808 | ✅ | 0.840 | ✅ | 0.888 | ✅ |
| num_words_masd_norm              | 0.715 | ✅ | 0.910 | ✅ | 1.000 | ✅ |
| polarity_masd_norm               | 0.577 | ✅ | 0.667 | ✅ | 0.593 | ✅ |
| subjectivity_masd_norm           | 0.558 | ✅ | 0.554 | ✅ | 0.465 | ❌ |
| vader_compound_masd_norm         | 0.420 | ❌ | 0.952 | ✅ | 0.676 | ✅ |
| vader_neg_masd_norm              | 0.442 | ❌ | 0.494 | ❌ | 0.365 | ❌ |
| vader_neu_masd_norm              | 0.587 | ✅ | 0.587 | ✅ | 0.487 | ❌ |
| vader_pos_masd_norm              | 0.587 | ✅ | 0.744 | ✅ | 0.644 | ✅ |
| verb_ratio_masd_norm             | 0.494 | ❌ | 0.567 | ✅ | 0.462 | ❌ |
| word_diversity_masd_norm         | 0.811 | ✅ | 0.862 | ✅ | 0.997 | ✅ |

---

### B4. ML Classification: Human vs LLM_with_history

**Classifier:** RandomForestClassifier | **CV:** 5-fold GroupKFold (by author_id)  
**Features:** 20 CE CV features

#### B4a. Combined (Imbalanced, DS+CL35+G4OM)

**N:** 1,648 (412 human + 412×3 LLM_with_history)

| Metric       | Mean    | Std     |
|--------------|---------|---------|
| Accuracy     | 0.9332  | ±0.0109 |
| AUC          | 0.9771  | ±0.0057 |
| F1 Score     | 0.8557  | ±0.0271 |
| Human Recall | 0.7960  | ±0.0429 |

**Top 5 feature importances (combined):**

| Rank | Feature                 | Importance |
|------|-------------------------|------------|
| 1    | avg_sentence_length_cv  | 17.21%     |
| 2    | Agreeableness_cv        | 16.11%     |
| 3    | word_diversity_cv       | 9.06%      |
| 4    | Neuroticism_cv          | 8.72%      |
| 5    | gunning_fog_cv          | 6.02%      |

#### B4b. Balanced Per-Model (412 Human vs 412 LLM_with_history each)

| Model | Accuracy | AUC | F1 | Human Recall | N |
|-------|----------|-----|----|--------------|---|
| DS    | 0.8967 ± 0.0209 | 0.9654 ± 0.0053 | 0.8941 ± 0.0218 | 0.8736 ± 0.0288 | 824 |
| CL35  | 0.9745 ± 0.0105 | 0.9954 ± 0.0022 | 0.9745 ± 0.0103 | 0.9708 ± 0.0097 | 824 |
| G4OM  | 0.9988 ± 0.0024 | 0.9996 ± 0.0008 | 0.9988 ± 0.0025 | 0.9976 ± 0.0049 | 824 |

**Top 5 feature importances (per model):**

| Rank | DS | CL35 | G4OM |
|------|----|------|------|
| 1 | Agreeableness_cv (15.94%) | vader_compound_cv (22.94%) | num_words_cv (34.88%) |
| 2 | avg_sentence_length_cv (12.99%) | avg_sentence_length_cv (14.66%) | word_diversity_cv (23.76%) |
| 3 | Neuroticism_cv (9.56%) | Agreeableness_cv (10.14%) | avg_sentence_length_cv (10.62%) |
| 4 | gunning_fog_cv (6.85%) | Neuroticism_cv (8.76%) | Agreeableness_cv (7.83%) |
| 5 | word_diversity_cv (6.05%) | flesch_reading_ease_cv (8.48%) | vader_compound_cv (4.56%) |

---

### B5. Drift Binomial Test

**H₁:** P(Human total drift > LLM_with_history total drift) > 0.5 | Raw p-values, α = 0.05 | N = 412

| Space | Model | Human Wins | Win Rate | p-value     | Sig |
|-------|-------|-----------|---------|-------------|-----|
| CE    | DS    | 323 / 412 | 0.784   | 1.45e-32    | ✅  |
| CE    | CL35  | 341 / 412 | 0.828   | 1.05e-43    | ✅  |
| CE    | G4OM  | 407 / 412 | 0.988   | 9.24e-114   | ✅  |
| SBERT | DS    | 334 / 412 | 0.811   | 4.10e-39    | ✅  |
| SBERT | CL35  | 354 / 412 | 0.859   | 3.28e-53    | ✅  |
| SBERT | G4OM  | 318 / 412 | 0.772   | 7.73e-30    | ✅  |
| TFIDF | DS    | 127 / 412 | 0.308   | 1.000       | ❌  |
| TFIDF | CL35  | 43 / 412  | 0.104   | 1.000       | ❌  |
| TFIDF | G4OM  | 120 / 412 | 0.291   | 1.000       | ❌  |

**Mean drift per author (CE space):** Human = 686, DS = 450, CL35 = 332, G4OM = 55  
**Mean drift per author (SBERT space):** Human = 3.87, DS = 3.72, CL35 = 3.62, G4OM = 3.62  
**Mean drift per author (TFIDF space):** Human = 0.46, DS = 0.49, CL35 = 0.68, G4OM = 0.56

---

---

## Part C: Side-by-Side Comparison

### C1. CE Variability Summary

| Test | Metric | Instance-wise DS | With-Hist DS | Instance-wise CL35 | With-Hist CL35 | Instance-wise G4OM | With-Hist G4OM |
|------|--------|-----------------|-------------|-------------------|---------------|-------------------|---------------|
| CV       | Sig. / 20    | 14 / 20 | 14 / 20 | 16 / 20 | 16 / 20 | 15 / 20 | 15 / 20 |
| CV       | Mean win rate | 0.662  | 0.665   | 0.679   | 0.684   | 0.707   | 0.711   |
| RMSSD_norm | Sig. / 20  | 16 / 20 | 15 / 20 | 16 / 20 | 16 / 20 | 14 / 20 | 15 / 20 |
| RMSSD_norm | Mean win rate | 0.635 | 0.645   | 0.660   | 0.671   | 0.690   | 0.692   |
| MASD_norm  | Sig. / 20  | 14 / 20 | 14 / 20 | 16 / 20 | 16 / 20 | 14 / 20 | 15 / 20 |
| MASD_norm  | Mean win rate | 0.619 | 0.628   | 0.646   | 0.654   | 0.683   | 0.687   |

### C2. ML Classification Summary

#### Combined (Imbalanced, DS+CL35+G4OM)

| Metric       | Instance-wise | LLM_with_history | Δ       |
|--------------|--------------|------------------|---------|
| Accuracy     | 0.9363       | 0.9332           | −0.0031 |
| AUC          | 0.9770       | 0.9771           | +0.0001 |
| F1 Score     | 0.8631       | 0.8557           | −0.0074 |
| Human Recall | 0.8082       | 0.7960           | −0.0122 |

#### Balanced Per-Model (412 Human vs 412 LLM)

| Model | Metric | Instance-wise | LLM_with_history | Δ |
|-------|--------|--------------|-----------------|---|
| DS | Accuracy | 0.9064 | 0.8967 | −0.0097 |
| DS | AUC | 0.9632 | 0.9654 | +0.0022 |
| DS | F1 | 0.9058 | 0.8941 | −0.0117 |
| DS | Human Recall | 0.8980 | 0.8736 | −0.0244 |
| CL35 | Accuracy | 0.9733 | 0.9745 | +0.0012 |
| CL35 | AUC | 0.9961 | 0.9954 | −0.0007 |
| CL35 | F1 | 0.9732 | 0.9745 | +0.0013 |
| CL35 | Human Recall | 0.9684 | 0.9708 | +0.0024 |
| G4OM | Accuracy | 0.9988 | 0.9988 | 0.0000 |
| G4OM | AUC | 0.9996 | 0.9996 | 0.0000 |
| G4OM | F1 | 0.9988 | 0.9988 | 0.0000 |
| G4OM | Human Recall | 0.9976 | 0.9976 | 0.0000 |

**Finding:** DS shows a small decrease in human recall (−2.4%) with history; CL35 is essentially unchanged (±0.1%); G4OM is identical. The difficulty ranking (G4OM easiest → CL35 → DS hardest) is stable across both conditions.

### C3. Drift Comparison

| Space | Model | Instance-wise win rate | With-Hist win rate | Δ      | Direction |
|-------|-------|----------------------|-------------------|--------|-----------|
| CE    | DS    | 0.757 ✅             | 0.784 ✅           | +0.027 | CE gap **wider** |
| CE    | CL35  | 0.825 ✅             | 0.828 ✅           | +0.003 | stable |
| CE    | G4OM  | 0.973 ✅             | 0.988 ✅           | +0.015 | CE gap wider |
| SBERT | DS    | 0.830 ✅             | 0.811 ✅           | −0.019 | SBERT gap **slightly smaller** |
| SBERT | CL35  | 0.852 ✅             | 0.859 ✅           | +0.007 | stable |
| SBERT | G4OM  | 0.748 ✅             | 0.772 ✅           | +0.024 | stable |
| TFIDF | DS    | 0.328 ❌             | 0.308 ❌           | −0.020 | LLM already > Human |
| TFIDF | CL35  | 0.201 ❌             | 0.104 ❌           | −0.097 | TFIDF inversion **stronger** |
| TFIDF | G4OM  | 0.306 ❌             | 0.291 ❌           | −0.015 | stable |

**CE mean drift (Human = 686 in both):**

| Model | Instance-wise LLM | LLM_with_history | Δ (LLM drift) |
|-------|------------------|-----------------|---------------|
| DS    | 498              | 450             | −48 (less CE drift with history) |
| CL35  | 332              | 332             | ≈ 0 |
| G4OM  | 51               | 55              | +4 |

**TFIDF mean drift (Human = 0.46 in both):**

| Model | Instance-wise LLM | LLM_with_history | Δ (LLM drift) |
|-------|------------------|-----------------|---------------|
| DS    | 0.49             | 0.49            | ≈ 0 |
| CL35  | 0.58             | 0.68            | +0.10 (more lexical drift with history) |
| G4OM  | 0.51             | 0.56            | +0.05 |

---

## Part D: Conclusions

### D1. Temporal Flattening Persists Regardless of Conversational History

All three variability operators (CV, RMSSD_norm, MASD_norm) show essentially the same number of significant features and win rates in both conditions. The ML classifier achieves ~93% accuracy and ~0.977 AUC in both cases. Providing history does not help LLMs close the temporal variability gap.

### D2. CE Drift Gap Widened With History

Humans drift significantly more than both LLM conditions in CE space. The gap is slightly larger under LLM_with_history: DS LLM drifts less with history (mean 450 vs 498), increasing the human win rate from 0.757 to 0.784. This suggests that conversational memory causes models to stay more cognitively/emotionally consistent over time — but this consistency is exactly the flattening effect.

### D3. SBERT Drift Gap Is Stable

Mean SBERT drift is virtually identical between conditions (~3.87 human vs ~3.62 LLM). The small DS win-rate decrease (0.830→0.811) is not accompanied by any mean-level change and should be treated as noise.

### D4. TFIDF Inversion Strengthened With History

LLM TFIDF drift already exceeded human drift in the instance-wise condition. With history, this inversion is amplified — CL35 mean TFIDF drift rises from 0.58 to 0.68 (human stays at 0.46), and the human win rate collapses from 0.201 to 0.104. This is consistent with history causing more surface-level lexical rephrasing without changing the underlying stylistic trajectory.

### D5. Key Takeaway

> Providing incremental conversational history to LLMs does not reduce temporal flattening in CE variability or semantic drift. If anything, it suppresses CE drift further (widening the human advantage in CE space) while amplifying lexical churn (widening the TFIDF inversion). Temporal flattening is a fundamental property of current LLMs, robust to both prompt design and conversational context.

---

## Data Provenance

| Source | Path | N |
|--------|------|---|
| Human | `dataset/process/human/{academic,blogs,news}/` | 412 author-field pairs |
| Instance-wise LLM | `dataset/process/LLM/{DS,CL35,G4OM}/LV3/{academic,blogs,news}/` | 412 × 3 |
| LLM_with_history | `dataset/process/LLM_with_history/{DS,CL35,G4OM}/LV3/{academic,blogs,news}/` | 412 × 3 |

CE variability stats sourced from `author_timeseries_stats_merged.csv` (generated from `combined_merged_outliers_removed.csv`).  
Drift stats sourced from `{ce,sbert,tfidf}_drift.csv` (generated from raw `combined_merged.csv` / `combined_with_embeddings.csv`).
