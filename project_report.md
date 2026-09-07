# Fake Job Posting Detection — Project Report

## 1. Dataset Overview

- **Source:** Real/Fake Job Postings dataset (Kaggle), 17,880 postings, 18 raw columns.
- **Target:** `fraudulent` (0 = real, 1 = fraudulent).
- **Class balance:** highly imbalanced — 866 fraudulent postings (4.84%) vs. 17,014 real (95.16%).
  This imbalance is the single most important fact shaping the whole project: accuracy alone is
  a poor evaluation metric here, since a model predicting "real" for everything would score ~95% accuracy
  while catching zero fraud.
- **Missing values:** `salary_range` (84% missing), `department` (65%), `required_education` (45%),
  `benefits` (40%), `required_experience` (39%) were the most sparse fields. Core text fields
  (`description`, `requirements`, `company_profile`) were mostly populated.

## 2. Preprocessing

- Dropped `salary_range` (too sparse/inconsistent) and `job_id`/`department` (no predictive value or too sparse).
- Text fields (`title`, `company_profile`, `description`, `requirements`, `benefits`) had missing values
  filled with empty strings, then cleaned: lowercased, HTML tags and URLs stripped, non-alphabetic
  characters removed, whitespace collapsed.
- Categorical fields (`employment_type`, `required_experience`, `required_education`, `industry`,
  `function`, `location`) had missing values filled with `"Unknown"` — treating "not specified" as
  its own informative category rather than discarding rows.
- All five cleaned text fields were concatenated into a single `full_text` feature, which captures
  the combined narrative signal of a posting.
- A `text_length` feature (word count) was engineered, since fraudulent postings tend to be vaguer/shorter.

## 3. Feature Engineering

- **Text:** TF-IDF vectorization of `full_text`, unigrams + bigrams, top 5,000 features, English stop
  words removed, `min_df=3` to drop rare noise terms.
- **Categorical:** one-hot encoding of `employment_type`, `required_experience`, `required_education`,
  `industry`, `function`.
- **Numeric/structured:** `text_length`, `telecommuting`, `has_company_logo`, `has_questions` passed
  through directly.

A quick fraud-rate breakdown by structured flags showed strong signal even before modeling:

| Flag | Value | Fraud rate |
|---|---|---|
| has_company_logo | 0 (no logo) | 15.9% |
| has_company_logo | 1 (has logo) | 2.0% |
| telecommuting | 0 | 4.7% |
| telecommuting | 1 | 8.3% |

Missing a company logo is one of the strongest single predictors in the dataset — consistent with the
intuition that fraudulent postings are often thrown together quickly without a proper company brand presence.

## 4. Train/Test Split

An 80/20 **stratified** split was used to preserve the ~4.8% fraud rate in both sets, ensuring the
test set is a fair, representative evaluation of real-world performance.

## 5. Models Trained

1. **Logistic Regression** (`class_weight="balanced"`) — interpretable linear baseline; coefficients
   directly show which words push toward "fraud" or "real."
2. **Multinomial Naive Bayes** — classic text-classification baseline, fast and simple, assumes
   feature independence.
3. **Random Forest** (`class_weight="balanced"`, 200 trees) — non-linear ensemble model, captures
   feature interactions between text and structured signals.

## 6. Evaluation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.980 | 1.000 | 0.590 | 0.742 | 0.992 |
| **Logistic Regression** | 0.963 | 0.577 | 0.908 | 0.706 | 0.986 |
| **Naive Bayes** | 0.943 | 0.419 | 0.434 | 0.426 | 0.841 |

**Why accuracy is misleading here:** all three models score 94%+ accuracy, but Naive Bayes only
catches 43% of actual fraud cases. Accuracy is dominated by the 95% majority class and hides this gap —
exactly why Precision, Recall, F1, and ROC-AUC are reported together.

### Model-by-model interpretation

- **Naive Bayes** is the weakest performer (F1 = 0.43). Its independence assumption between words
  doesn't hold well for job-posting text, where phrases (bigrams like "high school", "data entry")
  carry meaning as combinations, not just as isolated tokens.
- **Logistic Regression** has the **highest recall (0.91)** — it catches the most actual fraud cases —
  but at the cost of precision (0.58), meaning it also flags a fair number of legitimate postings as
  fraudulent (115 false positives in the test set). This makes sense for a linear model with
  `class_weight="balanced"`: it's tuned to not miss the minority class, at the expense of over-flagging.
- **Random Forest** has **perfect precision (1.00)** — every posting it flags as fraudulent actually is —
  but lower recall (0.59), meaning it misses about 41% of fraud cases (71 false negatives). It's the
  most "conservative" and trustworthy when it does raise an alarm, and it has the best overall ROC-AUC (0.992).

### Which model is "best" depends on the use case

- If the goal is **flagging postings for human review** (a triage system where false positives cost
  reviewer time but false negatives let fraud through), **Logistic Regression's higher recall** is more
  valuable — better to over-flag and let a human filter.
- If the goal is **automatically blocking/removing postings without human review**, **Random Forest's
  perfect precision** is safer — you don't want to auto-remove legitimate job listings.
- By raw F1-score (balancing both), **Random Forest edges out Logistic Regression** (0.742 vs 0.706),
  and it was selected as the default model in the prediction system, but Logistic Regression remains
  available and is arguably the better choice for a recall-sensitive deployment.

## 7. Interpretability — What the Model Has Learned

Inspecting Logistic Regression's coefficients on TF-IDF terms shows genuinely intuitive patterns:

**Words pushing toward "Fraudulent":** *aptitude, money, earn, financing, link, information security,
email, data entry, signing, high school, cash, bonus, compensation package* — language associated with
vague, urgency-driven, "quick cash" job scam framing.

**Words pushing toward "Real":** *team, companies, growing, based, search, english, digital, recruitment,
client, php, join team, years experience* — language associated with specific, professional
recruitment/agency and technical-role phrasing.

This aligns with known characteristics of job scams: vague promises of easy money, urgency, and
minimal specificity, versus real postings that reference specific teams, technologies, and requirements.

## 8. Prediction System

A `predict_job_posting()` function wraps the trained pipelines (TF-IDF + one-hot + classifier) so a
new, unseen posting (raw text + structured fields) can be classified directly, returning a label and
a fraud probability. It was validated on two hand-crafted examples:

- A posting written with classic scam patterns ("earn cash from home," no experience needed, no
  company profile, no logo) → correctly classified as **Fraudulent** by all three models
  (probability 0.65–1.00).
- A detailed, realistic senior engineering posting with a full company profile, specific requirements,
  and a company logo → correctly classified as **Real** by all three models (probability 0.02–0.05).

## 9. Limitations & Real-World Considerations

- **Severe class imbalance (4.8% fraud):** even with `class_weight="balanced"`, the models see far
  fewer fraud examples, which limits how well they generalize to fraud patterns not well-represented
  in this dataset.
- **Dataset is static and dated:** scam tactics evolve. A model trained on this dataset may not catch
  newer fraud phrasing/patterns that emerge after the data was collected.
- **Structured features can be gamed:** a bad actor could easily add a fake company logo or copy a
  legitimate company profile, defeating features like `has_company_logo` if the model over-relies on them.
- **Text-only signal has ceiling:** the model doesn't verify facts (e.g., does this company actually
  exist, is this a real address) — it only detects *stylistic and linguistic* patterns correlated
  with fraud in the training data, not ground-truth verification.
- **Precision/recall trade-off has real consequences:** in production, false negatives mean a scam
  reaches job seekers, while false positives mean a legitimate employer's posting gets removed/flagged —
  the acceptable trade-off should be a deliberate business/product decision, not just "pick the highest F1."
- **No temporal or company-level validation:** postings from the same company could appear in both
  train and test splits (if a company posted multiple postings), which could slightly inflate
  performance versus true out-of-sample generalization. A stricter validation would group-split by company.

## 10. Files Produced

- `01_eda_preprocessing.py` — EDA and cleaning
- `02_modeling.py` — feature engineering, model training, evaluation
- `03_prediction_system.py` — reusable prediction function + demo
- `cleaned_data.csv` — preprocessed dataset
- `model_comparison.csv` — metric comparison table
- `best_model_pipeline.joblib` / `all_model_pipelines.joblib` — trained, ready-to-use pipelines
- Plots: class distribution, missing values, text length distribution, confusion matrices (×3),
  ROC curve comparison, model comparison bar chart, top predictive terms
