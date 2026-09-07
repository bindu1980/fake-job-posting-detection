"""
Fake Job Posting Detection - Step 2: Modeling & Evaluation
============================================================
Loads the cleaned dataset, builds TF-IDF text features (+ structured
features), trains multiple classifiers, evaluates them with several
metrics, and saves a comparison table + plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

OUT_DIR = r"C:\Users\allad\Downloads\Fake Job Posting Detection\Fake Job Posting Detection\outputs"
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------------
df = pd.read_csv(f"{OUT_DIR}/cleaned_data.csv")
df["full_text"] = df["full_text"].fillna("")

X = df.drop(columns=["fraudulent"])
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train fraud rate: %.3f | Test fraud rate: %.3f" %
      (y_train.mean(), y_test.mean()))

# ---------------------------------------------------------------
# 2. Feature engineering: TF-IDF on text + One-Hot on categoricals
#    + pass-through numeric/binary flags
# ---------------------------------------------------------------
text_feature = "full_text"
cat_features = ["employment_type", "required_experience", "required_education",
                 "industry", "function"]
num_features = ["text_length", "telecommuting", "has_company_logo", "has_questions"]

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english",
                                   ngram_range=(1, 2), min_df=3), text_feature),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ]
)

# ---------------------------------------------------------------
# 3. Define models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                random_state=RANDOM_STATE),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1),
}

results = []
fitted_pipelines = {}

plt.figure(figsize=(7, 6))
ax = plt.gca()

for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-score": f1, "ROC-AUC": auc
    })

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fraudulent"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # Confusion matrix plot
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fraud"], yticklabels=["Real", "Fraud"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_{safe_name}.png", dpi=120)
    plt.close()

    # ROC curve (added to shared figure)
    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_title("ROC Curves - Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves_comparison.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 4. Comparison table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
print("\n=== Model Comparison (sorted by F1-score) ===")
print(results_df.to_string(index=False))

plt.figure(figsize=(8, 5))
melted = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
sns.barplot(data=melted, x="variable", y="value", hue="Model")
plt.title("Model Comparison Across Metrics")
plt.ylabel("Score")
plt.xlabel("")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison_bars.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 5. Top predictive TF-IDF terms for Logistic Regression (interpretability)
# ---------------------------------------------------------------
lr_pipe = fitted_pipelines["Logistic Regression"]
tfidf_vectorizer = lr_pipe.named_steps["prep"].named_transformers_["tfidf"]
feature_names = tfidf_vectorizer.get_feature_names_out()
lr_coefs = lr_pipe.named_steps["clf"].coef_[0][:len(feature_names)]

coef_df = pd.DataFrame({"term": feature_names, "coef": lr_coefs})
top_fraud_terms = coef_df.sort_values("coef", ascending=False).head(20)
top_real_terms = coef_df.sort_values("coef", ascending=True).head(20)

print("\nTop 20 terms pushing toward FRAUDULENT:")
print(top_fraud_terms.to_string(index=False))
print("\nTop 20 terms pushing toward REAL:")
print(top_real_terms.to_string(index=False))

top_fraud_terms.to_csv(f"{OUT_DIR}/top_fraud_terms.csv", index=False)
top_real_terms.to_csv(f"{OUT_DIR}/top_real_terms.csv", index=False)

plt.figure(figsize=(8, 6))
combined = pd.concat([top_fraud_terms.head(15), top_real_terms.head(15)])
combined = combined.sort_values("coef")
colors = ["#d62728" if c > 0 else "#1f77b4" for c in combined["coef"]]
plt.barh(combined["term"], combined["coef"], color=colors)
plt.title("Most Influential Terms (Logistic Regression)\nRed = pushes toward Fraud, Blue = pushes toward Real")
plt.xlabel("Coefficient weight")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/top_terms.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 6. Save the best pipeline (by F1) for the prediction system
# ---------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_pipelines[best_model_name]
joblib.dump(best_pipeline, f"{OUT_DIR}/best_model_pipeline.joblib")
print(f"\nBest model by F1-score: {best_model_name} -> saved to best_model_pipeline.joblib")

# Also save all pipelines in case user wants to compare/predict with any
joblib.dump(fitted_pipelines, f"{OUT_DIR}/all_model_pipelines.joblib")
