"""
Fake Job Posting Detection - Step 1: EDA & Preprocessing
==========================================================
Loads the raw dataset, explores it, cleans text fields, engineers
a unified text feature, and saves a cleaned CSV for the modeling step.
"""

import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

RAW_PATH = "fake_job_postings.csv"
OUT_DIR = r"C:\Users\allad\OneDrive\Desktop\fake job posting"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print("Shape:", df.shape)
print("\nClass balance:")
print(df["fraudulent"].value_counts())
print(df["fraudulent"].value_counts(normalize=True) * 100)

# ---------------------------------------------------------------
# 2. Missing value overview
# ---------------------------------------------------------------
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print("\nMissing values:\n", missing_summary[missing_summary.missing_count > 0])

# Plot class imbalance
plt.figure(figsize=(5, 4))
sns.countplot(x="fraudulent", data=df)
plt.title("Class Distribution (0 = Real, 1 = Fraudulent)")
plt.xlabel("Fraudulent")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=120)
plt.close()

# Plot missing values
plt.figure(figsize=(8, 6))
missing_summary[missing_summary.missing_count > 0]["missing_pct"].sort_values().plot(kind="barh")
plt.title("Missing Values by Column (%)")
plt.xlabel("% missing")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/missing_values.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 3. Handle missing values
# ---------------------------------------------------------------
# Text columns: missing -> empty string (we'll concatenate them anyway)
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
for col in text_cols:
    df[col] = df[col].fillna("")

# Categorical columns: missing -> "Unknown" (kept for optional structured features)
cat_cols = ["location", "department", "employment_type", "required_experience",
            "required_education", "industry", "function"]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# salary_range has ~84% missing and is inconsistent (ranges as strings/floats) -> drop
df = df.drop(columns=["salary_range", "job_id", "department"])

# ---------------------------------------------------------------
# 4. Text cleaning
# ---------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)        # keep letters only
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    return text

for col in text_cols:
    df[col + "_clean"] = df[col].apply(clean_text)

# Combine all textual signal into one field for vectorization
df["full_text"] = (
    df["title_clean"] + " " +
    df["company_profile_clean"] + " " +
    df["description_clean"] + " " +
    df["requirements_clean"] + " " +
    df["benefits_clean"]
)

# Basic text-length feature (often useful: fake postings tend to be shorter/vaguer)
df["text_length"] = df["full_text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="text_length", hue="fraudulent", bins=50,
             stat="density", common_norm=False, element="step")
plt.title("Text Length Distribution: Real vs Fraudulent")
plt.xlim(0, 1000)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/text_length_distribution.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 5. Structured feature quick look (fraud rate by flag)
# ---------------------------------------------------------------
for flag in ["telecommuting", "has_company_logo", "has_questions"]:
    print(f"\nFraud rate by {flag}:")
    print(df.groupby(flag)["fraudulent"].mean().round(3))

# ---------------------------------------------------------------
# 6. Save cleaned dataset
# ---------------------------------------------------------------
keep_cols = ["full_text", "text_length", "telecommuting", "has_company_logo",
             "has_questions", "employment_type", "required_experience",
             "required_education", "industry", "function", "fraudulent"]
clean_df = df[keep_cols]
clean_df.to_csv(f"{OUT_DIR}/cleaned_data.csv", index=False)
print("\nSaved cleaned dataset:", clean_df.shape)
print(clean_df.head(3))
