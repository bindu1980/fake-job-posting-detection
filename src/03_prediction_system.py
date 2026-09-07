"""
Fake Job Posting Detection - Step 3: Prediction System
=========================================================
Loads the trained pipeline and exposes a function to classify a new,
unseen job posting as Real or Fraudulent, with a confidence score.
"""

import re
import joblib
import pandas as pd

OUT_DIR = r"C:\Users\hansh\Downloads\files\outputs"

# Load all trained pipelines (each is a full sklearn Pipeline:
# TF-IDF + one-hot + classifier, so raw text/categoricals go in directly)
pipelines = joblib.load(f"{OUT_DIR}/all_model_pipelines.joblib")
DEFAULT_MODEL = "Random Forest"  # best F1-score / precision in our comparison


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_job_posting(
    title="", company_profile="", description="", requirements="", benefits="",
    telecommuting=0, has_company_logo=1, has_questions=0,
    employment_type="Unknown", required_experience="Unknown",
    required_education="Unknown", industry="Unknown", function="Unknown",
    model_name=DEFAULT_MODEL,
):
    """
    Classify a single job posting as Real (0) or Fraudulent (1).

    Returns a dict with the predicted label, fraud probability, and
    which model was used.
    """
    full_text = " ".join([
        clean_text(title), clean_text(company_profile), clean_text(description),
        clean_text(requirements), clean_text(benefits)
    ])
    text_length = len(full_text.split())

    row = pd.DataFrame([{
        "full_text": full_text,
        "text_length": text_length,
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
    }])

    pipe = pipelines[model_name]
    pred = pipe.predict(row)[0]
    proba = pipe.predict_proba(row)[0][1]

    return {
        "prediction": "Fraudulent" if pred == 1 else "Real",
        "fraud_probability": round(float(proba), 4),
        "model_used": model_name,
    }


#if __name__ == "__main__":
    # --- Demo 1: a posting with classic fraud red flags ---
    example_fraud = dict(
        title="Data Entry Clerk - Work From Home",
        company_profile="",
        description=(
            "Earn cash from home! No experience needed, just a computer and "
            "internet. Immediate hiring, flexible hours, weekly payment via "
            "email. Click the link below to start earning money today."
        ),
        requirements="No experience required. High school diploma preferred.",
        benefits="Bonus, cash payments, flexible schedule",
        telecommuting=1,
        has_company_logo=0,
        has_questions=0,
        employment_type="Contract",
        required_education="High School or equivalent",
    )

    # --- Demo 2: a realistic, detailed posting ---
    example_real = dict(
        title="Senior Software Engineer, Backend",
        company_profile=(
            "We are a growing fintech company based in San Francisco, building "
            "payment infrastructure for small businesses across North America. "
            "Our team of 120 engineers ships to millions of merchants."
        ),
        description=(
            "We are looking for a Senior Backend Engineer to join our Payments "
            "team. You will design and build scalable APIs, collaborate with "
            "product and design, and mentor junior engineers. This role reports "
            "to the VP of Engineering and is based in our downtown office with "
            "hybrid flexibility."
        ),
        requirements=(
            "5+ years of experience in backend development, proficiency in "
            "Python or Go, experience with distributed systems, strong "
            "communication skills, Bachelor's degree in Computer Science or "
            "equivalent experience."
        ),
        benefits="Health insurance, 401k matching, equity, unlimited PTO",
        telecommuting=0,
        has_company_logo=1,
        has_questions=1,
        employment_type="Full-time",
        required_experience="Mid-Senior level",
        required_education="Bachelor's Degree",
        industry="Financial Services",
        function="Engineering",
    )

    for label, example in [("FRAUD-PATTERN EXAMPLE", example_fraud),
                            ("LEGITIMATE EXAMPLE", example_real)]:
        print(f"\n--- {label} ---")
        for model_name in pipelines:
            result = predict_job_posting(**example, model_name=model_name)
            print(f"  [{model_name}] -> {result['prediction']} "
                  f"(fraud probability: {result['fraud_probability']})")
if __name__ == "__main__":
    my_posting = dict(
        title="Marketing Coordinator",
        company_profile="We are a mid-sized retail company based in Chicago...",
        description="We're looking for a Marketing Coordinator to join our team. "
                     "You'll manage social media campaigns and coordinate with "
                     "the design team on promotional materials...",
        requirements="2+ years marketing experience, Bachelor's degree preferred, "
                     "strong writing skills...",
        benefits="Health insurance, paid time off, 401k",
        telecommuting=0,
        has_company_logo=1,
        has_questions=1,
        employment_type="Full-time",
        required_experience="Associate",
        required_education="Bachelor's Degree",
        industry="Retail",
        function="Marketing",
    )

    result = predict_job_posting(**my_posting)
    print(result)