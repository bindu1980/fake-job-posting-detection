#  Fake Job Posting Detection

A Machine Learning project that detects whether a job posting is **Real** or **Fraudulent** using Natural Language Processing (NLP) and job-posting features.

##  Overview

The project analyzes job descriptions and other attributes to identify patterns commonly associated with fake job postings.

### Features

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* NLP using TF-IDF
* Feature engineering
* Handles imbalanced data
* Compares multiple ML models
* Predicts whether a new job posting is real or fraudulent

## Machine Learning Models

* Logistic Regression
* Multinomial Naive Bayes
* Random Forest

**Random Forest** achieved the best overall F1-score and ROC-AUC in the project.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

##  Project Structure

```text
fake-job-posting/
│src
  ├── 01_eda_preprocessing.py
  ├── 02_modeling.py
  ├── 03_prediction_system.py
├── fake_job_postings.csv
├── outputs/
└── README.md
```

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fake-job-posting-detection.git
cd fake-job-posting-detection](https://github.com/bindu1980/fake-job-posting-detection.git
```

```

```

```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

Run the files in order:

```bash
python 01_eda_preprocessing.py
python 02_modeling.py
python 03_prediction_system.py
```

##  Note

This project is intended for **educational purposes**. The prediction indicates whether a job posting resembles fraudulent postings in the dataset and does not guarantee that a real-world job is legitimate.
