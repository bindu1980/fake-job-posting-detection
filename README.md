# Fake Job Posting Detection

A Machine Learning project that detects whether a job posting is **Real** or **Fraudulent** using Natural Language Processing (NLP) and job-posting features.

## Overview

The project analyzes job descriptions and other attributes to identify patterns commonly associated with fraudulent job postings.

## Features

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Natural Language Processing using TF-IDF
* Feature engineering
* Handling of imbalanced data
* Comparison of multiple Machine Learning models
* Prediction of whether a job posting is real or fraudulent

## Machine Learning Models

The project evaluates the following classification models:

* Logistic Regression
* Multinomial Naive Bayes
* Random Forest

**Random Forest achieved the best overall F1-score and ROC-AUC in the project.**

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

## Project Structure

```text
fake-job-posting-detection/
│
├── src/
│   ├── modeling.py
│   ├── prediction_system.py
│   └── EDA_preprocessing.py
│
├
├── model_comparison.csv
├── model_comparison_bars.png
├── ROC_curves_comparison.png
└── top_terms.png
│
├── outputs/
│
└── project_report.md
│
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/bindu1980/fake-job-posting-detection.git
cd fake-job-posting-detection
```

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

The project workflow consists of three main Python files:

### 1. Data Preprocessing and EDA

```bash
python src/EDA_preprocessing.py
```

### 2. Model Training and Evaluation

```bash
python src/modeling.py
```

### 3. Prediction

```bash
python src/prediction_system.py
```

## Results

The repository contains the model comparison results and visualizations generated during the project.

### Model Comparison

The detailed model comparison is available in:

`results/model_comparison.csv`

### Model Performance

![Model Comparison](results/model_comparison_bars.png)

### ROC Curve Comparison

![ROC Curves](results/ROC_curves_comparison.png)

### Top Terms

![Top Terms](results/top_terms.png)

## Dataset

The project uses a Fake Job Postings dataset for training and evaluation.

The original and cleaned datasets are **not included in this repository because of their large file sizes**.

The data preprocessing code used to prepare the dataset is available in:

`src/EDA_preprocessing.py`

## Project Report

The detailed project report containing the methodology, preprocessing, model development, evaluation, and findings is available at:

`report/project_report.md`

## Note

This project is intended for **educational purposes**. The prediction indicates whether a job posting resembles fraudulent postings present in the dataset and does not guarantee that a real-world job posting is legitimate.

## Author

**Bindu**
